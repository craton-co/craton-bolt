// SPDX-License-Identifier: Apache-2.0
//! `MERGE`: update, delete and insert a table's rows from a source relation.
//!
//! For `MERGE INTO t USING s ON <cond> WHEN ... THEN <action> ...`, the
//! target's rows get a row id and are matched against the source by
//! ordinary queries:
//!
//! * a join of target and source picks, per matched target row, the first
//!   `WHEN MATCHED [AND c]` clause that applies and computes its new values
//!   (a target row matched by more than one source row is an error when a
//!   clause applies to it);
//! * a source `LEFT JOIN` target finds the unmatched source rows and the
//!   first `WHEN NOT MATCHED [BY TARGET] [AND c] THEN INSERT` that applies;
//! * `WHEN NOT MATCHED BY SOURCE` clauses run over the unmatched target
//!   rows.
//!
//! The table is then rewritten in one step, like `UPDATE`: one snapshot for
//! a catalog table, a replacement for an in-memory one.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::compute::take;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{new_null_array, Array, ArrayRef, Int64Array, RecordBatch, UInt32Array};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use sqlparser::ast::{
    AssignmentTarget, Expr, MergeAction, MergeClause, MergeClauseKind, MergeInsertKind,
    TableFactor,
};

use crate::error::{BoltError, BoltResult};
use crate::exec::engine::Engine;
use crate::exec::engine_ddl::{align_batch, object_name};
use crate::exec::engine_dml::{concat, quote};

/// Engine table holding the target's rows (plus [`RID`]) during a MERGE.
const MERGE_TARGET: &str = "__bolt_merge_target";
/// Row id column of [`MERGE_TARGET`].
const RID: &str = "__bolt_rid";
/// Chosen-clause column of the matching queries (0: none applies).
const CLAUSE: &str = "__bolt_clause";

fn err(msg: impl std::fmt::Display) -> BoltError {
    BoltError::Sql(format!("MERGE: {msg}"))
}

/// What a clause does, with its values as SQL text.
enum Action {
    /// `(column, expression)` assignments.
    Update(Vec<(String, String)>),
    Delete,
    /// `(column, expression)` values.
    Insert(Vec<(String, String)>),
}

struct Clause {
    condition: Option<String>,
    action: Action,
}

/// The three clause groups of a MERGE, each in statement order.
#[derive(Default)]
struct Clauses {
    matched: Vec<Clause>,
    not_matched: Vec<Clause>,
    by_source: Vec<Clause>,
}

impl Clauses {
    fn parse(clauses: Vec<MergeClause>, columns: &[String]) -> BoltResult<Clauses> {
        let mut out = Clauses::default();
        let column = |name: &str| -> BoltResult<String> {
            columns
                .iter()
                .find(|c| c.eq_ignore_ascii_case(name))
                .cloned()
                .ok_or_else(|| err(format!("column '{name}' does not exist in the target")))
        };
        for c in clauses {
            let action = match c.action {
                MergeAction::Delete => Action::Delete,
                MergeAction::Update { assignments } => {
                    let mut set: Vec<(String, String)> = Vec::new();
                    for a in assignments {
                        let AssignmentTarget::ColumnName(name) = a.target else {
                            return Err(err("UPDATE SET (a, b) = ... is not supported"));
                        };
                        let col = column(&name.0.last().map(|i| i.value.clone()).unwrap_or_default())?;
                        if set.iter().any(|(c, _)| *c == col) {
                            return Err(err(format!("column '{col}' is assigned more than once")));
                        }
                        set.push((col, a.value.to_string()));
                    }
                    Action::Update(set)
                }
                MergeAction::Insert(ins) => {
                    let MergeInsertKind::Values(values) = ins.kind else {
                        return Err(err("INSERT ROW is not supported; list the values"));
                    };
                    let [row] = values.rows.as_slice() else {
                        return Err(err("INSERT takes exactly one VALUES row"));
                    };
                    let names: Vec<String> = if ins.columns.is_empty() {
                        columns.to_vec()
                    } else {
                        ins.columns
                            .iter()
                            .map(|i| column(&i.value))
                            .collect::<BoltResult<_>>()?
                    };
                    if names.len() != row.len() {
                        return Err(err(format!(
                            "INSERT names {} columns but supplies {} values",
                            names.len(),
                            row.len()
                        )));
                    }
                    Action::Insert(names.into_iter().zip(row.iter().map(Expr::to_string)).collect())
                }
            };
            let clause = Clause {
                condition: c.predicate.as_ref().map(Expr::to_string),
                action,
            };
            let insert = matches!(clause.action, Action::Insert(_));
            match c.clause_kind {
                MergeClauseKind::Matched if !insert => out.matched.push(clause),
                MergeClauseKind::NotMatched | MergeClauseKind::NotMatchedByTarget if insert => {
                    out.not_matched.push(clause)
                }
                MergeClauseKind::NotMatchedBySource if !insert => out.by_source.push(clause),
                kind => {
                    return Err(err(format!(
                        "WHEN {kind} cannot {}",
                        if insert { "INSERT" } else { "UPDATE / DELETE" }
                    )))
                }
            }
        }
        Ok(out)
    }
}

/// `CASE WHEN c1 THEN 1 WHEN c2 THEN 2 ... ELSE 0 END`.
fn clause_choice(clauses: &[Clause]) -> String {
    let arms: String = clauses
        .iter()
        .enumerate()
        .map(|(k, c)| {
            format!(
                " WHEN ({}) THEN {}",
                c.condition.as_deref().unwrap_or("1 = 1"),
                k + 1
            )
        })
        .collect();
    format!("CASE{arms} ELSE 0 END AS {}", quote(CLAUSE))
}

/// The value column `name` takes under the chosen clause: `CASE WHEN c1
/// THEN v1 ... ELSE fallback END`, or `None` when no clause sets it.
fn chosen_value(clauses: &[Clause], name: &str, fallback: &str) -> Option<String> {
    let value = |c: &Clause| match &c.action {
        Action::Update(set) | Action::Insert(set) => set
            .iter()
            .find(|(col, _)| col == name)
            .map(|(_, e)| e.clone()),
        Action::Delete => None,
    };
    if clauses.iter().all(|c| value(c).is_none()) {
        return None;
    }
    let arms: String = clauses
        .iter()
        .map(|c| {
            format!(
                " WHEN ({}) THEN ({})",
                c.condition.as_deref().unwrap_or("1 = 1"),
                value(c).unwrap_or_else(|| fallback.to_string())
            )
        })
        .collect();
    Some(format!("CASE{arms} ELSE {fallback} END AS {}", quote(name)))
}

/// Rows of one matching query, by row id (or source row).
struct Chosen {
    batch: RecordBatch,
    /// Row index → 1-based clause (0: none).
    clause: Vec<usize>,
}

impl Chosen {
    fn of(batch: RecordBatch) -> BoltResult<Chosen> {
        let c = arrow::compute::cast(
            batch
                .column_by_name(CLAUSE)
                .ok_or_else(|| err("internal: clause column missing"))?,
            &ArrowDataType::Int64,
        )
        .map_err(err)?;
        let c = c.as_primitive::<Int64Type>();
        let clause = (0..c.len())
            .map(|i| if c.is_null(i) { 0 } else { c.value(i) as usize })
            .collect();
        Ok(Chosen { batch, clause })
    }

    fn rids(&self) -> BoltResult<Vec<Option<i64>>> {
        let r = self
            .batch
            .column_by_name(RID)
            .ok_or_else(|| err("internal: row id column missing"))?;
        let r = arrow::compute::cast(r, &ArrowDataType::Int64).map_err(err)?;
        let r = r.as_primitive::<Int64Type>();
        Ok((0..r.len()).map(|i| (!r.is_null(i)).then(|| r.value(i))).collect())
    }
}

impl Engine {
    /// `MERGE INTO target USING source ON cond WHEN ...`. Returns the rows
    /// inserted, updated or deleted.
    pub(crate) fn execute_merge(
        &mut self,
        table: TableFactor,
        source: TableFactor,
        on: Expr,
        clauses: Vec<MergeClause>,
    ) -> BoltResult<u64> {
        let TableFactor::Table { name, alias, .. } = &table else {
            return Err(err("the target must be a table"));
        };
        let target = object_name(name);
        let qualifier = alias
            .as_ref()
            .map(|a| a.name.value.clone())
            .unwrap_or_else(|| name.0.last().map(|i| i.value.clone()).unwrap_or_default());
        let schema = self.writable_schema(&target)?;
        let columns: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
        let clauses = Clauses::parse(clauses, &columns)?;
        if clauses.matched.is_empty() && clauses.not_matched.is_empty() && clauses.by_source.is_empty()
        {
            return Err(err("needs at least one WHEN clause"));
        }
        // Target rows with their row ids.
        let rows = self.full_rows(&target, &schema)?;
        let n = rows.num_rows();
        let mut fields: Vec<ArrowField> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
        fields.push(ArrowField::new(RID, ArrowDataType::Int64, false));
        let mut cols = rows.columns().to_vec();
        cols.push(Arc::new(Int64Array::from_iter_values(0..n as i64)));
        let with_rid = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), cols)
            .map_err(err)?;
        if self.has_table(MERGE_TARGET) {
            self.deregister_table(MERGE_TARGET)?;
        }
        self.register_table(MERGE_TARGET, with_rid)?;
        let r = self.merged_rows(&rows, &qualifier, &source.to_string(), &on.to_string(), &clauses);
        self.deregister_table(MERGE_TARGET)?;
        let (new_rows, affected) = r?;
        if affected == 0 {
            return Ok(0);
        }
        let new_rows = align_batch(&new_rows, &schema, &[])?;
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            if let Some(meta) = cat.table(&target) {
                let all: Vec<String> = meta.files.iter().map(|f| f.path.clone()).collect();
                cat.rewrite_files(&target, &all, std::slice::from_ref(&new_rows))?;
                self.register_catalog_table(&target)?;
                self.after_table_change(&target)?;
                return Ok(affected);
            }
        }
        self.replace_table(target.clone(), new_rows)?;
        self.after_table_change(&target)?;
        Ok(affected)
    }

    /// The target's rows after the MERGE (`rows`: before), and the count
    /// affected.
    fn merged_rows(
        &mut self,
        rows: &RecordBatch,
        qualifier: &str,
        source: &str,
        on: &str,
        clauses: &Clauses,
    ) -> BoltResult<(RecordBatch, u64)> {
        let schema = rows.schema();
        let tq = quote(qualifier);
        let target_from = format!("{MERGE_TARGET} AS {tq}");
        let col = |c: &str| format!("{tq}.{}", quote(c));
        // Per target column: its value under the chosen clause, or the
        // current one.
        let select_values = |cl: &[Clause]| -> String {
            schema
                .fields()
                .iter()
                .filter_map(|f| chosen_value(cl, f.name(), &col(f.name())))
                .map(|e| format!(", {e}"))
                .collect()
        };
        let mut affected = 0u64;
        // Matched target rows.
        let mut matched_ids: HashSet<i64> = HashSet::new();
        let mut actions: HashMap<i64, (usize, usize, bool)> = HashMap::new(); // rid -> (row, clause, by_source)
        let matched = if clauses.matched.is_empty() && clauses.by_source.is_empty() {
            None
        } else {
            let q = format!(
                "SELECT {} AS {}, {}{} FROM {target_from} JOIN {source} ON {on}",
                col(RID),
                quote(RID),
                if clauses.matched.is_empty() {
                    format!("0 AS {}", quote(CLAUSE))
                } else {
                    clause_choice(&clauses.matched)
                },
                select_values(&clauses.matched),
            );
            let chosen = Chosen::of(self.sql(&q)?.into_record_batch())?;
            for (i, rid) in chosen.rids()?.into_iter().enumerate() {
                let Some(rid) = rid else { continue };
                matched_ids.insert(rid);
                let k = chosen.clause[i];
                if k == 0 {
                    continue;
                }
                if actions.insert(rid, (i, k, false)).is_some() {
                    return Err(err(
                        "a target row matches more than one source row; make the ON \
                         condition match at most one",
                    ));
                }
            }
            Some(chosen)
        };
        // Unmatched target rows.
        let by_source = if clauses.by_source.is_empty() {
            None
        } else {
            let q = format!(
                "SELECT {} AS {}, {}{} FROM {target_from}",
                col(RID),
                quote(RID),
                clause_choice(&clauses.by_source),
                select_values(&clauses.by_source),
            );
            let chosen = Chosen::of(self.sql(&q)?.into_record_batch())?;
            for (i, rid) in chosen.rids()?.into_iter().enumerate() {
                let Some(rid) = rid else { continue };
                if !matched_ids.contains(&rid) && chosen.clause[i] > 0 {
                    actions.insert(rid, (i, chosen.clause[i], true));
                }
            }
            Some(chosen)
        };
        // Unmatched source rows to insert.
        let inserted = if clauses.not_matched.is_empty() {
            None
        } else {
            let values: String = schema
                .fields()
                .iter()
                .filter_map(|f| chosen_value(&clauses.not_matched, f.name(), "NULL"))
                .map(|e| format!(", {e}"))
                .collect();
            let q = format!(
                "SELECT {} AS {}, {}{values} FROM {source} LEFT JOIN {target_from} ON {on}",
                col(RID),
                quote(RID),
                clause_choice(&clauses.not_matched),
            );
            let chosen = Chosen::of(self.sql(&q)?.into_record_batch())?;
            let picked: Vec<u32> = chosen
                .rids()?
                .into_iter()
                .enumerate()
                .filter(|(i, rid)| rid.is_none() && chosen.clause[*i] > 0)
                .map(|(i, _)| i as u32)
                .collect();
            Some((chosen, picked))
        };

        // Assemble: kept rows, updated rows, inserted rows.
        let n = rows.num_rows();
        let mut kept: Vec<u32> = Vec::new();
        let mut updated: [Vec<(u32, u32)>; 2] = [Vec::new(), Vec::new()]; // (rid, result row)
        for rid in 0..n as i64 {
            match actions.get(&rid) {
                None => kept.push(rid as u32),
                Some(&(row, k, src)) => {
                    let group = if src { &clauses.by_source } else { &clauses.matched };
                    affected += 1;
                    match group[k - 1].action {
                        Action::Delete => {}
                        _ => updated[usize::from(src)].push((rid as u32, row as u32)),
                    }
                }
            }
        }
        let pick = |idx: &[u32]| -> BoltResult<Vec<ArrayRef>> {
            let idx = UInt32Array::from(idx.to_vec());
            rows.columns()
                .iter()
                .map(|c| take(c.as_ref(), &idx, None).map_err(err))
                .collect()
        };
        let mut parts: Vec<RecordBatch> = Vec::new();
        let batch_of = |cols: Vec<ArrayRef>| -> BoltResult<RecordBatch> {
            let fields: Vec<ArrowField> = schema
                .fields()
                .iter()
                .zip(&cols)
                .map(|(f, c)| ArrowField::new(f.name(), c.data_type().clone(), true))
                .collect();
            let b = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), cols)
                .map_err(err)?;
            align_batch(&b, &schema, &[])
        };
        parts.push(batch_of(pick(&kept)?)?);
        for (src, list) in updated.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            let chosen = if src == 1 { by_source.as_ref() } else { matched.as_ref() }
                .expect("rows came from it");
            let rids: Vec<u32> = list.iter().map(|(r, _)| *r).collect();
            let result_rows = UInt32Array::from(list.iter().map(|(_, i)| *i).collect::<Vec<_>>());
            let original = pick(&rids)?;
            let cols = schema
                .fields()
                .iter()
                .zip(original)
                .map(|(f, orig)| match chosen.batch.column_by_name(f.name()) {
                    Some(c) => take(c.as_ref(), &result_rows, None).map_err(err),
                    None => Ok(orig),
                })
                .collect::<BoltResult<Vec<_>>>()?;
            parts.push(batch_of(cols)?);
        }
        if let Some((chosen, picked)) = &inserted {
            affected += picked.len() as u64;
            let idx = UInt32Array::from(picked.clone());
            let cols = schema
                .fields()
                .iter()
                .map(|f| match chosen.batch.column_by_name(f.name()) {
                    Some(c) => take(c.as_ref(), &idx, None).map_err(err),
                    None => Ok(new_null_array(f.data_type(), picked.len())),
                })
                .collect::<BoltResult<Vec<_>>>()?;
            parts.push(batch_of(cols)?);
        }
        Ok((concat(&schema, &parts)?, affected))
    }
}
