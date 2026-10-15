// SPDX-License-Identifier: Apache-2.0
//! SQL NULL semantics for the fused GPU kernels.
//!
//! The kernel IR computes values only: a NULL row holds an arbitrary value
//! (zero), and validity bitmaps are read only by `IS [NOT] NULL`. Before a
//! predicate or projection over nullable columns is compiled, it is
//! rewritten into explicit two-valued expressions over those validity checks:
//!
//! * a **predicate** `p` becomes `T(p)`, "p is TRUE" — so a comparison with a
//!   NULL operand drops the row, and `AND` / `OR` / `NOT` / `CASE` follow
//!   Kleene logic;
//! * a **value** `e` becomes `V(e)` (its value whenever it is not NULL, with
//!   `CASE` conditions tested by `T`), plus `known(e)`, "e is not NULL",
//!   which the projection path stores as a hidden validity output.
//!
//! An expression that reads no nullable column and no NULL literal is left
//! untouched (the emitted kernel is unchanged). Every construct the codegen
//! lowers has a rule; others (subqueries, function calls) are reported as
//! unsupported and left to the codegen, which rejects them.

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{BinaryOp, DataType, Expr, Literal, Schema, UnaryOp};

/// Name prefix of the hidden validity output of projection output `j`.
pub(crate) const HIDDEN_VALIDITY_PREFIX: &str = "__bolt_valid_";

fn unsupported(e: &Expr) -> BoltError {
    BoltError::Plan(format!(
        "GPU codegen: no NULL-semantics rule for {e:?}; evaluated on the host"
    ))
}

fn lit(b: bool) -> Expr {
    Expr::Literal(Literal::Bool(b))
}

fn and(a: Expr, b: Expr) -> Expr {
    match (&a, &b) {
        (Expr::Literal(Literal::Bool(true)), _) => b,
        (_, Expr::Literal(Literal::Bool(true))) => a,
        (Expr::Literal(Literal::Bool(false)), _) | (_, Expr::Literal(Literal::Bool(false))) => {
            lit(false)
        }
        _ => Expr::Binary {
            op: BinaryOp::And,
            left: Box::new(a),
            right: Box::new(b),
        },
    }
}

fn or(a: Expr, b: Expr) -> Expr {
    match (&a, &b) {
        (Expr::Literal(Literal::Bool(false)), _) => b,
        (_, Expr::Literal(Literal::Bool(false))) => a,
        (Expr::Literal(Literal::Bool(true)), _) | (_, Expr::Literal(Literal::Bool(true))) => {
            lit(true)
        }
        _ => Expr::Binary {
            op: BinaryOp::Or,
            left: Box::new(a),
            right: Box::new(b),
        },
    }
}

fn not(a: Expr) -> Expr {
    match a {
        Expr::Literal(Literal::Bool(b)) => lit(!b),
        other => Expr::Unary {
            op: UnaryOp::Not,
            operand: Box::new(other),
        },
    }
}

fn and_opt(a: Option<Expr>, b: Option<Expr>) -> Option<Expr> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(a), Some(b)) => Some(and(a, b)),
    }
}

/// Rewrites against one scan schema.
pub(crate) struct NullSemantics<'a> {
    schema: &'a Schema,
}

impl<'a> NullSemantics<'a> {
    pub(crate) fn new(schema: &'a Schema) -> Self {
        NullSemantics { schema }
    }

    fn nullable(&self, column: &str) -> bool {
        // Rewriter-synthesised dictionary columns encode NULL rows
        // themselves (index sentinel / rank guard).
        !column.starts_with("__idx_")
            && !column.starts_with("__rank_")
            && self.schema.field(column).is_ok_and(|f| f.nullable)
    }

    /// `true` when `e` reads a nullable column or a NULL literal.
    pub(crate) fn touches_nulls(&self, e: &Expr) -> bool {
        let mut found = false;
        self.visit(e, &mut found);
        found
    }

    fn visit(&self, e: &Expr, found: &mut bool) {
        if *found {
            return;
        }
        match e {
            Expr::Column(c) => *found = self.nullable(c),
            Expr::Literal(Literal::Null) => *found = true,
            Expr::Literal(_) => {}
            Expr::Alias(x, _) => self.visit(x, found),
            Expr::Binary { left, right, .. } => {
                self.visit(left, found);
                self.visit(right, found);
            }
            Expr::Unary { operand, .. } => self.visit(operand, found),
            Expr::Cast { expr, .. } => self.visit(expr, found),
            Expr::Case {
                branches,
                else_branch,
            } => {
                for (w, t) in branches {
                    self.visit(w, found);
                    self.visit(t, found);
                }
                match else_branch {
                    Some(x) => self.visit(x, found),
                    // No ELSE: NULL when no branch matches.
                    None => *found = true,
                }
            }
            // Anything else is treated as touching NULLs so it goes through
            // the rewrite (and is reported unsupported there).
            _ => *found = true,
        }
    }

    /// `e` is TRUE (two-valued; `e` must be Boolean).
    pub(crate) fn is_true(&self, e: &Expr) -> BoltResult<Expr> {
        if !self.touches_nulls(e) {
            return Ok(e.clone());
        }
        Ok(match e {
            Expr::Alias(x, _) => self.is_true(x)?,
            Expr::Literal(Literal::Bool(b)) => lit(*b),
            Expr::Literal(Literal::Null) => lit(false),
            Expr::Binary {
                op: BinaryOp::And,
                left,
                right,
            } => and(self.is_true(left)?, self.is_true(right)?),
            Expr::Binary {
                op: BinaryOp::Or,
                left,
                right,
            } => or(self.is_true(left)?, self.is_true(right)?),
            Expr::Unary {
                op: UnaryOp::Not,
                operand,
            } => self.is_false(operand)?,
            Expr::Unary { op, operand } => self.null_test(*op, operand)?,
            Expr::Case {
                branches,
                else_branch,
            } => Expr::Case {
                branches: branches
                    .iter()
                    .map(|(w, t)| Ok((self.is_true(w)?, self.is_true(t)?)))
                    .collect::<BoltResult<_>>()?,
                else_branch: Some(Box::new(match else_branch {
                    Some(x) => self.is_true(x)?,
                    None => lit(false),
                })),
            },
            other => match self.known(other)? {
                None => self.value(other)?,
                Some(k) => and(k, self.value(other)?),
            },
        })
    }

    /// `e` is FALSE (two-valued; `e` must be Boolean).
    fn is_false(&self, e: &Expr) -> BoltResult<Expr> {
        if !self.touches_nulls(e) {
            return Ok(not(e.clone()));
        }
        Ok(match e {
            Expr::Alias(x, _) => self.is_false(x)?,
            Expr::Literal(Literal::Bool(b)) => lit(!*b),
            Expr::Literal(Literal::Null) => lit(false),
            Expr::Binary {
                op: BinaryOp::And,
                left,
                right,
            } => or(self.is_false(left)?, self.is_false(right)?),
            Expr::Binary {
                op: BinaryOp::Or,
                left,
                right,
            } => and(self.is_false(left)?, self.is_false(right)?),
            Expr::Unary {
                op: UnaryOp::Not,
                operand,
            } => self.is_true(operand)?,
            Expr::Unary { op, operand } => not(self.null_test(*op, operand)?),
            Expr::Case {
                branches,
                else_branch,
            } => Expr::Case {
                branches: branches
                    .iter()
                    .map(|(w, t)| Ok((self.is_true(w)?, self.is_false(t)?)))
                    .collect::<BoltResult<_>>()?,
                else_branch: Some(Box::new(match else_branch {
                    Some(x) => self.is_false(x)?,
                    None => lit(false),
                })),
            },
            other => match self.known(other)? {
                None => not(self.value(other)?),
                Some(k) => and(k, not(self.value(other)?)),
            },
        })
    }

    /// `x IS NULL` / `x IS NOT NULL`, two-valued for any operand.
    fn null_test(&self, op: UnaryOp, x: &Expr) -> BoltResult<Expr> {
        let want_null = matches!(op, UnaryOp::IsNull);
        // A bare column keeps its direct validity check.
        let mut bare = x;
        while let Expr::Alias(inner, _) = bare {
            bare = inner;
        }
        if let Expr::Column(_) = bare {
            return Ok(Expr::Unary {
                op,
                operand: Box::new(bare.clone()),
            });
        }
        Ok(match self.known(x)? {
            None => lit(!want_null),
            Some(k) if want_null => not(k),
            Some(k) => k,
        })
    }

    /// `e` is not NULL; `None` when it never is.
    pub(crate) fn known(&self, e: &Expr) -> BoltResult<Option<Expr>> {
        if !self.touches_nulls(e) {
            return Ok(None);
        }
        Ok(match e {
            Expr::Column(c) => Some(Expr::Unary {
                op: UnaryOp::IsNotNull,
                operand: Box::new(Expr::Column(c.clone())),
            }),
            Expr::Literal(Literal::Null) => Some(lit(false)),
            Expr::Literal(_) => None,
            Expr::Alias(x, _) => self.known(x)?,
            Expr::Binary {
                op: BinaryOp::And | BinaryOp::Or,
                ..
            }
            | Expr::Unary {
                op: UnaryOp::Not, ..
            } => Some(or(self.is_true(e)?, self.is_false(e)?)),
            Expr::Unary { .. } => None,
            Expr::Binary { left, right, .. } => and_opt(self.known(left)?, self.known(right)?),
            Expr::Cast {
                expr, safe: false, ..
            } => self.known(expr)?,
            Expr::Case {
                branches,
                else_branch,
            } => {
                let mut out = Vec::with_capacity(branches.len());
                for (w, t) in branches {
                    out.push((
                        self.is_true(w)?,
                        self.known(t)?.unwrap_or_else(|| lit(true)),
                    ));
                }
                let otherwise = match else_branch {
                    Some(x) => self.known(x)?.unwrap_or_else(|| lit(true)),
                    None => lit(false),
                };
                Some(Expr::Case {
                    branches: out,
                    else_branch: Some(Box::new(otherwise)),
                })
            }
            other => return Err(unsupported(other)),
        })
    }

    /// The value of `e` wherever it is not NULL (`CASE` conditions tested
    /// with SQL semantics).
    pub(crate) fn value(&self, e: &Expr) -> BoltResult<Expr> {
        if !self.touches_nulls(e) {
            return Ok(e.clone());
        }
        Ok(match e {
            Expr::Column(_) | Expr::Literal(_) => e.clone(),
            Expr::Alias(x, n) => Expr::Alias(Box::new(self.value(x)?), n.clone()),
            Expr::Binary { op, left, right } => Expr::Binary {
                op: *op,
                left: Box::new(self.value(left)?),
                right: Box::new(self.value(right)?),
            },
            Expr::Unary {
                op: UnaryOp::Not,
                operand,
            } => not(self.value(operand)?),
            Expr::Unary { op, operand } => self.null_test(*op, operand)?,
            Expr::Cast {
                expr,
                target,
                safe: false,
            } => Expr::Cast {
                expr: Box::new(self.value(expr)?),
                target: *target,
                safe: false,
            },
            Expr::Case {
                branches,
                else_branch,
            } => {
                // A NULL arm (an explicit NULL, or the missing ELSE) takes
                // the value of a non-NULL arm: `known` marks those rows NULL,
                // and the kernel's CASE needs a value of the result type.
                let is_null = |e: &Expr| matches!(e, Expr::Literal(Literal::Null));
                let filler = branches
                    .iter()
                    .map(|(_, t)| t)
                    .chain(else_branch.as_deref())
                    .find(|t| !is_null(t))
                    .ok_or_else(|| unsupported(e))?;
                let filler = self.value(filler)?;
                let arm = |t: &Expr| -> BoltResult<Expr> {
                    if is_null(t) {
                        Ok(filler.clone())
                    } else {
                        self.value(t)
                    }
                };
                Expr::Case {
                    branches: branches
                        .iter()
                        .map(|(w, t)| Ok((self.is_true(w)?, arm(t)?)))
                        .collect::<BoltResult<_>>()?,
                    else_branch: Some(Box::new(match else_branch {
                        Some(x) => arm(x)?,
                        None => filler.clone(),
                    })),
                }
            }
            other => return Err(unsupported(other)),
        })
    }

    /// `true` when `e`'s NULL behaviour is "NULL if any input is NULL", so a
    /// kernel output's validity is the AND of its inputs' validity (the
    /// fused aggregate pre-stage relies on that).
    pub(crate) fn is_strict(&self, e: &Expr) -> bool {
        if !self.touches_nulls(e) {
            return true;
        }
        match e {
            Expr::Column(_) => true,
            Expr::Literal(Literal::Null) => false,
            Expr::Literal(_) => true,
            Expr::Alias(x, _) => self.is_strict(x),
            Expr::Binary {
                op: BinaryOp::And | BinaryOp::Or,
                ..
            } => false,
            Expr::Binary { left, right, .. } => self.is_strict(left) && self.is_strict(right),
            Expr::Cast {
                expr, safe: false, ..
            } => self.is_strict(expr),
            Expr::Unary {
                op: UnaryOp::Not,
                operand,
            } => self.is_strict(operand),
            _ => false,
        }
    }
}

/// The dtype `DataType::Bool`, for hidden validity outputs.
pub(crate) const VALIDITY_DTYPE: DataType = DataType::Bool;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::Field;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("n", DataType::Int64, true),
        ])
    }

    fn col(c: &str) -> Expr {
        Expr::Column(c.into())
    }

    fn lt(l: Expr, v: i64) -> Expr {
        Expr::Binary {
            op: BinaryOp::Lt,
            left: Box::new(l),
            right: Box::new(Expr::Literal(Literal::Int64(v))),
        }
    }

    #[test]
    fn non_nullable_expressions_are_untouched() {
        let s = schema();
        let ns = NullSemantics::new(&s);
        let p = lt(col("a"), 5);
        assert!(!ns.touches_nulls(&p));
        assert_eq!(format!("{:?}", ns.is_true(&p).unwrap()), format!("{p:?}"));
        assert!(ns.known(&p).unwrap().is_none());
    }

    #[test]
    fn comparisons_on_nullable_columns_require_validity() {
        let s = schema();
        let ns = NullSemantics::new(&s);
        let t = ns.is_true(&lt(col("n"), 5)).unwrap();
        let text = format!("{t:?}");
        assert!(text.contains("IsNotNull"), "{text}");
        // NOT (n < 5) must also require n to be known.
        let f = ns.is_true(&not(lt(col("n"), 5))).unwrap();
        let text = format!("{f:?}");
        assert!(text.contains("IsNotNull") && text.contains("Not"), "{text}");
    }

    #[test]
    fn or_with_a_true_side_is_known() {
        let s = schema();
        let ns = NullSemantics::new(&s);
        let e = or(lt(col("n"), 5), lt(col("a"), 3));
        let k = ns.known(&e).unwrap().expect("depends on n");
        // known(n < 5 OR a < 3) = T(e) OR F(e) — includes `a < 3`.
        assert!(format!("{k:?}").contains("Column(\"a\")"));
        assert!(!ns.is_strict(&e));
        assert!(ns.is_strict(&lt(col("n"), 5)));
    }
}
