// SPDX-License-Identifier: Apache-2.0

//! "Did you mean...?" suggestion helpers for unknown SQL identifiers.
//!
//! When the planner fails to resolve an identifier (column, table qualifier,
//! aggregate function name), we want to add a short hint pointing at the
//! closest in-scope name. The "closest" rule is bounded Levenshtein edit
//! distance with a cap of 2 — past that point a suggestion is more likely to
//! confuse than help, so we just omit the hint.
//!
//! The helpers are pure and have no dependencies outside `std`. They are
//! cheap: the edit-distance matrix is O(n*m) per candidate, and identifier
//! lengths in practice are short (rarely more than ~30 chars). Crucially,
//! [`closest_match`] performs a case-insensitive comparison so that a typo
//! like `Naem` against a schema field named `name` still surfaces a hint.

/// Bounded Levenshtein edit distance.
///
/// Returns the edit distance between `a` and `b`, but only if it is `<=
/// limit`. Otherwise returns `None`. The early-out lets the cap-aware
/// caller avoid finishing the full matrix when the candidate is clearly
/// too far away.
///
/// The implementation uses the classic O(n*m) match-character matrix; for
/// the short identifier strings this is invoked on (typically under 30
/// chars), this is fine — see the module-level note.
pub fn edit_distance_within(a: &str, b: &str, limit: usize) -> Option<usize> {
    // Cheap length-based prune: if the strings differ in length by more
    // than `limit` characters, no alignment can possibly be within the
    // cap. This is a hot fast-path because most candidate-name comparisons
    // are against unrelated names of very different length.
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let la = a_chars.len();
    let lb = b_chars.len();
    if la.abs_diff(lb) > limit {
        return None;
    }
    if la == 0 {
        return if lb <= limit { Some(lb) } else { None };
    }
    if lb == 0 {
        return if la <= limit { Some(la) } else { None };
    }

    // Standard two-row DP. Row `prev` holds distances for the previous
    // `a` prefix; `cur` is reused per row.
    let mut prev: Vec<usize> = (0..=lb).collect();
    let mut cur: Vec<usize> = vec![0; lb + 1];
    for i in 1..=la {
        cur[0] = i;
        let mut row_min = cur[0];
        for j in 1..=lb {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            let del = prev[j] + 1;
            let ins = cur[j - 1] + 1;
            let sub = prev[j - 1] + cost;
            cur[j] = del.min(ins).min(sub);
            if cur[j] < row_min {
                row_min = cur[j];
            }
        }
        // If every cell in this row already exceeds the cap, no completion
        // can bring the final distance back under it — bail out early.
        if row_min > limit {
            return None;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let dist = prev[lb];
    if dist <= limit {
        Some(dist)
    } else {
        None
    }
}

/// Find the closest candidate to `needle` within edit distance 2 (case-
/// insensitive). Returns the candidate's original casing.
///
/// On ties, the first candidate in iteration order wins — caller controls
/// ordering by passing an ordered iterator (typically [`Schema::fields`]
/// in declaration order).
///
/// Returns `None` if no candidate is within distance 2, or if `needle` is
/// empty.
pub fn closest_match<'a, I>(needle: &str, candidates: I) -> Option<&'a str>
where
    I: IntoIterator<Item = &'a str>,
{
    const MAX_DISTANCE: usize = 2;
    if needle.is_empty() {
        return None;
    }
    let needle_lc = needle.to_ascii_lowercase();
    let mut best: Option<(usize, &'a str)> = None;
    for cand in candidates {
        if cand.is_empty() {
            continue;
        }
        let cand_lc = cand.to_ascii_lowercase();
        if let Some(d) = edit_distance_within(&needle_lc, &cand_lc, MAX_DISTANCE) {
            // Exact (after case-fold) shouldn't really happen at the call
            // sites — those only ask once the case-sensitive lookup has
            // already failed — but if it does, surface it immediately.
            if d == 0 {
                return Some(cand);
            }
            match best {
                None => best = Some((d, cand)),
                Some((bd, _)) if d < bd => best = Some((d, cand)),
                _ => {}
            }
        }
    }
    best.map(|(_, s)| s)
}

/// Format the standard " (did you mean 'X'?)" suffix for an unknown
/// identifier, or the empty string if no candidate is close enough.
///
/// Call sites concatenate this directly to the existing error message so
/// the wire format stays "<original message>(did you mean '<X>'?)" with a
/// leading space already supplied here. When no close match exists, the
/// returned string is empty and the error message is unchanged.
pub fn did_you_mean_suffix<'a, I>(needle: &str, candidates: I) -> String
where
    I: IntoIterator<Item = &'a str>,
{
    match closest_match(needle, candidates) {
        Some(c) => format!(" (did you mean '{c}'?)"),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_distance_basic_cases() {
        // Identical strings -> distance 0.
        assert_eq!(edit_distance_within("name", "name", 2), Some(0));
        // One substitution.
        assert_eq!(edit_distance_within("naem", "name", 2), Some(2));
        // One transposition counts as two substitutions in plain
        // Levenshtein (we deliberately don't use Damerau-Levenshtein —
        // simpler, and the cap covers the common single-typo cases).
        assert_eq!(edit_distance_within("nmae", "name", 2), Some(2));
        // One insertion.
        assert_eq!(edit_distance_within("nam", "name", 2), Some(1));
        // One deletion.
        assert_eq!(edit_distance_within("names", "name", 2), Some(1));
        // Too far apart -> None.
        assert_eq!(edit_distance_within("naem", "address", 2), None);
        // Empty needle.
        assert_eq!(edit_distance_within("", "name", 2), None);
        assert_eq!(edit_distance_within("", "ab", 2), Some(2));
    }

    #[test]
    fn closest_match_picks_nearest_candidate() {
        let candidates = ["name", "age", "address", "email"];
        // Single-char typo on the obvious target.
        assert_eq!(
            closest_match("naem", candidates.iter().copied()),
            Some("name")
        );
        assert_eq!(
            closest_match("emial", candidates.iter().copied()),
            Some("email")
        );
        // Too far -> no suggestion.
        assert_eq!(closest_match("zzzzzzz", candidates.iter().copied()), None);
        // Empty needle -> no suggestion.
        assert_eq!(closest_match("", candidates.iter().copied()), None);
    }

    #[test]
    fn closest_match_is_case_insensitive_but_preserves_candidate_casing() {
        let candidates = ["Name", "Age"];
        assert_eq!(
            closest_match("naem", candidates.iter().copied()),
            Some("Name")
        );
    }

    #[test]
    fn did_you_mean_suffix_formats_correctly() {
        let candidates = ["name", "age"];
        assert_eq!(
            did_you_mean_suffix("naem", candidates.iter().copied()),
            " (did you mean 'name'?)"
        );
        assert_eq!(did_you_mean_suffix("zzzz", candidates.iter().copied()), "");
    }
}
