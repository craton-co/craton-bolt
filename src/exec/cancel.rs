// SPDX-License-Identifier: Apache-2.0

//! Query cancellation and statement timeouts.
//!
//! A [`CancellationToken`] is handed to an entry point such as
//! [`Engine::sql_with_cancel`](crate::Engine::sql_with_cancel). While that
//! call runs, the token is the thread's *active* token: the engine polls it
//! (one atomic load, plus a clock read when a deadline is set) before every
//! physical operator, at the head of every morsel / partition / chunk loop,
//! and between the synchronised passes of multi-kernel operators. A kernel
//! that is already running is never interrupted; the statement stops at the
//! next check and returns [`BoltError::Cancelled`].
//!
//! Statements that change data (`INSERT`, `UPDATE`, `DELETE`, `MERGE`,
//! `CREATE TABLE AS`, `ALTER TABLE`, `COPY ... FROM`, materialized-view
//! refreshes, transaction control) poll the token one last time at their
//! commit point ([`commit_point`]); past it the statement runs to completion,
//! so a cancelled statement either changed nothing or completed.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{BoltError, BoltResult};

/// Why a statement stopped early; carried by [`BoltError::Cancelled`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CancelReason {
    /// [`CancellationToken::cancel`] was called.
    Cancelled,
    /// The token's deadline ([`CancellationToken::with_deadline`] /
    /// [`CancellationToken::with_timeout`]) passed.
    DeadlineExceeded,
}

impl std::fmt::Display for CancelReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            CancelReason::Cancelled => "query cancelled",
            CancelReason::DeadlineExceeded => "query deadline exceeded",
        })
    }
}

/// A cheap, cloneable handle that stops a running statement.
///
/// Clones share one flag: calling [`cancel`](Self::cancel) on any clone —
/// from any thread — cancels every statement running with any of them. A
/// token may also carry a deadline, which makes a statement timeout just a
/// token: `CancellationToken::new().with_timeout(Duration::from_secs(5))`.
///
/// ```
/// use std::time::Duration;
/// use craton_bolt::CancellationToken;
///
/// let token = CancellationToken::new();
/// let timed = token.with_timeout(Duration::from_secs(30));
/// assert!(!timed.is_cancelled());
/// token.cancel(); // also cancels `timed`, which shares the flag
/// assert!(timed.is_cancelled());
/// ```
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

impl CancellationToken {
    /// A token that is not cancelled and has no deadline.
    pub fn new() -> Self {
        Self::default()
    }

    /// A clone of this token that also expires at `deadline` (the earlier of
    /// `deadline` and any deadline it already has). It shares the cancel
    /// flag with `self`.
    #[must_use]
    pub fn with_deadline(&self, deadline: Instant) -> Self {
        Self {
            flag: Arc::clone(&self.flag),
            deadline: Some(self.deadline.map_or(deadline, |d| d.min(deadline))),
        }
    }

    /// [`with_deadline`](Self::with_deadline) at `timeout` from now.
    #[must_use]
    pub fn with_timeout(&self, timeout: Duration) -> Self {
        // A timeout too large for `Instant` is no deadline at all.
        match Instant::now().checked_add(timeout) {
            Some(deadline) => self.with_deadline(deadline),
            None => self.clone(),
        }
    }

    /// Cancel every statement running with this token or a clone of it.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }

    /// The token's deadline, if it has one.
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// `true` once the token is cancelled or its deadline has passed.
    pub fn is_cancelled(&self) -> bool {
        self.reason().is_some()
    }

    /// Why the token stops statements, if it does.
    pub fn reason(&self) -> Option<CancelReason> {
        if self.flag.load(Ordering::Acquire) {
            Some(CancelReason::Cancelled)
        } else if self.deadline.is_some_and(|d| Instant::now() >= d) {
            Some(CancelReason::DeadlineExceeded)
        } else {
            None
        }
    }

    /// `Err(BoltError::Cancelled)` once the token stops statements.
    pub(crate) fn check(&self) -> BoltResult<()> {
        match self.reason() {
            Some(reason) => Err(BoltError::Cancelled(reason)),
            None => Ok(()),
        }
    }
}

/// The token of the statement running on this thread.
struct Active {
    token: CancellationToken,
    /// Set while a [`CommitGuard`] is alive: the statement is applying its
    /// change and must not stop.
    committing: bool,
}

thread_local! {
    static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) };
}

/// Restores the previously active token when dropped.
#[must_use]
pub(crate) struct ScopeGuard {
    previous: Option<Active>,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        ACTIVE.with(|a| *a.borrow_mut() = previous);
    }
}

/// Make `token` the active token of this thread until the guard drops.
pub(crate) fn scope(token: &CancellationToken) -> ScopeGuard {
    let previous = ACTIVE.with(|a| {
        a.borrow_mut().replace(Active {
            token: token.clone(),
            committing: false,
        })
    });
    ScopeGuard { previous }
}

/// Stop the running statement if its token says so: the single decision
/// every check point calls. A no-op without an active token, and while a
/// statement is past its [`commit_point`].
pub(crate) fn check() -> BoltResult<()> {
    ACTIVE.with(|a| match a.borrow().as_ref() {
        Some(active) if !active.committing => active.token.check(),
        _ => Ok(()),
    })
}

/// Rows between two checks of a row-at-a-time host loop.
pub(crate) const ROWS_PER_CHECK: usize = 1 << 16;

/// [`check`] on every [`ROWS_PER_CHECK`]-th `row` of a row-at-a-time loop.
#[inline]
pub(crate) fn check_row(row: usize) -> BoltResult<()> {
    if row % ROWS_PER_CHECK == 0 {
        check()
    } else {
        Ok(())
    }
}

/// Keeps the statement uninterruptible until dropped; see [`commit_point`].
#[must_use]
pub(crate) struct CommitGuard {
    previous: bool,
}

impl Drop for CommitGuard {
    fn drop(&mut self) {
        let previous = self.previous;
        ACTIVE.with(|a| {
            if let Some(active) = a.borrow_mut().as_mut() {
                active.committing = previous;
            }
        });
    }
}

/// The last check of a data-changing operation: once it returns, checks are
/// no-ops until the guard drops, so the change about to be applied (and the
/// work that follows it, such as refreshing dependent views) is applied
/// whole. Hold the guard to the end of the operation.
pub(crate) fn commit_point() -> BoltResult<CommitGuard> {
    check()?;
    Ok(uninterruptible())
}

/// Run until the guard drops without cancellation checks (for operations
/// that must never stop half-way, such as transaction control).
pub(crate) fn uninterruptible() -> CommitGuard {
    let previous = ACTIVE.with(|a| match a.borrow_mut().as_mut() {
        Some(active) => std::mem::replace(&mut active.committing, true),
        None => false,
    });
    CommitGuard { previous }
}

/// The active token of this thread, if any (to carry it into work that
/// runs later, such as a lazily-pulled result stream).
pub(crate) fn active_token() -> Option<CancellationToken> {
    ACTIVE.with(|a| a.borrow().as_ref().map(|act| act.token.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_cancelled(r: BoltResult<()>, want: CancelReason) -> bool {
        matches!(r, Err(BoltError::Cancelled(got)) if got == want)
    }

    #[test]
    fn new_token_is_not_cancelled() {
        let t = CancellationToken::new();
        assert!(!t.is_cancelled());
        assert_eq!(t.reason(), None);
        assert!(t.check().is_ok());
        assert_eq!(t.deadline(), None);
    }

    #[test]
    fn cancel_is_shared_by_clones_and_across_threads() {
        let t = CancellationToken::new();
        let c = t.clone();
        std::thread::spawn(move || c.cancel()).join().unwrap();
        assert!(t.is_cancelled());
        assert!(is_cancelled(t.check(), CancelReason::Cancelled));
    }

    #[test]
    fn deadline_expires_and_keeps_the_earlier_one() {
        let t = CancellationToken::new();
        let past = t.with_deadline(Instant::now() - Duration::from_millis(1));
        assert!(is_cancelled(past.check(), CancelReason::DeadlineExceeded));
        let far = t.with_timeout(Duration::from_secs(3600));
        assert!(!far.is_cancelled());
        let earlier = far.with_timeout(Duration::from_secs(60));
        assert!(earlier.deadline() < far.deadline());
        assert_eq!(
            earlier.with_timeout(Duration::from_secs(7200)).deadline(),
            earlier.deadline()
        );
        // A timeout too large for `Instant` means no deadline.
        assert!(!t.with_timeout(Duration::MAX).is_cancelled());
    }

    #[test]
    fn derived_token_shares_the_flag_and_cancel_wins_over_deadline() {
        let t = CancellationToken::new();
        let d = t.with_deadline(Instant::now() - Duration::from_millis(1));
        t.cancel();
        assert!(is_cancelled(d.check(), CancelReason::Cancelled));
    }

    #[test]
    fn check_row_polls_every_interval() {
        let t = CancellationToken::new();
        t.cancel();
        let _g = scope(&t);
        assert!(check_row(1).is_ok());
        assert!(check_row(ROWS_PER_CHECK - 1).is_ok());
        assert!(is_cancelled(check_row(0), CancelReason::Cancelled));
        assert!(is_cancelled(
            check_row(ROWS_PER_CHECK),
            CancelReason::Cancelled
        ));
    }

    #[test]
    fn check_without_scope_is_ok() {
        assert!(check().is_ok());
        assert!(active_token().is_none());
    }

    #[test]
    fn scope_installs_and_restores_nested_tokens() {
        let outer = CancellationToken::new();
        let inner = CancellationToken::new();
        inner.cancel();
        {
            let _g = scope(&outer);
            assert!(check().is_ok());
            {
                let _g = scope(&inner);
                assert!(is_cancelled(check(), CancelReason::Cancelled));
            }
            assert!(check().is_ok());
            outer.cancel();
            assert!(is_cancelled(check(), CancelReason::Cancelled));
        }
        assert!(check().is_ok());
    }

    #[test]
    fn commit_point_suspends_until_the_guard_drops() {
        let t = CancellationToken::new();
        let _g = scope(&t);
        {
            let _commit = commit_point().unwrap();
            t.cancel();
            // Past the commit point the operation completes...
            assert!(check().is_ok());
            // ...including nested commit points inside it.
            let nested = commit_point().unwrap();
            drop(nested);
            assert!(check().is_ok());
        }
        // After it, the statement sees the cancellation again.
        assert!(is_cancelled(check(), CancelReason::Cancelled));
        assert!(commit_point().is_err());
        let _u = uninterruptible();
        assert!(check().is_ok());
    }

    #[test]
    fn cancelled_commit_point_does_not_suspend() {
        let t = CancellationToken::new();
        t.cancel();
        let _g = scope(&t);
        assert!(is_cancelled(
            commit_point().map(drop),
            CancelReason::Cancelled
        ));
        assert!(is_cancelled(check(), CancelReason::Cancelled));
    }

    #[test]
    fn guards_without_scope_are_no_ops() {
        let _c = commit_point().unwrap();
        let _u = uninterruptible();
        assert!(check().is_ok());
    }

    #[test]
    fn error_display_names_the_reason() {
        assert_eq!(
            BoltError::Cancelled(CancelReason::Cancelled).to_string(),
            "query cancelled"
        );
        assert_eq!(
            BoltError::Cancelled(CancelReason::DeadlineExceeded).to_string(),
            "query deadline exceeded"
        );
        assert_eq!(BoltError::Cancelled(CancelReason::Cancelled).span(), None);
    }
}
