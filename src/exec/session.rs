// SPDX-License-Identifier: Apache-2.0
//! Shared runtime and sessions.
//!
//! An [`Engine`] owns process-global CUDA state and is used from one thread
//! at a time. A [`Database`] turns it into a shared service: it builds the
//! engine on a dedicated **runtime thread** and owns it there, and any number
//! of [`Session`]s — cloned freely, used from any thread — submit statements
//! to it through a bounded FIFO queue. Statements execute one at a time in
//! arrival order (the GPU runtime is not shared between concurrent kernels
//! from different queries), so many clients get fair, bounded-resource access
//! without each needing its own CUDA context.
//!
//! ```ignore
//! use craton_bolt::{Database, Engine};
//!
//! let db = Database::open(Engine::builder().database("warehouse"))?;
//! let s = db.session();
//! s.sql("CREATE TABLE t AS SELECT * FROM (VALUES (1), (2)) AS v(x)")?;
//! let rows = s.sql("SELECT SUM(x) FROM t")?;
//! // From another thread (or async task):
//! let pending = db.session().submit("SELECT COUNT(*) FROM t");
//! let rows = pending.wait()?;
//! ```
//!
//! [`Session::submit`] returns a [`PendingResult`] that can be waited on,
//! polled as a [`Future`], or dropped to cancel a statement that has not
//! started. A panic inside one statement is caught and reported as that
//! statement's error; the runtime keeps serving.
//!
//! Temporary tables (`CREATE TEMP TABLE`) are database-wide: every session
//! sees them, and they disappear when the database closes.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use arrow_array::RecordBatch;

use crate::error::{BoltError, BoltResult};
use crate::exec::engine::{Engine, EngineBuilder};

type Job = Box<dyn FnOnce(&mut Engine) + Send>;

/// Default capacity of the statement queue.
pub const DEFAULT_QUEUE_CAPACITY: usize = 1024;

/// Counters of a [`Database`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct DatabaseStats {
    /// Sessions opened.
    pub sessions: u64,
    /// Statements executed (success or failure).
    pub statements: u64,
    /// Statements that returned an error (panics included).
    pub failed: u64,
    /// Statements cancelled before they started.
    pub cancelled: u64,
    /// Total time statements waited in the queue.
    pub total_queue_wait: Duration,
    /// Longest single queue wait.
    pub max_queue_wait: Duration,
}

#[derive(Default)]
struct Counters {
    sessions: AtomicU64,
    statements: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
    wait_nanos: AtomicU64,
    max_wait_nanos: AtomicU64,
}

struct Inner {
    tx: Mutex<Option<SyncSender<Job>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    // Jobs hold only the counters, never `Inner`: the last handle must not
    // drop (and join the runtime) from the runtime thread itself.
    counters: Arc<Counters>,
    next_session: AtomicU64,
}

impl Inner {
    fn sender(&self) -> BoltResult<SyncSender<Job>> {
        self.tx
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .ok_or_else(|| BoltError::Plan("the database is closed".into()))
    }

    fn shutdown(&self) {
        // Dropping the sender ends the runtime loop once the queue drains.
        self.tx.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(h) = self.worker.lock().unwrap_or_else(|p| p.into_inner()).take() {
            let _ = h.join();
        }
    }
}

/// A shared GPU runtime serving [`Session`]s. See the [module docs](self).
///
/// Cloning is cheap (a handle). The runtime shuts down — finishing queued
/// statements, then dropping the engine on its own thread — when
/// [`Database::close`] is called or the last handle *and* session drop.
#[derive(Clone)]
pub struct Database {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Database {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Database")
            .field("stats", &self.stats())
            .finish()
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl Database {
    /// Build an engine from `builder` on a new runtime thread and serve it.
    ///
    /// # Errors
    /// The engine could not be built (no device, another engine is active,
    /// the catalog is locked, …).
    pub fn open(builder: EngineBuilder) -> BoltResult<Database> {
        Self::open_with_capacity(builder, DEFAULT_QUEUE_CAPACITY)
    }

    /// [`Database::open`] with a statement-queue capacity: a submission
    /// blocks ([`Session::submit`]) or fails ([`Session::try_submit`]) while
    /// `capacity` statements are already queued.
    pub fn open_with_capacity(builder: EngineBuilder, capacity: usize) -> BoltResult<Database> {
        let (tx, rx) = mpsc::sync_channel::<Job>(capacity.max(1));
        let (ready_tx, ready_rx) = mpsc::channel::<BoltResult<()>>();
        let worker = std::thread::Builder::new()
            .name("bolt-runtime".into())
            .spawn(move || {
                let mut engine = match builder.build() {
                    Ok(e) => {
                        let _ = ready_tx.send(Ok(()));
                        e
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                while let Ok(job) = rx.recv() {
                    job(&mut engine);
                }
            })
            .map_err(|e| BoltError::Other(format!("starting the runtime thread: {e}")))?;
        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = worker.join();
                return Err(e);
            }
            Err(_) => {
                let _ = worker.join();
                return Err(BoltError::Other(
                    "the runtime thread exited during startup".into(),
                ));
            }
        }
        Ok(Database {
            inner: Arc::new(Inner {
                tx: Mutex::new(Some(tx)),
                worker: Mutex::new(Some(worker)),
                counters: Arc::new(Counters::default()),
                next_session: AtomicU64::new(1),
            }),
        })
    }

    /// Open a new session.
    pub fn session(&self) -> Session {
        self.inner.counters.sessions.fetch_add(1, Ordering::Relaxed);
        Session {
            inner: Arc::clone(&self.inner),
            id: self.inner.next_session.fetch_add(1, Ordering::Relaxed),
            settings: Arc::new(Mutex::new(SessionSettings::default())),
            statements: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Run `f` with exclusive access to the engine on the runtime thread
    /// (registration, configuration, anything beyond SQL), in queue order.
    pub fn with_engine<R, F>(&self, f: F) -> BoltResult<R>
    where
        R: Send + 'static,
        F: FnOnce(&mut Engine) -> R + Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        let job: Job = Box::new(move |e| {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(e)));
            let _ = tx.send(r);
        });
        self.inner
            .sender()?
            .send(job)
            .map_err(|_| BoltError::Plan("the database is closed".into()))?;
        match rx.recv() {
            Ok(Ok(r)) => Ok(r),
            Ok(Err(p)) => Err(BoltError::Other(format!(
                "engine task panicked: {}",
                panic_message(&p)
            ))),
            Err(_) => Err(BoltError::Other("the runtime thread stopped".into())),
        }
    }

    /// Counters since open.
    pub fn stats(&self) -> DatabaseStats {
        let c = &self.inner.counters;
        DatabaseStats {
            sessions: c.sessions.load(Ordering::Relaxed),
            statements: c.statements.load(Ordering::Relaxed),
            failed: c.failed.load(Ordering::Relaxed),
            cancelled: c.cancelled.load(Ordering::Relaxed),
            total_queue_wait: Duration::from_nanos(c.wait_nanos.load(Ordering::Relaxed)),
            max_queue_wait: Duration::from_nanos(c.max_wait_nanos.load(Ordering::Relaxed)),
        }
    }

    /// Stop accepting statements, finish the queued ones, and drop the
    /// engine. Sessions then fail with "the database is closed".
    pub fn close(&self) {
        self.inner.shutdown();
    }
}

fn panic_message(p: &Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".into())
}

/// Per-session settings.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SessionSettings {
    /// Reject statements that change tables (queries only).
    pub read_only: bool,
}

/// A client's handle on a [`Database`]. Cheap to clone; clones share the
/// session's settings and counters.
#[derive(Clone)]
pub struct Session {
    inner: Arc<Inner>,
    id: u64,
    settings: Arc<Mutex<SessionSettings>>,
    statements: Arc<AtomicU64>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("id", &self.id)
            .field("settings", &self.settings())
            .finish()
    }
}

struct Slot {
    result: Option<BoltResult<RecordBatch>>,
    waker: Option<Waker>,
}

struct Shared {
    slot: Mutex<Slot>,
    ready: std::sync::Condvar,
    cancelled: AtomicBool,
}

/// The eventual result of a [`Session::submit`]ted statement.
///
/// Wait with [`PendingResult::wait`] or `.await` it. Dropping it before the
/// statement starts cancels the statement.
pub struct PendingResult {
    shared: Arc<Shared>,
}

impl PendingResult {
    fn failed(e: BoltError) -> Self {
        let shared = Arc::new(Shared {
            slot: Mutex::new(Slot {
                result: Some(Err(e)),
                waker: None,
            }),
            ready: std::sync::Condvar::new(),
            cancelled: AtomicBool::new(false),
        });
        PendingResult { shared }
    }

    /// Block until the statement finishes.
    pub fn wait(self) -> BoltResult<RecordBatch> {
        let mut slot = self.shared.slot.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(r) = slot.result.take() {
                return r;
            }
            slot = self
                .shared
                .ready
                .wait(slot)
                .unwrap_or_else(|p| p.into_inner());
        }
    }

    /// The result if the statement has finished.
    pub fn try_take(&mut self) -> Option<BoltResult<RecordBatch>> {
        self.shared
            .slot
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .result
            .take()
    }
}

impl Drop for PendingResult {
    fn drop(&mut self) {
        self.shared.cancelled.store(true, Ordering::Release);
    }
}

impl Future for PendingResult {
    type Output = BoltResult<RecordBatch>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut slot = self.shared.slot.lock().unwrap_or_else(|p| p.into_inner());
        match slot.result.take() {
            Some(r) => Poll::Ready(r),
            None => {
                slot.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

impl Session {
    /// Session id (unique within the database).
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Current settings.
    pub fn settings(&self) -> SessionSettings {
        self.settings
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Make the session read-only (queries only) or writable.
    pub fn set_read_only(&self, read_only: bool) {
        self.settings
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .read_only = read_only;
    }

    /// Statements this session has submitted.
    pub fn statement_count(&self) -> u64 {
        self.statements.load(Ordering::Relaxed)
    }

    /// Execute `sql` (one or more statements, see
    /// [`Engine::execute_sql`]) and wait for the result.
    pub fn sql(&self, sql: &str) -> BoltResult<RecordBatch> {
        self.submit(sql).wait()
    }

    /// Queue `sql`, blocking while the queue is full; returns immediately
    /// with a [`PendingResult`].
    pub fn submit(&self, sql: &str) -> PendingResult {
        match self.make_job(sql) {
            Ok((job, pending)) => match self.inner.sender() {
                Ok(tx) => match tx.send(job) {
                    Ok(()) => pending,
                    Err(_) => {
                        PendingResult::failed(BoltError::Plan("the database is closed".into()))
                    }
                },
                Err(e) => PendingResult::failed(e),
            },
            Err(e) => PendingResult::failed(e),
        }
    }

    /// [`Session::submit`] without blocking: fails when the queue is full.
    pub fn try_submit(&self, sql: &str) -> BoltResult<PendingResult> {
        let (job, pending) = self.make_job(sql)?;
        match self.inner.sender()?.try_send(job) {
            Ok(()) => Ok(pending),
            Err(TrySendError::Full(_)) => {
                Err(BoltError::Memory("the statement queue is full".into()))
            }
            Err(TrySendError::Disconnected(_)) => {
                Err(BoltError::Plan("the database is closed".into()))
            }
        }
    }

    fn make_job(&self, sql: &str) -> BoltResult<(Job, PendingResult)> {
        self.statements.fetch_add(1, Ordering::Relaxed);
        let read_only = self.settings().read_only;
        let shared = Arc::new(Shared {
            slot: Mutex::new(Slot {
                result: None,
                waker: None,
            }),
            ready: std::sync::Condvar::new(),
            cancelled: AtomicBool::new(false),
        });
        let worker_side = Arc::clone(&shared);
        let counters = Arc::clone(&self.inner.counters);
        let sql = sql.to_string();
        let queued = Instant::now();
        let job: Job = Box::new(move |engine: &mut Engine| {
            let c = &counters;
            if worker_side.cancelled.load(Ordering::Acquire) {
                c.cancelled.fetch_add(1, Ordering::Relaxed);
                return;
            }
            let waited = queued.elapsed().as_nanos() as u64;
            c.wait_nanos.fetch_add(waited, Ordering::Relaxed);
            c.max_wait_nanos.fetch_max(waited, Ordering::Relaxed);
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if read_only {
                    engine.sql(&sql)
                } else {
                    engine.execute_sql(&sql)
                }
                .map(crate::exec::engine::QueryHandle::into_record_batch)
            }));
            let result = match run {
                Ok(r) => r,
                Err(p) => Err(BoltError::Other(format!(
                    "statement panicked: {}",
                    panic_message(&p)
                ))),
            };
            c.statements.fetch_add(1, Ordering::Relaxed);
            if result.is_err() {
                c.failed.fetch_add(1, Ordering::Relaxed);
            }
            let mut slot = worker_side.slot.lock().unwrap_or_else(|p| p.into_inner());
            slot.result = Some(result);
            if let Some(w) = slot.waker.take() {
                w.wake();
            }
            worker_side.ready.notify_all();
        });
        Ok((job, PendingResult { shared }))
    }
}
