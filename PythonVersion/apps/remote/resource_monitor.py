"""Background scheduler for durable server resource sampling.

The scheduler deliberately knows nothing about SSH or SQLite.  RelayService
provides narrow callbacks for eligibility, one sampling attempt, settings, and
retention cleanup.  This keeps lifecycle/concurrency policy independently
testable while every durable operation continues to use short-lived database
connections.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import threading
import time


class ResourceMonitor:
    """Schedule at most one sampling attempt per server at a time."""

    def __init__(
        self,
        *,
        eligible_server_ids: Callable[[], Sequence[str]],
        sample_once: Callable[[str], bool],
        sample_interval_seconds: Callable[[], float],
        prune_history: Callable[[], object],
        cleanup_interval_seconds: float = 300.0,
        maximum_backoff_seconds: float = 60.0,
        max_concurrent_samples: int = 4,
    ) -> None:
        if cleanup_interval_seconds <= 0:
            raise ValueError("cleanup_interval_seconds must be positive")
        if maximum_backoff_seconds <= 0:
            raise ValueError("maximum_backoff_seconds must be positive")
        if max_concurrent_samples < 1:
            raise ValueError("max_concurrent_samples must be at least 1")
        self._eligible_server_ids = eligible_server_ids
        self._sample_once = sample_once
        self._sample_interval_seconds = sample_interval_seconds
        self._prune_history = prune_history
        self._cleanup_interval_seconds = float(cleanup_interval_seconds)
        self._maximum_backoff_seconds = float(maximum_backoff_seconds)
        self._max_concurrent_samples = int(max_concurrent_samples)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._scheduler: threading.Thread | None = None
        self._lock = threading.RLock()
        self._workers: dict[str, threading.Thread] = {}
        self._next_due: dict[str, float] = {}
        self._failure_counts: dict[str, int] = {}

    def start(self) -> None:
        with self._lock:
            if self._scheduler is not None and self._scheduler.is_alive():
                if self._stop.is_set():
                    raise RuntimeError(
                        "resource monitor is still stopping; wait for shutdown "
                        "to finish before restarting"
                    )
                return
            self._stop.clear()
            self._wake.clear()
            self._scheduler = threading.Thread(
                target=self._scheduler_loop,
                name="edolab-relay-resource-monitor",
                daemon=True,
            )
            self._scheduler.start()

    def stop(self, timeout: float | None = 10.0) -> None:
        self.request_stop()
        self.join(timeout=timeout)

    def request_stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def join(self, timeout: float | None = 10.0) -> None:
        budget = 10.0 if timeout is None else max(0.0, float(timeout))
        deadline = time.monotonic() + budget
        scheduler = self._scheduler
        if scheduler is not None:
            scheduler.join(timeout=max(0.0, deadline - time.monotonic()))
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))

    def wake(self, *, reschedule: bool = False) -> None:
        if reschedule:
            with self._lock:
                now = time.monotonic()
                for server_id in self._next_due:
                    self._next_due[server_id] = now
        self._wake.set()

    @property
    def running(self) -> bool:
        scheduler = self._scheduler
        return scheduler is not None and scheduler.is_alive()

    def _scheduler_loop(self) -> None:
        next_cleanup = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_cleanup:
                try:
                    self._prune_history()
                except Exception:
                    # Cleanup must never suppress live monitoring.  SQLite
                    # errors are retried on the next bounded maintenance tick.
                    pass
                next_cleanup = now + self._cleanup_interval_seconds

            try:
                eligible = set(self._eligible_server_ids())
                interval = max(1.0, float(self._sample_interval_seconds()))
            except Exception:
                eligible = set()
                interval = 3.0

            with self._lock:
                for server_id, worker in tuple(self._workers.items()):
                    if not worker.is_alive():
                        self._workers.pop(server_id, None)
                for server_id in tuple(self._next_due):
                    if server_id not in eligible:
                        self._next_due.pop(server_id, None)
                        self._failure_counts.pop(server_id, None)
                for server_id in sorted(eligible):
                    active_count = sum(
                        worker.is_alive() for worker in self._workers.values()
                    )
                    if active_count >= self._max_concurrent_samples:
                        break
                    self._next_due.setdefault(server_id, now)
                    worker = self._workers.get(server_id)
                    if (
                        (worker is None or not worker.is_alive())
                        and now >= self._next_due[server_id]
                        and not self._stop.is_set()
                    ):
                        thread = threading.Thread(
                            target=self._sample_worker,
                            args=(server_id, interval),
                            name=f"edolab-relay-resource-{server_id[:8]}",
                            daemon=True,
                        )
                        self._workers[server_id] = thread
                        # Reserve the nominal next slot before starting.  The
                        # worker may push it further out after a failure.
                        self._next_due[server_id] = now + interval
                        thread.start()

            due_times = [
                due
                for server_id, due in self._next_due.items()
                if server_id in eligible
            ]
            next_due = min(due_times, default=now + 0.5)
            wait_seconds = max(0.05, min(0.5, next_due - now))
            self._wake.wait(wait_seconds)
            self._wake.clear()

    def _sample_worker(self, server_id: str, interval: float) -> None:
        succeeded = False
        try:
            succeeded = bool(self._sample_once(server_id))
        except Exception:
            succeeded = False
        finally:
            completed = time.monotonic()
            try:
                current_interval = max(
                    1.0, float(self._sample_interval_seconds())
                )
            except Exception:
                current_interval = interval
            with self._lock:
                if succeeded:
                    self._failure_counts.pop(server_id, None)
                    # The scheduler reserves ``started_at + interval`` before
                    # launching this worker.  Keep that deadline so the
                    # configured cadence is start-to-start; replacing it with
                    # ``completed + interval`` would silently add SSH latency
                    # to every sampling period.  A concurrent settings update
                    # may also have moved the deadline earlier via
                    # ``wake(reschedule=True)``, which must be preserved.
                else:
                    failures = self._failure_counts.get(server_id, 0) + 1
                    self._failure_counts[server_id] = failures
                    delay = min(
                        self._maximum_backoff_seconds,
                        current_interval * (2 ** min(failures, 6)),
                    )
                    self._next_due[server_id] = completed + delay
            self._wake.set()


__all__ = ["ResourceMonitor"]
