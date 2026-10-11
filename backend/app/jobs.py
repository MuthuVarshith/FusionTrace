"""In-process scan queue with a fixed-size worker pool.

Scans are CPU-bound model inference, so they run on plain worker threads
(``FUSIONTRACE_MAX_CONCURRENT_SCANS``, default 1) instead of Starlette's
request thread pool. A scan stays ``queued`` in the database until a worker
picks it up. The queue lives in memory: run a single application process.
What happens to scans interrupted by a restart is decided at startup by
``ScanService.recover_interrupted_scans``.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

logger = logging.getLogger("fusiontrace")

_STOP = object()


class QueueFull(Exception):
    """The queue already holds the configured maximum number of waiting scans."""


class ScanQueue:
    def __init__(self, handler: Callable[[str], None], workers: int = 1, max_pending: int = 50) -> None:
        self._handler = handler
        self._worker_count = max(1, workers)
        self._max_pending = max(1, max_pending)
        self._queue: queue.Queue[object] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._active = 0

    @property
    def workers(self) -> int:
        return self._worker_count

    @property
    def running(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)

    def start(self) -> None:
        if self.running:
            return
        self._threads = [
            threading.Thread(target=self._work, name=f"scan-worker-{index}", daemon=True) for index in range(self._worker_count)
        ]
        for thread in self._threads:
            thread.start()
        logger.info("Scan queue started with %d worker(s)", self._worker_count)

    def stop(self, timeout: float = 5.0) -> None:
        """Ask workers to exit after their current scan. Waiting scans stay queued in the database."""
        for _ in self._threads:
            self._queue.put(_STOP)
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads = []

    def pending(self) -> int:
        """Scans waiting for a worker (excludes the ones being processed)."""
        return self._queue.qsize()

    def active(self) -> int:
        with self._lock:
            return self._active

    def submit(self, scan_id: str) -> None:
        if self.pending() >= self._max_pending:
            raise QueueFull(scan_id)
        self._queue.put(scan_id)

    def has_capacity(self) -> bool:
        return self.pending() < self._max_pending

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                with self._lock:
                    self._active += 1
                try:
                    self._handler(str(item))
                except Exception:  # the handler records failures itself; never let a worker die
                    logger.exception("Unhandled error while processing scan %s", item)
                finally:
                    with self._lock:
                        self._active -= 1
            finally:
                self._queue.task_done()

    def join(self) -> None:
        """Block until every submitted scan has been processed (tests)."""
        self._queue.join()
