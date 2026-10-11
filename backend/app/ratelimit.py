"""Simple in-memory sliding-window rate limiter.

Per process only: limits reset on restart and are not shared between
processes. That matches the single-process deployment the in-process job
queue already assumes. Behind a reverse proxy, run uvicorn with
``--proxy-headers --forwarded-allow-ips=<proxy>`` so per-IP limits see the real
client address.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque


class RateLimiter:
    """Allow at most ``limit`` events per ``window_seconds`` per key. ``limit <= 0`` disables it."""

    _SWEEP_EVERY = 1024  # prune idle keys every N checks so memory stays bounded

    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window = float(window_seconds)
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._checks = 0

    def check(self, key: str) -> float | None:
        """Record one event for ``key``. Return None if allowed, else seconds until a slot frees."""
        if self.limit <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            self._checks += 1
            if self._checks % self._SWEEP_EVERY == 0:
                self._sweep(now)
            events = self._events.setdefault(key, deque())
            while events and events[0] <= now - self.window:
                events.popleft()
            if len(events) >= self.limit:
                return max(0.0, events[0] + self.window - now)
            events.append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def _sweep(self, now: float) -> None:
        stale = [key for key, events in self._events.items() if not events or events[-1] <= now - self.window]
        for key in stale:
            del self._events[key]


def retry_after_header(seconds: float) -> dict[str, str]:
    return {"Retry-After": str(max(1, math.ceil(seconds)))}
