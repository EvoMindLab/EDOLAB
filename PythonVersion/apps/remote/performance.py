"""Bounded process-local diagnostics; never retain URLs, SQL or request bodies."""

from collections import Counter, deque
from contextlib import contextmanager
from functools import wraps
import math
import threading
import time


class PerformanceMetrics:
    def __init__(self, *, capacity=256, operation_limit=128, slow_limit=64):
        self.capacity = capacity
        self.operation_limit = operation_limit
        self._lock = threading.Lock()
        self._operations = {}
        self._counters = Counter()
        self._slow = deque(maxlen=slow_limit)
        self._active = self._peak = 0

    def request_started(self):
        with self._lock:
            self._active += 1
            self._peak = max(self._peak, self._active)

    def request_finished(self):
        with self._lock:
            self._active -= 1

    def increment(self, name, value=1):
        with self._lock:
            key = name if name in self._counters or len(self._counters) < self.operation_limit else "other"
            self._counters[key] += value

    def record(self, name, seconds, *, failed=False):
        with self._lock:
            key = name if name in self._operations or len(self._operations) < self.operation_limit else "other"
            entry = self._operations.setdefault(key, {"count": 0, "failures": 0, "max_seconds": 0.0, "durations": deque(maxlen=self.capacity)})
            entry["count"] += 1
            entry["failures"] += int(failed)
            entry["max_seconds"] = max(entry["max_seconds"], seconds)
            entry["durations"].append(seconds)
            if seconds >= 1:
                self._slow.append({"operation": key, "seconds": seconds, "failed": failed, "timestamp": time.time()})

    @contextmanager
    def measure(self, name):
        start = time.perf_counter()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self.record(name, time.perf_counter() - start, failed=failed)

    def snapshot(self):
        with self._lock:
            operations = {}
            for name, entry in self._operations.items():
                values = sorted(entry["durations"])
                operations[name] = {k: v for k, v in entry.items() if k != "durations"}
                operations[name].update(window_count=len(values), p50_seconds=values[math.ceil(len(values) * .5) - 1], p95_seconds=values[math.ceil(len(values) * .95) - 1])
            return {"active_requests": self._active, "peak_requests": self._peak, "window_capacity": self.capacity, "operations": operations, "counters": dict(self._counters), "slow_operations": list(self._slow)}


metrics = PerformanceMetrics()


def measured(name):
    def decorate(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            with metrics.measure(name):
                return function(*args, **kwargs)
        return wrapper
    return decorate


def sample_stacks(seconds=5.0, interval=0.01, depth=6):
    """Sample every live thread's Python stack and count where each one spends its time.

    Only code locations (file:line function) are kept, never local variables, so the result is safe to show.
    A thread that sits in the same innermost frames in most samples is the one keeping the process busy.
    """
    import sys

    names = {thread.ident: thread.name for thread in threading.enumerate()}
    me = threading.get_ident()
    stacks: dict = {}
    totals: Counter = Counter()
    samples = 0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for ident, frame in sys._current_frames().items():
            if ident == me:
                continue
            path = []
            while frame is not None and len(path) < depth:
                code = frame.f_code
                path.append(f"{code.co_filename.rsplit('/', 2)[-1]}:{frame.f_lineno} {code.co_name}")
                frame = frame.f_back
            name = names.get(ident, str(ident))
            stacks.setdefault(name, Counter())[" <- ".join(path)] += 1
            totals[name] += 1
        samples += 1
        time.sleep(interval)
    return {
        "samples": samples,
        "threads": [
            {"thread": name, "samples": totals[name],
             "top": [{"count": n, "stack": s} for s, n in stacks[name].most_common(4)]}
            for name in sorted(stacks, key=lambda key: -totals[key])
        ],
    }
