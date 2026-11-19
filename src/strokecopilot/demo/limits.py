"""The Space's spending and fairness limits (DEC-32 §5, §7). All in memory: a restart or a sleep resets them, so the
hard cap is the Anthropic workspace's monthly limit, which Parsa sets. These are the soft ones in front of it.

- `RateLimiter`: live runs per visitor per hour and per day. A visitor is a salted hash of the client address, kept
  only in memory and never logged.
- `SpendCap`: estimated spend per UTC day; a run is admitted only if the cap leaves room for a cold run's estimate.
- `Slots`: at most N live runs at once; the next one falls back at once instead of waiting.
- `ResultCache`: finished live runs by input hash, so a repeat costs nothing.
Every class takes a `clock` (seconds) so the tests can move time.
"""

import hashlib
import os
import threading
import time
from collections import OrderedDict, deque

PER_HOUR = 3
PER_DAY = 10
DAILY_CAP_USD = 1.00          # DEC-32 §7 (Parsa: fewer than 20 live runs expected); a Space variable can change it
COLD_RUN_ESTIMATE_USD = 0.12  # EXP-37's logged usage at live prices, cache cold (DEC-32 §7); the dummy measures it
MAX_CONCURRENT = 3
CACHE_SIZE = 500


def visitor_id(address, salt=None):
    """A salted hash of the client address: enough to count runs, useless to identify anyone."""
    salt = salt if salt is not None else os.environ.get("SPACE_SIGNING_SECRET", "local")
    return hashlib.sha256(f"{salt}:{address}".encode()).hexdigest()[:16]


class RateLimiter:
    def __init__(self, per_hour=PER_HOUR, per_day=PER_DAY, clock=time.time):
        self.per_hour, self.per_day, self.clock = per_hour, per_day, clock
        self._runs = {}
        self._lock = threading.Lock()

    def _recent(self, visitor, now):
        runs = self._runs.setdefault(visitor, deque())
        while runs and now - runs[0] > 86400:
            runs.popleft()
        return runs

    def allow(self, visitor):
        """(allowed, reason). Records the run when allowed."""
        with self._lock:
            now = self.clock()
            runs = self._recent(visitor, now)
            if sum(now - t <= 3600 for t in runs) >= self.per_hour:
                return False, f"you have used your {self.per_hour} live runs for this hour"
            if len(runs) >= self.per_day:
                return False, f"you have used your {self.per_day} live runs for today"
            runs.append(now)
            return True, ""


class SpendCap:
    def __init__(self, cap_usd=DAILY_CAP_USD, run_estimate_usd=COLD_RUN_ESTIMATE_USD, clock=time.time):
        self.cap, self.estimate, self.clock = float(cap_usd), float(run_estimate_usd), clock
        self._day, self.spent, self.reserved = None, 0.0, 0.0
        self._lock = threading.Lock()

    def _roll(self):
        day = int(self.clock() // 86400)
        if day != self._day:
            self._day, self.spent, self.reserved = day, 0.0, 0.0

    def reserve(self):
        """Hold a cold run's estimate. (allowed, reason)."""
        with self._lock:
            self._roll()
            if self.spent + self.reserved + self.estimate > self.cap:
                return False, "today's budget for live runs is used up"
            self.reserved += self.estimate
            return True, ""

    def settle(self, actual_usd):
        """Replace a reservation with what the run actually cost (0 for a run that made no call)."""
        with self._lock:
            self._roll()
            self.reserved = max(0.0, self.reserved - self.estimate)
            self.spent += max(0.0, float(actual_usd or 0.0))

    def status(self):
        with self._lock:
            self._roll()
            return {"spent_today_usd": round(self.spent, 4), "cap_usd": self.cap, "reserved_usd": round(self.reserved, 4)}


class Slots:
    def __init__(self, n=MAX_CONCURRENT):
        self._sem = threading.BoundedSemaphore(n)
        self.n = n

    def try_acquire(self):
        return self._sem.acquire(blocking=False)

    def release(self):
        self._sem.release()


class ResultCache:
    def __init__(self, size=CACHE_SIZE):
        self.size = size
        self._items = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            return None

    def put(self, key, value):
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def __len__(self):
        return len(self._items)
