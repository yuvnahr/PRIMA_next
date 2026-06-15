"""Simple synchronous token-bucket rate limiter."""
import time
from threading import Lock
from typing import Optional


class RateLimiter:
    """Token-bucket limiter. Not distributed — suitable for single-process use.

    rate_per_minute: number of tokens refilled per minute (capacity)
    """

    def __init__(self, rate_per_minute: int = 60):
        self.capacity = max(1, int(rate_per_minute))
        self.tokens = self.capacity
        self.fill_interval = 60.0 / self.capacity
        self.last = time.monotonic()
        self.lock = Lock()

    def allow(self, cost: int = 1) -> bool:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last
            # refill based on elapsed time
            refill = int(elapsed / self.fill_interval)
            if refill > 0:
                self.tokens = min(self.capacity, self.tokens + refill)
                self.last = now
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False
