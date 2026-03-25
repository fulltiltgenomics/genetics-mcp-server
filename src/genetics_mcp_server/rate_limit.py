"""Per-user rate limiting for chat API requests.

Uses sliding window counters stored in memory with both hourly and daily limits.
Configured via environment variables:
    RATE_LIMIT_PER_HOUR: max requests per hour per user (default: 20)
    RATE_LIMIT_PER_DAY: max requests per day per user (default: 100)
"""

import logging
import time
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)

_lock = Lock()
_requests: dict[str, list[float]] = defaultdict(list)

_max_per_hour: int = 20
_max_per_day: int = 100

_HOUR = 3600
_DAY = 86400


def configure(max_per_hour: int, max_per_day: int) -> None:
    """Set rate limit parameters. Call once at startup."""
    global _max_per_hour, _max_per_day
    _max_per_hour = max_per_hour
    _max_per_day = max_per_day
    logger.info(f"Rate limit configured: {max_per_hour}/hour, {max_per_day}/day")


def check_rate_limit(user: str) -> tuple[bool, str | None]:
    """Check if user is within both hourly and daily rate limits.

    Returns (allowed, reason) where reason is None if allowed or a description of which limit was hit.
    """
    now = time.monotonic()
    day_cutoff = now - _DAY
    hour_cutoff = now - _HOUR

    with _lock:
        # prune entries older than 24h
        _requests[user] = timestamps = [t for t in _requests[user] if t > day_cutoff]

        hour_count = sum(1 for t in timestamps if t > hour_cutoff)

        if hour_count >= _max_per_hour:
            return False, f"hourly limit {_max_per_hour}/hour"

        if len(timestamps) >= _max_per_day:
            return False, f"daily limit {_max_per_day}/day"

        timestamps.append(now)
        return True, None
