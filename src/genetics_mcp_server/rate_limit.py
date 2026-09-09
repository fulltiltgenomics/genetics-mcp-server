"""Per-user rate limiting for chat API requests.

Sliding-window counters kept in memory, one list of timestamps per user, checked against
hourly, daily and weekly limits. Configured via environment variables:
    RATE_LIMIT_PER_HOUR: max requests per hour per user
    RATE_LIMIT_PER_DAY: max requests per day per user
    RATE_LIMIT_PER_WEEK: max requests per week per user
"""

import logging
import time
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)

_lock = Lock()
_requests: dict[str, list[float]] = defaultdict(list)

_max_per_hour: int = 20
_max_per_day: int = 40
_max_per_week: int = 100

_HOUR = 3600
_DAY = 86400
_WEEK = 7 * _DAY


def configure(max_per_hour: int, max_per_day: int, max_per_week: int) -> None:
    """Set rate limit parameters. Call once at startup."""
    global _max_per_hour, _max_per_day, _max_per_week
    _max_per_hour = max_per_hour
    _max_per_day = max_per_day
    _max_per_week = max_per_week
    logger.info(f"Rate limit configured: {max_per_hour}/hour, {max_per_day}/day, {max_per_week}/week")


def check_rate_limit(user: str) -> tuple[bool, str | None]:
    """Check if user is within the hourly, daily and weekly rate limits.

    Returns (allowed, reason) where reason is None if allowed or a description of which limit was hit.
    """
    now = time.monotonic()
    week_cutoff = now - _WEEK
    day_cutoff = now - _DAY
    hour_cutoff = now - _HOUR

    with _lock:
        # the weekly window is the longest, so it bounds what is worth keeping
        _requests[user] = timestamps = [t for t in _requests[user] if t > week_cutoff]

        hour_count = sum(1 for t in timestamps if t > hour_cutoff)
        day_count = sum(1 for t in timestamps if t > day_cutoff)

        if hour_count >= _max_per_hour:
            return False, f"hourly limit {_max_per_hour}/hour"

        if day_count >= _max_per_day:
            return False, f"daily limit {_max_per_day}/day"

        if len(timestamps) >= _max_per_week:
            return False, f"weekly limit {_max_per_week}/week"

        timestamps.append(now)
        return True, None
