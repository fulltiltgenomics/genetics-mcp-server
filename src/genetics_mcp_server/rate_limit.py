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
    logger.info(
        f"Rate limit configured: {max_per_hour}/hour, {max_per_day}/day, {max_per_week}/week"
    )


def _wait_text(seconds: float) -> str:
    minutes = max(1, -(-int(seconds) // 60))
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    if hours < 24:
        return f"{hours} h {rest} min" if rest else f"{hours} h"
    days, hours = divmod(hours, 24)
    return f"{days} d {hours} h" if hours else f"{days} d"


def check_rate_limit(user: str) -> tuple[bool, str | None, int | None]:
    """Check if user is within the hourly, daily and weekly rate limits.

    Returns (allowed, reason, retry_after_s). For a refusal, reason names the limit and how
    long until it lifts, and retry_after_s is that wait in seconds; both are None if allowed.
    """
    now = time.monotonic()

    with _lock:
        # the weekly window is the longest, so it bounds what is worth keeping
        _requests[user] = timestamps = [t for t in _requests[user] if t > now - _WEEK]

        # a refusal reports the longest wait among the limits it hit: quoting the hourly
        # one to a user who is also over the daily one promises a retry that will fail
        refusal: tuple[float, str] | None = None
        for window, limit, name in (
            (_HOUR, _max_per_hour, "hourly limit {}/hour"),
            (_DAY, _max_per_day, "daily limit {}/day"),
            (_WEEK, _max_per_week, "weekly limit {}/week"),
        ):
            in_window = [t for t in timestamps if t > now - window]
            if len(in_window) < limit:
                continue
            # the request that must age out before one more fits; a limit of 0 never lifts
            wait = in_window[len(in_window) - limit] + window - now if limit > 0 else window
            if refusal is None or wait > refusal[0]:
                refusal = (wait, name.format(limit))

        if refusal is not None:
            wait, name = refusal
            return False, f"{name}, lifts in about {_wait_text(wait)}", max(1, int(wait) + 1)

        timestamps.append(now)
        return True, None, None
