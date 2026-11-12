"""The three sliding windows of the per-user chat limiter."""

import pytest

from genetics_mcp_server import rate_limit

HOUR = rate_limit._HOUR
DAY = rate_limit._DAY


@pytest.fixture
def clock(monkeypatch):
    """A monotonic clock the test advances by hand; the limiter never sleeps."""
    state = {"now": 1_000_000.0}
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: state["now"])
    rate_limit._requests.clear()
    yield state
    rate_limit._requests.clear()


@pytest.fixture
def limits(monkeypatch):
    monkeypatch.setattr(rate_limit, "_max_per_hour", 2)
    monkeypatch.setattr(rate_limit, "_max_per_day", 3)
    monkeypatch.setattr(rate_limit, "_max_per_week", 4)


def _send(user="u", n=1):
    results = [rate_limit.check_rate_limit(user) for _ in range(n)]
    return results[-1]


def test_hourly_limit_names_itself_and_lifts_after_an_hour(clock, limits):
    assert _send(n=2) == (True, None, None)
    assert _send() == (False, "hourly limit 2/hour, lifts in about 1 h", HOUR + 1)
    clock["now"] += HOUR + 1
    assert _send() == (True, None, None)


def test_daily_limit_binds_after_the_hourly_window_has_rolled(clock, limits):
    _send(n=2)
    clock["now"] += HOUR + 1
    assert _send() == (True, None, None)
    assert _send()[1].startswith("daily limit 3/day, lifts in about 23 h")
    clock["now"] += DAY
    assert _send() == (True, None, None)


def test_weekly_limit_binds_across_days_and_lifts_after_seven(clock, limits):
    for _ in range(2):
        _send(n=2)
        clock["now"] += DAY + 1
    assert _send()[1].endswith("weekly limit 4/week, lifts in about 5 d")
    # the first day's two requests age out of the week only once seven days have passed
    clock["now"] += 5 * DAY
    assert _send() == (True, None, None)


def test_users_do_not_share_a_window(clock, limits):
    _send("a", n=2)
    assert _send("a")[0] is False
    assert _send("b") == (True, None, None)


def test_refusal_quotes_the_longest_wait_among_the_limits_hit(clock, limits):
    _send(n=2)
    clock["now"] += HOUR + 1
    _send()
    # over the daily limit only: the hourly window holds one request
    allowed, reason, retry_after = _send()
    assert not allowed and reason.startswith("daily limit")
    assert retry_after == DAY - HOUR


def test_configure_sets_all_three_limits():
    rate_limit.configure(max_per_hour=1, max_per_day=2, max_per_week=3)
    assert (rate_limit._max_per_hour, rate_limit._max_per_day, rate_limit._max_per_week) == (
        1,
        2,
        3,
    )
