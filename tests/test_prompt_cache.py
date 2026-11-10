"""The zero-output requests that keep prompt-cache entries alive between real ones."""

import asyncio
from types import SimpleNamespace

import pytest

from genetics_mcp_server import prompt_cache
from genetics_mcp_server.prompt_cache import PromptCacheKeeper, cache_usage


def _message(read=1000, written=0, written_1h=None):
    usage = SimpleNamespace(
        input_tokens=3, cache_read_input_tokens=read, cache_creation_input_tokens=written
    )
    if written_1h is not None:
        usage.cache_creation = SimpleNamespace(ephemeral_1h_input_tokens=written_1h)
    return SimpleNamespace(usage=usage, model="claude-fable-5-2")


class _Harness:
    """A keeper on a clock that only moves when the keeper sleeps."""

    def __init__(self, *, keepalive_s=1200.0, prewarm=True, replies=None, stop_after=None):
        self.now = 0.0
        self.sent: list[tuple[float, dict]] = []
        self.spend: list[tuple[str, int, dict]] = []
        self.replies = list(replies or [])
        self.stop_after = stop_after
        self.done = asyncio.Event()
        self.keeper = PromptCacheKeeper(
            self._send,
            self._on_spend,
            keepalive_s=keepalive_s,
            prewarm=prewarm,
            clock=lambda: self.now,
            sleep=self._sleep,
        )

    async def _sleep(self, seconds):
        self.now += seconds
        await asyncio.sleep(0)

    async def _send(self, params):
        self.sent.append((self.now, params))
        if self.stop_after and len(self.sent) >= self.stop_after:
            self.done.set()
        reply = self.replies.pop(0) if self.replies else _message()
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def _on_spend(self, kind, messages, **who):
        self.spend.append((kind, len(messages), who))


REQUEST = {
    "model": "claude-fable-5-2",
    "max_tokens": 16384,
    "thinking": {"type": "adaptive", "display": "summarized"},
    "tools": [{"name": "lookup", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
    "system": [
        {"type": "text", "text": "SHARED", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"type": "text", "text": "THIS USER", "cache_control": {"type": "ephemeral"}},
    ],
    "messages": [{"role": "user", "content": "hi"}],
}


class TestCacheUsage:
    def test_reads_the_one_hour_share_of_a_write(self):
        assert cache_usage(_message(read=10, written=50, written_1h=20).usage) == (3, 10, 50, 20)

    def test_a_write_with_no_breakdown_is_all_one_hour(self):
        assert cache_usage(_message(read=10, written=50).usage) == (3, 10, 50, 50)


class TestKeepAlive:
    async def test_re_sends_the_turns_last_request_until_the_window_closes(self):
        h = _Harness()
        h.keeper.end_turn("s1", REQUEST, sent_at=0.0, user="u1", secret=False)
        await asyncio.gather(*h.keeper._sessions.values())

        # every 270s while under the 1200s window: the last one keeps the entry to 1380s
        assert [t for t, _ in h.sent] == [270.0, 540.0, 810.0, 1080.0]
        assert all(p == {**REQUEST, "max_tokens": 0} for _, p in h.sent)
        assert h.spend == [
            (
                "keep-alive",
                4,
                {"model": "claude-fable-5-2", "session_id": "s1", "user": "u1", "secret": False},
            )
        ]
        assert h.keeper._sessions == {}

    async def test_the_first_re_send_is_timed_from_when_the_request_went_out(self):
        """The entry lapses five minutes after the request STARTED; a long final answer
        has used most of that by the time the turn ends."""
        h = _Harness()
        h.now = 200.0
        h.keeper.end_turn("s1", REQUEST, sent_at=0.0, user=None, secret=False)
        await asyncio.gather(*h.keeper._sessions.values())
        assert h.sent[0][0] == 270.0

    async def test_the_next_turn_stops_it_and_what_was_sent_is_still_recorded(self):
        h = _Harness(stop_after=2)
        h.keeper.end_turn("s1", REQUEST, sent_at=0.0, user="u1", secret=False)
        task = h.keeper._sessions["s1"]
        await h.done.wait()
        h.keeper.begin_turn("s1")
        await asyncio.gather(task, return_exceptions=True)

        assert task.cancelled()
        assert len(h.sent) == 2
        assert [(kind, n) for kind, n, _ in h.spend] == [("keep-alive", 2)]

    async def test_stops_when_a_re_send_writes_instead_of_reading(self):
        h = _Harness(replies=[_message(), _message(read=0, written=5000)])
        h.keeper.end_turn("s1", REQUEST, sent_at=0.0, user=None, secret=False)
        await asyncio.gather(*h.keeper._sessions.values())
        assert len(h.sent) == 2
        assert [(kind, n) for kind, n, _ in h.spend] == [("keep-alive", 2)]

    async def test_a_failed_re_send_ends_the_run_quietly(self):
        h = _Harness(replies=[RuntimeError("overloaded")])
        h.keeper.end_turn("s1", REQUEST, sent_at=0.0, user=None, secret=False)
        await asyncio.gather(*h.keeper._sessions.values())
        assert len(h.sent) == 1
        assert h.spend == []

    @pytest.mark.parametrize("session_id, keepalive_s", [("s1", 0.0), (None, 1200.0)])
    async def test_nothing_is_scheduled_when_off_or_without_a_session(
        self, session_id, keepalive_s
    ):
        h = _Harness(keepalive_s=keepalive_s)
        h.keeper.end_turn(session_id, REQUEST, sent_at=0.0, user=None, secret=False)
        assert h.keeper._sessions == {}

    async def test_a_new_turn_end_replaces_the_previous_run(self):
        h = _Harness()
        h.keeper.end_turn("s1", REQUEST, sent_at=0.0, user=None, secret=False)
        first = h.keeper._sessions["s1"]
        h.keeper.end_turn("s1", REQUEST, sent_at=0.0, user=None, secret=False)
        await asyncio.gather(first, return_exceptions=True)
        assert first.cancelled()
        assert h.keeper._sessions["s1"] is not first
        await h.keeper.close()


class TestPrewarm:
    async def test_re_reads_the_shared_prefix_before_the_hour_is_up(self):
        h = _Harness(stop_after=2)
        h.keeper.note_prefix(REQUEST)
        await h.done.wait()
        await h.keeper.close()

        times = [t for t, _ in h.sent]
        assert times == [prompt_cache.PREWARM_INTERVAL_S, 2 * prompt_cache.PREWARM_INTERVAL_S]
        # the tools and the shared block only, exactly as the real request carries them
        assert h.sent[0][1] == {
            "model": REQUEST["model"],
            "thinking": REQUEST["thinking"],
            "tools": REQUEST["tools"],
            "system": REQUEST["system"][:1],
            "max_tokens": 0,
            "messages": [{"role": "user", "content": "warmup"}],
        }
        assert h.spend[0] == (
            "pre-warm",
            1,
            {"model": "claude-fable-5-2", "session_id": None, "user": None, "secret": False},
        )

    async def test_the_same_prefix_is_registered_once(self):
        h = _Harness()
        h.keeper.note_prefix(REQUEST)
        h.keeper.note_prefix({**REQUEST, "messages": [{"role": "user", "content": "other"}]})
        h.keeper.note_prefix({**REQUEST, "system": [REQUEST["system"][0]]})
        assert len(h.keeper._prefixes) == 1
        h.keeper.note_prefix({**REQUEST, "model": "claude-opus-5"})
        assert len(h.keeper._prefixes) == 2
        await h.keeper.close()

    async def test_a_prefix_nothing_uses_is_dropped(self):
        h = _Harness()
        h.keeper.note_prefix(REQUEST)
        while h.keeper._prefixes:
            await asyncio.sleep(0)
        await h.keeper.close()
        assert h.now > prompt_cache.PREWARM_IDLE_DROP_S
        assert h.sent[-1][0] <= prompt_cache.PREWARM_IDLE_DROP_S

    async def test_a_prefix_whose_re_sends_keep_writing_is_dropped(self):
        """A write inside the hour means the re-send is not the prompt real requests
        send; carrying on would pay for a full write every time and warm nothing."""
        wrote = _message(read=0, written=90_000)
        h = _Harness(replies=[wrote, wrote])
        h.keeper.note_prefix(REQUEST)
        while h.keeper._prefixes:
            await asyncio.sleep(0)
        await h.keeper.close()
        assert len(h.sent) == 2

    async def test_keeps_only_the_most_recently_used_prefixes(self):
        h = _Harness()
        for i in range(prompt_cache.MAX_PREFIXES + 1):
            h.now = float(i)
            h.keeper.note_prefix({**REQUEST, "model": f"claude-fable-5-{i}"})
        models = {p.params["model"] for p in h.keeper._prefixes.values()}
        assert len(models) == prompt_cache.MAX_PREFIXES
        assert "claude-fable-5-0" not in models
        await h.keeper.close()

    async def test_off_or_without_a_cached_system_block_registers_nothing(self):
        off = _Harness(prewarm=False)
        off.keeper.note_prefix(REQUEST)
        assert off.keeper._prefixes == {}
        on = _Harness()
        on.keeper.note_prefix({k: v for k, v in REQUEST.items() if k != "system"})
        assert on.keeper._prefixes == {}
