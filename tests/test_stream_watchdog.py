"""Tests for the stream idle watchdog: runs may take unlimited time as long as
they keep producing items; only a stalled stream is aborted."""

from __future__ import annotations

import asyncio

import pytest

from agent.stream_watchdog import StreamIdleTimeout, idle_bounded_agen


async def _agen_items(*items, delay: float = 0.0):
    for item in items:
        if delay:
            await asyncio.sleep(delay)
        yield item


@pytest.mark.asyncio
async def test_yields_all_items_when_active():
    items = [i async for i in idle_bounded_agen(_agen_items(1, 2, 3), 5.0)]
    assert items == [1, 2, 3]


@pytest.mark.asyncio
async def test_activity_below_idle_keeps_running():
    # Total runtime > idle_seconds, but each gap stays below it: must not abort.
    items = [i async for i in idle_bounded_agen(_agen_items(1, 2, 3, delay=0.08), 0.15)]
    assert items == [1, 2, 3]


@pytest.mark.asyncio
async def test_stalled_stream_aborts():
    async def stalled():
        yield 1
        await asyncio.sleep(5)
        yield 2  # pragma: no cover - unreachable

    seen = []
    with pytest.raises(StreamIdleTimeout) as excinfo:
        async for item in idle_bounded_agen(stalled(), 0.05):
            seen.append(item)
    assert seen == [1]
    assert excinfo.value.idle_seconds == 0.05


@pytest.mark.asyncio
async def test_idle_before_first_item():
    async def never():
        await asyncio.sleep(5)
        yield 1  # pragma: no cover - unreachable

    with pytest.raises(StreamIdleTimeout):
        async for _ in idle_bounded_agen(never(), 0.05):
            pass


@pytest.mark.asyncio
async def test_empty_stream_ends_cleanly():
    async def empty():
        return
        yield  # pragma: no cover

    items = [i async for i in idle_bounded_agen(empty(), 5.0)]
    assert items == []


@pytest.mark.asyncio
async def test_non_token_items_count_as_activity():
    # Any yielded dict — tool progress, subagent events — resets the idle clock.
    async def events():
        for i in range(4):
            await asyncio.sleep(0.04)
            yield {"tool_progress": i}
        yield {"content": "done"}

    items = [i async for i in idle_bounded_agen(events(), 0.1)]
    assert items[-1] == {"content": "done"}
    assert len(items) == 5
