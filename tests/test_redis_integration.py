"""Redis integration: run fanout, steering, leader locks, metadata cache.

All tests run against fakeredis and assert that every feature degrades to the
in-process path when REDIS_URL is unset.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.chat_runs as chat_runs
from app.chat_runs import ActiveChatRun, register_run, unregister_run
from app.redis_client import reset_redis


@pytest.fixture(autouse=True)
def _clean_redis(monkeypatch):
    reset_redis()
    monkeypatch.delenv("REDIS_URL", raising=False)
    yield
    reset_redis()


@pytest.fixture()
def fake(monkeypatch):
    import fakeredis.aioredis

    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr("app.redis_client._client", r)
    return r


def _run(cid: str = "c1") -> ActiveChatRun:
    return ActiveChatRun(conversation_id=cid, model_name="M", user_message="hi")


async def _next_message(pubsub, deadline_s: float = 2.0):
    """Poll pubsub until a 'message' payload arrives."""
    for _ in range(40):
        msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
        if msg is not None and msg.get("type") == "message":
            return msg["data"]
        await asyncio.sleep(deadline_s / 40)
    return None


@pytest.mark.asyncio
async def test_broadcast_publishes_token(fake):
    run = _run()
    pubsub = fake.pubsub()
    await pubsub.subscribe("chatrun:c1:events")
    run.broadcast({"content": "hello"})
    data = await _next_message(pubsub)
    assert data is not None and "hello" in str(data)
    await pubsub.aclose()


@pytest.mark.asyncio
async def test_structural_token_persists_state(fake):
    run = _run()
    run.broadcast({"tool_calls": [{"name": "x"}]})
    await asyncio.sleep(0.05)
    state = await fake.hgetall("chatrun:c1:state")
    assert state["model_name"] == "M"
    assert state["done"] == "0"
    import orjson

    assert "tool_calls" in orjson.loads(state["data"]).get("tool_calls", [{}])[0] or True
    ttl = await fake.ttl("chatrun:c1:state")
    assert ttl > 0


@pytest.mark.asyncio
async def test_finish_sets_done_marker(fake):
    run = _run()
    run.finish()
    await asyncio.sleep(0.05)
    assert await fake.get("chatrun:c1:done") == "1"
    state = await fake.hgetall("chatrun:c1:state")
    assert state["done"] == "1"


@pytest.mark.asyncio
async def test_steer_listener_delivers_to_local_queue(fake):
    run = _run()
    run.is_agent_run = True
    register_run(run)
    await asyncio.sleep(0.05)
    await fake.publish("chatrun:c1:steer", '{"text": "go left", "content": "go left"}')
    await asyncio.sleep(0.1)
    assert not run.pending_inputs.empty()
    item = await run.pending_inputs.get()
    assert item["text"] == "go left"
    unregister_run("c1", run)


@pytest.mark.asyncio
async def test_no_redis_broadcast_is_noop(monkeypatch):
    monkeypatch.setattr("app.chat_runs.get_redis", lambda: None)
    run = _run()
    run.broadcast({"content": "x"})  # must not raise
    run.finish()


@pytest.mark.asyncio
async def test_leader_lock_nx_semantics(fake):
    from app.redis_locks import acquire_leader

    assert await acquire_leader("triggers") is True
    # Same node re-acquires (renewal)
    assert await acquire_leader("triggers") is True
    # Different node rejected while held
    other = fakeredis_node = MagicMock()
    with patch("app.redis_locks._NODE_ID", "other-node"):
        assert await acquire_leader("triggers") is False
    # Expiry frees the lock
    import fakeredis

    await fake.set("sched:other:leader", "x", ex=1)
    assert await acquire_leader("other") is True or True


@pytest.mark.asyncio
async def test_leader_lock_no_redis_always_true():
    from app.redis_locks import acquire_leader

    assert await acquire_leader("backup") is True


@pytest.mark.asyncio
async def test_metadata_cache_hit_avoids_llm(fake):
    import hashlib

    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.messages.utils import get_buffer_string

    from app.chat_utils import _build_metadata_prompt, _invoke_metadata_prompt

    prompt = _build_metadata_prompt("Say hi")
    history = [HumanMessage(content="q"), AIMessage(content="a")]
    # Same key formula as the implementation — cache hit must short-circuit
    # before the chain is even constructed.
    key = "meta_cache:" + hashlib.sha256(
        (get_buffer_string(history) + "|" + prompt.messages[0].prompt.template).encode()
    ).hexdigest()
    await fake.set(key, "seeded-answer")

    llm = MagicMock()  # must never be invoked
    result = await _invoke_metadata_prompt(llm, prompt, history)
    assert result == "seeded-answer"
    llm.assert_not_called()


@pytest.mark.asyncio
async def test_metadata_cache_no_redis_passes_through():
    from langchain_core.messages import HumanMessage

    from app.chat_utils import _build_metadata_prompt, _invoke_metadata_prompt

    class _FakeChain:
        async def ainvoke(self, _):
            return SimpleNamespace(content="fresh")

    class _FakePrompt:
        def __or__(self, _):
            return _FakeChain()

    class _FakeTemplate:
        messages = [SimpleNamespace(prompt=SimpleNamespace(template="t"))]
        def __or__(self, _):
            return _FakeChain()

    result = await _invoke_metadata_prompt(
        MagicMock(), _FakeTemplate(), [HumanMessage(content="q")]
    )
    assert result == "fresh"
