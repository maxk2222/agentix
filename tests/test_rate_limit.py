"""Per-API-key rate limiting: Redis fixed-windows + in-process fallback."""

from __future__ import annotations

import pytest

from app.rate_limit import check_rate_limit, reset_rate_limit
from app.redis_client import reset_redis


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    reset_redis()
    reset_rate_limit()
    monkeypatch.delenv("REDIS_URL", raising=False)
    yield
    reset_redis()
    reset_rate_limit()


@pytest.fixture()
def fake(monkeypatch):
    import fakeredis.aioredis

    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr("app.redis_client._client", r)
    return r


@pytest.mark.asyncio
async def test_unlimited_when_zero(fake):
    for _ in range(10):
        assert await check_rate_limit("k1", 0, 0) is None


@pytest.mark.asyncio
async def test_rpm_limit_hits(fake):
    assert await check_rate_limit("k1", 2, 0) is None
    assert await check_rate_limit("k1", 2, 0) is None
    result = await check_rate_limit("k1", 2, 0)
    assert result is not None
    retry, limit, _reset = result
    assert 0 < retry <= 60 and limit == 2


@pytest.mark.asyncio
async def test_rpd_independent_of_rpm(fake):
    # 100 rpm never trips the minute window; 2 rpd does.
    assert await check_rate_limit("k1", 100, 2) is None
    assert await check_rate_limit("k1", 100, 2) is None
    result = await check_rate_limit("k1", 100, 2)
    assert result is not None
    retry, limit, _ = result
    assert retry > 60 and limit == 2


@pytest.mark.asyncio
async def test_local_fallback_limits():
    # No redis → in-process counters enforce the same semantics.
    assert await check_rate_limit("k1", 1, 0) is None
    assert await check_rate_limit("k1", 1, 0) is not None


@pytest.mark.asyncio
async def test_keys_are_independent(fake):
    assert await check_rate_limit("a", 1, 0) is None
    assert await check_rate_limit("a", 1, 0) is not None  # a is now limited
    assert await check_rate_limit("b", 1, 0) is None      # b unaffected
