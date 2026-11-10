"""Leader election for single-winner background loops.

``acquire_leader`` claims ``sched:{name}:leader`` with ``SET NX EX`` so only
one replica ticks the triggers/backup schedulers at a time. Without Redis
every node is its own leader — preserving the single-node behaviour.
"""

from __future__ import annotations

import logging
import socket

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_NODE_ID = f"{socket.gethostname()}-{id(object())}"


async def acquire_leader(name: str, ttl: int = 300) -> bool:
    """Return True when this process may run the ``name`` scheduler tick.

    The lock holder renews by re-acquiring on each tick (EX refresh); a dead
    leader loses it within ``ttl`` seconds."""
    r = get_redis()
    if r is None:
        return True
    key = f"sched:{name}:leader"
    try:
        if await r.set(key, _NODE_ID, nx=True, ex=ttl):
            return True
        # Already hold it → refresh the expiry.
        if (await r.get(key)) == _NODE_ID:
            await r.expire(key, ttl)
            return True
        return False
    except Exception:
        # Fail closed: a transient Redis error must not let several replicas
        # tick the same scheduler — a delayed tick beats a duplicate run.
        logger.warning("leader lock %s failed — skipping tick", name, exc_info=True)
        return False
