"""Per-API-key request rate limiting.

Fixed-window counters (requests/minute + requests/day) backed by Redis when
available, else an in-process dict (exact on a single node, approximate across
replicas). ``check_rate_limit`` returns ``None`` when the request is allowed or
``(retry_after_seconds, limit, reset_epoch)`` when it must be rejected.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from app.redis_client import get_redis

_WINDOW_MINUTE = 60
_WINDOW_DAY = 86400

# key_id -> {window_key: (window_start, count)}
_local: Dict[str, Dict[str, Tuple[int, int]]] = {}


def _windows(now: int) -> Dict[str, Tuple[int, int]]:
    """{key-suffix: (window_start_epoch, ttl)} for the active minute+day windows."""
    minute = (now // _WINDOW_MINUTE) * _WINDOW_MINUTE
    day = (now // _WINDOW_DAY) * _WINDOW_DAY
    return {
        "m": (minute, _WINDOW_MINUTE + 5),
        "d": (day, _WINDOW_DAY + 60),
    }


async def check_rate_limit(
    key_id: str, rpm: int = 0, rpd: int = 0
) -> Optional[Tuple[int, int, int]]:
    """Increment the key's counters; return retry info when a limit is hit.

    ``rpm``/``rpd`` of 0 mean unlimited. A rejected request still counts —
    standard fixed-window semantics, avoids check-then-incr races.
    """
    if rpm <= 0 and rpd <= 0:
        return None
    now = int(time.time())
    windows = _windows(now)
    r = get_redis()
    if r is not None:
        try:
            counts = {}
            for suffix, (_start, ttl) in windows.items():
                key = f"rate:{key_id}:{suffix}:{_start}"
                counts[suffix] = await r.incr(key)
                if counts[suffix] == 1:
                    await r.expire(key, ttl)
        except Exception:
            counts = None
        if counts is not None:
            if rpm > 0 and counts.get("m", 0) > rpm:
                start, _ = windows["m"]
                return (start + _WINDOW_MINUTE - now, rpm, start + _WINDOW_MINUTE)
            if rpd > 0 and counts.get("d", 0) > rpd:
                start, _ = windows["d"]
                return (start + _WINDOW_DAY - now, rpd, start + _WINDOW_DAY)
            return None

    # In-process fallback
    buckets = _local.setdefault(key_id, {})
    counts = {}
    for suffix, (start, _ttl) in windows.items():
        wkey = f"{suffix}:{start}"
        prev = buckets.get(wkey)
        count = (prev[1] if prev and prev[0] == start else 0) + 1
        buckets[wkey] = (start, count)
        counts[suffix] = count
        # Prune stale windows opportunistically.
        for stale in [k for k in buckets if k != wkey and k.split(":")[0] == suffix]:
            if buckets[stale][0] != start:
                buckets.pop(stale, None)
    if rpm > 0 and counts.get("m", 0) > rpm:
        start, _ = windows["m"]
        return (start + _WINDOW_MINUTE - now, rpm, start + _WINDOW_MINUTE)
    if rpd > 0 and counts.get("d", 0) > rpd:
        start, _ = windows["d"]
        return (start + _WINDOW_DAY - now, rpd, start + _WINDOW_DAY)
    return None


def reset_rate_limit() -> None:
    """Drop all in-process counters (tests)."""
    _local.clear()
