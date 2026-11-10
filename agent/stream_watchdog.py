"""Activity-based watchdog for streaming agent runs.

Replaces fixed total-run timeouts: a run may work for as long as it keeps
producing stream items (tokens, reasoning, tool/subagent events). Only when no
item arrives for ``idle_seconds`` does the watchdog abort the run.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

DEFAULT_STREAM_IDLE_SECONDS = 300.0
# Run-level watchdog: aborts a run when no *real* stream item (token, reasoning,
# tool/subagent event) arrives for this long — synthetic tool keepalives do not
# count. Sized above the sandbox budgets so a healthy code_execution
# (install ~240s + exec ~600s) fits inside one silent window.
DEFAULT_RUN_IDLE_SECONDS = 900.0


class StreamIdleTimeout(Exception):
    """Raised when a stream produced no item for ``idle_seconds``."""

    def __init__(self, idle_seconds: float) -> None:
        super().__init__(f"No stream activity for {idle_seconds:g}s")
        self.idle_seconds = idle_seconds


async def idle_bounded_agen(
    agen: AsyncIterator[Any],
    idle_seconds: float,
) -> AsyncIterator[Any]:
    """Yield items from ``agen``; abort with :class:`StreamIdleTimeout` when no
    item has arrived for ``idle_seconds``.

    Every yielded item counts as activity — tokens, reasoning, tool progress
    and subagent events alike — so a working run is never cut off; only a
    genuinely stalled stream is.
    """
    deadline = time.monotonic() + idle_seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StreamIdleTimeout(idle_seconds)
            try:
                item = await asyncio.wait_for(agen.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                aclose = getattr(agen, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass
                raise StreamIdleTimeout(idle_seconds) from None
            deadline = time.monotonic() + idle_seconds
            yield item
    finally:
        # Forward the close to the wrapped generator — abandoning it would
        # defer its cleanup (subagent shutdown, observer aclose) to asyncgen
        # GC finalization instead of running it in-band.
        aclose = getattr(agen, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass
