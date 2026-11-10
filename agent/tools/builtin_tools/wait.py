"""Small bounded delay tool available to every agent."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, Dict, Optional


class WaitTool:
    name = "wait"

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Wait briefly before polling an external resource or retrying. The delay is capped "
                    "by max_seconds. Returns early when a new message is injected into this run."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seconds": {"type": "number", "description": "How many seconds to wait (0.1 to 120)."},
                        "reason": {"type": "string", "description": "Optional short reason for waiting."},
                    },
                    "required": ["seconds"],
                },
            },
        }

    async def execute(
        self,
        seconds: float,
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        context = context or {}
        config = context.get("tool_config") or {}
        try:
            cap = min(120.0, max(0.1, float(config.get("max_seconds", 120) or 120)))
            delay = min(cap, max(0.1, float(seconds)))
        except (TypeError, ValueError) as exc:
            raise ValueError("seconds must be a number") from exc
        suffix = f" ({reason[:200]})" if reason else ""

        queue = context.get("injected_input_queue")
        event = context.get("input_event")

        # Input queued before the wait even started — return immediately so the
        # runner can drain it at the next iteration boundary.
        if queue is not None and not queue.empty():
            return f"Interrupted by pending input (planned {delay:g}s){suffix}."

        if event is not None:
            started = time.monotonic()
            sleep_task = asyncio.create_task(asyncio.sleep(delay))
            event_task = asyncio.create_task(event.wait())
            done, _ = await asyncio.wait(
                {sleep_task, event_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in (sleep_task, event_task):
                if task not in done:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            if event_task in done:
                elapsed = time.monotonic() - started
                return f"Interrupted by new input after {elapsed:g}s (planned {delay:g}s){suffix}."
            return f"Waited {delay:g}s{suffix}."

        if queue is not None:
            # Queue without an event: poll in short slices so newly queued
            # input still ends the wait promptly.
            deadline = time.monotonic() + delay
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(remaining, 0.5))
                if not queue.empty():
                    elapsed = delay - max(0.0, deadline - time.monotonic())
                    return f"Interrupted by new input after {elapsed:g}s (planned {delay:g}s){suffix}."
            return f"Waited {delay:g}s{suffix}."

        await asyncio.sleep(delay)
        return f"Waited {delay:g}s{suffix}."
