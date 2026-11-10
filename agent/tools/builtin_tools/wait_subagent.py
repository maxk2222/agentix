"""Wait for subagents spawned by the current agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class WaitSubagentTool:
    name = "wait_subagent"

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Wait for your background subagents and collect their answers and activity reports. "
                    "mode='first' returns as soon as the next subagent finishes (finished ones are reported, "
                    "the rest keeps running); mode='all' waits for everything (default). "
                    "Also returns early when new input is injected into this run."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subagent_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional ids to wait for; omit to consider all your subagents.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["all", "first"],
                            "description": "'all' waits for every selected subagent (default); 'first' returns once the next one finishes.",
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(
        self,
        subagent_ids: Optional[List[str]] = None,
        mode: str = "all",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = (context or {}).get("subagent_state")
        if state is None:
            raise RuntimeError("Subagents are not enabled for this run")
        return await state.manager.wait(
            subagent_ids, state, mode=mode,
            input_event=(context or {}).get("input_event"),
        )
