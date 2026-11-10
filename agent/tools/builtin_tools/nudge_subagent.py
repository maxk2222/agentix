"""Tool that injects a mid-run guidance message into a running subagent."""

from __future__ import annotations

from typing import Any, Dict, Optional


class NudgeSubagentTool:
    name = "nudge_subagent"

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Send a mid-run guidance message into one of your running subagents. The message is "
                    "injected into its context before its next iteration (it also wakes a subagent that is "
                    "currently waiting). The subagent cannot reply to the nudge — collect its answer via "
                    "wait_subagent. Only works on subagents you spawned."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subagent_id": {
                            "type": "string",
                            "description": "Id of your running subagent (from spawn_subagent / subagent_status).",
                        },
                        "message": {
                            "type": "string",
                            "description": "Short guidance to inject, e.g. reprioritized task or extra context.",
                        },
                    },
                    "required": ["subagent_id", "message"],
                },
            },
        }

    async def execute(
        self,
        subagent_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = (context or {}).get("subagent_state")
        if state is None:
            raise RuntimeError("Subagents are not enabled for this run")
        return await state.manager.nudge(subagent_id, message, state)
