"""Non-blocking status query for the current agent's subagents."""

from __future__ import annotations

from typing import Any, Dict, Optional


class SubagentStatusTool:
    name = "subagent_status"

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Check live status and activity of your subagents without waiting. Use full_answer=true to fetch a complete answer when a report was truncated.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subagent_id": {
                            "type": "string",
                            "description": "Optional specific subagent id; omit to list all yours.",
                        },
                        "full_answer": {
                            "type": "boolean",
                            "description": "Return the subagent's complete answer instead of the compact preview. Use when a wait_subagent report says '[answer truncated]'.",
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(
        self,
        subagent_id: Optional[str] = None,
        full_answer: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = (context or {}).get("subagent_state")
        if state is None:
            raise RuntimeError("Subagents are not enabled for this run")
        return state.manager.status(subagent_id, state, full_answer=bool(full_answer))
