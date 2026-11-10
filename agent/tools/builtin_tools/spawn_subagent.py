"""Tool that starts an isolated background clone of the current assistant."""

from __future__ import annotations

from typing import Any, Dict, Optional


class SpawnSubagentTool:
    name = "spawn_subagent"

    def __init__(self, rag_app: Any) -> None:
        self._rag_app = rag_app

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Start an isolated background clone of yourself for a self-contained subtask. "
                    "It has the same instructions and inherits your full tool set, but no access to this conversation. "
                    "Returns an id immediately; use wait_subagent to collect its answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Self-contained task, including all context and expected output.",
                        },
                        "label": {
                            "type": "string",
                            "description": "Short human-readable title shown in chat.",
                        },
                    },
                    "required": ["task"],
                },
            },
        }

    async def execute(
        self,
        task: str,
        label: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = (context or {}).get("subagent_state")
        if state is None:
            raise RuntimeError("Subagents are not enabled for this run")
        record = await state.manager.spawn(task=task, label=label, owner=state)
        return {
            "content": f"Subagent {record.id} ({record.label}) started in the background. Use wait_subagent to collect its result.",
            "metadata": {"subagent": record.to_dict()},
        }
