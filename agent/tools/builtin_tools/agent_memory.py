"""Agent memory builtin tool."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AgentMemoryTool:
    """Persistent scratchpad for agent runs within a single conversation."""

    name = "agent_memory"
    description = (
        "A persistent scratchpad for THIS conversation. Use it to store durable facts, "
        "user preferences, verification or coverage checklists, and important intermediate "
        "results that should persist across tool calls — earlier tool outputs may be "
        "compressed during long runs, memory is not. "
        "Do NOT store raw tool outputs, hallucinated information, or data that is better "
        "fetched again from a source tool. Actions: read (retrieve), set (overwrite), append (add), delete (clear)."
    )

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["read", "set", "append", "delete"],
                            "description": "The action to perform on the persistent memory.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to store. Required for 'set' and 'append'.",
                        },
                    },
                    "required": ["action"],
                },
            },
        }

    async def execute(self, action: str, content: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> str:
        ctx = context or {}
        store = ctx.get("agent_memory_store")
        conversation_id = ctx.get("conversation_id")
        assistant_name = ctx.get("assistant_name") or "default"

        if not store:
            raise RuntimeError("AgentMemoryStore not available")
        if not conversation_id:
            raise RuntimeError("conversation_id required for agent_memory")

        action = (action or "").lower().strip()
        if action not in ("read", "set", "append", "delete"):
            raise ValueError(f"Unknown action: {action}")

        if action in ("set", "append") and content is None:
            raise ValueError("content is required for set/append")

        if action == "read":
            memory = await store.get_memory(conversation_id, assistant_name)
            if memory:
                return f"Current memory:\n{memory}"
            return "Memory is empty."

        if action == "set":
            await store.set_memory(conversation_id, assistant_name, content or "")
            return "Memory updated."

        if action == "append":
            await store.append_memory(conversation_id, assistant_name, content or "")
            return "Memory appended."

        # action == "delete"
        await store.delete_memory(conversation_id, assistant_name)
        return "Memory cleared."
