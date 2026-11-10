"""Built-in tool that delegates a task to another assistant."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Prefix used for all assistant-derived tool names
ASSISTANT_TOOL_PREFIX = "assistant__"


class AssistantCallerTool:
    """Tool wrapper that invokes another assistant via ``RAGApplication.run()``.

    Each instance represents a single target assistant.  The tool schema is
    generated dynamically from the assistant name and description.
    """

    def __init__(
        self,
        rag_app: Any,
        assistant_name: str,
        description: str = "",
    ) -> None:
        self._rag_app = rag_app
        self.assistant_name = assistant_name
        # Safe tool name: prefix + lowercase with underscores
        safe_name = "".join(c if c.isalnum() else "_" for c in assistant_name).lower()
        self.name = f"{ASSISTANT_TOOL_PREFIX}{safe_name}"
        self._description = (
            description or f"Invoke the '{assistant_name}' assistant with a task."
        )

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self._description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task or question to send to the assistant.",
                        },
                    },
                    "required": ["task"],
                },
            },
        }

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        caller = (context or {}).get("assistant_name")

        # Prevent direct self-call
        if caller == self.assistant_name:
            return (
                f"[Error: Assistant '{self.assistant_name}' cannot call itself.]"
            )

        if not self._rag_app or not getattr(self._rag_app, "ready", False):
            return "[Error: RAG application is not ready.]"

        try:
            from app.chat_utils import extract_answer_text

            result = await self._rag_app.run(
                input_text=task,
                model_name=self.assistant_name,
                chat_history=[],
            )
            answer = extract_answer_text(result)
            return answer or "[No answer returned by assistant.]"
        except Exception as exc:
            logger.warning(
                "AssistantCallerTool failed for assistant '%s': %s",
                self.assistant_name,
                exc,
                exc_info=True,
            )
            return f"[Error calling assistant '{self.assistant_name}': {exc}]"
