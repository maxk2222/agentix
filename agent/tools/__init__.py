"""Tool runtime for agent assistants."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from agent.tools.builtin_tools import discover_builtin_tool_classes
from agent.tools.venv_executor import VenvExecutor


def _build_runner_script(code: str, params: Dict[str, Any]) -> str:
    """Wrap user code so that parameters are injected and output is captured."""
    params_json = json.dumps(params)
    return (
        "import json, sys\n"
        f"params = json.loads({json.dumps(params_json)})\n"
        "\n"
        "# User tool code\n"
        f"{code}\n"
        "\n"
        "# If the code defines a 'run' function, call it with params\n"
        "if 'run' in dir():\n"
        '    result = run(**params)\n'
        '    print(json.dumps({"ok": True, "content": str(result)}))\n'
    )

logger = logging.getLogger(__name__)


class CustomToolWrapper:
    """Wraps a DB-stored custom tool for execution in a venv or Docker sandbox."""

    def __init__(self, tool_entry: Any) -> None:
        self.entry = tool_entry
        self.name = tool_entry.name
        self.enabled = tool_entry.enabled
        # Always keep a venv executor ready as fallback
        self._executor: Optional[VenvExecutor] = None
        self._executor = VenvExecutor(tool_entry.id, tool_entry.venv_config or {})

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return self.entry.tool_schema or {}

    async def execute(self, **params: Any) -> str:
        context = params.pop("context", None)
        code = self.entry.code
        if not code:
            raise RuntimeError("Tool has no code")

        # Try Docker if configured and we have a conversation context
        if getattr(self.entry, "runtime_mode", "venv") == "docker":
            conversation_id = (context or {}).get("conversation_id")
            if conversation_id:
                try:
                    from agent.tools.docker_sandbox import ConversationSandboxManager
                    manager = ConversationSandboxManager()
                    handle = await manager.get_or_create(conversation_id)
                    runner = _build_runner_script(code, params)
                    result = await manager.execute_in(handle, runner, timeout=30)
                    if not result.get("ok"):
                        raise RuntimeError(result.get("error", "Docker execution failed"))
                    stdout = result.get("stdout", "")
                    # Try to parse JSON output from the last non-empty line
                    output_lines = [ln for ln in stdout.splitlines() if ln.strip()]
                    if output_lines:
                        last_line = output_lines[-1]
                        try:
                            parsed = json.loads(last_line)
                            if isinstance(parsed, dict):
                                if not parsed.get("ok", True):
                                    raise RuntimeError(parsed.get("error", "Tool error"))
                                return str(parsed.get("content", stdout))
                        except json.JSONDecodeError:
                            pass
                    return stdout
                except RuntimeError as exc:
                    if "Docker sandbox unavailable" in str(exc) or "Docker client" in str(exc):
                        logger.warning("Docker unavailable for tool %s, falling back to venv", self.name)
                    else:
                        raise
            else:
                logger.warning("No conversation_id for Docker tool %s, falling back to venv", self.name)

        # Venv execution (primary or fallback)
        if self._executor:
            result = await self._executor.execute(code, params)
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "Unknown tool error"))
            return str(result.get("content", ""))

        # Fallback: direct eval for simple inline code without venv
        local_ns: Dict[str, Any] = {"params": params}
        exec(code, {"__builtins__": {}}, local_ns)  # type: ignore[arg-type]
        if "run" in local_ns:
            return str(local_ns["run"](**params))
        return ""


class ToolRegistry:
    """Registry of available tools for agent assistants."""

    def __init__(self) -> None:
        discovered = discover_builtin_tool_classes()
        self._builtins: Dict[str, Any] = {}
        for name, cls in discovered.items():
            try:
                self._builtins[name] = cls()
            except TypeError:
                # Some tools like CollectionManagerTool require constructor args
                logger.debug("Skipping builtin tool %s: requires constructor arguments", name)
        self._custom: Dict[str, Any] = {}
        self._tool_configs: Dict[str, Dict[str, Any]] = {}
        self._assistant: Dict[str, Any] = {}
        self._mcp_manager: Any = None

    def set_mcp_manager(self, manager: Any) -> None:
        self._mcp_manager = manager

    async def load_custom_tools(self, tools_store: Any) -> None:
        """Load enabled custom tools from the DB store and cache tool configs."""
        tools = await tools_store.list_tools(enabled_only=True)
        self._custom.clear()
        self._tool_configs.clear()
        for entry in tools:
            self._tool_configs[entry.name] = entry.config or {}
            if entry.kind == "custom":
                self.register_custom(entry.name, CustomToolWrapper(entry))
            elif entry.kind == "mcp" and self._mcp_manager is not None:
                # MCP-discovered remote tools: rebuild the proxy from the
                # persisted entry (no live session required) so enabled/schema
                # changes apply on every reload.
                proxy = self._mcp_manager.get_proxy(entry)
                if proxy is not None:
                    self.register_custom(entry.name, proxy)

    async def refresh_schemas(self) -> None:
        """Call refresh_schema on all tools that support it."""
        for tool in list(self._builtins.values()) + list(self._custom.values()):
            if hasattr(tool, "refresh_schema"):
                try:
                    await tool.refresh_schema()
                except Exception as exc:
                    logger.warning("Failed to refresh schema for tool %s: %s", getattr(tool, "name", "?"), exc)

    def register_builtin(self, name: str, tool: Any) -> None:
        self._builtins[name] = tool

    def register_custom(self, name: str, tool: Any) -> None:
        self._custom[name] = tool

    def register_assistant(self, name: str, tool: Any) -> None:
        self._assistant[name] = tool

    def clear_assistants(self) -> None:
        self._assistant.clear()

    def get_tool(self, name: str) -> Optional[Any]:
        if name in self._builtins:
            return self._builtins[name]
        if name in self._custom:
            return self._custom[name]
        return self._assistant.get(name)

    def list_tools(self, enabled_only: bool = True) -> List[str]:
        # Built-ins and assistant tools are always considered enabled
        names = list(self._builtins.keys()) + list(self._assistant.keys())
        for name, tool in self._custom.items():
            if not enabled_only or getattr(tool, "enabled", True):
                names.append(name)
        return names

    def tool_schemas(
        self,
        tool_names: List[str],
        *,
        tool_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        schemas = []
        for name in tool_names:
            tool = self.get_tool(name)
            if tool is None:
                continue
            override = (tool_overrides or {}).get(name)
            if override and hasattr(tool, "schema_with_config"):
                schema = tool.schema_with_config(override)
            else:
                schema = getattr(tool, "tool_schema", None)
            if schema is not None:
                schemas.append(schema)
        return schemas

    async def execute(
        self,
        name: str,
        params: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
        assistant_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tool = self.get_tool(name)
        if tool is None:
            return {"ok": False, "error": f"Tool '{name}' not found", "content": ""}
        try:
            if hasattr(tool, "execute"):
                # Forward context only to tools that accept it
                import inspect
                sig = inspect.signature(tool.execute)
                tool_config = self._tool_configs.get(name, {})
                if assistant_override:
                    tool_config = {**tool_config, **assistant_override}
                enriched_context = {**(context or {}), "tool_config": tool_config}
                if "context" in sig.parameters:
                    result = await tool.execute(**params, context=enriched_context)
                else:
                    result = await tool.execute(**params)
                if isinstance(result, dict) and "content" in result:
                    # `images` carries rendered page-image data URIs (e.g.
                    # workspace `view`). Only well-formed image data URIs pass,
                    # capped so a buggy tool cannot blow up the request.
                    images = [
                        u for u in (result.get("images") or [])
                        if isinstance(u, str) and u.startswith("data:image/")
                    ][:8]
                    normalized = {
                        "ok": bool(result.get("ok", True)),
                        "content": str(result.get("content", "")),
                        "error": result.get("error"),
                        "metadata": result.get("metadata"),
                    }
                    if images:
                        normalized["images"] = images
                    return normalized
                return {"ok": True, "content": str(result), "error": None}
            return {"ok": False, "error": "Tool has no execute method", "content": ""}
        except Exception as exc:
            logger.warning("Tool '%s' execution failed: %s", name, exc, exc_info=True)
            return {"ok": False, "error": str(exc), "content": ""}
