"""Built-in agent tools — facade that re-exports everything from builtin_tools.

This module remains for backward compatibility. All tool implementations have
moved to individual modules under ``agent.tools.builtin_tools``.

Attributes are discovered automatically at import time; nothing is hard-coded.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _scan() -> tuple[Dict[str, Any], Dict[str, str]]:
    """Scan ``builtin_tools/`` and return:

    - ``common``: mapping ``name -> object`` for every public name exported
      from ``common.py`` (plus auto-generated ``_foo`` aliases).
    - ``tools``: mapping ``ClassName -> module_path`` for every tool class.
    """
    import agent.tools.builtin_tools as _pkg
    from agent.tools.builtin_tools import common as _common

    # 1. Re-export everything from common.py + auto aliases
    common: Dict[str, Any] = {}
    for name in dir(_common):
        if name.startswith("__"):
            continue
        obj = getattr(_common, name)
        common[name] = obj
        # Auto-alias: foo -> _foo for backward compatibility
        if not name.startswith("_"):
            alias = f"_{name}"
            if alias not in common:
                common[alias] = obj

    # 2. Discover tool classes
    tools: Dict[str, str] = {}
    for _, modname, ispkg in pkgutil.iter_modules(_pkg.__path__):
        if ispkg or modname.startswith("_") or modname == "common":
            continue
        try:
            mod = importlib.import_module(f"agent.tools.builtin_tools.{modname}")
        except Exception as exc:
            logger.warning("Failed to import builtin tool module %s: %s", modname, exc)
            continue
        for _name, obj in inspect.getmembers(mod, inspect.isclass):
            if getattr(obj, "name", None) and hasattr(obj, "tool_schema") and hasattr(obj, "execute"):
                tools[_name] = f"agent.tools.builtin_tools.{modname}"

    return common, tools


_COMMON, _TOOL_MAP = _scan()

# Dynamically build __all__ from scanned names
__all__ = sorted(
    set(_COMMON.keys()) | set(_TOOL_MAP.keys()) | {"discover_builtin_tool_classes"}
)


def __getattr__(name: str) -> Any:
    if name in _COMMON:
        return _COMMON[name]
    mod = _TOOL_MAP.get(name)
    if mod:
        module = importlib.import_module(mod)
        return getattr(module, name)
    if name == "discover_builtin_tool_classes":
        from agent.tools.builtin_tools import discover_builtin_tool_classes
        return discover_builtin_tool_classes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
