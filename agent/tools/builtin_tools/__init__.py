"""Built-in agent tools — each tool lives in its own module."""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Any, Dict, List, Type

logger = logging.getLogger(__name__)


def discover_builtin_tool_classes() -> Dict[str, Type[Any]]:
    """Discover all tool classes in this package.

    A valid tool class has:
    - a ``name`` attribute (str)
    - a ``tool_schema`` property
    - an ``execute`` coroutine method
    """
    from . import common  # noqa: F401 — ensure common is loaded first

    classes: Dict[str, Type[Any]] = {}
    for importer, modname, ispkg in pkgutil.iter_modules(__path__):
        if ispkg or modname.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{__package__}.{modname}")
        except Exception as exc:
            logger.warning("Failed to import builtin tool module %s: %s", modname, exc)
            continue
        for _name, obj in inspect.getmembers(mod, inspect.isclass):
            if getattr(obj, "name", None) and hasattr(obj, "tool_schema") and hasattr(obj, "execute"):
                classes[obj.name] = obj
    return dict(sorted(classes.items()))
