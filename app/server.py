"""Thin launcher that runs the FastAPI app using the configured host/port flags."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import uvicorn

ROOT_DIR = Path(__file__).resolve().parents[1]
root_str = str(ROOT_DIR)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from agent.config import AppConfig

logger = logging.getLogger(__name__)


def _ensure_supported_python() -> None:
    """Fail fast on Python versions known to break LangChain/Pydantic internals."""
    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "Python 3.14+ is not supported by the current LangChain/Pydantic stack. "
            "Please use Python 3.11.x (recommended) or 3.12/3.13 if validated in your environment."
        )


def _reload_dirs(config: AppConfig) -> list[Path]:
    """Directories that trigger an auto-reload in development."""
    reload_dirs = {ROOT_DIR}
    reload_dirs.add(config.instructions_file.parent)
    reload_dirs.add(config.knowledge_base_root)
    return list(reload_dirs)


def _run_granian(config: AppConfig) -> None:
    # Granian is a Rust HTTP server; workers must stay at 1 because the app
    # keeps in-memory state (stores, locks, run registry) in a single process.
    # Interface must be ASGI explicitly — granian defaults to RSGI.
    from granian import Granian
    from granian.constants import Interfaces

    granian = Granian(
        "app.main:app",
        address=config.api_host,
        port=config.api_port,
        interface=Interfaces.ASGI,
        workers=1,
        reload=config.api_reload,
        reload_paths=_reload_dirs(config) if config.api_reload else [],
    )
    granian.serve()


def _run_uvicorn(config: AppConfig) -> None:
    reload_kwargs = {}
    if config.api_reload:
        reload_kwargs["reload_dirs"] = [str(d) for d in _reload_dirs(config)]

    uvicorn.run(
        "app.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.api_reload,
        **reload_kwargs,
    )


def main() -> None:
    """Load :class:`agent.config.AppConfig` and start the configured HTTP server."""
    _ensure_supported_python()
    config = AppConfig()
    # granian does not configure Python logging the way uvicorn does
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    config.bootstrap_default_runtime_data()

    server = (config.api_server or "").strip().lower()
    if server == "granian":
        _run_granian(config)
    else:
        if server and server != "uvicorn":
            logger.warning(
                "Unknown API_SERVER %r — falling back to uvicorn", config.api_server
            )
        _run_uvicorn(config)


if __name__ == "__main__":
    main()
