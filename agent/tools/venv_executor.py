"""Venv-based execution for custom agent tools.

Each custom tool can optionally run in its own persistent virtual environment,
managed by this executor.  The venv is created on first use (or after a
"rebuild" request) and re-used across subsequent calls for performance.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

VENV_BASE_DIR = Path.home() / ".agentix" / "tool_envs"
MAX_OUTPUT_BYTES = 50_000
DEFAULT_TIMEOUT_SECONDS = 30


class VenvExecutor:
    """Manage per-tool virtual environments and execute custom tool code inside them."""

    def __init__(self, tool_id: str, venv_config: Optional[Dict[str, Any]] = None) -> None:
        self.tool_id = tool_id
        self.venv_config = venv_config or {}
        self._venv_path = VENV_BASE_DIR / tool_id

    @property
    def venv_python(self) -> Path:
        """Path to the Python interpreter inside the venv."""
        if sys.platform == "win32":
            return self._venv_path / "Scripts" / "python.exe"
        return self._venv_path / "bin" / "python"

    @property
    def exists(self) -> bool:
        return self.venv_python.exists()

    async def ensure_venv(self) -> None:
        """Create the venv if it does not yet exist, then install requirements."""
        if self.exists:
            return

        VENV_BASE_DIR.mkdir(parents=True, exist_ok=True)
        python_version = self.venv_config.get("python_version", "")
        python_bin = self._resolve_python(python_version)

        logger.info("Creating venv for tool %s using %s", self.tool_id, python_bin)
        try:
            subprocess.run(
                [python_bin, "-m", "venv", str(self._venv_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to create venv for tool {self.tool_id}: {exc.stderr}") from exc

        requirements = self.venv_config.get("requirements", "")
        if requirements and requirements.strip():
            await self._install_requirements(requirements)

    async def _install_requirements(self, requirements: str) -> None:
        """Install packages from a requirements string into the venv."""
        req_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        req_file.write(requirements)
        req_file.close()
        try:
            logger.info("Installing requirements for tool %s", self.tool_id)
            uv_bin = shutil.which("uv")
            if uv_bin:
                cmd = [uv_bin, "pip", "install", "--python", str(self.venv_python), "-r", req_file.name]
            else:
                cmd = [str(self.venv_python), "-m", "pip", "install", "-r", req_file.name]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.warning(
                    "Requirements install for tool %s had issues: %s",
                    self.tool_id,
                    result.stderr[:500],
                )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Requirements install timed out for tool {self.tool_id}") from exc
        finally:
            try:
                os.unlink(req_file.name)
            except OSError:
                pass

    async def execute(
        self,
        code: str,
        params: Dict[str, Any],
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> Dict[str, Any]:
        """Run custom tool code inside the venv.

        Returns a dict with ``ok``, ``content``, and ``error`` keys.
        """
        await self.ensure_venv()

        # Write a temporary runner script
        runner = self._build_runner_script(code, params)
        runner_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        runner_file.write(runner)
        runner_file.close()

        try:
            logger.debug("Executing tool %s in venv", self.tool_id)
            result = subprocess.run(
                [str(self.venv_python), runner_file.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if result.returncode != 0:
                return {
                    "ok": False,
                    "content": "",
                    "error": f"Tool exited with code {result.returncode}: {stderr[:1_000]}",
                }

            # Try to parse JSON output from the last non-empty line
            output_lines = [ln for ln in stdout.splitlines() if ln.strip()]
            if output_lines:
                last_line = output_lines[-1]
                try:
                    parsed = json.loads(last_line)
                    if isinstance(parsed, dict):
                        return {
                            "ok": parsed.get("ok", True),
                            "content": str(parsed.get("content", stdout)),
                            "error": parsed.get("error"),
                        }
                except json.JSONDecodeError:
                    pass

            # Fall back to raw stdout
            truncated = stdout[:MAX_OUTPUT_BYTES]
            if len(stdout) > MAX_OUTPUT_BYTES:
                truncated += "\n...[output truncated]"
            return {"ok": True, "content": truncated, "error": None}

        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "content": "",
                "error": f"Tool timed out after {timeout} seconds",
            }
        except Exception as exc:
            logger.warning("Venv execution failed for tool %s: %s", self.tool_id, exc, exc_info=True)
            return {"ok": False, "content": "", "error": str(exc)}
        finally:
            try:
                os.unlink(runner_file.name)
            except OSError:
                pass

    async def rebuild(self) -> None:
        """Delete and recreate the venv from scratch."""
        if self._venv_path.exists():
            shutil.rmtree(self._venv_path, ignore_errors=True)
        await self.ensure_venv()

    def _resolve_python(self, python_version: str) -> str:
        """Resolve the host Python interpreter to use for venv creation."""
        if python_version:
            # Try pyenv-style python versions first
            candidates = [
                f"python{python_version}",
                f"python{python_version.replace('.', '')}",
                python_version,
            ]
            for cand in candidates:
                path = shutil.which(cand)
                if path:
                    return path
        # Fallback to the current interpreter
        return sys.executable

    def _build_runner_script(self, code: str, params: Dict[str, Any]) -> str:
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
            "    result = run(**params)\n"
            "    print(json.dumps({\"ok\": True, \"content\": str(result)}))\n"
        )
