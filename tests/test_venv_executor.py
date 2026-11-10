"""Tests for agent.tools.venv_executor."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from agent.tools.venv_executor import VenvExecutor, VENV_BASE_DIR


class TestVenvExecutor:
    def test_venv_python_path(self):
        exc = VenvExecutor("tool-abc", {"python_version": "3.11"})
        expected = VENV_BASE_DIR / "tool-abc" / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python")
        assert exc.venv_python == expected

    def test_exists_false_when_no_venv(self, tmp_path):
        exc = VenvExecutor("x", {})
        exc._venv_path = tmp_path / "nonexistent"
        assert exc.exists is False

    @pytest.mark.asyncio
    async def test_ensure_venv_skips_when_exists(self, tmp_path):
        venv_dir = tmp_path / "venv"
        bin_dir = venv_dir / ("Scripts" if sys.platform == "win32" else "bin")
        bin_dir.mkdir(parents=True)
        (bin_dir / ("python.exe" if sys.platform == "win32" else "python")).touch()

        exc = VenvExecutor("t1", {})
        exc._venv_path = venv_dir
        with patch("subprocess.run") as mock_run:
            await exc.ensure_venv()
            mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_simple_code(self, tmp_path):
        exc = VenvExecutor("t1", {})
        code = "print('hello from venv')"

        # Mock the subprocess execution
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "hello from venv\n"
        mock_proc.stderr = ""

        with patch("subprocess.run", return_value=mock_proc):
            result = await exc.execute(code, {"query": "test"})

        assert result["ok"] is True
        assert "hello from venv" in result["content"]

    @pytest.mark.asyncio
    async def test_execute_timeout(self, tmp_path):
        exc = VenvExecutor("t1", {})
        exc._venv_path = tmp_path / "venv"
        (exc._venv_path / ("Scripts" if sys.platform == "win32" else "bin")).mkdir(parents=True)
        (exc._venv_path / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python")).touch()
        code = "import time; time.sleep(100)"

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="python", timeout=1)):
            result = await exc.execute(code, {})

        assert result["ok"] is False
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_nonzero_return(self):
        exc = VenvExecutor("t1", {})
        code = "raise RuntimeError('boom')"

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "RuntimeError: boom\n"

        with patch("subprocess.run", return_value=mock_proc):
            result = await exc.execute(code, {})

        assert result["ok"] is False
        assert "boom" in result["error"]

    @pytest.mark.asyncio
    async def test_rebuild_deletes_and_recreates(self, tmp_path):
        exc = VenvExecutor("t1", {"requirements": "requests"})
        exc._venv_path = tmp_path / "venv"
        exc._venv_path.mkdir(parents=True)
        (exc._venv_path / "placeholder").touch()

        with patch("shutil.rmtree") as mock_rm:
            with patch.object(exc, "ensure_venv") as mock_ensure:
                await exc.rebuild()
                mock_rm.assert_called_once_with(exc._venv_path, ignore_errors=True)
                mock_ensure.assert_called_once()
