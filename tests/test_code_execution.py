"""Tests for the code_execution built-in tool and Docker sandbox manager."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.tools.builtin import CodeExecutionTool
from agent.tools.docker_sandbox import (
    ConversationSandboxManager,
    SandboxHandle,
    _container_name,
    _sanitize_conversation_id,
)
from agent.tools import ToolRegistry


class TestCodeExecutionTool:
    def test_tool_schema(self):
        tool = CodeExecutionTool()
        schema = tool.tool_schema
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "code_execution"
        props = func["parameters"]["properties"]
        assert "code" in props
        assert "packages" in props
        assert "timeout" in props
        assert "action" in props
        assert "cmd" in props
        assert "job_id" in props
        assert "tail" in props
        assert "stream" in props
        assert "text" in props
        assert func["parameters"]["required"] == []

    @pytest.mark.asyncio
    async def test_execute_without_conversation_id(self):
        tool = CodeExecutionTool()
        result = await tool.execute(code="print(1)")
        assert "conversation_id" in result
        assert "requires" in result.lower() or "Error" in result

    @pytest.mark.asyncio
    async def test_execute_with_mocked_manager(self):
        tool = CodeExecutionTool()
        mock_handle = MagicMock()
        mock_handle.container_name = "agentix-sandbox-test-123"
        mock_handle.host_port = 9999

        with patch.object(
            ConversationSandboxManager, "get_or_create", new_callable=AsyncMock, return_value=mock_handle
        ) as mock_get, patch.object(
            ConversationSandboxManager, "execute_in", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = {
                "ok": True,
                "stdout": "hello",
                "stderr": "",
                "error": None,
            }
            result = await tool.execute(
                code="print('hello')",
                context={"conversation_id": "conv-123"},
            )
            mock_get.assert_awaited_once_with("conv-123", sandbox_config={}, agent_workspace_dir=None)
            mock_exec.assert_awaited_once()
            assert "hello" in result

    @pytest.mark.asyncio
    async def test_execute_with_packages(self):
        tool = CodeExecutionTool()
        mock_handle = MagicMock()
        mock_handle.host_port = 9999

        with patch.object(
            ConversationSandboxManager, "get_or_create", new_callable=AsyncMock, return_value=mock_handle
        ), patch.object(
            ConversationSandboxManager, "install_in", new_callable=AsyncMock
        ) as mock_install, patch.object(
            ConversationSandboxManager, "execute_in", new_callable=AsyncMock
        ) as mock_exec:
            mock_install.return_value = {"ok": True, "stdout": "", "stderr": "", "error": None}
            mock_exec.return_value = {"ok": True, "stdout": "42", "stderr": "", "error": None}

            result = await tool.execute(
                code="print(42)",
                packages=["numpy"],
                context={"conversation_id": "conv-456"},
            )
            # Install budget = min(exec_timeout * 4, INSTALL_TIMEOUT + 30) with
            # the default exec timeout of 120 → capped at 270.
            mock_install.assert_awaited_once_with(mock_handle, ["numpy"], timeout=270)
            assert "42" in result

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_default_exec_timeout_is_120(self):
        tool = CodeExecutionTool()
        mock_handle = MagicMock()
        mock_handle.container_name = "agentix-sandbox-test-123"

        with patch.object(
            ConversationSandboxManager, "get_or_create", new_callable=AsyncMock, return_value=mock_handle
        ), patch.object(
            ConversationSandboxManager, "execute_in", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = {"ok": True, "stdout": "x", "stderr": "", "error": None}
            await tool.execute(code="print(1)", context={"conversation_id": "conv-1"})
            # LLM omitted 'timeout' → the 120s default reaches the sandbox.
            assert mock_exec.await_args.kwargs["timeout"] == 120

    @pytest.mark.asyncio
    async def test_execute_install_failure(self):
        tool = CodeExecutionTool()
        mock_handle = MagicMock()
        mock_handle.host_port = 9999

        with patch.object(
            ConversationSandboxManager, "get_or_create", new_callable=AsyncMock, return_value=mock_handle
        ), patch.object(
            ConversationSandboxManager, "install_in", new_callable=AsyncMock
        ) as mock_install:
            mock_install.return_value = {
                "ok": False,
                "stdout": "",
                "stderr": "no such package",
                "error": "pip failed",
            }

            result = await tool.execute(
                code="print(1)",
                packages=["fake-package-xyz"],
                context={"conversation_id": "conv-789"},
            )
            assert "Package installation failed" in result
            assert "no such package" in result

    @pytest.mark.asyncio
    async def test_execute_graceful_docker_unavailable(self):
        tool = CodeExecutionTool()
        with patch.object(
            ConversationSandboxManager, "get_or_create", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = RuntimeError("Docker sandbox unavailable: no docker")
            result = await tool.execute(
                code="print(1)",
                context={"conversation_id": "conv-000"},
            )
            assert "Docker sandbox unavailable" in result


class TestCodeExecutionJobs:
    """action-based dispatch onto the sandbox /jobs endpoint."""

    def _patches(self, jobs_result):
        handle = MagicMock()
        handle.mount_warning = ""
        get = patch.object(
            ConversationSandboxManager, "get_or_create",
            new_callable=AsyncMock, return_value=handle,
        )
        jobs = patch.object(
            ConversationSandboxManager, "jobs_in",
            new_callable=AsyncMock, return_value=jobs_result,
        )
        return handle, get, jobs

    CTX = {"conversation_id": "conv-jobs"}

    @pytest.mark.asyncio
    async def test_shell_start_payload(self):
        tool = CodeExecutionTool()
        handle, get, jobs = self._patches({"ok": True, "job_id": "j1", "pid": 42})
        with get, jobs as mock_jobs:
            res = await tool.execute(
                action="shell_start", cmd="python srv.py", cwd="/app",
                context=self.CTX,
            )
        assert "Started job 'j1'" in res and "pid 42" in res
        (h, payload), kw = mock_jobs.await_args
        assert h is handle
        assert payload == {"action": "start", "cmd": "python srv.py", "cwd": "/app"}
        assert kw["timeout"] == 60

    @pytest.mark.asyncio
    async def test_output_payload_and_format(self):
        tool = CodeExecutionTool()
        _, get, jobs = self._patches({
            "ok": True, "status": "running", "rc": None, "runtime_s": 12.5,
            "stdout": "line1\n", "stderr": "warn\n",
        })
        with get, jobs as mock_jobs:
            res = await tool.execute(
                action="output", job_id="j1", tail=50, stream="stderr",
                context=self.CTX,
            )
        (h, payload), _ = mock_jobs.await_args
        assert payload == {"action": "output", "job_id": "j1", "tail": 50, "stream": "stderr"}
        assert "job j1 — running" in res and "warn" in res

    @pytest.mark.asyncio
    async def test_write_and_kill_payloads(self):
        tool = CodeExecutionTool()
        _, get, jobs = self._patches({"ok": True, "bytes": 4})
        with get, jobs as mock_jobs:
            res = await tool.execute(
                action="write", job_id="j1", text="abc", context=self.CTX)
        assert mock_jobs.await_args[0][1] == {"action": "write", "job_id": "j1", "text": "abc"}
        assert "4 bytes" in res

        _, get, jobs = self._patches({"ok": True, "rc": -15})
        with get, jobs as mock_jobs:
            res = await tool.execute(
                action="kill", job_id="j1", context=self.CTX)
        assert mock_jobs.await_args[0][1] == {"action": "kill", "job_id": "j1"}
        assert "terminated" in res and "rc=-15" in res

    @pytest.mark.asyncio
    async def test_wait_blocks_and_reports(self):
        tool = CodeExecutionTool()
        _, get, jobs = self._patches(
            {"ok": True, "status": "exited", "rc": 0, "runtime_s": 3.0})
        with get, jobs as mock_jobs:
            res = await tool.execute(
                action="wait", job_id="j2", timeout=45, context=self.CTX)
        (h, payload), kw = mock_jobs.await_args
        assert payload == {"action": "wait", "job_id": "j2", "timeout": 45}
        assert kw["timeout"] == 75  # server wait cap + slack
        assert "exited (rc=0)" in res

    @pytest.mark.asyncio
    async def test_wait_woken_by_input_event(self):
        """User steering wakes a waiting job call — same semantics as wait."""
        tool = CodeExecutionTool()
        ev = asyncio.Event()
        ev.set()

        async def _never(*a, **k):
            await asyncio.sleep(60)

        _, get, jobs = self._patches({})
        with get, patch.object(
            ConversationSandboxManager, "jobs_in",
            new=MagicMock(side_effect=_never),
        ):
            res = await tool.execute(
                action="wait", job_id="j9",
                context={**self.CTX, "input_event": ev},
            )
        assert "Woken early" in res and "j9" in res

    @pytest.mark.asyncio
    async def test_shell_one_shot(self):
        tool = CodeExecutionTool()
        _, get, jobs = self._patches({
            "ok": True, "rc": 0, "stdout": "hi\n", "stderr": "",
        })
        with get, jobs as mock_jobs:
            res = await tool.execute(
                action="shell", cmd="echo hi", timeout=10, context=self.CTX)
        (h, payload), kw = mock_jobs.await_args
        assert payload == {"action": "shell", "cmd": "echo hi", "timeout": 10}
        assert kw["timeout"] == 40
        assert "rc=0" in res and "hi" in res

    @pytest.mark.asyncio
    async def test_jobs_list_format(self):
        tool = CodeExecutionTool()
        _, get, jobs = self._patches({
            "ok": True, "active": 1,
            "jobs": [
                {"job_id": "j1", "cmd": "sleep 9", "cwd": "/tmp",
                 "status": "running", "rc": None, "runtime_s": 3.2},
                {"job_id": "j2", "cmd": "make", "cwd": "/w",
                 "status": "exited", "rc": 0, "runtime_s": 40.0},
            ],
        })
        with get, jobs:
            res = await tool.execute(action="jobs", context=self.CTX)
        assert "1 running" in res
        assert "j1" in res and "sleep 9" in res
        assert "exited(rc=0)" in res

    @pytest.mark.asyncio
    async def test_required_args(self):
        tool = CodeExecutionTool()
        _, get, _ = self._patches({})
        with get:
            for kwargs, need in (
                ({"action": "shell_start"}, "'cmd'"),
                ({"action": "shell"}, "'cmd'"),
                ({"action": "output"}, "'job_id'"),
                ({"action": "write", "job_id": "j1"}, "'text'"),
                ({"action": "wait"}, "'job_id'"),
                ({"action": "exec"}, "'code'"),
            ):
                res = await tool.execute(context=self.CTX, **kwargs)
                assert "Error" in res and need in res, kwargs

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = CodeExecutionTool()
        handle, get, _ = self._patches({})
        with get:
            res = await tool.execute(action="wat", context=self.CTX)
        assert "unknown action 'wat'" in res

    @pytest.mark.asyncio
    async def test_exec_still_works_without_action(self):
        """Backward compat: a code-only call goes down the exec path."""
        tool = CodeExecutionTool()
        handle = MagicMock()
        handle.mount_warning = ""
        with patch.object(
            ConversationSandboxManager, "get_or_create",
            new_callable=AsyncMock, return_value=handle,
        ), patch.object(
            ConversationSandboxManager, "execute_in", new_callable=AsyncMock
        ) as mock_exec, patch.object(
            ConversationSandboxManager, "jobs_in", new_callable=AsyncMock
        ) as mock_jobs:
            mock_exec.return_value = {"ok": True, "stdout": "x", "stderr": "", "error": None}
            res = await tool.execute(code="print('x')", context=self.CTX)
        mock_exec.assert_awaited_once()
        mock_jobs.assert_not_awaited()
        assert "x" in res

    @pytest.mark.asyncio
    async def test_job_error_passthrough(self):
        tool = CodeExecutionTool()
        _, get, jobs = self._patches({"ok": False, "error": "Unknown job: j7"})
        with get, jobs:
            res = await tool.execute(action="output", job_id="j7", context=self.CTX)
        assert "Error" in res and "Unknown job" in res

    def test_call_conflict_key(self):
        tool = CodeExecutionTool()
        for action in ("output", "write", "kill", "wait"):
            key = tool.call_conflict_key({"action": action, "job_id": "j1"})
            assert key == "job:j1", action
        for action in ("shell", "shell_start", "jobs"):
            assert tool.call_conflict_key({"action": action}) is None, action
        # Foreground execs serialize (shared sandbox namespace/lock);
        # background tasks return immediately and need no serialization.
        assert tool.call_conflict_key({"action": "exec"}) == "sandbox-exec"
        assert tool.call_conflict_key({"action": "exec", "background": True}) is None
        assert tool.call_conflict_key({}) == "sandbox-exec"  # default action is foreground exec
        assert tool.call_conflict_key({"action": "output"}) is None  # no job_id


class TestCodeExecutionOutputFiles:
    @pytest.mark.asyncio
    async def test_output_files_saved_as_artifacts(self, tmp_path):
        tool = CodeExecutionTool(tmp_path / "artifacts")
        mock_handle = MagicMock()

        with patch.object(
            ConversationSandboxManager, "get_or_create", new_callable=AsyncMock, return_value=mock_handle
        ), patch.object(
            ConversationSandboxManager, "execute_in", new_callable=AsyncMock
        ) as mock_exec, patch.object(
            ConversationSandboxManager, "read_file_in", new_callable=AsyncMock
        ) as mock_read:
            mock_exec.return_value = {"ok": True, "stdout": "done", "stderr": "", "error": None}
            mock_read.return_value = {
                "ok": True, "name": "result.png", "size": 5, "data": b"pngbytes",
            }
            result = await tool.execute(
                code="open('/tmp/result.png','wb').write(b'pngbytes')",
                output_files=["/tmp/result.png"],
                context={"conversation_id": "conv-1", "assistant_name": "AssistantX"},
            )
            mock_read.assert_awaited_once()
            assert isinstance(result, dict) and result["ok"] is True
            files = result["metadata"]["files"]
            assert len(files) == 1
            assert files[0]["name"] == "result.png"
            assert files[0]["kind"] == "png"  # extension-derived for chat previews
            # artifact actually written
            assert tool._store.resolve("conv-1", "result.png") is not None

    @pytest.mark.asyncio
    async def test_output_files_read_failure_is_a_line_not_a_crash(self, tmp_path):
        tool = CodeExecutionTool(tmp_path / "artifacts")
        mock_handle = MagicMock()

        with patch.object(
            ConversationSandboxManager, "get_or_create", new_callable=AsyncMock, return_value=mock_handle
        ), patch.object(
            ConversationSandboxManager, "execute_in", new_callable=AsyncMock
        ) as mock_exec, patch.object(
            ConversationSandboxManager, "read_file_in", new_callable=AsyncMock
        ) as mock_read:
            mock_exec.return_value = {"ok": True, "stdout": "ok", "stderr": "", "error": None}
            mock_read.return_value = {"ok": False, "error": "Not a file: /tmp/missing"}
            result = await tool.execute(
                code="pass",
                output_files=["/tmp/missing"],
                context={"conversation_id": "conv-1"},
            )
            # no metadata.files, error reported in content
            assert isinstance(result, str)
            assert "Not a file" in result

    @pytest.mark.asyncio
    async def test_output_files_without_store_reports_unavailable(self):
        tool = CodeExecutionTool()  # no artifact_root → no store
        mock_handle = MagicMock()

        with patch.object(
            ConversationSandboxManager, "get_or_create", new_callable=AsyncMock, return_value=mock_handle
        ), patch.object(
            ConversationSandboxManager, "execute_in", new_callable=AsyncMock
        ) as mock_exec, patch.object(
            ConversationSandboxManager, "read_file_in", new_callable=AsyncMock
        ) as mock_read:
            mock_exec.return_value = {"ok": True, "stdout": "", "stderr": "", "error": None}
            result = await tool.execute(
                code="pass",
                output_files=["/tmp/x"],
                context={"conversation_id": "conv-1"},
            )
            assert "unavailable" in result
            mock_read.assert_not_awaited()


class TestConversationSandboxManager:
    def test_singleton(self):
        m1 = ConversationSandboxManager()
        m2 = ConversationSandboxManager()
        assert m1 is m2

    def test_sanitize_conversation_id(self):
        assert _sanitize_conversation_id("abc-123") == "abc-123"
        assert _sanitize_conversation_id("abc/123") == "abc123"
        assert _sanitize_conversation_id("abc 123") == "abc123"
        assert _sanitize_conversation_id("") == "unknown"

    def test_container_name(self):
        name = _container_name("chat-abc")
        assert name.startswith("agentix-sandbox-chat-abc")
        assert len(name) < 70

    @pytest.mark.asyncio
    async def test_get_or_create_docker_not_available(self):
        manager = ConversationSandboxManager()
        with patch.object(
            manager, "_ensure_client", new_callable=AsyncMock
        ) as mock_client:
            mock_client.side_effect = RuntimeError("Docker not available")
            with pytest.raises(RuntimeError, match="Docker"):
                await manager.get_or_create("conv-1")

    @pytest.mark.asyncio
    async def test_schedule_and_cancel_cleanup(self):
        manager = ConversationSandboxManager()
        manager._ttl_seconds = 1

        handle = MagicMock()
        handle.conversation_id = "conv-test"
        handle.timer_handle = None

        manager._schedule_cleanup(handle)
        assert handle.timer_handle is not None
        assert not handle.timer_handle.cancelled()
        timer = handle.timer_handle

        manager._cancel_cleanup(handle)
        assert timer.cancelled()

    @pytest.mark.asyncio
    async def test_destroy_sync_schedules_async(self):
        manager = ConversationSandboxManager()
        with patch.object(manager, "destroy", new_callable=AsyncMock) as mock_destroy:
            manager._destroy_sync("conv-1")
            # Give the event loop a chance to schedule the task
            await asyncio.sleep(0.01)
            mock_destroy.assert_awaited_once_with("conv-1")


class TestToolRegistryContext:
    @pytest.mark.asyncio
    async def test_execute_forwards_context(self):
        registry = ToolRegistry()
        tool = registry.get_tool("code_execution")

        with patch.object(
            ConversationSandboxManager, "get_or_create", new_callable=AsyncMock
        ) as mock_get, patch.object(
            ConversationSandboxManager, "execute_in", new_callable=AsyncMock
        ) as mock_exec:
            mock_handle = MagicMock()
            mock_handle.container_name = "agentix-sandbox-test"
            mock_handle.host_port = 9999
            mock_get.return_value = mock_handle
            mock_exec.return_value = {"ok": True, "stdout": "hi", "stderr": "", "error": None}

            result = await registry.execute(
                "code_execution",
                {"code": "print('hi')"},
                context={"conversation_id": "conv-registry"},
            )
            mock_get.assert_awaited_once_with("conv-registry", sandbox_config={}, agent_workspace_dir=None)
            assert result["ok"] is True
            assert "hi" in result["content"]


class TestConversationSandboxManagerCleanup:
    @pytest.mark.asyncio
    async def test_destroy_all_cleans_multiple_containers(self):
        manager = ConversationSandboxManager()
        handle1 = SandboxHandle(conversation_id="conv-a", container_name="agentix-sandbox-conv-a-123")
        handle2 = SandboxHandle(conversation_id="conv-b", container_name="agentix-sandbox-conv-b-456")
        manager._cache["conv-a"] = handle1
        manager._cache["conv-b"] = handle2

        with patch.object(manager, "destroy", new_callable=AsyncMock) as mock_destroy:
            await manager.destroy_all()
            assert mock_destroy.await_count == 2
            calls = {call.args[0] for call in mock_destroy.await_args_list}
            assert calls == {"conv-a", "conv-b"}

    @pytest.mark.asyncio
    async def test_destroy_graceful_stop_then_kill_then_remove(self):
        manager = ConversationSandboxManager()
        mock_container = MagicMock()
        mock_container.stop.side_effect = RuntimeError("stop failed")

        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container
        with patch.object(manager, "_ensure_client", new_callable=AsyncMock, return_value=mock_client):
            await manager.destroy("conv-test")
            mock_container.stop.assert_called_once_with(timeout=5)
            mock_container.kill.assert_called_once()
            mock_container.remove.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_destroy_remove_failure_is_logged_not_raised(self):
        manager = ConversationSandboxManager()
        mock_container = MagicMock()
        mock_container.remove.side_effect = RuntimeError("remove failed")

        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container
        with patch.object(manager, "_ensure_client", new_callable=AsyncMock, return_value=mock_client):
            await manager.destroy("conv-test")
            mock_container.stop.assert_called_once_with(timeout=5)
            mock_container.remove.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_destroy_all_empty_cache_is_noop(self):
        manager = ConversationSandboxManager()
        manager._cache.clear()
        # Should not raise or call destroy
        with patch.object(manager, "destroy", new_callable=AsyncMock) as mock_destroy:
            await manager.destroy_all()
            mock_destroy.assert_not_awaited()


class TestValidationLimits:
    @pytest.mark.asyncio
    async def test_code_timeout_clamped(self):
        from agent.tools.validation import validate_and_normalize_args
        from agent.tools.builtin import CodeExecutionTool

        tool = CodeExecutionTool()
        schema = tool.tool_schema
        normalized, error, _notes = validate_and_normalize_args(
            "code_execution",
            {"code": "print(1)", "timeout": 9999},
            schema,
        )
        assert error is None
        assert normalized["timeout"] == 180


class TestSandboxWorkspaceMount:
    @pytest.mark.asyncio
    async def test_cache_hit_with_matching_label(self, tmp_path):
        from agent.tools.docker_sandbox import _workspace_mount_label

        manager = ConversationSandboxManager()
        manager._cache.clear()
        ws = str(tmp_path / "ws")
        from agent.tools.docker_sandbox import (
            _egress_label, _resolve_egress_policy,
        )
        handle = SandboxHandle(
            conversation_id="conv-ws",
            container_name="agentix-sandbox-conv-ws-1",
            workspace_label=_workspace_mount_label(ws),
            egress_label=_egress_label(_resolve_egress_policy(None)),
        )
        manager._cache["conv-ws"] = handle
        with patch.object(manager, "_check_docker"), patch.object(
            manager, "_ensure_network", new_callable=AsyncMock
        ), patch.object(
            manager, "_container_is_running", new_callable=AsyncMock, return_value=True
        ):
            result = await manager.get_or_create("conv-ws", agent_workspace_dir=ws)
            assert result is handle

    @pytest.mark.asyncio
    async def test_stale_workspace_label_forces_recreate(self, tmp_path):
        """A cached container created before the workspace was enabled must be
        dropped — otherwise /agent_ws silently misses the mount (the daemon-
        side ws_ok check never runs for cached handles)."""
        manager = ConversationSandboxManager()
        manager._cache.clear()
        stale = SandboxHandle(
            conversation_id="conv-ws2",
            container_name="agentix-sandbox-conv-ws2-1",
            workspace_label="none",
        )
        manager._cache["conv-ws2"] = stale
        fresh = SandboxHandle(conversation_id="conv-ws2", container_name="fresh")
        with patch.object(manager, "_check_docker"), patch.object(
            manager, "_ensure_network", new_callable=AsyncMock
        ), patch.object(
            # step 1: cached container still running; step 2: skip adoption
            manager, "_container_is_running", new_callable=AsyncMock,
            side_effect=[True, False],
        ), patch.object(
            manager, "_start_container", new_callable=AsyncMock, return_value=fresh
        ) as mock_start:
            result = await manager.get_or_create(
                "conv-ws2", agent_workspace_dir=str(tmp_path / "ws")
            )
        assert result is fresh
        mock_start.assert_awaited_once()

    @staticmethod
    def _start_kwargs_client() -> MagicMock:
        client = MagicMock()
        client.containers.get.side_effect = Exception("no leftover")
        container = MagicMock()
        container.ports = {}
        # The same mock serves the sandbox AND the egress sidecar call —
        # wait() must return a clean rc for the sidecar path.
        container.wait.return_value = {"StatusCode": 0}
        client.containers.run.return_value = container
        return client

    @pytest.mark.asyncio
    async def test_start_container_hardening_kwargs(self):
        manager = ConversationSandboxManager()
        client = self._start_kwargs_client()
        with patch.object(
            manager, "_ensure_client", new_callable=AsyncMock, return_value=client
        ), patch.object(
            manager, "_ensure_sandbox_image", new_callable=AsyncMock, return_value="img"
        ), patch.object(
            manager, "_wait_for_ready", new_callable=AsyncMock,
            return_value="http://sandbox:8080",
        ):
            handle = await manager._start_container("conv-h", "name-h")
        # call 0 = sandbox, call 1 = netctl egress sidecar
        kwargs = client.containers.run.call_args_list[0].kwargs
        assert kwargs["cap_drop"] == ["ALL"]
        assert kwargs["pids_limit"] == 512
        assert "no-new-privileges:true" in kwargs["security_opt"]
        assert kwargs["dns_search"] == ["."]
        assert kwargs["init"] is True
        assert kwargs["sysctls"]["net.ipv4.ip_forward"] == "0"
        assert handle.workspace_label == "none"
        assert handle.mount_warning == ""
        # egress sidecar: shares the sandbox netns, NET_ADMIN only — and
        # must run as root (the image's USER sandboxuser would leave
        # cap_add'd NET_ADMIN out of the effective set → iptables fails).
        sidecar = client.containers.run.call_args_list[1].kwargs
        assert sidecar["network_mode"] == "container:name-h"
        assert sidecar["cap_add"] == ["NET_ADMIN"]
        assert sidecar["cap_drop"] == ["ALL"]
        assert sidecar["user"] == "0"
        # v4 chain must be `&&`-joined to the v6 block — a `;` separator
        # masks a failed v4 chain when ip6tables is absent (fail-open).
        script = sidecar["command"][2]
        assert "; if " not in script
        assert " && if [ -f /proc/net/if_inet6 ]" in script

    @pytest.mark.asyncio
    async def test_suspicious_agent_workspace_source_skips_mount(self):
        """AGENT_WORKSPACE_ROOT=/ (or a failed derivation) must not bind-mount
        a host subtree into the sandbox — fail closed, warn visibly."""
        manager = ConversationSandboxManager()
        client = self._start_kwargs_client()
        with patch.object(
            manager, "_ensure_client", new_callable=AsyncMock, return_value=client
        ), patch.object(
            manager, "_ensure_sandbox_image", new_callable=AsyncMock, return_value="img"
        ), patch.object(
            manager, "_wait_for_ready", new_callable=AsyncMock,
            return_value="http://sandbox:8080",
        ):
            handle = await manager._start_container(
                "conv-s", "name-s", agent_workspace_dir="/"
            )
        vols = client.containers.run.call_args_list[0].kwargs.get("volumes") or {}
        assert all(v.get("bind") != "/agent_ws" for v in vols.values())
        assert "suspicious" in handle.mount_warning


class TestMountWarningSurfacing:
    @pytest.mark.asyncio
    async def test_mount_warning_surfaced_once(self):
        tool = CodeExecutionTool()
        mock_handle = MagicMock()
        mock_handle.mount_warning = "/agent_ws mount skipped (suspicious bind source /)"
        with patch.object(
            ConversationSandboxManager, "get_or_create",
            new_callable=AsyncMock, return_value=mock_handle,
        ), patch.object(
            ConversationSandboxManager, "execute_in", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = {
                "ok": True, "stdout": "x", "stderr": "", "error": None,
            }
            r1 = await tool.execute(code="1", context={"conversation_id": "c"})
            r2 = await tool.execute(code="2", context={"conversation_id": "c"})
        assert "mount skipped" in r1
        assert "mount skipped" not in r2


class TestEgressPolicy:
    """Pure-function egress policy tests — no Docker client needed."""

    def test_parse_cidr_list_filters_invalid(self):
        from agent.tools.docker_sandbox import _parse_cidr_list

        valid, invalid = _parse_cidr_list("10.0.0.0/8, not-a-cidr, ::1/128, 1.2.3.4")
        assert valid == ["10.0.0.0/8", "::1/128", "1.2.3.4/32"]
        assert invalid == ["not-a-cidr"]

    def test_parse_cidr_list_accepts_list(self):
        from agent.tools.docker_sandbox import _parse_cidr_list

        valid, invalid = _parse_cidr_list(["192.168.1.0/24", "bogus"])
        assert valid == ["192.168.1.0/24"]
        assert invalid == ["bogus"]

    def test_defaults_internet_on_lan_off(self):
        from agent.tools.docker_sandbox import _resolve_egress_policy

        pol = _resolve_egress_policy(None)
        assert pol["internet"] is True
        assert pol["local_network"] is False
        assert pol["blocked"] == []
        assert pol["invalid"] == []

    def test_toggles_and_blocked(self):
        from agent.tools.docker_sandbox import _resolve_egress_policy

        pol = _resolve_egress_policy(
            {
                "egress_internet": False,
                "egress_local_network": True,
                "egress_blocked": ["203.0.113.0/24", "bad entry"],
            }
        )
        assert pol["internet"] is False
        assert pol["local_network"] is True
        assert "203.0.113.0/24" in pol["blocked"]
        assert pol["invalid"] == ["bad entry"]

    def test_env_floor_forces_internet_off(self):
        import agent.tools.docker_sandbox as ds
        from agent.tools.docker_sandbox import _resolve_egress_policy

        with patch.object(ds, "SANDBOX_NETWORK_INTERNAL", True):
            pol = _resolve_egress_policy({"egress_internet": True})
        assert pol["internet"] is False

    def test_env_blocked_cidrs_union_and_dedup(self):
        import agent.tools.docker_sandbox as ds
        from agent.tools.docker_sandbox import _resolve_egress_policy

        with patch.object(ds, "SANDBOX_EGRESS_BLOCKED_CIDRS", "10.9.0.0/16"):
            pol = _resolve_egress_policy(
                {"egress_blocked": ["10.9.0.0/16", "172.31.0.0/16"]}
            )
        assert pol["blocked"] == ["10.9.0.0/16", "172.31.0.0/16"]

    def test_label_changes_with_policy(self):
        from agent.tools.docker_sandbox import (
            _egress_label,
            _resolve_egress_policy,
        )

        base = _egress_label(_resolve_egress_policy(None))
        assert _egress_label(_resolve_egress_policy(None)) == base
        changed = _egress_label(
            _resolve_egress_policy({"egress_internet": False})
        )
        assert changed != base

    def test_rules_order_and_families(self):
        from agent.tools.docker_sandbox import (
            _egress_rules,
            _resolve_egress_policy,
        )

        pol = _resolve_egress_policy(
            {"egress_blocked": ["10.20.0.0/16", "fd00::/8"]}
        )
        v4 = _egress_rules(pol)
        v6 = _egress_rules(pol, ipv6=True)
        v4_joined = [" ".join(r) for r in v4]
        v6_joined = [" ".join(r) for r in v6]
        # v4 ranges never reach ip6tables and vice versa
        assert any("10.20.0.0/16" in r for r in v4_joined)
        assert not any("fd00" in r for r in v4_joined)
        assert any("fd00" in r for r in v6_joined)
        assert not any("10.20.0.0/16" in r for r in v6_joined)
        # metadata drop precedes the LAN drops so LAN-on cannot reopen it
        meta_idx = next(
            i for i, r in enumerate(v4_joined) if "169.254.169.254" in r
        )
        lan_idx = next(
            i for i, r in enumerate(v4_joined) if "192.168.0.0/16" in r
        )
        assert meta_idx < lan_idx

    def test_internet_off_drops_dns_and_ends_with_drop(self):
        from agent.tools.docker_sandbox import (
            _egress_rules,
            _resolve_egress_policy,
        )

        pol = _resolve_egress_policy({"egress_internet": False})
        rules = [" ".join(r) for r in _egress_rules(pol)]
        dns_idx = next(i for i, r in enumerate(rules) if "127.0.0.11" in r)
        lo_idx = next(i for i, r in enumerate(rules) if r == "-o lo -j ACCEPT")
        # embedded-DNS kill must precede the loopback accept (127.0.0.11
        # traffic egresses via lo) — otherwise DNS-tunnel exfil stays open.
        assert dns_idx < lo_idx
        assert rules[-1] == "-j DROP"

    def test_internet_on_keeps_dns(self):
        from agent.tools.docker_sandbox import (
            _egress_rules,
            _resolve_egress_policy,
        )

        pol = _resolve_egress_policy({"egress_internet": True})
        rules = [" ".join(r) for r in _egress_rules(pol)]
        assert not any("127.0.0.11" in r for r in rules)
        assert rules[-1] != "-j DROP"

    def test_sidecar_script_is_valid_and_v4_failure_is_fail_closed(self, tmp_path):
        import subprocess

        from agent.tools.docker_sandbox import (
            _egress_sidecar_script,
            _resolve_egress_policy,
        )

        script = _egress_sidecar_script(_resolve_egress_policy(None))
        syntax = subprocess.run(
            ["/bin/sh", "-n", "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert syntax.returncode == 0, syntax.stderr

        # Simulate an unavailable/denied iptables command. Even when IPv6
        # rules are skipped, the generated script must retain a nonzero rc.
        fake_iptables = tmp_path / "iptables"
        fake_iptables.write_text("#!/bin/sh\nexit 1\n")
        fake_iptables.chmod(0o755)
        env = dict(os.environ, PATH=str(tmp_path))
        result = subprocess.run(
            ["/bin/sh", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0

    @pytest.mark.asyncio
    async def test_sidecar_failure_fails_closed(self):
        """A failing netctl sidecar must destroy the sandbox container —
        never hand out an unfiltered egress sandbox."""
        manager = ConversationSandboxManager()
        client = MagicMock()
        client.containers.get.side_effect = Exception("no leftover")
        container = MagicMock()
        sidecar = MagicMock()
        sidecar.wait.return_value = {"StatusCode": 2}
        sidecar.logs.return_value = b"iptables: Permission denied"
        client.containers.run.side_effect = [container, sidecar]
        with patch.object(
            manager, "_ensure_client", new_callable=AsyncMock, return_value=client
        ), patch.object(
            manager, "_ensure_sandbox_image", new_callable=AsyncMock,
            return_value="img",
        ):
            with pytest.raises(RuntimeError, match="egress policy"):
                await manager._start_container("conv-x", "name-x")
        container.stop.assert_called_once()
        container.remove.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_policy_change_recreates_cached_sandbox(self):
        """Same conversation, new egress config → old handle is dropped."""
        manager = ConversationSandboxManager()
        manager._cache.clear()
        from agent.tools.docker_sandbox import (
            _egress_label,
            _resolve_egress_policy,
        )

        handle = SandboxHandle(
            conversation_id="conv-p",
            container_name="agentix-sandbox-conv-p-1",
            egress_label=_egress_label(_resolve_egress_policy(None)),
        )
        manager._cache["conv-p"] = handle
        fresh = SandboxHandle(conversation_id="conv-p", container_name="fresh")
        with patch.object(manager, "_check_docker"), patch.object(
            manager, "_ensure_network", new_callable=AsyncMock
        ), patch.object(
            manager, "_container_is_running", new_callable=AsyncMock,
            side_effect=[True, False],
        ), patch.object(
            manager, "_start_container", new_callable=AsyncMock,
            return_value=fresh,
        ) as mock_start:
            result = await manager.get_or_create(
                "conv-p", sandbox_config={"egress_internet": False}
            )
        assert result is fresh
        mock_start.assert_awaited_once()


def test_sandbox_server_scrubs_secrets_from_environ():
    """Importing the in-container server must strip SANDBOX_TOKEN and the
    pip-policy vars from os.environ so exec'd code cannot read them back."""
    import json
    import subprocess
    import sys

    env = dict(os.environ)
    env["SANDBOX_TOKEN"] = "supersecret"
    env["SANDBOX_PIP_INDEX_URL"] = "https://mirror.internal/simple"
    env["SANDBOX_PIP_ONLY_BINARY"] = "1"
    code = (
        "import json, os, importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location(\n"
        "    'sandbox_server', sys.argv[1])\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "print(json.dumps({\n"
        "    'token': os.environ.get('SANDBOX_TOKEN', 'GONE'),\n"
        "    'pip_url': os.environ.get('SANDBOX_PIP_INDEX_URL', 'GONE'),\n"
        "    'pip_bin': os.environ.get('SANDBOX_PIP_ONLY_BINARY', 'GONE'),\n"
        "    'expected': mod._EXPECTED_TOKEN,\n"
        "}))\n"
    )
    from agent.tools.docker_sandbox import _SANDBOX_SERVER_SCRIPT

    out = subprocess.run(
        [sys.executable, "-c", code, str(_SANDBOX_SERVER_SCRIPT)],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout.strip().splitlines()[-1])
    assert data["token"] == "GONE"
    assert data["pip_url"] == "GONE"
    assert data["pip_bin"] == "GONE"
    # the server still remembers the token it was started with
    assert data["expected"] == "supersecret"


class TestBackgroundExecTasks:
    """background=true exec tasks: async runs, result collection, nudges."""

    CTX = {"conversation_id": "conv-bg"}

    def _patches(self, exec_result=None, exec_side_effect=None):
        handle = MagicMock()
        handle.mount_warning = ""
        get = patch.object(
            ConversationSandboxManager, "get_or_create",
            new_callable=AsyncMock, return_value=handle,
        )
        exe = patch.object(
            ConversationSandboxManager, "execute_in",
            new_callable=AsyncMock,
            return_value=exec_result if exec_side_effect is None else None,
            side_effect=exec_side_effect,
        )
        return handle, get, exe

    def _ok(self, stdout="hi", error=None):
        return {"ok": error is None, "stdout": stdout, "stderr": "", "error": error}

    @pytest.mark.asyncio
    async def test_background_returns_task_id_and_completes(self):
        tool = CodeExecutionTool()
        _, get, exe = self._patches(self._ok("hello bg"))
        with get, exe:
            res = await tool.execute(
                action="exec", code="print('hi')", background=True, context=self.CTX,
            )
            assert res["ok"] is True
            task_id = res["metadata"]["exec_task"]
            assert task_id.startswith("task-")
            assert task_id in res["content"]
            record = tool._exec_tasks["conv-bg"][task_id]
            assert record["status"] == "running"
            await record["task"]
            assert record["status"] == "done"
            assert "hello bg" in record["content"]

    @pytest.mark.asyncio
    async def test_result_collects_finished_task(self):
        tool = CodeExecutionTool()
        _, get, exe = self._patches(self._ok("payload"))
        with get as mock_get, exe:
            res = await tool.execute(code="print(1)", background=True, context=self.CTX)
            task_id = res["metadata"]["exec_task"]
            await tool._exec_tasks["conv-bg"][task_id]["task"]
            calls_after_start = mock_get.call_count
            out = await tool.execute(action="result", job_id=task_id, context=self.CTX)
            assert "payload" in out
            # result must NOT boot a sandbox — get_or_create count unchanged
            assert mock_get.call_count == calls_after_start

    @pytest.mark.asyncio
    async def test_result_on_running_task_reports_status(self):
        tool = CodeExecutionTool()
        gate = asyncio.Event()

        async def slow(*a, **k):
            await gate.wait()
            return self._ok()

        _, get, exe = self._patches(exec_side_effect=slow)
        with get, exe:
            res = await tool.execute(code="x=1", background=True, context=self.CTX)
            task_id = res["metadata"]["exec_task"]
            out = await tool.execute(action="result", job_id=task_id, context=self.CTX)
            assert "still running" in out
            gate.set()
            await tool._exec_tasks["conv-bg"][task_id]["task"]

    @pytest.mark.asyncio
    async def test_completion_nudge_reaches_queue(self):
        tool = CodeExecutionTool()
        queue = asyncio.Queue()
        event = asyncio.Event()
        ctx = {**self.CTX, "injected_input_queue": queue, "input_event": event}
        _, get, exe = self._patches(self._ok("nudge-me"))
        with get, exe:
            res = await tool.execute(code="print(1)", background=True, context=ctx)
            task_id = res["metadata"]["exec_task"]
            await tool._exec_tasks["conv-bg"][task_id]["task"]
            item = queue.get_nowait()
            assert item["source"] == "code_execution"
            assert task_id in item["content"]
            assert "result" in item["content"]
            assert event.is_set()  # no consumer in this test — set() must stick
            text = item["content"]
            assert "nudge-me" in text  # short tail included

    @pytest.mark.asyncio
    async def test_wait_returns_result_when_done(self):
        tool = CodeExecutionTool()
        _, get, exe = self._patches(self._ok("waited"))
        with get, exe:
            res = await tool.execute(code="print(1)", background=True, context=self.CTX)
            task_id = res["metadata"]["exec_task"]
            await tool._exec_tasks["conv-bg"][task_id]["task"]
            out = await tool.execute(action="wait", job_id=task_id, context=self.CTX)
            assert "waited" in out

    @pytest.mark.asyncio
    async def test_wait_blocks_until_done(self):
        tool = CodeExecutionTool()
        gate = asyncio.Event()

        async def slow(*a, **k):
            await gate.wait()
            return self._ok("late")

        _, get, exe = self._patches(exec_side_effect=slow)
        with get, exe:
            res = await tool.execute(code="x=1", background=True, context=self.CTX)
            task_id = res["metadata"]["exec_task"]

            async def release():
                await asyncio.sleep(0.05)
                gate.set()

            asyncio.ensure_future(release())
            out = await tool.execute(action="wait", job_id=task_id, timeout=30, context=self.CTX)
            assert "late" in out

    @pytest.mark.asyncio
    async def test_wait_woken_by_input_event(self):
        tool = CodeExecutionTool()
        gate = asyncio.Event()

        async def slow(*a, **k):
            await gate.wait()
            return self._ok()

        _, get, exe = self._patches(exec_side_effect=slow)
        input_event = asyncio.Event()
        ctx = {**self.CTX, "input_event": input_event}
        with get, exe:
            res = await tool.execute(code="x=1", background=True, context=ctx)
            task_id = res["metadata"]["exec_task"]

            async def wake():
                await asyncio.sleep(0.05)
                input_event.set()

            asyncio.ensure_future(wake())
            out = await tool.execute(action="wait", job_id=task_id, timeout=60, context=ctx)
            assert "Woken early" in out
            gate.set()
            await tool._exec_tasks["conv-bg"][task_id]["task"]

    @pytest.mark.asyncio
    async def test_kill_cancels_task(self):
        tool = CodeExecutionTool()
        gate = asyncio.Event()

        async def slow(*a, **k):
            await gate.wait()
            return self._ok()

        _, get, exe = self._patches(exec_side_effect=slow)
        with get, exe:
            res = await tool.execute(code="x=1", background=True, context=self.CTX)
            task_id = res["metadata"]["exec_task"]
            out = await tool.execute(action="kill", job_id=task_id, context=self.CTX)
            assert "cancelled" in out
            rec = tool._exec_tasks["conv-bg"][task_id]
            try:
                await rec["task"]
            except asyncio.CancelledError:
                pass
            assert rec["status"] == "cancelled"
            out = await tool.execute(action="kill", job_id=task_id, context=self.CTX)
            assert "already" in out
            gate.set()

    @pytest.mark.asyncio
    async def test_jobs_lists_tasks_alongside_shell_jobs(self):
        tool = CodeExecutionTool()
        handle = MagicMock()
        handle.mount_warning = ""
        get = patch.object(
            ConversationSandboxManager, "get_or_create",
            new_callable=AsyncMock, return_value=handle,
        )
        exe = patch.object(
            ConversationSandboxManager, "execute_in",
            new_callable=AsyncMock, return_value=self._ok(),
        )
        jobs = patch.object(
            ConversationSandboxManager, "jobs_in",
            new_callable=AsyncMock,
            return_value={"ok": True, "jobs": [{"job_id": "job-1", "status": "running"}]},
        )
        with get, exe, jobs:
            res = await tool.execute(code="x=1", background=True, context=self.CTX)
            task_id = res["metadata"]["exec_task"]
            out = await tool.execute(action="jobs", context=self.CTX)
            assert "job-1" in out
            assert task_id in out
            assert "Exec tasks" in out
            rec = tool._exec_tasks["conv-bg"][task_id]
            if not rec["task"].done():
                rec["task"].cancel()
                try:
                    await rec["task"]
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_cross_type_actions_rejected(self):
        tool = CodeExecutionTool()
        # result on a shell job id
        out = await tool.execute(action="result", job_id="job-1", context=self.CTX)
        assert "task" in out.lower() and "Error" in out
        # output/write on a task id — dispatched before any sandbox boot
        out = await tool.execute(action="output", job_id="task-deadbeef", context=self.CTX)
        assert "shell jobs" in out
        out = await tool.execute(action="write", job_id="task-deadbeef", text="x", context=self.CTX)
        assert "shell jobs" in out
        # wait/result on unknown task
        out = await tool.execute(action="wait", job_id="task-deadbeef", context=self.CTX)
        assert "no exec task" in out

    @pytest.mark.asyncio
    async def test_running_task_cap(self):
        tool = CodeExecutionTool()
        gate = asyncio.Event()

        async def slow(*a, **k):
            await gate.wait()
            return self._ok()

        _, get, exe = self._patches(exec_side_effect=slow)
        with get, exe:
            started = []
            for _ in range(10):
                r = await tool.execute(code="x=1", background=True, context=self.CTX)
                started.append(r["metadata"]["exec_task"])
            over = await tool.execute(code="x=1", background=True, context=self.CTX)
            assert "Error" in over and "10" in over
            gate.set()
            for tid in started:
                await tool._exec_tasks["conv-bg"][tid]["task"]

    @pytest.mark.asyncio
    async def test_exec_error_status_and_boot_failure(self):
        tool = CodeExecutionTool()
        _, get, exe = self._patches(self._ok("", error="boom"))
        with get, exe:
            res = await tool.execute(code="1/0", background=True, context=self.CTX)
            task_id = res["metadata"]["exec_task"]
            await tool._exec_tasks["conv-bg"][task_id]["task"]
            rec = tool._exec_tasks["conv-bg"][task_id]
            assert rec["status"] == "error"
            out = await tool.execute(action="result", job_id=task_id, context=self.CTX)
            assert "boom" in out

    @pytest.mark.asyncio
    async def test_sandbox_boot_inside_task(self):
        """Cold-boot failures land in the task result, not the start call."""
        tool = CodeExecutionTool()
        get = patch.object(
            ConversationSandboxManager, "get_or_create",
            new_callable=AsyncMock, side_effect=RuntimeError("docker offline"),
        )
        with get:
            res = await tool.execute(code="x=1", background=True, context=self.CTX)
            task_id = res["metadata"]["exec_task"]
            assert res["ok"] is True  # start itself never blocks on Docker
            await tool._exec_tasks["conv-bg"][task_id]["task"]
            rec = tool._exec_tasks["conv-bg"][task_id]
            assert rec["status"] == "error"
            assert "docker offline" in rec["content"]

    @pytest.mark.asyncio
    async def test_result_delivers_output_files(self, tmp_path):
        store_root = tmp_path / "artifacts"
        tool = CodeExecutionTool(artifact_root=store_root)
        handle = MagicMock()
        handle.mount_warning = ""
        get = patch.object(
            ConversationSandboxManager, "get_or_create",
            new_callable=AsyncMock, return_value=handle,
        )
        exe = patch.object(
            ConversationSandboxManager, "execute_in",
            new_callable=AsyncMock, return_value=self._ok("made a file"),
        )
        read = patch.object(
            ConversationSandboxManager, "read_file_in",
            new_callable=AsyncMock,
            return_value={"ok": True, "name": "out.txt", "data": b"payload"},
        )
        with get, exe, read:
            res = await tool.execute(
                code="open('/tmp/out.txt','w')", background=True,
                output_files=["/tmp/out.txt"], context=self.CTX,
            )
            task_id = res["metadata"]["exec_task"]
            await tool._exec_tasks["conv-bg"][task_id]["task"]
            out = await tool.execute(action="result", job_id=task_id, context=self.CTX)
            assert out["metadata"]["files"]
            assert out["metadata"]["files"][0]["name"] == "out.txt"
            assert (store_root / "conv-bg" / "out.txt").exists()

    def test_schema_has_background(self):
        tool = CodeExecutionTool()
        props = tool.tool_schema["function"]["parameters"]["properties"]
        assert props["background"]["type"] == "boolean"
        assert "result" in tool.tool_schema["function"]["parameters"]["properties"]["action"]["enum"]

    def test_conflict_key_serializes_same_task_only(self):
        tool = CodeExecutionTool()
        k1 = tool.call_conflict_key({"action": "wait", "job_id": "task-1"})
        k2 = tool.call_conflict_key({"action": "wait", "job_id": "task-2"})
        k3 = tool.call_conflict_key({"action": "result", "job_id": "task-1"})
        assert k1 == k3 and k1 != k2
        assert tool.call_conflict_key({"action": "exec", "background": True}) is None

    @pytest.mark.asyncio
    async def test_background_task_installs_packages(self):
        tool = CodeExecutionTool()
        handle = MagicMock()
        handle.mount_warning = ""
        get = patch.object(
            ConversationSandboxManager, "get_or_create",
            new_callable=AsyncMock, return_value=handle,
        )
        inst = patch.object(
            ConversationSandboxManager, "install_in",
            new_callable=AsyncMock, return_value={"ok": True},
        )
        exe = patch.object(
            ConversationSandboxManager, "execute_in",
            new_callable=AsyncMock,
            return_value={"ok": True, "stdout": "pkg ok", "stderr": "", "error": None},
        )
        with get, inst as mock_inst, exe:
            res = await tool.execute(
                code="import requests", background=True, packages=["requests"],
                context=self.CTX,
            )
            task_id = res["metadata"]["exec_task"]
            await tool._exec_tasks["conv-bg"][task_id]["task"]
            rec = tool._exec_tasks["conv-bg"][task_id]
            assert rec["status"] == "done"
            assert "pkg ok" in rec["content"]
            mock_inst.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_background_task_install_failure_is_error(self):
        tool = CodeExecutionTool()
        handle = MagicMock()
        handle.mount_warning = ""
        get = patch.object(
            ConversationSandboxManager, "get_or_create",
            new_callable=AsyncMock, return_value=handle,
        )
        inst = patch.object(
            ConversationSandboxManager, "install_in",
            new_callable=AsyncMock,
            return_value={"ok": False, "error": "no network", "stderr": ""},
        )
        exe = patch.object(ConversationSandboxManager, "execute_in", new_callable=AsyncMock)
        with get, inst, exe as mock_exec:
            res = await tool.execute(
                code="x=1", background=True, packages=["nonexistent-pkg"],
                context=self.CTX,
            )
            task_id = res["metadata"]["exec_task"]
            await tool._exec_tasks["conv-bg"][task_id]["task"]
            rec = tool._exec_tasks["conv-bg"][task_id]
            assert rec["status"] == "error"
            assert "no network" in rec["content"]
            mock_exec.assert_not_awaited()


# ---------------------------------------------------------------------------
# queue_wait plumbing — busy-sandbox semantics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_foreground_exec_passes_short_queue_wait():
    """A busy sandbox must fail the interactive call fast, not stall it."""
    from agent.tools.builtin_tools.code_execution import _BUSY_QUEUE_WAIT_S

    tool = CodeExecutionTool()
    handle = MagicMock()
    handle.mount_warning = ""
    with patch.object(
        ConversationSandboxManager, "get_or_create",
        new_callable=AsyncMock, return_value=handle,
    ), patch.object(
        ConversationSandboxManager, "execute_in", new_callable=AsyncMock,
        return_value={"ok": True, "stdout": "", "stderr": "", "error": None},
    ) as mock_exec:
        await tool.execute(code="print(1)", context={"conversation_id": "conv-qw"})
        assert mock_exec.await_args.kwargs["queue_wait"] == _BUSY_QUEUE_WAIT_S


@pytest.mark.asyncio
async def test_background_task_passes_full_queue_wait():
    """Background tasks queue behind a running exec for the exec cap."""
    from agent.tools.docker_sandbox import EXEC_MAX_TIMEOUT

    tool = CodeExecutionTool()
    handle = MagicMock()
    handle.mount_warning = ""
    with patch.object(
        ConversationSandboxManager, "get_or_create",
        new_callable=AsyncMock, return_value=handle,
    ), patch.object(
        ConversationSandboxManager, "execute_in", new_callable=AsyncMock,
        return_value={"ok": True, "stdout": "", "stderr": "", "error": None},
    ) as mock_exec:
        res = await tool.execute(
            code="x=1", background=True, context={"conversation_id": "conv-qw2"}
        )
        task_id = res["metadata"]["exec_task"]
        await tool._exec_tasks["conv-qw2"][task_id]["task"]
        assert mock_exec.await_args.kwargs["queue_wait"] == EXEC_MAX_TIMEOUT
