"""Code execution builtin tool."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_OUTPUT_FILES = 10
_MAX_OUTPUT_FILE_BYTES = 32 * 1024 * 1024
_JOB_WAIT_MAX = 120
_MAX_BG_TASKS = 10
_MAX_FINISHED_TASKS = 50
_NUDGE_TAIL_CHARS = 300
# Interactive execs get only a short lock queue wait — a sandbox busy with a
# long exec should fail fast with an actionable error instead of stalling the
# tool call for its full timeout. Background tasks queue much longer.
_BUSY_QUEUE_WAIT_S = 5.0


class CodeExecutionTool:
    """Run Python code inside a conversation-persistent Docker sandbox."""

    name = "code_execution"
    description = (
        "Execute Python code in an isolated sandbox container. "
        "The container persists across multiple calls in the same conversation, "
        "so variables, imports, written files and installed packages survive. "
        "The working directory is /tmp (writable); the rest of the filesystem "
        "may be read-only — always write files under /tmp (or /workspace if "
        "it exists) using absolute paths. When the assistant has an agent "
        "workspace enabled, it is mounted at /agent_ws — files stored there "
        "persist beyond this conversation's sandbox lifetime. "
        "Outbound network access is controlled by the administrator's egress "
        "policy (internet/local-network toggles and blocked IP ranges) — if "
        "network code fails, assume the destination is disallowed rather "
        "than retrying. "
        "Persistent background jobs (tmux-style): action='shell_start' + "
        "'cmd' launches a long-running process that KEEPS RUNNING between "
        "calls and returns job_id — run several jobs in parallel (e.g. a "
        "server + its test suite). action='output' (job_id, optional "
        "'tail'/'stream') reads captured stdout/stderr, 'write' sends a "
        "line to the job's stdin, 'wait' blocks until the job exits "
        "(returns early when a user message arrives), 'kill' terminates "
        "the whole process group, 'jobs' lists all jobs. action='shell' "
        "runs one shell command synchronously. "
        "Asynchronous Python tasks: action='exec' + background=true returns "
        "'task-N' immediately (sandbox boot included) — you keep working, "
        "get auto-notified on completion, and collect the full output and "
        "attached files via action='result' + job_id='task-N'; 'wait' blocks "
        "until done (returns early on new input), 'kill' cancels, 'jobs' "
        "lists exec tasks alongside shell jobs. Up to 10 tasks can be "
        "registered per conversation — they SHARE the sandbox namespace and "
        "execute strictly one at a time (a second task queues behind the "
        "running exec, it does NOT run in parallel). A foreground 'exec' on "
        "a busy sandbox fails fast with a 'still running' error — collect or "
        "'wait' on the task instead of retrying. 'timeout' applies to "
        "background tasks too (default 120s, max 600). Check 'jobs'/'result' "
        "before finishing when tasks may still be running. "
        "egress policy and filesystem; /tmp is noexec — invoke scripts via "
        "'sh x.sh'/'python x.py', never './x.sh'. Max 20 jobs; output "
        "keeps the last ~4000 lines per stream. "
        "For action='exec' (the default when 'action' is omitted): 'code' "
        "(required), optional 'packages' (list of pip package "
        "names to install first — requires network egress; if installation "
        "fails with a network/isolation error, rewrite the code to use only "
        "the standard library instead of retrying), optional 'timeout' "
        "(seconds, default 120 — ask for more for long computations, capped "
        "by the sandbox's exec limit), and optional 'output_files': up to "
        "10 absolute file paths "
        "(max 32 MB each) created inside the sandbox that are delivered to "
        "the user as downloadable chat attachments — e.g. /tmp/result.png. "
        "Do NOT retry output_files delivery repeatedly: if it reports an "
        "error once, report the error and the sandbox path to the user "
        "instead of looping."
    )

    def __init__(self, artifact_root: Optional[Path] = None) -> None:
        self._store = None
        if artifact_root is not None:
            from app.artifacts import ArtifactStore

            self._store = ArtifactStore(Path(artifact_root))
        # conv_id -> {task_id: record} for background exec tasks; records
        # persist past the run so 'result'/'wait' can collect them later.
        self._exec_tasks: Dict[str, Dict[str, Dict[str, Any]]] = {}

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
                            "enum": [
                                "exec", "shell", "shell_start", "jobs",
                                "output", "write", "kill", "wait", "result",
                            ],
                            "description": "'exec' (default): run 'code' (background=true makes it async). Job actions: 'shell_start' launches a persistent process; 'jobs' lists jobs AND exec tasks; 'output'/'write'/'wait'/'kill' manage a job via job_id; 'result' collects a finished exec task (task-N); 'shell' runs a single shell command.",
                        },
                        "cmd": {
                            "type": "string",
                            "description": "Shell command for 'shell_start'/'shell' (e.g. 'python server.py').",
                        },
                        "job_id": {
                            "type": "string",
                            "description": "Job id ('job-N') for output/write/wait/kill, or exec task id ('task-N') for result/wait/kill.",
                        },
                        "background": {
                            "type": "boolean",
                            "description": "For 'exec': run asynchronously — returns 'task-N' immediately; the result auto-notifies on completion and is collected via action='result'.",
                        },
                        "tail": {
                            "type": "integer",
                            "description": "For 'output': last N lines per stream (default 100, max 1000).",
                        },
                        "stream": {
                            "type": "string",
                            "enum": ["stdout", "stderr", "both"],
                            "description": "For 'output': which stream to return (default 'both').",
                        },
                        "text": {
                            "type": "string",
                            "description": "For 'write': line sent to the job's stdin (newline appended).",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory for 'shell_start'/'shell' (default /tmp).",
                        },
                        "code": {
                            "type": "string",
                            "description": "The Python source code to execute (required for action='exec').",
                        },
                        "packages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of pip-installable packages to install before execution.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Max execution time in seconds (default 120, capped by the sandbox exec limit).",
                            "default": 120,
                        },
                        "output_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional absolute paths (max 10) of files created inside the sandbox to attach to the chat for the user to download.",
                        },
                    },
                    "required": [],
                },
            },
        }

    def call_conflict_key(self, args: Dict[str, Any]) -> Optional[str]:
        """Serialize operations on the same job inside a tool-call batch."""
        action = str(args.get("action") or "exec").strip().lower()
        if action in ("output", "write", "kill", "wait", "result"):
            job_id = args.get("job_id")
            if job_id:
                return f"job:{job_id}"
        # Foreground execs share one sandbox namespace + exec lock — two in
        # the same batch must run sequentially, not race into a 429.
        if action == "exec" and not args.get("background"):
            return "sandbox-exec"
        return None

    _JOB_ACTIONS = frozenset(
        {"shell", "shell_start", "jobs", "output", "write", "kill", "wait", "result"}
    )

    async def execute(
        self,
        code: Optional[str] = None,
        packages: Optional[List[str]] = None,
        timeout: int = 120,
        output_files: Optional[List[str]] = None,
        action: Optional[str] = None,
        cmd: Optional[str] = None,
        job_id: Optional[str] = None,
        tail: Optional[int] = None,
        stream: Optional[str] = None,
        text: Optional[str] = None,
        cwd: Optional[str] = None,
        background: Optional[bool] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        from agent.tools.docker_sandbox import (
            ConversationSandboxManager,
            EXEC_MAX_TIMEOUT,
            INSTALL_TIMEOUT,
        )

        action = (action or "exec").strip().lower()
        try:
            timeout = min(max(int(timeout or 120), 1), EXEC_MAX_TIMEOUT)
        except (TypeError, ValueError):
            timeout = 120

        # Validate before touching Docker — a malformed call must not boot
        # a sandbox container just to be rejected.
        if action in self._JOB_ACTIONS:
            if action in ("shell", "shell_start") and not (cmd or "").strip():
                return f"[Error: 'cmd' is required for action='{action}']"
            if action in ("output", "write", "kill", "wait", "result") and not (job_id or "").strip():
                return f"[Error: 'job_id' is required for action='{action}']"
            if action == "write" and text is None:
                return "[Error: 'text' is required for action='write']"
        elif action != "exec":
            return (
                f"[Error: unknown action '{action}' — use exec|shell|shell_start|"
                "jobs|output|write|kill|wait|result]"
            )
        elif not code:
            return "[Error: 'code' is required for action='exec']"

        conversation_id = (context or {}).get("conversation_id")
        if not conversation_id:
            return "[Error: code_execution requires a conversation_id, but none was provided.]"

        # Exec-task actions are local registry ops — dispatch BEFORE booting a
        # sandbox: reading a finished result must not require Docker.
        jid = (job_id or "").strip()
        if jid.startswith("task-"):
            if action in ("output", "write"):
                return f"[Error: '{action}' is for shell jobs only — use 'result' for exec tasks]"
            if action in ("wait", "kill", "result"):
                return await self._execute_task_action(
                    conversation_id, action, job_id=jid,
                    timeout=timeout, context=context,
                )
        elif action == "result":
            return "[Error: 'result' expects an exec task id ('task-N'), not a shell job]"

        manager = ConversationSandboxManager()
        sandbox_config = (context or {}).get("tool_config", {})
        agent_ws = (context or {}).get("agent_workspace_dir")

        if action == "exec" and background:
            return await self._start_exec_task(
                manager, conversation_id, sandbox_config=sandbox_config,
                agent_ws=agent_ws, code=code, packages=packages,
                timeout=timeout, output_files=output_files, context=context,
            )

        try:
            handle = await manager.get_or_create(
                conversation_id, sandbox_config=sandbox_config, agent_workspace_dir=agent_ws
            )
        except RuntimeError as exc:
            return f"[Error: {exc}]"

        # Surface a misconfigured/skipped workspace mount once per sandbox —
        # otherwise the agent just sees an empty /agent_ws with no clue why.
        mount_note = getattr(handle, "mount_warning", "")
        if not isinstance(mount_note, str):
            mount_note = ""
        if mount_note:
            handle.mount_warning = ""

        if action in self._JOB_ACTIONS:
            job_result = await self._execute_job_action(
                manager, handle, action,
                cmd=cmd, cwd=cwd, job_id=job_id, tail=tail, stream=stream,
                text=text, timeout=timeout, context=context,
                mount_note=mount_note,
            )
            if action == "jobs":
                job_result = str(job_result) + self._format_task_listing(conversation_id)
            return job_result

        # Install requested packages first
        if packages:
            try:
                # Client HTTP cap tracks the in-container pip timeout (+slack)
                # so a configured SANDBOX_INSTALL_TIMEOUT isn't undercut here.
                install_result = await manager.install_in(
                    handle, packages, timeout=min(timeout * 4, INSTALL_TIMEOUT + 30)
                )
            except Exception as exc:
                return f"[Error during package installation: {exc}]"
            if not install_result.get("ok"):
                err = install_result.get("error", "Unknown installation error")
                stderr = install_result.get("stderr", "")
                return f"[Package installation failed: {err}]\nstderr:\n{stderr}"

        # Execute code — only a short lock queue wait so a sandbox busy with
        # a background task fails fast instead of stalling this tool call.
        try:
            result = await manager.execute_in(
                handle, code, timeout=timeout, queue_wait=_BUSY_QUEUE_WAIT_S
            )
        except Exception as exc:
            return f"[Error during code execution: {exc}]"

        content, file_entries = await self._collect_output(
            result, output_files, manager, handle, conversation_id,
            context=context, mount_note=mount_note,
        )
        if file_entries:
            return {"ok": True, "content": content, "metadata": {"files": file_entries}}
        return content

    async def _collect_output(
        self,
        result: Dict[str, Any],
        output_files: Optional[List[str]],
        manager: Any,
        handle: Any,
        conversation_id: str,
        *,
        context: Optional[Dict[str, Any]],
        mount_note: str = "",
    ) -> tuple:
        """Format exec output + attach output_files; shared by sync and
        background task paths so both produce identical results."""
        parts: List[str] = []
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        error = result.get("error")

        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        if error:
            parts.append(f"error:\n{error}")

        # Collect requested sandbox files into the artifact store so the chat
        # renders them as downloadable attachment cards (metadata.files).
        file_entries: List[Dict[str, Any]] = []
        if output_files:
            for raw_path in list(output_files)[:_MAX_OUTPUT_FILES]:
                path = str(raw_path or "").strip()
                if not path:
                    continue
                if self._store is None:
                    parts.append(f"output_files: skipped '{path}' — artifact store unavailable.")
                    continue
                try:
                    read_result = await manager.read_file_in(
                        handle, path, max_bytes=_MAX_OUTPUT_FILE_BYTES
                    )
                except Exception as exc:
                    parts.append(f"output_files: failed to read '{path}': {exc}")
                    continue
                if not read_result.get("ok"):
                    parts.append(f"output_files: {read_result.get('error', 'read failed')}")
                    continue
                try:
                    model = str((context or {}).get("assistant_name") or "")
                    entry = await self._store.save(
                        conversation_id, model, read_result["name"], read_result["data"], "file"
                    )
                    file_entries.append(entry)
                    parts.append(f"output_files: attached '{entry['name']}' ({entry['size']} bytes)")
                except Exception as exc:
                    parts.append(f"output_files: failed to store '{path}': {exc}")

        if not parts:
            parts.append("(No output)")
        if mount_note:
            parts.insert(0, f"[Note: {mount_note}]")
        return "\n\n".join(parts), file_entries

    # -- background exec tasks ------------------------------------------------

    def _conv_tasks(self, conversation_id: str) -> Dict[str, Dict[str, Any]]:
        return self._exec_tasks.setdefault(conversation_id, {})

    def _evict_finished(self, tasks: Dict[str, Dict[str, Any]]) -> None:
        """Cap stored finished records so the registry stays bounded."""
        done_ids = [tid for tid, r in tasks.items() if r["status"] != "running"]
        if len(done_ids) > _MAX_FINISHED_TASKS:
            done_ids.sort(key=lambda t: tasks[t].get("finished_at", 0))
            for tid in done_ids[: len(done_ids) - _MAX_FINISHED_TASKS]:
                tasks.pop(tid, None)

    async def _start_exec_task(
        self,
        manager: Any,
        conversation_id: str,
        *,
        sandbox_config: Any,
        agent_ws: Any,
        code: Optional[str],
        packages: Optional[List[str]],
        timeout: int,
        output_files: Optional[List[str]],
        context: Optional[Dict[str, Any]],
    ) -> Any:
        tasks = self._conv_tasks(conversation_id)
        self._evict_finished(tasks)
        running = sum(1 for r in tasks.values() if r["status"] == "running")
        if running >= _MAX_BG_TASKS:
            return (
                f"[Error: {_MAX_BG_TASKS} background tasks already running — "
                "'wait'/'kill' one first]"
            )
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        hint = " ".join(str(code or "").split())[:60]
        record: Dict[str, Any] = {
            "task_id": task_id,
            "status": "running",
            "code_hint": hint,
            "started_at": time.time(),
            "finished_at": None,
            "content": "",
            "files": [],
            "task": None,
        }
        ctx = context or {}
        record["task"] = asyncio.ensure_future(
            self._run_exec_task(
                record, manager, conversation_id,
                sandbox_config=sandbox_config, agent_ws=agent_ws,
                code=code, packages=packages, timeout=timeout,
                output_files=output_files, model=str(ctx.get("assistant_name") or ""),
                queue=ctx.get("injected_input_queue"),
                event=ctx.get("input_event"),
            )
        )
        tasks[task_id] = record
        return {
            "ok": True,
            "content": (
                f"Started background task '{task_id}' — keep working; you will be "
                f"notified when it finishes. Collect it via action='result' + "
                f"job_id='{task_id}' ('wait' blocks, 'kill' cancels)."
            ),
            "metadata": {"exec_task": task_id},
        }

    async def _run_exec_task(
        self,
        record: Dict[str, Any],
        manager: Any,
        conversation_id: str,
        *,
        sandbox_config: Any,
        agent_ws: Any,
        code: Optional[str],
        packages: Optional[List[str]],
        timeout: int,
        output_files: Optional[List[str]],
        model: str,
        queue: Any,
        event: Any,
    ) -> None:
        """Boots the sandbox inside the task (cold-start does not block the
        caller), installs packages, runs the code, stores files, then nudges."""
        from agent.tools.docker_sandbox import EXEC_MAX_TIMEOUT, INSTALL_TIMEOUT

        try:
            try:
                handle = await manager.get_or_create(
                    conversation_id, sandbox_config=sandbox_config,
                    agent_workspace_dir=agent_ws,
                )
            except RuntimeError as exc:
                record["status"] = "error"
                record["content"] = f"[Error: {exc}]"
                return
            mount_note = getattr(handle, "mount_warning", "")
            if not isinstance(mount_note, str):
                mount_note = ""
            if mount_note:
                handle.mount_warning = ""
            if packages:
                install_result = await manager.install_in(
                    handle, packages, timeout=min(timeout * 4, INSTALL_TIMEOUT + 30),
                    queue_wait=EXEC_MAX_TIMEOUT,
                )
                if not install_result.get("ok"):
                    record["status"] = "error"
                    record["content"] = (
                        f"[Package installation failed: "
                        f"{install_result.get('error', 'Unknown installation error')}]\n"
                        f"stderr:\n{install_result.get('stderr', '')}"
                    )
                    return
            # Background tasks queue behind a running exec for the full exec
            # cap — they have no caller waiting on a prompt.
            result = await manager.execute_in(
                handle, code, timeout=timeout, queue_wait=EXEC_MAX_TIMEOUT
            )
            record["content"], record["files"] = await self._collect_output(
                result, output_files, manager, handle, conversation_id,
                context={"assistant_name": model}, mount_note=mount_note,
            )
            record["status"] = (
                "error" if (result.get("error") or not result.get("ok")) else "done"
            )
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 - surface any failure in the result
            logger.warning("exec task %s failed: %s", record["task_id"], exc)
            record["status"] = "error"
            record["content"] = f"[Error during background execution: {exc}]"
        finally:
            record["finished_at"] = time.time()
            if record["status"] != "running":
                self._nudge(record, queue, event)

    def _nudge(self, record: Dict[str, Any], queue: Any, event: Any) -> None:
        """Inject a short completion notice into the active run; the result
        itself stays in the registry until 'result' collects it."""
        if queue is None:
            return
        tail = (record.get("content") or "").strip()[-_NUDGE_TAIL_CHARS:]
        tail = f"\n{tail}" if tail else ""
        text = (
            f"[code_execution task '{record['task_id']}' finished — "
            f"{record['status']}]{tail}\n"
            f"→ call action='result' + job_id='{record['task_id']}' for the "
            "full output and attached files."
        )
        try:
            queue.put_nowait({"content": text, "text": text, "source": "code_execution"})
            if event is not None:
                event.set()
        except Exception:  # noqa: BLE001 - queue gone; result stays pollable
            pass

    async def _execute_task_action(
        self,
        conversation_id: str,
        action: str,
        *,
        job_id: str,
        timeout: int,
        context: Optional[Dict[str, Any]],
    ) -> Any:
        record = self._exec_tasks.get(conversation_id, {}).get(job_id)
        if record is None:
            return f"[Error: no exec task '{job_id}' in this conversation]"
        if action == "kill":
            task = record.get("task")
            if record["status"] == "running" and task is not None and not task.done():
                task.cancel()
                record["status"] = "cancelled"
                record["finished_at"] = time.time()
                return (
                    f"Task '{job_id}' cancelled — the sandbox-side process may "
                    "still finish up to its timeout."
                )
            return f"Task '{job_id}' already {record['status']}."
        if action == "wait":
            task = record.get("task")
            if record["status"] != "running" or task is None or task.done():
                return self._task_result_payload(record)
            input_event = (context or {}).get("input_event")
            waiters = {task}
            event_wait = None
            if input_event is not None:
                event_wait = asyncio.ensure_future(input_event.wait())
                waiters.add(event_wait)
            wait_cap = max(1, min(int(timeout or _JOB_WAIT_MAX), _JOB_WAIT_MAX))
            try:
                done, _ = await asyncio.wait(
                    waiters, timeout=wait_cap, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                if event_wait is not None and not event_wait.done():
                    event_wait.cancel()
            if task.done():
                # Prefer the real result when the task finished while input
                # also arrived — the queued input drains next iteration anyway.
                return self._task_result_payload(record)
            if event_wait is not None and event_wait in done:
                return (
                    f"[Woken early — new input arrived; task '{job_id}' may "
                    "still be running. Poll via action='result'.]"
                )
            return (
                f"Task '{job_id}' still running after {wait_cap}s — call "
                "'wait' again or fetch action='result'."
            )
        # result
        return self._task_result_payload(record)

    def _task_result_payload(self, record: Dict[str, Any]) -> Any:
        if record["status"] == "running":
            started = record.get("started_at", time.time())
            elapsed = round(time.time() - started)
            return (
                f"Task '{record['task_id']}' still running ({elapsed}s) — "
                "'wait' blocks until done or poll 'result' again."
            )
        content = record.get("content") or "(No output)"
        files = record.get("files") or []
        if record["status"] == "cancelled":
            content = f"Task '{record['task_id']}' was cancelled.\n\n{content}"
        if files:
            return {"ok": True, "content": content, "metadata": {"files": files}}
        return content

    def _format_task_listing(self, conversation_id: str) -> str:
        tasks = self._exec_tasks.get(conversation_id) or {}
        if not tasks:
            return ""
        lines = ["", "Exec tasks (background):"]
        for tid, rec in sorted(tasks.items(), key=lambda kv: kv[1].get("started_at", 0)):
            end = rec.get("finished_at") or time.time()
            elapsed = round(end - rec.get("started_at", end))
            lines.append(f'  {tid}  {rec["status"]}  {elapsed}s  "{rec.get("code_hint", "")}"')
        return "\n".join(lines)

    # -- persistent background jobs ------------------------------------------

    async def _execute_job_action(
        self,
        manager: Any,
        handle: Any,
        action: str,
        *,
        cmd: Optional[str],
        cwd: Optional[str],
        job_id: Optional[str],
        tail: Optional[int],
        stream: Optional[str],
        text: Optional[str],
        timeout: int,
        context: Optional[Dict[str, Any]],
        mount_note: str,
    ) -> str:
        server_action = {"shell_start": "start", "jobs": "list"}.get(action, action)
        payload: Dict[str, Any] = {"action": server_action}
        if action in ("shell", "shell_start"):
            payload["cmd"] = cmd
            if cwd:
                payload["cwd"] = cwd
            if action == "shell":
                payload["timeout"] = min(timeout, _JOB_WAIT_MAX)
        elif action in ("output", "write", "kill", "wait"):
            payload["job_id"] = job_id
            if action == "output":
                if tail is not None:
                    try:
                        payload["tail"] = int(tail)
                    except (TypeError, ValueError):
                        pass
                if stream:
                    payload["stream"] = stream
            elif action == "write":
                payload["text"] = text
            elif action == "wait":
                payload["timeout"] = min(timeout, _JOB_WAIT_MAX)

        # 'wait' can block server-side for the full timeout — client waits a
        # bit longer; a new user message wakes the call early (same steering
        # semantics as the wait/wait_subagent tools).
        client_timeout = (
            int(payload.get("timeout", 30)) + 30 if action in ("wait", "shell") else 60
        )
        job_req = asyncio.ensure_future(
            manager.jobs_in(handle, payload, timeout=client_timeout)
        )
        input_event = (context or {}).get("input_event") if action == "wait" else None
        if input_event is not None:
            input_task = asyncio.ensure_future(input_event.wait())
            done, _ = await asyncio.wait(
                {job_req, input_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if input_task in done:
                job_req.cancel()
                return (
                    f"[Woken early — new input arrived; job '{job_id}' may "
                    "still be running. Poll it with action='output'.]"
                )
            input_task.cancel()
        result = await job_req

        if not isinstance(result, dict):
            return f"[Error: unexpected sandbox job response: {result!r}]"
        if not result.get("ok"):
            return f"[Error: {result.get('error') or 'job request failed'}]"

        prefix = f"[Note: {mount_note}]\n" if mount_note else ""
        return prefix + self._format_job_result(action, job_id, result)

    @staticmethod
    def _format_job_result(action: str, job_id: Optional[str], res: Dict[str, Any]) -> str:
        if action == "shell_start":
            return (
                f"Started job '{res.get('job_id')}' (pid {res.get('pid')}). "
                "Poll with action='output' + job_id, block with action='wait', "
                "stop with action='kill'."
            )
        if action == "jobs":
            jobs = res.get("jobs") or []
            if not jobs:
                return "No jobs in this sandbox."
            lines = [f"Jobs ({res.get('active', 0)} running):"]
            for j in jobs:
                status = "running" if j.get("status") == "running" else f"exited(rc={j.get('rc')})"
                lines.append(
                    f"  {j.get('job_id')}  {status}  {j.get('runtime_s')}s"
                    f"  {j.get('cwd', '')}  \"{j.get('cmd', '')}\""
                )
            return "\n".join(lines)
        if action == "output":
            parts = [
                f"job {job_id} — {res.get('status')}"
                + (f" (rc={res.get('rc')})" if res.get("status") == "exited" else "")
                + f", {res.get('runtime_s')}s"
            ]
            if res.get("stdout") is not None:
                parts.append(f"stdout:\n{res.get('stdout') or '(empty)'}")
            if res.get("stderr") is not None:
                parts.append(f"stderr:\n{res.get('stderr') or '(empty)'}")
            return "\n".join(parts)
        if action == "write":
            return f"Sent {res.get('bytes', 0)} bytes to stdin of job '{job_id}'."
        if action == "kill":
            note = f" — {res['note']}" if res.get("note") else ""
            return f"Job '{job_id}' terminated (rc={res.get('rc')}){note}."
        if action == "wait":
            if res.get("status") == "exited":
                return f"Job '{job_id}' exited (rc={res.get('rc')})."
            return (
                f"Job '{job_id}' still running "
                f"({res.get('runtime_s')}s elapsed). Keep polling or call 'wait' again."
            )
        if action == "shell":
            return (
                f"rc={res.get('rc')}\n"
                f"stdout:\n{res.get('stdout') or '(empty)'}\n"
                f"stderr:\n{res.get('stderr') or '(empty)'}"
            )
        return str(res)
