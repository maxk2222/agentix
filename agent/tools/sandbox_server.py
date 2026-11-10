"""In-container HTTP executor for the code-execution sandbox.

This script is injected into a Docker sandbox container and runs a tiny
HTTP server that accepts Python code, executes it in a persistent global
namespace, and returns stdout/stderr.  It has zero external dependencies
beyond the Python standard library.
"""

from __future__ import annotations

import ctypes
import hmac
import importlib
import io
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Persistent globals so variables survive across /exec calls — shared, so
# execs must serialize (parallel execs would corrupt each other's stdout
# capture and interleave _globals mutations).
_globals: dict = {}
_EXEC_LOCK = threading.Lock()

# Shared secret provisioned by the sandbox manager at container start. The
# executor is reachable on a bridge network shared with every other sandbox
# (and via a published host port), so unauthenticated requests must never run
# code. Fail closed when the token is absent.
_EXPECTED_TOKEN = os.environ.get("SANDBOX_TOKEN", "")

# pip mirror/binary policy forwarded by the manager — captured here and
# applied to the pip subprocess env in _handle_install only, so the values
# are never left visible to exec'd code via os.environ.
_PIP_INDEX_URL = os.environ.get("SANDBOX_PIP_INDEX_URL", "")
_PIP_EXTRA_INDEX_URL = os.environ.get("SANDBOX_PIP_EXTRA_INDEX_URL", "")
_PIP_ONLY_BINARY = os.environ.get("SANDBOX_PIP_ONLY_BINARY", "").lower() in (
    "1", "true", "yes",
)

# Scrub secrets from the process environment — exec'd code shares os.environ
# and pip subprocesses inherit it, so the token (and base-image leftovers
# like GPG_KEY) must not linger there. The token survives only in
# _EXPECTED_TOKEN. Residual: /proc/self/environ still exposes the original
# env — the token's blast radius stays bounded to this container's own
# executor API, which the code inside already controls.
for _k in (
    "SANDBOX_TOKEN",
    "GPG_KEY",
    "SANDBOX_PIP_INDEX_URL",
    "SANDBOX_PIP_EXTRA_INDEX_URL",
    "SANDBOX_PIP_ONLY_BINARY",
):
    os.environ.pop(_k, None)

# Dedicated exec-capable tmpfs mount — the system site-packages are root-owned
# (and the whole fs is read-only in read_only mode), so `pip --user` cannot
# work reliably. `--target` into this dir works in every sandbox mode.
_PACKAGES_DIR = os.environ.get("SANDBOX_PACKAGES_DIR", "/var/sandbox-packages")
if _PACKAGES_DIR not in sys.path:
    sys.path.insert(0, _PACKAGES_DIR)

# When the sandbox runs on an internal (egress-less) bridge the manager sets
# this flag so /install can fail fast with a clear reason instead of leaving
# the agent to decode pip DNS errors.
_NETWORK_ISOLATED = os.environ.get("SANDBOX_NETWORK_ISOLATED", "").lower() in (
    "1", "true", "yes",
)

# pip subprocess cap — large trees (pandas, torch) legitimately exceed 120s.
_INSTALL_TIMEOUT = int(os.environ.get("SANDBOX_INSTALL_TIMEOUT", "240"))

# Hard cap for a single /exec — forwarded by the sandbox manager so app-side
# config reaches the in-container executor. Client HTTP waits add slack on top.
_EXEC_MAX_TIMEOUT = int(os.environ.get("SANDBOX_EXEC_MAX_TIMEOUT", "600"))

# Longest an /exec may wait for the shared-namespace lock before the 429.
# Background exec tasks pass a generous queue_wait so they queue behind the
# running exec instead of failing; interactive execs pass a short one so a
# busy sandbox fails fast instead of stalling the tool call.
_EXEC_QUEUE_WAIT_MAX = float(_EXEC_MAX_TIMEOUT)

# ---------------------------------------------------------------------------
# Persistent background jobs (tmux-style, Popen-based)
#
# The sandbox container lives for the whole conversation, so processes
# spawned here keep running between /exec calls. stdout/stderr are pumped
# into bounded in-memory scrollback deques (like tmux's history-limit) —
# deliberately NOT /tmp files: /tmp is a small noexec tmpfs shared with
# everything else.
# ---------------------------------------------------------------------------
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_JOB_IDS = itertools.count(1)
_JOB_MAX = 20
_JOB_SCROLLBACK_LINES = 4000
_JOB_WAIT_MAX = 120.0


def _job_env() -> dict:
    """Environment for job subprocesses.

    os.environ is already scrubbed at import (token/PIP_* are gone), so a
    plain copy stays clean — SANDBOX_TOKEN is popped again anyway (defense
    in depth). PYTHONPATH points at the pip --target dir so jobs can import
    packages installed via /install (sys.path tweaks do not propagate to
    child processes).
    """
    env = dict(os.environ)
    env.setdefault("HOME", "/tmp")
    env.pop("SANDBOX_TOKEN", None)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        _PACKAGES_DIR if not existing else f"{_PACKAGES_DIR}:{existing}"
    )
    return env


def _pump(stream, buf: "deque") -> None:
    """Copy a job stream into its scrollback deque until EOF.

    readline(size): an overlong line without \n is read in fragments — a
    single unbounded readline() could load megabytes into one deque slot.
    """
    try:
        for line in iter(lambda: stream.readline(4096), ""):
            buf.append(line)
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _terminate_job(proc) -> Optional[int]:
    """SIGTERM the job's whole process group, escalate to SIGKILL."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        return proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except Exception:
            pass
    try:
        return proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        return proc.poll()


def _close_job(job: dict) -> None:
    proc = job["popen"]
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if stream is not None:
                stream.close()
        except Exception:
            pass


def _job_status(job: dict) -> dict:
    rc = job["popen"].poll()
    return {
        "status": "running" if rc is None else "exited",
        "rc": rc,
        "runtime_s": round(time.time() - job["started_at"], 1),
    }


class _ExecTimeout(Exception):
    """Raised inside the exec thread by the watchdog timer."""


def _async_raise(tid: int) -> None:
    """Best-effort interrupt of a running thread (CPython only).

    Delivers ``_ExecTimeout`` at the next bytecode boundary, which stops
    pure-Python hangs (``while True`` loops, heavy computation). A thread
    blocked inside a C call (``time.sleep``, blocking I/O) finishes the
    call first — combined with the threaded HTTP server such a zombie can
    no longer starve other requests.
    """
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid), ctypes.py_object(_ExecTimeout)
    )
    if res > 1:  # more than one thread affected — undo and give up
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)


class _SandboxHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: ARG002
        # Silence default request logging to keep output clean
        pass

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def do_POST(self) -> None:  # noqa: N802
        provided = self.headers.get("X-Sandbox-Token", "")
        try:
            authorized = bool(_EXPECTED_TOKEN) and hmac.compare_digest(
                provided, _EXPECTED_TOKEN
            )
        except (TypeError, ValueError):
            authorized = False
        if not authorized:
            self._send_json(401, {"ok": False, "error": "Unauthorized"})
            return
        path = self.path
        body = self._read_body()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON"})
            return

        if path == "/exec":
            self._handle_exec(payload)
        elif path == "/install":
            self._handle_install(payload)
        elif path == "/files/read":
            self._handle_files_read(payload)
        elif path == "/jobs":
            self._handle_jobs(payload)
        else:
            self._send_json(404, {"ok": False, "error": f"Unknown endpoint: {path}"})

    def _handle_exec(self, payload: dict) -> None:
        code = payload.get("code", "")
        if not code:
            self._send_json(400, {"ok": False, "error": "Missing 'code' field"})
            return

        try:
            timeout = min(max(float(payload.get("timeout") or 30), 1.0), _EXEC_MAX_TIMEOUT)
        except (TypeError, ValueError):
            timeout = 30.0

        # Only one exec at a time (shared _globals + stdout capture). The
        # lock wait is the payload's queue_wait — interactive execs pass a
        # few seconds so a busy sandbox fails fast, background tasks pass
        # minutes so they queue behind the running exec. A hung predecessor
        # still cannot stall a request forever: at worst queue_wait + the
        # predecessor's own watchdog.
        try:
            raw_qw = payload.get("queue_wait")
            queue_wait = min(
                max(float(timeout if raw_qw is None else raw_qw), 0.0),
                _EXEC_QUEUE_WAIT_MAX,
            )
        except (TypeError, ValueError):
            queue_wait = timeout
        if not _EXEC_LOCK.acquire(timeout=queue_wait):
            self._send_json(429, {
                "ok": False,
                "error": (
                    "Another execution is still running in this sandbox "
                    f"(waited {queue_wait:g}s for it) — wait for it to finish "
                    "or retry; list/block via the tool's jobs/wait actions."
                ),
                "stdout": "",
                "stderr": "",
            })
            return

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        # Interrupt this handler thread if the exec overruns — a single
        # hung exec must not block the (threaded) server forever. The
        # `finished` flag disarms a late-firing watchdog so its exception
        # cannot land inside the finally and skip the lock release.
        finished = threading.Event()
        handler_tid = threading.get_ident()  # captured HERE — inside the timer
        # callback get_ident() would return the timer's own thread instead.
        watchdog = threading.Timer(
            timeout,
            lambda: None if finished.is_set() else _async_raise(handler_tid),
        )
        watchdog.daemon = True
        watchdog.start()

        error = None
        try:
            exec(code, _globals)  # noqa: S102
        except _ExecTimeout:
            error = f"Execution timed out after {timeout:g} seconds (SANDBOX_EXEC_MAX_TIMEOUT={_EXEC_MAX_TIMEOUT})"
        except Exception:
            error = traceback.format_exc()
        finally:
            # A watchdog callback that already passed the `finished` check
            # can deliver _ExecTimeout on ANY bytecode inside this cleanup —
            # retry the idempotent steps until they complete (at most one
            # pending interrupt can exist, so 3 attempts are ample).
            for _attempt in range(3):
                try:
                    finished.set()
                    watchdog.cancel()
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    try:
                        _EXEC_LOCK.release()
                    except RuntimeError:
                        pass  # released by an interrupted attempt
                    break
                except _ExecTimeout:
                    continue

        self._send_json(
            200 if error is None else 500,
            {
                "ok": error is None,
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue(),
                "error": error,
            },
        )

    def _handle_install(self, payload: dict) -> None:
        packages = payload.get("packages", [])
        if not packages:
            self._send_json(400, {"ok": False, "error": "Missing 'packages' field"})
            return
        # Package names only — flag-shaped args like '--index-url' would let
        # injected code pull packages from attacker-controlled indexes.
        if not isinstance(packages, list) or any(
            str(p).strip().startswith("-") or not str(p).strip() for p in packages
        ):
            self._send_json(400, {"ok": False, "error": "Invalid package name"})
            return

        if _NETWORK_ISOLATED:
            self._send_json(500, {
                "ok": False,
                "error": (
                    "Sandbox network is isolated — egress policy disables "
                    "internet access (egress_internet=off / "
                    "SANDBOX_NETWORK_INTERNAL). Package installation is "
                    "unavailable; write code using the standard library only."
                ),
                "stdout": "",
                "stderr": "",
            })
            return

        Path(_PACKAGES_DIR).mkdir(parents=True, exist_ok=True)
        # Serialize with /exec: an exec importing a package while pip is
        # still extracting it crashes on half-written files — installs and
        # execs share the interpreter state, so they share the lock too.
        # queue_wait mirrors /exec: short for interactive, long for tasks.
        try:
            raw_qw = payload.get("queue_wait")
            install_qw = min(
                max(float(10.0 if raw_qw is None else raw_qw), 0.0),
                _EXEC_QUEUE_WAIT_MAX,
            )
        except (TypeError, ValueError):
            install_qw = 10.0
        if not _EXEC_LOCK.acquire(timeout=install_qw):
            self._send_json(429, {
                "ok": False,
                "error": "Sandbox busy with a running execution — retry the install shortly.",
                "stdout": "",
                "stderr": "",
            })
            return
        try:
            env = dict(os.environ)
            env.setdefault("HOME", "/tmp")
            env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
            # Defense in depth — already scrubbed at import; keep the
            # installer's env free of the token even if a future env path
            # reintroduces it.
            env.pop("SANDBOX_TOKEN", None)
            installer = "pip"
            uv_bin = shutil.which("uv")
            if uv_bin:
                installer = "uv"
                cmd = [
                    uv_bin, "pip", "install",
                    "--python", sys.executable,
                    "--target", _PACKAGES_DIR,
                    "--no-cache",
                    # uv prefers wheels natively and has no --prefer-binary;
                    # source builds would fail under read_only's noexec /tmp
                    # anyway and stay blocked via --only-binary below.
                ]
            else:
                cmd = [
                    sys.executable, "-m", "pip", "install",
                    "--target", _PACKAGES_DIR,
                    "--no-cache-dir",
                    "--disable-pip-version-check",
                    "--prefer-binary",
                ]
            # Mirror/binary policy reaches the installer only — never
            # os.environ: pip via env vars, uv via equivalent CLI flags.
            if _PIP_INDEX_URL:
                if installer == "pip":
                    env["PIP_INDEX_URL"] = _PIP_INDEX_URL
                else:
                    cmd += ["--index-url", _PIP_INDEX_URL]
            if _PIP_EXTRA_INDEX_URL:
                if installer == "pip":
                    env["PIP_EXTRA_INDEX_URL"] = _PIP_EXTRA_INDEX_URL
                else:
                    cmd += ["--extra-index-url", _PIP_EXTRA_INDEX_URL]
            if _PIP_ONLY_BINARY:
                # Harder than --prefer-binary: no sdist setup.py ever executes
                # — supply-chain attacks via arbitrary build code become inert.
                cmd += ["--only-binary", ":all:"]
            cmd += list(packages)
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=_INSTALL_TIMEOUT,
                    check=False,
                    env=env,
                )
                # sys.path was primed at module load; make sure of it again
                # in case SANDBOX_PACKAGES_DIR pointed elsewhere at import.
                if _PACKAGES_DIR not in sys.path:
                    sys.path.insert(0, _PACKAGES_DIR)
                # The FileFinder for _PACKAGES_DIR caches the directory
                # listing on first use — without this, packages installed
                # after the first import attempt are invisible to `import`.
                importlib.invalidate_caches()
                self._send_json(
                    200 if result.returncode == 0 else 500,
                    {
                        "ok": result.returncode == 0,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "error": None if result.returncode == 0 else f"{installer} exited with code {result.returncode}",
                    },
                )
            except subprocess.TimeoutExpired:
                self._send_json(500, {"ok": False, "error": f"{installer} install timed out", "stdout": "", "stderr": ""})
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc), "stdout": "", "stderr": ""})
        finally:
            _EXEC_LOCK.release()


    def _handle_files_read(self, payload: dict) -> None:
        import base64

        raw = str(payload.get("path") or "").strip()
        if not raw:
            self._send_json(400, {"ok": False, "error": "Missing 'path' field"})
            return
        try:
            max_bytes = int(payload.get("max_bytes") or 32 * 1024 * 1024)
        except (TypeError, ValueError):
            max_bytes = 32 * 1024 * 1024
        path = Path(raw)
        try:
            if not path.is_file():
                self._send_json(404, {"ok": False, "error": f"Not a file: {raw}"})
                return
            size = path.stat().st_size
            if size > max_bytes:
                self._send_json(413, {"ok": False, "error": f"File too large ({size} bytes, max {max_bytes})"})
                return
            data = path.read_bytes()
        except OSError as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})
            return
        self._send_json(
            200,
            {
                "ok": True,
                "name": path.name,
                "size": size,
                "data_b64": base64.b64encode(data).decode("ascii"),
            },
        )

    # -- background jobs ----------------------------------------------------

    def _lookup_job(self, payload: dict):
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            self._send_json(400, {"ok": False, "error": "Missing 'job_id' field"})
            return None, None
        job = _JOBS.get(job_id)
        if job is None:
            self._send_json(404, {"ok": False, "error": f"Unknown job: {job_id}"})
            return None, None
        return job_id, job

    def _handle_jobs(self, payload: dict) -> None:
        action = str(payload.get("action") or "").strip().lower()

        if action == "start":
            cmd = str(payload.get("cmd") or "")
            if not cmd.strip():
                self._send_json(400, {"ok": False, "error": "Missing 'cmd' field"})
                return
            cwd = str(payload.get("cwd") or "/tmp").strip() or "/tmp"
            with _JOBS_LOCK:
                if len(_JOBS) >= _JOB_MAX:
                    # Evict finished jobs (oldest first) before refusing.
                    finished = sorted(
                        (jid for jid, j in _JOBS.items()
                         if j["popen"].poll() is not None),
                        key=lambda jid: _JOBS[jid]["started_at"],
                    )
                    for jid in finished[: len(_JOBS) - _JOB_MAX + 1]:
                        _close_job(_JOBS.pop(jid))
                if len(_JOBS) >= _JOB_MAX:
                    self._send_json(429, {
                        "ok": False,
                        "error": f"Job limit reached ({_JOB_MAX}) — kill or reap jobs first",
                    })
                    return
                job_id = f"j{next(_JOB_IDS)}"
            try:
                # start_new_session: own process group so kill can reap the
                # whole tree (a job spawning children must not leak them).
                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    cwd=cwd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    # errors="replace": binary garbage on a stream must not
                    # kill the pump thread — a dead pump means a full pipe
                    # and the job blocks forever on write.
                    errors="replace",
                    bufsize=1,
                    env=_job_env(),
                    start_new_session=True,
                )
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": f"Failed to start job: {exc}"})
                return
            job = {
                "popen": proc,
                "cmd": cmd,
                "cwd": cwd,
                "started_at": time.time(),
                "out": deque(maxlen=_JOB_SCROLLBACK_LINES),
                "err": deque(maxlen=_JOB_SCROLLBACK_LINES),
                "stdin_lock": threading.Lock(),
            }
            threading.Thread(target=_pump, args=(proc.stdout, job["out"]), daemon=True).start()
            threading.Thread(target=_pump, args=(proc.stderr, job["err"]), daemon=True).start()
            with _JOBS_LOCK:
                _JOBS[job_id] = job
            self._send_json(200, {"ok": True, "job_id": job_id, "pid": proc.pid})
            return

        if action == "list":
            jobs = []
            for jid, job in _JOBS.items():
                st = _job_status(job)
                jobs.append({
                    "job_id": jid,
                    "cmd": job["cmd"][:120],
                    "cwd": job["cwd"],
                    **st,
                })
            self._send_json(200, {
                "ok": True,
                "jobs": jobs,
                "active": sum(1 for j in jobs if j["status"] == "running"),
            })
            return

        if action == "output":
            job_id, job = self._lookup_job(payload)
            if job is None:
                return
            try:
                tail = max(1, min(int(payload.get("tail") or 100), 1000))
            except (TypeError, ValueError):
                tail = 100
            stream = str(payload.get("stream") or "both").lower()
            resp = {"ok": True, "job_id": job_id, **_job_status(job)}
            if stream in ("stdout", "both"):
                resp["stdout"] = "".join(list(job["out"])[-tail:])
            if stream in ("stderr", "both"):
                resp["stderr"] = "".join(list(job["err"])[-tail:])
            self._send_json(200, resp)
            return

        if action == "write":
            job_id, job = self._lookup_job(payload)
            if job is None:
                return
            text = payload.get("text")
            if text is None:
                self._send_json(400, {"ok": False, "error": "Missing 'text' field"})
                return
            proc = job["popen"]
            if proc.poll() is not None:
                self._send_json(409, {"ok": False, "error": f"Job {job_id} already exited (rc={proc.returncode})"})
                return
            data = str(text) + "\n"
            try:
                with job["stdin_lock"]:
                    proc.stdin.write(data)
                    proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as exc:
                self._send_json(409, {"ok": False, "error": f"stdin closed: {exc}"})
                return
            self._send_json(200, {"ok": True, "bytes": len(data)})
            return

        if action == "kill":
            job_id, job = self._lookup_job(payload)
            if job is None:
                return
            proc = job["popen"]
            if proc.poll() is not None:
                self._send_json(200, {"ok": True, "rc": proc.returncode, "note": "already exited"})
                return
            rc = _terminate_job(proc)
            try:
                proc.stdin.close()
            except Exception:
                pass
            self._send_json(200, {"ok": True, "rc": rc})
            return

        if action == "wait":
            job_id, job = self._lookup_job(payload)
            if job is None:
                return
            try:
                timeout = max(0.1, min(float(payload.get("timeout") or 30), _JOB_WAIT_MAX))
            except (TypeError, ValueError):
                timeout = 30.0
            try:
                job["popen"].wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._send_json(200, {"ok": True, **_job_status(job)})
                return
            self._send_json(200, {"ok": True, **_job_status(job)})
            return

        if action == "shell":
            cmd = str(payload.get("cmd") or "")
            if not cmd.strip():
                self._send_json(400, {"ok": False, "error": "Missing 'cmd' field"})
                return
            cwd = str(payload.get("cwd") or "/tmp").strip() or "/tmp"
            try:
                timeout = max(1.0, min(float(payload.get("timeout") or 30), _JOB_WAIT_MAX))
            except (TypeError, ValueError):
                timeout = 30.0
            try:
                result = subprocess.run(
                    cmd, shell=True, cwd=cwd, capture_output=True,
                    text=True, timeout=timeout, env=_job_env(),
                )
            except subprocess.TimeoutExpired:
                self._send_json(500, {"ok": False, "error": f"shell command timed out after {timeout:g}s"})
                return
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
                return
            # A completed command is a successful request — the exit code is
            # data, not an error (grep finding nothing must still surface).
            self._send_json(200, {
                "ok": True,
                "rc": result.returncode,
                "stdout": result.stdout[-100_000:],
                "stderr": result.stderr[-100_000:],
            })
            return

        self._send_json(400, {
            "ok": False,
            "error": f"Unknown jobs action '{action}' — use start|list|output|write|kill|wait|shell",
        })


def main() -> None:
    port = 8080
    # Threaded so one hung /exec cannot starve /install, /files or later
    # /exec calls — daemon threads let the container exit cleanly on TTL.
    server = ThreadingHTTPServer(("0.0.0.0", port), _SandboxHandler)
    server.daemon_threads = True
    print(f"Sandbox server listening on port {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
