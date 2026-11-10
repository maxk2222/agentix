"""Security/isolation tests for the Docker sandbox + conversation cleanup."""

from __future__ import annotations

import json
import os
import threading
import time
import types
import urllib.request
import urllib.error
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# sandbox_server token auth
# ---------------------------------------------------------------------------

def _start_sandbox_server(monkeypatch, token: str):
    """Run the real sandbox_server handler on an ephemeral localhost port."""
    import agent.tools.sandbox_server as srv

    monkeypatch.setattr(srv, "_EXPECTED_TOKEN", token)
    from http.server import HTTPServer

    server = HTTPServer(("127.0.0.1", 0), srv._SandboxHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def _post(port: int, path: str, payload: dict, token: str | None = None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token is not None:
        req.add_header("X-Sandbox-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_sandbox_server_rejects_missing_and_wrong_token(monkeypatch):
    server, port = _start_sandbox_server(monkeypatch, "secret-token")
    try:
        status, _ = _post(port, "/exec", {"code": "pass"})
        assert status == 401
        status, _ = _post(port, "/exec", {"code": "pass"}, token="wrong")
        assert status == 401
        status, _ = _post(port, "/files/read", {"path": "/etc/hostname"})
        assert status == 401
    finally:
        server.shutdown()


def test_sandbox_server_accepts_correct_token(monkeypatch):
    server, port = _start_sandbox_server(monkeypatch, "secret-token")
    try:
        status, body = _post(port, "/exec", {"code": "print('hi')"}, token="secret-token")
        assert status == 200 and body["ok"] and "hi" in body["stdout"]
    finally:
        server.shutdown()


def test_sandbox_server_fails_closed_without_env_token(monkeypatch):
    server, port = _start_sandbox_server(monkeypatch, "")
    try:
        status, _ = _post(port, "/exec", {"code": "pass"})
        assert status == 401
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# image tag / workspace root helpers
# ---------------------------------------------------------------------------

def test_sandbox_image_name_env_override(monkeypatch):
    import agent.tools.docker_sandbox as ds

    monkeypatch.setattr(ds, "SANDBOX_IMAGE", "custom/img:1")
    assert ds._sandbox_image_name() == "custom/img:1"


def test_sandbox_image_name_content_hash_changes(monkeypatch, tmp_path):
    import agent.tools.docker_sandbox as ds

    monkeypatch.setattr(ds, "SANDBOX_IMAGE", "")
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text("v1")
    f2.write_text("docker")
    monkeypatch.setattr(ds, "_SANDBOX_SERVER_SCRIPT", f1)
    monkeypatch.setattr(ds, "_SANDBOX_DOCKERFILE", f2)
    tag1 = ds._sandbox_image_name()
    f1.write_text("v2-changed")
    tag2 = ds._sandbox_image_name()
    assert tag1.startswith("agentix/sandbox:")
    assert tag1 != tag2


def test_workspace_dir_is_per_conversation(tmp_path, monkeypatch):
    import agent.tools.docker_sandbox as ds

    monkeypatch.setenv("SANDBOX_WORKSPACE_ROOT", str(tmp_path))
    a = ds._conversation_workspace_dir("conv-a")
    b = ds._conversation_workspace_dir("conv-b")
    assert a != b and a.parent == tmp_path and b.parent == tmp_path


def test_remove_conversation_workspace(tmp_path, monkeypatch):
    import agent.tools.docker_sandbox as ds

    monkeypatch.setenv("SANDBOX_WORKSPACE_ROOT", str(tmp_path))
    d = ds._conversation_workspace_dir("conv-x")
    d.mkdir(parents=True)
    (d / "f.txt").write_text("x")
    ds.remove_conversation_workspace("conv-x")
    assert not d.exists()


# ---------------------------------------------------------------------------
# _start_container wiring (token env/label + per-conv workspace mount)
# ---------------------------------------------------------------------------

class _FakeContainers:
    def __init__(self):
        self.run_kwargs = None
        self.run_calls = []

    def get(self, name):
        raise Exception("not found")

    def run(self, **kwargs):
        self.run_calls.append(kwargs)
        # run_kwargs keeps the FIRST call — the sandbox itself; later calls
        # (e.g. the netctl egress sidecar) are inspected via run_calls.
        if self.run_kwargs is None:
            self.run_kwargs = kwargs
        c = types.SimpleNamespace(
            labels=kwargs.get("labels") or {},
            attrs={},
            reload=lambda: None,
            ports={"8080/tcp": [{"HostPort": "49123"}]},
            stop=lambda timeout=5: None,
            remove=lambda force=True: None,
            logs=lambda tail=0: b"",
            wait=lambda timeout=None: {"StatusCode": 0},
        )
        return c

    def list(self, **kwargs):
        return []


class _FakeDockerClient:
    def __init__(self):
        self.containers = _FakeContainers()
        self.images = types.SimpleNamespace(get=lambda n: object())
        self.networks = types.SimpleNamespace(
            get=lambda n: types.SimpleNamespace(
                attrs={"IPAM": {"Config": [{"Gateway": "172.20.0.1"}]}},
                connect=lambda c, aliases=None: None,
            )
        )


@pytest.mark.asyncio
async def test_start_container_sets_token_and_conv_workspace(tmp_path, monkeypatch):
    import agent.tools.docker_sandbox as ds

    monkeypatch.setenv("SANDBOX_WORKSPACE_ROOT", str(tmp_path))
    client = _FakeDockerClient()
    manager = ds.ConversationSandboxManager()
    monkeypatch.setattr(manager, "_client", client)
    monkeypatch.setattr(manager, "_docker_available", True)
    monkeypatch.setattr(manager, "_ensure_network", lambda: _ait())
    monkeypatch.setattr(manager, "_wait_for_ready", lambda urls, timeout=30, token="": _ait(urls[0]))
    monkeypatch.setattr(manager, "_schedule_cleanup", lambda h: None)

    handle = await manager._start_container(
        "conv-ws", "agentix-sandbox-conv-ws-x", sandbox_config={"sandbox_mode": "writable_workspace"}
    )

    kw = client.containers.run_kwargs
    assert kw["environment"]["SANDBOX_TOKEN"] == handle.token and len(handle.token) >= 32
    assert kw["labels"]["agentix.sandbox.token"] == handle.token
    ws = tmp_path / "conv-ws"
    assert kw["volumes"] == {str(ws.resolve()): {"bind": "/workspace", "mode": "rw"}}
    assert ws.is_dir()


@pytest.mark.asyncio
async def test_start_container_read_only_has_no_workspace_mount(tmp_path, monkeypatch):
    import agent.tools.docker_sandbox as ds

    monkeypatch.setenv("SANDBOX_WORKSPACE_ROOT", str(tmp_path))
    client = _FakeDockerClient()
    manager = ds.ConversationSandboxManager()
    monkeypatch.setattr(manager, "_client", client)
    monkeypatch.setattr(manager, "_docker_available", True)
    monkeypatch.setattr(manager, "_ensure_network", lambda: _ait())
    monkeypatch.setattr(manager, "_wait_for_ready", lambda urls, timeout=30, token="": _ait(urls[0]))
    monkeypatch.setattr(manager, "_schedule_cleanup", lambda h: None)

    await manager._start_container(
        "conv-ro", "agentix-sandbox-conv-ro-x", sandbox_config={"sandbox_mode": "read_only"}
    )
    assert "volumes" not in client.containers.run_kwargs


async def _ait(v=None):
    return v


# ---------------------------------------------------------------------------
# cleanup_conversation_data
# ---------------------------------------------------------------------------

class _SpyStore:
    def __init__(self):
        self.deleted = []

    async def delete_conversation(self, cid):
        self.deleted.append(cid)


class _SpyConvStore(_SpyStore):
    async def delete(self, cid):
        self.deleted.append(cid)


class _SpyMemory:
    def __init__(self):
        self.deleted = []

    async def delete_memories_for_conversation(self, cid):
        self.deleted.append(cid)
        return 1


@pytest.mark.asyncio
async def test_cleanup_conversation_data_covers_all_stores(monkeypatch):
    from app import conversation_cleanup as cc

    conv, art, upl, mem = _SpyConvStore(), _SpyStore(), _SpyStore(), _SpyMemory()
    destroyed, ws_removed = [], []

    class _Engine:
        pass

    monkeypatch.setattr("app.stores.AgentMemoryStore", lambda e: mem)
    import agent.tools.docker_sandbox as ds

    class _Mgr:
        async def destroy(self, cid):
            destroyed.append(cid)

    monkeypatch.setattr(ds, "ConversationSandboxManager", _Mgr)
    monkeypatch.setattr(ds, "remove_conversation_workspace", lambda cid: ws_removed.append(cid))

    state = types.SimpleNamespace(
        _chat_conversations_store=conv,
        artifact_store=art,
        chat_attachment_store=upl,
        rag_application=types.SimpleNamespace(engine=_Engine()),
    )
    out = await cc.cleanup_conversation_data(state, "conv-9")

    assert conv.deleted == ["conv-9"] and art.deleted == ["conv-9"] and upl.deleted == ["conv-9"]
    assert mem.deleted == ["conv-9"]
    assert destroyed == ["conv-9"] and ws_removed == ["conv-9"]
    assert out["conversation_row"] and out["agent_memory"] == 1


@pytest.mark.asyncio
async def test_cleanup_conversation_data_isolates_failures(monkeypatch):
    from app import conversation_cleanup as cc

    class _BoomStore:
        async def delete_conversation(self, cid):
            raise RuntimeError("boom")

    upl = _SpyStore()
    monkeypatch.setattr(
        "agent.tools.docker_sandbox.ConversationSandboxManager",
        types.SimpleNamespace,
    )
    state = types.SimpleNamespace(
        _chat_conversations_store=None,
        artifact_store=_BoomStore(),
        chat_attachment_store=upl,
        rag_application=None,
    )
    # Must not raise despite the artifact store exploding.
    await cc.cleanup_conversation_data(state, "conv-err")
    assert upl.deleted == ["conv-err"]


# ---------------------------------------------------------------------------
# artifact download owner check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_artifact_access_denied_for_foreign_owned_conversation(monkeypatch):
    from app.routers import chat as chat_router
    from fastapi import HTTPException

    monkeypatch.setattr(
        chat_router,
        "_artifact_access",
        lambda r, cid, f: {"store": object(), "entry": {"model": "m"}},
    )
    monkeypatch.setattr(chat_router, "_model_access_allowed", lambda r, m: _ait(True))
    monkeypatch.setattr(
        chat_router, "_attachment_conv_owner_allowed", lambda r, cid: _ait(False)
    )
    req = types.SimpleNamespace(app=types.SimpleNamespace(state=types.SimpleNamespace()))

    with pytest.raises(HTTPException) as exc:
        await chat_router._check_artifact_access(req, "other-conv", "f.docx")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_artifact_access_allowed_for_rowless_conversation(monkeypatch):
    """Row-less conversations keep bearer semantics (unguessable uuid)."""
    from app.routers import chat as chat_router

    resolved = {"store": object(), "entry": {"model": "m"}}
    monkeypatch.setattr(chat_router, "_artifact_access", lambda r, cid, f: resolved)
    monkeypatch.setattr(chat_router, "_model_access_allowed", lambda r, m: _ait(True))
    monkeypatch.setattr(
        chat_router, "_attachment_conv_owner_allowed", lambda r, cid: _ait(True)
    )
    req = types.SimpleNamespace(app=types.SimpleNamespace(state=types.SimpleNamespace()))

    out = await chat_router._check_artifact_access(req, "no-row-conv", "f.docx")
    assert out is resolved


def test_sandbox_server_install_rejects_flag_injection(monkeypatch):
    """'--index-url' style args must never reach pip (supply-chain vector)."""
    server, port = _start_sandbox_server(monkeypatch, "tok")
    try:
        status, body = _post(
            port, "/install", {"packages": ["requests", "--index-url", "https://evil.example"]}, token="tok"
        )
        assert status == 400 and not body["ok"]
        status, _ = _post(port, "/install", {"packages": ["--user", "requests"]}, token="tok")
        assert status == 400
    finally:
        server.shutdown()


def test_sandbox_server_install_pip_command_has_target(monkeypatch, tmp_path):
    """Verify the install command uses --target into the exec-capable dir."""
    import agent.tools.sandbox_server as srv

    monkeypatch.setattr(srv, "_PACKAGES_DIR", str(tmp_path / "pkgs"))
    monkeypatch.setattr(srv.shutil, "which", lambda name: None)  # force pip path
    calls = []

    class _FakeResult:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(srv.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _FakeResult())
    monkeypatch.setattr(srv, "_EXPECTED_TOKEN", "tok")
    handler = srv._SandboxHandler.__new__(srv._SandboxHandler)
    sent = []
    handler._send_json = lambda status, data: sent.append((status, data))
    handler._handle_install({"packages": ["requests"]})

    cmd = calls[0]
    assert "--target" in cmd and "--no-cache-dir" in cmd
    assert cmd[cmd.index("--target") + 1] == srv._PACKAGES_DIR
    assert sent[0][0] == 200


def test_sandbox_server_install_uv_command_when_available(monkeypatch, tmp_path):
    """When the uv binary is present the install goes through `uv pip install`
    with equivalent flags (pip stays the fallback)."""
    import agent.tools.sandbox_server as srv

    monkeypatch.setattr(srv, "_PACKAGES_DIR", str(tmp_path / "pkgs"))
    monkeypatch.setattr(srv.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(srv, "_PIP_INDEX_URL", "https://mirror.example/simple")
    monkeypatch.setattr(srv, "_PIP_EXTRA_INDEX_URL", "https://extra.example/simple")
    calls = []

    class _FakeResult:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(srv.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _FakeResult())
    monkeypatch.setattr(srv, "_EXPECTED_TOKEN", "tok")
    handler = srv._SandboxHandler.__new__(srv._SandboxHandler)
    sent = []
    handler._send_json = lambda status, data: sent.append((status, data))
    handler._handle_install({"packages": ["requests"]})

    cmd = calls[0]
    assert cmd[0] == "/usr/bin/uv" and cmd[1:3] == ["pip", "install"]
    assert "--no-cache" in cmd and "--target" in cmd
    assert "--no-cache-dir" not in cmd and "--prefer-binary" not in cmd
    assert cmd[cmd.index("--index-url") + 1] == "https://mirror.example/simple"
    assert cmd[cmd.index("--extra-index-url") + 1] == "https://extra.example/simple"
    assert sent[0][0] == 200


def test_sandbox_server_install_prefers_binary_and_invalidates_caches(monkeypatch, tmp_path):
    """--prefer-binary avoids sdist builds (noexec /tmp), and the path
    importer cache must be invalidated so freshly installed packages import."""
    import agent.tools.sandbox_server as srv

    monkeypatch.setattr(srv, "_PACKAGES_DIR", str(tmp_path / "pkgs"))
    monkeypatch.setattr(srv, "_NETWORK_ISOLATED", False)
    monkeypatch.setattr(srv.shutil, "which", lambda name: None)  # force pip path
    calls = []
    invalidated = []

    class _FakeResult:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(srv.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _FakeResult())
    monkeypatch.setattr(srv.importlib, "invalidate_caches", lambda: invalidated.append(True))
    monkeypatch.setattr(srv, "_EXPECTED_TOKEN", "tok")
    handler = srv._SandboxHandler.__new__(srv._SandboxHandler)
    sent = []
    handler._send_json = lambda status, data: sent.append((status, data))
    handler._handle_install({"packages": ["requests"]})

    cmd = calls[0]
    assert "--prefer-binary" in cmd
    assert invalidated, "invalidate_caches must run after install"
    assert sent[0][0] == 200


def test_sandbox_server_install_fails_fast_when_network_isolated(monkeypatch):
    """On an egress-less bridge /install explains itself instead of surfacing
    cryptic pip DNS failures."""
    import agent.tools.sandbox_server as srv

    monkeypatch.setattr(srv, "_NETWORK_ISOLATED", True)
    monkeypatch.setattr(srv, "_EXPECTED_TOKEN", "tok")
    handler = srv._SandboxHandler.__new__(srv._SandboxHandler)
    sent = []
    handler._send_json = lambda status, data: sent.append((status, data))
    handler._handle_install({"packages": ["requests"]})
    status, body = sent[0]
    assert status == 500 and body["ok"] is False
    assert "isolated" in body["error"]


@pytest.mark.asyncio
async def test_start_container_attaches_only_dedicated_network(tmp_path, monkeypatch):
    """Without `network=` the container would ALSO sit on the default bridge —
    silently defeating SANDBOX_NETWORK_INTERNAL isolation."""
    import agent.tools.docker_sandbox as ds

    monkeypatch.setenv("SANDBOX_WORKSPACE_ROOT", str(tmp_path))
    client = _FakeDockerClient()
    manager = ds.ConversationSandboxManager()
    monkeypatch.setattr(manager, "_client", client)
    monkeypatch.setattr(manager, "_docker_available", True)
    monkeypatch.setattr(manager, "_ensure_network", lambda: _ait())
    monkeypatch.setattr(manager, "_wait_for_ready", lambda urls, timeout=30, token="": _ait(urls[0]))
    monkeypatch.setattr(manager, "_schedule_cleanup", lambda h: None)

    await manager._start_container(
        "conv-net", "agentix-sandbox-conv-net-x", sandbox_config={"sandbox_mode": "read_only"}
    )
    kw = client.containers.run_kwargs
    assert kw["network"] == ds.SANDBOX_NETWORK


@pytest.mark.asyncio
async def test_start_container_flags_isolated_network_and_tmpfs_mode(tmp_path, monkeypatch):
    """The isolated-network flag reaches the in-container executor and the
    packages tmpfs is writable for uid-1000 runs."""
    import agent.tools.docker_sandbox as ds

    monkeypatch.setenv("SANDBOX_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(ds, "SANDBOX_NETWORK_INTERNAL", True)
    client = _FakeDockerClient()
    manager = ds.ConversationSandboxManager()
    monkeypatch.setattr(manager, "_client", client)
    monkeypatch.setattr(manager, "_docker_available", True)
    monkeypatch.setattr(manager, "_ensure_network", lambda: _ait())
    monkeypatch.setattr(manager, "_wait_for_ready", lambda urls, timeout=30, token="": _ait(urls[0]))
    monkeypatch.setattr(manager, "_schedule_cleanup", lambda h: None)

    await manager._start_container(
        "conv-iso", "agentix-sandbox-conv-iso-x", sandbox_config={"sandbox_mode": "read_only"}
    )
    kw = client.containers.run_kwargs
    assert kw["environment"]["SANDBOX_NETWORK_ISOLATED"] == "1"
    # The in-container server honours the app-side timeout knobs only when
    # they are forwarded into the container env.
    assert kw["environment"]["SANDBOX_INSTALL_TIMEOUT"] == str(ds.INSTALL_TIMEOUT)
    assert kw["environment"]["SANDBOX_EXEC_MAX_TIMEOUT"] == str(ds.EXEC_MAX_TIMEOUT)
    # exec flag required: Docker tmpfs defaults to noexec — C-extension
    # wheels (numpy) install fine but fail at import without it.
    assert kw["tmpfs"]["/var/sandbox-packages"] == "size=512m,mode=1777,exec"


def test_sandbox_server_exec_enforces_timeout(monkeypatch):
    """A hung exec is interrupted server-side by the watchdog — with a
    single-threaded server it would otherwise starve every later request."""
    server, port = _start_sandbox_server(monkeypatch, "tok")
    try:
        status, body = _post(
            port, "/exec", {"code": "while True: pass", "timeout": 1}, token="tok"
        )
        assert status == 500
        assert body["ok"] is False
        assert "timed out" in body["error"]
        # The server must still answer afterwards — the watchdog freed it.
        status, body = _post(port, "/exec", {"code": "print('alive')"}, token="tok")
        assert status == 200 and body["ok"] and "alive" in body["stdout"]
    finally:
        server.shutdown()


def test_sandbox_server_exec_serializes_and_reports_busy(monkeypatch):
    """Concurrent execs are rejected fast instead of queueing behind a hung
    one — shared _globals and stdout capture make parallel execs unsafe."""
    import agent.tools.sandbox_server as srv

    server, port = _start_sandbox_server(monkeypatch, "tok")
    try:
        srv._EXEC_LOCK.acquire()
        try:
            status, body = _post(
                port, "/exec", {"code": "print('x')", "timeout": 1}, token="tok"
            )
            assert status == 429
            assert body["ok"] is False and "still running" in body["error"]
        finally:
            srv._EXEC_LOCK.release()
    finally:
        server.shutdown()


def test_sandbox_server_exec_max_timeout_env(monkeypatch):
    """SANDBOX_EXEC_MAX_TIMEOUT clamps the per-request timeout."""
    import agent.tools.sandbox_server as srv

    assert srv._EXEC_MAX_TIMEOUT == int(os.environ.get("SANDBOX_EXEC_MAX_TIMEOUT", "600"))


# ---------------------------------------------------------------------------
# sandbox_server /jobs — persistent background processes
# ---------------------------------------------------------------------------

def _fresh_jobs(monkeypatch):
    """Isolate the module-global job registry for one test."""
    import agent.tools.sandbox_server as srv
    import itertools

    monkeypatch.setattr(srv, "_JOBS", {})
    monkeypatch.setattr(srv, "_JOB_IDS", itertools.count(1))
    return srv


def test_sandbox_jobs_lifecycle(monkeypatch):
    srv = _fresh_jobs(monkeypatch)
    server, port = _start_sandbox_server(monkeypatch, "tok")
    try:
        status, body = _post(
            port, "/jobs", {"action": "start", "cmd": "printf 'boot\\n'; sleep 30"},
            token="tok",
        )
        assert status == 200 and body["ok"] and body["job_id"] == "j1"
        assert body["pid"] > 0

        status, body = _post(port, "/jobs", {"action": "list"}, token="tok")
        assert status == 200 and body["active"] == 1
        assert body["jobs"][0]["status"] == "running"

        # the pump thread needs a moment to deliver 'boot' into the deque
        out = ""
        for _ in range(40):
            _, body = _post(
                port, "/jobs", {"action": "output", "job_id": "j1"}, token="tok"
            )
            out = body.get("stdout", "")
            if "boot" in out:
                break
            time.sleep(0.05)
        assert "boot" in out

        status, body = _post(
            port, "/jobs", {"action": "kill", "job_id": "j1"}, token="tok"
        )
        assert status == 200 and body["ok"]
        status, body = _post(port, "/jobs", {"action": "list"}, token="tok")
        assert body["active"] == 0
        assert body["jobs"][0]["status"] == "exited"
    finally:
        server.shutdown()


def test_sandbox_jobs_wait_until_exit_and_timeout(monkeypatch):
    srv = _fresh_jobs(monkeypatch)
    server, port = _start_sandbox_server(monkeypatch, "tok")
    try:
        _, body = _post(
            port, "/jobs", {"action": "start", "cmd": "sleep 0.2"}, token="tok"
        )
        jid = body["job_id"]
        _, body = _post(
            port, "/jobs", {"action": "wait", "job_id": jid, "timeout": 10},
            token="tok",
        )
        assert body["status"] == "exited" and body["rc"] == 0

        _, body = _post(
            port, "/jobs", {"action": "start", "cmd": "sleep 30"}, token="tok"
        )
        jid2 = body["job_id"]
        _, body = _post(
            port, "/jobs", {"action": "wait", "job_id": jid2, "timeout": 0.5},
            token="tok",
        )
        assert body["status"] == "running"
        _post(port, "/jobs", {"action": "kill", "job_id": jid2}, token="tok")
    finally:
        server.shutdown()


def test_sandbox_jobs_write_stdin_and_shell(monkeypatch):
    srv = _fresh_jobs(monkeypatch)
    server, port = _start_sandbox_server(monkeypatch, "tok")
    try:
        _, body = _post(
            port, "/jobs", {"action": "start", "cmd": "cat"}, token="tok"
        )
        jid = body["job_id"]
        _, body = _post(
            port, "/jobs", {"action": "write", "job_id": jid, "text": "ping"},
            token="tok",
        )
        assert body["ok"] and body["bytes"] == 5
        out = ""
        for _ in range(40):
            _, body = _post(
                port, "/jobs", {"action": "output", "job_id": jid}, token="tok"
            )
            out = body.get("stdout", "")
            if "ping" in out:
                break
            time.sleep(0.05)
        assert "ping" in out
        _post(port, "/jobs", {"action": "kill", "job_id": jid}, token="tok")

        status, body = _post(
            port, "/jobs",
            {"action": "shell", "cmd": "echo one-shot", "timeout": 10},
            token="tok",
        )
        assert status == 200 and body["ok"]
        assert body["rc"] == 0 and "one-shot" in body["stdout"]

        # A non-zero exit is a completed command, not a request failure —
        # rc is data (e.g. grep with no match), stdout/stderr must survive.
        status, body = _post(
            port, "/jobs",
            {"action": "shell", "cmd": "echo oops >&2; exit 3", "timeout": 10},
            token="tok",
        )
        assert status == 200 and body["ok"]
        assert body["rc"] == 3 and "oops" in body["stderr"]
    finally:
        server.shutdown()


def test_sandbox_jobs_unknown_and_missing(monkeypatch):
    _fresh_jobs(monkeypatch)
    server, port = _start_sandbox_server(monkeypatch, "tok")
    try:
        status, body = _post(
            port, "/jobs", {"action": "output", "job_id": "j99"}, token="tok"
        )
        assert status == 404 and "Unknown job" in body["error"]
        status, body = _post(
            port, "/jobs", {"action": "start"}, token="tok"
        )
        assert status == 400 and "cmd" in body["error"]
        status, body = _post(
            port, "/jobs", {"action": "frob"}, token="tok"
        )
        assert status == 400 and "Unknown jobs action" in body["error"]
    finally:
        server.shutdown()


def test_sandbox_jobs_env_has_no_token_and_pip_path(monkeypatch):
    """Job env inherits the import-time scrub — no SANDBOX_TOKEN — and gets
    PYTHONPATH for /install'ed packages."""
    srv = _fresh_jobs(monkeypatch)
    env = srv._job_env()
    assert "SANDBOX_TOKEN" not in env
    assert srv._PACKAGES_DIR in env["PYTHONPATH"].split(":")


def test_sandbox_jobs_cap_evicts_finished(monkeypatch):
    srv = _fresh_jobs(monkeypatch)
    monkeypatch.setattr(srv, "_JOB_MAX", 3)
    server, port = _start_sandbox_server(monkeypatch, "tok")
    try:
        ids = []
        for i in range(3):
            _, body = _post(
                port, "/jobs", {"action": "start", "cmd": "true"}, token="tok"
            )
            ids.append(body["job_id"])
        # let them exit, then a 4th start must evict the oldest finished
        for jid in ids:
            srv._JOBS[jid]["popen"].wait(timeout=5)
        _, body = _post(
            port, "/jobs", {"action": "start", "cmd": "sleep 30"}, token="tok"
        )
        assert body["ok"] and body["job_id"] == "j4"
        _post(port, "/jobs", {"action": "kill", "job_id": "j4"}, token="tok")
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# /exec queue_wait semantics (real handler, real _EXEC_LOCK)
# ---------------------------------------------------------------------------

def test_packages_tmpfs_mount_allows_exec(tmp_path, monkeypatch):
    """C-extension wheels (numpy) must be loadable — the tmpfs needs 'exec'."""
    import agent.tools.docker_sandbox as ds

    monkeypatch.setenv("SANDBOX_WORKSPACE_ROOT", str(tmp_path))
    client = _FakeDockerClient()
    manager = ds.ConversationSandboxManager()
    monkeypatch.setattr(manager, "_client", client)
    monkeypatch.setattr(manager, "_docker_available", True)
    monkeypatch.setattr(manager, "_ensure_network", lambda: _ait())
    monkeypatch.setattr(manager, "_wait_for_ready", lambda urls, timeout=30, token="": _ait(urls[0]))
    monkeypatch.setattr(manager, "_schedule_cleanup", lambda h: None)

    import asyncio
    asyncio.run(
        manager._start_container(
            "conv-pkgs", "agentix-sandbox-conv-pkgs-x",
            sandbox_config={"sandbox_mode": "read_only"},
        )
    )
    tmpfs = client.containers.run_kwargs["tmpfs"]
    assert "/var/sandbox-packages" in tmpfs
    assert "exec" in tmpfs["/var/sandbox-packages"]
    assert "noexec" not in tmpfs["/var/sandbox-packages"]


def test_exec_queue_wait_429s_fast_on_busy_lock(monkeypatch):
    """queue_wait bounds the lock wait — a busy sandbox 429s quickly."""
    import agent.tools.sandbox_server as srv

    server, port = _start_sandbox_server(monkeypatch, "tok")
    try:
        srv._EXEC_LOCK.acquire()
        try:
            status, data = _post(
                port, "/exec",
                {"code": "pass", "queue_wait": 0.2}, token="tok",
            )
            assert status == 429
            assert "still running" in data["error"]
        finally:
            srv._EXEC_LOCK.release()
        # Once free, the same request executes normally.
        status, data = _post(
            port, "/exec", {"code": "print('free')"}, token="tok",
        )
        assert status == 200 and "free" in data["stdout"]
    finally:
        server.shutdown()


def test_exec_queue_wait_blocks_until_lock_released(monkeypatch):
    """A queued exec runs once the predecessor frees the lock."""
    import agent.tools.sandbox_server as srv

    server, port = _start_sandbox_server(monkeypatch, "tok")
    try:
        srv._EXEC_LOCK.acquire()
        try:
            released = []
            import threading as _t

            def _release():
                released.append(
                    _post(port, "/exec",
                          {"code": "print('queued')", "queue_wait": 3},
                          token="tok")
                )

            t = _t.Thread(target=_release, daemon=True)
            t.start()
            time.sleep(0.2)  # let the request reach the lock wait
            srv._EXEC_LOCK.release()
            t.join(timeout=5)
            assert released and released[0][0] == 200
            assert "queued" in released[0][1]["stdout"]
        finally:
            if srv._EXEC_LOCK.locked():
                srv._EXEC_LOCK.release()
    finally:
        server.shutdown()
