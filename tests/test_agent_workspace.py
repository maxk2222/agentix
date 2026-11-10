"""Tests for the per-assistant agent workspace store (app/agent_workspace.py)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.agent_workspace import AgentWorkspaceStore, WorkspaceError


@pytest.fixture
def store(tmp_path):
    return AgentWorkspaceStore(tmp_path / "agent_workspaces")


# -- scope dirs ---------------------------------------------------------------


def test_scope_dir_assistant_shared(store):
    d = store.scope_dir("My Assistant", "assistant")
    assert d == store.root / "My_Assistant" / "shared"


def test_scope_dir_user(store):
    d = store.scope_dir("a", "user", user_id="chat-user-1")
    assert d.name == "u_chat-user-1"
    # no user identity -> falls back to conversation scope
    d2 = store.scope_dir("a", "user", conversation_id="conv1", user_id=None)
    assert d2.name == "c_conv1"


def test_scope_dir_conversation_sanitized(store):
    d = store.scope_dir("a", "conversation", conversation_id="c/../evil")
    assert ".." not in str(d)
    assert d.parent.name == "a"


def test_scope_isolation(store):
    assert store.scope_dir("a", "user", user_id="u1") != store.scope_dir("a", "user", user_id="u2")
    assert store.scope_dir("a", "conversation", "c1") != store.scope_dir("a", "conversation", "c2")
    assert store.scope_dir("a1", "assistant") != store.scope_dir("a2", "assistant")


# -- safe resolution ------------------------------------------------------------


def test_resolve_rejects_traversal(store):
    d = store.scope_dir("a", "assistant")
    for bad in ("../x", "a/../../x", "/etc/passwd", "..\\x", "a\x00b"):
        with pytest.raises(WorkspaceError):
            store.resolve(d, bad)


def test_resolve_rejects_symlink_escape(store, tmp_path):
    d = store.scope_dir("a", "assistant")
    d.mkdir(parents=True)
    outside = tmp_path / "secret.txt"
    outside.write_text("x")
    link = d / "link"
    link.symlink_to(outside)
    with pytest.raises(WorkspaceError):
        store.resolve(d, "link")
    sub = d / "sub"
    sub.mkdir()
    (sub / "inner").symlink_to(tmp_path)
    with pytest.raises(WorkspaceError):
        store.resolve(d, "sub/inner/file.txt")


def test_resolve_ok(store):
    d = store.scope_dir("a", "assistant")
    p = store.resolve(d, "notes/todo.md")
    assert str(p).startswith(str(d.resolve()))
    assert p.name == "todo.md"


# -- operations -----------------------------------------------------------------


def test_write_read_list(store):
    d = store.scope_dir("a", "assistant")
    entry = store.write(d, "notes/hi.txt", b"hello")
    assert entry["path"] == "notes/hi.txt"
    assert store.read(d, "notes/hi.txt") == b"hello"
    listing = store.list_entries(d, "notes")
    assert [e["name"] for e in listing] == ["hi.txt"]


def test_write_is_atomic(store):
    d = store.scope_dir("a", "assistant")
    store.write(d, "f.txt", b"v1")
    store.write(d, "f.txt", b"v2")
    assert store.read(d, "f.txt") == b"v2"
    # no temp leftovers
    assert [e["name"] for e in store.list_entries(d)] == ["f.txt"]


def test_append(store):
    d = store.scope_dir("a", "assistant")
    store.write(d, "f.txt", b"a")
    store.write(d, "f.txt", b"b", append=True)
    assert store.read(d, "f.txt") == b"ab"


def test_quota_file_size(store):
    d = store.scope_dir("a", "assistant")
    cfg = {"max_files": 1}
    store.write(d, "a.txt", b"x", cfg)
    with pytest.raises(WorkspaceError):
        store.write(d, "b.txt", b"x", cfg)


def test_quota_total(store):
    d = store.scope_dir("a", "assistant")
    cfg = {"max_total_mb": 1, "max_file_mb": 50}
    big = b"x" * (1024 * 1024)
    store.write(d, "a.bin", big, cfg)
    with pytest.raises(WorkspaceError):
        store.write(d, "b.bin", b"y", cfg)


def test_delete_and_move(store):
    d = store.scope_dir("a", "assistant")
    store.write(d, "a/b.txt", b"x")
    store.move(d, "a/b.txt", "c/d.txt")
    assert store.read(d, "c/d.txt") == b"x"
    with pytest.raises(WorkspaceError):
        store.read(d, "a/b.txt")
    store.delete(d, "c")
    with pytest.raises(WorkspaceError):
        store.read(d, "c/d.txt")
    assert [e["name"] for e in store.list_entries(d)] == ["a"]


def test_usage_counts_files(store):
    d = store.scope_dir("a", "assistant")
    store.write(d, "x.txt", b"12345")
    store.write(d, "y.txt", b"123")
    u = store.usage(d)
    assert u == {"files": 2, "bytes": 8}


def test_usage_counts_hidden_dirs_recursively(store):
    d = store.scope_dir("a", "assistant")
    store.write(d, ".pkgs/lib/a.py", b"x")
    store.write(d, ".pkgs/lib/b.py", b"x")
    store.write(d, "top.txt", b"x")
    assert store.usage(d)["files"] == 3


def test_file_breakdown_groups_by_top_level(store):
    d = store.scope_dir("a", "assistant")
    store.write(d, ".pkgs/a/f1.txt", b"x")
    store.write(d, ".pkgs/b/f2.txt", b"x")
    store.write(d, ".pkgs/b/f3.txt", b"x")
    store.write(d, "reports/r1.txt", b"x")
    store.write(d, "loose.txt", b"x")
    bd = store.file_breakdown(d)
    assert bd[0] == {"name": ".pkgs", "files": 3}
    names = {t["name"] for t in bd}
    assert "reports" in names and "(root)" in names
    assert sum(t["files"] for t in bd) == 5


def test_file_breakdown_skips_symlinks(store):
    d = store.scope_dir("a", "assistant")
    store.write(d, "real.txt", b"x")
    link = d / "link.txt"
    try:
        link.symlink_to(d / "real.txt")
    except OSError:
        pytest.skip("symlinks unsupported")
    assert store.file_breakdown(d) == [{"name": "(root)", "files": 1}]


def test_quota_error_lists_top_dirs(store):
    d = store.scope_dir("a", "assistant")
    store.write(d, ".pkgs/a.txt", b"x")
    store.write(d, ".pkgs/b.txt", b"x")
    cfg = {"max_files": 2}
    with pytest.raises(WorkspaceError) as exc:
        store.write(d, "new.txt", b"x", cfg)
    msg = str(exc.value)
    assert "file limit (2)" in msg
    assert ".pkgs" in msg and "2" in msg


def test_list_empty_and_missing(store):
    d = store.scope_dir("a", "assistant")
    assert store.list_entries(d) == []


def test_delete_assistant_tree(store):
    store.write(store.scope_dir("a", "assistant"), "f.txt", b"x")
    store.write(store.scope_dir("a", "conversation", "c1"), "g.txt", b"y")
    store.delete_assistant("a")
    assert not (store.root / "a").exists()


def test_orphan_assistant_dirs(store):
    store.write(store.scope_dir("dead", "assistant"), "f.txt", b"x")
    store.write(store.scope_dir("alive", "assistant"), "f.txt", b"x")
    orphans = store.orphan_assistant_dirs({"alive"})
    assert [p.name for p in orphans] == ["dead"]


# -- agent_workspace tool: chat-upload import ---------------------------------

@pytest.fixture
def uploads_store(tmp_path):
    from app.chat_attachments import ChatAttachmentStore
    return ChatAttachmentStore(tmp_path / "chat_uploads")


@pytest.fixture
def workspace_scope(tmp_path):
    scope = tmp_path / "ws_scope"
    scope.mkdir(parents=True)
    return scope


@pytest.fixture
def ws_tool(store, uploads_store):
    from agent.tools.builtin_tools.agent_workspace import AgentWorkspaceTool
    return AgentWorkspaceTool(store=store, attachment_store=uploads_store)


def _ws_ctx(scope_dir: Path, conv: str = "conv-1") -> dict:
    return {
        "agent_workspace_dir": str(scope_dir),
        "agent_workspace_cfg": {},
        "agent_workspace_conv": conv,
        "conversation_id": conv,
    }


@pytest.mark.asyncio
async def test_tool_list_uploads_empty(ws_tool, workspace_scope):
    result = await ws_tool.execute("list_uploads", context=_ws_ctx(workspace_scope))
    assert "no uploaded files" in result


@pytest.mark.asyncio
async def test_tool_import_upload_by_name(ws_tool, uploads_store, workspace_scope):
    await uploads_store.save("conv-1", "data.csv", b"a,b\n1,2\n")
    listing = await ws_tool.execute("list_uploads", context=_ws_ctx(workspace_scope))
    assert "data.csv" in listing

    result = await ws_tool.execute(
        "import_upload", attachment="data.csv", context=_ws_ctx(workspace_scope)
    )
    assert "uploads/data.csv" in result
    assert (workspace_scope / "uploads" / "data.csv").read_bytes() == b"a,b\n1,2\n"


@pytest.mark.asyncio
async def test_tool_import_upload_by_id_and_dest(ws_tool, uploads_store, workspace_scope):
    meta = await uploads_store.save("conv-1", "image.png", b"\x89PNG")
    result = await ws_tool.execute(
        "import_upload",
        attachment=meta["id"],
        to_path="inputs/pic.png",
        context=_ws_ctx(workspace_scope),
    )
    assert "inputs/pic.png" in result
    assert (workspace_scope / "inputs" / "pic.png").read_bytes() == b"\x89PNG"


@pytest.mark.asyncio
async def test_tool_import_upload_subagent_uses_root_conv(ws_tool, uploads_store, workspace_scope):
    """Subagent contexts carry the root conv id so uploads resolve correctly."""
    await uploads_store.save("conv-1", "root.txt", b"root-file")
    ctx = {
        "agent_workspace_dir": str(workspace_scope),
        "agent_workspace_cfg": {},
        "agent_workspace_conv": "conv-1",        # root conv
        "conversation_id": "conv-1.sa_1",        # derived subagent conv
    }
    result = await ws_tool.execute("import_upload", attachment="root.txt", context=ctx)
    assert "uploads/root.txt" in result


@pytest.mark.asyncio
async def test_tool_import_upload_wrong_conv(ws_tool, uploads_store, workspace_scope):
    await uploads_store.save("conv-A", "secret.txt", b"x")
    result = await ws_tool.execute("list_uploads", context=_ws_ctx(workspace_scope, conv="conv-B"))
    assert "no uploaded files" in result


@pytest.mark.asyncio
async def test_tool_import_upload_quota_enforced(ws_tool, uploads_store, workspace_scope):
    await uploads_store.save("conv-1", "one.bin", b"x" * 64)
    await uploads_store.save("conv-1", "two.bin", b"y" * 64)
    ctx = _ws_ctx(workspace_scope)
    ctx["agent_workspace_cfg"] = {"max_files": 1}
    result = await ws_tool.execute("import_upload", attachment="one.bin", context=ctx)
    assert "Error" not in result
    result = await ws_tool.execute("import_upload", attachment="two.bin", context=ctx)
    assert "file limit" in result


# -- agent_workspace tool: document view/inspect -------------------------------

@pytest.fixture
def pdf_in_scope(workspace_scope):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}")
    pdf = workspace_scope / "report.pdf"
    doc.save(str(pdf))
    doc.close()
    return pdf


def _vision_ctx(scope_dir: Path) -> dict:
    ctx = _ws_ctx(scope_dir)
    ctx["agent_workspace_cfg"] = {"vision_read": True}
    return ctx


@pytest.mark.asyncio
async def test_tool_view_gated_off(ws_tool, workspace_scope, pdf_in_scope):
    result = await ws_tool.execute("view", path="report.pdf", context=_ws_ctx(workspace_scope))
    assert "vision_read" in result


@pytest.mark.asyncio
async def test_tool_view_pdf_returns_images(ws_tool, workspace_scope, pdf_in_scope, monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_RENDER_CACHE", str(tmp_path / "cache"))
    result = await ws_tool.execute(
        "view", path="report.pdf", pages="1-2", context=_vision_ctx(workspace_scope)
    )
    assert isinstance(result, dict)
    assert "Rendered 2" in result["content"]
    assert len(result["images"]) == 2
    assert all(u.startswith("data:image/png;base64,") for u in result["images"])


@pytest.mark.asyncio
async def test_tool_view_respects_containment(ws_tool, workspace_scope):
    result = await ws_tool.execute(
        "view", path="../escape.pdf", context=_vision_ctx(workspace_scope)
    )
    assert "Error" in result


@pytest.mark.asyncio
async def test_tool_inspect_pdf(ws_tool, workspace_scope, pdf_in_scope):
    result = await ws_tool.execute("inspect", path="report.pdf", context=_ws_ctx(workspace_scope))
    assert "PDF: 3 page(s)" in result
    assert "page 1:" in result


@pytest.mark.asyncio
async def test_tool_inspect_extract_media(ws_tool, workspace_scope, tmp_path):
    # build a pptx with an embedded image
    pptx = pytest.importorskip("pptx")
    from PIL import Image
    import io as _io

    img = Image.new("RGB", (32, 32), (9, 9, 9))
    buf = _io.BytesIO()
    img.save(buf, "PNG")
    pic = tmp_path / "p.png"
    pic.write_bytes(buf.getvalue())
    prs = pptx.Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.shapes.add_picture(str(pic), pptx.util.Inches(1), pptx.util.Inches(1))
    prs.save(str(workspace_scope / "deck.pptx"))

    result = await ws_tool.execute(
        "inspect", path="deck.pptx", extract_media=True, context=_ws_ctx(workspace_scope)
    )
    assert "PPTX: 1 slide" in result
    assert "Extracted 1 media file" in result
    extracted = list((workspace_scope / "extracted" / "deck").glob("*.png"))
    assert extracted, "media should land under extracted/<stem>/"


@pytest.mark.asyncio
async def test_tool_read_binary_hint(ws_tool, workspace_scope, pdf_in_scope):
    result = await ws_tool.execute("read", path="report.pdf", context=_ws_ctx(workspace_scope))
    assert "inspect" in result and "view" in result


@pytest.mark.asyncio
async def test_tool_view_missing_file(ws_tool, workspace_scope):
    result = await ws_tool.execute(
        "view", path="missing.pdf", context=_vision_ctx(workspace_scope)
    )
    assert "Error" in result


# -- agent_workspace tool: edit action -------------------------------------------


@pytest.mark.asyncio
async def test_tool_edit_sequential_batch(ws_tool, workspace_scope):
    ctx = _ws_ctx(workspace_scope)
    await ws_tool.execute("write", path="notes.md", content="a\nb\nc\n", context=ctx)
    result = await ws_tool.execute(
        "edit", path="notes.md",
        edits=[
            {"operation": "replace", "old_text": "b", "content": "B"},
            {"operation": "insert", "line": -1, "content": "d\n"},
            {"operation": "remove", "text_snippet": "a\n"},
        ],
        context=ctx,
    )
    assert "applied 3 edit(s) (replace, insert, remove)" in result
    assert (workspace_scope / "notes.md").read_text() == "B\nc\nd\n"


@pytest.mark.asyncio
async def test_tool_edit_evolving_line_state(ws_tool, workspace_scope):
    """Later ops see the state left by earlier ops in the same batch."""
    ctx = _ws_ctx(workspace_scope)
    await ws_tool.execute("write", path="f.txt", content="1\n2\n3\n", context=ctx)
    await ws_tool.execute(
        "edit", path="f.txt",
        edits=[
            {"operation": "remove", "lines_start": 1, "lines_end": 1},
            {"operation": "insert", "line": 1, "content": "X\n"},
        ],
        context=ctx,
    )
    assert (workspace_scope / "f.txt").read_text() == "X\n2\n3\n"


@pytest.mark.asyncio
async def test_tool_edit_failure_saves_nothing(ws_tool, workspace_scope):
    ctx = _ws_ctx(workspace_scope)
    await ws_tool.execute("write", path="f.txt", content="a\nb\n", context=ctx)
    result = await ws_tool.execute(
        "edit", path="f.txt",
        edits=[
            {"operation": "replace", "old_text": "a", "content": "A"},
            {"operation": "replace", "old_text": "MISSING", "content": "x"},
        ],
        context=ctx,
    )
    assert "Error" in result
    assert "edit #1" in result and "no changes saved" in result
    assert (workspace_scope / "f.txt").read_text() == "a\nb\n"


@pytest.mark.asyncio
async def test_tool_edit_binary_rejected(ws_tool, workspace_scope, pdf_in_scope):
    result = await ws_tool.execute(
        "edit", path="report.pdf",
        edits=[{"operation": "replace", "old_text": "a", "content": "b"}],
        context=_ws_ctx(workspace_scope),
    )
    assert "binary" in result


@pytest.mark.asyncio
async def test_tool_edit_missing_file(ws_tool, workspace_scope):
    result = await ws_tool.execute(
        "edit", path="nope.txt",
        edits=[{"operation": "remove", "text_snippet": "x"}],
        context=_ws_ctx(workspace_scope),
    )
    assert "Error" in result


@pytest.mark.asyncio
async def test_tool_edit_quota_enforced(ws_tool, workspace_scope, monkeypatch):
    """Edits route through store.write → the file-size quota still applies."""
    import app.agent_workspace as aw
    monkeypatch.setattr(aw, "DEFAULT_MAX_FILE_BYTES", 4)
    ctx = _ws_ctx(workspace_scope)
    await ws_tool.execute("write", path="f.txt", content="x", context=ctx)
    result = await ws_tool.execute(
        "edit", path="f.txt",
        edits=[{"operation": "insert", "line": -1, "content": "more"}],
        context=ctx,
    )
    assert "Error" in result
    assert (workspace_scope / "f.txt").read_text() == "x"  # rolled back


@pytest.mark.asyncio
async def test_tool_edit_rejects_non_utf8(ws_tool, workspace_scope):
    """A latin-1 file must fail cleanly — decoding with errors='replace'
    would silently write U+FFFD back and corrupt the file."""
    raw = b"ligne accentu\xe9e\n"  # latin-1, invalid UTF-8
    (workspace_scope / "latin.md").write_bytes(raw)
    result = await ws_tool.execute(
        "edit", path="latin.md",
        edits=[{"operation": "replace", "old_text": "x", "content": "y"}],
        context=_ws_ctx(workspace_scope),
    )
    assert "UTF-8" in result
    assert (workspace_scope / "latin.md").read_bytes() == raw  # untouched


@pytest.mark.asyncio
async def test_tool_read_line_numbers(ws_tool, workspace_scope):
    ctx = _ws_ctx(workspace_scope)
    await ws_tool.execute("write", path="f.txt", content="l1\nl2\n", context=ctx)
    result = await ws_tool.execute("read", path="f.txt", line_numbers=True, context=ctx)
    assert "1│ l1" in result and "2│ l2" in result


# -- daemon path derivation / mount safety --------------------------------------


def test_parse_mountinfo_filters_virtual_fs():
    from app.agent_workspace import _parse_mountinfo

    text = (
        "26 22 0:22 / / rw,relatime - overlay overlay rw,lowerdir=/l1\n"
        "30 26 8:1 /deploy/data /app/data rw,relatime - ext4 /dev/sda1 rw\n"
        "31 26 8:1 /var/lib/docker/volumes/x/_data /app/logs rw - ext4 /dev/sda1 rw\n"
        "32 26 0:5 / /dev/shm rw,nosuid,nodev - tmpfs shm rw\n"
    )
    entries = _parse_mountinfo(text)
    mps = {str(e[1]) for e in entries}
    assert mps == {"/app/data", "/app/logs"}


def test_daemon_bind_source_compose_layout():
    from app.agent_workspace import _parse_mountinfo, daemon_bind_source

    entries = _parse_mountinfo(
        "26 22 0:22 / / rw,relatime - overlay overlay rw\n"
        "30 26 8:1 /deploy/data /app/data rw - ext4 /dev/sda1 rw\n"
    )
    assert daemon_bind_source(
        Path("/app/data/agent_workspaces/A/c_c1"), mountinfo=entries
    ) == Path("/deploy/data/agent_workspaces/A/c_c1")


def test_daemon_bind_source_longest_prefix_wins():
    from app.agent_workspace import daemon_bind_source

    entries = [
        (Path("/host/data"), Path("/app/data"), "ext4"),
        (Path("/host/nested"), Path("/app/data/nested"), "ext4"),
    ]
    assert daemon_bind_source(
        Path("/app/data/nested/f"), mountinfo=entries
    ) == Path("/host/nested/f")


def test_daemon_bind_source_no_coverage():
    from app.agent_workspace import daemon_bind_source

    entries = [(Path("/host/d"), Path("/app/data"), "ext4")]
    assert daemon_bind_source(Path("/tmp/x"), mountinfo=entries) is None


def test_daemon_bind_source_escaped_space():
    from app.agent_workspace import _parse_mountinfo, daemon_bind_source

    entries = _parse_mountinfo(
        "30 26 8:1 /deploy/my\\040data /app/data rw - ext4 /dev/sda1 rw\n"
    )
    assert daemon_bind_source(Path("/app/data/ws"), mountinfo=entries) == Path(
        "/deploy/my data/ws"
    )


def test_mount_root_env_wins(monkeypatch, tmp_path):
    import app.agent_workspace as aw

    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "host" / "ws"))
    monkeypatch.setattr(aw, "daemon_bind_source", lambda p: Path("/unused"))
    assert aw._workspace_mount_root() == tmp_path / "host" / "ws"


def test_mount_root_derives_from_mountinfo(monkeypatch):
    import app.agent_workspace as aw

    monkeypatch.delenv("AGENT_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("AGENT_WORKSPACE_APP_ROOT", raising=False)
    monkeypatch.setattr(
        aw, "daemon_bind_source", lambda p: Path("/host/data/agent_workspaces")
    )
    assert aw._workspace_mount_root() == Path("/host/data/agent_workspaces")


def test_mount_root_fallback_warns_only_in_container(monkeypatch, caplog):
    import app.agent_workspace as aw

    monkeypatch.delenv("AGENT_WORKSPACE_ROOT", raising=False)
    monkeypatch.setattr(aw, "daemon_bind_source", lambda p: None)
    monkeypatch.setattr(aw, "_warned_no_daemon_root", False)
    # native dev (no /.dockerenv): fallback is correct via Docker Desktop —
    # no warning.
    monkeypatch.setattr(aw, "_app_runs_in_container", lambda: False)
    with caplog.at_level("WARNING", logger="app.agent_workspace"):
        aw._workspace_mount_root()
    assert "AGENT_WORKSPACE_ROOT" not in caplog.text
    # containerized app without env/derivation: warn once.
    monkeypatch.setattr(aw, "_app_runs_in_container", lambda: True)
    with caplog.at_level("WARNING", logger="app.agent_workspace"):
        root = aw._workspace_mount_root()
    assert "AGENT_WORKSPACE_ROOT" in caplog.text
    assert root == aw._workspace_root().resolve()


def test_mount_source_outside_root_warns(store, tmp_path, caplog):
    outside = tmp_path / "elsewhere"
    with caplog.at_level("WARNING", logger="app.agent_workspace"):
        src = store.mount_source(outside)
    assert src == outside
    assert "outside the workspace root" in caplog.text


@pytest.mark.parametrize(
    "p", ["/", "/data", "/etc/ws", "/var/lib/x", "/usr/local/ws", "/root/x", "rel/dir/x"]
)
def test_suspicious_mount_root_true(p):
    from app.agent_workspace import is_suspicious_mount_root

    assert is_suspicious_mount_root(Path(p))


@pytest.mark.parametrize(
    "p",
    ["/opt/app/data/ws", "/home/u/data/agent_workspaces", "/deploy/data/ws", "/srv/x/y"],
)
def test_suspicious_mount_root_false(p):
    from app.agent_workspace import is_suspicious_mount_root

    assert not is_suspicious_mount_root(Path(p))


def test_write_dirs_world_writable_for_sandbox(store):
    import stat

    d = store.scope_dir("a", "assistant")
    store.write(d, "sub/dir/f.txt", b"x")
    for p in (d, d / "sub", d / "sub" / "dir"):
        assert stat.S_IMODE(p.stat().st_mode) == 0o777, p


def test_mkdir_dir_world_writable_for_sandbox(store):
    import stat

    d = store.scope_dir("a", "assistant")
    d.mkdir(parents=True)
    store.mkdir(d, "made/inside")
    assert stat.S_IMODE((d / "made" / "inside").stat().st_mode) == 0o777


def test_written_files_are_sandbox_writable(store):
    """Files must be world-writable (0o666): the bind-mounted sandbox runs as
    uid 1000 and would otherwise be unable to overwrite app-created files."""
    import stat

    d = store.scope_dir("a", "assistant")
    store.write(d, "sub/f.txt", b"x")
    f = (d / "sub" / "f.txt")
    assert stat.S_IMODE(f.stat().st_mode) & 0o666 == 0o666
    assert stat.S_IMODE((d / "sub").stat().st_mode) & 0o777 == 0o777


def test_moved_file_keeps_sandbox_writable_mode(store, tmp_path):
    """move() preserves modes — re-apply 0o666 so a file moved from an
    umask-created source stays writable for the sandbox user."""
    import stat

    d = store.scope_dir("a", "assistant")
    store.write(d, "a.txt", b"x")
    # Simulate a restrictive source file as it could arrive via os.replace.
    os.chmod(d / "a.txt", 0o600)
    store.move(d, "a.txt", "b.txt")
    assert stat.S_IMODE((d / "b.txt").stat().st_mode) & 0o666 == 0o666
