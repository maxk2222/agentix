"""Per-assistant persistent workspace folders ("agent workspace").

Storage layout::

    <root>/<assistant_slug>/shared/          scope="assistant"   (all users/convs share)
    <root>/<assistant_slug>/u_<user_id>/     scope="user"        (per chat user)
    <root>/<assistant_slug>/c_<conv_id>/     scope="conversation"

All path resolution goes through :class:`AgentWorkspaceStore` — callers only ever
see paths *relative* to the scope directory. ``resolve()`` guards against
traversal, absolute paths and symlink escapes; writes are atomic (tmp +
``os.replace``) so a sandbox/agent reader never sees a partial file.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.artifacts import sanitize_filename

logger = logging.getLogger(__name__)

DEFAULT_MAX_FILE_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 500 * 1024 * 1024
DEFAULT_MAX_FILES = 5000

_SCOPES = ("assistant", "user", "conversation")


class WorkspaceError(Exception):
    """Workspace operation failed (traversal, quota, missing entry…)."""


def _workspace_root() -> Path:
    """App-visible root. ``AGENT_WORKSPACE_APP_ROOT`` overrides (containerized
    app where the daemon-side ``AGENT_WORKSPACE_ROOT`` differs); falls back to
    ``<DATA_DIR>/agent_workspaces``."""
    explicit = os.environ.get("AGENT_WORKSPACE_APP_ROOT", "").strip()
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("DATA_DIR", "data")) / "agent_workspaces"


# Top-level dirs whose subtrees must never be bind-mounted into a sandbox —
# a mount root resolving here means a misconfigured AGENT_WORKSPACE_ROOT or a
# failed path derivation that would expose system or unrelated host data.
_SUSPICIOUS_ROOT_PARTS = {
    "etc", "bin", "boot", "dev", "proc", "sys", "usr", "var",
    "root", "run", "sbin", "lib", "lib64",
}


def _suspicious_parts(parts: tuple) -> bool:
    if len(parts) < 3:  # "/" itself or "/<dir>"
        return True
    return parts[0] != "/" or parts[1] in _SUSPICIOUS_ROOT_PARTS


def is_suspicious_mount_root(path: Path) -> bool:
    """True when *path* is too shallow or system-owned to be a workspace root.

    A daemon bind source this broad would mount a large, unrelated host
    subtree into the sandbox — e.g. ``AGENT_WORKSPACE_ROOT=/`` exposes the
    entire host filesystem. Legitimate roots (``/opt/app/data/...``,
    ``/home/user/data/...``) are at least 3 components deep under a
    non-system prefix. Checked both lexically and resolved — e.g. ``/etc``
    resolves to ``/private/etc`` on macOS and would slip a resolved-only
    check.
    """
    raw = Path(path)
    if _suspicious_parts(raw.parts):
        return True
    try:
        return _suspicious_parts(raw.resolve().parts)
    except OSError:
        return False


def _unescape_mountinfo_path(raw: str) -> str:
    """Decode octal escapes (``\\040`` etc.) used for spaces in mountinfo."""
    out, i = [], 0
    while i < len(raw):
        if raw[i] == "\\" and i + 3 < len(raw) and raw[i + 1 : i + 4].isdigit():
            out.append(chr(int(raw[i + 1 : i + 4], 8)))
            i += 4
        else:
            out.append(raw[i])
            i += 1
    return "".join(out)


# Filesystem types that can never be a host directory bind — the container's
# own rootfs (overlay) and kernel-virtual filesystems. Treating them as bind
# sources would translate the app path to itself, silently defeating the
# whole derivation. Docker-Desktop shares (virtiofs/fuse variants) are real
# binds and intentionally NOT listed.
_VIRTUAL_FS_TYPES = {
    "overlay", "tmpfs", "proc", "sysfs", "devpts", "devtmpfs", "cgroup",
    "cgroup2", "mqueue", "shm", "fusectl", "debugfs", "tracefs",
    "securityfs", "pstore", "bpf", "hugetlbfs", "configfs", "autofs",
    "nsfs", "ramfs", "selinuxfs", "efivarfs", "binfmt_misc", "rpc_pipefs",
    "squashfs",
}


_MOUNTINFO_ENTRIES: Optional[List[tuple]] = None


def _mountinfo_entries() -> List[tuple]:
    """Parsed ``/proc/self/mountinfo`` as (root, mount_point, fstype) — cached."""
    global _MOUNTINFO_ENTRIES
    if _MOUNTINFO_ENTRIES is not None:
        return _MOUNTINFO_ENTRIES
    try:
        text = Path("/proc/self/mountinfo").read_text()
    except OSError:
        _MOUNTINFO_ENTRIES = []
        return []
    _MOUNTINFO_ENTRIES = _parse_mountinfo(text)
    return _MOUNTINFO_ENTRIES


def _parse_mountinfo(text: str) -> List[tuple]:
    entries: List[tuple] = []
    for line in text.splitlines():
        fields = line.split(" ")
        # Format: id parent major:minor root mount_point options... - fstype src opts
        try:
            sep = fields.index("-")
            fstype = fields[sep + 1]
        except (ValueError, IndexError):
            continue
        if len(fields) < 5 or fstype in _VIRTUAL_FS_TYPES:
            continue
        try:
            entries.append(
                (
                    Path(_unescape_mountinfo_path(fields[3])),
                    Path(_unescape_mountinfo_path(fields[4])),
                    fstype,
                )
            )
        except ValueError:
            continue
    return entries


def daemon_bind_source(
    app_path: Path, *, mountinfo: Optional[List[tuple]] = None
) -> Optional[Path]:
    """Translate an app-side path to the daemon-side bind source.

    When the app runs in a container, bind mounts of the data directory are
    visible in ``/proc/self/mountinfo``: the ``root`` field holds the
    daemon-visible (host) path of the mount's source. The longest mount point
    covering *app_path* wins. Returns ``None`` when no real filesystem mount
    covers the path (e.g. plain overlay filesystem — then nothing can be
    bind-mounted anyway).
    """
    if mountinfo is None:
        mountinfo = _mountinfo_entries()
    app_path = Path(app_path).resolve()
    best: Optional[tuple] = None
    for entry in mountinfo:
        root, mount_point = entry[0], entry[1]
        try:
            rel = app_path.relative_to(mount_point)
        except ValueError:
            continue
        if best is None or len(mount_point.parts) > len(best[0].parts):
            best = (mount_point, root, rel)
    if best is None:
        return None
    return best[1] / best[2]


_warned_no_daemon_root = False


def _app_runs_in_container() -> bool:
    """Best-effort container detection for warning gating only."""
    if os.environ.get("RUNNING_IN_DOCKER", "").lower() in ("1", "true", "yes"):
        return True
    return os.path.exists("/.dockerenv")


def _workspace_mount_root() -> Path:
    """Daemon-visible bind-source root (absolute for dockerd).

    ``AGENT_WORKSPACE_ROOT`` wins. Unset, the mount source is derived from
    ``/proc/self/mountinfo`` so the common containerized deployment
    (``./data:/app/data`` + docker.sock) works without configuration. Last
    resort is the app path itself — correct only when app and daemon share
    the filesystem; otherwise the sandbox silently mounts an empty dir.
    """
    global _warned_no_daemon_root
    daemon = os.environ.get("AGENT_WORKSPACE_ROOT", "").strip()
    if daemon:
        return Path(daemon)
    derived = daemon_bind_source(_workspace_root())
    if derived is not None:
        return derived
    # Only warn inside a container — on a native dev machine (Docker Desktop
    # shares the host fs) the app-path fallback below is correct, so a
    # warning there would be a false alarm.
    if not _warned_no_daemon_root and _app_runs_in_container():
        _warned_no_daemon_root = True
        logger.warning(
            "AGENT_WORKSPACE_ROOT is unset and no host bind mount covers %s; "
            "sandbox /agent_ws mounts will be empty. Set AGENT_WORKSPACE_ROOT "
            "to the host-visible workspace path.",
            _workspace_root(),
        )
    return _workspace_root().resolve()


def normalize_scope(scope: Optional[str]) -> str:
    return scope if scope in _SCOPES else "conversation"


class AgentWorkspaceStore:
    """Filesystem store for per-assistant agent workspace folders."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self._root = Path(root) if root is not None else _workspace_root()

    @property
    def root(self) -> Path:
        return self._root

    # -- scope dirs -----------------------------------------------------------

    @staticmethod
    def assistant_slug(assistant_name: str) -> str:
        return sanitize_filename(assistant_name or "assistant")

    def scope_dir(
        self,
        assistant_name: str,
        scope: Optional[str],
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Path:
        """Directory backing the requested scope.

        ``user`` scope falls back to ``conversation`` when no user identity is
        available (anonymous callers) so an agent never loses its folder.
        """
        scope = normalize_scope(scope)
        base = self._root / self.assistant_slug(assistant_name)
        if scope == "assistant":
            return base / "shared"
        if scope == "user" and user_id:
            return base / f"u_{sanitize_filename(user_id)}"
        conv = conversation_id or "shared"
        return base / f"c_{sanitize_filename(conv)}"

    def mount_source(self, scope_dir: Path) -> Path:
        """Daemon-side bind source for an app-side scope dir."""
        try:
            rel = Path(scope_dir).resolve().relative_to(self._root.resolve())
        except ValueError:
            logger.warning(
                "Agent workspace scope dir %s is outside the workspace root %s; "
                "using it verbatim as bind source — check AGENT_WORKSPACE_*_ROOT",
                scope_dir, self._root,
            )
            return Path(scope_dir)
        return _workspace_mount_root() / rel

    @staticmethod
    def _chmod_dir_chain(scope_dir: Path, target_dir: Path) -> None:
        """Best-effort 0o777 on scope_dir and each dir down to *target_dir*.

        The sandbox mounts the scope dir and runs as uid 1000 — directories
        created with the default umask (755) would not be writable inside the
        container, so store-created dirs are made world-writable (the scope
        root itself is already created 0o777 by the sandbox manager).
        """
        try:
            rel_parts = (
                Path(target_dir).resolve().relative_to(Path(scope_dir).resolve()).parts
            )
        except ValueError:
            return
        cursor = Path(scope_dir).resolve()
        dirs = [cursor]
        for part in rel_parts:
            cursor = cursor / part
            dirs.append(cursor)
        for entry in dirs:
            try:
                os.chmod(entry, 0o777)
            except OSError:
                pass

    @staticmethod
    def _chmod_file(path: Path) -> None:
        """Best-effort 0o666 — the bind-mounted sandbox runs as uid 1000 and
        must be able to modify/overwrite files the app created (default umask
        644 would make them read-only for the sandbox user)."""
        try:
            os.chmod(path, 0o666)
        except OSError:
            pass

    # -- safe path resolution -------------------------------------------------

    def resolve(self, scope_dir: Path, rel_path: str) -> Path:
        """Resolve ``rel_path`` inside ``scope_dir``; raises WorkspaceError on
        traversal/absolute paths/symlink escapes."""
        rel = (rel_path or "").strip().replace("\\", "/")
        if not rel or rel in (".", "/"):
            raise WorkspaceError("path required")
        if "\x00" in rel:
            raise WorkspaceError("invalid path")
        if rel.startswith("/") or rel.split("/")[0] == "..":
            raise WorkspaceError("path must be relative")
        root = Path(scope_dir).resolve()
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise WorkspaceError("path escapes workspace")
        # Reject symlinks anywhere in the resolved chain (root itself may be a
        # legit symlink target — compare against the already-resolved path).
        cursor = root
        for part in candidate.relative_to(root).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise WorkspaceError("symlinks are not allowed")
        return candidate

    # -- operations ------------------------------------------------------------

    def list_entries(self, scope_dir: Path, rel_dir: str = ".") -> List[Dict[str, Any]]:
        # Resolve once — ``relative_to`` is lexical, so entries under a
        # resolved target would not match an unresolved (e.g. symlinked)
        # scope_dir and silently vanish.
        scope_dir = Path(scope_dir).resolve()
        if not scope_dir.exists():
            return []
        target = scope_dir if rel_dir in ("", ".", "/") else self.resolve(scope_dir, rel_dir)
        if not target.is_dir():
            raise WorkspaceError("not a directory")
        entries: List[Dict[str, Any]] = []
        for item in sorted(target.iterdir(), key=lambda p: p.name):
            try:
                st = item.lstat()
                if item.is_symlink():
                    continue
                entries.append(
                    {
                        "name": item.name,
                        "path": str(item.relative_to(scope_dir)).replace(os.sep, "/"),
                        "type": "dir" if item.is_dir() else "file",
                        "size": st.st_size if item.is_file() else None,
                        "mtime": st.st_mtime,
                    }
                )
            except OSError:
                continue
        return entries

    def usage(self, scope_dir: Path) -> Dict[str, int]:
        files = 0
        total = 0
        scope_dir = Path(scope_dir)
        if scope_dir.exists():
            for dirpath, dirnames, filenames in os.walk(scope_dir, followlinks=False):
                for fn in filenames:
                    fp = Path(dirpath) / fn
                    try:
                        if fp.is_symlink():
                            continue
                        files += 1
                        total += fp.stat().st_size
                    except OSError:
                        continue
        return {"files": files, "bytes": total}

    def file_breakdown(self, scope_dir: Path, limit: int = 3) -> List[Dict[str, Any]]:
        """File counts grouped by the top-level entry inside ``scope_dir``.

        Used to explain quota errors: same walk semantics as ``usage()``
        (recursive, hidden dirs included, symlinks skipped), but aggregated
        per first path component so the report points at the culprit dir.
        """
        counts: Dict[str, int] = {}
        scope_dir = Path(scope_dir)
        if scope_dir.exists():
            for dirpath, _dirnames, filenames in os.walk(scope_dir, followlinks=False):
                rel = Path(dirpath).relative_to(scope_dir)
                top = rel.parts[0] if rel.parts else ""
                for fn in filenames:
                    fp = Path(dirpath) / fn
                    try:
                        if fp.is_symlink():
                            continue
                    except OSError:
                        continue
                    counts[top] = counts.get(top, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [{"name": name or "(root)", "files": n} for name, n in ranked[:limit]]

    def read(self, scope_dir: Path, rel_path: str, max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> bytes:
        path = self.resolve(scope_dir, rel_path)
        if not path.is_file():
            raise WorkspaceError("not a file")
        size = path.stat().st_size
        if size > max_bytes:
            raise WorkspaceError(f"file exceeds {max_bytes // (1024 * 1024)} MB limit")
        return path.read_bytes()

    def _check_quota(
        self,
        scope_dir: Path,
        final_file_bytes: int,
        added_bytes: int,
        cfg: Optional[Dict[str, Any]],
        replacing_bytes: int = 0,
        new_file: bool = True,
    ) -> None:
        cfg = cfg or {}
        max_file = int(cfg.get("max_file_mb") or 0) * 1024 * 1024 or DEFAULT_MAX_FILE_BYTES
        max_total = int(cfg.get("max_total_mb") or 0) * 1024 * 1024 or DEFAULT_MAX_TOTAL_BYTES
        max_files = int(cfg.get("max_files") or 0) or DEFAULT_MAX_FILES
        if final_file_bytes > max_file:
            raise WorkspaceError(f"file exceeds {max_file // (1024 * 1024)} MB limit")
        usage = self.usage(scope_dir)
        if new_file and usage["files"] + 1 > max_files:
            top = self.file_breakdown(scope_dir)
            detail = ", ".join(f"{t['name']}/: {t['files']}" if t["name"] != "(root)" else f"(root): {t['files']}" for t in top)
            suffix = f" — {detail}, total {usage['files']}" if detail else f" — total {usage['files']}"
            raise WorkspaceError(f"workspace file limit ({max_files}) reached{suffix}")
        # ``replacing_bytes`` are freed by the overwrite — don't count them twice.
        if usage["bytes"] - replacing_bytes + added_bytes > max_total:
            raise WorkspaceError(f"workspace size limit ({max_total // (1024 * 1024)} MB) reached")

    def write(
        self,
        scope_dir: Path,
        rel_path: str,
        data: bytes,
        cfg: Optional[Dict[str, Any]] = None,
        append: bool = False,
    ) -> Dict[str, Any]:
        path = self.resolve(scope_dir, rel_path)
        replacing = path.exists()
        old_size = path.stat().st_size if replacing else 0
        final_size = len(data) + (old_size if append else 0)
        self._check_quota(
            scope_dir, final_size, len(data), cfg,
            replacing_bytes=0 if append else old_size, new_file=not replacing,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._chmod_dir_chain(scope_dir, path.parent)
        if append:
            with path.open("ab") as fh:
                fh.write(data)
        else:
            tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
            try:
                tmp.write_bytes(data)
                os.replace(tmp, path)
            finally:
                if tmp.exists():
                    tmp.unlink()
        self._chmod_file(path)
        st = path.stat()
        return {
            "name": path.name,
            "path": str(path.relative_to(Path(scope_dir).resolve())).replace(os.sep, "/"),
            "type": "file",
            "size": st.st_size,
            "mtime": st.st_mtime,
        }

    def mkdir(self, scope_dir: Path, rel_path: str) -> Dict[str, Any]:
        path = self.resolve(scope_dir, rel_path)
        path.mkdir(parents=True, exist_ok=True)
        self._chmod_dir_chain(scope_dir, path)
        return {"name": path.name, "path": rel_path, "type": "dir"}

    def delete(self, scope_dir: Path, rel_path: str) -> Dict[str, Any]:
        import shutil

        path = self.resolve(scope_dir, rel_path)
        if not path.exists():
            raise WorkspaceError("not found")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return {"name": path.name, "path": rel_path, "deleted": True}

    def move(self, scope_dir: Path, src_rel: str, dst_rel: str) -> Dict[str, Any]:
        src = self.resolve(scope_dir, src_rel)
        dst = self.resolve(scope_dir, dst_rel)
        if not src.exists():
            raise WorkspaceError("source not found")
        dst.parent.mkdir(parents=True, exist_ok=True)
        self._chmod_dir_chain(scope_dir, dst.parent)
        os.replace(src, dst)
        if dst.is_dir():
            self._chmod_dir_chain(scope_dir, dst)
        else:
            self._chmod_file(dst)
        return {"name": dst.name, "path": dst_rel, "type": "dir" if dst.is_dir() else "file"}

    # -- lifecycle --------------------------------------------------------------

    def delete_tree(self, scope_dir: Path) -> None:
        import shutil

        try:
            shutil.rmtree(Path(scope_dir), ignore_errors=True)
        except Exception as exc:
            logger.debug("Workspace delete failed for %s: %s", scope_dir, exc)

    def delete_assistant(self, assistant_name: str) -> None:
        """Remove the whole workspace tree of an assistant (all scopes)."""
        self.delete_tree(self._root / self.assistant_slug(assistant_name))

    def rename_assistant(self, old_name: str, new_name: str) -> bool:
        """Atomically move an assistant's workspace tree to a new slug.

        No-op (returns False) when the source does not exist; refuses to
        clobber an existing destination. Best-effort — failures are logged,
        never raised, so a config save cannot be broken by the filesystem.
        """
        old_slug, new_slug = self.assistant_slug(old_name), self.assistant_slug(new_name)
        if old_slug == new_slug:
            return False
        src, dst = self._root / old_slug, self._root / new_slug
        try:
            if not src.is_dir() or dst.exists():
                return False
            os.replace(src, dst)
            logger.info("Renamed agent workspace %s -> %s", old_slug, new_slug)
            return True
        except OSError as exc:
            logger.warning("Agent workspace rename %s -> %s failed: %s", src, dst, exc)
            return False

    def orphan_assistant_dirs(self, live_slugs: set) -> List[Path]:
        """Dirs under the root whose assistant no longer exists."""
        if not self._root.exists():
            return []
        orphans = []
        for entry in self._root.iterdir():
            if entry.is_dir() and entry.name not in live_slugs:
                orphans.append(entry)
        return orphans
