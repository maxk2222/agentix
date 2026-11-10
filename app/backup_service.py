"""Database backup & restore orchestration.

Creates consistent ``pg_dump -Fc`` snapshots of the Postgres database,
encrypts them with chunked AES-256-GCM (:mod:`app.backup_crypto`) and
uploads them to a WebDAV target (e.g. Sciebo) alongside a JSON manifest
containing checksums, sizes and per-table row counts.

Restore downloads + verifies an archive, takes a mandatory pre-restore
safety dump and performs an in-place ``pg_restore --clean`` while a
maintenance flag pauses the backup and trigger schedulers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from agent.config import AppConfig
from app.backup_crypto import (
    BackupCryptoError,
    decrypt_file,
    encrypt_file,
    load_encryption_key,
    sha256_file,
)
from app.backup_store import BackupConfigStore, BackupJobStore
from app.webdav_sync import WebDAVClient, WebDAVError

logger = logging.getLogger(__name__)

BACKUP_FILE_RE = re.compile(r"^agentix-backup-[\w.-]+\.dump\.enc$")
MANIFEST_SUFFIX = ".manifest.json"

# Tables whose exact row counts are captured in the manifest for
# post-restore validation (all base tables in the current schema).


class BackupServiceError(Exception):
    """Generic backup/restore failure with a user-facing message."""


class BackupService:
    """Orchestrates dump -> encrypt -> upload -> verify -> retain."""

    def __init__(
        self,
        engine: AsyncEngine,
        config: AppConfig,
        config_store: BackupConfigStore,
        job_store: BackupJobStore,
        app_state: Any = None,
    ) -> None:
        self.engine = engine
        self.config = config
        self.config_store = config_store
        self.job_store = job_store
        self._app_state = app_state
        self._lock = asyncio.Lock()
        self._maintenance = False
        self._cancel = asyncio.Event()

    # ------------------------------------------------------------------
    # state helpers
    # ------------------------------------------------------------------
    @property
    def in_maintenance(self) -> bool:
        """True while a destructive restore is running (pauses schedulers).

        The flag is mirrored onto ``app.state`` so it survives the hot-reload
        that recreates this service instance at the end of a restore.
        """
        if self._maintenance:
            return True
        if self._app_state is not None:
            return bool(getattr(self._app_state, "backup_maintenance", False))
        return False

    def _set_maintenance(self, active: bool) -> None:
        self._maintenance = active
        if self._app_state is not None:
            self._app_state.backup_maintenance = active

    @property
    def busy(self) -> bool:
        return self._lock.locked() or self._maintenance

    def cancel(self) -> None:
        """Request cancellation of the currently running job."""
        self._cancel.set()

    def _check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise BackupServiceError("Backup job cancelled by user")

    # ------------------------------------------------------------------
    # config / client helpers
    # ------------------------------------------------------------------
    def _tmp_dir(self) -> Path:
        d = Path(self.config.backup_tmp_dir)
        d.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass
        return d

    def _instance_name(self) -> str:
        return self.config.backup_instance or socket.gethostname() or "agentix"

    async def _require_config(self) -> Dict[str, Any]:
        cfg = await self.config_store.get_config_internal()
        if cfg is None or not cfg.get("webdav_url"):
            raise BackupServiceError("Backup target is not configured")
        url = cfg["webdav_url"]
        if not url.startswith("https://"):
            raise BackupServiceError("Only HTTPS WebDAV URLs are allowed")
        if not cfg.get("username"):
            raise BackupServiceError("WebDAV username is not configured")
        return cfg

    def _client(self, cfg: Dict[str, Any]) -> WebDAVClient:
        return WebDAVClient(
            cfg["webdav_url"],
            cfg.get("username") or "",
            cfg.get("password") or "",
            timeout=float(self.config.backup_upload_timeout),
        )

    @staticmethod
    def _check_remote_path(remote_path: str) -> str:
        if not remote_path.startswith("/"):
            raise BackupServiceError("remote_path must start with '/'")
        if ".." in remote_path.split("/"):
            raise BackupServiceError("remote_path must not contain '..'")
        return remote_path.rstrip("/") or "/"

    # ------------------------------------------------------------------
    # subprocess helpers
    # ------------------------------------------------------------------
    def _pg_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env["PGPASSWORD"] = self.config.pg_password  # never on argv
        return env

    def _pg_conn_args(self) -> List[str]:
        return [
            "-h", self.config.pg_host,
            "-p", str(self.config.pg_port),
            "-U", self.config.pg_user,
        ]

    async def _run_pg_tool(self, tool: str, args: List[str], timeout: float) -> str:
        """Run pg_dump/pg_restore and return combined stdout+stderr."""
        try:
            proc = await asyncio.create_subprocess_exec(
                tool, *args,
                env=self._pg_env(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            raise BackupServiceError(
                f"'{tool}' not found in PATH - install postgresql-client "
                "(see docs/backup-and-recovery.md)"
            )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise BackupServiceError(f"'{tool}' timed out after {timeout}s")
        out = (stdout + stderr).decode("utf-8", "replace").strip()
        if proc.returncode != 0:
            raise BackupServiceError(f"'{tool}' failed (rc={proc.returncode}): {out[:2000]}")
        return out

    async def _dump_database(self, dest: Path) -> None:
        cfg = self.config
        await self._run_pg_tool(
            "pg_dump",
            self._pg_conn_args()
            + ["-Fc", "-f", str(dest), cfg.pg_database],
            timeout=float(cfg.backup_dump_timeout),
        )

    async def _pg_dump_version(self) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "pg_dump", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            return stdout.decode("utf-8", "replace").strip()
        except Exception:
            return "unavailable"

    async def _server_version(self) -> str:
        try:
            async with self.engine.connect() as conn:
                res = await conn.execute(text("SHOW server_version"))
                return str(res.scalar() or "unknown")
        except Exception:
            return "unknown"

    async def _table_row_counts(self) -> Dict[str, int]:
        """Exact row counts for every base table in the current schema."""
        counts: Dict[str, int] = {}
        async with self.engine.connect() as conn:
            res = await conn.execute(
                text(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = current_schema() AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                    """
                )
            )
            tables = [r[0] for r in res.fetchall()]
            for name in tables:
                quoted = '"' + name.replace('"', '""') + '"'
                try:
                    cres = await conn.execute(text(f"SELECT COUNT(*) FROM {quoted}"))
                    counts[name] = int(cres.scalar() or 0)
                except Exception as exc:
                    logger.warning("Row count failed for table %s: %s", name, exc)
        return counts

    async def _check_disk_space(self, tmp: Path) -> None:
        """Abort early when tmp space is clearly insufficient."""
        async with self.engine.connect() as conn:
            res = await conn.execute(
                text("SELECT pg_database_size(current_database())")
            )
            db_size = int(res.scalar() or 0)
        free = shutil.disk_usage(tmp).free
        # pg_dump -Fc compresses, but encrypted output + safety margin:
        needed = int(db_size * 1.5)
        if free < needed:
            raise BackupServiceError(
                f"Insufficient disk space in {tmp}: {free} bytes free, "
                f"~{needed} bytes required"
            )

    # ------------------------------------------------------------------
    # backup pipeline
    # ------------------------------------------------------------------
    async def run_backup(
        self, kind: str = "manual", triggered_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the full backup pipeline. Raises BackupServiceError on failure."""
        if self._maintenance:
            raise BackupServiceError("Restore in progress - backups are paused")
        if self._lock.locked():
            raise BackupServiceError("Another backup job is already running")

        job_id = await self.job_store.create_job(kind, triggered_by)
        started = time.perf_counter()
        error: Optional[str] = None
        filename: Optional[str] = None
        remote_path: Optional[str] = None
        size_bytes: Optional[int] = None
        sha_plain: Optional[str] = None
        sha_enc: Optional[str] = None
        tmp_files: List[Path] = []

        async with self._lock:
            try:
                self._cancel.clear()
                key = load_encryption_key()  # fail-closed when unset
                cfg = await self._require_config()
                remote_path = self._check_remote_path(cfg.get("remote_path") or "/agentix-backups")

                tmp = self._tmp_dir()
                await self._check_disk_space(tmp)
                self._check_cancelled()

                ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                basename = f"agentix-backup-{self._instance_name()}-{ts}"
                dump_path = tmp / f"{basename}.dump"
                enc_path = tmp / f"{basename}.dump.enc"
                manifest_path = tmp / f"{basename}{MANIFEST_SUFFIX}"
                tmp_files += [dump_path, enc_path, manifest_path]
                filename = enc_path.name

                # 1. dump
                logger.info("Backup job %s: pg_dump starting", job_id)
                await self._dump_database(dump_path)
                self._check_cancelled()

                # 2. encrypt (streaming, hashes computed inline)
                enc_info = encrypt_file(dump_path, enc_path, key)
                sha_plain = str(enc_info["sha256_plain"])
                sha_enc = str(enc_info["sha256_enc"])
                size_bytes = int(enc_info["size_enc"])
                self._check_cancelled()

                # 3. manifest
                manifest = {
                    "format": "agentix-backup/1",
                    "filename": filename,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "instance": self._instance_name(),
                    "pg_dump_version": await self._pg_dump_version(),
                    "pg_server_version": await self._server_version(),
                    "app_version": "1.0.0",
                    "encryption": "aes-256-gcm-chunked",
                    "sha256_plain": sha_plain,
                    "sha256_enc": sha_enc,
                    "size_plain": enc_info["size_plain"],
                    "size_enc": size_bytes,
                    "row_counts": await self._table_row_counts(),
                }
                manifest_path.write_text(json.dumps(manifest, indent=2))

                # 4. upload (with retries + backoff)
                remote_file = f"{remote_path}/{filename}"
                remote_manifest = f"{remote_path}/{manifest_path.name}"
                await self._upload_with_retry(cfg, remote_path, [
                    (enc_path, remote_file),
                    (manifest_path, remote_manifest),
                ])
                self._check_cancelled()

                # 5. verify remote size
                async with self._client(cfg) as client:
                    info = await client.stat(remote_file)
                if not info or int(info.get("content_length") or -1) != size_bytes:
                    raise BackupServiceError(
                        "Remote verification failed: uploaded file size mismatch"
                    )

                # 6. retention
                pruned = await self._apply_retention(cfg, remote_path)
                if pruned:
                    logger.info("Backup retention removed %d old archive(s)", pruned)

            except BackupServiceError as exc:
                error = str(exc)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Backup job %s failed unexpectedly", job_id)
                error = f"unexpected error: {exc}"
            finally:
                for p in tmp_files:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        logger.warning("Failed to clean temp file %s", p)

        duration_ms = int((time.perf_counter() - started) * 1000)
        status = "cancelled" if error == "Backup job cancelled by user" else (
            "error" if error else "success"
        )
        await self.job_store.finish_job(
            job_id, status,
            filename=filename, remote_path=remote_path, size_bytes=size_bytes,
            sha256_plain=sha_plain, sha256_enc=sha_enc,
            duration_ms=duration_ms, error=error,
        )
        await self.config_store.update_last_run()
        if error:
            raise BackupServiceError(error)
        return {
            "job_id": job_id,
            "filename": filename,
            "size_bytes": size_bytes,
            "sha256": sha_enc,
            "duration_ms": duration_ms,
        }

    async def _upload_with_retry(
        self,
        cfg: Dict[str, Any],
        remote_dir: str,
        files: List[tuple],
    ) -> None:
        retries = int(self.config.backup_upload_retries)
        delay = 2.0
        last_exc: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            self._check_cancelled()
            uploaded: List[str] = []
            try:
                async with self._client(cfg) as client:
                    await client.ensure_collection(remote_dir)
                    for local, remote in files:
                        await client.upload(remote, local)
                        uploaded.append(remote)
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Backup upload attempt %d/%d failed: %s", attempt, retries, exc
                )
                # clean up partial uploads
                try:
                    async with self._client(cfg) as client:
                        for remote in uploaded:
                            await client.delete(remote)
                except Exception:
                    logger.warning("Cleanup of partial upload failed")
                if attempt < retries:
                    await asyncio.sleep(delay)
                    delay *= 2
        raise BackupServiceError(f"Upload failed after {retries} attempts: {last_exc}")

    async def _apply_retention(self, cfg: Dict[str, Any], remote_dir: str) -> int:
        count = int(cfg.get("retention_count") or 10)
        days = cfg.get("retention_days")
        try:
            async with self._client(cfg) as client:
                files = await client.list_files(remote_dir.lstrip("/"))
        except Exception as exc:
            logger.warning("Retention listing failed: %s", exc)
            return 0

        archives = [
            f for f in files
            if BACKUP_FILE_RE.match(Path(f["href"]).name)
            and "/pre-restore/" not in f["href"]
        ]
        archives.sort(key=lambda e: Path(e["href"]).name)
        to_delete: List[Dict[str, Any]] = []
        if len(archives) > count:
            to_delete += archives[: len(archives) - count]
        if days:
            cutoff = datetime.now(timezone.utc).timestamp() - int(days) * 86400
            for e in archives:
                lm = e.get("last_modified")
                if lm and lm.timestamp() < cutoff and e not in to_delete:
                    to_delete.append(e)

        deleted = 0
        async with self._client(cfg) as client:
            for e in to_delete:
                href = e["href"]
                try:
                    await client.delete(href)
                    manifest_href = re.sub(r"\.dump\.enc$", MANIFEST_SUFFIX, href)
                    await client.delete(manifest_href)
                    deleted += 1
                except Exception as exc:
                    logger.warning("Retention delete failed for %s: %s", href, exc)
        return deleted

    # ------------------------------------------------------------------
    # remote inspection / verification
    # ------------------------------------------------------------------
    async def list_remote(self) -> List[Dict[str, Any]]:
        cfg = await self._require_config()
        remote_dir = self._check_remote_path(cfg.get("remote_path") or "/agentix-backups")
        async with self._client(cfg) as client:
            try:
                files = await client.list_files(remote_dir.lstrip("/"))
            except Exception as exc:
                raise BackupServiceError(f"Remote listing failed: {exc}")
        out = []
        for f in files:
            name = Path(f["href"]).name
            out.append({
                "name": name,
                "href": f["href"],
                "size_bytes": f.get("content_length"),
                "last_modified": f.get("last_modified"),
                "is_backup": bool(BACKUP_FILE_RE.match(name)),
                "is_manifest": name.endswith(MANIFEST_SUFFIX),
            })
        out.sort(key=lambda e: e["name"], reverse=True)
        return out

    async def test_connection(
        self, webdav_url: str, username: str, password: str, remote_path: str
    ) -> Dict[str, Any]:
        """Write-test a WebDAV target: MKCOL + PUT + PROPFIND + DELETE."""
        if not webdav_url.startswith("https://"):
            raise BackupServiceError("Only HTTPS WebDAV URLs are allowed")
        remote_path = self._check_remote_path(remote_path)
        probe_name = f".agentix-backup-probe-{int(time.time())}.txt"
        tmp = self._tmp_dir() / probe_name
        tmp.write_bytes(b"agentix backup connection test\n")
        try:
            async with self._client({
                "webdav_url": webdav_url, "username": username, "password": password
            }) as client:
                await client.ensure_collection(remote_path)
                await client.upload(f"{remote_path}/{probe_name}", tmp)
                info = await client.stat(f"{remote_path}/{probe_name}")
                if not info:
                    raise BackupServiceError("Uploaded probe file is not visible")
                await client.delete(f"{remote_path}/{probe_name}")
            return {"success": True}
        except BackupServiceError:
            raise
        except Exception as exc:
            raise BackupServiceError(f"Connection test failed: {exc}")
        finally:
            tmp.unlink(missing_ok=True)

    async def verify_remote(self, filename: str) -> Dict[str, Any]:
        """Download + decrypt a remote backup and check both checksums."""
        if not BACKUP_FILE_RE.match(filename):
            raise BackupServiceError("Invalid backup filename")
        cfg = await self._require_config()
        key = load_encryption_key()
        remote_dir = self._check_remote_path(cfg.get("remote_path") or "/agentix-backups")

        tmp = self._tmp_dir()
        enc_path = tmp / f"verify-{filename}"
        dec_path = tmp / f"verify-{filename}.dec"
        manifest_path = tmp / re.sub(r"\.dump\.enc$", MANIFEST_SUFFIX, filename)
        try:
            async with self._client(cfg) as client:
                await client.download_to_file(f"{remote_dir}/{filename}", enc_path)
                manifest_name = manifest_path.name
                await client.download_to_file(
                    f"{remote_dir}/{manifest_name}", manifest_path
                )
            manifest = json.loads(manifest_path.read_text())
            if sha256_file(enc_path) != manifest.get("sha256_enc"):
                return {"valid": False, "detail": "encrypted artifact checksum mismatch"}
            info = decrypt_file(enc_path, dec_path, key)
            if info["sha256_plain"] != manifest.get("sha256_plain"):
                return {"valid": False, "detail": "decrypted dump checksum mismatch"}
            return {
                "valid": True,
                "detail": "checksums verified",
                "size_bytes": info["size_plain"],
            }
        except BackupCryptoError as exc:
            return {"valid": False, "detail": str(exc)}
        finally:
            for p in (enc_path, dec_path, manifest_path):
                p.unlink(missing_ok=True)

    async def delete_remote(self, filename: str) -> bool:
        """Delete a single remote backup artifact and its manifest."""
        if not BACKUP_FILE_RE.match(filename):
            raise BackupServiceError("Invalid backup filename")
        cfg = await self._require_config()
        remote_dir = self._check_remote_path(cfg.get("remote_path") or "/agentix-backups")
        manifest_name = re.sub(r"\.dump\.enc$", MANIFEST_SUFFIX, filename)
        async with self._client(cfg) as client:
            removed = await client.delete(f"{remote_dir}/{filename}")
            try:
                await client.delete(f"{remote_dir}/{manifest_name}")
            except Exception:
                pass
        return removed

    # ------------------------------------------------------------------
    # restore
    # ------------------------------------------------------------------
    async def restore_backup(self, filename: str, app: Any = None) -> Dict[str, Any]:
        """Restore a remote backup into the live database (in-place).

        Steps: download+verify -> mandatory pre-restore safety dump ->
        maintenance mode -> terminate foreign DB connections ->
        pg_restore --clean -> post-validation -> hot reload.
        """
        if not BACKUP_FILE_RE.match(filename):
            raise BackupServiceError("Invalid backup filename")
        if self._lock.locked():
            raise BackupServiceError("Another backup job is already running")

        job_id = await self.job_store.create_job("restore", filename)
        started = time.perf_counter()
        tmp = self._tmp_dir()
        enc_path = tmp / f"restore-{filename}"
        dec_path = tmp / f"restore-{filename}.dump"
        manifest_local = tmp / re.sub(r"\.dump\.enc$", MANIFEST_SUFFIX, filename)
        cleanup = [enc_path, dec_path, manifest_local]
        error: Optional[str] = None
        detail: Dict[str, Any] = {}

        async with self._lock:
            self._set_maintenance(True)
            try:
                cfg = await self._require_config()
                key = load_encryption_key()
                remote_dir = self._check_remote_path(cfg.get("remote_path") or "/agentix-backups")

                # 1. download + verify
                async with self._client(cfg) as client:
                    await client.download_to_file(f"{remote_dir}/{filename}", enc_path)
                    try:
                        await client.download_to_file(
                            f"{remote_dir}/{manifest_local.name}", manifest_local
                        )
                    except Exception:
                        logger.warning("Manifest not found remotely for %s", filename)
                expected_enc = self._manifest_hash(manifest_local, "sha256_enc")
                if expected_enc and sha256_file(enc_path) != expected_enc:
                    raise BackupServiceError("Encrypted artifact checksum mismatch")
                info = decrypt_file(enc_path, dec_path, key)
                expected_plain = self._manifest_hash(manifest_local, "sha256_plain")
                if expected_plain and info["sha256_plain"] != expected_plain:
                    raise BackupServiceError("Decrypted dump checksum mismatch")
                await self._run_pg_tool(
                    "pg_restore", ["--list", str(dec_path)],
                    timeout=60.0,
                )

                # 2. mandatory pre-restore safety dump (best-effort upload)
                pre = await self._pre_restore_dump(cfg, remote_dir)
                detail["pre_restore_backup"] = pre

                # 3. terminate foreign connections (privilege check)
                await self._terminate_connections()

                # 4. in-place restore
                await self._run_pg_tool(
                    "pg_restore",
                    self._pg_conn_args()
                    + ["--clean", "--if-exists", "--no-owner", "--no-privileges",
                       "-d", self.config.pg_database, str(dec_path)],
                    timeout=float(self.config.backup_dump_timeout) * 2,
                )

                # 5. post-validation (missing tables are fatal, count diffs warn)
                detail["validation"] = await self._validate_restore(manifest_local)

                # 6. hot-reload the application onto the restored DB
                if app is not None:
                    try:
                        from app.main import reload_rag_application
                        await reload_rag_application(app)
                        detail["hot_reload"] = "done"
                    except Exception as exc:
                        logger.warning("Post-restore hot reload failed: %s", exc)
                        detail["hot_reload"] = f"failed: {exc}"

            except BackupServiceError as exc:
                error = str(exc)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Restore job %s failed unexpectedly", job_id)
                error = f"unexpected error: {exc}"
            finally:
                self._set_maintenance(False)
                for p in cleanup:
                    p.unlink(missing_ok=True)

        duration_ms = int((time.perf_counter() - started) * 1000)
        status = "error" if error else "success"
        await self.job_store.finish_job(
            job_id, status,
            filename=filename, duration_ms=duration_ms,
            error=error or (json.dumps(detail) if detail else None),
        )
        if error:
            raise BackupServiceError(error)
        return {"job_id": job_id, "filename": filename, **detail}

    @staticmethod
    def _manifest_hash(path: Path, key: str) -> Optional[str]:
        try:
            return json.loads(path.read_text()).get(key)
        except Exception:
            return None

    async def _pre_restore_dump(self, cfg: Dict[str, Any], remote_dir: str) -> str:
        """Mandatory safety dump before restore. Uploaded under pre-restore/."""
        tmp = self._tmp_dir()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"pre-restore-{self._instance_name()}-{ts}.dump"
        dump_path = tmp / name
        enc_path = tmp / f"{name}.enc"
        try:
            await self._dump_database(dump_path)
            encrypt_file(dump_path, enc_path, load_encryption_key())
            remote = f"{remote_dir}/pre-restore/{enc_path.name}"
            try:
                async with self._client(cfg) as client:
                    await client.ensure_collection(f"{remote_dir}/pre-restore")
                    await client.upload(remote, enc_path)
                return remote
            except Exception as exc:
                # keep local copy as fallback - it is inside tmp and will be
                # removed by the caller cleanup, so copy it to a stable place
                keep = tmp.parent / enc_path.name
                shutil.copy2(enc_path, keep)
                logger.warning(
                    "Pre-restore dump upload failed (%s); local copy kept at %s",
                    exc, keep,
                )
                return f"local:{keep}"
        finally:
            dump_path.unlink(missing_ok=True)
            enc_path.unlink(missing_ok=True)

    async def _terminate_connections(self) -> None:
        """Terminate other sessions on the target database.

        Requires the pg_signal_backend privilege (PG14+) or ownership;
        aborts the restore with a clear error when the privilege is missing.
        """
        async with self.engine.begin() as conn:
            res = await conn.execute(
                text(
                    """
                    SELECT pid FROM pg_stat_activity
                    WHERE datname = current_database() AND pid <> pg_backend_pid()
                    """
                )
            )
            pids = [r[0] for r in res.fetchall()]
            if not pids:
                return
            try:
                res = await conn.execute(
                    text(
                        """
                        SELECT pid, pg_terminate_backend(pid) FROM pg_stat_activity
                        WHERE datname = current_database() AND pid <> pg_backend_pid()
                        """
                    )
                )
                terminated = [r[0] for r in res.fetchall() if r[1]]
                logger.info("Terminated %d DB connection(s) for restore", len(terminated))
            except Exception as exc:
                raise BackupServiceError(
                    "Insufficient privilege to terminate active DB connections "
                    f"(pg_signal_backend required): {exc}"
                )

    async def _validate_restore(self, manifest_path: Path) -> Dict[str, Any]:
        """Compare post-restore table counts against the backup manifest."""
        result: Dict[str, Any] = {"ok": True, "missing_tables": [], "count_diffs": {}}
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            result["note"] = "manifest unavailable - skipped count validation"
            return result
        expected = manifest.get("row_counts") or {}
        if not expected:
            result["note"] = "manifest has no row_counts"
            return result
        actual = await self._table_row_counts()
        for table, exp in expected.items():
            if table not in actual:
                result["missing_tables"].append(table)
                result["ok"] = False
            elif actual[table] != exp:
                result["count_diffs"][table] = {"expected": exp, "actual": actual[table]}
        if result["count_diffs"]:
            # runtime tables legitimately change (job rows, metrics) - warn only
            result["note"] = "row count differences present (runtime tables may legitimately differ)"
        if not result["ok"]:
            raise BackupServiceError(
                f"Restore validation failed: missing tables {result['missing_tables']}"
            )
        return result
