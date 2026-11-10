"""Tests for app.backup_service (pipeline orchestration, mocked internals)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.backup_service import BackupService, BackupServiceError


def _service(tmp_path: Path, cfg_row: Dict[str, Any] | None = None) -> BackupService:
    config = SimpleNamespace(
        pg_host="h", pg_port=5432, pg_user="u", pg_password="p",
        pg_database="db", backup_tmp_dir=tmp_path,
        backup_dump_timeout=10, backup_upload_timeout=10,
        backup_upload_retries=1, backup_instance="test",
    )
    config_store = MagicMock()
    config_store.get_config_internal = AsyncMock(return_value=cfg_row)
    config_store.get_config = AsyncMock(return_value=cfg_row or {})
    config_store.update_next_run = AsyncMock()
    config_store.update_last_run = AsyncMock()
    job_store = MagicMock()
    job_store.create_job = AsyncMock(return_value=1)
    job_store.finish_job = AsyncMock()
    return BackupService(
        engine=MagicMock(), config=config,
        config_store=config_store, job_store=job_store,
    )


CFG = {
    "webdav_url": "https://sciebo.de/dav",
    "remote_path": "/agentix-backups",
    "username": "u", "password": "p",
    "retention_count": 5, "retention_days": None,
}


@pytest.mark.asyncio
async def test_run_backup_fails_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("BACKUP_ENCRYPTION_KEY", raising=False)
    svc = _service(tmp_path, CFG)
    with pytest.raises(BackupServiceError):
        await svc.run_backup()


@pytest.mark.asyncio
async def test_run_backup_fails_without_config(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", os.urandom(32).hex())
    svc = _service(tmp_path, cfg_row=None)
    with pytest.raises(BackupServiceError, match="not configured"):
        await svc.run_backup()


@pytest.mark.asyncio
async def test_run_backup_rejects_http(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", os.urandom(32).hex())
    bad = dict(CFG, webdav_url="http://insecure.example")
    svc = _service(tmp_path, bad)
    with pytest.raises(BackupServiceError, match="HTTPS"):
        await svc.run_backup()


@pytest.mark.asyncio
async def test_run_backup_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", os.urandom(32).hex())
    svc = _service(tmp_path, CFG)

    async def fake_dump(dest: Path):
        dest.write_bytes(b"PGDUMP-fake-content")

    svc._dump_database = fake_dump
    svc._check_disk_space = AsyncMock()
    svc._pg_dump_version = AsyncMock(return_value="pg_dump 16")
    svc._server_version = AsyncMock(return_value="16.0")
    svc._table_row_counts = AsyncMock(return_value={"t": 1})
    svc._upload_with_retry = AsyncMock()
    svc._apply_retention = AsyncMock(return_value=0)

    client = MagicMock()
    client.stat = AsyncMock(return_value={"content_length": None})
    # remote size must match the produced .enc file; patch stat lazily
    async def fake_client_ctx(*a, **k):
        return client

    class FakeClient:
        async def __aenter__(self):
            return client

        async def __aexit__(self, *a):
            return None

    monkeypatch.setattr("app.backup_service.WebDAVClient", lambda *a, **k: FakeClient())

    # stat must report the encrypted size - compute lazily
    async def stat_side_effect(path):
        # find the enc file in tmp
        encs = list(tmp_path.glob("*.dump.enc"))
        return {"content_length": encs[0].stat().st_size if encs else -1}
    client.stat = AsyncMock(side_effect=stat_side_effect)

    result = await svc.run_backup(kind="manual", triggered_by="test")
    assert result["filename"].startswith("agentix-backup-test-")
    assert result["filename"].endswith(".dump.enc")
    svc._upload_with_retry.assert_awaited_once()
    svc.job_store.finish_job.assert_awaited_once()
    args, kwargs = svc.job_store.finish_job.call_args
    assert args[1] == "success"


@pytest.mark.asyncio
async def test_concurrent_backup_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", os.urandom(32).hex())
    svc = _service(tmp_path, CFG)
    svc._lock = MagicMock()
    svc._lock.locked = MagicMock(return_value=True)
    with pytest.raises(BackupServiceError, match="already running"):
        await svc.run_backup()


@pytest.mark.asyncio
async def test_remote_path_validation(tmp_path):
    svc = _service(tmp_path, CFG)
    assert svc._check_remote_path("/backups/") == "/backups"
    with pytest.raises(BackupServiceError):
        svc._check_remote_path("relative/path")
    with pytest.raises(BackupServiceError):
        svc._check_remote_path("/../escape")


@pytest.mark.asyncio
async def test_restore_rejects_bad_filename(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", os.urandom(32).hex())
    svc = _service(tmp_path, CFG)
    with pytest.raises(BackupServiceError, match="Invalid backup filename"):
        await svc.restore_backup("../../etc/passwd")
    with pytest.raises(BackupServiceError, match="Invalid backup filename"):
        await svc.restore_backup("plain.sql")


@pytest.mark.asyncio
async def test_maintenance_flag_blocks_backup(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", os.urandom(32).hex())
    svc = _service(tmp_path, CFG)
    svc._set_maintenance(True)
    assert svc.in_maintenance is True
    with pytest.raises(BackupServiceError, match="paused"):
        await svc.run_backup()
    svc._set_maintenance(False)
    assert svc.in_maintenance is False
