"""Tests for the /admin/backup endpoints."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.routers.backup import (
    BackupConfigPayload,
    RestorePayload,
    backup_get_config,
    backup_list_jobs,
    backup_restore,
    backup_run_now,
    backup_update_config,
)


class FakeService:
    def __init__(self):
        self.cancelled = False

    async def run_backup(self, kind, triggered_by):
        return {"filename": "agentix-backup-x.dump.enc"}

    async def restore_backup(self, filename, app=None):
        return {"job_id": 5, "filename": filename}


def _make_request(service=None, config_store=None, job_store=None):
    state = {}
    if service is not None:
        state["_backup_service"] = service
    if config_store is not None:
        state["_backup_config_store"] = config_store
    if job_store is not None:
        state["_backup_job_store"] = job_store
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(**state)),
        cookies={},
    )


@pytest.fixture(autouse=True)
def mock_require_permission():
    with patch("app.routers.backup.require_permission", new_callable=AsyncMock) as m:
        yield m


@pytest.mark.asyncio
async def test_get_config_shape():
    store = MagicMock()
    store.get_config = AsyncMock(return_value={
        "enabled": False, "remote_path": "/agentix-backups",
    })
    req = _make_request(config_store=store)
    cfg = await backup_get_config(req)
    assert "encryption_configured" in cfg
    assert "pg_dump_available" in cfg


@pytest.mark.asyncio
async def test_update_config_rejects_http_url():
    store = MagicMock()
    req = _make_request(config_store=store)
    payload = BackupConfigPayload(webdav_url="http://insecure.example")
    with pytest.raises(Exception) as exc_info:
        await backup_update_config(req, payload)
    assert "400" in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_config_rejects_bad_remote_path():
    store = MagicMock()
    req = _make_request(config_store=store)
    payload = BackupConfigPayload(remote_path="/../escape")
    with pytest.raises(Exception) as exc_info:
        await backup_update_config(req, payload)
    assert "400" in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_config_rejects_bad_cron():
    store = MagicMock()
    req = _make_request(config_store=store)
    payload = BackupConfigPayload(schedule_cron="not a cron")
    with pytest.raises(Exception) as exc_info:
        await backup_update_config(req, payload)
    assert "400" in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_config_saves_and_schedules():
    store = MagicMock()
    store.get_config = AsyncMock(return_value={"enabled": False})
    store.save_config = AsyncMock(return_value={"enabled": True})
    req = _make_request(config_store=store)
    payload = BackupConfigPayload(
        enabled=True, schedule_type="interval", schedule_minutes=60
    )
    cfg = await backup_update_config(req, payload)
    kwargs = store.save_config.call_args.kwargs
    assert kwargs["enabled"] is True
    assert kwargs["next_run_at"] is not None  # schedule computed on enable


@pytest.mark.asyncio
async def test_run_now_invokes_service():
    req = _make_request(service=FakeService())
    result = await backup_run_now(req)
    assert result["filename"].endswith(".enc")


@pytest.mark.asyncio
async def test_restore_requires_matching_confirm():
    req = _make_request(service=FakeService())
    payload = RestorePayload(filename="a.enc", confirm="b.enc")
    with pytest.raises(Exception) as exc_info:
        await backup_restore(req, payload)
    assert "400" in str(exc_info.value)


@pytest.mark.asyncio
async def test_restore_runs_with_matching_confirm():
    svc = FakeService()
    req = _make_request(service=svc)
    payload = RestorePayload(filename="a.enc", confirm="a.enc")
    result = await backup_restore(req, payload)
    assert result["filename"] == "a.enc"


@pytest.mark.asyncio
async def test_list_jobs():
    store = MagicMock()
    store.list_jobs = AsyncMock(return_value=[{"id": 1}])
    store.summary = AsyncMock(return_value={"total_jobs": 1})
    req = _make_request(job_store=store)
    out = await backup_list_jobs(req)
    assert len(out["jobs"]) == 1
