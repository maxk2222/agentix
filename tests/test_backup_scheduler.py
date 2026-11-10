"""Tests for app.backup_scheduler."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.backup_scheduler import BackupScheduler, compute_next_run


def test_compute_next_run_interval():
    cfg = {"schedule_type": "interval", "schedule_minutes": 60}
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    nxt = compute_next_run(cfg, base=base)
    assert nxt == base + timedelta(hours=1)


def test_compute_next_run_cron():
    cfg = {
        "schedule_type": "cron", "schedule_cron": "0 3 * * *",
        "timezone": "UTC",
    }
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    nxt = compute_next_run(cfg, base=base)
    assert nxt.hour == 3 and nxt > base


def test_compute_next_run_cron_timezone():
    cfg = {
        "schedule_type": "cron", "schedule_cron": "0 3 * * *",
        "timezone": "Europe/Berlin",
    }
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    nxt = compute_next_run(cfg, base=base)
    # 03:00 Europe/Berlin = 01:00/02:00 UTC depending on DST
    assert nxt.tzinfo == timezone.utc
    assert nxt.hour in (1, 2)


def test_compute_next_run_invalid_cron():
    cfg = {"schedule_type": "cron", "schedule_cron": "not a cron"}
    assert compute_next_run(cfg) is None


@pytest.mark.asyncio
async def test_check_and_run_executes_due_backup():
    store = MagicMock()
    store.is_due = AsyncMock(return_value=True)
    store.get_config = AsyncMock(return_value={
        "schedule_type": "interval", "schedule_minutes": 60,
    })
    store.update_next_run = AsyncMock()
    store.update_last_run = AsyncMock()
    service = MagicMock()
    service.in_maintenance = False
    service.run_backup = AsyncMock(return_value={"filename": "x.enc"})

    sched = BackupScheduler(store, service)
    await sched.check_and_run()
    service.run_backup.assert_awaited_once()
    store.update_next_run.assert_awaited_once()
    store.update_last_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_and_run_skips_when_not_due():
    store = MagicMock()
    store.is_due = AsyncMock(return_value=False)
    service = MagicMock()
    service.run_backup = AsyncMock()
    sched = BackupScheduler(store, service)
    await sched.check_and_run()
    service.run_backup.assert_not_called()


@pytest.mark.asyncio
async def test_check_and_run_skips_in_maintenance():
    store = MagicMock()
    store.is_due = AsyncMock(return_value=True)
    service = MagicMock()
    service.in_maintenance = True
    service.run_backup = AsyncMock()
    sched = BackupScheduler(store, service)
    await sched.check_and_run()
    service.run_backup.assert_not_called()
    store.is_due.assert_not_called()


@pytest.mark.asyncio
async def test_check_and_run_continues_after_failure():
    store = MagicMock()
    store.is_due = AsyncMock(return_value=True)
    store.get_config = AsyncMock(return_value={
        "schedule_type": "interval", "schedule_minutes": 60,
    })
    store.update_next_run = AsyncMock()
    store.update_last_run = AsyncMock()
    service = MagicMock()
    service.in_maintenance = False
    service.run_backup = AsyncMock(side_effect=RuntimeError("boom"))

    sched = BackupScheduler(store, service)
    await sched.check_and_run()  # must not raise
    service.run_backup.assert_awaited_once()
    store.update_last_run.assert_awaited_once()
