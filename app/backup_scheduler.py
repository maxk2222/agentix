"""Async scheduler for automated database backups.

Mirrors :class:`app.triggers_scheduler.TriggersScheduler`: a lightweight
30-second loop consults ``agentix_backup_config.next_run_at`` (persisted in
Postgres, so schedules survive restarts) and executes due backup jobs via
:class:`app.backup_service.BackupService`.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from croniter import croniter

from app.backup_store import BackupConfigStore

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 30.0


def compute_next_run(cfg: Dict[str, Any], base: Optional[datetime] = None) -> Optional[datetime]:
    """Compute the next due timestamp (UTC) for a backup config row."""
    base = base or datetime.now(timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)

    schedule_type = cfg.get("schedule_type") or "interval"
    if schedule_type == "cron":
        expr = (cfg.get("schedule_cron") or "").strip()
        if not expr:
            return None
        try:
            tz = ZoneInfo(cfg.get("timezone") or "UTC")
        except Exception:
            tz = ZoneInfo("UTC")
        try:
            itr = croniter(expr, base.astimezone(tz))
            return itr.get_next(datetime).astimezone(timezone.utc)
        except Exception:
            logger.warning("Invalid backup cron expression: %s", expr)
            return None

    # interval (minutes)
    minutes = int(cfg.get("schedule_minutes") or 1440)
    if minutes <= 0:
        minutes = 1440
    return base + timedelta(minutes=minutes)


class BackupScheduler:
    """Periodically checks the backup config and runs due backup jobs."""

    def __init__(self, config_store: BackupConfigStore, service: Any) -> None:
        self.config_store = config_store
        self.service = service
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return
        logger.info("Starting backup scheduler loop.")
        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        logger.info("Stopping backup scheduler loop.")
        self._stop_event.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.check_and_run()
            except Exception:
                logger.exception("Backup scheduler check_and_run failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=CHECK_INTERVAL_SECONDS
                )
            except asyncio.TimeoutError:
                pass

    async def check_and_run(self) -> None:
        """Run a scheduled backup when the config marks one as due."""
        if getattr(self.service, "in_maintenance", False):
            logger.debug("Skipping backup check: maintenance mode active")
            return
        from app.redis_locks import acquire_leader

        if not await acquire_leader("backup"):
            return
        if not await self.config_store.is_due():
            return

        cfg = await self.config_store.get_config()
        # Push next_run_at forward BEFORE running so a crash or slow job does
        # not cause immediate re-runs on the next tick.
        await self.config_store.update_next_run(compute_next_run(cfg))

        try:
            result = await self.service.run_backup(kind="scheduled", triggered_by="scheduler")
            logger.info(
                "Scheduled backup finished: %s (%s bytes)",
                result.get("filename"), result.get("size_bytes"),
            )
        except Exception as exc:
            logger.error("Scheduled backup failed: %s", exc)
        await self.config_store.update_last_run()
