"""Admin endpoints for database backup & restore (/admin/backup).

Manages the WebDAV/Sciebo backup target, triggers manual backups, lists job
history and remote archives, verifies archive integrity and runs restores
with an explicit typed confirmation.
"""

from __future__ import annotations

import shutil
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.auth import require_permission
from app.backup_crypto import BackupEncryptionKeyMissing, encryption_configured
from app.backup_scheduler import compute_next_run
from app.backup_service import BackupService, BackupServiceError
from app.backup_store import BackupConfigStore, BackupJobStore

router = APIRouter(prefix="/admin/backup", tags=["Admin Backup"])


# ------------------------------------------------------------------------------
# Payloads
# ------------------------------------------------------------------------------
class BackupConfigPayload(BaseModel):
    enabled: Optional[bool] = None
    schedule_type: Optional[str] = None  # "interval" | "cron"
    schedule_minutes: Optional[int] = None
    schedule_cron: Optional[str] = None
    timezone: Optional[str] = None
    webdav_url: Optional[str] = None
    remote_path: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None  # None = keep stored password
    retention_count: Optional[int] = None
    retention_days: Optional[int] = None


class BackupTestPayload(BaseModel):
    webdav_url: str
    remote_path: str = "/agentix-backups"
    username: str
    password: Optional[str] = None  # falls back to stored password


class RestorePayload(BaseModel):
    filename: str
    confirm: str


# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------
def _get_service(request: Request) -> BackupService:
    service = getattr(request.app.state, "_backup_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Backup service unavailable")
    return service


def _get_config_store(request: Request) -> BackupConfigStore:
    store = getattr(request.app.state, "_backup_config_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Backup store unavailable")
    return store


def _get_job_store(request: Request) -> BackupJobStore:
    store = getattr(request.app.state, "_backup_job_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Backup store unavailable")
    return store


def _map_error(exc: BackupServiceError) -> HTTPException:
    msg = str(exc)
    lowered = msg.lower()
    if "not found in path" in lowered or "encryption_key" in lowered or "not set" in lowered:
        return HTTPException(status_code=503, detail=msg)
    if "already running" in lowered or "paused" in lowered or "maintenance" in lowered:
        return HTTPException(status_code=409, detail=msg)
    return HTTPException(status_code=400, detail=msg)


# ------------------------------------------------------------------------------
# config
# ------------------------------------------------------------------------------
@router.get("/config")
async def backup_get_config(request: Request) -> Dict[str, Any]:
    await require_permission(request, "backup:read")
    store = _get_config_store(request)
    cfg = await store.get_config()
    cfg["encryption_configured"] = encryption_configured()
    cfg["pg_dump_available"] = shutil.which("pg_dump") is not None
    return cfg


@router.put("/config")
async def backup_update_config(request: Request, payload: BackupConfigPayload) -> Dict[str, Any]:
    await require_permission(request, "backup:write")
    store = _get_config_store(request)

    if payload.schedule_type is not None and payload.schedule_type not in ("interval", "cron"):
        raise HTTPException(status_code=400, detail="schedule_type must be 'interval' or 'cron'")
    if payload.webdav_url is not None and not payload.webdav_url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Only HTTPS WebDAV URLs are allowed")
    if payload.remote_path is not None:
        rp = payload.remote_path
        if not rp.startswith("/") or ".." in rp.split("/"):
            raise HTTPException(status_code=400, detail="Invalid remote_path")
    if payload.schedule_cron is not None and payload.schedule_cron.strip():
        from croniter import croniter
        from datetime import datetime, timezone
        try:
            croniter(payload.schedule_cron.strip(), datetime.now(timezone.utc))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cron expression")

    merged = await store.get_config()
    merged.update({k: v for k, v in payload.model_dump().items() if v is not None})
    next_run = compute_next_run(merged) if merged.get("enabled") else None

    try:
        cfg = await store.save_config(
            enabled=payload.enabled,
            schedule_type=payload.schedule_type,
            schedule_minutes=payload.schedule_minutes,
            schedule_cron=(payload.schedule_cron.strip() if payload.schedule_cron else None),
            timezone_name=payload.timezone,
            webdav_url=payload.webdav_url,
            remote_path=payload.remote_path,
            username=payload.username,
            password=payload.password if payload.password else None,
            retention_count=payload.retention_count,
            retention_days=payload.retention_days,
            next_run_at=next_run,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {exc}")
    if not cfg.get("enabled") and cfg.get("next_run_at") is not None:
        await store.update_next_run(None)
        cfg["next_run_at"] = None
    cfg["encryption_configured"] = encryption_configured()
    cfg["pg_dump_available"] = shutil.which("pg_dump") is not None
    return cfg


@router.post("/test")
async def backup_test_connection(request: Request, payload: BackupTestPayload) -> Dict[str, Any]:
    await require_permission(request, "backup:write")
    service = _get_service(request)
    password = payload.password
    if not password:
        cfg = await _get_config_store(request).get_config_internal()
        password = (cfg or {}).get("password") or ""
        if not password:
            raise HTTPException(
                status_code=400,
                detail="No stored password - provide one or save the config first",
            )
    try:
        return await service.test_connection(
            payload.webdav_url, payload.username, password, payload.remote_path
        )
    except BackupServiceError as exc:
        raise _map_error(exc)


# ------------------------------------------------------------------------------
# jobs
# ------------------------------------------------------------------------------
@router.post("/run")
async def backup_run_now(request: Request) -> Dict[str, Any]:
    await require_permission(request, "backup:write")
    service = _get_service(request)
    try:
        return await service.run_backup(kind="manual", triggered_by="dashboard")
    except BackupEncryptionKeyMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except BackupServiceError as exc:
        raise _map_error(exc)


@router.post("/cancel")
async def backup_cancel(request: Request) -> Dict[str, Any]:
    await require_permission(request, "backup:write")
    service = _get_service(request)
    service.cancel()
    return {"success": True}


@router.get("/jobs")
async def backup_list_jobs(request: Request, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    await require_permission(request, "backup:read")
    store = _get_job_store(request)
    limit = max(1, min(limit, 200))
    jobs = await store.list_jobs(limit=limit, offset=offset)
    summary = await store.summary()
    return {"jobs": jobs, "summary": summary}


@router.get("/jobs/{job_id}")
async def backup_get_job(request: Request, job_id: int) -> Dict[str, Any]:
    await require_permission(request, "backup:read")
    store = _get_job_store(request)
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ------------------------------------------------------------------------------
# remote archive management
# ------------------------------------------------------------------------------
@router.get("/remote")
async def backup_list_remote(request: Request) -> Dict[str, Any]:
    await require_permission(request, "backup:read")
    service = _get_service(request)
    try:
        files = await service.list_remote()
    except BackupServiceError as exc:
        raise _map_error(exc)
    return {"files": files}


@router.post("/verify/{filename}")
async def backup_verify(request: Request, filename: str) -> Dict[str, Any]:
    await require_permission(request, "backup:read")
    service = _get_service(request)
    try:
        return await service.verify_remote(filename)
    except BackupEncryptionKeyMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except BackupServiceError as exc:
        raise _map_error(exc)


@router.delete("/remote/{filename}")
async def backup_delete_remote(request: Request, filename: str) -> Dict[str, Any]:
    await require_permission(request, "backup:write")
    service = _get_service(request)
    try:
        removed = await service.delete_remote(filename)
    except BackupServiceError as exc:
        raise _map_error(exc)
    if not removed:
        raise HTTPException(status_code=404, detail="Backup not found on remote")
    return {"success": True}


# ------------------------------------------------------------------------------
# restore
# ------------------------------------------------------------------------------
@router.post("/restore")
async def backup_restore(request: Request, payload: RestorePayload) -> Dict[str, Any]:
    await require_permission(request, "backup:restore")
    service = _get_service(request)
    if payload.confirm != payload.filename:
        raise HTTPException(
            status_code=400,
            detail="Confirmation mismatch: 'confirm' must equal the backup filename",
        )
    try:
        return await service.restore_backup(payload.filename, app=request.app)
    except BackupEncryptionKeyMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except BackupServiceError as exc:
        raise _map_error(exc)
