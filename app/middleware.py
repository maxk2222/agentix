"""FastAPI middlewares for the cybersecurity RAG assistant.

Provides API key enforcement and structured request/response logging.
All middlewares are compatible with OpenAI-style clients and run on the
ASGI stack via ``BaseHTTPMiddleware``.
"""
from __future__ import annotations

import hmac
import logging
import time
from typing import Any, AsyncGenerator, Callable, Optional

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.auth import get_serializer, read_session_cookie
from app.orjson_response import ORJSONResponse
from app.logging_utils import colorize_process_time
from app.metrics import BlockedRequestRecord, BlockedRequestStore

logger = logging.getLogger(__name__)


class SessionRefreshMiddleware(BaseHTTPMiddleware):
    """Sliding renewal for the dashboard session cookie.

    The signed cookie normally dies exactly ``dashboard_session_ttl_seconds``
    after login — even while a long agent run is still streaming. Whenever a
    request carries a valid session older than half the TTL we re-sign the
    payload and attach a fresh ``Set-Cookie`` so the TTL counts from the last
    activity instead of the login. Login responses and requests that set the
    cookie themselves are skipped so the fresh login cookie is never clobbered.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        config = getattr(request.app.state, "app_config", None)
        raw = request.cookies.get("dashboard_session") if config else None
        secret = getattr(config, "dashboard_session_secret", "") if config else ""
        refresh_cookie: Optional[str] = None
        if raw and secret:
            try:
                serializer = get_serializer(secret)
                ttl = int(getattr(config, "dashboard_session_ttl_seconds", 3600) or 3600)
                payload, issued_at = serializer.loads(raw, max_age=ttl, return_timestamp=True)
                age = time.time() - issued_at.timestamp()
                if isinstance(payload, dict) and age > ttl / 2:
                    refresh_cookie = serializer.dumps(payload)
            except Exception:
                refresh_cookie = None

        response = await call_next(request)

        if refresh_cookie is not None and not request.url.path.startswith("/admin/auth"):
            existing = " ".join(response.headers.getlist("set-cookie"))
            if "dashboard_session" not in existing:
                ttl = int(getattr(config, "dashboard_session_ttl_seconds", 3600) or 3600)
                parts = [
                    f"dashboard_session={refresh_cookie}",
                    "Path=/",
                    "HttpOnly",
                    "SameSite=Strict",
                    f"Max-Age={ttl}",
                ]
                if getattr(config, "_is_https", False):
                    parts.append("Secure")
                response.headers.append("Set-Cookie", "; ".join(parts))
        return response


def _has_admin_session(request: Request) -> bool:
    """Check if the request carries a valid dashboard admin session cookie."""
    config = getattr(request.app.state, "app_config", None)
    if config is None:
        return False
    try:
        session = read_session_cookie(request, config)
        return session is not None and session.get("role") == "admin"
    except Exception:
        return False


def _has_dashboard_session(request: Request) -> bool:
    """Check for any valid dashboard session (including owner accounts)."""
    config = getattr(request.app.state, "app_config", None)
    if config is None:
        return False
    try:
        return read_session_cookie(request, config) is not None
    except Exception:
        return False


def _reason_for_status(status: int) -> str:
    if status == 401:
        return "Invalid API key"
    if status == 403:
        return "Forbidden"
    if status == 413:
        return "Request body too large"
    if status == 415:
        return "Unsupported media type"
    if status == 422:
        return "Validation failed"
    if status == 429:
        return "Rate limited"
    if status == 500:
        return "Internal server error"
    if status == 502:
        return "Upstream LLM error"
    if status == 503:
        return "Service unavailable"
    return "Blocked"


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Enforce API keys from the AppApiKeyStore with optional legacy fallback."""

    def __init__(self, app: FastAPI, legacy_key: Optional[str] = None) -> None:
        super().__init__(app)
        self.legacy_key = (legacy_key or "").strip()
        self._excluded_paths = {"/health"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Always exempt health, dashboard, admin, widget, public-chat, and
        # branding routes (admin has its own auth; widget/public-chat/branding
        # are public endpoints with their own session handling where needed).
        if (
            path in self._excluded_paths
            or path.startswith("/dashboard")
            or path.startswith("/admin")
            or path.startswith("/widget")
            or path.startswith("/public-chat")
            or path.startswith("/branding")
        ):
            return await call_next(request)

        # Check if API keys are globally disabled via runtime config
        api_keys_enabled = True
        app_config = getattr(request.app.state, "app_config", None)
        if app_config is not None:
            api_keys_enabled = bool(getattr(app_config, "api_keys_enabled", True))
            # Runtime config override
            runtime_store = getattr(request.app.state, "_runtime_config_store", None)
            if runtime_store is not None:
                try:
                    stored = await runtime_store.get("api_keys_enabled")
                    if stored is not None:
                        api_keys_enabled = bool(stored)
                except Exception:
                    pass
        if not api_keys_enabled:
            return await call_next(request)

        provided = request.headers.get("x-api-key")
        if not provided:
            auth_header = request.headers.get("authorization", "").strip()
            if auth_header.lower().startswith("bearer "):
                provided = auth_header[7:].strip()

        # If no key is provided, check if any keys are configured at all.
        # If not, run in open mode (no API key required).
        key_store: Optional[Any] = getattr(request.app.state, "_api_key_store", None)
        has_configured_keys = False
        if key_store is not None:
            from app.api_keys import AppApiKeyStore
            if isinstance(key_store, AppApiKeyStore):
                keys = await key_store.list_keys()
                has_configured_keys = any(k.enabled for k in keys) or bool(self.legacy_key)

        if not provided:
            # Allow admin session to bypass API key check (e.g. dashboard widget)
            if _has_admin_session(request):
                return await call_next(request)
            # The chat endpoint itself accepts authenticated dashboard
            # sessions, including the legacy `owner` role. Keep that contract
            # consistent for the dashboard widget preview without bypassing
            # API-key enforcement on unrelated API routes.
            if path == "/v1/chat/completions" and _has_dashboard_session(request):
                return await call_next(request)
            if not has_configured_keys:
                return await call_next(request)
            return await self._reject(request, "Missing API key")

        # Try modern store first
        if key_store is not None:
            from app.api_keys import AppApiKeyStore
            if isinstance(key_store, AppApiKeyStore):
                key_record = await key_store.verify_key(provided)
                if key_record is not None:
                    request.state.api_key_id = key_record.id
                    request.state.allowed_models = key_record.allowed_models or []
                    from app.rate_limit import check_rate_limit

                    limited = await check_rate_limit(
                        key_record.id, key_record.rate_limit_rpm, key_record.rate_limit_rpd
                    )
                    if limited is not None:
                        retry_after, _limit, _reset = limited
                        return await self._reject(
                            request, "Rate limit exceeded", status_code=429, retry_after=retry_after
                        )
                    return await call_next(request)

        # Fallback to legacy single key
        if self.legacy_key and hmac.compare_digest(str(provided), str(self.legacy_key)):
            request.state.api_key_id = "legacy"
            request.state.allowed_models = []
            return await call_next(request)

        return await self._reject(request, "Invalid API key")

    async def _reject(
        self, request: Request, reason: str, status_code: int = 401, retry_after: Optional[int] = None
    ) -> Response:
        logger.warning("Rejected request | path=%s | reason=%s", request.url.path, reason)
        store: Optional[BlockedRequestStore] = getattr(
            request.app.state, "_blocked_store", None
        )
        if store is not None:
            try:
                await store.save(
                    BlockedRequestRecord(
                        path=request.url.path,
                        method=request.method,
                        status_code=status_code,
                        reason=reason,
                    )
                )
            except Exception:
                logger.exception("Failed to save blocked request for %s", request.url.path)
        headers = {"Retry-After": str(retry_after)} if retry_after else {}
        return ORJSONResponse(
            status_code=status_code,
            content={"detail": reason, "retry_after": retry_after} if retry_after else {"detail": reason},
            headers=headers,
        )


class LoggingMiddleware(BaseHTTPMiddleware):
    """Record structured request/response metadata, including SSE statistics."""

    def __init__(self, app: FastAPI, app_config: Any) -> None:
        super().__init__(app)
        self.app_config = app_config

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Duplicate the request body, invoke the next handler, and log the outcome."""

        # Mask sensitive headers for logs
        def _mask_headers(headers: dict) -> dict:
            out = {}
            for k, v in headers.items():
                lk = k.lower()
                if lk in ("authorization", "x-api-key") and v:
                    try:
                        out[k] = v[:4] + "..." + v[-4:]
                    except Exception:
                        out[k] = "***"
                else:
                    out[k] = v
            return out

        logger.info("Request: %s %s", request.method, request.url.path)
        req_body_bytes = await request.body()
        if req_body_bytes and self.app_config.debug_logging_enabled:
            logger.debug(
                "Request Body: %s",
                req_body_bytes.decode("utf-8", errors="ignore"),
            )
        if self.app_config.debug_logging_enabled:
            try:
                headers = dict(request.headers)
                logger.debug("Request Headers: %s", _mask_headers(headers))
            except Exception:
                pass

        async def receive() -> dict:
            return {"type": "http.request", "body": req_body_bytes}

        request_with_body = Request(request.scope, receive)
        start_time = time.perf_counter()
        response = await call_next(request_with_body)

        async def body_logger(
            iterator: AsyncGenerator[bytes, None],
        ) -> AsyncGenerator[bytes, None]:
            """Count response chunks and reconstruct non-streaming bodies for logging."""
            content_type = response.headers.get("content-type", "").lower()
            is_streaming = "text/event-stream" in content_type
            body_chunks: list[bytes] = []
            chunk_count = 0
            total_bytes = 0

            async for chunk in iterator:
                chunk_bytes = chunk or b""
                chunk_count += 1
                total_bytes += len(chunk_bytes)

                if not is_streaming:
                    body_chunks.append(chunk_bytes)

                yield chunk

            process_time = (time.perf_counter() - start_time) * 1000
            colored_process_time = colorize_process_time(process_time)
            if is_streaming:
                logger.info(
                    "Response: %s | Process Time: %s | Stream Chunks: %s | Bytes: %s",
                    response.status_code,
                    colored_process_time,
                    chunk_count,
                    total_bytes,
                    extra={
                        "response_status": response.status_code,
                        "process_time": process_time,
                    },
                )
            else:
                full_body_bytes = b"".join(body_chunks)
                logger.info(
                    "Response: %s | Process Time: %s | Bytes: %s",
                    response.status_code,
                    colored_process_time,
                    len(full_body_bytes),
                    extra={
                        "response_status": response.status_code,
                        "process_time": process_time,
                    },
                )
                if self.app_config.debug_logging_enabled:
                    logger.debug(
                        "Response Body: %s",
                        full_body_bytes.decode("utf-8", errors="ignore"),
                    )

        response.body_iterator = body_logger(response.body_iterator)
        return response
