"""Webhook POST builtin tool."""

from __future__ import annotations

import fnmatch
import ipaddress
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import aiohttp

from .common import get_headers

logger = logging.getLogger(__name__)


class WebhookPostTool:
    """POST a JSON payload to an HTTPS endpoint."""

    name = "webhook_post"
    description = (
        "Send a JSON payload to an HTTPS webhook URL. "
        "Use ONLY 'url' (required), 'payload' (JSON object, default {}), "
        "'headers' (optional dict of extra headers), and 'timeout' (seconds, default 30, max 60)."
    )

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The target HTTPS URL. Must start with https://",
                        },
                        "payload": {
                            "type": "object",
                            "description": "JSON payload to send. Default is empty object.",
                            "default": {},
                        },
                        "headers": {
                            "type": "object",
                            "description": "Optional additional HTTP headers.",
                            "default": {},
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Request timeout in seconds (default 30, max 60).",
                            "default": 30,
                        },
                    },
                    "required": ["url"],
                },
            },
        }

    async def execute(
        self,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not url.startswith("https://"):
            raise ValueError("URL must start with https:// (http:// is not allowed for security)")

        cfg = (context or {}).get("tool_config") or {}
        # urlparse strips the port AND []-brackets — string splitting would
        # turn "https://[::1]/" into "[" and silently bypass both checks.
        hostname = (urlparse(url).hostname or "").lower()

        allowed = [str(p).strip().lower() for p in (cfg.get("allowed_hosts") or []) if str(p).strip()]
        if allowed and not any(fnmatch.fnmatch(hostname, pat) for pat in allowed):
            raise ValueError(
                f"Webhook host '{hostname}' is not in allowed_hosts — refused."
            )

        if not cfg.get("allow_private_ips", True):
            try:
                ip = ipaddress.ip_address(hostname)
            except ValueError:
                ip = None
            if ip is not None and (
                ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified
            ):
                raise ValueError(
                    f"Webhook target '{hostname}' is a non-public IP — refused."
                )

        effective_timeout = min(max(timeout, 1), 60)
        aio_timeout = aiohttp.ClientTimeout(total=effective_timeout)

        post_headers = get_headers()
        post_headers["Content-Type"] = "application/json"
        if headers:
            for k, v in headers.items():
                post_headers[str(k)] = str(v)

        try:
            async with aiohttp.ClientSession(timeout=aio_timeout) as session:
                async with session.post(url, json=payload or {}, headers=post_headers) as resp:
                    body = await resp.text()
                    truncated = body[:5_000]
                    if len(body) > 5_000:
                        truncated += "\n[response truncated]"
                    return f"Status: {resp.status}\nBody:\n{truncated}"
        except Exception as exc:
            raise RuntimeError(f"Webhook POST failed: {exc}")
