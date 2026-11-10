"""Email sender builtin tool."""

from __future__ import annotations

import asyncio
import base64
import logging
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import smtplib
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_ATTACHMENTS = 3
_MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024


def _validate_header(value: str) -> str:
    """Reject header-injection payloads in any email header field."""
    if "\r" in value or "\n" in value or "\x00" in value:
        raise ValueError("Invalid characters in email header value.")
    return value


class EmailSenderTool:
    """Send plain-text or HTML emails via SMTP using server-wide configuration."""

    name = "email_sender"
    description = (
        "Send an email via SMTP. Supports plain-text, HTML (multipart), CC, BCC, reply-to, "
        "and file attachments. Use 'to' (required), 'subject' (required), 'body' (required plain text), "
        "and optionally 'html_body', 'cc', 'bcc', 'reply_to', 'attachments'. "
        "Attachment items accept either {'filename','content_base64'} or "
        "{'artifact':'<name>'} to attach a generated document/artifact from this "
        "conversation (e.g. a file created by document_generator or code_execution). "
        "SMTP credentials are configured in the tool settings (dashboard → Tools → email_sender)."
    )

    def __init__(self, artifact_root: Optional[Path] = None) -> None:
        self._store = None
        if artifact_root is not None:
            from app.artifacts import ArtifactStore

            self._store = ArtifactStore(Path(artifact_root))

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
                        "to": {"type": "string", "description": "Primary recipient email address."},
                        "subject": {"type": "string", "description": "Email subject line."},
                        "body": {"type": "string", "description": "Plain-text email body (always included for fallback)."},
                        "html_body": {"type": "string", "description": "Optional HTML email body. If provided, a multipart message is sent with both plain-text and HTML."},
                        "cc": {"type": "array", "items": {"type": "string"}, "description": "Optional list of CC recipient email addresses."},
                        "bcc": {"type": "array", "items": {"type": "string"}, "description": "Optional list of BCC recipient email addresses."},
                        "reply_to": {"type": "string", "description": "Optional reply-to email address."},
                        "attachments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "filename": {"type": "string"},
                                    "content_base64": {"type": "string"},
                                    "mime_type": {"type": "string"},
                                    "artifact": {"type": "string", "description": "Name of a generated artifact in this conversation to attach (alternative to content_base64)."},
                                },
                            },
                            "description": "Optional file attachments (max 3, max 5 MB total). Each item takes {'filename','content_base64','mime_type'?} or {'artifact':'<name>'} to attach a generated conversation artifact.",
                        },
                    },
                    "required": ["to", "subject", "body"],
                },
            },
        }

    async def execute(
        self,
        to: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        reply_to: Optional[str] = None,
        attachments: Optional[List[Dict[str, str]]] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        cfg = (context or {}).get("tool_config", {})
        smtp_host = cfg.get("smtp_host", "").strip()
        smtp_port = int(cfg.get("smtp_port", "587"))
        smtp_user = cfg.get("smtp_user", "").strip()
        smtp_password = cfg.get("smtp_password", "").strip()
        from_addr = cfg.get("smtp_from", smtp_user).strip()
        smtp_use_ssl = bool(cfg.get("smtp_use_ssl", False))
        smtp_timeout = int(cfg.get("smtp_timeout", "30"))

        if not smtp_host:
            raise RuntimeError(
                "SMTP host is not configured. Please configure the email_sender tool "
                "in the dashboard under Tools → email_sender with smtp_host, smtp_user, "
                "and smtp_password."
            )

        msg = MIMEMultipart("mixed")
        msg["From"] = from_addr or smtp_user
        msg["To"] = _validate_header(to)
        msg["Subject"] = _validate_header(subject)

        if cc:
            msg["Cc"] = ", ".join(_validate_header(a) for a in cc)
        if bcc:
            msg["Bcc"] = ", ".join(_validate_header(a) for a in bcc)
        if reply_to:
            msg["Reply-To"] = _validate_header(reply_to)

        if html_body:
            alt = MIMEMultipart("alternative")
            alt.attach(MIMEText(body, "plain", "utf-8"))
            alt.attach(MIMEText(html_body, "html", "utf-8"))
            msg.attach(alt)
        else:
            msg.attach(MIMEText(body, "plain", "utf-8"))

        if attachments:
            if len(attachments) > _MAX_ATTACHMENTS:
                raise ValueError(f"Too many attachments (max {_MAX_ATTACHMENTS}).")
            total_bytes = 0
            for att in attachments:
                artifact_name = (att.get("artifact") or "").strip()
                if artifact_name:
                    if self._store is None:
                        raise RuntimeError("Artifact attachments are unavailable on this server.")
                    conversation_id = (context or {}).get("conversation_id")
                    if not conversation_id:
                        raise RuntimeError("Artifact attachments require a conversation context.")
                    resolved = self._store.resolve(conversation_id, artifact_name)
                    if resolved is None:
                        raise ValueError(f"Artifact '{artifact_name}' not found in this conversation.")
                    payload = await asyncio.to_thread(resolved.read_bytes)
                    filename = att.get("filename") or resolved.name
                    from app.artifacts import mime_for
                    mime_type = att.get("mime_type") or mime_for(resolved.name)
                else:
                    filename = att.get("filename", "attachment")
                    content_b64 = att.get("content_base64", "")
                    mime_type = att.get("mime_type", "application/octet-stream")
                    if not content_b64:
                        continue
                    try:
                        payload = base64.b64decode(content_b64, validate=True)
                    except Exception:
                        raise ValueError(f"Invalid base64 content for attachment '{filename}'")
                total_bytes += len(payload)
                if total_bytes > _MAX_ATTACHMENT_BYTES:
                    raise ValueError(
                        f"Attachments exceed the {_MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB total limit."
                    )
                filename = str(filename).replace('"', "'").replace("\r", "").replace("\n", "")
                main_type, _, sub_type = mime_type.partition("/")
                part = MIMEBase(main_type or "application", sub_type or "octet-stream")
                part.set_payload(payload)
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f'attachment; filename="{filename}"',
                )
                msg.attach(part)

        all_recipients = [to]
        if cc:
            all_recipients.extend(cc)
        if bcc:
            all_recipients.extend(bcc)

        try:
            loop = asyncio.get_running_loop()

            def _send() -> str:
                if smtp_use_ssl:
                    with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=smtp_timeout) as server:
                        if smtp_user and smtp_password:
                            server.login(smtp_user, smtp_password)
                        server.sendmail(from_addr or smtp_user, all_recipients, msg.as_string())
                else:
                    with smtplib.SMTP(smtp_host, smtp_port, timeout=smtp_timeout) as server:
                        if smtp_user and smtp_password:
                            server.starttls()
                            server.login(smtp_user, smtp_password)
                        server.sendmail(from_addr or smtp_user, all_recipients, msg.as_string())
                recipients = ", ".join(all_recipients)
                return f"Email sent to {recipients}"

            result = await asyncio.wait_for(
                loop.run_in_executor(None, _send),
                timeout=60,
            )
            return result
        except asyncio.TimeoutError:
            raise RuntimeError("SMTP connection timed out after 60s")
        except Exception as exc:
            raise RuntimeError(f"Failed to send email: {exc}")
