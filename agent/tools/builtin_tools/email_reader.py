"""IMAP inbox reader builtin tool.

Read-only mailbox access: list envelopes and read message bodies over IMAP.
Credentials live in the per-tool configuration (dashboard → Tools →
email_reader); they are never logged or returned to the model.
"""

from __future__ import annotations

import asyncio
import email
import email.header
import email.utils
import imaplib
import logging
from typing import Any, Dict, List, Optional

from agent.tools.builtin_tools.common import extract_text_from_html

logger = logging.getLogger(__name__)

_MAX_LIST = 50
_MAX_BODY_CHARS = 20_000


def _decode_header(value: Optional[str]) -> str:
    if not value:
        return ""
    parts: List[str] = []
    try:
        for text, charset in email.header.decode_header(value):
            if isinstance(text, bytes):
                parts.append(text.decode(charset or "utf-8", errors="replace"))
            else:
                parts.append(text)
    except Exception:
        return str(value)
    return "".join(parts)


def _extract_body(msg: email.message.Message) -> str:
    """Prefer text/plain; fall back to stripped text/html."""
    plain, html = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp.lower():
                continue
            try:
                payload = part.get_payload(decode=True)
            except Exception:
                payload = None
            if not payload:
                continue
            charset = part.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if ctype == "text/plain" and not plain:
                plain = text
            elif ctype == "text/html" and not html:
                html = text
    else:
        try:
            payload = msg.get_payload(decode=True)
        except Exception:
            payload = None
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/html":
                html = text
            else:
                plain = text
    if plain.strip():
        return plain.strip()
    if html.strip():
        return extract_text_from_html(html).strip()
    return ""


class EmailReaderTool:
    """Read the configured IMAP mailbox (list envelopes / read bodies)."""

    name = "email_reader"
    description = (
        "Read emails from the configured IMAP inbox. "
        "action='list' returns recent envelopes [{uid, from, subject, date, seen}] — "
        "options: 'folder' (default INBOX), 'limit' (default 10, max 50), "
        "'unseen_only' (bool), 'search' (matches sender or subject). "
        "action='read' fetches one message body by 'uid' (plain text preferred, "
        "HTML stripped); 'mark_read' (bool, default false) sets the \\Seen flag. "
        "IMAP credentials are configured in the tool settings (dashboard → Tools → email_reader)."
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
                        "action": {
                            "type": "string",
                            "enum": ["list", "read"],
                            "description": "'list' = recent envelopes, 'read' = one message body (requires 'uid').",
                        },
                        "folder": {"type": "string", "description": "Mailbox folder (default 'INBOX')."},
                        "limit": {"type": "integer", "description": "Max envelopes for 'list' (default 10, max 50).", "default": 10},
                        "unseen_only": {"type": "boolean", "description": "Only unread messages for 'list' (default false).", "default": False},
                        "search": {"type": "string", "description": "Optional sender/subject filter for 'list'."},
                        "uid": {"type": "string", "description": "Message uid for action='read' (from the same 'folder' it was listed in)."},
                        "mark_read": {"type": "boolean", "description": "Mark message as read on action='read' (default false).", "default": False},
                    },
                    "required": ["action"],
                },
            },
        }

    def _connect(self, cfg: Dict[str, Any]) -> Any:
        host = str(cfg.get("imap_host", "") or "").strip()
        if not host:
            raise RuntimeError(
                "IMAP host is not configured. Please configure the email_reader tool "
                "in the dashboard under Tools → email_reader with imap_host, imap_user "
                "and imap_password."
            )
        port = int(cfg.get("imap_port", 993) or 993)
        use_ssl = bool(cfg.get("imap_use_ssl", True))
        timeout = int(cfg.get("imap_timeout", 30) or 30)
        cls = imaplib.IMAP4_SSL if use_ssl else imaplib.IMAP4
        conn = cls(host, port, timeout=timeout)
        user = str(cfg.get("imap_user", "") or "").strip()
        password = str(cfg.get("imap_password", "") or "").strip()
        if user or password:
            conn.login(user, password)
        return conn

    @staticmethod
    def _safe_folder(folder: str) -> str:
        # Folder names are quoted into the IMAP SELECT command — strip
        # characters that could break out of the quoting.
        return (folder or "INBOX").replace('"', "").replace("\\", "").strip() or "INBOX"

    def _list(self, cfg: Dict[str, Any], folder: str, limit: int,
              unseen_only: bool, search: str) -> List[Dict[str, Any]]:
        conn = self._connect(cfg)
        try:
            status, _ = conn.select(f'"{self._safe_folder(folder)}"', readonly=True)
            if status != "OK":
                raise RuntimeError(f"Cannot open folder '{folder}'.")
            criteria = "UNSEEN" if unseen_only else "ALL"
            status, data = conn.uid("search", None, criteria)
            if status != "OK":
                raise RuntimeError("IMAP search failed.")
            uids = (data[0] or b"").split()
            envelopes: List[Dict[str, Any]] = []
            needle = search.lower()
            for uid in reversed(uids):
                if len(envelopes) >= limit:
                    break
                status, msg_data = conn.uid(
                    "fetch", uid, "(FLAGS BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"
                )
                if status != "OK" or not msg_data:
                    continue
                header_blob = b""
                flags_blob = ""
                for item in msg_data:
                    if isinstance(item, tuple):
                        flags_blob += item[0].decode("utf-8", errors="replace") if isinstance(item[0], bytes) else str(item[0])
                        header_blob += item[1] if isinstance(item[1], bytes) else b""
                msg = email.message_from_bytes(header_blob)
                sender = _decode_header(msg.get("From"))
                subject = _decode_header(msg.get("Subject"))
                if needle and needle not in sender.lower() and needle not in subject.lower():
                    continue
                envelopes.append({
                    "uid": uid.decode("utf-8", errors="replace") if isinstance(uid, bytes) else str(uid),
                    "from": sender,
                    "subject": subject,
                    "date": str(msg.get("Date") or ""),
                    "seen": "\\Seen" in flags_blob,
                })
            return envelopes
        finally:
            try:
                conn.logout()
            except Exception:
                pass

    def _read(self, cfg: Dict[str, Any], uid: str, folder: str,
              mark_read: bool) -> Dict[str, Any]:
        conn = self._connect(cfg)
        try:
            status, _ = conn.select(f'"{self._safe_folder(folder)}"', readonly=not mark_read)
            if status != "OK":
                raise RuntimeError(f"Cannot open folder '{self._safe_folder(folder)}'.")
            status, msg_data = conn.uid("fetch", uid, "(BODY.PEEK[])")
            if status != "OK" or not msg_data or msg_data[0] is None:
                raise RuntimeError(f"Message uid '{uid}' not found.")
            raw = b""
            for item in msg_data:
                if isinstance(item, tuple) and isinstance(item[1], bytes):
                    raw += item[1]
            msg = email.message_from_bytes(raw)
            body = _extract_body(msg)
            if len(body) > _MAX_BODY_CHARS:
                body = body[:_MAX_BODY_CHARS] + "\n...[body truncated]"
            if mark_read:
                try:
                    conn.uid("store", uid, "+FLAGS", "(\\Seen)")
                except Exception:
                    pass
            attachments = [
                str(part.get_filename() or "")
                for part in msg.walk()
                if "attachment" in str(part.get("Content-Disposition") or "").lower()
            ]
            return {
                "uid": uid,
                "from": _decode_header(msg.get("From")),
                "to": _decode_header(msg.get("To")),
                "subject": _decode_header(msg.get("Subject")),
                "date": str(msg.get("Date") or ""),
                "body": body,
                "attachments": [a for a in attachments if a],
            }
        finally:
            try:
                conn.logout()
            except Exception:
                pass

    async def execute(
        self,
        action: str,
        folder: str = "INBOX",
        limit: int = 10,
        unseen_only: bool = False,
        search: Optional[str] = None,
        uid: Optional[str] = None,
        mark_read: bool = False,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        cfg = (context or {}).get("tool_config", {})
        action = (action or "").strip().lower()
        limit = max(1, min(_MAX_LIST, int(limit or 10)))

        if action == "list":
            folder = (folder or "INBOX").strip() or "INBOX"
            envelopes = await asyncio.to_thread(
                self._list, cfg, folder, limit, bool(unseen_only), (search or "").strip()
            )
            if not envelopes:
                return "No matching messages."
            lines = [f"{len(envelopes)} message(s) in {folder}:"]
            for env in envelopes:
                flag = " " if env["seen"] else "*"
                lines.append(
                    f"{flag} uid={env['uid']} | {env['date']}\n"
                    f"    from: {env['from']}\n    subject: {env['subject']}"
                )
            lines.append("(* = unread) Use action='read' with a uid to read a message.")
            return "\n".join(lines)

        if action == "read":
            if not uid or not str(uid).strip():
                return "[Error: action='read' requires 'uid'.]"
            folder = (folder or "INBOX").strip() or "INBOX"
            result = await asyncio.to_thread(
                self._read, cfg, str(uid).strip(), folder, bool(mark_read)
            )
            parts = [
                f"From: {result['from']}",
                f"To: {result['to']}",
                f"Date: {result['date']}",
                f"Subject: {result['subject']}",
            ]
            if result["attachments"]:
                parts.append(f"Attachments: {', '.join(result['attachments'])} (not downloaded)")
            parts.append("")
            parts.append(result["body"] or "(no readable body)")
            return "\n".join(parts)

        return "[Error: unknown action '%s' — use 'list' or 'read'.]" % action
