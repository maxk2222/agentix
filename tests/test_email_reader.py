"""Tests for the email_reader builtin tool (imaplib fully mocked)."""

from __future__ import annotations

import email
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.builtin_tools.email_reader import EmailReaderTool


@pytest.fixture()
def tool() -> EmailReaderTool:
    return EmailReaderTool()


IMAP_CFG = {
    "imap_host": "imap.example.com",
    "imap_port": 993,
    "imap_user": "user@example.com",
    "imap_password": "secret",
    "imap_use_ssl": True,
}


def _msg(frm: str, subj: str, date: str = "Mon, 1 Jan 2024 10:00:00 +0000",
         body: str = "Hello body") -> bytes:
    m = email.message.EmailMessage()
    m["From"] = frm
    m["To"] = "me@example.com"
    m["Subject"] = subj
    m["Date"] = date
    m.set_content(body)
    return m.as_bytes()


def _header_block(frm: str, subj: str) -> bytes:
    return f"From: {frm}\r\nSubject: {subj}\r\nDate: Mon, 1 Jan 2024 10:00:00 +0000\r\n\r\n".encode()


class _FakeIMAP:
    """Minimal fake for the calls email_reader makes."""

    def __init__(self, *args, **kwargs):
        self.uids = [b"101", b"102", b"103"]
        self.bodies = {
            b"101": _msg("alice@example.com", "First"),
            b"102": _msg("bob@example.com", "Second"),
            b"103": _msg("carol@example.com", "Third"),
        }
        self.headers = {
            b"101": _header_block("alice@example.com", "First"),
            b"102": _header_block("bob@example.com", "Second"),
            b"103": _header_block("carol@example.com", "Third"),
        }
        self.stored = []

    def login(self, user, pw):
        return "OK", [b"logged in"]

    def select(self, folder, readonly=True):
        return "OK", [b"3"]

    def uid(self, cmd, *args):
        if cmd == "search":
            return "OK", [b" ".join(self.uids)]
        if cmd == "fetch":
            uid = args[0]
            uid = uid.encode() if isinstance(uid, str) else uid
            spec = args[1] if len(args) > 1 else ""
            if "HEADER.FIELDS" in spec:
                return "OK", [(b"1 (FLAGS (\\Seen) BODY[])", self.headers.get(uid, b""))]
            return "OK", [(b"1 (BODY[])", self.bodies.get(uid, b""))]
        if cmd == "store":
            self.stored.append(args)
            return "OK", []
        return "OK", []

    def logout(self):
        return "OK", []


class TestEmailReaderList:
    @pytest.mark.asyncio
    async def test_list_returns_envelopes_newest_first(self, tool):
        with patch("imaplib.IMAP4_SSL", return_value=_FakeIMAP()):
            result = await tool.execute(
                action="list", context={"tool_config": IMAP_CFG}
            )
        assert "uid=103" in result
        assert "uid=101" in result
        # newest first
        assert result.index("uid=103") < result.index("uid=101")
        assert "Third" in result

    @pytest.mark.asyncio
    async def test_list_search_filters(self, tool):
        with patch("imaplib.IMAP4_SSL", return_value=_FakeIMAP()):
            result = await tool.execute(
                action="list", search="bob", context={"tool_config": IMAP_CFG}
            )
        assert "uid=102" in result
        assert "uid=101" not in result

    @pytest.mark.asyncio
    async def test_missing_config_errors(self, tool):
        with pytest.raises(RuntimeError, match="not configured"):
            await tool.execute(action="list", context={"tool_config": {}})


class TestEmailReaderRead:
    @pytest.mark.asyncio
    async def test_read_returns_body(self, tool):
        with patch("imaplib.IMAP4_SSL", return_value=_FakeIMAP()):
            result = await tool.execute(
                action="read", uid="102", context={"tool_config": IMAP_CFG}
            )
        assert "Subject: Second" in result
        assert "Hello body" in result
        assert "bob@example.com" in result

    @pytest.mark.asyncio
    async def test_read_requires_uid(self, tool):
        result = await tool.execute(action="read", context={"tool_config": IMAP_CFG})
        assert "requires 'uid'" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        result = await tool.execute(action="delete", context={"tool_config": IMAP_CFG})
        assert "unknown action" in result
