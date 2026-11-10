"""Tests for the built-in email_sender tool."""

from __future__ import annotations

import base64

import pytest
from email import message_from_string
from unittest.mock import MagicMock, patch

from agent.tools.builtin import EmailSenderTool


@pytest.fixture
def email_tool() -> EmailSenderTool:
    return EmailSenderTool()


@pytest.fixture
def smtp_config() -> dict:
    return {
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "smtp_user": "sender@example.com",
        "smtp_password": "secret",
        "smtp_from": "noreply@example.com",
        "smtp_use_ssl": False,
        "smtp_timeout": 30,
    }


class TestEmailSenderPlain:
    @pytest.mark.asyncio
    async def test_plain_text_send(self, email_tool, smtp_config):
        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_server = MagicMock()
            mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

            result = await email_tool.execute(
                to="recipient@example.com",
                subject="Hello",
                body="This is a test.",
                context={"tool_config": smtp_config},
            )

            assert "sent to" in result
            assert "recipient@example.com" in result
            mock_smtp_class.assert_called_once_with("smtp.example.com", 587, timeout=30)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("sender@example.com", "secret")
            mock_server.sendmail.assert_called_once()
            from_addr, recipients, raw_msg = mock_server.sendmail.call_args[0]
            assert from_addr == "noreply@example.com"
            assert recipients == ["recipient@example.com"]

            msg = message_from_string(raw_msg)
            assert msg["To"] == "recipient@example.com"
            assert msg["Subject"] == "Hello"
            assert msg["From"] == "noreply@example.com"
            assert msg.is_multipart()
            parts = msg.get_payload()
            assert len(parts) == 1
            assert parts[0].get_content_type() == "text/plain"
            decoded = parts[0].get_payload(decode=True)
            if decoded:
                assert b"This is a test." in decoded
            else:
                assert "This is a test." in parts[0].get_payload()


class TestEmailSenderHtml:
    @pytest.mark.asyncio
    async def test_html_multipart(self, email_tool, smtp_config):
        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_server = MagicMock()
            mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

            result = await email_tool.execute(
                to="recipient@example.com",
                subject="HTML Test",
                body="Plain text fallback.",
                html_body="<h1>Hello</h1><p>World</p>",
                context={"tool_config": smtp_config},
            )

            assert "sent to" in result
            mock_server.sendmail.assert_called_once()
            from_addr, recipients, raw_msg = mock_server.sendmail.call_args[0]
            msg = message_from_string(raw_msg)
            assert msg.is_multipart()
            top_parts = msg.get_payload()
            assert len(top_parts) == 1
            alt = top_parts[0]
            assert alt.get_content_type() == "multipart/alternative"
            alt_parts = alt.get_payload()
            assert len(alt_parts) == 2
            assert alt_parts[0].get_content_type() == "text/plain"
            plain_decoded = alt_parts[0].get_payload(decode=True)
            if plain_decoded:
                assert b"Plain text fallback." in plain_decoded
            else:
                assert "Plain text fallback." in alt_parts[0].get_payload()
            assert alt_parts[1].get_content_type() == "text/html"
            html_decoded = alt_parts[1].get_payload(decode=True)
            if html_decoded:
                assert b"<h1>Hello</h1>" in html_decoded
            else:
                assert "<h1>Hello</h1>" in alt_parts[1].get_payload()


class TestEmailSenderCcBccReplyTo:
    @pytest.mark.asyncio
    async def test_cc_bcc_reply_to(self, email_tool, smtp_config):
        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_server = MagicMock()
            mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

            result = await email_tool.execute(
                to="to@example.com",
                subject="Test",
                body="Body",
                cc=["cc1@example.com", "cc2@example.com"],
                bcc=["bcc@example.com"],
                reply_to="reply@example.com",
                context={"tool_config": smtp_config},
            )

            mock_server.sendmail.assert_called_once()
            from_addr, recipients, raw_msg = mock_server.sendmail.call_args[0]
            assert recipients == ["to@example.com", "cc1@example.com", "cc2@example.com", "bcc@example.com"]
            msg = message_from_string(raw_msg)
            assert msg["Cc"] == "cc1@example.com, cc2@example.com"
            assert msg["Bcc"] == "bcc@example.com"
            assert msg["Reply-To"] == "reply@example.com"


class TestEmailSenderSsl:
    @pytest.mark.asyncio
    async def test_smtp_ssl(self, email_tool, smtp_config):
        smtp_config["smtp_use_ssl"] = True
        smtp_config["smtp_port"] = 465
        with patch("smtplib.SMTP_SSL") as mock_ssl_class:
            mock_server = MagicMock()
            mock_ssl_class.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_ssl_class.return_value.__exit__ = MagicMock(return_value=False)

            result = await email_tool.execute(
                to="recipient@example.com",
                subject="SSL Test",
                body="Body",
                context={"tool_config": smtp_config},
            )

            assert "sent to" in result
            mock_ssl_class.assert_called_once_with("smtp.example.com", 465, timeout=30)
            mock_server.starttls.assert_not_called()
            mock_server.login.assert_called_once_with("sender@example.com", "secret")


class TestEmailSenderAttachments:
    @pytest.mark.asyncio
    async def test_single_attachment(self, email_tool, smtp_config):
        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_server = MagicMock()
            mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

            content = b"file content"
            b64 = base64.b64encode(content).decode()

            result = await email_tool.execute(
                to="recipient@example.com",
                subject="Attachment",
                body="See attached.",
                attachments=[{
                    "filename": "report.txt",
                    "content_base64": b64,
                    "mime_type": "text/plain",
                }],
                context={"tool_config": smtp_config},
            )

            mock_server.sendmail.assert_called_once()
            from_addr, recipients, raw_msg = mock_server.sendmail.call_args[0]
            msg = message_from_string(raw_msg)
            assert msg.is_multipart()
            parts = msg.get_payload()
            assert len(parts) == 2
            attachment_parts = [p for p in parts if p.get_content_disposition() == "attachment"]
            assert len(attachment_parts) == 1
            att_part = attachment_parts[0]
            assert att_part.get_filename() == "report.txt"
            assert att_part.get_content_type() == "text/plain"
            payload = att_part.get_payload(decode=True)
            if payload:
                assert payload == content
            else:
                assert base64.b64decode(att_part.get_payload()) == content

    @pytest.mark.asyncio
    async def test_html_with_attachment(self, email_tool, smtp_config):
        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_server = MagicMock()
            mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

            content = b"binary data"
            b64 = base64.b64encode(content).decode()

            result = await email_tool.execute(
                to="recipient@example.com",
                subject="HTML + Attachment",
                body="Plain text.",
                html_body="<p>HTML</p>",
                attachments=[{"filename": "data.bin", "content_base64": b64}],
                context={"tool_config": smtp_config},
            )

            assert "sent to" in result
            from_addr, recipients, raw_msg = mock_server.sendmail.call_args[0]
            msg = message_from_string(raw_msg)
            assert msg.is_multipart()
            parts = msg.get_payload()
            assert len(parts) == 2  # alternative container + attachment
            alt = [p for p in parts if p.get_content_type() == "multipart/alternative"][0]
            att = [p for p in parts if p.get_content_disposition() == "attachment"][0]
            assert alt is not None
            assert att.get_filename() == "data.bin"

    @pytest.mark.asyncio
    async def test_invalid_base64_attachment(self, email_tool, smtp_config):
        with patch("smtplib.SMTP"):
            with pytest.raises(ValueError, match="Invalid base64"):
                await email_tool.execute(
                    to="recipient@example.com",
                    subject="Bad",
                    body="Body",
                    attachments=[{"filename": "bad.txt", "content_base64": "!!!not_valid!!!"}],
                    context={"tool_config": smtp_config},
                )


class TestEmailSenderArtifactAttachments:
    @pytest.mark.asyncio
    async def test_artifact_attachment_resolved(self, smtp_config, tmp_path):
        from app.artifacts import ArtifactStore

        store = ArtifactStore(tmp_path / "artifacts")
        data = b"%PDF-1.4 content"
        await store.save("conv-1", "AssistantX", "report.pdf", data, "pdf")
        tool = EmailSenderTool(tmp_path / "artifacts")

        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_server = MagicMock()
            mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

            result = await tool.execute(
                to="recipient@example.com",
                subject="With artifact",
                body="Attached.",
                attachments=[{"artifact": "report.pdf"}],
                context={"tool_config": smtp_config, "conversation_id": "conv-1"},
            )
            assert "sent to" in result
            _, _, raw_msg = mock_server.sendmail.call_args[0]
            msg = message_from_string(raw_msg)
            atts = [p for p in msg.get_payload() if p.get_content_disposition() == "attachment"]
            assert len(atts) == 1
            assert atts[0].get_filename() == "report.pdf"
            assert atts[0].get_content_type() == "application/pdf"
            assert atts[0].get_payload(decode=True) == data

    @pytest.mark.asyncio
    async def test_artifact_missing_fails(self, smtp_config, tmp_path):
        tool = EmailSenderTool(tmp_path / "artifacts")
        with patch("smtplib.SMTP"):
            with pytest.raises(ValueError, match="not found"):
                await tool.execute(
                    to="recipient@example.com",
                    subject="T",
                    body="B",
                    attachments=[{"artifact": "nope.pdf"}],
                    context={"tool_config": smtp_config, "conversation_id": "conv-1"},
                )

    @pytest.mark.asyncio
    async def test_artifact_other_conversation_invisible(self, smtp_config, tmp_path):
        from app.artifacts import ArtifactStore

        store = ArtifactStore(tmp_path / "artifacts")
        await store.save("conv-other", "M", "secret.pdf", b"data", "pdf")
        tool = EmailSenderTool(tmp_path / "artifacts")
        with patch("smtplib.SMTP"):
            with pytest.raises(ValueError, match="not found"):
                await tool.execute(
                    to="recipient@example.com",
                    subject="T",
                    body="B",
                    attachments=[{"artifact": "secret.pdf"}],
                    context={"tool_config": smtp_config, "conversation_id": "conv-1"},
                )

    @pytest.mark.asyncio
    async def test_artifact_without_store_unavailable(self, email_tool, smtp_config):
        with patch("smtplib.SMTP"):
            with pytest.raises(RuntimeError, match="unavailable"):
                await email_tool.execute(
                    to="recipient@example.com",
                    subject="T",
                    body="B",
                    attachments=[{"artifact": "x.pdf"}],
                    context={"tool_config": smtp_config, "conversation_id": "conv-1"},
                )


class TestEmailSenderLimitsAndInjection:
    @pytest.mark.asyncio
    async def test_too_many_attachments(self, email_tool, smtp_config):
        b64 = base64.b64encode(b"x").decode()
        with patch("smtplib.SMTP"):
            with pytest.raises(ValueError, match="Too many attachments"):
                await email_tool.execute(
                    to="recipient@example.com",
                    subject="T",
                    body="B",
                    attachments=[{"filename": f"{i}.txt", "content_base64": b64} for i in range(4)],
                    context={"tool_config": smtp_config},
                )

    @pytest.mark.asyncio
    async def test_total_size_limit(self, email_tool, smtp_config):
        big = base64.b64encode(b"x" * (3 * 1024 * 1024)).decode()
        with patch("smtplib.SMTP"):
            with pytest.raises(ValueError, match="MB total limit"):
                await email_tool.execute(
                    to="recipient@example.com",
                    subject="T",
                    body="B",
                    attachments=[
                        {"filename": "a.bin", "content_base64": big},
                        {"filename": "b.bin", "content_base64": big},
                    ],
                    context={"tool_config": smtp_config},
                )

    @pytest.mark.asyncio
    async def test_header_injection_rejected(self, email_tool, smtp_config):
        with patch("smtplib.SMTP"):
            with pytest.raises(ValueError, match="Invalid characters"):
                await email_tool.execute(
                    to="victim@example.com\r\nBcc: evil@example.com",
                    subject="T",
                    body="B",
                    context={"tool_config": smtp_config},
                )


class TestEmailSenderMissingConfig:
    @pytest.mark.asyncio
    async def test_missing_smtp_host(self, email_tool):
        with pytest.raises(RuntimeError, match="SMTP host is not configured"):
            await email_tool.execute(
                to="recipient@example.com",
                subject="Test",
                body="Body",
                context={"tool_config": {}},
            )
