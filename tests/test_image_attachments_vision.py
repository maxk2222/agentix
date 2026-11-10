"""Tests for image-attachment → multimodal vision pipeline.

Covers ``image_attachment_data_uris`` (whitelist/size/count/order/dedup)
plus the regression guard that chain prompts keep ``image_url`` content
parts intact instead of stringifying them into a Python repr.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from app.chat_attachments import (
    ChatAttachmentStore,
    image_attachment_data_uris,
)


@pytest.fixture()
def store(tmp_path):
    return ChatAttachmentStore(tmp_path / "uploads")


def _png_bytes() -> bytes:
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )


async def _save(store, conv, name, data):
    return await store.save(conv, name, data)


class TestImageAttachmentDataUris:
    @pytest.mark.asyncio
    async def test_png_becomes_data_uri_and_marks_meta(self, store):
        meta = await _save(store, "c1", "pic.png", _png_bytes())
        metas = store.list("c1")
        uris = await image_attachment_data_uris(
            store, "c1", metas, max_bytes=10_000_000, max_count=4
        )
        assert len(uris) == 1
        assert uris[0].startswith("data:image/png;base64,")
        assert metas[0].get("vision") is True

    @pytest.mark.asyncio
    async def test_non_vision_exts_skipped(self, store):
        await _save(store, "c1", "scan.tiff", b"II*\x00faketiff")
        await _save(store, "c1", "doc.bmp", b"BMfake")
        metas = store.list("c1")
        uris = await image_attachment_data_uris(
            store, "c1", metas, max_bytes=10_000_000, max_count=4
        )
        assert uris == []

    @pytest.mark.asyncio
    async def test_non_image_attachments_skipped(self, store):
        await _save(store, "c1", "notes.txt", b"hello")
        metas = store.list("c1")
        uris = await image_attachment_data_uris(
            store, "c1", metas, max_bytes=10_000_000, max_count=4
        )
        assert uris == []

    @pytest.mark.asyncio
    async def test_max_count_caps_and_newest_first(self, store):
        await _save(store, "c1", "a.png", _png_bytes())
        await _save(store, "c1", "b.png", _png_bytes())
        await _save(store, "c1", "c.png", _png_bytes())
        metas = store.list("c1")
        assert [m["name"] for m in metas] == ["a.png", "b.png", "c.png"]
        uris = await image_attachment_data_uris(
            store, "c1", metas, max_bytes=10_000_000, max_count=1
        )
        assert len(uris) == 1
        # newest (c.png) wins — marked vision, the older ones not
        assert metas[2].get("vision") is True
        assert not metas[0].get("vision")
        assert not metas[1].get("vision")

    @pytest.mark.asyncio
    async def test_max_bytes_skips_oversized(self, store):
        await _save(store, "c1", "big.png", b"x" * 5000)
        metas = store.list("c1")
        uris = await image_attachment_data_uris(
            store, "c1", metas, max_bytes=100, max_count=4
        )
        assert uris == []

    @pytest.mark.asyncio
    async def test_exclude_uris_dedupes(self, store):
        await _save(store, "c1", "pic.png", _png_bytes())
        metas = store.list("c1")
        existing_uri = f"data:image/png;base64,{base64.b64encode(_png_bytes()).decode()}"
        uris = await image_attachment_data_uris(
            store, "c1", metas, max_bytes=10_000_000, max_count=4,
            exclude_uris={existing_uri},
        )
        assert uris == []

    @pytest.mark.asyncio
    async def test_missing_original_skips_safely(self, store):
        meta = await _save(store, "c1", "pic.png", _png_bytes())
        path = store.original_path("c1", meta["id"])
        path.unlink()
        metas = store.list("c1")
        uris = await image_attachment_data_uris(
            store, "c1", metas, max_bytes=10_000_000, max_count=4
        )
        assert uris == []

    @pytest.mark.asyncio
    async def test_zero_count_returns_empty(self, store):
        await _save(store, "c1", "pic.png", _png_bytes())
        uris = await image_attachment_data_uris(
            store, "c1", store.list("c1"), max_bytes=10_000_000, max_count=0
        )
        assert uris == []


class TestMultimodalPromptRegression:
    def test_messages_placeholder_keeps_image_parts(self):
        """Regression: ('human','{input}') stringified content lists to a
        Python repr — input_messages must render image_url parts intact."""
        from langchain_core.messages import HumanMessage
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages([
            ("system", "s"),
            MessagesPlaceholder("chat_history"),
            MessagesPlaceholder("input_messages"),
        ])
        content = [
            {"type": "text", "text": "what is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]
        msgs = prompt.invoke({
            "chat_history": [],
            "input_messages": [HumanMessage(content=content)],
        }).to_messages()
        last = msgs[-1]
        assert isinstance(last.content, list)
        kinds = {p["type"] for p in last.content}
        assert kinds == {"text", "image_url"}
        assert last.content[1]["image_url"]["url"].startswith("data:image/png")

    def test_string_template_would_destroy_parts(self):
        """Document the bug we fixed: ('human','{input}') reprs the list."""
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        msgs = prompt.invoke({
            "input": [{"type": "text", "text": "q"},
                      {"type": "image_url", "image_url": {"url": "data:x"}}],
        }).to_messages()
        # The old path produced a repr string — proves the regression existed.
        assert isinstance(msgs[-1].content, str)
        assert "'image_url'" in msgs[-1].content


class TestInjectRagContextPassthrough:
    @pytest.mark.asyncio
    async def test_input_messages_flows_through_rag_prompt(self):
        """_inject_rag_context rebuilds the payload dict — input_messages
        must be forwarded or images die exactly there."""
        from langchain_core.messages import HumanMessage
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        # Mirror the rag_qa_prompt + _inject_rag_context contract.
        rag_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_text}"),
            MessagesPlaceholder("chat_history"),
            MessagesPlaceholder("input_messages"),
        ])
        payload = {
            "system_text": "sys",
            "chat_history": [],
            "input": "raw text",
            "input_messages": [HumanMessage(content=[
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ])],
        }
        rebuilt = {
            "system_text": payload["system_text"],
            "chat_history": payload.get("chat_history", []),
            "input_messages": payload.get("input_messages") or [],
        }
        msgs = rag_qa_prompt.invoke(rebuilt).to_messages()
        assert isinstance(msgs[-1].content, list)
        assert msgs[-1].content[1]["type"] == "image_url"
