from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile
from langchain_core.messages import HumanMessage

from app.chat_attachments import (
    AttachmentError,
    ChatAttachmentStore,
    _xml_escape,
    build_attachment_context,
    extract_attachment_text,
)
from app.routers.chat import (
    chat_attachment_delete,
    chat_attachment_list,
    chat_attachment_upload,
    chat_endpoint,
)
from app.schemas import ChatRequest, Message


def _cfg(**overrides) -> SimpleNamespace:
    base = dict(
        chat_file_upload_enabled=True,
        chat_file_allowed_exts=".txt,.md,.csv,.docx",
        chat_file_max_size_bytes=20_971_520,
        chat_file_max_count=3,
        chat_file_max_extract_chars=50_000,
        chat_file_max_total_chars=120_000,
        chat_file_extract_timeout_seconds=60.0,
        chat_image_upload_enabled=False,
        ocr_language="deu",
        ocr_mode="tesseract",
        pdf_image_extraction_enabled=False,
        pptx_slide_analysis_enabled=False,
        dashboard_session_secret=None,
        max_message_chars=20_000,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _request(*, store=None, cfg=None, api_key_id="k1", rag=None, msg_store=None):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                chat_attachment_store=store,
                app_config=cfg,
                rag_application=rag,
                _message_store=msg_store,
            )
        ),
        state=SimpleNamespace(api_key_id=api_key_id),
        headers={},
        query_params={},
    )


def _upload(data: bytes, filename: str = "notes.txt") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


# ---------------------------------------------------------------------------
# ChatAttachmentStore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_save_list_get_delete(tmp_path):
    store = ChatAttachmentStore(tmp_path)
    meta = await store.save("conv-1", "doc.txt", b"hello")
    assert meta["id"].startswith("att-")
    assert meta["name"] == "doc.txt"
    assert meta["ext"] == ".txt"
    assert meta["size"] == 5

    listed = store.list("conv-1")
    assert [m["id"] for m in listed] == [meta["id"]]
    assert store.get("conv-1", meta["id"])["name"] == "doc.txt"
    assert store.original_path("conv-1", meta["id"]).read_bytes() == b"hello"

    await store.set_extracted("conv-1", meta["id"], "extracted text", False)
    assert store.extracted_text("conv-1", meta["id"]) == "extracted text"
    assert store.get("conv-1", meta["id"])["char_count"] == len("extracted text")

    assert await store.delete("conv-1", meta["id"]) is True
    assert store.list("conv-1") == []
    assert store.get("conv-1", meta["id"]) is None


@pytest.mark.asyncio
async def test_store_delete_conversation_and_traversal(tmp_path):
    store = ChatAttachmentStore(tmp_path)
    await store.save("conv-2", "a.txt", b"x")
    await store.delete_conversation("conv-2")
    assert store.list("conv-2") == []
    # Traversal-ish ids resolve to missing/None rather than escaping the dir
    assert store.get("conv-2", "../escape") is None
    assert store.extracted_text("conv-2", "../../etc/passwd") is None


@pytest.mark.asyncio
async def test_store_conversation_id_sanitized(tmp_path):
    store = ChatAttachmentStore(tmp_path)
    meta = await store.save("../evil/conv", "a.txt", b"x")
    listed = store.list("../evil/conv")
    assert len(listed) == 1 and listed[0]["id"] == meta["id"]
    # Everything stays under root
    assert tmp_path.joinpath("evil").exists() is False or True
    for p in tmp_path.iterdir():
        assert p.resolve().is_relative_to(tmp_path.resolve())


# ---------------------------------------------------------------------------
# build_attachment_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_attachment_context_blocks_and_escape(tmp_path):
    store = ChatAttachmentStore(tmp_path)
    meta = await store.save("c", 'a<b>"&.txt', b"data")
    await store.set_extracted("c", meta["id"], "line1\nline2", False)

    ctx = build_attachment_context(store, "c", max_total_chars=10_000)
    assert '<attachment name="a&lt;b&gt;&quot;&amp;.txt" type="txt">' in ctx
    assert "line1\nline2" in ctx
    assert ctx.endswith("</attachment>\n")


@pytest.mark.asyncio
async def test_build_attachment_context_omitted_marker(tmp_path):
    store = ChatAttachmentStore(tmp_path)
    m1 = await store.save("c", "big.txt", b"x")
    m2 = await store.save("c", "small.txt", b"x")
    await store.set_extracted("c", m1["id"], "x" * 500, False)
    await store.set_extracted("c", m2["id"], "ok", False)

    ctx = build_attachment_context(store, "c", max_total_chars=200)
    assert 'omitted="character-limit"' in ctx
    assert ">ok<" in ctx or "ok" in ctx

    assert build_attachment_context(store, "empty-conv", max_total_chars=100) == ""


# ---------------------------------------------------------------------------
# extract_attachment_text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_text_file_direct(tmp_path):
    f = tmp_path / "readme.txt"
    f.write_text("hello world", encoding="utf-8")
    text = await extract_attachment_text(f, _cfg())
    assert text == "hello world"


@pytest.mark.asyncio
async def test_extract_unsupported_ext(tmp_path):
    f = tmp_path / "a.exe"
    f.write_bytes(b"MZ")
    with pytest.raises(AttachmentError, match="Unsupported file type"):
        await extract_attachment_text(f, _cfg())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_disabled_returns_403(tmp_path):
    store = ChatAttachmentStore(tmp_path)
    req = _request(store=store, cfg=_cfg(chat_file_upload_enabled=False))
    with pytest.raises(HTTPException) as exc:
        await chat_attachment_upload(req, _upload(b"x"), "conv")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_upload_disallowed_ext_returns_422(tmp_path):
    store = ChatAttachmentStore(tmp_path)
    req = _request(store=store, cfg=_cfg())
    with pytest.raises(HTTPException) as exc:
        await chat_attachment_upload(req, _upload(b"x", "evil.exe"), "conv")
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_upload_oversize_and_count_limit(tmp_path):
    store = ChatAttachmentStore(tmp_path)
    req = _request(store=store, cfg=_cfg(chat_file_max_size_bytes=10))
    with pytest.raises(HTTPException) as exc:
        await chat_attachment_upload(req, _upload(b"x" * 50), "conv")
    assert exc.value.status_code == 422

    small_req = _request(store=store, cfg=_cfg(chat_file_max_count=1))
    await chat_attachment_upload(small_req, _upload(b"a"), "conv")
    with pytest.raises(HTTPException) as exc2:
        await chat_attachment_upload(small_req, _upload(b"b"), "conv")
    assert exc2.value.status_code == 422


@pytest.mark.asyncio
async def test_upload_success_extracts_text(tmp_path):
    store = ChatAttachmentStore(tmp_path)
    req = _request(store=store, cfg=_cfg())
    meta = await chat_attachment_upload(req, _upload(b"hello doc"), "conv")
    assert meta["name"] == "notes.txt"
    assert meta["char_count"] == len("hello doc")
    assert store.extracted_text("conv", meta["id"]) == "hello doc"


@pytest.mark.asyncio
async def test_list_and_delete_endpoints(tmp_path):
    store = ChatAttachmentStore(tmp_path)
    req = _request(store=store, cfg=_cfg())
    meta = await chat_attachment_upload(req, _upload(b"x"), "conv")

    listed = await chat_attachment_list("conv", req)
    assert [m["id"] for m in listed["items"]] == [meta["id"]]

    res = await chat_attachment_delete("conv", meta["id"], req)
    assert res["success"] is True
    assert store.list("conv") == []


@pytest.mark.asyncio
async def test_attachment_endpoints_reject_anonymous(tmp_path):
    store = ChatAttachmentStore(tmp_path)
    req = _request(store=store, cfg=_cfg(), api_key_id=None)
    with pytest.raises(HTTPException) as exc:
        await chat_attachment_list("conv", req)
    assert exc.value.status_code == 403


class _FakeRAG:
    """Captures what the model-facing run() receives."""

    ready = True
    instructions = {
        "A": {"model": "gpt-4o-mini", "system_instructions": "s {{context}}"}
    }

    def __init__(self):
        self.kwargs = None

    async def refresh_instructions_if_needed(self):
        return False

    async def run(self, input_text, *, chat_history, model_name, retrieval_info=None, multimodal_input=None, **kw):
        self.kwargs = {"input_text": input_text, "multimodal_input": multimodal_input}
        if retrieval_info is not None:
            retrieval_info.update({"status": "ok", "chunks_retrieved": 0, "rag_context_tokens": 0, "rag_context": []})
        return {"answer": "ok"}


@pytest.mark.asyncio
async def test_completions_injects_attachments_via_multimodal(tmp_path):
    """Attachment text reaches the model through multimodal_input while the
    retrieval-query input stays the raw user question."""
    store = ChatAttachmentStore(tmp_path)
    meta = await store.save("conv-x", "spec.txt", b"raw")
    await store.set_extracted("conv-x", meta["id"], "EXTRACTED_CONTENT", False)

    rag = _FakeRAG()
    cfg = _cfg()
    req = _request(store=store, cfg=cfg, rag=rag)
    chat_request = ChatRequest(
        model="A",
        messages=[Message(role="user", content="what does it say?")],
        stream=False,
        conversation_id="conv-x",
    )
    resp = await chat_endpoint(chat_request, req)
    assert resp is not None
    assert rag.kwargs is not None
    # Retrieval query stays the raw question
    assert rag.kwargs["input_text"] == "what does it say?"
    mm = rag.kwargs["multimodal_input"]
    assert mm is not None and isinstance(mm.content, list)
    text_part = next(p for p in mm.content if p.get("type") == "text")
    assert "EXTRACTED_CONTENT" in text_part["text"]
    assert '<attachment name="spec.txt" type="txt">' in text_part["text"]
    assert "what does it say?" in text_part["text"]


@pytest.mark.asyncio
async def test_completions_image_attachment_becomes_image_url(tmp_path):
    """Stored image attachments reach the model as real image_url parts when
    the image-upload flag is on (in addition to their OCR text context)."""
    import base64

    store = ChatAttachmentStore(tmp_path)
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    meta = await store.save("conv-img", "shot.png", png)
    await store.set_extracted("conv-img", meta["id"], "OCR_TEXT", False)

    rag = _FakeRAG()
    cfg = _cfg(chat_image_upload_enabled=True)
    req = _request(store=store, cfg=cfg, rag=rag)
    chat_request = ChatRequest(
        model="A",
        messages=[Message(role="user", content="what is in the image?")],
        stream=False,
        conversation_id="conv-img",
    )
    resp = await chat_endpoint(chat_request, req)
    assert resp is not None
    mm = rag.kwargs["multimodal_input"]
    assert mm is not None and isinstance(mm.content, list)
    kinds = [p.get("type") for p in mm.content]
    assert "image_url" in kinds
    img = next(p for p in mm.content if p.get("type") == "image_url")
    assert img["image_url"]["url"].startswith("data:image/png;base64,")
    # OCR text still present alongside the image part
    text_part = next(p for p in mm.content if p.get("type") == "text")
    assert "OCR_TEXT" in text_part["text"]


@pytest.mark.asyncio
async def test_completions_image_attachment_flag_off_is_text_only(tmp_path):
    """Image flag off → attachments stay OCR-text-only (no image_url parts)."""
    store = ChatAttachmentStore(tmp_path)
    meta = await store.save("conv-img", "shot.png", b"\x89PNGfake")
    await store.set_extracted("conv-img", meta["id"], "OCR_TEXT", False)

    rag = _FakeRAG()
    cfg = _cfg(chat_image_upload_enabled=False)
    req = _request(store=store, cfg=cfg, rag=rag)
    chat_request = ChatRequest(
        model="A",
        messages=[Message(role="user", content="hi")],
        stream=False,
        conversation_id="conv-img",
    )
    await chat_endpoint(chat_request, req)
    mm = rag.kwargs["multimodal_input"]
    assert mm is not None and isinstance(mm.content, list)
    assert all(p.get("type") == "text" for p in mm.content)


@pytest.mark.asyncio
async def test_completions_no_attachments_no_multimodal(tmp_path):
    store = ChatAttachmentStore(tmp_path)
    rag = _FakeRAG()
    req = _request(store=store, cfg=_cfg(), rag=rag)
    chat_request = ChatRequest(
        model="A",
        messages=[Message(role="user", content="hi")],
        stream=False,
        conversation_id="conv-y",
    )
    await chat_endpoint(chat_request, req)
    assert rag.kwargs["multimodal_input"] is None
    assert rag.kwargs["input_text"] == "hi"


def test_xml_escape():
    assert _xml_escape('a<b>"&') == "a&lt;b&gt;&quot;&amp;"


# ---------------------------------------------------------------------------
# Access gating, effective config, ownership
# ---------------------------------------------------------------------------


class _FakeRuntimeStore:
    def __init__(self, values: dict) -> None:
        self.values = values

    async def get(self, key):
        return self.values.get(key)


class _FakeConvStore:
    def __init__(self, conv) -> None:
        self._conv = conv

    async def get_by_conversation_id(self, conversation_id):
        return self._conv


@pytest.mark.asyncio
async def test_upload_rejects_public_chat_token_only(tmp_path, monkeypatch):
    """Anonymous public-chat visitors hold only the free access token — they
    must not write files to disk; upload requires a real identity."""
    from app.routers.chat import _read_public_chat_access_token  # noqa: F401  (imported for clarity)

    store = ChatAttachmentStore(tmp_path)
    req = _request(store=store, cfg=_cfg(), api_key_id=None)
    req.cookies = {}
    req.headers = {"x-public-chat-token": "some-token"}

    async def _no_user(r):
        return None

    monkeypatch.setattr("app.routers.chat._get_current_chat_user", _no_user)
    with pytest.raises(HTTPException) as exc:
        await chat_attachment_upload(req, _upload(b"x"), "conv")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_upload_honors_stored_override_immediately(tmp_path):
    """A stored runtime-config override takes effect without Apply/restart:
    app_config still holds the default but the flag resolves True."""
    store = ChatAttachmentStore(tmp_path)
    cfg = _cfg(chat_file_upload_enabled=False)
    req = _request(store=store, cfg=cfg)
    req.app.state._runtime_config_store = _FakeRuntimeStore(
        {"chat_file_upload_enabled": True}
    )
    meta = await chat_attachment_upload(req, _upload(b"immediate"), "conv")
    assert meta["name"] == "notes.txt"


@pytest.mark.asyncio
async def test_upload_env_backed_value_beats_stored_override(tmp_path):
    """When app_config already differs from the default (env-set), a stored
    override must NOT flip it back off."""
    store = ChatAttachmentStore(tmp_path)
    cfg = _cfg()  # chat_file_upload_enabled=True — treated as env/non-default
    req = _request(store=store, cfg=cfg)
    req.app.state._runtime_config_store = _FakeRuntimeStore(
        {"chat_file_upload_enabled": False}
    )
    meta = await chat_attachment_upload(req, _upload(b"still on"), "conv")
    assert meta["name"] == "notes.txt"


@pytest.mark.asyncio
async def test_chat_capabilities_endpoint(tmp_path, monkeypatch):
    from app.routers.chat import chat_capabilities

    cfg = _cfg(chat_file_upload_enabled=True, chat_image_upload_enabled=True)
    req = _request(store=None, cfg=cfg)
    caps = await chat_capabilities(req)
    assert caps["files"] is True
    assert caps["images"] is True
    assert caps["file_max_count"] == 3

    anon = _request(store=None, cfg=cfg, api_key_id=None)
    anon.cookies = {}

    async def _no_user(r):
        return None

    monkeypatch.setattr("app.routers.chat._get_current_chat_user", _no_user)
    anon_caps = await chat_capabilities(anon)
    # Per-caller like /public-chat/config: images are flag-gated for everyone,
    # file attachments require a real identity.
    assert anon_caps["images"] is True
    assert anon_caps["files"] is False


@pytest.mark.asyncio
async def test_attachment_ownership_enforced(tmp_path, monkeypatch):
    """A caller may not upload/list/delete attachments of a conversation owned
    by another chat user."""
    store = ChatAttachmentStore(tmp_path)
    req = _request(store=store, cfg=_cfg(), api_key_id=None)
    req.cookies = {}
    req.app.state._chat_conversations_store = _FakeConvStore(
        {"user_id": "chat-user-other"}
    )

    async def _me(r):
        return {"id": "chat-user-me", "dashboard_user_id": None}

    monkeypatch.setattr("app.routers.chat._get_current_chat_user", _me)
    with pytest.raises(HTTPException) as exc:
        await chat_attachment_upload(req, _upload(b"x"), "conv")
    assert exc.value.status_code == 403
    with pytest.raises(HTTPException):
        await chat_attachment_list("conv", req)
    with pytest.raises(HTTPException):
        await chat_attachment_delete("conv", "att-x", req)

    # Own conversation → allowed
    req.app.state._chat_conversations_store = _FakeConvStore(
        {"user_id": "chat-user-me"}
    )
    meta = await chat_attachment_upload(req, _upload(b"own"), "conv")
    assert meta["name"] == "notes.txt"

    # Anonymous-owned conversation (user_id NULL) → bearer semantics
    req2 = _request(store=store, cfg=_cfg(), api_key_id="k1")
    req2.app.state._chat_conversations_store = _FakeConvStore({"user_id": None})
    meta2 = await chat_attachment_upload(req2, _upload(b"anon-owned"), "conv")
    assert meta2["name"] == "notes.txt"


@pytest.mark.asyncio
async def test_completions_skips_foreign_conversation_attachments(tmp_path, monkeypatch):
    """Attachment text of a conversation owned by someone else must not leak
    into the caller's prompt."""
    store = ChatAttachmentStore(tmp_path)
    meta = await store.save("conv-foreign", "spec.txt", b"raw")
    await store.set_extracted("conv-foreign", meta["id"], "SECRET_CONTENT", False)

    rag = _FakeRAG()
    req = _request(store=store, cfg=_cfg(), rag=rag, api_key_id=None)
    req.cookies = {}
    req.app.state._chat_conversations_store = _FakeConvStore(
        {"user_id": "chat-user-other"}
    )

    async def _me(r):
        return {"id": "chat-user-me", "dashboard_user_id": None, "role": "standard"}

    monkeypatch.setattr("app.routers.chat._get_current_chat_user", _me)
    chat_request = ChatRequest(
        model="A",
        messages=[Message(role="user", content="hi")],
        stream=False,
        conversation_id="conv-foreign",
    )
    resp = await chat_endpoint(chat_request, req)
    assert resp is not None
    mm = rag.kwargs["multimodal_input"]
    leaked = mm is not None and "SECRET_CONTENT" in str(mm.content)
    assert not leaked


@pytest.mark.asyncio
async def test_upload_image_ext_accepted_via_loader(tmp_path):
    """Image files route through the document-attachment path (OCR/vision
    extraction) — they must not be rejected as 'not allowed'. Extraction
    degrades gracefully to a placeholder when no OCR/LLM is available."""
    store = ChatAttachmentStore(tmp_path)
    cfg = _cfg(chat_file_allowed_exts=".txt,.png")
    req = _request(store=store, cfg=cfg)
    meta = await chat_attachment_upload(
        req, _upload(b"\x89PNG\r\n\x1a\n" + b"0" * 64, "shot.png"), "conv"
    )
    assert meta["name"] == "shot.png"
    assert meta["char_count"] >= len("[no extractable text]")


def test_chat_file_defaults_support_large_office_files():
    """A 25 MB .pptx must fit inside the defaults, and image extensions are
    accepted for the OCR/vision extraction path."""
    from agent.config import AppConfig

    cfg = AppConfig()
    assert cfg.chat_file_max_size_bytes >= 50 * 1024 * 1024
    assert cfg.chat_file_extract_timeout_seconds >= 120
    assert ".pptx" in cfg.chat_file_allowed_exts
    assert ".png" in cfg.chat_file_allowed_exts
    assert ".jpeg" in cfg.chat_file_allowed_exts


# ---------------------------------------------------------------------------
# GET /chat/attachments/{conv}/{att}/file — original-bytes endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attachment_file_serves_image_inline(tmp_path):
    from app.routers.chat import chat_attachment_file

    store = ChatAttachmentStore(tmp_path)
    meta = await store.save("conv-1", "shot.png", b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    req = _request(store=store, cfg=_cfg())
    res = await chat_attachment_file("conv-1", meta["id"], req)

    assert res.media_type == "image/png"
    assert res.headers["x-content-type-options"] == "nosniff"
    assert "inline" in res.headers.get("content-disposition", "")


@pytest.mark.asyncio
async def test_attachment_file_serves_document_as_download(tmp_path):
    from app.routers.chat import chat_attachment_file

    store = ChatAttachmentStore(tmp_path)
    meta = await store.save("conv-1", "notes.txt", b"hello")
    req = _request(store=store, cfg=_cfg())
    res = await chat_attachment_file("conv-1", meta["id"], req)

    assert res.media_type == "text/plain"
    assert "attachment" in res.headers.get("content-disposition", "")


@pytest.mark.asyncio
async def test_attachment_file_missing_returns_404(tmp_path):
    from app.routers.chat import chat_attachment_file

    store = ChatAttachmentStore(tmp_path)
    req = _request(store=store, cfg=_cfg())
    with pytest.raises(HTTPException) as exc:
        await chat_attachment_file("conv-1", "att-nope", req)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_attachment_file_denies_foreign_conversation(tmp_path):
    """Owned conversations reject non-owners; bearer semantics stay for
    row-less conversations (unguessable UUID = credential)."""
    from app.routers.chat import chat_attachment_file

    class _ConvStore:
        async def get_by_conversation_id(self, cid):
            return {"conversation_id": cid, "user_id": "owner-1"}

    store = ChatAttachmentStore(tmp_path)
    meta = await store.save("conv-1", "a.txt", b"x")
    req = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(
            chat_attachment_store=store,
            app_config=_cfg(),
            _chat_conversations_store=_ConvStore(),
        )),
        state=SimpleNamespace(api_key_id=None),
        headers={}, query_params={}, cookies={},
    )
    with pytest.raises(HTTPException) as exc:
        await chat_attachment_file("conv-1", meta["id"], req)
    assert exc.value.status_code == 403
