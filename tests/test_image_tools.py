"""Tests for image_generate / image_edit builtin tools + image_common helpers."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.tools.builtin_tools.image_common as ic
import agent.tools.builtin_tools.image_edit as ie
import agent.tools.builtin_tools.image_generate as ig
from agent.tools.builtin_tools.image_edit import ImageEditTool
from agent.tools.builtin_tools.image_generate import ImageGenerateTool
from app.artifacts import ArtifactStore
from app.chat_attachments import ChatAttachmentStore
from app.llm_providers import LLMProvider

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 32
WEBP_BYTES = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 24


# ---------------------------------------------------------------------------
# fixtures / fakes
# ---------------------------------------------------------------------------

def _provider(
    pid: str = "img-prov",
    name: str = "ImageProv",
    *,
    enabled: bool = True,
    is_default: bool = False,
    base_url: str = "https://img.example.com/api",
) -> LLMProvider:
    return LLMProvider(
        id=pid, name=name, base_url=base_url,
        enabled=enabled, is_default=is_default,
    )


def _store(providers: Dict[str, LLMProvider], default: Any = None, key: str = "sk-test"):
    store = MagicMock()
    store.get_provider = AsyncMock(side_effect=lambda pid: providers.get(pid))
    store.get_api_key = AsyncMock(return_value=key)
    store.get_default_provider = AsyncMock(return_value=default)
    return store


def _app(tmp_path: Path, store: Any = None, *, env_llm: bool = False):
    return SimpleNamespace(
        config=SimpleNamespace(
            data_dir=tmp_path,
            llm_api_base="https://env-llm.example.com" if env_llm else "",
            llm_api_key="env-key" if env_llm else "",
        ),
        _provider_store=store,
    )


def _ctx(tmp_path: Path, cfg: Dict[str, Any] | None = None, **extra) -> Dict[str, Any]:
    ctx = {
        "conversation_id": "conv-1",
        "assistant_name": "tester",
        "tool_config": cfg or {},
    }
    ctx.update(extra)
    return ctx


def _b64_response(data: bytes = PNG_BYTES) -> Dict[str, Any]:
    return {"data": [{"b64_json": base64.b64encode(data).decode("ascii")}]}


@pytest.fixture
def gen_tool(tmp_path):
    return ImageGenerateTool(_app(tmp_path))


@pytest.fixture
def edit_tool(tmp_path):
    return ImageEditTool(_app(tmp_path))


# ---------------------------------------------------------------------------
# sniff_image
# ---------------------------------------------------------------------------

class TestSniff:
    def test_formats(self):
        assert ic.sniff_image(PNG_BYTES) == (".png", "image/png")
        assert ic.sniff_image(JPG_BYTES) == (".jpg", "image/jpeg")
        assert ic.sniff_image(WEBP_BYTES) == (".webp", "image/webp")
        assert ic.sniff_image(b"GIF89a" + b"\x00" * 10) == (".gif", "image/gif")
        assert ic.sniff_image(b"BM" + b"\x00" * 10) == (".bmp", "image/bmp")
        assert ic.sniff_image(b"II*\x00" + b"\x00" * 10) == (".tiff", "image/tiff")
        assert ic.sniff_image(b'{"not": "an image"}') is None
        assert ic.sniff_image(b"") is None


# ---------------------------------------------------------------------------
# endpoint + model resolution
# ---------------------------------------------------------------------------

class TestEndpointResolution:
    @pytest.mark.asyncio
    async def test_explicit_provider(self, tmp_path):
        prov = _provider()
        app = _app(tmp_path, _store({"img-prov": prov}))
        ep = await ic.resolve_endpoint(app, {"provider_id": "img-prov"})
        assert ep.base_url == "https://img.example.com/api/v1"
        assert ep.api_key == "sk-test"
        assert ep.headers()["Authorization"] == "Bearer sk-test"

    @pytest.mark.asyncio
    async def test_base_url_v1_not_doubled(self, tmp_path):
        prov = _provider(base_url="https://img.example.com/v1")
        app = _app(tmp_path, _store({"img-prov": prov}))
        ep = await ic.resolve_endpoint(app, {"provider_id": "img-prov"})
        assert ep.base_url == "https://img.example.com/v1"

    @pytest.mark.asyncio
    async def test_default_provider_fallback(self, tmp_path):
        prov = _provider(pid="d1", is_default=True)
        app = _app(tmp_path, _store({"d1": prov}, default=prov))
        ep = await ic.resolve_endpoint(app, {})
        assert ep.label == "provider 'ImageProv'"

    @pytest.mark.asyncio
    async def test_env_default_explicit(self, tmp_path):
        app = _app(tmp_path, _store({}), env_llm=True)
        ep = await ic.resolve_endpoint(app, {"provider_id": "env-default"})
        assert ep.base_url == "https://env-llm.example.com/v1"
        assert ep.api_key == "env-key"

    @pytest.mark.asyncio
    async def test_env_fallback_when_no_default(self, tmp_path):
        app = _app(tmp_path, _store({}), env_llm=True)
        ep = await ic.resolve_endpoint(app, {})
        assert ep.label == "environment LLM provider"

    @pytest.mark.asyncio
    async def test_missing_provider_error(self, tmp_path):
        app = _app(tmp_path, _store({}))
        with pytest.raises(RuntimeError, match="not found"):
            await ic.resolve_endpoint(app, {"provider_id": "nope"})

    @pytest.mark.asyncio
    async def test_disabled_provider_error(self, tmp_path):
        app = _app(tmp_path, _store({"p": _provider("p", enabled=False)}))
        with pytest.raises(RuntimeError, match="disabled"):
            await ic.resolve_endpoint(app, {"provider_id": "p"})

    @pytest.mark.asyncio
    async def test_nothing_configured_error(self, tmp_path):
        app = _app(tmp_path, _store({}))
        with pytest.raises(RuntimeError, match="no image endpoint"):
            await ic.resolve_endpoint(app, {})

    def test_model_required(self):
        with pytest.raises(RuntimeError, match="no image model"):
            ic.resolve_model({})
        assert ic.resolve_model({"model": "dall-e-3"}) == "dall-e-3"


# ---------------------------------------------------------------------------
# data URI decoding
# ---------------------------------------------------------------------------

class TestDataUri:
    def test_base64(self):
        uri = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"
        data, mime = ic._decode_data_uri(uri)
        assert data == PNG_BYTES and mime == "image/png"

    def test_malformed(self):
        with pytest.raises(RuntimeError, match="malformed"):
            ic._decode_data_uri("data:image/png")
        with pytest.raises(RuntimeError, match="malformed"):
            ic._decode_data_uri("data:image/png;base64,!!!not-b64!!!")


# ---------------------------------------------------------------------------
# image_generate
# ---------------------------------------------------------------------------

class TestImageGenerate:
    @pytest.mark.asyncio
    async def test_happy_path(self, tmp_path, gen_tool):
        app = _app(tmp_path, _store({"p": _provider("p")}))
        gen_tool._app = app
        cfg = {"provider_id": "p", "model": "gpt-image-1", "size": "1024x1024"}
        captured = {}

        async def fake_post(ep, path, body, timeout):
            captured.update(body)
            captured["path"] = path
            return _b64_response()

        with patch.object(ig, "post_json", side_effect=fake_post):
            res = await gen_tool.execute("a red fox", context=_ctx(tmp_path, cfg))

        assert res["ok"] is True
        assert captured["path"] == "/images/generations"
        assert captured["model"] == "gpt-image-1"
        assert captured["prompt"] == "a red fox"
        assert captured["size"] == "1024x1024"
        assert captured["n"] == 1
        files = res["metadata"]["files"]
        assert len(files) == 1
        assert files[0]["name"].endswith(".png")
        assert "images" not in res  # return_images off by default
        # artifact really stored
        store = ArtifactStore(tmp_path / "artifacts")
        entry = store.get_entry("conv-1", files[0]["name"])
        assert entry is not None and store.resolve("conv-1", files[0]["name"]).read_bytes() == PNG_BYTES

    @pytest.mark.asyncio
    async def test_arg_overrides_config(self, tmp_path, gen_tool):
        app = _app(tmp_path, _store({"p": _provider("p")}))
        gen_tool._app = app
        cfg = {"provider_id": "p", "model": "m", "size": "256x256", "n": 2}
        captured = {}

        async def fake_post(ep, path, body, timeout):
            captured.update(body)
            return _b64_response()

        with patch.object(ig, "post_json", side_effect=fake_post):
            await gen_tool.execute("x", size="512x512", n=3,
                                   context=_ctx(tmp_path, cfg))
        assert captured["size"] == "512x512"  # arg wins
        assert captured["n"] == 3

    @pytest.mark.asyncio
    async def test_extra_body_merged_but_args_win(self, tmp_path, gen_tool):
        app = _app(tmp_path, _store({"p": _provider("p")}))
        gen_tool._app = app
        cfg = {"provider_id": "p", "model": "m",
               "extra_body": {"moderation": "low", "size": "999x999"}}
        captured = {}

        async def fake_post(ep, path, body, timeout):
            captured.update(body)
            return _b64_response()

        with patch.object(ig, "post_json", side_effect=fake_post):
            await gen_tool.execute("x", context=_ctx(tmp_path, cfg))
        assert captured["moderation"] == "low"
        assert captured["size"] == "999x999"  # extra_body fills unset keys

    @pytest.mark.asyncio
    async def test_endpoint_error_honest(self, tmp_path, gen_tool):
        app = _app(tmp_path, _store({}))
        gen_tool._app = app
        res = await gen_tool.execute("x", context=_ctx(tmp_path, {}))
        assert res["ok"] is False
        assert "no image model" in res["error"] or "no image endpoint" in res["error"]

    @pytest.mark.asyncio
    async def test_api_error_honest(self, tmp_path, gen_tool):
        app = _app(tmp_path, _store({"p": _provider("p")}))
        gen_tool._app = app

        async def fail(ep, path, body, timeout):
            raise RuntimeError("image API /images/generations returned HTTP 500")

        with patch.object(ig, "post_json", side_effect=fail):
            res = await gen_tool.execute(
                "x", context=_ctx(tmp_path, {"provider_id": "p", "model": "m"}))
        assert res["ok"] is False and "HTTP 500" in res["error"]

    @pytest.mark.asyncio
    async def test_no_conversation_id(self, tmp_path, gen_tool):
        app = _app(tmp_path, _store({"p": _provider("p")}))
        gen_tool._app = app

        async def fake_post(ep, path, body, timeout):
            return _b64_response()

        ctx = {"tool_config": {"provider_id": "p", "model": "m"}}
        with patch.object(ig, "post_json", side_effect=fake_post):
            res = await gen_tool.execute("x", context=ctx)
        assert res["ok"] is False and "conversation_id" in res["error"]

    @pytest.mark.asyncio
    async def test_multiple_outputs_numbered(self, tmp_path, gen_tool):
        app = _app(tmp_path, _store({"p": _provider("p")}))
        gen_tool._app = app
        resp = {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode()}] * 3}

        async def fake_post(ep, path, body, timeout):
            return resp

        with patch.object(ig, "post_json", side_effect=fake_post):
            res = await gen_tool.execute(
                "x", filename="out.png",
                context=_ctx(tmp_path, {"provider_id": "p", "model": "m"}))
        names = [f["name"] for f in res["metadata"]["files"]]
        assert names == ["out.png", "out-2.png", "out-3.png"]

    @pytest.mark.asyncio
    async def test_return_images_flag(self, tmp_path, gen_tool):
        app = _app(tmp_path, _store({"p": _provider("p")}))
        gen_tool._app = app

        async def fake_post(ep, path, body, timeout):
            return _b64_response()

        cfg = {"provider_id": "p", "model": "m", "return_images": True}
        with patch.object(ig, "post_json", side_effect=fake_post):
            res = await gen_tool.execute("x", context=_ctx(tmp_path, cfg))
        assert res["images"] and res["images"][0].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_workspace_copy(self, tmp_path, gen_tool):
        app = _app(tmp_path, _store({"p": _provider("p")}))
        gen_tool._app = app
        ws_dir = tmp_path / "ws"
        ws_dir.mkdir()

        async def fake_post(ep, path, body, timeout):
            return _b64_response()

        ctx = _ctx(tmp_path, {"provider_id": "p", "model": "m"},
                   agent_workspace_dir=str(ws_dir), agent_workspace_cfg={})
        with patch.object(ig, "post_json", side_effect=fake_post):
            res = await gen_tool.execute(
                "x", workspace_path="img/pic.png", context=ctx)
        assert res["metadata"].get("workspace_path")
        assert (ws_dir / "img" / "pic.png").read_bytes() == PNG_BYTES

    def test_conflict_key(self, gen_tool):
        assert gen_tool.call_conflict_key({"workspace_path": "a/b.png"}) == "ws:a/b.png"
        assert gen_tool.call_conflict_key({}) is None


# ---------------------------------------------------------------------------
# image_edit — source resolution
# ---------------------------------------------------------------------------

async def _mk_attachment(att_root: Path, conv: str, name: str, data: bytes):
    store = ChatAttachmentStore(att_root)
    meta = await store.save(conv, name, data)
    return store, meta


class TestSourceResolution:
    @pytest.mark.asyncio
    async def test_data_uri(self, edit_tool, tmp_path):
        uri = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"
        data, mime, label = await ic.resolve_image_source(
            uri, _ctx(tmp_path), {},
            artifacts=edit_tool._artifacts, attachments=edit_tool._att,
            ws_store=edit_tool._ws, timeout_s=30)
        assert data == PNG_BYTES and mime == "image/png"

    @pytest.mark.asyncio
    async def test_artifact_source(self, edit_tool, tmp_path):
        store = ArtifactStore(tmp_path / "artifacts")
        await store.save("conv-1", "m", "pic.png", PNG_BYTES, "png")
        data, mime, label = await ic.resolve_image_source(
            "artifact:pic.png", _ctx(tmp_path), {},
            artifacts=edit_tool._artifacts, attachments=edit_tool._att,
            ws_store=edit_tool._ws, timeout_s=30)
        assert data == PNG_BYTES

    @pytest.mark.asyncio
    async def test_attachment_source(self, edit_tool, tmp_path):
        await _mk_attachment(tmp_path / "chat_uploads", "conv-1", "photo.jpg", JPG_BYTES)
        data, mime, label = await ic.resolve_image_source(
            "attachment:photo.jpg", _ctx(tmp_path), {},
            artifacts=edit_tool._artifacts, attachments=edit_tool._att,
            ws_store=edit_tool._ws, timeout_s=30)
        assert data == JPG_BYTES and mime == "image/jpeg"

    @pytest.mark.asyncio
    async def test_workspace_source(self, edit_tool, tmp_path):
        ws_dir = tmp_path / "ws"
        (ws_dir / "d").mkdir(parents=True)
        (ws_dir / "d" / "pic.png").write_bytes(PNG_BYTES)
        ctx = _ctx(tmp_path, agent_workspace_dir=str(ws_dir))
        data, mime, label = await ic.resolve_image_source(
            "workspace:d/pic.png", ctx, {},
            artifacts=edit_tool._artifacts, attachments=edit_tool._att,
            ws_store=edit_tool._ws, timeout_s=30)
        assert data == PNG_BYTES

    @pytest.mark.asyncio
    async def test_bare_name_fallback_artifact(self, edit_tool, tmp_path):
        store = ArtifactStore(tmp_path / "artifacts")
        await store.save("conv-1", "m", "gen.png", PNG_BYTES, "png")
        data, _m, _l = await ic.resolve_image_source(
            "gen.png", _ctx(tmp_path), {},
            artifacts=edit_tool._artifacts, attachments=edit_tool._att,
            ws_store=edit_tool._ws, timeout_s=30)
        assert data == PNG_BYTES

    @pytest.mark.asyncio
    async def test_unknown_source_error(self, edit_tool, tmp_path):
        with pytest.raises(RuntimeError, match="not found"):
            await ic.resolve_image_source(
                "nope.png", _ctx(tmp_path), {},
                artifacts=edit_tool._artifacts, attachments=edit_tool._att,
                ws_store=edit_tool._ws, timeout_s=30)

    def test_url_guards(self):
        cfg = {"allowed_hosts": ["*.example.com"], "allow_private_ips": True}
        with pytest.raises(RuntimeError, match="allowed_hosts"):
            ic._check_url_guards("https://evil.com/x.png", cfg, check_hosts=True)
        cfg2 = {"allow_private_ips": False}
        with pytest.raises(RuntimeError, match="non-public IP"):
            ic._check_url_guards("https://10.0.0.5/x.png", cfg2, check_hosts=True)
        # allowed host passes
        ic._check_url_guards("https://cdn.example.com/x.png", cfg, check_hosts=True)


# ---------------------------------------------------------------------------
# image_edit — full path
# ---------------------------------------------------------------------------

class TestImageEdit:
    @pytest.mark.asyncio
    async def test_requires_image(self, edit_tool, tmp_path):
        res = await edit_tool.execute(image=None, prompt="make it blue",
                                      context=_ctx(tmp_path, {"model": "m"}))
        assert res["ok"] is False and "image is required" in res["error"]

    @pytest.mark.asyncio
    async def test_happy_path_attachment(self, tmp_path, edit_tool):
        app = _app(tmp_path, _store({"p": _provider("p")}))
        edit_tool._app = app
        await _mk_attachment(tmp_path / "chat_uploads", "conv-1", "in.png", PNG_BYTES)
        captured = {}

        async def fake_multi(ep, path, fields, files, timeout):
            captured["fields"] = fields
            captured["files"] = files
            return _b64_response(JPG_BYTES)

        cfg = {"provider_id": "p", "model": "gpt-image-1"}
        with patch.object(ie, "post_multipart", side_effect=fake_multi):
            res = await edit_tool.execute(
                image=["attachment:in.png"], prompt="make it blue",
                context=_ctx(tmp_path, cfg))
        assert res["ok"] is True
        assert captured["fields"]["model"] == "gpt-image-1"
        assert captured["fields"]["prompt"] == "make it blue"
        files = captured["files"]
        assert len(files) == 1 and files[0][0] == "image" and files[0][2] == PNG_BYTES
        # result is the jpg the API returned — ext sniffed honestly
        out = res["metadata"]["files"][0]
        assert out["name"].endswith(".jpg")

    @pytest.mark.asyncio
    async def test_multi_source_uses_image_list_fields(self, tmp_path, edit_tool):
        app = _app(tmp_path, _store({"p": _provider("p")}))
        edit_tool._app = app
        await _mk_attachment(tmp_path / "chat_uploads", "conv-1", "a.png", PNG_BYTES)
        await _mk_attachment(tmp_path / "chat_uploads", "conv-1", "b.png", PNG_BYTES)
        captured = {}

        async def fake_multi(ep, path, fields, files, timeout):
            captured["files"] = files
            return _b64_response()

        with patch.object(ie, "post_multipart", side_effect=fake_multi):
            res = await edit_tool.execute(
                image=["attachment:a.png", "attachment:b.png"], prompt="merge",
                context=_ctx(tmp_path, {"provider_id": "p", "model": "m"}))
        assert res["ok"] is True
        assert all(f[0] == "image[]" for f in captured["files"])

    @pytest.mark.asyncio
    async def test_mask_sent_as_file(self, tmp_path, edit_tool):
        app = _app(tmp_path, _store({"p": _provider("p")}))
        edit_tool._app = app
        await _mk_attachment(tmp_path / "chat_uploads", "conv-1", "a.png", PNG_BYTES)
        await _mk_attachment(tmp_path / "chat_uploads", "conv-1", "m.png", PNG_BYTES)
        captured = {}

        async def fake_multi(ep, path, fields, files, timeout):
            captured["files"] = files
            return _b64_response()

        with patch.object(ie, "post_multipart", side_effect=fake_multi):
            await edit_tool.execute(
                image=["attachment:a.png"], mask="attachment:m.png", prompt="x",
                context=_ctx(tmp_path, {"provider_id": "p", "model": "m"}))
        names = [f[0] for f in captured["files"]]
        assert "image" in names and "mask" in names

    @pytest.mark.asyncio
    async def test_rejects_non_image_source(self, tmp_path, edit_tool):
        app = _app(tmp_path, _store({"p": _provider("p")}))
        edit_tool._app = app
        store = ArtifactStore(tmp_path / "artifacts")
        await store.save("conv-1", "m", "notes.txt", b"hello text", "txt")
        res = await edit_tool.execute(
            image=["artifact:notes.txt"], prompt="x",
            context=_ctx(tmp_path, {"provider_id": "p", "model": "m"}))
        assert res["ok"] is False and "does not look like an image" in res["error"]

    @pytest.mark.asyncio
    async def test_missing_source_honest_error(self, tmp_path, edit_tool):
        app = _app(tmp_path, _store({"p": _provider("p")}))
        edit_tool._app = app
        res = await edit_tool.execute(
            image=["artifact:missing.png"], prompt="x",
            context=_ctx(tmp_path, {"provider_id": "p", "model": "m"}))
        assert res["ok"] is False and "not found" in res["error"]


# ---------------------------------------------------------------------------
# decode_image_items
# ---------------------------------------------------------------------------

class TestDecodeItems:
    @pytest.mark.asyncio
    async def test_url_response_fetched(self):
        with patch.object(ic, "fetch_url", new=AsyncMock(return_value=(PNG_BYTES, "image/png"))):
            images, revised = await ic.decode_image_items(
                {"data": [{"url": "https://cdn.example.com/x.png", "revised_prompt": "rev"}]},
                {}, 30)
        assert images[0][0] == PNG_BYTES and revised == "rev"

    @pytest.mark.asyncio
    async def test_empty_data_error(self):
        with pytest.raises(RuntimeError, match="no images"):
            await ic.decode_image_items({"data": []}, {}, 30)

    @pytest.mark.asyncio
    async def test_neither_field_error(self):
        with pytest.raises(RuntimeError, match="neither b64_json nor url"):
            await ic.decode_image_items({"data": [{"foo": 1}]}, {}, 30)
