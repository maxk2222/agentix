"""Image editing builtin tool.

Calls ``POST {provider}/v1/images/edits`` (multipart/form-data, the classic
OpenAI-compatible schema that compatible servers — LiteLLM, LocalAI, etc. —
implement). Input images come from chat uploads, conversation artifacts, the
agent workspace, URLs or inline data URIs; results are stored as artifacts
like ``image_generate`` does.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tools.builtin_tools.image_common import (
    _clamp_float,
    _clamp_int,
    _err,
    decode_image_items,
    deliver_images,
    post_multipart,
    resolve_endpoint,
    resolve_image_source,
    resolve_model,
    sniff_image,
)
from app.agent_workspace import AgentWorkspaceStore
from app.artifacts import ArtifactStore
from app.chat_attachments import ChatAttachmentStore

logger = logging.getLogger(__name__)

_MAX_SOURCE_IMAGES = 8


def _looks_like_image(data: bytes, mime: str) -> bool:
    """Cheap sanity gate: reject sources that are clearly not images while
    still letting exotic-but-valid formats through (the endpoint decides)."""
    if sniff_image(data) is not None:
        return True
    return (mime or "").lower().startswith("image/")


def _upload_name(index: int, data: bytes, mime: str) -> str:
    """Multipart filename — the extension must match the real format or
    strict endpoints reject the part."""
    sniffed = sniff_image(data)
    if sniffed:
        ext = sniffed[0]
    elif (mime or "").startswith("image/"):
        ext = f".{mime.split('/', 1)[1].split('+')[0]}"
    else:
        ext = ".png"
    return f"image{'' if index == 0 else f'-{index + 1}'}{ext}"


class ImageEditTool:
    """Edit/transform existing image(s) with a prompt via /v1/images/edits."""

    name = "image_edit"
    description = (
        "Edit or transform existing image(s) with a text prompt using the "
        "configured image endpoint (OpenAI-compatible /v1/images/edits API — "
        "model, endpoint and defaults are set in the tool settings, NOT here). "
        "The result is attached to the chat as a downloadable preview card. "
        "Use ONLY 'image' (required — one source string or a list of them) "
        "and 'prompt' (required — describe the change). "
        "A source can be: 'attachment:<name>' (or 'upload:<name>') for a file "
        "the user uploaded to this chat, 'artifact:<name>' for a file "
        "generated earlier in this conversation, 'workspace:<path>' (or "
        "'file:<path>') for a file in the agent workspace, 'https://...' for a "
        "remote image, 'data:<mime>;base64,<data>' for inline content, or a "
        "bare filename (tries artifact, then upload, then workspace). "
        "Optional: 'mask' (same source syntax — transparent regions mark the "
        "edit area where the model supports it), 'n' (1-4), 'size', "
        "'filename', 'workspace_path'. "
        "NOTE: the source image content is sent to the external image endpoint."
    )

    def __init__(self, app: Any) -> None:
        self._app = app
        data_dir = Path(app.config.data_dir)
        self._artifacts = ArtifactStore(data_dir / "artifacts")
        self._att = ChatAttachmentStore(data_dir / "chat_uploads")
        self._ws = AgentWorkspaceStore()

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
                        "image": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Source image(s) to edit — a single source string "
                                "or a list. Prefixes: attachment:/upload:, "
                                "artifact:, workspace:/file:, https://, data:. "
                                "A bare filename tries artifact, upload, workspace."
                            ),
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Text description of the desired edit.",
                        },
                        "mask": {
                            "type": "string",
                            "description": "Optional mask image (same source syntax as 'image'); transparent regions mark the editable area on models that support masking.",
                        },
                        "n": {
                            "type": "integer",
                            "description": "Number of edit variants (1-4, default 1).",
                            "default": 1,
                        },
                        "size": {
                            "type": "string",
                            "description": "Output size, e.g. '1024x1024' — valid values depend on the model. Omit for the configured/model default.",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Optional output filename; the extension is forced to the actual image format.",
                        },
                        "workspace_path": {
                            "type": "string",
                            "description": "Optional path relative to the agent workspace root — when set, a copy is also written there (requires workspace enabled).",
                        },
                    },
                    "required": ["image", "prompt"],
                },
            },
        }

    def call_conflict_key(self, args: Dict[str, Any]) -> Optional[str]:
        ws = str(args.get("workspace_path") or "").strip().replace("\\", "/")
        if ws:
            return f"ws:{ws}"
        return None

    async def execute(
        self,
        image: Optional[Any] = None,
        prompt: Optional[str] = None,
        mask: Optional[str] = None,
        n: Optional[int] = None,
        size: Optional[str] = None,
        filename: Optional[str] = None,
        workspace_path: Optional[str] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = context or {}
        cfg = ctx.get("tool_config") or {}
        prompt = (prompt or "").strip()
        if not prompt:
            return _err("prompt is required")

        specs = image if isinstance(image, list) else ([image] if image else [])
        specs = [str(s).strip() for s in specs if str(s or "").strip()]
        if not specs:
            return _err(
                "image is required — pass a source like 'attachment:photo.png', "
                "'artifact:img.png', 'workspace:pic.png', an https URL or a data URI"
            )
        if len(specs) > _MAX_SOURCE_IMAGES:
            return _err(f"too many source images ({len(specs)} > {_MAX_SOURCE_IMAGES})")

        try:
            ep = await resolve_endpoint(self._app, cfg)
            model = resolve_model(cfg)
        except RuntimeError as exc:
            return _err(str(exc))
        timeout_s = _clamp_float(cfg.get("timeout_seconds"), 5, 900, 300)

        sources: List[tuple] = []
        try:
            for spec in specs:
                sources.append(
                    await resolve_image_source(
                        spec, ctx, cfg,
                        artifacts=self._artifacts,
                        attachments=self._att,
                        ws_store=self._ws,
                        timeout_s=timeout_s,
                    )
                )
        except RuntimeError as exc:
            return _err(str(exc))

        for data, mime, label in sources:
            if not _looks_like_image(data, mime):
                return _err(
                    f"source {label} does not look like an image "
                    f"(mime '{mime or 'unknown'}') — image_edit needs raster "
                    "images (png/jpg/webp/gif/...)"
                )

        mask_src: Optional[tuple] = None
        if mask:
            try:
                mask_src = await resolve_image_source(
                    mask, ctx, cfg,
                    artifacts=self._artifacts,
                    attachments=self._att,
                    ws_store=self._ws,
                    timeout_s=timeout_s,
                )
            except RuntimeError as exc:
                return _err(f"mask: {exc}")
            if not _looks_like_image(mask_src[0], mask_src[1]):
                return _err(f"mask source {mask_src[2]} does not look like an image")

        n_eff = _clamp_int(n if n is not None else cfg.get("n"), 1, 4, 1)
        fields: Dict[str, str] = {"model": model, "prompt": prompt, "n": str(n_eff)}
        size_v = str(size if size is not None else (cfg.get("size") or "")).strip()
        if size_v:
            fields["size"] = size_v
        for key in ("background", "output_format", "response_format"):
            value = str(cfg.get(key) or "").strip()
            if value:
                fields[key] = value
        extra = cfg.get("extra_body")
        if isinstance(extra, dict):
            for k, v in extra.items():
                fields.setdefault(
                    str(k), v if isinstance(v, str) else json.dumps(v)
                )

        files: List[tuple] = []
        multi = len(sources) > 1
        for i, (data, mime, _label) in enumerate(sources):
            field = "image[]" if multi else "image"
            files.append(
                (field, _upload_name(i, data, mime), data,
                 mime if (mime or "").startswith("image/") else "application/octet-stream")
            )
        if mask_src is not None:
            mdata, mmime, _mlabel = mask_src
            mext = sniff_image(mdata)
            files.append(
                ("mask", f"mask{mext[0] if mext else '.png'}", mdata,
                 mmime if (mmime or "").startswith("image/") else "image/png")
            )

        try:
            data = await post_multipart(
                ep, "/images/edits", fields, files, timeout_s
            )
            images, revised = await decode_image_items(data, cfg, timeout_s)
        except RuntimeError as exc:
            return _err(str(exc))

        try:
            entries, ws_meta, ws_note, uris = await deliver_images(
                ctx, cfg, images,
                artifacts=self._artifacts,
                ws_store=self._ws,
                filename=filename,
                slug_base=f"edit {prompt}",
                workspace_path=workspace_path,
            )
        except RuntimeError as exc:
            return _err(str(exc))

        names = "', '".join(e["name"] for e in entries)
        total_kb = sum(int(e.get("size") or 0) for e in entries) / 1024
        msg = (
            f"Edited {len(sources)} source image(s) into {len(entries)} "
            f"result(s): '{names}' ({total_kb:.0f} KB total, model '{model}')."
            f"{ws_note} The user can preview and download "
            f"{'it' if len(entries) == 1 else 'them'} in the chat."
        )
        if revised:
            msg += f" Endpoint revised prompt: {revised}"
        metadata: Dict[str, Any] = {"files": entries}
        if ws_meta:
            metadata["workspace_path"] = ws_meta
        result: Dict[str, Any] = {"ok": True, "content": msg, "metadata": metadata}
        if uris:
            result["images"] = uris
        return result
