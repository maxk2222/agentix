"""Image generation builtin tool.

Calls ``POST {provider}/v1/images/generations`` on the OpenAI-compatible
endpoint selected in the tool settings (``provider_id`` + ``model``, falling
back to the default provider / environment LLM). Generated images are stored
as conversation artifacts so they appear as preview cards in the chat; an
optional copy can be dropped into the agent workspace.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from agent.tools.builtin_tools.image_common import (
    _clamp_float,
    _clamp_int,
    _err,
    decode_image_items,
    deliver_images,
    post_json,
    resolve_endpoint,
    resolve_model,
)
from app.agent_workspace import AgentWorkspaceStore
from app.artifacts import ArtifactStore

logger = logging.getLogger(__name__)


class ImageGenerateTool:
    """Generate image(s) from a text prompt via an OpenAI-compatible API."""

    name = "image_generate"
    description = (
        "Generate image file(s) from a text prompt using the configured image "
        "endpoint (OpenAI-compatible /v1/images API — model, endpoint and "
        "defaults are set in the tool settings, NOT here). The finished image "
        "is attached to the chat as a downloadable preview card. "
        "Use ONLY 'prompt' (required — describe the desired image in detail) "
        "and optionally 'n' (1-4 images, default 1), 'size' (e.g. '1024x1024', "
        "'1024x1536', '1536x1024' — valid values depend on the model), "
        "'quality', 'filename' (output name, extension auto-detected) and "
        "'workspace_path' (also save a copy into the agent workspace). "
        "To modify an existing image use image_edit instead."
    )

    def __init__(self, app: Any) -> None:
        self._app = app
        data_dir = Path(app.config.data_dir)
        self._artifacts = ArtifactStore(data_dir / "artifacts")
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
                        "prompt": {
                            "type": "string",
                            "description": "Text description of the image to generate.",
                        },
                        "n": {
                            "type": "integer",
                            "description": "Number of images to generate (1-4, default 1). Not all models support n>1.",
                            "default": 1,
                        },
                        "size": {
                            "type": "string",
                            "description": "Image size, e.g. '1024x1024' — valid values depend on the model. Omit for the configured/model default.",
                        },
                        "quality": {
                            "type": "string",
                            "description": "Quality tier, e.g. 'auto', 'low', 'medium', 'high', 'standard', 'hd' — valid values depend on the model. Omit for the configured default.",
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
                    "required": ["prompt"],
                },
            },
        }

    def call_conflict_key(self, args: Dict[str, Any]) -> Optional[str]:
        """Serialize same-path workspace copies inside a parallel batch."""
        ws = str(args.get("workspace_path") or "").strip().replace("\\", "/")
        if ws:
            return f"ws:{ws}"
        return None

    async def execute(
        self,
        prompt: Optional[str] = None,
        n: Optional[int] = None,
        size: Optional[str] = None,
        quality: Optional[str] = None,
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
        try:
            ep = await resolve_endpoint(self._app, cfg)
            model = resolve_model(cfg)
        except RuntimeError as exc:
            return _err(str(exc))

        n_eff = _clamp_int(n if n is not None else cfg.get("n"), 1, 4, 1)
        body: Dict[str, Any] = {"model": model, "prompt": prompt, "n": n_eff}
        for key, arg in (("size", size), ("quality", quality)):
            value = str(arg if arg is not None else (cfg.get(key) or "")).strip()
            if value:
                body[key] = value
        for key in ("background", "output_format", "style", "response_format"):
            value = str(cfg.get(key) or "").strip()
            if value:
                body[key] = value
        extra = cfg.get("extra_body")
        if isinstance(extra, dict):
            for k, v in extra.items():
                body.setdefault(str(k), v)

        timeout_s = _clamp_float(cfg.get("timeout_seconds"), 5, 900, 300)
        try:
            data = await post_json(ep, "/images/generations", body, timeout_s)
            images, revised = await decode_image_items(data, cfg, timeout_s)
        except RuntimeError as exc:
            return _err(str(exc))

        try:
            entries, ws_meta, ws_note, uris = await deliver_images(
                ctx, cfg, images,
                artifacts=self._artifacts,
                ws_store=self._ws,
                filename=filename,
                slug_base=prompt,
                workspace_path=workspace_path,
            )
        except RuntimeError as exc:
            return _err(str(exc))

        names = "', '".join(e["name"] for e in entries)
        total_kb = sum(int(e.get("size") or 0) for e in entries) / 1024
        msg = (
            f"Generated {len(entries)} image(s): '{names}' "
            f"({total_kb:.0f} KB total, model '{model}')."
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
