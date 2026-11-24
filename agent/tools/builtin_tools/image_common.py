"""Shared helpers for the image generation/editing builtin tools.

Both tools call an OpenAI-compatible ``/v1/images/*`` endpoint resolved
through the LLM provider store — the endpoint's base URL, API key and extra
headers live on the provider row (managed like any other LLM provider in the
admin UI), never inside the tool config. The tool config only references a
provider by id plus the image model and generation defaults; an empty
``provider_id`` falls back to the default provider and then to the
environment LLM endpoint, exactly like the chat model resolution does.

This module is not a tool itself — it is imported by ``image_generate`` and
``image_edit`` and skipped by builtin-tool discovery (no ``name``/
``tool_schema``/``execute`` on module level).
"""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import ipaddress
import json
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from app.agent_workspace import AgentWorkspaceStore
from app.artifacts import ArtifactStore, mime_for, sanitize_filename
from app.chat_attachments import ChatAttachmentStore
from app.llm_providers import build_provider_endpoint, normalize_base_url

logger = logging.getLogger(__name__)

# Per-input byte cap for image sources (data URIs, uploads, workspace files,
# fetched URLs). OpenAI caps dall-e-2 edits at 4 MB; gpt-image-1 allows ~50 MB —
# 32 MB covers realistic use without letting a multi-hundred-MB file blow up
# a multipart request.
_MAX_SOURCE_BYTES = 32 * 1024 * 1024
# Response images processed per call — also the runner's cap on returned
# ``images`` feedback entries.
_MAX_OUTPUT_IMAGES = 8
# Cap on the JSON response body — 8 base64 images of ~5 MB each fit easily;
# a runaway/compromised endpoint cannot OOM the process with resp.text().
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024

_SOURCE_SCHEMES = (
    ("attachment:", "attachment"),
    ("upload:", "attachment"),
    ("artifact:", "artifact"),
    ("workspace:", "workspace"),
    ("file:", "workspace"),
)


def _err(msg: str) -> Dict[str, Any]:
    """User-facing tool failure — the runner surfaces it as [Error: msg]."""
    return {"ok": False, "error": msg, "content": msg}


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^\w\-]+", "-", (text or "").strip().lower()).strip("-")
    return (s or "image")[:max_len].strip("-") or "image"


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


@dataclass
class ImageEndpoint:
    """Resolved wire target for one images API call.

    ``base_url`` already carries the ``/v1`` suffix; ``label`` describes the
    source for honest error messages. Secrets never appear in ``repr``.
    """

    base_url: str
    api_key: str = field(repr=False)
    extra_headers: Dict[str, str] = field(repr=False)
    label: str = ""

    def headers(self) -> Dict[str, str]:
        h = dict(self.extra_headers)
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h


def _env_endpoint(app: Any) -> Optional[ImageEndpoint]:
    """Endpoint from the legacy env LLM config (same fallback the chat uses)."""
    cfg = getattr(app, "config", None)
    base = normalize_base_url(getattr(cfg, "llm_api_base", "") or "")
    if not base:
        return None
    return ImageEndpoint(
        base_url=f"{base}/v1",
        api_key=(getattr(cfg, "llm_api_key", "") or "").strip(),
        extra_headers={},
        label="environment LLM provider",
    )


async def resolve_endpoint(app: Any, cfg: Dict[str, Any]) -> ImageEndpoint:
    """Pick the images endpoint.

    Order: ``provider_id`` config/env pin (``env-default`` selects the env
    fallback explicitly) → enabled default provider → env LLM config. Raises
    RuntimeError with a setup hint when nothing is reachable.
    """
    store = getattr(app, "_provider_store", None)
    pid = str(
        cfg.get("provider_id") or os.environ.get("IMAGE_PROVIDER_ID") or ""
    ).strip()

    if pid == "env-default":
        ep = _env_endpoint(app)
        if ep is None:
            raise RuntimeError(
                "provider_id 'env-default' selected but no environment LLM "
                "endpoint (llm_api_base) is configured"
            )
        return ep

    provider = None
    if pid:
        if store is None:
            raise RuntimeError(
                "provider store unavailable — cannot resolve provider_id "
                f"'{pid}' for the image endpoint"
            )
        provider = await store.get_provider(pid)
        if provider is None:
            raise RuntimeError(
                f"image provider '{pid}' not found — pick an existing provider "
                "in the tool settings"
            )
        if not provider.enabled:
            raise RuntimeError(
                f"image provider '{provider.name}' ({pid}) is disabled"
            )
    elif store is not None:
        try:
            provider = await store.get_default_provider()
        except Exception:
            logger.warning("image tools: default provider lookup failed", exc_info=True)
            provider = None
        if provider is not None and not provider.enabled:
            provider = None

    if provider is not None:
        api_key = await store.get_api_key(provider.id)
        ep = build_provider_endpoint(provider, api_key, fallback_key="")
        return ImageEndpoint(
            base_url=ep.base_url,
            api_key=ep.api_key,
            extra_headers=dict(ep.extra_headers or {}),
            label=f"provider '{provider.name}'",
        )

    ep = _env_endpoint(app)
    if ep is not None:
        return ep
    raise RuntimeError(
        "no image endpoint configured — set 'provider_id' in the tool settings "
        "to a provider serving /v1/images, mark a provider as default, or set "
        "the IMAGE_PROVIDER_ID environment variable"
    )


def resolve_model(cfg: Dict[str, Any]) -> str:
    """Image model id — config, then IMAGE_MODEL env, then an honest error."""
    model = str(cfg.get("model") or os.environ.get("IMAGE_MODEL") or "").strip()
    if not model:
        raise RuntimeError(
            "no image model configured — set 'model' in the tool settings "
            "(e.g. gpt-image-1 or the model name your endpoint serves)"
        )
    return model


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


async def _bounded_text(resp: aiohttp.ClientResponse, path: str) -> str:
    raw = await resp.content.read(_MAX_RESPONSE_BYTES + 1)
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise RuntimeError(
            f"image API {path} response exceeds "
            f"{_MAX_RESPONSE_BYTES // (1024 * 1024)} MB — refusing to parse"
        )
    return raw.decode("utf-8", errors="replace")


async def post_json(
    ep: ImageEndpoint, path: str, body: Dict[str, Any], timeout_s: float
) -> Dict[str, Any]:
    url = f"{ep.base_url}{path}"
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=body, headers=ep.headers()) as resp:
                text = await _bounded_text(resp, path)
    except asyncio.TimeoutError:
        raise RuntimeError(f"image API {path} timed out after {timeout_s:.0f}s")
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"image API {path} request failed: {exc}")
    if resp.status != 200:
        raise RuntimeError(
            f"image API {path} on {ep.label} returned HTTP {resp.status}: {text[:400]}"
        )
    try:
        return json.loads(text)
    except ValueError:
        raise RuntimeError(
            f"image API {path} returned a non-JSON response: {text[:200]}"
        )


async def post_multipart(
    ep: ImageEndpoint,
    path: str,
    fields: Dict[str, str],
    files: List[Tuple[str, str, bytes, str]],
    timeout_s: float,
) -> Dict[str, Any]:
    """POST multipart/form-data. ``files`` entries are
    ``(field_name, filename, data, mime)`` — repeated field names are allowed."""
    url = f"{ep.base_url}{path}"
    form = aiohttp.FormData()
    for key, value in fields.items():
        form.add_field(key, value)
    for field_name, filename, data, mime in files:
        form.add_field(field_name, data, filename=filename, content_type=mime)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=form, headers=ep.headers()) as resp:
                text = await _bounded_text(resp, path)
    except asyncio.TimeoutError:
        raise RuntimeError(f"image API {path} timed out after {timeout_s:.0f}s")
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"image API {path} request failed: {exc}")
    if resp.status != 200:
        raise RuntimeError(
            f"image API {path} on {ep.label} returned HTTP {resp.status}: {text[:400]}"
        )
    try:
        return json.loads(text)
    except ValueError:
        raise RuntimeError(
            f"image API {path} returned a non-JSON response: {text[:200]}"
        )


# ---------------------------------------------------------------------------
# URL fetch + response decoding
# ---------------------------------------------------------------------------


def _check_url_guards(url: str, cfg: Dict[str, Any], *, check_hosts: bool) -> None:
    """allowed_hosts (fnmatch on hostname) + allow_private_ips — same
    semantics as ``webhook_post``. Raises RuntimeError when refused."""
    hostname = (urllib.parse.urlparse(url).hostname or "").lower()
    if check_hosts:
        allowed = [
            str(p).strip().lower()
            for p in (cfg.get("allowed_hosts") or [])
            if str(p).strip()
        ]
        if allowed and not any(fnmatch.fnmatch(hostname, pat) for pat in allowed):
            raise RuntimeError(
                f"image URL host '{hostname}' is not in allowed_hosts — refused."
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
            raise RuntimeError(
                f"image URL '{hostname}' is a non-public IP — refused."
            )


async def fetch_url(
    url: str, cfg: Dict[str, Any], timeout_s: float, *, check_hosts: bool = True
) -> Tuple[bytes, str]:
    """Fetch ``url`` → (bytes, mime). ``check_hosts`` applies the
    ``allowed_hosts`` allowlist — only agent-supplied ``url:`` sources are
    guarded; URLs returned by the configured endpoint itself are trusted.
    Redirects are followed manually so the host/IP guards re-apply at every
    hop — otherwise an allowed host could bounce the request anywhere."""
    timeout = aiohttp.ClientTimeout(total=min(timeout_s, 120))
    data = b""
    mime = ""
    current = url
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for _ in range(6):  # initial request + up to 5 redirects
                _check_url_guards(current, cfg, check_hosts=check_hosts)
                async with session.get(current, allow_redirects=False) as resp:
                    if resp.status in (301, 302, 303, 307, 308):
                        loc = str(resp.headers.get("Location") or "").strip()
                        if not loc:
                            raise RuntimeError(
                                "image URL redirect without Location header"
                            )
                        current = urllib.parse.urljoin(current, loc)
                        if not current.lower().startswith(("http://", "https://")):
                            raise RuntimeError(
                                "image URL redirect to a non-http(s) target — refused"
                            )
                        continue
                    if resp.status != 200:
                        raise RuntimeError(
                            f"image URL fetch returned HTTP {resp.status}"
                        )
                    data = await resp.content.read(_MAX_SOURCE_BYTES + 1)
                    mime = (
                        resp.headers.get("Content-Type", "").split(";")[0].strip()
                    )
                    break
            else:
                raise RuntimeError("image URL fetch exceeded 5 redirects")
    except asyncio.TimeoutError:
        raise RuntimeError("image URL fetch timed out")
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"image URL fetch failed: {exc}")
    if len(data) > _MAX_SOURCE_BYTES:
        raise RuntimeError(
            f"image URL exceeds the {_MAX_SOURCE_BYTES // (1024 * 1024)} MB limit"
        )
    return data, mime


def sniff_image(data: bytes) -> Optional[Tuple[str, str]]:
    """(extension, mime) for common raster image magic bytes, else None."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return (".png", "image/png")
    if data[:3] == b"\xff\xd8\xff":
        return (".jpg", "image/jpeg")
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return (".webp", "image/webp")
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return (".gif", "image/gif")
    if data[:2] == b"BM":
        return (".bmp", "image/bmp")
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return (".tiff", "image/tiff")
    return None


async def decode_image_items(
    data: Dict[str, Any], cfg: Dict[str, Any], timeout_s: float
) -> Tuple[List[Tuple[bytes, str]], Optional[str]]:
    """Normalize an images API response → ``([(bytes, mime)], revised_prompt)``.

    Accepts ``b64_json`` and ``url`` entries (the two shapes OpenAI-compatible
    endpoints use). ``mime`` may be empty — the caller sniffs when needed.
    """
    items = data.get("data")
    if not isinstance(items, list) or not items:
        raise RuntimeError("image API returned no images (empty 'data' list)")
    out: List[Tuple[bytes, str]] = []
    for item in items[:_MAX_OUTPUT_IMAGES]:
        if not isinstance(item, dict):
            continue
        b64 = item.get("b64_json")
        if b64:
            try:
                out.append((base64.b64decode(b64), ""))
            except Exception:
                raise RuntimeError("image API returned invalid base64 image data")
            continue
        url = item.get("url")
        if url:
            raw, mime = await fetch_url(
                str(url), cfg, timeout_s, check_hosts=False
            )
            out.append((raw, mime))
    if not out:
        raise RuntimeError(
            "image API response contained neither b64_json nor url entries"
        )
    revised = next(
        (
            it["revised_prompt"]
            for it in items
            if isinstance(it, dict) and it.get("revised_prompt")
        ),
        None,
    )
    return out, revised


# ---------------------------------------------------------------------------
# Image source resolution (edit inputs)
# ---------------------------------------------------------------------------


def _decode_data_uri(uri: str) -> Tuple[bytes, str]:
    header, sep, payload = uri.partition(",")
    if not sep:
        raise RuntimeError("malformed data URI (missing ',')")
    m = re.match(r"^data:([^;,]*)(;base64)?$", header, re.IGNORECASE)
    if not m:
        raise RuntimeError("malformed data URI header")
    mime = m.group(1) or "application/octet-stream"
    try:
        if m.group(2):
            data = base64.b64decode(payload)
        else:
            data = urllib.parse.unquote_to_bytes(payload)
    except Exception:
        raise RuntimeError("malformed data URI payload")
    if len(data) > _MAX_SOURCE_BYTES:
        raise RuntimeError(
            f"inline image exceeds the {_MAX_SOURCE_BYTES // (1024 * 1024)} MB limit"
        )
    return data, mime


def _find_upload_meta(
    attachments: ChatAttachmentStore, conversation_id: str, query: str
) -> Optional[Dict[str, Any]]:
    """Match an upload by id or (case-insensitive) filename — newest last, so
    the most recent match wins."""
    q = (query or "").strip().lower()
    if not q:
        return None
    metas = attachments.list(conversation_id)
    for m in metas:
        if str(m.get("id", "")).lower() == q:
            return m
    for m in reversed(metas):
        if str(m.get("name", "")).lower() == q:
            return m
    for m in reversed(metas):
        if q in str(m.get("name", "")).lower():
            return m
    return None


async def _read_source_file(
    path: Path, label: str
) -> Tuple[bytes, str, str]:
    try:
        data = await asyncio.to_thread(path.read_bytes)
    except OSError as exc:
        raise RuntimeError(f"cannot read {label}: {exc}")
    if len(data) > _MAX_SOURCE_BYTES:
        raise RuntimeError(
            f"{label} exceeds the {_MAX_SOURCE_BYTES // (1024 * 1024)} MB limit"
        )
    sniffed = sniff_image(data)
    mime = sniffed[1] if sniffed else mime_for(path.name)
    return data, mime, label


async def _named_source(
    kind: str,
    value: str,
    ctx: Dict[str, Any],
    *,
    artifacts: ArtifactStore,
    attachments: ChatAttachmentStore,
    ws_store: AgentWorkspaceStore,
) -> Tuple[bytes, str, str]:
    conv = ctx.get("conversation_id")
    if kind == "artifact":
        if not conv:
            raise RuntimeError("artifact: sources require a conversation context")
        path = artifacts.resolve(conv, value)
        if path is None:
            raise RuntimeError(f"artifact '{value}' not found in this conversation")
        return await _read_source_file(path, f"artifact '{value}'")
    if kind == "attachment":
        if not conv:
            raise RuntimeError("attachment: sources require a conversation context")
        meta = _find_upload_meta(attachments, conv, value)
        if meta is None:
            raise RuntimeError(f"no uploaded file matching '{value}' in this chat")
        path = attachments.original_path(conv, str(meta.get("id") or ""))
        if path is None:
            raise RuntimeError(f"uploaded file '{value}' has no stored original")
        return await _read_source_file(path, f"upload '{meta.get('name') or value}'")
    # workspace
    scope_dir = ctx.get("agent_workspace_dir")
    if not scope_dir:
        raise RuntimeError(
            f"workspace source '{value}' requested but the agent workspace is "
            "not enabled for this assistant"
        )
    try:
        data = await asyncio.to_thread(
            ws_store.read, Path(scope_dir), value, _MAX_SOURCE_BYTES
        )
    except Exception as exc:
        raise RuntimeError(f"workspace file '{value}': {exc}")
    sniffed = sniff_image(data)
    mime = sniffed[1] if sniffed else mime_for(value)
    return data, mime, f"workspace '{value}'"


async def resolve_image_source(
    spec: Any,
    ctx: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    artifacts: ArtifactStore,
    attachments: ChatAttachmentStore,
    ws_store: AgentWorkspaceStore,
    timeout_s: float,
) -> Tuple[bytes, str, str]:
    """Resolve one image source spec → ``(bytes, mime, label)``.

    Schemes: ``attachment:``/``upload:`` (chat uploads), ``artifact:``
    (generated files), ``workspace:``/``file:`` (agent workspace),
    ``http(s)://``, ``data:`` — bare names try artifact → upload → workspace.
    Raises RuntimeError with an actionable message.
    """
    spec = str(spec or "").strip()
    if not spec:
        raise RuntimeError("empty image source")
    low = spec.lower()
    if low.startswith("data:"):
        data, mime = _decode_data_uri(spec)
        return data, mime, "inline data URI"
    if low.startswith("http://") or low.startswith("https://"):
        data, mime = await fetch_url(spec, cfg, timeout_s, check_hosts=True)
        if not mime:
            sniffed = sniff_image(data)
            mime = sniffed[1] if sniffed else "application/octet-stream"
        return data, mime, spec
    for prefix, kind in _SOURCE_SCHEMES:
        if low.startswith(prefix):
            return await _named_source(
                kind, spec[len(prefix):].strip(), ctx,
                artifacts=artifacts, attachments=attachments, ws_store=ws_store,
            )
    # Bare name: artifact → upload → workspace, first hit wins.
    errors: List[str] = []
    for kind in ("artifact", "attachment", "workspace"):
        try:
            return await _named_source(
                kind, spec, ctx,
                artifacts=artifacts, attachments=attachments, ws_store=ws_store,
            )
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError(
        f"image source '{spec}' not found (tried artifact, upload, workspace): "
        + "; ".join(errors)
    )


# ---------------------------------------------------------------------------
# Output delivery (artifacts + optional workspace copy)
# ---------------------------------------------------------------------------


async def deliver_images(
    ctx: Dict[str, Any],
    cfg: Dict[str, Any],
    images: List[Tuple[bytes, str]],
    *,
    artifacts: ArtifactStore,
    ws_store: AgentWorkspaceStore,
    filename: Optional[str],
    slug_base: str,
    workspace_path: Optional[str],
    max_images: int = _MAX_OUTPUT_IMAGES,
) -> Tuple[List[Dict[str, Any]], Optional[str], str, List[str]]:
    """Persist generated images as chat artifacts (+ optional workspace copy).

    Returns ``(metadata_files, ws_meta_path, ws_note, feedback_uris)``. The
    output extension is sniffed from the actual bytes — a 'jpg' filename on
    PNG bytes would produce a card that previews wrong.
    """
    conv = ctx.get("conversation_id")
    if not conv:
        raise RuntimeError("conversation_id required to store image artifacts")
    model = str(ctx.get("assistant_name") or "")
    entries: List[Dict[str, Any]] = []
    ws_paths: List[str] = []
    uris: List[str] = []

    base_stem = (
        Path(sanitize_filename(filename)).stem if filename else _slug(slug_base)
    )
    for i, (data, _mime) in enumerate(images[:max_images]):
        sniffed = sniff_image(data)
        ext = sniffed[0] if sniffed else ".png"
        mime = sniffed[1] if sniffed else "image/png"
        suffix = "" if i == 0 else f"-{i + 1}"
        desired = f"{base_stem}{suffix}{ext}"
        try:
            entry = await artifacts.save(
                conv, model, desired, data, ext.lstrip("."), overwrite=False
            )
        except Exception as exc:
            logger.warning("image artifact save failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Failed to store image artifact: {exc}")
        entries.append(entry)
        if cfg.get("return_images") and len(uris) < _MAX_OUTPUT_IMAGES:
            uris.append(
                f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
            )

    ws_note = ""
    ws_meta: Optional[str] = None
    ws_rel_raw = (workspace_path or "").strip().replace("\\", "/")
    if ws_rel_raw:
        scope_dir = ctx.get("agent_workspace_dir")
        if not scope_dir:
            ws_note = " Workspace copy skipped: agent workspace is not enabled."
        else:
            for i, (data, _mime) in enumerate(images[: len(entries)]):
                sniffed = sniff_image(data)
                ext = sniffed[0] if sniffed else ".png"
                suffix = "" if i == 0 else f"-{i + 1}"
                ws_stem = Path(ws_rel_raw).stem
                ws_parent = str(Path(ws_rel_raw).parent)
                rel = f"{ws_stem}{suffix}{ext}"
                ws_rel = (
                    f"{ws_parent}/{rel}" if ws_parent not in (".", "") else rel
                )
                try:
                    ws_entry = await asyncio.to_thread(
                        ws_store.write,
                        Path(scope_dir),
                        ws_rel,
                        data,
                        ctx.get("agent_workspace_cfg") or {},
                    )
                    ws_paths.append(ws_entry["path"])
                except Exception as exc:
                    logger.warning(
                        "image workspace copy failed (%s): %s", ws_rel, exc,
                        exc_info=True,
                    )
                    ws_note += f" Workspace copy to '{ws_rel}' failed: {exc}"
            if ws_paths:
                ws_meta = ", ".join(ws_paths)
                ws_note += (
                    f" Also saved to the workspace at '{ws_meta}'."
                    if len(ws_paths) == 1
                    else f" Also saved {len(ws_paths)} copies to the workspace: {ws_meta}."
                )
    return entries, ws_meta, ws_note, uris
