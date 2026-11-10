"""OpenAI-compatible chat completion router with streaming and metadata helpers."""

import asyncio
import contextlib
import datetime
import json
import logging
import mimetypes
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

import orjson
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from pathlib import Path

from app.chat_attachments import (
    AttachmentError,
    build_attachment_context,
    extract_attachment_text,
    image_attachment_data_uris,
)

from agent.recovery import RecoveryManager
from app.messages import MessageStore
from app.metrics import ChatMetricRecord, VectorstoreMetricRecord
from app.schemas import ChatRequest
from types import SimpleNamespace

from app.auto_title import ANON_USER_ID, schedule_auto_title
from app.redis_client import get_redis
from app.chat_runs import (
    _done_key,
    _events_channel,
    _state_key,
    _steer_channel,
)
from app.chat_utils import (
    aux_usage_totals,
    classify_message,
    convert_openai_messages,
    build_multimodal_content,
    count_tokens,
    MAX_PLAUSIBLE_TPS,
    estimate_full_prompt_tokens,
    extract_answer_text,
    extract_reasoning_text,
    extract_usage_metadata,
    collect_tool_names,
    build_message_metrics,
    finalize_token_fields,
    generate_tags,
    generate_title,
    limit_chat_history,
    parse_chat_history_block,
    render_system_instructions,
    split_history_and_latest_user,
    stringify_message_content,
)
from app.auth import read_session_cookie
from app.agent_chains_store import AgentChainsStore
from app.config_access import effective_config_bool, effective_config_value
from app.public_chat_config_store import PublicChatConfigStore
from app.chat_conversations_store import ChatConversationsStore
from app.chat_group_access import resolve_effective_allowlist
from app.routers.public_chat_auth import (
    _get_current_chat_user,
    _read_public_chat_access_token,
)
from app.chat_runs import ActiveChatRun, get_run, register_run, unregister_run
from app.orjson_response import ORJSONResponse
from app.history_budget import build_rolling_history, resolve_rolling_history_settings
from app.model_context import get_provider_context_windows
from agent.stream_watchdog import DEFAULT_RUN_IDLE_SECONDS
from app.emotions import (
    detect_text_emotion,
    validate_visual_emotion,
    fuse_emotions,
    build_emotion_prompt_hint,
)

router = APIRouter(
    prefix="/v1",
    tags=["Chat"],
)

logger = logging.getLogger(__name__)

_ALLOWED_ROLES = {"system", "user", "assistant"}


def _serialize_chat_history(messages: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        role = getattr(message, "type", None) or getattr(message, "role", None) or "unknown"
        serialized.append(
            {
                "role": str(role),
                "content": stringify_message_content(getattr(message, "content", "")),
            }
        )
    return serialized


def _get_message_store(request: Request) -> MessageStore | None:
    """Lazily build a MessageStore from the shared RAGApplication engine."""
    rag_app = request.app.state.rag_application
    engine = getattr(rag_app, "engine", None)
    if engine is None:
        return None
    store = getattr(request.app.state, "_message_store", None)
    if store is None:
        store = MessageStore(engine)
        request.app.state._message_store = store
    return store


async def _save_vectorstore_metric(
    request: Request,
    *,
    conversation_id: str,
    model_name: str,
    collection_names: str,
    started_at: str,
    retrieval_info: dict[str, object],
) -> None:
    """Persist a vectorstore metric row, swallowing errors."""
    store = getattr(request.app.state, "_vectorstore_metrics_store", None)
    if store is None:
        return
    status = str(retrieval_info.get("status", "skipped"))
    record = VectorstoreMetricRecord(
        conversation_id=conversation_id,
        model_name=model_name,
        collection_names=collection_names,
        retrieval_time_ms=float(retrieval_info.get("retrieval_time_ms", 0.0)),
        chunks_retrieved=int(retrieval_info.get("chunks_retrieved", 0)),
        status=status,
        started_at=started_at,
        error_type=str(retrieval_info.get("error_type")) if retrieval_info.get("error_type") else None,
        rag_context_tokens=int(retrieval_info.get("rag_context_tokens", 0) or 0) if status == "ok" else None,
    )
    try:
        await store.save(record)
    except Exception:
        logger.exception("Failed to save vectorstore metric for %s", conversation_id)


_MAX_PLAUSIBLE_TPS = MAX_PLAUSIBLE_TPS


def _compute_stored_tps(
    completion_tokens: Optional[float],
    llm_generation_ms: Optional[float],
    response_time_ms: Optional[float],
) -> Optional[float]:
    """Persistable tokens/s from stored fields, hardened against bad inputs.

    ``llm_generation_ms`` is only trusted when physically plausible
    (>= 1 ms and within 1.5x the total request time); corrupted values
    degrade to the whole-request denominator instead of exploding the
    rate. Implausible results return ``None`` — store unknown rather
    than garbage.
    """
    if not completion_tokens or completion_tokens <= 0:
        return None
    eff_ms: Optional[float] = None
    if (
        llm_generation_ms is not None
        and llm_generation_ms >= 1
        and llm_generation_ms <= (response_time_ms or float("inf")) * 1.5
    ):
        eff_ms = llm_generation_ms
    elif response_time_ms and response_time_ms >= 1:
        eff_ms = response_time_ms
    if not eff_ms or eff_ms <= 0:
        return None
    tps = completion_tokens / (eff_ms / 1000.0)
    if not (0 < tps <= _MAX_PLAUSIBLE_TPS):
        return None
    return tps


async def _save_chat_metric(
    request: Request,
    *,
    conversation_id: str,
    model_name: str,
    message_type: str,
    stream: bool,
    status_code: int,
    started_at: str,
    response_time_ms: float,
    time_to_first_token_ms: Optional[float] = None,
    time_to_first_reasoning_token_ms: Optional[float] = None,
    time_to_first_answer_token_ms: Optional[float] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    answer_tokens: Optional[int] = None,
    message_count: Optional[int] = None,
    input_chars: Optional[int] = None,
    output_chars: Optional[int] = None,
    rag_context_tokens: Optional[int] = None,
    reasoning_tokens: Optional[int] = None,
    error_type: Optional[str] = None,
    api_key_id: Optional[str] = None,
    tokens_per_second: Optional[float] = None,
    chat_user_id: Optional[str] = None,
    agent_iterations: Optional[int] = None,
    tool_names: Optional[list] = None,
    llm_generation_ms: Optional[float] = None,
    cache_tokens: Optional[int] = None,
    tool_result_tokens: Optional[int] = None,
) -> None:
    """Persist a chat metric row, swallowing errors to avoid breaking responses."""
    store = getattr(request.app.state, "_metrics_store", None)
    if store is None:
        return
    record = ChatMetricRecord(
        conversation_id=conversation_id,
        model_name=model_name,
        message_type=message_type,
        stream=stream,
        status_code=status_code,
        started_at=started_at,
        response_time_ms=response_time_ms,
        time_to_first_token_ms=time_to_first_token_ms,
        time_to_first_reasoning_token_ms=time_to_first_reasoning_token_ms,
        time_to_first_answer_token_ms=time_to_first_answer_token_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        answer_tokens=answer_tokens,
        tokens_per_second=(
            tokens_per_second
            if tokens_per_second is not None and 0 < tokens_per_second <= _MAX_PLAUSIBLE_TPS
            else _compute_stored_tps(completion_tokens, llm_generation_ms, response_time_ms)
        ),
        message_count=message_count,
        input_chars=input_chars,
        output_chars=output_chars,
        rag_context_tokens=rag_context_tokens,
        reasoning_tokens=reasoning_tokens,
        error_type=error_type,
        api_key_id=api_key_id,
        chat_user_id=chat_user_id,
        agent_iterations=agent_iterations,
        tool_names=tool_names,
        llm_generation_ms=llm_generation_ms,
        cache_tokens=cache_tokens,
        tool_result_tokens=tool_result_tokens,
    )
    try:
        await store.save(record)
    except Exception:
        logger.exception("Failed to save chat metric for %s", conversation_id)


def _has_disallowed_control_chars(text: str) -> bool:
    # Reject NUL always; other control chars (except common whitespace) are usually abusive.
    for ch in text:
        code = ord(ch)
        if code == 0:
            return True
        if code < 32 and ch not in ("\n", "\r", "\t"):
            return True
    return False


def _validate_openai_messages_payload(
    raw_messages: list,
    *,
    max_messages: int,
    max_message_chars: int,
    max_total_chars: int,
    require_last_user: bool,
    allow_non_string_content: bool,
    control_chars_mode: str,
) -> Optional[str]:
    if not raw_messages:
        return "No messages provided."

    if max_messages > 0 and len(raw_messages) > max_messages:
        return f"Too many messages (max {max_messages})."

    last = raw_messages[-1]
    last_role = getattr(last, "role", None) or (last.get("role") if isinstance(last, dict) else None) or "user"
    if require_last_user and str(last_role).lower() != "user":
        return "The final message must come from the user."

    total_chars = 0
    for idx, message in enumerate(raw_messages):
        role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None) or "user"
        role_norm = str(role).lower()
        if role_norm not in _ALLOWED_ROLES:
            return f"Invalid role at messages[{idx}]."

        content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
        if not allow_non_string_content and not isinstance(content, str):
            return f"Invalid content type at messages[{idx}] (must be string)."

        text = stringify_message_content(content)
        if role_norm == "user" and idx == len(raw_messages) - 1 and not text.strip():
            return "User message content cannot be empty."

        if control_chars_mode.lower() == "reject" and _has_disallowed_control_chars(text):
            return "Message contains disallowed control characters."

        if max_message_chars > 0 and len(text) > max_message_chars:
            return f"Message too long at messages[{idx}] (max {max_message_chars} chars)."

        total_chars += len(text)
        if max_total_chars > 0 and total_chars > max_total_chars:
            return f"Request too large (max {max_total_chars} chars across messages)."

    return None

@router.get("/knowledge/okf-graph")
async def okf_graph_endpoint(request: Request, collection: str):
    """Return all OKF concepts for a given collection.

    Accessible by both dashboard and public chat users (same auth as /v1/chat/completions).
    """
    rag_application = request.app.state.rag_application
    if not getattr(rag_application, "ready", False):
        return ORJSONResponse(
            status_code=503,
            content={"detail": "RAG pipeline is still initializing."}
        )
    try:
        concepts = await rag_application.get_okf_concepts(collection)
        return {"concepts": concepts}
    except Exception as exc:
        logger.warning("Failed to fetch OKF concepts for '%s': %s", collection, exc)
        raise HTTPException(status_code=500, detail=str(exc))


def _format_run_chunk(run: ActiveChatRun, delta: dict[str, Any], finish_reason: Optional[str] = None) -> str:
    chunk_payload = {
        "id": run.conversation_id,
        "object": "chat.completion.chunk",
        "created": run.created_ts,
        "model": run.model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {orjson.dumps(chunk_payload).decode('utf-8')}\n\n"


def _is_stream_keepalive(token: object) -> bool:
    """Synthetic periodic items (tool_progress keepalives, incl. forwarded
    subagent ones) keep the SSE wire and dashboard progress alive but do NOT
    count as run activity for the run-level idle watchdog."""
    return isinstance(token, dict) and token.get("keepalive") is True


def _run_chunk_from_token(run: ActiveChatRun, token: object) -> Optional[str]:
    if not isinstance(token, dict):
        return None
    content = token.get("content", "")
    reasoning = token.get("reasoning", "")
    tool_execution = token.get("tool_execution")
    tool_calls = token.get("tool_calls")
    tool_progress = token.get("tool_progress")
    subagent = token.get("subagent")
    todo_list = token.get("todo_list")
    cancelled = token.get("cancelled")
    injected_user_message = token.get("injected_user_message")
    unconsumed_inputs = token.get("unconsumed_inputs")
    observer = token.get("observer")
    live_usage = token.get("live_usage")
    if not content and not reasoning and not tool_execution and not tool_calls and not tool_progress and not subagent and todo_list is None and not cancelled and not injected_user_message and not unconsumed_inputs and not observer and not live_usage:
        return None
    delta: dict[str, Any] = {}
    if content:
        delta["content"] = content
    if reasoning:
        delta["reasoning_content"] = reasoning
    if tool_execution:
        delta["tool_execution"] = tool_execution
    if tool_calls:
        delta["tool_calls"] = tool_calls
    if tool_progress:
        delta["tool_progress"] = tool_progress
    if subagent:
        delta["subagent"] = subagent
    if todo_list is not None:
        delta["todo_list"] = todo_list
    if cancelled:
        delta["cancelled"] = True
    if injected_user_message:
        delta["injected_user_message"] = injected_user_message
        if token.get("injected_source"):
            delta["injected_source"] = token["injected_source"]
    if unconsumed_inputs:
        delta["unconsumed_inputs"] = unconsumed_inputs
    if observer:
        delta["observer"] = observer
    if live_usage:
        delta["live_usage"] = live_usage
    return _format_run_chunk(run, delta)


async def _stream_run_events(
    run: ActiveChatRun,
    *,
    resume: bool,
    queue: Optional[asyncio.Queue] = None,
    auto_title: Optional[tuple] = None,
) -> AsyncGenerator[str, None]:
    """SSE view onto a chat run: replay accumulated state, then live tokens.

    The generator is only a subscriber — disconnecting it has no effect on the
    underlying run, which keeps streaming and persists its own turn.
    """
    # subscribe() and snapshot() must run back-to-back without an await between
    # them so no broadcast can slip through the gap: anything produced before
    # subscribe lands in the snapshot, anything after lands in the queue.
    if queue is None:
        queue = run.subscribe()
    # Snapshot must be taken immediately after subscribe() — a yield before it
    # would let a broadcast slip into both the snapshot and the live queue and
    # resume clients would see that token twice.
    snapshot = run.snapshot() if resume else None
    try:
        yield _format_run_chunk(run, {"role": "assistant"})
        if snapshot is not None:
            yield _format_run_chunk(run, snapshot)
        while True:
            token = await queue.get()
            if token is None:
                break
            chunk = _run_chunk_from_token(run, token)
            if chunk:
                yield chunk
    finally:
        run.unsubscribe(queue)
        if auto_title is not None and run.done:
            schedule_auto_title(auto_title[0], run.conversation_id, auto_title[1])

    retrieval_info = run.retrieval_info
    if retrieval_info:
        rag_payload = {
            "status": str(retrieval_info.get("status", "skipped")),
            "chunks_retrieved": int(retrieval_info.get("chunks_retrieved", 0) or 0),
            "rag_context_tokens": int(retrieval_info.get("rag_context_tokens", 0) or 0),
            "rag_context": retrieval_info.get("rag_context", []) if retrieval_info.get("status") == "ok" else [],
        }
        if retrieval_info.get("todo_list") is not None:
            rag_payload["todo_list"] = retrieval_info["todo_list"]
        yield _format_run_chunk(run, {"retrieval_info": rag_payload})
    if run.emotion_data is not None:
        yield _format_run_chunk(run, {"emotion": run.emotion_data})

    # Per-message metrics for the frontend hover infobox — the same payload is
    # persisted as retrieval_info["_msg_metrics"] so reloads render the badge.
    try:
        usage_meta_emit = retrieval_info.get("_llm_usage_metadata") if isinstance(retrieval_info, dict) else None
        prompt_est = None
        if not (isinstance(usage_meta_emit, dict) and usage_meta_emit.get("prompt_tokens") is not None):
            prompt_est = estimate_full_prompt_tokens(
                run.sys_instructions_rendered,
                run.history_for_metrics,
                int(retrieval_info.get("rag_context_tokens", 0) or 0) if retrieval_info else 0,
                run.user_message,
            )
        metrics = build_message_metrics(
            retrieval_info=retrieval_info,
            assistant_text="".join(run.assistant_buffer).strip(),
            reasoning_text="".join(run.reasoning_buffer).strip() or (
                retrieval_info.get("_llm_reasoning_text") if isinstance(retrieval_info, dict) else None
            ),
            response_ms=(time.perf_counter() - run.started_perf) * 1000,
            prompt_estimate=prompt_est,
            stream_start=run.stream_start,
            first_token_at=run.first_token_at,
            first_answer_token_at=run.first_answer_token_at,
            first_reasoning_token_at=run.first_reasoning_token_at,
            subagent_count=len(run.subagent_channels) if run.subagent_channels else None,
        )
        retrieval_info["_msg_metrics"] = metrics
        yield _format_run_chunk(run, {"metrics": metrics})
    except Exception:
        logger.debug("Failed to emit streaming metrics chunk for %s", run.conversation_id, exc_info=True)

    yield _format_run_chunk(run, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _model_access_allowed(request: Request, model_name: str) -> bool:
    """Mirror the access rules of POST /chat/completions for a given model."""
    if getattr(request.state, "api_key_id", None):
        allowed = getattr(request.state, "allowed_models", None)
        if allowed and model_name not in allowed:
            return False
        return True
    app_config = getattr(request.app.state, "app_config", None)
    if app_config is not None and read_session_cookie(request, app_config):
        return True

    chat_user = await _get_current_chat_user(request)
    pct_data = _read_public_chat_access_token(request, app_config)

    rag_application = request.app.state.rag_application
    engine = getattr(rag_application, "engine", None) or getattr(rag_application, "_engine", None)

    if pct_data is not None or chat_user is not None:
        if engine is None:
            return False
        public_config = await PublicChatConfigStore(engine).get()
        if not (public_config and public_config.get("enabled")):
            return False
        allowlist = await resolve_effective_allowlist(
            request, pct_data=pct_data, chat_user=chat_user
        )
        if allowlist is not None and model_name not in allowlist:
            return False
        return True

    # Anonymous API caller — api_accessible is the only gate.
    if model_name.startswith("chain:"):
        if engine is None:
            return False
        chain_store = AgentChainsStore(engine)
        await chain_store.ensure_tables()
        entry = await chain_store.get_chain_by_name(model_name[6:])
        if entry is None:
            entry = await chain_store.get_chain(model_name[6:])
        return bool(entry and entry.enabled and entry.api_accessible)
    return bool(rag_application.instructions.get(model_name, {}).get("api_accessible", True))


async def _run_access_allowed(request: Request, run: ActiveChatRun) -> bool:
    """Mirror the access rules of POST /chat/completions for an existing run."""
    return await _model_access_allowed(request, run.model_name)


@router.get("/chat/runs/{conversation_id}")
async def chat_run_status(conversation_id: str, request: Request):
    """Report whether a streaming run for this conversation is still resumable."""
    run = get_run(conversation_id)
    if run is None:
        return {"running": False}
    if not await _run_access_allowed(request, run):
        raise HTTPException(status_code=403, detail="Not allowed to access this run.")
    # Unconsumed injected inputs are handed out exactly once — the first
    # status read after the run ended gets them back for requeueing.
    unconsumed = run.unconsumed_inputs
    run.unconsumed_inputs = []
    return {
        "running": True,
        "model": run.model_name,
        "done": run.done,
        "agent": run.is_agent_run,
        "unconsumed_inputs": unconsumed,
    }


@router.get("/chat/runs/{conversation_id}/stream")
async def chat_run_stream(conversation_id: str, request: Request):
    """Resume an in-flight (or recently finished) streaming run via SSE.

    Sends a `resumed` snapshot of everything produced so far followed by the
    remaining live tokens, trailers and [DONE].
    """
    run = get_run(conversation_id)
    if run is None:
        remote = await _remote_run_stream(request, conversation_id)
        if remote is not None:
            return remote
        return ORJSONResponse({"running": False})
    if not await _run_access_allowed(request, run):
        raise HTTPException(status_code=403, detail="Not allowed to access this run.")
    return StreamingResponse(
        _stream_run_events(run, resume=True),
        media_type="text/event-stream",
    )


async def _remote_run_stream(request: Request, conversation_id: str):
    """Stream a run produced on another replica via Redis state + pub/sub."""
    r = get_redis()
    if r is None:
        return None
    try:
        state = await r.hgetall(_state_key(conversation_id))
    except Exception:
        return None
    if not state or not state.get("data"):
        return None
    model_name = state.get("model_name", "")
    if model_name and not await _model_access_allowed(request, model_name):
        raise HTTPException(status_code=403, detail="Not allowed to access this run.")
    if not await _attachment_conv_owner_allowed(request, conversation_id):
        raise HTTPException(status_code=403, detail="Not allowed to access this run.")
    return StreamingResponse(
        _stream_remote_events(conversation_id, state),
        media_type="text/event-stream",
    )


_REMOTE_IDLE_SECONDS = 15.0


async def _stream_remote_events(
    conversation_id: str, state: Dict[str, str]
) -> AsyncGenerator[str, None]:
    """Replay the stored snapshot, then live tokens off the Redis channel."""
    try:
        snapshot = orjson.loads(state["data"])
    except Exception:
        snapshot = {}
    stub = SimpleNamespace(
        conversation_id=conversation_id, model_name=state.get("model_name", "")
    )
    yield _format_run_chunk(stub, {"role": "assistant"})
    yield _format_run_chunk(stub, snapshot)
    r = get_redis()
    if r is None or state.get("done") == "1":
        yield _format_run_chunk(stub, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return
    pubsub = r.pubsub()
    try:
        await pubsub.subscribe(_events_channel(conversation_id))
        while True:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=_REMOTE_IDLE_SECONDS
            )
            if message is None:
                # Idle — a dead producer must not pin the stream open; the done
                # marker is the authoritative end signal.
                if await r.exists(_done_key(conversation_id)):
                    break
                yield ": keepalive\n\n"
                continue
            try:
                token = orjson.loads(message["data"])
            except Exception:
                continue
            if isinstance(token, dict) and token.get("__done__"):
                break
            chunk = _run_chunk_from_token(stub, token)
            if chunk:
                yield chunk
    finally:
        try:
            await pubsub.aclose()
        except Exception:
            pass
    yield _format_run_chunk(stub, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _remote_steer(request: Request, conversation_id: str) -> Optional[str]:
    """Deliver steering input to a run produced on another replica.

    Returns ``"remote"`` when published, ``"finished"`` for a done run, or
    ``None`` when no remote run exists. Text-only: multimodal payloads need
    the producing node."""
    r = get_redis()
    if r is None:
        return None
    try:
        state = await r.hgetall(_state_key(conversation_id))
        if not state:
            return None
        model_name = state.get("model_name") or ""
        if state.get("done") == "1":
            return "finished"
    except Exception:
        return None
    if model_name and not await _model_access_allowed(request, model_name):
        raise HTTPException(status_code=403, detail="Not allowed to access this run.")
    if not await _attachment_conv_owner_allowed(request, conversation_id):
        raise HTTPException(status_code=403, detail="Not allowed to access this conversation.")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, str) or not message.strip():
        raise HTTPException(status_code=422, detail="Field 'message' must be a non-empty string.")
    subagent_id = payload.get("subagent_id") if isinstance(payload, dict) else None
    if isinstance(subagent_id, str) and not subagent_id.strip():
        subagent_id = None
    item: Dict[str, Any] = {"text": message.strip(), "content": message.strip()}
    if subagent_id:
        item["subagent_id"] = subagent_id
    try:
        await r.publish(_steer_channel(conversation_id), orjson.dumps(item))
    except Exception:
        return None
    logger.info("Steering published for remote run %s", conversation_id)
    return "remote"


@router.post("/chat/runs/{conversation_id}/cancel")
async def chat_run_cancel(conversation_id: str, request: Request):
    """Ask the detached producer of a run to stop early.

    Idempotent: cancelling a finished or missing run is a no-op so clients can
    fire-and-forget during racy stop/send-now flows.
    """
    run = get_run(conversation_id)
    if run is None:
        return {"cancelled": False, "reason": "no_run"}
    if not await _run_access_allowed(request, run):
        raise HTTPException(status_code=403, detail="Not allowed to access this run.")
    run.cancel()
    return {"cancelled": True, "done": run.done}


@router.post("/chat/runs/{conversation_id}/input")
async def chat_run_input(conversation_id: str, request: Request):
    """Inject a user message into a running agent loop at the next iteration.

    The message is appended to the agent's in-flight message list so the next
    LLM call sees it as fresh user input. Agent-only: non-agent runs and
    chains reject with ``accepted: false``.
    """
    run = get_run(conversation_id)
    if run is None:
        remote_reason = await _remote_steer(request, conversation_id)
        if remote_reason is not None:
            return {"accepted": remote_reason == "remote", "reason": remote_reason}
        return {"accepted": False, "reason": "no_run"}
    if not await _run_access_allowed(request, run):
        raise HTTPException(status_code=403, detail="Not allowed to access this run.")
    # Injection mutates the run — require conversation ownership the same way
    # attachment writes do (anonymous-owned convs keep bearer semantics).
    if not await _attachment_conv_owner_allowed(request, conversation_id):
        raise HTTPException(status_code=403, detail="Not allowed to access this conversation.")
    if run.done:
        return {"accepted": False, "reason": "finished"}
    if not run.is_agent_run:
        return {"accepted": False, "reason": "not_agent"}

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, str) or not message.strip():
        raise HTTPException(status_code=422, detail="Field 'message' must be a non-empty string.")
    message = message.strip()

    # Optional: target one of the run's subagents — the message lands in that
    # child's nudge queue instead of the root steer queue.
    subagent_id = payload.get("subagent_id") if isinstance(payload, dict) else None
    if subagent_id is not None and (
        not isinstance(subagent_id, str) or not subagent_id.strip()
    ):
        raise HTTPException(status_code=422, detail="Field 'subagent_id' must be a non-empty string.")
    if isinstance(subagent_id, str):
        subagent_id = subagent_id.strip() or None

    app_config = getattr(request.app.state, "app_config", None)
    if app_config is not None:
        max_chars = int(getattr(app_config, "max_message_chars", 20_000) or 20_000)
        if max_chars > 0 and len(message) > max_chars:
            raise HTTPException(status_code=422, detail=f"Message too long (max {max_chars} chars).")
        if _has_disallowed_control_chars(message):
            raise HTTPException(status_code=422, detail="Message contains disallowed control characters.")

    images = payload.get("images") if isinstance(payload, dict) else None

    # All conversation attachments inject their extracted text into mid-run
    # inputs too (queued messages reach the agent at the next iteration with
    # the same context a fresh completion would see). ``attachment_ids`` is
    # accepted for API compat — explicit ids are a subset of the conv scope.
    attachment_ids = payload.get("attachment_ids") if isinstance(payload, dict) else None
    if attachment_ids and (
        not isinstance(attachment_ids, list) or not all(isinstance(a, str) for a in attachment_ids)
    ):
        raise HTTPException(status_code=422, detail="Invalid attachment_ids payload.")
    display_text = message
    att_store = _attachment_store(request)
    if att_store is not None and await effective_config_bool(request, "chat_file_upload_enabled"):
        attachment_ctx = build_attachment_context(
            att_store,
            conversation_id,
            max_total_chars=int(
                await effective_config_value(request, "chat_file_max_total_chars", 120_000) or 120_000
            ),
        )
        if attachment_ctx:
            message = message + "\n" + attachment_ctx

    if images:
        if not isinstance(images, list) or not all(isinstance(i, str) and i for i in images):
            raise HTTPException(status_code=422, detail="Invalid images payload.")
        if subagent_id:
            raise HTTPException(status_code=422, detail="Subagent nudges are text-only.")
        if not await effective_config_bool(request, "chat_image_upload_enabled"):
            raise HTTPException(status_code=422, detail="Chat image upload is disabled.")
        max_count = int(await effective_config_value(request, "chat_image_max_count", 4) or 4)
        if len(images) > max_count:
            raise HTTPException(status_code=422, detail=f"Too many images (max {max_count}).")

    content: Any = build_multimodal_content(message, images) if images else message

    # `text` is what subscribers display as the injected user message — keep it
    # raw so attachment context blocks never render in the chat transcript.
    if subagent_id:
        reject_reason = run.submit_subagent_input(
            subagent_id, {"text": display_text}
        )
        if reject_reason is not None:
            return {"accepted": False, "reason": reject_reason}
        logger.info(
            "User nudge queued for subagent %s of run %s (%d chars)",
            subagent_id, conversation_id, len(message),
        )
        return {"accepted": True, "subagent_id": subagent_id}
    if not run.submit_input({"text": display_text, "content": content, "images": images or []}):
        return {"accepted": False, "reason": "finished"}
    logger.info("User input queued for injection into run %s (%d chars)", conversation_id, len(message))
    return {"accepted": True}


def _artifact_access(request: Request, conversation_id: str, filename: str) -> Dict[str, Any]:
    """Resolve an artifact entry and enforce the owning model's access rules."""
    store = getattr(request.app.state, "artifact_store", None)
    if store is None:
        raise HTTPException(status_code=404, detail="Artifacts unavailable.")
    entry = store.get_entry(conversation_id, filename)
    if entry is None:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return {"store": store, "entry": entry}


async def _check_artifact_access(request: Request, conversation_id: str, filename: str) -> Dict[str, Any]:
    resolved = _artifact_access(request, conversation_id, filename)
    if not await _model_access_allowed(request, resolved["entry"]["model"]):
        raise HTTPException(status_code=403, detail="Not allowed to access this artifact.")
    # Owned conversations additionally require the caller to be that owner;
    # row-less/NULL-user conversations keep bearer semantics (unguessable id).
    if not await _attachment_conv_owner_allowed(request, conversation_id):
        raise HTTPException(status_code=403, detail="Not allowed to access this artifact.")
    return resolved


@router.get("/chat/files/{conversation_id}/{filename}")
async def chat_file_download(conversation_id: str, filename: str, request: Request):
    """Download a generated artifact file."""
    resolved = await _check_artifact_access(request, conversation_id, filename)
    store, entry = resolved["store"], resolved["entry"]
    path = store.resolve(conversation_id, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Artifact file missing.")
    return FileResponse(path, media_type=entry["mime"], filename=entry["name"])


@router.get("/chat/files/{conversation_id}/{filename}/preview")
async def chat_file_preview(conversation_id: str, filename: str, request: Request):
    """Preview metadata for an artifact: rendered page count or text outline."""
    resolved = await _check_artifact_access(request, conversation_id, filename)
    try:
        return await resolved["store"].preview(conversation_id, filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preview failed: {exc}")


@router.get("/chat/files/{conversation_id}/{filename}/preview/page/{page}")
async def chat_file_preview_page(conversation_id: str, filename: str, page: int, request: Request):
    """One rendered preview page as PNG."""
    resolved = await _check_artifact_access(request, conversation_id, filename)
    path = resolved["store"].page_path(conversation_id, filename, page)
    if path is None:
        raise HTTPException(status_code=404, detail="Preview page not found.")
    return FileResponse(path, media_type="image/png")


@router.get("/chat/files/{conversation_id}/{filename}/export/{fmt}")
async def chat_file_export(conversation_id: str, filename: str, fmt: str, request: Request):
    """Download the artifact converted to another format (pdf/odt/odp)."""
    resolved = await _check_artifact_access(request, conversation_id, filename)
    from app.artifacts import ArtifactError
    try:
        path = await resolved["store"].export(conversation_id, filename, fmt)
    except ArtifactError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    from app.artifacts import mime_for
    return FileResponse(path, media_type=mime_for(path.name), filename=path.name)


# ------------------------------------------------------------------------------
# Conversation file attachments
# ------------------------------------------------------------------------------


async def _attachment_access_allowed(request: Request) -> bool:
    """Uploads require a real identity — the free public-chat access token is
    deliberately *not* accepted here so anonymous visitors cannot write files
    to disk (or spend LLM extraction quota) unauthenticated."""
    if getattr(request.state, "api_key_id", None):
        return True
    app_config = getattr(request.app.state, "app_config", None)
    if app_config is not None and read_session_cookie(request, app_config):
        return True
    return await _get_current_chat_user(request) is not None


def _attachment_store(request: Request):
    return getattr(request.app.state, "chat_attachment_store", None)


async def _attachment_conv_owner_allowed(request: Request, conversation_id: str) -> bool:
    """Best-effort ownership check for attachment operations.

    - API-key callers are privileged and skip the check.
    - Conversations without a store record (pre-first-message) or with
      ``user_id IS NULL`` (anonymously created) use bearer semantics: the
      unguessable conversation id is the only credential.
    - Owned conversations require the caller's chat-user id or linked
      ``dashboard_user_id`` to match.
    """
    if getattr(request.state, "api_key_id", None):
        return True
    conv_store = getattr(request.app.state, "_chat_conversations_store", None)
    if conv_store is None:
        return True
    conv = await conv_store.get_by_conversation_id(conversation_id)
    if conv is None or conv.get("user_id") in (None, ANON_USER_ID):
        return True
    chat_user = await _get_current_chat_user(request)
    if chat_user is None:
        return False
    allowed = {chat_user.get("id"), chat_user.get("dashboard_user_id")} - {None}
    return conv.get("user_id") in allowed


@router.get("/chat/capabilities")
async def chat_capabilities(request: Request):
    """Unprivileged capability discovery for chat clients.

    Returns the *effective* upload flags/limits so the UI can render
    affordances only when the backend would accept them — admin runtime-config
    toggles take effect immediately without an "Apply" reload. Per-caller like
    /public-chat/config: images are flag-gated (body-only, no disk writes),
    files additionally require a real identity.
    """
    authenticated = await _attachment_access_allowed(request)
    return {
        "images": await effective_config_bool(request, "chat_image_upload_enabled"),
        "files": bool(authenticated) and await effective_config_bool(request, "chat_file_upload_enabled"),
        "image_max_size_bytes": await effective_config_value(request, "chat_image_max_size_bytes", 5_242_880),
        "image_max_count": await effective_config_value(request, "chat_image_max_count", 4),
        "image_allowed_types": await effective_config_value(request, "chat_image_allowed_types", ""),
        "file_max_size_bytes": await effective_config_value(request, "chat_file_max_size_bytes", 20_971_520),
        "file_max_count": await effective_config_value(request, "chat_file_max_count", 5),
        "file_allowed_exts": await effective_config_value(request, "chat_file_allowed_exts", ""),
    }


@router.post("/chat/attachments")
async def chat_attachment_upload(
    request: Request,
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    model: Optional[str] = Form(default=None),
):
    """Upload a document attachment for a conversation and extract its text.

    The extracted text is injected into subsequent completions as tagged
    ``<attachment>`` context blocks — the raw file is never served back.
    """
    app_config = getattr(request.app.state, "app_config", None)
    if not await effective_config_bool(request, "chat_file_upload_enabled"):
        raise HTTPException(status_code=403, detail="Chat file upload is disabled.")
    if not await _attachment_access_allowed(request):
        raise HTTPException(status_code=403, detail="Not allowed to upload attachments.")
    if not await _attachment_conv_owner_allowed(request, conversation_id):
        raise HTTPException(status_code=403, detail="Not allowed to access this conversation.")
    store = _attachment_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Attachment store unavailable.")

    name = Path(file.filename or "file").name
    ext = Path(name).suffix.lower()
    allowed_exts = {
        e.strip().lower()
        for e in str(await effective_config_value(request, "chat_file_allowed_exts", "") or "").split(",")
        if e.strip()
    }
    if allowed_exts and ext not in allowed_exts:
        raise HTTPException(status_code=422, detail=f"File type '{ext or 'unknown'}' is not allowed.")

    data = await file.read()
    max_size = int(await effective_config_value(request, "chat_file_max_size_bytes", 20_971_520) or 20_971_520)
    if not data:
        raise HTTPException(status_code=422, detail="Empty file.")
    if len(data) > max_size:
        raise HTTPException(status_code=422, detail=f"File exceeds max size ({max_size} bytes).")
    max_count = int(await effective_config_value(request, "chat_file_max_count", 5) or 5)
    if len(store.list(conversation_id)) >= max_count:
        raise HTTPException(status_code=422, detail=f"Too many attachments (max {max_count}).")

    meta = await store.save(conversation_id, name, data)
    try:
        original = store.original_path(conversation_id, meta["id"])
        if original is None:
            raise AttachmentError("Stored file missing.")
        # Vision-based extraction paths (pdf images / pptx slides) use the
        # conversation's model when provided, else the metadata LLM.
        llm_factory = None
        rag_application = getattr(request.app.state, "rag_application", None)
        if rag_application is not None:
            try:
                llm = await rag_application.get_metadata_llm(model_name=model, purpose="attachment")
                llm_factory = lambda: llm
            except Exception:
                llm_factory = None
        timeout_s = float(
            await effective_config_value(request, "chat_file_extract_timeout_seconds", 60.0) or 60.0
        )
        text = await asyncio.wait_for(
            extract_attachment_text(original, app_config, llm_factory=llm_factory),
            timeout=timeout_s,
        )
        if not text.strip():
            text = "[no extractable text]"
        max_chars = int(await effective_config_value(request, "chat_file_max_extract_chars", 50_000) or 50_000)
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars] + "\n…[truncated]"
        meta = await store.set_extracted(conversation_id, meta["id"], text, truncated)
    except AttachmentError as exc:
        await store.delete(conversation_id, meta["id"])
        raise HTTPException(status_code=422, detail=str(exc))
    except asyncio.TimeoutError:
        await store.delete(conversation_id, meta["id"])
        raise HTTPException(status_code=422, detail="Extraction timed out.")
    return meta


@router.get("/chat/attachments")
async def chat_attachment_list(conversation_id: str, request: Request):
    """List attachments stored for a conversation."""
    if not await _attachment_access_allowed(request):
        raise HTTPException(status_code=403, detail="Not allowed to list attachments.")
    if not await _attachment_conv_owner_allowed(request, conversation_id):
        raise HTTPException(status_code=403, detail="Not allowed to access this conversation.")
    store = _attachment_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Attachment store unavailable.")
    return {"items": store.list(conversation_id)}


@router.get("/chat/attachments/{conversation_id}/{attachment_id}/file")
async def chat_attachment_file(conversation_id: str, attachment_id: str, request: Request):
    """Serve an attachment's original bytes — inline for images so history
    messages can render thumbnails; attachment download otherwise."""
    if not await _attachment_access_allowed(request):
        raise HTTPException(status_code=403, detail="Not allowed to access attachments.")
    if not await _attachment_conv_owner_allowed(request, conversation_id):
        raise HTTPException(status_code=403, detail="Not allowed to access this conversation.")
    store = _attachment_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Attachment store unavailable.")
    meta = store.get(conversation_id, attachment_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Attachment not found.")
    path = store.original_path(conversation_id, attachment_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Attachment file missing.")
    media_type = mimetypes.guess_type(meta.get("name") or "")[0] or "application/octet-stream"
    return FileResponse(
        path,
        media_type=media_type,
        filename=meta.get("name") or path.name,
        content_disposition_type="inline" if media_type.startswith("image/") else "attachment",
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=300"},
    )


@router.delete("/chat/attachments/{conversation_id}/{attachment_id}")
async def chat_attachment_delete(conversation_id: str, attachment_id: str, request: Request):
    """Remove an attachment so it is no longer injected into completions."""
    if not await _attachment_access_allowed(request):
        raise HTTPException(status_code=403, detail="Not allowed to delete attachments.")
    if not await _attachment_conv_owner_allowed(request, conversation_id):
        raise HTTPException(status_code=403, detail="Not allowed to access this conversation.")
    store = _attachment_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Attachment store unavailable.")
    deleted = await store.delete(conversation_id, attachment_id)
    return {"success": deleted}


@router.delete("/chat/conversations/{conversation_id}")
async def chat_conversation_delete(conversation_id: str, request: Request):
    """Delete a chat conversation and ALL its on-disk data.

    Mirrors the public-chat delete: row + artifacts/ + chat_uploads/ dirs.
    File blobs of conversations the client has forgotten (e.g. dashboard
    "new conversation") would otherwise persist forever.
    """
    if not await _attachment_access_allowed(request):
        raise HTTPException(status_code=403, detail="Not allowed to delete conversations.")
    if not await _attachment_conv_owner_allowed(request, conversation_id):
        raise HTTPException(status_code=403, detail="Not allowed to access this conversation.")

    # Flag before deleting so a still-finalizing run cannot recreate the row.
    from app.chat_runs import get_run

    run = get_run(conversation_id)
    if run is not None:
        run.conversation_deleted = True
        run.cancel()

    from app.conversation_cleanup import cleanup_conversation_data

    await cleanup_conversation_data(request.app.state, conversation_id)
    return {"success": True}


@router.post("/chat/completions")
async def chat_endpoint(chat_request: ChatRequest, request: Request):
    """Handle OpenAI-style chat completions with optional SSE streaming support."""
    rag_application = request.app.state.rag_application
    await rag_application.refresh_instructions_if_needed()

    # Resolve public chat user (if logged in) for conversation linking
    chat_user = await _get_current_chat_user(request)
    user_id = chat_user["id"] if chat_user else None
    app_config = getattr(request.app.state, "app_config", None)
    if not getattr(rag_application, "ready", False):
        return ORJSONResponse(
            status_code=503,
            content={"detail": "RAG pipeline is still initializing."}
        )

    model_name = chat_request.model or "CyberSecurityAssistant"

    # --- Access context (resolved once) ---
    # Precedence: API key -> dashboard session -> public-chat context (signed
    # access token or logged-in chat user) -> anonymous.
    has_api_key = getattr(request.state, "api_key_id", None) is not None
    has_dashboard_session = False
    if app_config is not None:
        has_dashboard_session = read_session_cookie(request, app_config) is not None
    pc_ctx: Optional[Dict[str, Any]] = None
    if not has_api_key and not has_dashboard_session:
        pct_data = _read_public_chat_access_token(request, app_config)
        if pct_data is not None or chat_user is not None:
            engine = getattr(rag_application, "engine", None) or getattr(rag_application, "_engine", None)
            if engine is not None:
                pc_config = await PublicChatConfigStore(engine).get()
                pc_ctx = {
                    "enabled": bool(pc_config and pc_config.get("enabled")),
                    # Callers in public-chat context are restricted to the
                    # allowlist resolved from their groups/role whenever one
                    # is configured. Chains are stored as 'chain:<name>'.
                    "allowlist": await resolve_effective_allowlist(
                        request, pct_data=pct_data, chat_user=chat_user
                    ),
                }
    public_chat_allowed = False

    # --- Chain resolution ---
    chain_definition: Optional[dict] = None
    chain_exec_options: Optional[dict] = None
    chain_run_meta: Optional[dict] = None
    chain_name: Optional[str] = None
    if model_name.startswith("chain:"):
        chain_name = model_name[6:]
        engine = getattr(rag_application, "engine", None)
        if engine is None:
            raise HTTPException(status_code=503, detail="Database unavailable for chain lookup.")
        chain_store = AgentChainsStore(engine)
        await chain_store.ensure_tables()
        chain_entry = await chain_store.get_chain_by_name(chain_name)
        if chain_entry is None:
            chain_entry = await chain_store.get_chain(chain_name)
        if chain_entry is None:
            raise HTTPException(status_code=404, detail=f"Chain '{chain_name}' not found.")
        if not chain_entry.enabled:
            raise HTTPException(status_code=403, detail=f"Chain '{chain_name}' is disabled.")
        if not chain_entry.api_accessible:
            pc_ok = bool(
                pc_ctx
                and pc_ctx["enabled"]
                and (pc_ctx["allowlist"] is None or model_name in pc_ctx["allowlist"])
            )
            if not has_dashboard_session and not pc_ok:
                raise HTTPException(status_code=403, detail="Chain is not accessible via the chat API.")
        chain_definition = chain_entry.flow_definition
        chain_exec_options = chain_entry.execution_options
        chain_run_meta = {
            "chain_id": chain_entry.id,
            "chain_name": chain_entry.name,
            "source": "chat",
        }
    else:
        if model_name not in getattr(rag_application, "instructions", {}):
            raise HTTPException(status_code=400, detail="Unknown model/assistant.")

    allowed_models = getattr(request.state, "allowed_models", None)
    if allowed_models and model_name not in allowed_models:
        raise HTTPException(status_code=403, detail="Model not allowed for this API key.")

    # --- Public chat context gate ---
    # Only requests authenticated as public-chat clients (signed access token
    # or public-chat session cookie) are subject to the public-chat
    # availability settings. Anonymous callers skip this entirely — for them
    # api_accessible below is the single gate.
    if pc_ctx is not None:
        if not pc_ctx["enabled"]:
            raise HTTPException(status_code=403, detail="Public chat is disabled.")
        if pc_ctx["allowlist"] is not None and model_name not in pc_ctx["allowlist"]:
            raise HTTPException(status_code=403, detail="Model is not publicly accessible.")
        public_chat_allowed = pc_ctx["allowlist"] is None or model_name in pc_ctx["allowlist"]

    if chain_definition is None:
        assistant_entry = rag_application.instructions.get(model_name, {})
        is_internal = not assistant_entry.get("api_accessible", True)
        if is_internal and not public_chat_allowed and not has_dashboard_session:
            raise HTTPException(status_code=403, detail="Assistant is not accessible via the chat API.")

    if request.headers.get("content-type") and "application/json" not in request.headers.get("content-type", ""):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json.")

    # Effective per-request read so admin runtime-config toggles apply
    # immediately (no "Apply" reload needed for capability enforcement).
    chat_images_enabled = await effective_config_bool(request, "chat_image_upload_enabled")
    if app_config is not None:
        # When chat image upload is enabled or images are present, allow non-string content
        request_images = getattr(chat_request, "images", None)
        effective_allow_non_string = bool(getattr(app_config, "allow_non_string_content", False))
        if chat_images_enabled:
            effective_allow_non_string = True
        error = _validate_openai_messages_payload(
            list(chat_request.messages or []),
            max_messages=int(getattr(app_config, "max_messages", 100) or 100),
            max_message_chars=int(getattr(app_config, "max_message_chars", 20_000) or 20_000),
            max_total_chars=int(getattr(app_config, "max_total_chars", 120_000) or 120_000),
            require_last_user=bool(getattr(app_config, "require_last_user", True)),
            allow_non_string_content=effective_allow_non_string,
            control_chars_mode=str(getattr(app_config, "validation_control_chars_mode", "reject") or "reject"),
        )
        if error:
            if bool(getattr(app_config, "validation_enforce", True)):
                raise HTTPException(status_code=422, detail=error)
            logger.warning("Validation would block request but VALIDATION_ENFORCE=false: %s", error)

    # --- Image upload validation ---
    request_images = getattr(chat_request, "images", None)
    if request_images:
        if not chat_images_enabled:
            raise HTTPException(status_code=422, detail="Chat image upload is disabled.")
        max_count = int(await effective_config_value(request, "chat_image_max_count", 4) or 4)
        max_size = int(await effective_config_value(request, "chat_image_max_size_bytes", 5_242_880) or 5_242_880)
        allowed_types_str = str(
            await effective_config_value(
                request, "chat_image_allowed_types", "image/png,image/jpeg,image/webp,image/gif"
            ) or ""
        )
        allowed_types = {t.strip() for t in allowed_types_str.split(",") if t.strip()}
        if len(request_images) > max_count:
            raise HTTPException(status_code=422, detail=f"Too many images (max {max_count}).")
        for img_idx, img in enumerate(request_images):
            if not isinstance(img, str) or not img:
                raise HTTPException(status_code=422, detail=f"Invalid image at index {img_idx}.")
            # Check size: raw base64 length is a proxy for byte size
            raw_b64 = img.split(",", 1)[-1] if img.startswith("data:") else img
            if len(raw_b64) > max_size * 2:  # base64 is ~33% larger, allow 2x for safety
                raise HTTPException(status_code=422, detail=f"Image at index {img_idx} exceeds max size ({max_size} bytes).")
            # Check MIME type from data URI
            if img.startswith("data:"):
                mime_part = img.split(";", 1)[0].replace("data:", "")
                if allowed_types and mime_part not in allowed_types:
                    raise HTTPException(status_code=422, detail=f"Image type '{mime_part}' at index {img_idx} is not allowed.")

    # Validate inline multimodal images in message content (OpenAI format)
    if chat_images_enabled and not request_images:
        max_count = int(await effective_config_value(request, "chat_image_max_count", 4) or 4)
        max_size = int(await effective_config_value(request, "chat_image_max_size_bytes", 5_242_880) or 5_242_880)
        for msg_idx, msg in enumerate(chat_request.messages or []):
            content = msg.content
            if not isinstance(content, list):
                continue
            inline_img_count = 0
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    inline_img_count += 1
                    url = (part.get("image_url") or {}).get("url", "")
                    raw_b64 = url.split(",", 1)[-1] if url.startswith("data:") else url
                    if len(raw_b64) > max_size * 2:
                        raise HTTPException(status_code=422, detail=f"Inline image at messages[{msg_idx}] exceeds max size ({max_size} bytes).")
            if inline_img_count > max_count:
                raise HTTPException(status_code=422, detail=f"Too many inline images at messages[{msg_idx}] (max {max_count}).")

    lc_messages = convert_openai_messages(chat_request.messages)
    try:
        history_messages, latest_user_message = split_history_and_latest_user(lc_messages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if app_config is not None:
        max_history = int(getattr(app_config, "max_history_messages", 0) or 0)
        history_messages = limit_chat_history(history_messages, max_messages=max_history)

    user_message = stringify_message_content(latest_user_message.content)
    if not user_message and not request_images:
        raise HTTPException(status_code=400, detail="User message content cannot be empty.")

    # Conversation attachments inject their extracted text into every turn —
    # llm_user_message is what the model sees; user_message (what is persisted
    # and displayed) stays clean.
    attachment_ctx = ""
    attachment_metas: list[dict[str, Any]] = []
    attachment_images: list[str] = []
    conv_id_for_attachments = (chat_request.conversation_id or "").strip()
    if conv_id_for_attachments and await effective_config_bool(request, "chat_file_upload_enabled"):
        att_store = _attachment_store(request)
        # Skip (don't block) when the caller does not own the conversation —
        # prevents leaking another user's attachment text into this prompt.
        if att_store is not None and await _attachment_conv_owner_allowed(
            request, conv_id_for_attachments
        ):
            attachment_metas = att_store.list(conv_id_for_attachments)
            if attachment_metas:
                attachment_ctx = build_attachment_context(
                    att_store,
                    conv_id_for_attachments,
                    max_total_chars=int(
                        await effective_config_value(request, "chat_file_max_total_chars", 120_000)
                        or 120_000
                    ),
                )
                # Stored image attachments go to the model as real image_url
                # parts (gated on the image-upload flag), alongside their OCR
                # text in attachment_ctx. Request images take precedence in
                # the shared chat_image_max_count budget.
                if chat_images_enabled:
                    _img_max_bytes = int(
                        await effective_config_value(
                            request, "chat_image_max_size_bytes", 5_242_880
                        ) or 5_242_880
                    )
                    _img_max_count = int(
                        await effective_config_value(request, "chat_image_max_count", 4) or 4
                    )
                    attachment_images = await image_attachment_data_uris(
                        att_store,
                        conv_id_for_attachments,
                        attachment_metas,
                        max_bytes=_img_max_bytes,
                        max_count=max(0, _img_max_count - len(request_images or [])),
                        exclude_uris=set(request_images or []),
                    )
        elif att_store is not None:
            logger.warning(
                "Skipping attachment injection: caller does not own conversation %s",
                conv_id_for_attachments,
            )
    llm_user_message = user_message + attachment_ctx

    # Build multimodal content if images are present — inline request images
    # and/or stored image attachments (attachment_images is empty unless the
    # image-upload flag is on). Attachment-only turns go through the same
    # channel so `input_text` — which doubles as the retrieval query — stays
    # the raw user question.
    all_images = list(request_images or []) + attachment_images
    multimodal_msg: Optional[HumanMessage] = None
    if all_images:
        multimodal_content = build_multimodal_content(llm_user_message, all_images)
        latest_user_message = HumanMessage(content=multimodal_content)
        multimodal_msg = latest_user_message
    elif attachment_ctx:
        latest_user_message = HumanMessage(
            content=build_multimodal_content(llm_user_message, [])
        )
        multimodal_msg = latest_user_message
    elif isinstance(latest_user_message.content, list):
        # Inline multimodal content in the last user message (OpenAI format)
        multimodal_msg = latest_user_message

    message_type = classify_message(user_message)

    # --- Emotion Detection ---
    emotion_data: dict[str, Any] | None = None
    emotion_hint: str | None = None
    if app_config is not None and getattr(app_config, "emotion_detection_enabled", False):
        text_emotion = None
        if getattr(app_config, "emotion_text_model_enabled", True):
            text_emotion = detect_text_emotion(user_message, config=app_config)

        visual_pre_send = None
        audio_prosody = None
        phases = None
        emotion_context = getattr(chat_request, "emotion_context", None) or {}
        if getattr(app_config, "emotion_webcam_enabled", False):
            visual_pre_send = validate_visual_emotion(
                emotion_context.get("visual_pre_send"), app_config
            )
        if getattr(app_config, "emotion_audio_enabled", False):
            audio_prosody = validate_visual_emotion(
                emotion_context.get("audio_prosody"), app_config
            )

        raw_phases = emotion_context.get("phases")
        if isinstance(raw_phases, list):
            phases = {}
            for item in raw_phases:
                if isinstance(item, dict) and item.get("primary_emotion") and item.get("phase"):
                    phase_name = item["phase"]
                    existing = phases.get(phase_name)
                    if existing is None or item.get("confidence", 0) > existing.get("confidence", 0):
                        phases[phase_name] = item

        fused = fuse_emotions(
            text_emotion=text_emotion,
            visual_pre_send=visual_pre_send,
            audio_prosody=audio_prosody,
            phases=phases,
            config=app_config,
        )
        emotion_data = {
            "text_emotion": text_emotion,
            "visual_pre_send": visual_pre_send,
            "audio_prosody": audio_prosody,
            "fused_emotion": fused,
            "phases": phases,
            "consent": emotion_context.get("consent") or {"webcam_opt_in": False},
        }
        if getattr(app_config, "emotion_prompt_steering_enabled", True):
            emotion_hint = build_emotion_prompt_hint(
                fused,
                text_emotion=text_emotion,
                visual_pre=visual_pre_send,
                audio_prosody=audio_prosody,
                phases=phases,
            )

    if message_type == "chat" and app_config is not None:
        assistant_entry = getattr(rag_application, "instructions", {}).get(model_name, {})
        provider_windows = await get_provider_context_windows(app_config)
        settings = resolve_rolling_history_settings(app_config, assistant_entry, provider_windows=provider_windows)
        history_messages = await build_rolling_history(
            history_messages,
            latest_user_message,
            settings=settings,
            summary_llm_factory=lambda: asyncio.create_task(rag_application.get_metadata_llm(purpose="summary")),
        )
    metadata_history = list(history_messages)
    if message_type in {"title", "tags"} and not metadata_history:
        inferred_history = parse_chat_history_block(user_message)
        if inferred_history:
            if app_config is not None:
                max_history = int(getattr(app_config, "max_history_messages", 0) or 0)
                metadata_history = limit_chat_history(inferred_history, max_messages=max_history)
            else:
                metadata_history = inferred_history

    # --- Streaming Response Logic ---
    # Run-level watchdog budget: seconds without a *real* stream item (token,
    # reasoning, tool/subagent event) before the run is aborted. Synthetic
    # keepalives don't reset it. LLM token-level stalls are bounded separately
    # via stream_idle_timeout_seconds inside the agent runner.
    run_idle_seconds = DEFAULT_RUN_IDLE_SECONDS
    if app_config is not None:
        run_idle_seconds = max(
            0.0,
            float(getattr(app_config, "run_idle_timeout_seconds", DEFAULT_RUN_IDLE_SECONDS)),
        )

    if chat_request.stream and message_type == "chat":

        response_id = (
            chat_request.conversation_id
            if getattr(chat_request, "conversation_id", None)
            and str(chat_request.conversation_id).strip()
            else f"chatcmpl-{uuid4().hex}"
        )
        created_ts = int(time.time())
        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        stream_start_outer = time.perf_counter()
        message_count = len(chat_request.messages or [])
        input_chars = sum(len(stringify_message_content(m.content)) for m in (chat_request.messages or []))

        retrieval_info: dict[str, object] = {}
        if emotion_data is not None:
            retrieval_info["emotion"] = emotion_data
        if emotion_hint is not None:
            retrieval_info["_emotion_hint"] = emotion_hint
        if attachment_metas:
            retrieval_info["attachments"] = [
                {
                    "id": m.get("id"),
                    "name": m.get("name"),
                    "ext": m.get("ext"),
                    "size": m.get("size"),
                    "vision": bool(m.get("vision")),
                }
                for m in attachment_metas
            ]

        # The run lives independently of this request's connection: tokens are
        # broadcast to every subscribed queue and the turn is persisted by the
        # producer task even when all clients have disconnected.
        run = ActiveChatRun(
            conversation_id=response_id,
            model_name=model_name,
            user_message=user_message,
            retrieval_info=retrieval_info,
            emotion_data=emotion_data,
        )
        run.is_agent_run = chain_definition is None and bool(
            rag_application.instructions.get(model_name, {}).get("agent_enabled")
        )
        run.source = (
            "api" if has_api_key
            else "dashboard" if has_dashboard_session
            else "public" if pc_ctx
            else "anonymous"
        )
        run.created_ts = created_ts
        run.started_perf = stream_start_outer
        run.sys_instructions_rendered = render_system_instructions(
            rag_application.instructions.get(model_name, {}).get("system_instructions", "")
        )
        run.history_for_metrics = history_messages
        register_run(run)

        api_key_id_snapshot = getattr(request.state, "api_key_id", None)
        message_store = _get_message_store(request)

        async def _fallback_text(reason: str) -> str:
            logger.warning("Streaming fallback triggered for %s (%s)", response_id, reason)
            try:
                if chain_definition is not None:
                    fallback_result = await rag_application.run_chain(
                        chain_definition=chain_definition,
                        input_text=llm_user_message,
                        chat_history=history_messages,
                        conversation_id=response_id,
                        chat_user_id=user_id,
                        execution_options=chain_exec_options,
                        run_context={
                            **(chain_run_meta or {}),
                            "conversation_id": response_id,
                        },
                    )
                else:
                    fallback_result = await rag_application.run(
                        user_message,
                        chat_history=history_messages,
                        model_name=model_name,
                        max_tokens=chat_request.max_tokens,
                        multimodal_input=multimodal_msg,
                        chat_user_id=user_id,
                    )
                return extract_answer_text(fallback_result) or "No answer found."
            except Exception as exc:
                logger.warning("Streaming fallback failed for %s: %s", response_id, exc)
                return "Upstream LLM request failed (check LLM_API_URL/LLM_API_KEY)."

        async def _finalize() -> None:
            """Persist the finished turn + metrics — runs detached from the request."""
            reasoning_text = "".join(run.reasoning_buffer).strip() or run.retrieval_info.get(
                "_llm_reasoning_text"
            )

            # Persisted per-message metrics — same payload as the SSE trailer so
            # a reload-from-server renders the identical metrics badge.
            try:
                _usage_meta_persist = retrieval_info.get("_llm_usage_metadata") if isinstance(retrieval_info, dict) else None
                _prompt_est_persist = None
                if not (isinstance(_usage_meta_persist, dict) and _usage_meta_persist.get("prompt_tokens") is not None):
                    _prompt_est_persist = estimate_full_prompt_tokens(
                        run.sys_instructions_rendered,
                        run.history_for_metrics,
                        int(retrieval_info.get("rag_context_tokens", 0) or 0) if retrieval_info else 0,
                        run.user_message,
                    )
                retrieval_info["_msg_metrics"] = build_message_metrics(
                    retrieval_info=retrieval_info,
                    assistant_text="".join(run.assistant_buffer).strip(),
                    reasoning_text=reasoning_text,
                    response_ms=(time.perf_counter() - run.started_perf) * 1000,
                    prompt_estimate=_prompt_est_persist,
                    stream_start=run.stream_start,
                    first_token_at=run.first_token_at,
                    first_answer_token_at=run.first_answer_token_at,
                    first_reasoning_token_at=run.first_reasoning_token_at,
                    subagent_count=len(run.subagent_channels) if run.subagent_channels else None,
                )
            except Exception:
                logger.debug("Failed to build persisted message metrics for %s", response_id, exc_info=True)

            # Tool calls from streamed tokens (chain mode) or fallback to agent trace
            tool_calls = run.tool_calls_buffer or None
            if not tool_calls:
                agent_trace = run.retrieval_info.get("_agent_trace")
                tool_calls = agent_trace.get("tool_calls") if isinstance(agent_trace, dict) else None

            try:
                if message_store is not None:
                    final_assistant_text = "".join(run.assistant_buffer).strip()
                    if final_assistant_text:
                        if run.injected_messages:
                            retrieval_info["injected_user_messages"] = list(run.injected_messages)
                        final_user_content = user_message + (
                            "\n\n" + "\n\n".join(
                                m.get("text", "") for m in run.injected_messages
                            )
                            if run.injected_messages else ""
                        )
                        logger.info("Persisting streamed message for %s (model=%s, user=%s chars, assistant=%s chars)",
                                    response_id, model_name, len(final_user_content), len(final_assistant_text))
                        await message_store.save_turn(
                            conversation_id=response_id,
                            user_content=final_user_content,
                            assistant_content=final_assistant_text,
                            assistant_reasoning=reasoning_text,
                            model_name=model_name,
                            chat_history=_serialize_chat_history(history_messages),
                            retrieval_info=retrieval_info,
                            api_key_id=api_key_id_snapshot,
                            tool_calls=tool_calls,
                            emotion=emotion_data,
                            user_id=user_id,
                        )
                        if user_id and not getattr(run, "conversation_deleted", False):
                            conv_store = ChatConversationsStore(message_store._engine)
                            await conv_store.ensure_exists(
                                user_id=user_id,
                                conversation_id=response_id,
                                model_name=model_name,
                            )
            except Exception:
                logger.exception("Failed to persist streamed messages for %s", response_id)

            # Persist streamed metrics
            try:
                final_assistant_text = "".join(run.assistant_buffer).strip()
                output_chars = len(final_assistant_text)
                rag_context_tokens_val = int(retrieval_info.get("rag_context_tokens", 0) or 0) if retrieval_info else 0
                usage_meta = retrieval_info.get("_llm_usage_metadata")
                # rag_search tool results count as RAG context in the persisted
                # aggregate column (per-message badge stays auto-retrieval only).
                rag_context_tokens_val += int((usage_meta or {}).get("rag_tool_tokens") or 0)
                prompt_tokens: Optional[int] = None
                completion_tokens: Optional[int] = None
                total_tokens: Optional[int] = None
                reasoning_tokens: Optional[int] = None
                cache_tokens_val: Optional[int] = None
                if usage_meta is not None:
                    prompt_tokens = usage_meta.get("prompt_tokens")
                    completion_tokens = usage_meta.get("completion_tokens")
                    total_tokens = usage_meta.get("total_tokens")
                    reasoning_tokens = usage_meta.get("reasoning_tokens")
                    cache_tokens_val = usage_meta.get("cache_tokens")
                aux = aux_usage_totals(retrieval_info)
                if aux["total_tokens"] or aux["prompt_tokens"]:
                    prompt_tokens = (prompt_tokens or 0) + aux["prompt_tokens"]
                    completion_tokens = (completion_tokens or 0) + aux["completion_tokens"]
                    total_tokens = (total_tokens or 0) + aux["total_tokens"]
                    reasoning_tokens = (reasoning_tokens or 0) + aux["reasoning_tokens"]
                    cache_tokens_val = (cache_tokens_val or 0) + aux["cache_tokens"]
                if prompt_tokens is None:
                    prompt_tokens = estimate_full_prompt_tokens(
                        run.sys_instructions_rendered,
                        history_messages,
                        rag_context_tokens_val,
                        llm_user_message,
                    )
                token_fields = finalize_token_fields(
                    prompt=prompt_tokens,
                    completion=completion_tokens,
                    total=total_tokens,
                    reasoning=reasoning_tokens,
                    cache=cache_tokens_val,
                    est_answer=count_tokens(final_assistant_text) if completion_tokens is None else None,
                    est_reasoning=count_tokens(reasoning_text) if (reasoning_text and not reasoning_tokens) else None,
                )
                prompt_tokens = token_fields["prompt_tokens"]
                completion_tokens = token_fields["completion_tokens"]
                total_tokens = token_fields["total_tokens"]
                reasoning_tokens = token_fields["reasoning_tokens"]
                answer_tokens = token_fields["answer_tokens"]
                response_time_ms = (time.perf_counter() - stream_start_outer) * 1000
                ttft_ms = ((run.first_token_at - run.stream_start) * 1000) if run.first_token_at else None
                ttfa_ms = ((run.first_answer_token_at - run.stream_start) * 1000) if run.first_answer_token_at else None
                ttfrt_ms = ((run.first_reasoning_token_at - run.stream_start) * 1000) if run.first_reasoning_token_at else None
                error_type = (
                    "cancelled" if run.cancelled
                    else type(run.stream_error).__name__ if run.stream_error else None
                )
                status_code = (
                    200 if (not error_type or error_type == "cancelled")
                    else (502 if "upstream" in (error_type or "").lower() else 500)
                )
                final_answer_usage = retrieval_info.get("_final_answer_usage") if isinstance(retrieval_info, dict) else None
                # Output generation rate over LLM-active time — same value the
                # per-message badge shows (runner aggregate), with the final
                # answer's call rate as fallback.
                final_tps: Optional[float] = None
                if isinstance(usage_meta, dict):
                    agg_tps = usage_meta.get("tokens_per_second")
                    if isinstance(agg_tps, (int, float)) and 0 < agg_tps <= _MAX_PLAUSIBLE_TPS:
                        final_tps = float(agg_tps)
                if final_tps is None and isinstance(final_answer_usage, dict):
                    fa_completion = final_answer_usage.get("completion_tokens")
                    fa_ms = final_answer_usage.get("_response_time_ms")
                    # >=1ms gate: a real LLM call never completes sub-millisecond —
                    # corrupted timing degrades to the stored-TPS fallback.
                    if fa_completion and fa_ms and fa_ms >= 1:
                        final_tps = fa_completion / (fa_ms / 1000.0)
                agent_iterations = usage_meta.get("iterations") if isinstance(usage_meta, dict) else None
                agent_trace = retrieval_info.get("_agent_trace") if isinstance(retrieval_info, dict) else None
                tool_calls_src = run.tool_calls_buffer or (
                    agent_trace.get("tool_calls") if isinstance(agent_trace, dict) else None
                )
                tool_names = collect_tool_names(tool_calls_src)
                await _save_chat_metric(
                    request,
                    conversation_id=response_id,
                    model_name=model_name,
                    message_type="chat",
                    stream=True,
                    status_code=status_code,
                    started_at=started_at,
                    response_time_ms=response_time_ms,
                    time_to_first_token_ms=ttft_ms,
                    time_to_first_reasoning_token_ms=ttfrt_ms,
                    time_to_first_answer_token_ms=ttfa_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    answer_tokens=answer_tokens,
                    message_count=message_count,
                    input_chars=input_chars,
                    output_chars=output_chars,
                    rag_context_tokens=rag_context_tokens_val if retrieval_info else None,
                    reasoning_tokens=reasoning_tokens,
                    error_type=error_type,
                    api_key_id=api_key_id_snapshot,
                    tokens_per_second=final_tps,
                    chat_user_id=user_id,
                    agent_iterations=agent_iterations,
                    tool_names=tool_names,
                    llm_generation_ms=(usage_meta.get("llm_generation_ms") if isinstance(usage_meta, dict) else None),
                    cache_tokens=cache_tokens_val,
                    tool_result_tokens=(usage_meta.get("tool_result_tokens") if isinstance(usage_meta, dict) else None),
                )
            except Exception:
                logger.exception("Failed to save streaming metrics for %s", response_id)

            # Persist streamed vectorstore metrics
            try:
                collections = ",".join(rag_application.instructions.get(model_name, {}).get("collections", []))
                await _save_vectorstore_metric(
                    request,
                    conversation_id=response_id,
                    model_name=model_name,
                    collection_names=collections,
                    started_at=started_at,
                    retrieval_info=retrieval_info,
                )
            except Exception:
                logger.exception("Failed to save streaming vectorstore metrics for %s", response_id)

        async def _producer() -> None:
            """Detached producer: streams the run, broadcasts tokens to all
            subscribers, handles first-token timeout + fallback, then persists."""
            error: Optional[Exception] = None
            timed_out = False
            agen = None
            try:
                if chain_definition is not None:
                    agen = rag_application.run_chain_streaming(
                        chain_definition=chain_definition,
                        input_text=llm_user_message,
                        chat_history=history_messages,
                        conversation_id=response_id,
                        chat_user_id=user_id,
                        execution_options=chain_exec_options,
                        run_context={
                            **(chain_run_meta or {}),
                            "conversation_id": response_id,
                        },
                    )
                else:
                    agen = rag_application.run_streaming(
                        user_message,
                        chat_history=history_messages,
                        model_name=model_name,
                        retrieval_info=retrieval_info,
                        max_tokens=chat_request.max_tokens,
                        conversation_id=response_id,
                        multimodal_input=multimodal_msg,
                        injected_input_queue=run.pending_inputs,
                        input_event=run.input_event,
                        subagent_channel_registry=run.subagent_channels,
                        chat_user_id=user_id,
                    )
                # Idle watchdog, not a total-run deadline: the deadline slides
                # forward on every produced item, so a run may take as long as
                # it needs while it keeps emitting — only a stalled stream dies.
                last_activity = run.stream_start
                cancel_wait = asyncio.create_task(run.cancel_event.wait())
                try:
                    while True:
                        anext_task = asyncio.ensure_future(agen.__anext__())
                        timeout: Optional[float] = None
                        if run_idle_seconds > 0:
                            timeout = max(
                                0.0, last_activity + run_idle_seconds - time.monotonic()
                            )
                        try:
                            done, _ = await asyncio.wait(
                                {anext_task, cancel_wait},
                                timeout=timeout,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                        except asyncio.CancelledError:
                            anext_task.cancel()
                            with contextlib.suppress(BaseException):
                                await anext_task
                            raise
                        if cancel_wait in done:
                            # Graceful stop: close the agent generator so its
                            # finally-blocks run (subagent shutdown etc.), then
                            # reap the pending __anext__ call.
                            with contextlib.suppress(Exception):
                                await agen.aclose()
                            anext_task.cancel()
                            with contextlib.suppress(BaseException):
                                await anext_task
                            break
                        if anext_task not in done:
                            anext_task.cancel()
                            with contextlib.suppress(BaseException):
                                await anext_task
                            timed_out = True
                            logger.warning(
                                "Run idle for over %.1fs for %s", run_idle_seconds, response_id
                            )
                            break
                        try:
                            token = anext_task.result()
                        except StopAsyncIteration:
                            break
                        # Only real items count as activity — synthetic
                        # keepalives must not shield a stuck tool forever.
                        if not _is_stream_keepalive(token):
                            last_activity = time.monotonic()
                        run.broadcast(token)
                finally:
                    cancel_wait.cancel()
                    with contextlib.suppress(BaseException):
                        await cancel_wait
            except asyncio.CancelledError as exc:
                error = exc
            except Exception as exc:  # pragma: no cover - network failures
                error = exc
            finally:
                if agen is not None:
                    try:
                        await agen.aclose()
                    except Exception:
                        pass

            if run.stream_error is None:
                run.stream_error = error
            if error is not None and not isinstance(error, asyncio.CancelledError):
                logger.warning("Streaming encountered error for %s: %s", response_id, error)

            try:
                if not run.produced_content and not run.cancelled:
                    reason = "streaming timeout" if timed_out else "empty streaming output"
                    fallback_text = await _fallback_text(reason)
                    if fallback_text:
                        run.broadcast({"content": fallback_text})
            finally:
                leftover = [item for item in run.drain_pending_inputs() if item.get("text")]
                if leftover:
                    run.unconsumed_inputs = [
                        {"text": item["text"], "images": item.get("images") or []}
                        for item in leftover
                    ]
                    run.broadcast({"unconsumed_inputs": list(run.unconsumed_inputs)})
                if run.cancelled:
                    run.broadcast({"cancelled": True})
                # Aborted runs never deliver the subagent manager's final
                # "cancelled" events (multiplexer already exited) — broadcast
                # the last record state so badges don't stay stuck "running".
                # Best-effort only: a failure here must not skip run.finish.
                if run.subagent_channels and (run.cancelled or timed_out or error):
                    for rec in list(run.subagent_channels.values()):
                        try:
                            stage = {
                                "done": "finished",
                                "error": "finished",
                                "cancelled": "cancelled",
                            }.get(rec.status, "progress")
                            payload = {**rec.to_dict(), "stage": stage}
                            if stage == "finished":
                                payload["ok"] = rec.status == "done"
                            run.broadcast(
                                {"content": "", "reasoning": "", "subagent": payload}
                            )
                        except Exception:
                            logger.debug("Final subagent status broadcast failed", exc_info=True)
                run.finish(error)
            try:
                await _finalize()
            except Exception:
                logger.exception("Finalize failed for %s", response_id)
            finally:
                unregister_run(response_id, run)

        # Subscribe the response's queue before starting the producer so tokens
        # broadcast before the SSE generator first runs are still delivered.
        initial_queue = run.subscribe()
        asyncio.create_task(_producer())
        logger.info("Streaming SSE started for %s", response_id)
        title_args = (
            (request.app.state, metadata_history)
            if await effective_config_bool(request, "chat_auto_title_enabled")
            else None
        )
        return StreamingResponse(
            _stream_run_events(run, resume=False, queue=initial_queue, auto_title=title_args),
            media_type="text/event-stream",
        )

    raw_answer: str | list[dict[str, object]]
    conversation_id = (
        chat_request.conversation_id
        if getattr(chat_request, "conversation_id", None)
        and str(chat_request.conversation_id).strip()
        else f"chatcmpl-{uuid4().hex}"
    )
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    message_count = len(chat_request.messages or [])
    input_chars = sum(len(stringify_message_content(m.content)) for m in (chat_request.messages or []))
    error_type: Optional[str] = None
    status_code = 200
    response_time_ms = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    answer_tokens: Optional[int] = None
    final_tps: Optional[float] = None
    output_chars: Optional[int] = None
    agent_iterations: Optional[int] = None
    tool_names: Optional[list] = None

    start_time = time.perf_counter()
    if message_type == "chat":
        retrieval_info: dict[str, object] = {}
        if emotion_data is not None:
            retrieval_info["emotion"] = emotion_data
        if emotion_hint is not None:
            retrieval_info["_emotion_hint"] = emotion_hint
        if attachment_metas:
            retrieval_info["attachments"] = [
                {
                    "id": m.get("id"),
                    "name": m.get("name"),
                    "ext": m.get("ext"),
                    "size": m.get("size"),
                    "vision": bool(m.get("vision")),
                }
                for m in attachment_metas
            ]
        try:
            if chain_definition is not None:
                result = await rag_application.run_chain(
                    chain_definition=chain_definition,
                    input_text=llm_user_message,
                    chat_history=history_messages,
                    conversation_id=conversation_id,
                    chat_user_id=user_id,
                    execution_options=chain_exec_options,
                    run_context={
                        **(chain_run_meta or {}),
                        "conversation_id": conversation_id,
                    },
                )
            else:
                result = await rag_application.run(
                    user_message, chat_history=history_messages, model_name=model_name,
                    retrieval_info=retrieval_info,
                    max_tokens=chat_request.max_tokens,
                    conversation_id=conversation_id,
                    multimodal_input=multimodal_msg,
                    chat_user_id=user_id,
                )
        except Exception as exc:
            error_type = type(exc).__name__
            status_code = 502 if RecoveryManager.is_upstream_llm_error(exc) else 500
            response_time_ms = (time.perf_counter() - start_time) * 1000
            await _save_chat_metric(
                request,
                conversation_id=conversation_id,
                model_name=model_name,
                message_type="chat",
                stream=False,
                status_code=status_code,
                started_at=started_at,
                response_time_ms=response_time_ms,
                message_count=message_count,
                input_chars=input_chars,
                rag_context_tokens=int(retrieval_info.get("rag_context_tokens", 0) or 0) if retrieval_info else None,
                reasoning_tokens=None,
                error_type=error_type,
                api_key_id=getattr(request.state, "api_key_id", None),
                chat_user_id=user_id,
            )
            try:
                collections = ",".join(rag_application.instructions.get(model_name, {}).get("collections", []))
                await _save_vectorstore_metric(
                    request,
                    conversation_id=conversation_id,
                    model_name=model_name,
                    collection_names=collections,
                    started_at=started_at,
                    retrieval_info=retrieval_info,
                )
            except Exception:
                logger.exception("Failed to save vectorstore metrics for %s", conversation_id)
            if RecoveryManager.is_upstream_llm_error(exc):
                logger.warning("Upstream chat model failed for %s: %s", model_name, exc)
                raise HTTPException(
                    status_code=502,
                    detail="Upstream LLM request failed. Please retry later.",
                ) from exc
            raise
        raw_answer = extract_answer_text(result) or "No answer found."
        response_time_ms = (time.perf_counter() - start_time) * 1000
        assistant_text = raw_answer if isinstance(raw_answer, str) else extract_answer_text(raw_answer)
        output_chars = len(assistant_text or "")
        system_instructions = render_system_instructions(rag_application.instructions.get(model_name, {}).get("system_instructions", ""))
        rag_context_tokens_val = int(retrieval_info.get("rag_context_tokens", 0) or 0) if retrieval_info else 0
        usage_meta = retrieval_info.pop("_llm_usage_metadata", None)
        # rag_search tool results count as RAG context in the persisted
        # aggregate column (per-message badge stays auto-retrieval only).
        rag_context_tokens_val += int((usage_meta or {}).get("rag_tool_tokens") or 0)
        reasoning_tokens: Optional[int] = None
        cache_tokens_val: Optional[int] = None
        if usage_meta is not None:
            prompt_tokens = usage_meta.get("prompt_tokens")
            completion_tokens = usage_meta.get("completion_tokens")
            total_tokens = usage_meta.get("total_tokens")
            reasoning_tokens = usage_meta.get("reasoning_tokens")
            cache_tokens_val = usage_meta.get("cache_tokens")
            agent_iterations = usage_meta.get("iterations")
            # tokens/s is the run's output generation rate (completion over
            # LLM-active time) — the final answer's call rate is the fallback.
            agg_tps = usage_meta.get("tokens_per_second")
            if isinstance(agg_tps, (int, float)) and 0 < agg_tps <= _MAX_PLAUSIBLE_TPS:
                final_tps = float(agg_tps)
            fa_usage = usage_meta.get("final_answer_usage")
            if final_tps is None and isinstance(fa_usage, dict):
                fa_completion = fa_usage.get("completion_tokens")
                fa_ms = fa_usage.get("_response_time_ms")
                # >=1ms gate: a real LLM call never completes sub-millisecond.
                if fa_completion and fa_ms and fa_ms >= 1:
                    final_tps = fa_completion / (fa_ms / 1000.0)
        else:
            prompt_tokens = estimate_full_prompt_tokens(
                system_instructions,
                history_messages,
                rag_context_tokens_val,
                llm_user_message,
            )
        aux = aux_usage_totals(retrieval_info)
        if aux["total_tokens"] or aux["prompt_tokens"]:
            prompt_tokens = (prompt_tokens or 0) + aux["prompt_tokens"]
            completion_tokens = (completion_tokens or 0) + aux["completion_tokens"]
            total_tokens = (total_tokens or 0) + aux["total_tokens"]
            reasoning_tokens = (reasoning_tokens or 0) + aux["reasoning_tokens"]
            cache_tokens_val = (cache_tokens_val or 0) + aux["cache_tokens"]
        # Read reasoning text from result (always available)
        reasoning_text = extract_reasoning_text(result)
        token_fields = finalize_token_fields(
            prompt=prompt_tokens,
            completion=completion_tokens,
            total=total_tokens,
            reasoning=reasoning_tokens,
            cache=cache_tokens_val,
            est_answer=count_tokens(assistant_text or "") if completion_tokens is None else None,
            est_reasoning=count_tokens(reasoning_text) if (reasoning_text and not reasoning_tokens) else None,
        )
        prompt_tokens = token_fields["prompt_tokens"]
        completion_tokens = token_fields["completion_tokens"]
        total_tokens = token_fields["total_tokens"]
        reasoning_tokens = token_fields["reasoning_tokens"]
        answer_tokens = token_fields["answer_tokens"]
        # Persisted per-message metrics — same payload shape as the streaming
        # trailer so a reload-from-server renders the badge identically.
        try:
            retrieval_info["_msg_metrics"] = build_message_metrics(
                retrieval_info=retrieval_info,
                assistant_text=assistant_text or "",
                reasoning_text=reasoning_text,
                response_ms=response_time_ms,
                prompt_estimate=prompt_tokens,
                usage_meta=usage_meta,
            )
        except Exception:
            logger.debug("Failed to build persisted message metrics for %s", conversation_id, exc_info=True)
        # Extract tool calls from chain result or agent trace for persistence
        tool_calls = None
        if isinstance(result, dict) and result.get("tool_calls"):
            tool_calls = result["tool_calls"]
        if not tool_calls:
            agent_trace = retrieval_info.pop("_agent_trace", None)
            tool_calls = agent_trace.get("tool_calls") if isinstance(agent_trace, dict) else None
        tool_names = collect_tool_names(tool_calls)

        # Persist non-streaming chat messages
        try:
            store = _get_message_store(request)
            if store is not None:
                logger.info("Persisting non-streamed message for %s (model=%s)",
                            conversation_id, model_name)
                await store.save_turn(
                    conversation_id=conversation_id,
                    user_content=user_message,
                    assistant_content=assistant_text or "",
                    assistant_reasoning=reasoning_text,
                    model_name=model_name,
                    chat_history=_serialize_chat_history(history_messages),
                    retrieval_info=retrieval_info,
                    api_key_id=getattr(request.state, "api_key_id", None),
                    tool_calls=tool_calls,
                    emotion=emotion_data,
                    user_id=user_id,
                )
                if user_id:
                    conv_store = ChatConversationsStore(store._engine)
                    await conv_store.ensure_exists(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        model_name=model_name,
                    )
        except Exception:
            logger.exception("Failed to persist chat messages for model %s", model_name)
        # Persist vectorstore metrics
        try:
            collections = ",".join(rag_application.instructions.get(model_name, {}).get("collections", []))
            await _save_vectorstore_metric(
                request,
                conversation_id=conversation_id,
                model_name=model_name,
                collection_names=collections,
                started_at=started_at,
                retrieval_info=retrieval_info,
            )
        except Exception:
            logger.exception("Failed to save vectorstore metrics for %s", conversation_id)
    elif message_type == "title":
        metadata_llm = await rag_application.get_metadata_llm(purpose="title")
        try:
            title = await generate_title(metadata_llm, metadata_history)
        except Exception as exc:
            error_type = type(exc).__name__
            status_code = 502 if RecoveryManager.is_upstream_llm_error(exc) else 500
            response_time_ms = (time.perf_counter() - start_time) * 1000
            await _save_chat_metric(
                request,
                conversation_id=conversation_id,
                model_name=model_name,
                message_type="title",
                stream=False,
                status_code=status_code,
                started_at=started_at,
                response_time_ms=response_time_ms,
                message_count=message_count,
                input_chars=input_chars,
                error_type=error_type,
                api_key_id=getattr(request.state, "api_key_id", None),
            )
            if RecoveryManager.is_upstream_llm_error(exc):
                logger.warning("Upstream title model failed for %s: %s", model_name, exc)
                raise HTTPException(
                    status_code=502,
                    detail="Upstream LLM request failed. Please retry later.",
                ) from exc
            raise
        raw_answer = json.dumps({"title": title}, ensure_ascii=False)
        response_time_ms = (time.perf_counter() - start_time) * 1000
        output_chars = len(raw_answer)
        completion_tokens = count_tokens(raw_answer)
        prompt_tokens = count_tokens(llm_user_message)
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        reasoning_tokens = None
        answer_tokens = completion_tokens
    elif message_type == "tags":
        metadata_llm = await rag_application.get_metadata_llm(purpose="tags")
        try:
            tags = await generate_tags(metadata_llm, metadata_history)
        except Exception as exc:
            error_type = type(exc).__name__
            status_code = 502 if RecoveryManager.is_upstream_llm_error(exc) else 500
            response_time_ms = (time.perf_counter() - start_time) * 1000
            await _save_chat_metric(
                request,
                conversation_id=conversation_id,
                model_name=model_name,
                message_type="tags",
                stream=False,
                status_code=status_code,
                started_at=started_at,
                response_time_ms=response_time_ms,
                message_count=message_count,
                input_chars=input_chars,
                error_type=error_type,
                api_key_id=getattr(request.state, "api_key_id", None),
            )
            if RecoveryManager.is_upstream_llm_error(exc):
                logger.warning("Upstream tags model failed for %s: %s", model_name, exc)
                raise HTTPException(
                    status_code=502,
                    detail="Upstream LLM request failed. Please retry later.",
                ) from exc
            raise
        raw_answer = json.dumps({"tags": tags}, ensure_ascii=False)
        response_time_ms = (time.perf_counter() - start_time) * 1000
        output_chars = len(raw_answer)
        completion_tokens = count_tokens(raw_answer)
        prompt_tokens = count_tokens(llm_user_message)
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        reasoning_tokens = None
        answer_tokens = completion_tokens
    else:
        raw_answer = "Error: Unknown message type detected."
        response_time_ms = (time.perf_counter() - start_time) * 1000
        status_code = 400
        error_type = "UnknownMessageType"
        output_chars = len(raw_answer)
        reasoning_tokens = None

    # Save metrics for successful non-streaming paths
    if not error_type:
        await _save_chat_metric(
            request,
            conversation_id=conversation_id,
            model_name=model_name,
            message_type=message_type,
            stream=False,
            status_code=status_code,
            started_at=started_at,
            response_time_ms=response_time_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            message_count=message_count,
            input_chars=input_chars,
            output_chars=output_chars,
            rag_context_tokens=rag_context_tokens_val if retrieval_info else None,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
            api_key_id=getattr(request.state, "api_key_id", None),
            tokens_per_second=final_tps,
            chat_user_id=user_id,
            agent_iterations=agent_iterations,
            tool_names=tool_names,
            llm_generation_ms=usage_meta.get("llm_generation_ms") if usage_meta else None,
            cache_tokens=cache_tokens_val,
            tool_result_tokens=usage_meta.get("tool_result_tokens") if usage_meta else None,
        )

    response_payload = {
        "id": conversation_id,
        "conversation_id": conversation_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": raw_answer if isinstance(raw_answer, list) else raw_answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "answer_tokens": answer_tokens or 0,
            "total_tokens": total_tokens or 0,
            "reasoning_tokens": reasoning_tokens or 0,
            "response_time_ms": round(response_time_ms, 1),
            "tokens_per_second": round(
                final_tps if final_tps is not None
                else (completion_tokens / (response_time_ms / 1000.0) if completion_tokens and response_time_ms > 0 else 0.0),
                2,
            ) if (final_tps is not None or (completion_tokens and response_time_ms > 0)) else None,
        },
    }

    return ORJSONResponse(content=response_payload)

@router.get("/models")
async def list_models(request: Request):
    """Expose publicly accessible assistants and agent chains for OpenAI-compatible clients.

    Only ``api_accessible`` models are listed — public-chat availability
    settings intentionally do not affect the raw API surface.
    """
    rag_application = request.app.state.rag_application
    await rag_application.refresh_instructions_if_needed()

    models = []
    for name, data in rag_application.instructions.items():
        if not data.get("api_accessible", True):
            continue
        models.append(
            {
                "id": name,
                "object": "model",
                "owned_by": "Agentix-Project",
                "created": int(time.time()),
                "description": data.get("description", ""),
                "permission": [],
                "collections": data.get("collections", []),
                "type": "assistant",
            }
        )

    engine = getattr(rag_application, "engine", None) or getattr(rag_application, "_engine", None)
    if engine is not None:
        try:
            chain_store = AgentChainsStore(engine)
            for chain in await chain_store.list_chains():
                if not chain.enabled or not chain.api_accessible:
                    continue
                models.append(
                    {
                        "id": f"chain:{chain.name}",
                        "object": "model",
                        "owned_by": "Agentix-Project",
                        "created": int(chain.created_at.timestamp()) if chain.created_at else int(time.time()),
                        "description": chain.description or "",
                        "permission": [],
                        "collections": [],
                        "type": "chain",
                    }
                )
        except Exception:
            logger.warning("Failed to list agent chains for /v1/models", exc_info=True)

    return {"object": "list", "data": models}