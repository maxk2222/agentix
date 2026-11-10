"""Lightweight async metrics persistence to Postgres.

Tracks per-request performance data for chat completions, independent of
the message store, so historical aggregations remain cheap.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from agent.utils.pg_helpers import _quote_identifier

logger = logging.getLogger(__name__)


def _format_bucket(bucket_val) -> str:
    """Normalize a PostgreSQL date_trunc result to an ISO-8601 UTC string."""
    if hasattr(bucket_val, "strftime"):
        return bucket_val.strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(bucket_val)


@dataclass
class BlockedRequestRecord:
    """Structured blocked-request row before insertion."""

    path: str
    method: str
    status_code: int
    reason: str


class BlockedRequestStore:
    """Async store for requests blocked by middleware."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._tables_ensured = False

    async def ensure_tables(self) -> None:
        if self._tables_ensured:
            return
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS blocked_requests (
                        id TEXT PRIMARY KEY,
                        path TEXT NOT NULL,
                        method TEXT NOT NULL,
                        status_code INTEGER NOT NULL,
                        reason TEXT NOT NULL,
                        created_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )
        self._tables_ensured = True

    async def save(self, record: BlockedRequestRecord) -> None:
        if not self._tables_ensured:
            await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO blocked_requests (id, path, method, status_code, reason)
                    VALUES (:id, :path, :method, :status, :reason)
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "path": record.path,
                    "method": record.method,
                    "status": record.status_code,
                    "reason": record.reason,
                },
            )

    async def get_summary(self, range_hours: int) -> dict:
        if not self._tables_ensured:
            await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT
                        COUNT(*) AS total_blocked,
                        COALESCE(SUM(CASE WHEN status_code = 401 THEN 1 ELSE 0 END), 0) AS auth_blocks,
                        COALESCE(SUM(CASE WHEN status_code = 413 THEN 1 ELSE 0 END), 0) AS size_blocks,
                        COALESCE(SUM(CASE WHEN status_code = 422 THEN 1 ELSE 0 END), 0) AS validation_blocks,
                        COALESCE(SUM(CASE WHEN status_code NOT IN (401, 413, 422) THEN 1 ELSE 0 END), 0) AS other_blocks
                    FROM blocked_requests
                    WHERE created_at >= now() - INTERVAL '{range_hours} hours'
                    """
                ),
            )
            row = result.mappings().fetchone()
            if not row:
                return {
                    "total_blocked": 0,
                    "auth_blocks": 0,
                    "size_blocks": 0,
                    "validation_blocks": 0,
                    "other_blocks": 0,
                }
            return {
                "total_blocked": int(row["total_blocked"]),
                "auth_blocks": int(row["auth_blocks"]),
                "size_blocks": int(row["size_blocks"]),
                "validation_blocks": int(row["validation_blocks"]),
                "other_blocks": int(row["other_blocks"]),
            }

    async def get_history(self, range_hours: int, bucket: str) -> list[dict]:
        if not self._tables_ensured:
            await self.ensure_tables()
        if bucket not in ("hour", "day"):
            bucket = "hour"
        bucket_expr = "date_trunc('hour', created_at)" if bucket == "hour" else "date_trunc('day', created_at)"
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT
                        {bucket_expr} AS bucket,
                        COUNT(*) AS blocked,
                        COALESCE(SUM(CASE WHEN status_code = 401 THEN 1 ELSE 0 END), 0) AS auth_blocks,
                        COALESCE(SUM(CASE WHEN status_code = 413 THEN 1 ELSE 0 END), 0) AS size_blocks,
                        COALESCE(SUM(CASE WHEN status_code = 422 THEN 1 ELSE 0 END), 0) AS validation_blocks
                    FROM blocked_requests
                    WHERE created_at >= now() - INTERVAL '{range_hours} hours'
                    GROUP BY bucket
                    ORDER BY bucket ASC
                    """
                ),
            )
            rows = result.mappings().all()
            return [
                {
                    "bucket": _format_bucket(row["bucket"]),
                    "blocked": int(row["blocked"]),
                    "auth_blocks": int(row["auth_blocks"]),
                    "size_blocks": int(row["size_blocks"]),
                    "validation_blocks": int(row["validation_blocks"]),
                }
                for row in rows
            ]


@dataclass
class ChatMetricRecord:
    """Structured metric row before insertion."""

    conversation_id: str
    model_name: str
    message_type: str
    stream: bool
    status_code: int
    started_at: str  # ISO 8601
    response_time_ms: float
    time_to_first_token_ms: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    tokens_per_second: Optional[float]
    message_count: Optional[int]
    input_chars: Optional[int]
    output_chars: Optional[int]
    rag_context_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    error_type: Optional[str] = None
    api_key_id: Optional[str] = None
    time_to_first_reasoning_token_ms: Optional[float] = None
    answer_tokens: Optional[int] = None
    time_to_first_answer_token_ms: Optional[float] = None
    chat_user_id: Optional[str] = None
    agent_iterations: Optional[int] = None
    tool_names: Optional[list] = None
    llm_generation_ms: Optional[float] = None
    cache_tokens: Optional[int] = None
    tool_result_tokens: Optional[int] = None


@dataclass
class VectorstoreMetricRecord:
    """Structured vectorstore retrieval metric row before insertion."""

    conversation_id: str
    model_name: str
    collection_names: str
    retrieval_time_ms: float
    chunks_retrieved: int
    status: str  # ok | error | skipped
    started_at: str  # ISO 8601
    error_type: Optional[str] = None
    rag_context_tokens: Optional[int] = None


class MetricsStore:
    """Async metrics store backed by a single ``chat_metrics`` table."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._tables_ensured = False

    async def ensure_tables(self) -> None:
        if self._tables_ensured:
            return
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS chat_metrics (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        stream BOOLEAN NOT NULL,
                        status_code INTEGER NOT NULL,
                        started_at timestamptz NOT NULL,
                        response_time_ms DOUBLE PRECISION NOT NULL,
                        time_to_first_token_ms DOUBLE PRECISION,
                        prompt_tokens INTEGER,
                        completion_tokens INTEGER,
                        total_tokens INTEGER,
                        tokens_per_second DOUBLE PRECISION,
                        message_count INTEGER,
                        input_chars INTEGER,
                        output_chars INTEGER,
                        rag_context_tokens INTEGER,
                        llm_generation_ms DOUBLE PRECISION,
                        error_type TEXT,
                        api_key_id TEXT,
                        created_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    ALTER TABLE chat_metrics
                    ADD COLUMN IF NOT EXISTS rag_context_tokens INTEGER
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    ALTER TABLE chat_metrics
                    ADD COLUMN IF NOT EXISTS api_key_id TEXT
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    ALTER TABLE chat_metrics
                    ADD COLUMN IF NOT EXISTS reasoning_tokens INTEGER
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    ALTER TABLE chat_metrics
                    ADD COLUMN IF NOT EXISTS time_to_first_reasoning_token_ms DOUBLE PRECISION
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    ALTER TABLE chat_metrics
                    ADD COLUMN IF NOT EXISTS chat_user_id TEXT,
                    ADD COLUMN IF NOT EXISTS agent_iterations INTEGER,
                    ADD COLUMN IF NOT EXISTS tool_names JSONB,
                    ADD COLUMN IF NOT EXISTS llm_generation_ms DOUBLE PRECISION,
                    ADD COLUMN IF NOT EXISTS cache_tokens INTEGER,
                    ADD COLUMN IF NOT EXISTS tool_result_tokens INTEGER
                    """
                )
            )
            # Columns with backfills are created inside this DO block: the
            # UPDATE only runs when the column is actually new (detected via
            # information_schema), so later restarts never overwrite rows that
            # legitimately store NULL — e.g. reasoning-only responses keep
            # ttfa NULL and error rows keep answer_tokens NULL.
            await conn.execute(
                text(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'chat_metrics'
                              AND column_name = 'time_to_first_answer_token_ms'
                        ) THEN
                            ALTER TABLE chat_metrics
                            ADD COLUMN time_to_first_answer_token_ms DOUBLE PRECISION;
                            -- Pre-migration rows tracked "first content token" in
                            -- time_to_first_token_ms — that semantic is exactly
                            -- the new answer-start marker.
                            UPDATE chat_metrics
                            SET time_to_first_answer_token_ms = time_to_first_token_ms;
                        END IF;

                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'chat_metrics'
                              AND column_name = 'answer_tokens'
                        ) THEN
                            ALTER TABLE chat_metrics ADD COLUMN answer_tokens INTEGER;
                            -- Provider-shaped rows store reasoning inside
                            -- completion -> difference; estimated rows stored
                            -- completion as answer-only -> keep completion.
                            UPDATE chat_metrics SET answer_tokens = CASE
                                WHEN completion_tokens IS NULL THEN NULL
                                WHEN reasoning_tokens IS NOT NULL AND reasoning_tokens > 0
                                     AND completion_tokens > reasoning_tokens
                                     THEN completion_tokens - reasoning_tokens
                                ELSE completion_tokens
                            END;
                        END IF;
                    END $$;
                    """
                )
            )
            # Backfill the dashboard/chat user that owned each conversation.
            await conn.execute(
                text(
                    """
                    DO $$
                    BEGIN
                        IF to_regclass('chat_user_conversations') IS NOT NULL THEN
                            UPDATE chat_metrics cm
                            SET chat_user_id = c.user_id
                            FROM chat_user_conversations c
                            WHERE cm.conversation_id = c.conversation_id
                              AND cm.chat_user_id IS NULL;
                        END IF;
                    END $$;
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_metrics_api_key_id
                    ON chat_metrics (api_key_id)
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_metrics_chat_user_id
                    ON chat_metrics (chat_user_id)
                    """
                )
            )
        self._tables_ensured = True

    async def save(self, record: ChatMetricRecord) -> None:
        if not self._tables_ensured:
            await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO chat_metrics (
                        id, conversation_id, model_name, message_type, stream,
                        status_code, started_at, response_time_ms, time_to_first_token_ms,
                        time_to_first_reasoning_token_ms, time_to_first_answer_token_ms,
                        prompt_tokens, completion_tokens, total_tokens, tokens_per_second,
                        message_count, input_chars, output_chars, rag_context_tokens,
                        reasoning_tokens, answer_tokens, error_type, api_key_id,
                        chat_user_id, agent_iterations, tool_names, llm_generation_ms,
                        cache_tokens, tool_result_tokens
                    ) VALUES (
                        :id, :cid, :model, :msg_type, :stream, :status, :started,
                        :rt_ms, :ttft_ms, :ttfrt_ms, :ttfa_ms, :pt, :ct, :tt, :tps,
                        :mc, :ic, :oc, :rag_tokens, :reasoning_tokens, :answer_tokens,
                        :err, :api_key_id, :chat_user_id, :agent_iterations,
                        CAST(:tool_names AS jsonb), :llm_gen_ms,
                        :cache_tokens, :tool_result_tokens
                    )
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "cid": record.conversation_id,
                    "model": record.model_name,
                    "msg_type": record.message_type,
                    "stream": record.stream,
                    "status": record.status_code,
                    "started": datetime.fromisoformat(record.started_at),
                    "rt_ms": record.response_time_ms,
                    "ttft_ms": record.time_to_first_token_ms,
                    "ttfrt_ms": record.time_to_first_reasoning_token_ms,
                    "ttfa_ms": record.time_to_first_answer_token_ms,
                    "pt": record.prompt_tokens,
                    "ct": record.completion_tokens,
                    "tt": record.total_tokens,
                    "tps": record.tokens_per_second,
                    "mc": record.message_count,
                    "ic": record.input_chars,
                    "oc": record.output_chars,
                    "rag_tokens": record.rag_context_tokens,
                    "reasoning_tokens": record.reasoning_tokens,
                    "answer_tokens": record.answer_tokens,
                    "err": record.error_type,
                    "api_key_id": record.api_key_id,
                    "chat_user_id": record.chat_user_id,
                    "agent_iterations": record.agent_iterations,
                    "tool_names": (
                        json.dumps(record.tool_names) if record.tool_names else None
                    ),
                    "llm_gen_ms": record.llm_generation_ms,
                    "cache_tokens": record.cache_tokens,
                    "tool_result_tokens": record.tool_result_tokens,
                },
            )

    async def get_conversation_metrics(self, conversation_id: str) -> dict:
        """Aggregated chat metrics for a single conversation."""
        if not self._tables_ensured:
            await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT
                        COUNT(*) AS request_count,
                        COALESCE(AVG(response_time_ms) FILTER (WHERE status_code < 400), 0) AS avg_response_time_ms,
                        COALESCE(AVG(time_to_first_token_ms) FILTER (WHERE status_code < 400), 0) AS avg_time_to_first_token_ms,
                        COALESCE(AVG(time_to_first_reasoning_token_ms) FILTER (WHERE status_code < 400), 0) AS avg_time_to_first_reasoning_token_ms,
                        COALESCE(AVG(time_to_first_answer_token_ms) FILTER (WHERE status_code < 400), 0) AS avg_time_to_first_answer_token_ms,
                        COALESCE(AVG(COALESCE(time_to_first_reasoning_token_ms, time_to_first_token_ms)) FILTER (WHERE status_code < 400), 0) AS avg_time_to_first_generation_ms,
                        COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                        COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens,
                        COALESCE(SUM(answer_tokens), 0) AS total_answer_tokens,
                        COALESCE(SUM(total_tokens), 0) AS total_tokens,
                        COALESCE(SUM(prompt_tokens) FILTER (WHERE status_code < 400) * 1000.0 / NULLIF(SUM(GREATEST(1, CASE WHEN llm_generation_ms IS NOT NULL AND llm_generation_ms >= 1 AND llm_generation_ms <= COALESCE(response_time_ms, 1e15) * 1.5 THEN llm_generation_ms ELSE COALESCE(response_time_ms, 0) END)) FILTER (WHERE status_code < 400), 0), 0) AS avg_prompt_tokens_per_second,
                        COALESCE(SUM(COALESCE(completion_tokens, answer_tokens)) FILTER (WHERE status_code < 400) * 1000.0 / NULLIF(SUM(GREATEST(1, CASE WHEN llm_generation_ms IS NOT NULL AND llm_generation_ms >= 1 AND llm_generation_ms <= COALESCE(response_time_ms, 1e15) * 1.5 THEN llm_generation_ms ELSE COALESCE(response_time_ms, 0) END)) FILTER (WHERE status_code < 400), 0), 0) AS avg_completion_tokens_per_second,
                        COALESCE(SUM(input_chars), 0) AS total_input_chars,
                        COALESCE(SUM(output_chars), 0) AS total_output_chars,
                        COALESCE(SUM(rag_context_tokens), 0) AS total_rag_context_tokens,
                        COALESCE(SUM(reasoning_tokens), 0) AS total_reasoning_tokens
                    FROM chat_metrics
                    WHERE conversation_id = :conversation_id
                      AND message_type = 'chat'
                    """
                ),
                {"conversation_id": conversation_id},
            )
            row = result.mappings().fetchone()
            if not row:
                return {
                    "request_count": 0,
                    "avg_response_time_ms": 0.0,
                    "avg_time_to_first_token_ms": 0.0,
                    "avg_time_to_first_reasoning_token_ms": 0.0,
                    "avg_time_to_first_answer_token_ms": 0.0,
                    "avg_time_to_first_generation_ms": 0.0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_answer_tokens": 0,
                    "total_tokens": 0,
                    "avg_prompt_tokens_per_second": 0.0,
                    "avg_completion_tokens_per_second": 0.0,
                    "total_input_chars": 0,
                    "total_output_chars": 0,
                    "total_rag_context_tokens": 0,
                    "total_reasoning_tokens": 0,
                }
            return {
                "request_count": int(row["request_count"]),
                "avg_response_time_ms": float(row["avg_response_time_ms"]),
                "avg_time_to_first_token_ms": float(row["avg_time_to_first_token_ms"]),
                "avg_time_to_first_reasoning_token_ms": float(row["avg_time_to_first_reasoning_token_ms"]),
                "avg_time_to_first_answer_token_ms": float(row["avg_time_to_first_answer_token_ms"]),
                "avg_time_to_first_generation_ms": float(row["avg_time_to_first_generation_ms"]),
                "total_prompt_tokens": int(row["total_prompt_tokens"]),
                "total_completion_tokens": int(row["total_completion_tokens"]),
                "total_answer_tokens": int(row["total_answer_tokens"]),
                "total_tokens": int(row["total_tokens"]),
                "avg_prompt_tokens_per_second": float(row["avg_prompt_tokens_per_second"]),
                "avg_completion_tokens_per_second": float(row["avg_completion_tokens_per_second"]),
                "total_input_chars": int(row["total_input_chars"]),
                "total_output_chars": int(row["total_output_chars"]),
                "total_rag_context_tokens": int(row["total_rag_context_tokens"]),
                "total_reasoning_tokens": int(row["total_reasoning_tokens"]),
            }

    async def get_summary(self, range_hours: int) -> dict:
        """Overall summary within the last N hours."""
        if not self._tables_ensured:
            await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT
                        COUNT(*) AS total_requests,
                        COALESCE(SUM(message_count), 0) AS total_messages,
                        COALESCE(AVG(response_time_ms) FILTER (WHERE status_code < 400), 0) AS avg_response_time_ms,
                        COALESCE(AVG(time_to_first_token_ms) FILTER (WHERE status_code < 400), 0) AS avg_time_to_first_token_ms,
                        COALESCE(AVG(time_to_first_reasoning_token_ms) FILTER (WHERE status_code < 400), 0) AS avg_time_to_first_reasoning_token_ms,
                        COALESCE(AVG(time_to_first_answer_token_ms) FILTER (WHERE status_code < 400), 0) AS avg_time_to_first_answer_token_ms,
                        COALESCE(AVG(COALESCE(time_to_first_reasoning_token_ms, time_to_first_token_ms)) FILTER (WHERE status_code < 400), 0) AS avg_time_to_first_generation_ms,
                        COALESCE(percentile_cont(0.5) WITHIN GROUP (ORDER BY response_time_ms) FILTER (WHERE status_code < 400), 0) AS p50_response_time_ms,
                        COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY response_time_ms) FILTER (WHERE status_code < 400), 0) AS p95_response_time_ms,
                        COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY time_to_first_token_ms) FILTER (WHERE status_code < 400), 0) AS p95_time_to_first_token_ms,
                        COALESCE(percentile_cont(0.5) WITHIN GROUP (ORDER BY time_to_first_answer_token_ms) FILTER (WHERE status_code < 400), 0) AS p50_time_to_first_answer_token_ms,
                        COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY time_to_first_answer_token_ms) FILTER (WHERE status_code < 400), 0) AS p95_time_to_first_answer_token_ms,
                        COALESCE(SUM(prompt_tokens) FILTER (WHERE status_code < 400) * 1000.0 / NULLIF(SUM(GREATEST(1, CASE WHEN llm_generation_ms IS NOT NULL AND llm_generation_ms >= 1 AND llm_generation_ms <= COALESCE(response_time_ms, 1e15) * 1.5 THEN llm_generation_ms ELSE COALESCE(response_time_ms, 0) END)) FILTER (WHERE status_code < 400), 0), 0) AS avg_prompt_tokens_per_second,
                        COALESCE(SUM(COALESCE(completion_tokens, answer_tokens)) FILTER (WHERE status_code < 400) * 1000.0 / NULLIF(SUM(GREATEST(1, CASE WHEN llm_generation_ms IS NOT NULL AND llm_generation_ms >= 1 AND llm_generation_ms <= COALESCE(response_time_ms, 1e15) * 1.5 THEN llm_generation_ms ELSE COALESCE(response_time_ms, 0) END)) FILTER (WHERE status_code < 400), 0), 0) AS avg_completion_tokens_per_second,
                        COALESCE(SUM(COALESCE(completion_tokens, answer_tokens)) FILTER (WHERE status_code < 400) * 1000.0 / NULLIF(SUM(GREATEST(1, CASE WHEN llm_generation_ms IS NOT NULL AND llm_generation_ms >= 1 AND llm_generation_ms <= COALESCE(response_time_ms, 1e15) * 1.5 THEN llm_generation_ms ELSE COALESCE(response_time_ms, 0) END)) FILTER (WHERE status_code < 400), 0), 0) AS avg_tokens_per_second,
                        COALESCE(SUM(total_tokens), 0) AS total_tokens,
                        COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                        COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens,
                        COALESCE(SUM(answer_tokens), 0) AS total_answer_tokens,
                        COALESCE(SUM(cache_tokens), 0) AS total_cache_tokens,
                        COALESCE(SUM(tool_result_tokens), 0) AS total_tool_result_tokens,
                        COALESCE(SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END), 0) AS error_count,
                        COALESCE(SUM(rag_context_tokens), 0) AS total_rag_context_tokens,
                        COALESCE(SUM(reasoning_tokens), 0) AS total_reasoning_tokens
                    FROM chat_metrics
                    WHERE started_at >= now() - INTERVAL '{range_hours} hours'
                      AND message_type = 'chat'
                    """
                ),
            )
            row = result.mappings().fetchone()
            if not row:
                return {
                    "total_requests": 0,
                    "total_messages": 0,
                    "avg_response_time_ms": 0.0,
                    "avg_time_to_first_token_ms": 0.0,
                    "avg_time_to_first_reasoning_token_ms": 0.0,
                    "avg_time_to_first_answer_token_ms": 0.0,
                    "avg_time_to_first_generation_ms": 0.0,
                    "p50_response_time_ms": 0.0,
                    "p95_response_time_ms": 0.0,
                    "p95_time_to_first_token_ms": 0.0,
                    "p50_time_to_first_answer_token_ms": 0.0,
                    "p95_time_to_first_answer_token_ms": 0.0,
                    "avg_prompt_tokens_per_second": 0.0,
                    "avg_completion_tokens_per_second": 0.0,
                    "avg_tokens_per_second": 0.0,
                    "total_tokens": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_answer_tokens": 0,
                    "total_cache_tokens": 0,
                    "total_tool_result_tokens": 0,
                    "error_rate": 0.0,
                    "total_rag_context_tokens": 0,
                    "total_reasoning_tokens": 0,
                }
            total_requests = int(row["total_requests"])
            error_count = int(row["error_count"])
            error_rate = (error_count / total_requests) if total_requests > 0 else 0.0
            return {
                "total_requests": total_requests,
                "total_messages": int(row["total_messages"]),
                "avg_response_time_ms": float(row["avg_response_time_ms"]),
                "avg_time_to_first_token_ms": float(row["avg_time_to_first_token_ms"]),
                "avg_time_to_first_reasoning_token_ms": float(row["avg_time_to_first_reasoning_token_ms"]),
                "avg_time_to_first_answer_token_ms": float(row["avg_time_to_first_answer_token_ms"]),
                "avg_time_to_first_generation_ms": float(row["avg_time_to_first_generation_ms"]),
                "p50_response_time_ms": float(row["p50_response_time_ms"]),
                "p95_response_time_ms": float(row["p95_response_time_ms"]),
                "p95_time_to_first_token_ms": float(row["p95_time_to_first_token_ms"]),
                "p50_time_to_first_answer_token_ms": float(row["p50_time_to_first_answer_token_ms"]),
                "p95_time_to_first_answer_token_ms": float(row["p95_time_to_first_answer_token_ms"]),
                "avg_prompt_tokens_per_second": float(row["avg_prompt_tokens_per_second"]),
                "avg_completion_tokens_per_second": float(row["avg_completion_tokens_per_second"]),
                "avg_tokens_per_second": float(row["avg_tokens_per_second"]),
                "total_tokens": int(row["total_tokens"]),
                "total_prompt_tokens": int(row["total_prompt_tokens"]),
                "total_completion_tokens": int(row["total_completion_tokens"]),
                "total_answer_tokens": int(row["total_answer_tokens"]),
                "total_cache_tokens": int(row["total_cache_tokens"]),
                "total_tool_result_tokens": int(row["total_tool_result_tokens"]),
                "error_rate": round(error_rate, 4),
                "total_rag_context_tokens": int(row["total_rag_context_tokens"]),
                "total_reasoning_tokens": int(row["total_reasoning_tokens"]),
            }

    async def get_by_model(self, range_hours: int) -> list[dict]:
        """Aggregated metrics grouped by model_name."""
        if not self._tables_ensured:
            await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT
                        model_name,
                        COUNT(*) AS requests,
                        COALESCE(SUM(message_count), 0) AS total_messages,
                        COALESCE(AVG(response_time_ms) FILTER (WHERE status_code < 400), 0) AS avg_response_time_ms,
                        COALESCE(AVG(time_to_first_token_ms) FILTER (WHERE status_code < 400), 0) AS avg_time_to_first_token_ms,
                        COALESCE(AVG(time_to_first_reasoning_token_ms) FILTER (WHERE status_code < 400), 0) AS avg_time_to_first_reasoning_token_ms,
                        COALESCE(AVG(time_to_first_answer_token_ms) FILTER (WHERE status_code < 400), 0) AS avg_time_to_first_answer_token_ms,
                        COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY response_time_ms) FILTER (WHERE status_code < 400), 0) AS p95_response_time_ms,
                        COALESCE(SUM(prompt_tokens) FILTER (WHERE status_code < 400) * 1000.0 / NULLIF(SUM(GREATEST(1, CASE WHEN llm_generation_ms IS NOT NULL AND llm_generation_ms >= 1 AND llm_generation_ms <= COALESCE(response_time_ms, 1e15) * 1.5 THEN llm_generation_ms ELSE COALESCE(response_time_ms, 0) END)) FILTER (WHERE status_code < 400), 0), 0) AS avg_prompt_tokens_per_second,
                        COALESCE(SUM(COALESCE(completion_tokens, answer_tokens)) FILTER (WHERE status_code < 400) * 1000.0 / NULLIF(SUM(GREATEST(1, CASE WHEN llm_generation_ms IS NOT NULL AND llm_generation_ms >= 1 AND llm_generation_ms <= COALESCE(response_time_ms, 1e15) * 1.5 THEN llm_generation_ms ELSE COALESCE(response_time_ms, 0) END)) FILTER (WHERE status_code < 400), 0), 0) AS avg_completion_tokens_per_second,
                        COALESCE(SUM(total_tokens), 0) AS total_tokens,
                        COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                        COALESCE(SUM(answer_tokens), 0) AS total_answer_tokens,
                        COALESCE(SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END), 0) AS errors,
                        COALESCE(SUM(rag_context_tokens), 0) AS total_rag_context_tokens,
                        COALESCE(SUM(reasoning_tokens), 0) AS total_reasoning_tokens
                    FROM chat_metrics
                    WHERE started_at >= now() - INTERVAL '{range_hours} hours'
                      AND message_type = 'chat'
                    GROUP BY model_name
                    ORDER BY requests DESC
                    """
                ),
            )
            rows = result.mappings().all()
            out: list[dict] = []
            for row in rows:
                requests = int(row["requests"])
                errors = int(row["errors"])
                out.append(
                    {
                        "model_name": str(row["model_name"]),
                        "requests": requests,
                        "total_messages": int(row["total_messages"]),
                        "avg_response_time_ms": float(row["avg_response_time_ms"]),
                        "avg_time_to_first_token_ms": float(row["avg_time_to_first_token_ms"]),
                        "avg_time_to_first_reasoning_token_ms": float(row["avg_time_to_first_reasoning_token_ms"]),
                        "avg_time_to_first_answer_token_ms": float(row["avg_time_to_first_answer_token_ms"]),
                        "p95_response_time_ms": float(row["p95_response_time_ms"]),
                        "avg_prompt_tokens_per_second": float(row["avg_prompt_tokens_per_second"]),
                        "avg_completion_tokens_per_second": float(row["avg_completion_tokens_per_second"]),
                        "total_tokens": int(row["total_tokens"]),
                        "total_prompt_tokens": int(row["total_prompt_tokens"]),
                        "total_answer_tokens": int(row["total_answer_tokens"]),
                        "error_rate": round(errors / requests, 4) if requests > 0 else 0.0,
                        "total_rag_context_tokens": int(row["total_rag_context_tokens"]),
                        "total_reasoning_tokens": int(row["total_reasoning_tokens"]),
                    }
                )
            return out

    async def get_history(self, range_hours: int, bucket: str, model: Optional[str]) -> list[dict]:
        """Time-series buckets for charting."""
        if not self._tables_ensured:
            await self.ensure_tables()
        if bucket not in ("hour", "day"):
            bucket = "hour"
        bucket_expr = "date_trunc('hour', started_at)" if bucket == "hour" else "date_trunc('day', started_at)"
        async with self._engine.begin() as conn:
            query = f"""
                SELECT
                    {bucket_expr} AS bucket,
                    COUNT(*) AS requests,
                    COALESCE(SUM(message_count), 0) AS total_messages,
                    COALESCE(AVG(response_time_ms) FILTER (WHERE status_code < 400), 0) AS avg_response_time_ms,
                    COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY response_time_ms) FILTER (WHERE status_code < 400), 0) AS p95_response_time_ms,
                    COALESCE(AVG(time_to_first_token_ms) FILTER (WHERE status_code < 400), 0) AS avg_time_to_first_token_ms,
                    COALESCE(AVG(time_to_first_reasoning_token_ms) FILTER (WHERE status_code < 400), 0) AS avg_time_to_first_reasoning_token_ms,
                    COALESCE(AVG(time_to_first_answer_token_ms) FILTER (WHERE status_code < 400), 0) AS avg_time_to_first_answer_token_ms,
                    COALESCE(SUM(prompt_tokens) FILTER (WHERE status_code < 400) * 1000.0 / NULLIF(SUM(GREATEST(1, CASE WHEN llm_generation_ms IS NOT NULL AND llm_generation_ms >= 1 AND llm_generation_ms <= COALESCE(response_time_ms, 1e15) * 1.5 THEN llm_generation_ms ELSE COALESCE(response_time_ms, 0) END)) FILTER (WHERE status_code < 400), 0), 0) AS avg_prompt_tokens_per_second,
                    COALESCE(SUM(COALESCE(completion_tokens, answer_tokens)) FILTER (WHERE status_code < 400) * 1000.0 / NULLIF(SUM(GREATEST(1, CASE WHEN llm_generation_ms IS NOT NULL AND llm_generation_ms >= 1 AND llm_generation_ms <= COALESCE(response_time_ms, 1e15) * 1.5 THEN llm_generation_ms ELSE COALESCE(response_time_ms, 0) END)) FILTER (WHERE status_code < 400), 0), 0) AS avg_completion_tokens_per_second,
                    COALESCE(SUM(COALESCE(completion_tokens, answer_tokens)) FILTER (WHERE status_code < 400) * 1000.0 / NULLIF(SUM(GREATEST(1, CASE WHEN llm_generation_ms IS NOT NULL AND llm_generation_ms >= 1 AND llm_generation_ms <= COALESCE(response_time_ms, 1e15) * 1.5 THEN llm_generation_ms ELSE COALESCE(response_time_ms, 0) END)) FILTER (WHERE status_code < 400), 0), 0) AS avg_tokens_per_second,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens,
                    COALESCE(SUM(answer_tokens), 0) AS total_answer_tokens,
                    COALESCE(SUM(rag_context_tokens), 0) AS total_rag_context_tokens,
                    COALESCE(SUM(reasoning_tokens), 0) AS total_reasoning_tokens
                FROM chat_metrics
                WHERE started_at >= now() - INTERVAL '{range_hours} hours'
                AND message_type = 'chat'
                {'AND model_name = :model' if model else ''}
                GROUP BY bucket
                ORDER BY bucket ASC
            """
            params: dict = {}
            if model:
                params["model"] = model
            result = await conn.execute(text(query), params)
            rows = result.mappings().all()
            return [
                {
                    "bucket": _format_bucket(row["bucket"]),
                    "requests": int(row["requests"]),
                    "total_messages": int(row["total_messages"]),
                    "avg_response_time_ms": float(row["avg_response_time_ms"]),
                    "p95_response_time_ms": float(row["p95_response_time_ms"]),
                    "avg_time_to_first_token_ms": float(row["avg_time_to_first_token_ms"]),
                    "avg_time_to_first_reasoning_token_ms": float(row["avg_time_to_first_reasoning_token_ms"]),
                    "avg_time_to_first_answer_token_ms": float(row["avg_time_to_first_answer_token_ms"]),
                    "avg_prompt_tokens_per_second": float(row["avg_prompt_tokens_per_second"]),
                    "avg_completion_tokens_per_second": float(row["avg_completion_tokens_per_second"]),
                    "avg_tokens_per_second": float(row["avg_tokens_per_second"]),
                    "total_tokens": int(row["total_tokens"]),
                    "total_prompt_tokens": int(row["total_prompt_tokens"]),
                    "total_completion_tokens": int(row["total_completion_tokens"]),
                    "total_answer_tokens": int(row["total_answer_tokens"]),
                    "total_rag_context_tokens": int(row["total_rag_context_tokens"]),
                    "total_reasoning_tokens": int(row["total_reasoning_tokens"]),
                }
                for row in rows
            ]

    async def get_history_by_api_key(self, range_hours: int, bucket: str) -> list[dict]:
        """Time-series buckets for charting grouped by api_key_id."""
        if not self._tables_ensured:
            await self.ensure_tables()
        if bucket not in ("hour", "day"):
            bucket = "hour"
        bucket_expr = "date_trunc('hour', started_at)" if bucket == "hour" else "date_trunc('day', started_at)"
        async with self._engine.begin() as conn:
            # Get top api keys by total tokens in the range
            top_result = await conn.execute(
                text(
                    f"""
                    SELECT api_key_id, SUM(total_tokens) AS total
                    FROM chat_metrics
                    WHERE started_at >= now() - INTERVAL '{range_hours} hours'
                        AND api_key_id IS NOT NULL
                        AND message_type = 'chat'
                    GROUP BY api_key_id
                    ORDER BY total DESC
                    LIMIT 10
                    """
                )
            )
            top_keys = [str(row["api_key_id"]) for row in top_result.mappings().all()]
            if not top_keys:
                return []

            # Pivot query: one row per bucket, one column per api_key_id
            placeholders = ", ".join(f":k{i}" for i in range(len(top_keys)))
            pivot_cols = ", ".join(
                f"COALESCE(SUM(CASE WHEN api_key_id = :k{i} THEN total_tokens ELSE 0 END), 0) AS key_{i}"
                for i in range(len(top_keys))
            )
            query = f"""
                SELECT
                    {bucket_expr} AS bucket,
                    {pivot_cols}
                FROM chat_metrics
                WHERE started_at >= now() - INTERVAL '{range_hours} hours'
                    AND api_key_id IN ({placeholders})
                    AND message_type = 'chat'
                GROUP BY bucket
                ORDER BY bucket ASC
            """
            params = {f"k{i}": key for i, key in enumerate(top_keys)}
            result = await conn.execute(text(query), params)
            rows = result.mappings().all()
            out = []
            for row in rows:
                entry: dict = {"bucket": _format_bucket(row["bucket"])}
                for i, key in enumerate(top_keys):
                    entry[key] = int(row[f"key_{i}"])
                out.append(entry)
            return out

    async def get_usage_stats(self, range_hours: int) -> dict:
        """Usage-shape stats for the dashboard "Nutzung" tab.

        Activity heatmap, error/retrieval breakdowns, ratio metrics and top
        lists. All chat_metrics queries filter message_type='chat' so title/tag
        generation calls do not skew user-facing numbers.
        """
        if not self._tables_ensured:
            await self.ensure_tables()
        interval = f"now() - INTERVAL '{int(range_hours)} hours'"
        out: dict = {
            "hourly_activity": [],
            "error_breakdown": [],
            "stream_share_pct": 0.0,
            "avg_message_count": 0.0,
            "rag_share_pct": 0.0,
            "reasoning_share_pct": 0.0,
            "avg_agent_iterations": 0.0,
            "top_tools": [],
            "top_users": [],
            "top_collections": [],
            "retrieval_status": {"ok": 0, "error": 0, "skipped": 0},
        }
        async with self._engine.begin() as conn:
            # vectorstore_metrics / chat_users are owned by other stores and may
            # not exist yet on fresh installs — guard their queries.
            probe = await conn.execute(
                text(
                    """
                    SELECT to_regclass('vectorstore_metrics') AS vs,
                           to_regclass('chat_users') AS cu
                    """
                )
            )
            probes = probe.mappings().fetchone() or {}
            has_vs_metrics = probes.get("vs") is not None
            has_chat_users = probes.get("cu") is not None

            result = await conn.execute(
                text(
                    f"""
                    SELECT EXTRACT(dow FROM started_at)::int AS dow,
                           EXTRACT(hour FROM started_at)::int AS hour,
                           COUNT(*) AS count
                    FROM chat_metrics
                    WHERE started_at >= {interval} AND message_type = 'chat'
                    GROUP BY 1, 2
                    """
                )
            )
            out["hourly_activity"] = [
                {
                    "dow": int(r["dow"]),
                    "hour": int(r["hour"]),
                    "count": int(r["count"]),
                }
                for r in result.mappings().all()
            ]

            result = await conn.execute(
                text(
                    f"""
                    SELECT COALESCE(error_type, 'unknown') AS error_type,
                           COUNT(*) AS count
                    FROM chat_metrics
                    WHERE started_at >= {interval} AND message_type = 'chat'
                      AND (status_code >= 400 OR error_type IS NOT NULL)
                    GROUP BY error_type
                    ORDER BY count DESC
                    """
                )
            )
            out["error_breakdown"] = [
                {"error_type": str(r["error_type"]), "count": int(r["count"])}
                for r in result.mappings().all()
            ]

            result = await conn.execute(
                text(
                    f"""
                    SELECT
                        COALESCE(AVG(CASE WHEN stream THEN 1.0 ELSE 0.0 END) * 100, 0) AS stream_share_pct,
                        COALESCE(AVG(message_count), 0) AS avg_message_count,
                        COALESCE(SUM(rag_context_tokens) * 100.0 / NULLIF(SUM(prompt_tokens), 0), 0) AS rag_share_pct,
                        -- Share of requests that produced reasoning tokens
                        -- (proxy for "reasoning enabled"), not the token share.
                        COALESCE(COUNT(*) FILTER (WHERE reasoning_tokens > 0) * 100.0 / NULLIF(COUNT(*), 0), 0) AS reasoning_share_pct,
                        COALESCE(AVG(agent_iterations) FILTER (WHERE agent_iterations IS NOT NULL), 0) AS avg_agent_iterations
                    FROM chat_metrics
                    WHERE started_at >= {interval} AND message_type = 'chat'
                    """
                )
            )
            row = result.mappings().fetchone()
            if row:
                out["stream_share_pct"] = round(float(row["stream_share_pct"]), 1)
                out["avg_message_count"] = round(float(row["avg_message_count"]), 1)
                out["rag_share_pct"] = round(float(row["rag_share_pct"]), 1)
                out["reasoning_share_pct"] = round(float(row["reasoning_share_pct"]), 1)
                out["avg_agent_iterations"] = round(
                    float(row["avg_agent_iterations"]), 1
                )

            result = await conn.execute(
                text(
                    f"""
                    SELECT t.name AS name, COUNT(*) AS count
                    FROM chat_metrics,
                         jsonb_array_elements_text(tool_names) AS t(name)
                    WHERE started_at >= {interval} AND message_type = 'chat'
                    GROUP BY t.name
                    ORDER BY count DESC
                    LIMIT 10
                    """
                )
            )
            out["top_tools"] = [
                {"name": str(r["name"]), "count": int(r["count"])}
                for r in result.mappings().all()
            ]

            user_join = (
                "LEFT JOIN chat_users cu ON cu.id = cm.chat_user_id"
                if has_chat_users
                else ""
            )
            user_name_cols = (
                "MAX(cu.display_name) AS display_name, MAX(cu.username) AS username"
                if has_chat_users
                else "NULL AS display_name, NULL AS username"
            )
            result = await conn.execute(
                text(
                    f"""
                    SELECT cm.chat_user_id AS user_id,
                           COUNT(*) AS requests,
                           COALESCE(SUM(cm.total_tokens), 0) AS total_tokens,
                           {user_name_cols}
                    FROM chat_metrics cm
                    {user_join}
                    WHERE cm.started_at >= {interval} AND cm.message_type = 'chat'
                    GROUP BY cm.chat_user_id
                    ORDER BY requests DESC
                    LIMIT 10
                    """
                )
            )
            out["top_users"] = [
                {
                    "user_id": str(r["user_id"]) if r["user_id"] else None,
                    "display_name": (
                        str(r["display_name"] or r["username"])
                        if (r["display_name"] or r["username"])
                        else None
                    ),
                    "requests": int(r["requests"]),
                    "total_tokens": int(r["total_tokens"]),
                }
                for r in result.mappings().all()
            ]

            if has_vs_metrics:
                result = await conn.execute(
                    text(
                        f"""
                        SELECT btrim(c.name) AS name, COUNT(*) AS count
                        FROM vectorstore_metrics,
                             unnest(string_to_array(NULLIF(collection_names, ''), ',')) AS c(name)
                        WHERE started_at >= {interval}
                        GROUP BY btrim(c.name)
                        ORDER BY count DESC
                        LIMIT 10
                        """
                    )
                )
                out["top_collections"] = [
                    {"name": str(r["name"]), "count": int(r["count"])}
                    for r in result.mappings().all()
                ]

                result = await conn.execute(
                    text(
                        f"""
                        SELECT status, COUNT(*) AS count
                        FROM vectorstore_metrics
                        WHERE started_at >= {interval}
                        GROUP BY status
                        """
                    )
                )
                status_counts = {
                    str(r["status"]): int(r["count"])
                    for r in result.mappings().all()
                }
                out["retrieval_status"] = {
                    "ok": status_counts.get("ok", 0),
                    "error": status_counts.get("error", 0),
                    "skipped": status_counts.get("skipped", 0),
                }
        return out


class VectorstoreMetricsStore:
    """Async store for per-request vectorstore retrieval metrics."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._tables_ensured = False

    async def ensure_tables(self) -> None:
        if self._tables_ensured:
            return
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS vectorstore_metrics (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        collection_names TEXT NOT NULL,
                        retrieval_time_ms DOUBLE PRECISION NOT NULL,
                        chunks_retrieved INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        started_at timestamptz NOT NULL,
                        error_type TEXT,
                        rag_context_tokens INTEGER,
                        created_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    ALTER TABLE vectorstore_metrics
                    ADD COLUMN IF NOT EXISTS rag_context_tokens INTEGER
                    """
                )
            )
        self._tables_ensured = True

    async def save(self, record: VectorstoreMetricRecord) -> None:
        if not self._tables_ensured:
            await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO vectorstore_metrics (
                        id, conversation_id, model_name, collection_names,
                        retrieval_time_ms, chunks_retrieved, status, started_at, error_type,
                        rag_context_tokens
                    ) VALUES (
                        :id, :cid, :model, :collections, :rt_ms, :chunks, :status, :started, :err,
                        :rag_tokens
                    )
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "cid": record.conversation_id,
                    "model": record.model_name,
                    "collections": record.collection_names,
                    "rt_ms": record.retrieval_time_ms,
                    "chunks": record.chunks_retrieved,
                    "status": record.status,
                    "started": datetime.fromisoformat(record.started_at),
                    "err": record.error_type,
                    "rag_tokens": record.rag_context_tokens,
                },
            )

    async def get_conversation_metrics(self, conversation_id: str) -> dict:
        """Aggregated vectorstore retrieval metrics for a single conversation."""
        if not self._tables_ensured:
            await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT
                        COUNT(*) AS retrieval_count,
                        COALESCE(AVG(retrieval_time_ms), 0) AS avg_retrieval_time_ms,
                        COALESCE(SUM(chunks_retrieved), 0) AS total_chunks,
                        COALESCE(SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END), 0) AS error_count,
                        COALESCE(SUM(rag_context_tokens), 0) AS total_rag_context_tokens
                    FROM vectorstore_metrics
                    WHERE conversation_id = :conversation_id
                    """
                ),
                {"conversation_id": conversation_id},
            )
            row = result.mappings().fetchone()
            if not row:
                return {
                    "retrieval_count": 0,
                    "avg_retrieval_time_ms": 0.0,
                    "total_chunks": 0,
                    "error_count": 0,
                    "total_rag_context_tokens": 0,
                }
            retrieval_count = int(row["retrieval_count"])
            error_count = int(row["error_count"])
            return {
                "retrieval_count": retrieval_count,
                "avg_retrieval_time_ms": float(row["avg_retrieval_time_ms"]),
                "total_chunks": int(row["total_chunks"]),
                "error_rate": round(error_count / retrieval_count, 4) if retrieval_count > 0 else 0.0,
                "total_rag_context_tokens": int(row["total_rag_context_tokens"]),
            }

    async def get_summary(self, range_hours: int) -> dict:
        """Overall summary within the last N hours."""
        if not self._tables_ensured:
            await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT
                        COUNT(*) AS total_retrievals,
                        COALESCE(AVG(retrieval_time_ms), 0) AS avg_retrieval_time_ms,
                        COALESCE(SUM(chunks_retrieved), 0) AS total_chunks,
                        COALESCE(SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END), 0) AS error_count,
                        COALESCE(SUM(rag_context_tokens), 0) AS total_rag_context_tokens
                    FROM vectorstore_metrics
                    WHERE started_at >= now() - INTERVAL '{range_hours} hours'
                    """
                ),
            )
            row = result.mappings().fetchone()
            if not row:
                return {
                    "total_retrievals": 0,
                    "avg_retrieval_time_ms": 0.0,
                    "total_chunks": 0,
                    "error_rate": 0.0,
                    "total_rag_context_tokens": 0,
                }
            total_retrievals = int(row["total_retrievals"])
            error_count = int(row["error_count"])
            return {
                "total_retrievals": total_retrievals,
                "avg_retrieval_time_ms": float(row["avg_retrieval_time_ms"]),
                "total_chunks": int(row["total_chunks"]),
                "error_rate": round(error_count / total_retrievals, 4) if total_retrievals > 0 else 0.0,
                "total_rag_context_tokens": int(row["total_rag_context_tokens"]),
            }

    async def get_by_model(self, range_hours: int) -> list[dict]:
        """Aggregated retrieval metrics grouped by model_name."""
        if not self._tables_ensured:
            await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT
                        model_name,
                        COUNT(*) AS retrievals,
                        COALESCE(AVG(retrieval_time_ms), 0) AS avg_retrieval_time_ms,
                        COALESCE(SUM(chunks_retrieved), 0) AS total_chunks,
                        COALESCE(SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END), 0) AS errors,
                        COALESCE(SUM(rag_context_tokens), 0) AS total_rag_context_tokens
                    FROM vectorstore_metrics
                    WHERE started_at >= now() - INTERVAL '{range_hours} hours'
                    GROUP BY model_name
                    ORDER BY retrievals DESC
                    """
                ),
            )
            rows = result.mappings().all()
            out: list[dict] = []
            for row in rows:
                retrievals = int(row["retrievals"])
                errors = int(row["errors"])
                out.append(
                    {
                        "model_name": str(row["model_name"]),
                        "retrievals": retrievals,
                        "avg_retrieval_time_ms": float(row["avg_retrieval_time_ms"]),
                        "total_chunks": int(row["total_chunks"]),
                        "error_rate": round(errors / retrievals, 4) if retrievals > 0 else 0.0,
                        "total_rag_context_tokens": int(row["total_rag_context_tokens"]),
                    }
                )
            return out

    async def get_history(self, range_hours: int, bucket: str, model: Optional[str] = None) -> list[dict]:
        """Time-series buckets for charting."""
        if not self._tables_ensured:
            await self.ensure_tables()
        if bucket not in ("hour", "day"):
            bucket = "hour"
        bucket_expr = "date_trunc('hour', started_at)" if bucket == "hour" else "date_trunc('day', started_at)"
        async with self._engine.begin() as conn:
            query = f"""
                SELECT
                    {bucket_expr} AS bucket,
                    COUNT(*) AS retrievals,
                    COALESCE(AVG(retrieval_time_ms), 0) AS avg_retrieval_time_ms,
                    COALESCE(SUM(chunks_retrieved), 0) AS total_chunks,
                    COALESCE(SUM(rag_context_tokens), 0) AS total_rag_context_tokens
                FROM vectorstore_metrics
                WHERE started_at >= now() - INTERVAL '{range_hours} hours'
                {'AND model_name = :model' if model else ''}
                GROUP BY bucket
                ORDER BY bucket ASC
            """
            params: dict = {}
            if model:
                params["model"] = model
            result = await conn.execute(text(query), params)
            rows = result.mappings().all()
            return [
                {
                    "bucket": _format_bucket(row["bucket"]),
                    "retrievals": int(row["retrievals"]),
                    "avg_retrieval_time_ms": float(row["avg_retrieval_time_ms"]),
                    "total_chunks": int(row["total_chunks"]),
                    "total_rag_context_tokens": int(row["total_rag_context_tokens"]),
                }
                for row in rows
            ]
