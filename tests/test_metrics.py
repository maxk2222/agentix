from __future__ import annotations

from typing import Any, List

import pytest

from datetime import datetime

from app.metrics import ChatMetricRecord, MetricsStore, VectorstoreMetricRecord, VectorstoreMetricsStore, _format_bucket


class FakeResult:
    def __init__(self, rows: List[dict[str, Any]]) -> None:
        self._rows = rows

    def mappings(self):
        return FakeMappings(self._rows)


class FakeMappings:
    def __init__(self, rows: List[dict[str, Any]]) -> None:
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return self._rows


class FakeConnection:
    def __init__(self, results: List[FakeResult] | None = None) -> None:
        self.results = results or []
        self.executed: List[tuple[str, dict[str, Any]]] = []

    async def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params or {}))
        if self.results:
            return self.results.pop(0)
        return FakeResult([])

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


class FakeEngine:
    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn

    def begin(self):
        return self._conn


@pytest.mark.asyncio
async def test_ensure_tables():
    conn = FakeConnection()
    store = MetricsStore(FakeEngine(conn))
    await store.ensure_tables()
    assert len(conn.executed) == 10
    assert "CREATE TABLE IF NOT EXISTS chat_metrics" in conn.executed[0][0]


@pytest.mark.asyncio
async def test_save():
    conn = FakeConnection()
    store = MetricsStore(FakeEngine(conn))
    record = ChatMetricRecord(
        conversation_id="cid-1",
        model_name="CyberSecurityAssistant",
        message_type="chat",
        stream=False,
        status_code=200,
        started_at="2024-01-01T00:00:00",
        response_time_ms=123.0,
        time_to_first_token_ms=None,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        tokens_per_second=5.0,
        message_count=2,
        input_chars=100,
        output_chars=200,
        reasoning_tokens=5,
        error_type=None,
        time_to_first_reasoning_token_ms=None,
    )
    await store.save(record)
    assert len(conn.executed) == 11  # ensure_tables (10 stmts) + insert
    assert "INSERT INTO chat_metrics" in conn.executed[10][0]


@pytest.mark.asyncio
async def test_save_keeps_timezone_aware_started_at():
    conn = FakeConnection()
    store = MetricsStore(FakeEngine(conn))
    record = ChatMetricRecord(
        conversation_id="cid-1",
        model_name="CyberSecurityAssistant",
        message_type="chat",
        stream=False,
        status_code=200,
        started_at="2024-01-01T00:00:00+00:00",
        response_time_ms=123.0,
        time_to_first_token_ms=None,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        tokens_per_second=5.0,
        message_count=2,
        input_chars=100,
        output_chars=200,
        reasoning_tokens=5,
        error_type=None,
        time_to_first_reasoning_token_ms=None,
    )
    await store.save(record)
    assert conn.executed[10][1]["started"].tzinfo is not None
    assert conn.executed[10][1]["started"].isoformat() == "2024-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_get_summary():
    conn = FakeConnection(
        results=[
            FakeResult(
                [
                    {
                        "total_requests": 42,
                        "total_messages": 100,
                        "avg_response_time_ms": 150.5,
                        "avg_time_to_first_token_ms": 45.2,
                        "avg_time_to_first_reasoning_token_ms": 12.3,
                        "avg_time_to_first_answer_token_ms": 55.0,
                        "avg_time_to_first_generation_ms": 12.3,
                        "p50_response_time_ms": 140.0,
                        "p95_response_time_ms": 400.0,
                        "p95_time_to_first_token_ms": 90.0,
                        "p50_time_to_first_answer_token_ms": 50.0,
                        "p95_time_to_first_answer_token_ms": 120.0,
                        "avg_prompt_tokens_per_second": 5.0,
                        "avg_completion_tokens_per_second": 7.3,
                        "avg_tokens_per_second": 6.5,
                        "total_tokens": 5000,
                        "total_prompt_tokens": 3000,
                        "total_completion_tokens": 1900,
                        "total_answer_tokens": 1800,
                        "total_cache_tokens": 250,
                        "total_tool_result_tokens": 400,
                        "error_count": 2,
                        "total_rag_context_tokens": 0,
                        "total_reasoning_tokens": 100,
                    }
                ]
            )
        ]
    )
    store = MetricsStore(FakeEngine(conn))
    store._tables_ensured = True
    summary = await store.get_summary(24)
    assert summary["total_requests"] == 42
    assert summary["total_messages"] == 100
    assert summary["avg_response_time_ms"] == 150.5
    assert summary["avg_prompt_tokens_per_second"] == 5.0
    assert summary["avg_completion_tokens_per_second"] == 7.3
    assert summary["total_tokens"] == 5000
    assert summary["total_reasoning_tokens"] == 100
    assert summary["error_rate"] == pytest.approx(2 / 42, 0.001)
    assert summary["avg_time_to_first_generation_ms"] == 12.3
    assert summary["total_completion_tokens"] == 1900
    assert summary["total_cache_tokens"] == 250
    assert summary["total_tool_result_tokens"] == 400


@pytest.mark.asyncio
async def test_summary_sql_filters_errors_and_adds_fields():
    """Latency aggregates ignore failed rows; TTFT prefers reasoning first."""
    conn = FakeConnection(results=[FakeResult([])])
    store = MetricsStore(FakeEngine(conn))
    store._tables_ensured = True
    await store.get_summary(24)
    sql = conn.executed[0][0]
    assert sql.count("FILTER (WHERE status_code < 400)") == 16
    assert (
        "AVG(COALESCE(time_to_first_reasoning_token_ms, time_to_first_token_ms))"
        " FILTER (WHERE status_code < 400)"
    ) in sql
    assert "SUM(cache_tokens)" in sql
    assert "SUM(tool_result_tokens)" in sql
    assert "SUM(completion_tokens)" in sql


@pytest.mark.asyncio
async def test_get_by_model():
    conn = FakeConnection(
        results=[
            FakeResult(
                [
                    {
                        "model_name": "CyberSecurityAssistant",
                        "requests": 10,
                        "total_messages": 25,
                        "avg_response_time_ms": 200.0,
                        "avg_time_to_first_token_ms": 60.0,
                        "avg_time_to_first_reasoning_token_ms": 15.0,
                        "avg_time_to_first_answer_token_ms": 70.0,
                        "p95_response_time_ms": 450.0,
                        "avg_prompt_tokens_per_second": 3.0,
                        "avg_completion_tokens_per_second": 5.0,
                        "total_tokens": 1000,
                        "total_prompt_tokens": 600,
                        "total_answer_tokens": 350,
                        "errors": 1,
                        "total_rag_context_tokens": 0,
                        "total_reasoning_tokens": 50,
                    }
                ]
            )
        ]
    )
    store = MetricsStore(FakeEngine(conn))
    store._tables_ensured = True
    rows = await store.get_by_model(24)
    assert len(rows) == 1
    assert rows[0]["model_name"] == "CyberSecurityAssistant"
    assert rows[0]["requests"] == 10
    assert rows[0]["error_rate"] == pytest.approx(0.1, 0.001)
    assert rows[0]["total_reasoning_tokens"] == 50


@pytest.mark.asyncio
async def test_get_history():
    conn = FakeConnection(
        results=[
            FakeResult(
                [
                    {
                        "bucket": "2024-01-01T00:00:00Z",
                        "requests": 5,
                        "total_messages": 12,
                        "avg_response_time_ms": 180.0,
                        "avg_time_to_first_token_ms": 50.0,
                        "avg_time_to_first_reasoning_token_ms": 10.0,
                        "avg_time_to_first_answer_token_ms": 60.0,
                        "p95_response_time_ms": 320.0,
                        "avg_prompt_tokens_per_second": 4.0,
                        "avg_completion_tokens_per_second": 6.0,
                        "avg_tokens_per_second": 5.0,
                        "total_tokens": 600,
                        "total_prompt_tokens": 300,
                        "total_completion_tokens": 300,
                        "total_answer_tokens": 270,
                        "total_rag_context_tokens": 0,
                        "total_reasoning_tokens": 30,
                    }
                ]
            )
        ]
    )
    store = MetricsStore(FakeEngine(conn))
    store._tables_ensured = True
    rows = await store.get_history(24, "hour", None)
    assert len(rows) == 1
    assert rows[0]["bucket"] == "2024-01-01T00:00:00Z"
    assert rows[0]["requests"] == 5
    assert rows[0]["total_reasoning_tokens"] == 30


@pytest.mark.asyncio
async def test_vectorstore_ensure_tables():
    conn = FakeConnection()
    store = VectorstoreMetricsStore(FakeEngine(conn))
    await store.ensure_tables()
    assert len(conn.executed) == 2
    assert "CREATE TABLE IF NOT EXISTS vectorstore_metrics" in conn.executed[0][0]


@pytest.mark.asyncio
async def test_vectorstore_save():
    conn = FakeConnection()
    store = VectorstoreMetricsStore(FakeEngine(conn))
    record = VectorstoreMetricRecord(
        conversation_id="cid-1",
        model_name="CyberSecurityAssistant",
        collection_names="BSI",
        retrieval_time_ms=45.0,
        chunks_retrieved=5,
        status="ok",
        started_at="2024-01-01T00:00:00",
        error_type=None,
    )
    await store.save(record)
    assert len(conn.executed) == 3
    assert "INSERT INTO vectorstore_metrics" in conn.executed[2][0]


@pytest.mark.asyncio
async def test_vectorstore_get_summary():
    conn = FakeConnection(
        results=[
            FakeResult(
                [
                    {
                        "total_retrievals": 20,
                        "avg_retrieval_time_ms": 55.0,
                        "total_chunks": 100,
                        "error_count": 1,
                        "total_rag_context_tokens": 0,
                    }
                ]
            )
        ]
    )
    store = VectorstoreMetricsStore(FakeEngine(conn))
    store._tables_ensured = True
    summary = await store.get_summary(24)
    assert summary["total_retrievals"] == 20
    assert summary["avg_retrieval_time_ms"] == 55.0
    assert summary["total_chunks"] == 100
    assert summary["error_rate"] == pytest.approx(1 / 20, 0.001)


@pytest.mark.asyncio
async def test_vectorstore_get_by_model():
    conn = FakeConnection(
        results=[
            FakeResult(
                [
                    {
                        "model_name": "CyberSecurityAssistant",
                        "retrievals": 10,
                        "avg_retrieval_time_ms": 40.0,
                        "total_chunks": 50,
                        "errors": 2,
                        "total_rag_context_tokens": 0,
                    }
                ]
            )
        ]
    )
    store = VectorstoreMetricsStore(FakeEngine(conn))
    store._tables_ensured = True
    rows = await store.get_by_model(24)
    assert len(rows) == 1
    assert rows[0]["model_name"] == "CyberSecurityAssistant"
    assert rows[0]["retrievals"] == 10
    assert rows[0]["error_rate"] == pytest.approx(0.2, 0.001)


@pytest.mark.asyncio
async def test_vectorstore_get_history():
    conn = FakeConnection(
        results=[
            FakeResult(
                [
                    {
                        "bucket": "2024-01-01T00:00:00Z",
                        "retrievals": 5,
                        "avg_retrieval_time_ms": 60.0,
                        "total_chunks": 25,
                        "total_rag_context_tokens": 0,
                    }
                ]
            )
        ]
    )
    store = VectorstoreMetricsStore(FakeEngine(conn))
    store._tables_ensured = True
    rows = await store.get_history(24, "hour", None)
    assert len(rows) == 1
    assert rows[0]["bucket"] == "2024-01-01T00:00:00Z"
    assert rows[0]["retrievals"] == 5
    assert rows[0]["total_chunks"] == 25


def test_format_bucket():
    assert _format_bucket("2024-01-01T12:34:56") == "2024-01-01T12:34:56"
    assert _format_bucket(datetime(2024, 1, 1, 12, 34, 56)) == "2024-01-01T12:34:56Z"
    assert _format_bucket(datetime(2024, 1, 1, 0, 0, 0)) == "2024-01-01T00:00:00Z"


@pytest.mark.asyncio
async def test_save_persists_llm_generation_ms():
    conn = FakeConnection()
    store = MetricsStore(FakeEngine(conn))
    record = ChatMetricRecord(
        conversation_id="cid-1",
        model_name="CyberSecurityAssistant",
        message_type="chat",
        stream=False,
        status_code=200,
        started_at="2024-01-01T00:00:00",
        response_time_ms=123.0,
        time_to_first_token_ms=None,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        tokens_per_second=5.0,
        message_count=2,
        input_chars=100,
        output_chars=200,
        llm_generation_ms=45.5,
    )
    await store.save(record)
    _, params = conn.executed[-1]
    assert params["llm_gen_ms"] == 45.5
    assert "llm_generation_ms" in conn.executed[-1][0]


@pytest.mark.asyncio
async def test_rate_aggregates_use_llm_generation_time():
    """inputPerSec/answerPerSec divide by LLM-active time, falling back to
    wall-clock response_time_ms for rows without usage metadata."""
    conn = FakeConnection(results=[FakeResult([{
        "total_requests": 1, "total_messages": 1, "avg_response_time_ms": 10.0,
        "avg_time_to_first_token_ms": 5.0, "avg_time_to_first_reasoning_token_ms": 2.0,
        "avg_time_to_first_answer_token_ms": 4.0, "avg_time_to_first_generation_ms": 2.0,
        "p50_response_time_ms": 10.0,
        "p95_response_time_ms": 10.0, "p95_time_to_first_token_ms": 5.0,
        "p50_time_to_first_answer_token_ms": 8.0,
        "p95_time_to_first_answer_token_ms": 12.0,
        "avg_prompt_tokens_per_second": 5.0, "avg_completion_tokens_per_second": 7.0,
        "avg_tokens_per_second": 6.0, "total_tokens": 30, "total_prompt_tokens": 10,
        "total_completion_tokens": 20,
        "total_answer_tokens": 20, "total_cache_tokens": 0, "total_tool_result_tokens": 0,
        "error_count": 0, "total_rag_context_tokens": 0,
        "total_reasoning_tokens": 0,
    }])])
    store = MetricsStore(FakeEngine(conn))
    store._tables_ensured = True
    await store.get_summary(24)
    sql = conn.executed[-1][0]
    # Weighted: SUM(tokens) / SUM(gated generation ms), llm_generation_ms
    # used only when sane (>=1ms, <= response_time*1.5), else response_time.
    assert "SUM(COALESCE(completion_tokens, answer_tokens)) FILTER (WHERE status_code < 400)" in sql
    assert "llm_generation_ms >= 1" in sql
    assert "COALESCE(response_time_ms, 1e15) * 1.5" in sql


# ---------------------------------------------------------------------------
# _compute_stored_tps — write-side sanitize for tokens_per_second
# ---------------------------------------------------------------------------

def test_compute_stored_tps_normal():
    from app.routers.chat import _compute_stored_tps
    assert _compute_stored_tps(100, 4000, 5000) == pytest.approx(25.0)


def test_compute_stored_tps_sub_ms_generation_falls_back():
    from app.routers.chat import _compute_stored_tps
    # 0.25 ms generation is corrupt — denominator must be response_time.
    assert _compute_stored_tps(100, 0.25, 5000) == pytest.approx(20.0)


def test_compute_stored_tps_oversized_generation_falls_back():
    from app.routers.chat import _compute_stored_tps
    # gen_ms > response_ms * 1.5 is impossible — degrade to response_time.
    assert _compute_stored_tps(100, 9000, 5000) == pytest.approx(20.0)


def test_compute_stored_tps_no_valid_timing_returns_none():
    from app.routers.chat import _compute_stored_tps
    assert _compute_stored_tps(100, 0.25, 0) is None
    assert _compute_stored_tps(100, None, None) is None
    assert _compute_stored_tps(0, 4000, 5000) is None


def test_compute_stored_tps_clamps_implausible_rate():
    from app.routers.chat import _compute_stored_tps
    # 128 tokens in 0.25ms response would be 512k t/s → reject.
    assert _compute_stored_tps(128, None, 0.25) is None


def test_metrics_summary_sql_weighted_tps():
    """avg_tokens_per_second is SUM(tokens)/SUM(valid ms), never AVG of rates."""
    conn = FakeConnection(results=[FakeResult([])])
    store = MetricsStore(FakeEngine(conn))
    store._tables_ensured = True
    import asyncio
    sql = asyncio.run(store.get_summary(24))
    q = conn.executed[0][0]
    assert "AVG(tokens_per_second)" not in q
    assert "SUM(COALESCE(completion_tokens, answer_tokens))" in q
