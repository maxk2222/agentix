"""Embedding utilities for OpenAI-compatible and Ollama-style endpoints.

These helpers wrap ``httpx`` plus LangChain's ``Embeddings`` interface, add resilience
features (endpoint fallback, structured error handling, retry wrappers), and expose async
APIs needed by the PGVector ingestion pipeline.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import random
from typing import Any, List, Optional
from urllib.parse import urlparse

import httpx
from langchain.embeddings.base import Embeddings
from app.logging_utils import colorize_by_status, colorize_process_time

logger = logging.getLogger(__name__)


class CustomOpenAICompatibleEmbeddings(Embeddings):
    """LangChain ``Embeddings`` implementation for OpenAI-compatible deployments.

    The class encapsulates httpx client management, handles Ollama-specific payloads,
    and can transparently fall back from ``/v1/embeddings`` to ``/api/embeddings`` when
    needed.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        model: str = "text-embedding-nomic-embed-text-v1.5",
        api_key: str = "none",
        timeout: int = 30,
        client: Optional[httpx.AsyncClient] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        max_chars: int = 0,
    ) -> None:
        if endpoint:
            self.endpoint = endpoint
        else:
            base = os.getenv("LLM_API_URL", "http://127.0.0.1:5001").rstrip("/")
            self.endpoint = f"{base}/v1/embeddings"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
            self._client = httpx.AsyncClient(timeout=self.timeout, limits=limits)
            self._owns_client = True
        self._fallback_endpoint = self._build_fallback_endpoint(self.endpoint)
        self._using_fallback = False
        self._is_ollama_endpoint = self._looks_like_ollama(self.endpoint)
        self.max_retries = int(max_retries)
        self.backoff_factor = float(backoff_factor)
        self.max_chars = int(max_chars)

    async def close(self) -> None:
        """Close the owned HTTP client when the embedding instance is disposed."""
        if self._owns_client and self._client:
            await self._client.aclose()

    @staticmethod
    def _build_fallback_endpoint(endpoint: str | None) -> str | None:
        """Translate ``/v1/embeddings`` URLs into legacy ``/api/embeddings`` targets."""
        if not endpoint:
            return None
        if "/v1/embeddings" in endpoint:
            base = endpoint.split("/v1/embeddings", 1)[0].rstrip("/")
            return f"{base}/api/embeddings"
        return None

    @staticmethod
    def _looks_like_ollama(endpoint: str | None) -> bool:
        """Return ``True`` when the endpoint resembles an Ollama embeddings route."""
        if not endpoint:
            return False
        normalized = endpoint.rstrip("/")
        return normalized.endswith("/api/embeddings") or normalized.endswith(":11434/embeddings")

    def _try_fallback_endpoint(self) -> bool:
        """Switch to the fallback endpoint once per instance when the primary fails."""
        if self._fallback_endpoint and not self._using_fallback:
            logger.info("Switching embedding endpoint to fallback %s", self._fallback_endpoint)
            self.endpoint = self._fallback_endpoint
            self._using_fallback = True
            self._is_ollama_endpoint = self._looks_like_ollama(self.endpoint)
            return True
        return False

    async def _get_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Call the configured endpoint and return embeddings for each supplied text."""
        if self.max_chars > 0:
            truncated = []
            for t in texts:
                if len(t) > self.max_chars:
                    logger.warning(
                        "Truncating text from %s to %s chars before embedding",
                        len(t), self.max_chars,
                    )
                    truncated.append(t[:self.max_chars])
                else:
                    truncated.append(t)
            texts = truncated

        headers = {"Content-Type": "application/json"}
        if self.api_key != "none":
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            if self._is_ollama_endpoint:
                return await self._get_ollama_embeddings(texts, headers)

            # Non-ollama: attempt a batched embeddings request with retries
            start_time = time.perf_counter()
            response = await self._request_with_retries(
                method="post",
                url=self.endpoint,
                json={"input": texts, "model": self.model},
                headers=headers,
            )
            data = response.json()
            raw_embeddings = [item.get("embedding") for item in data.get("data", [])]
            if len(raw_embeddings) != len(texts):
                raise ValueError("Embedding response missing expected vectors")

            embeddings: List[List[float]] = []
            for embedding in raw_embeddings:
                if not embedding:
                    raise ValueError("Embedding response contained empty vector")
                embeddings.append(embedding)

            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "HTTP Request: POST %s \"HTTP/1.1 %s\" | Process Time: %s | Texts: %d",
                self.endpoint,
                colorize_by_status(response.status_code),
                colorize_process_time(duration_ms),
                len(texts),
                extra={"process_time": duration_ms},
            )
            return embeddings
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                if exc.response.status_code in (301, 302, 303, 307, 308):
                    location = exc.response.headers.get("Location")
                    if location:
                        logger.warning("Embedding endpoint redirected to %s", location)
                if exc.response.status_code == 404 and self._try_fallback_endpoint():
                    logger.warning(
                        "Embedding endpoint %s returned 404. Retrying with fallback %s.",
                        exc.request.url,
                        self.endpoint,
                    )
                    return await self._get_embeddings_async(texts)

            # Redact sensitive details from logs; only surface endpoint host and status
            parsed = urlparse(self.endpoint or "")
            detail_parts: List[str] = [f"type={type(exc).__name__}", f"host={parsed.netloc}"]
            with contextlib.suppress(Exception):
                if getattr(exc, "request", None) is not None and getattr(exc.request, "url", None) is not None:
                    detail_parts.append(f"path={urlparse(str(exc.request.url)).path}")
            if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                with contextlib.suppress(Exception):
                    detail_parts.append(f"status={exc.response.status_code}")
                with contextlib.suppress(Exception):
                    body = exc.response.text
                    if body:
                        snippet = body.strip().replace("\n", " ")
                        detail_parts.append(f"body_snippet={snippet[:200]}")

            logger.error("Embedding request failed (%s)", " | ".join(detail_parts))
            raise RuntimeError("Embedding request failed") from exc

    async def _request_with_retries(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Perform an HTTP request with retries and exponential backoff.

        Retries on network errors and 5xx responses. Will not log sensitive headers.
        """
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt < self.max_retries:
            attempt += 1
            try:
                resp = await self._client.request(method, url, timeout=self.timeout, **kwargs)
                # Raise for HTTP errors; allow handlers to inspect status
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                # Retry on server errors (5xx)
                if status and 500 <= status < 600 and attempt < self.max_retries:
                    last_exc = exc
                else:
                    raise
            except httpx.RequestError as exc:
                last_exc = exc

            # Backoff with jitter
            backoff = self.backoff_factor * (2 ** (attempt - 1))
            jitter = random.uniform(0, backoff * 0.1)
            await asyncio.sleep(backoff + jitter)

        if last_exc:
            raise last_exc

    async def _get_ollama_embeddings(self, texts: List[str], headers: dict) -> List[List[float]]:
        """Request Ollama embeddings with limited concurrency.

        Ollama expects single-prompt payloads, so we send individual requests
        but do this concurrently with a small semaphore to improve throughput.
        """
        sem = asyncio.Semaphore(5)
        results: List[Optional[List[float]]] = [None] * len(texts)

        async def _call(idx: int, txt: str):
            payload = {"model": self.model, "prompt": txt}
            start_time = time.perf_counter()
            async with sem:
                resp = await self._request_with_retries("post", self.endpoint, json=payload, headers=headers)
            body = resp.json()
            embedding = body.get("embedding")
            if not embedding:
                raise ValueError("Ollama embeddings response missing 'embedding'")
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "HTTP Request: POST %s %s | Process Time: %s",
                urlparse(self.endpoint).netloc,
                colorize_by_status(resp.status_code),
                colorize_process_time(duration_ms),
                extra={"process_time": duration_ms},
            )
            results[idx] = embedding

        tasks = [_call(i, t) for i, t in enumerate(texts)]
        await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Synchronous wrapper around :meth:`aembed_documents` for LangChain compatibility."""
        return self._run_sync(self.aembed_documents(texts))

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string synchronously."""
        return self._run_sync(self.aembed_query(text))

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async interface required by LangChain vector stores."""
        return await self._get_embeddings_async(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Return just the first embedding from the async bulk helper."""
        embeddings = await self._get_embeddings_async([text])
        return embeddings[0] if embeddings else []

    @staticmethod
    def _run_sync(coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # Inside an active event loop: schedule in a thread pool to avoid blocking
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()


class SlowSystemEmbeddingWrapper:
    """Retry-heavy wrapper for embedding providers with strict rate/latency limits.

    Processes documents in small concurrent batches rather than one-by-one to
    improve throughput while still respecting downstream rate limits.
    """

    def __init__(
        self,
        base_embedding: CustomOpenAICompatibleEmbeddings,
        max_retries: int = 5,
        delay: float = 1.5,
        batch_size: int = 5,
        max_concurrency: int = 3,
        max_chars: int = 0,
    ) -> None:
        self.base_embedding = base_embedding
        self.max_retries = max_retries
        self.delay = delay
        self.batch_size = max(1, batch_size)
        self._sem = asyncio.Semaphore(max(1, max_concurrency))
        if max_chars and not getattr(base_embedding, "max_chars", 0):
            base_embedding.max_chars = int(max_chars)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents in concurrent batches with per-batch retry logic."""
        if not texts:
            return []

        total = len(texts)
        results: List[List[float]] = [[] for _ in range(total)]

        async def _embed_batch(start: int, batch: List[str]) -> None:
            embeddings = await self._retry(
                lambda: self.base_embedding.aembed_documents(batch)
            )
            if not embeddings or len(embeddings) != len(batch):
                raise ValueError("Received empty or mismatched embedding batch after retries")
            for idx, emb in enumerate(embeddings):
                results[start + idx] = emb

        tasks: List[asyncio.Task] = []
        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            tasks.append(asyncio.create_task(_embed_batch(i, batch)))

        if tasks:
            await asyncio.gather(*tasks)

        logger.debug("Embedding completed for %s texts in %s batches", total, len(tasks))
        return results

    async def aembed_query(self, text: str) -> List[float]:
        """Retry the wrapped embedder for single queries and surface failures early."""
        embedding = await self._retry(lambda: self.base_embedding.aembed_query(text))
        if not embedding:
            raise ValueError("Query embedding empty after retries")
        return embedding

    async def _retry(self, func):
        """Run ``func`` with incremental delays, raising after ``max_retries`` attempts."""
        async with self._sem:
            for attempt in range(1, self.max_retries + 1):
                try:
                    return await func()
                except Exception as exc:
                    logger.warning("Embedding attempt %s/%s failed: %s", attempt, self.max_retries, exc)
                    await asyncio.sleep(self.delay * attempt)
        raise ValueError(f"Embedding failed after {self.max_retries} attempts")

