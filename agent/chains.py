"""LangChain pipeline builders used by the RAG application."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain.chains import create_history_aware_retriever
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from agent.custom_llm import ChatOpenAIWithReasoning
from agent.agent_llm import ChatOpenAIAgentLLM

from app.chat_utils import extract_answer_text, extract_reasoning_text, render_system_instructions
from app.reasoning_params import (
    deep_merge_extra_body,
    match_family,
    probe_overrides_family,
    rule_from_capabilities,
    translate_to_wire,
)
from app.history_budget import count_text_tokens, _truncate_summary
from agent.config import AppConfig
from agent.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ChainBuilder:
    """Constructs streaming, conversational, and LLM-only LangChain chains."""

    def __init__(
        self,
        config: AppConfig,
        instructions: Dict[str, Any],
        provider_store: Any = None,
    ) -> None:
        self.config = config
        self.instructions = instructions
        self._provider_store = provider_store
        self._llm_cache: Dict[Tuple[Any, ...], ChatOpenAIWithReasoning] = {}
        self._agent_llm_cache: Dict[Tuple[Any, ...], ChatOpenAIAgentLLM] = {}
        self._cache_lock = asyncio.Lock()
        self._provider_model_cache: Dict[str, Tuple[str, str, Optional[str]]] = {}
        """model_name -> (provider_id, base_url, api_key)"""

    def set_provider_store(self, store: Any) -> None:
        self._provider_store = store

    def set_provider_model_cache(self, cache: Dict[str, Tuple[str, str, Optional[str]]]) -> None:
        self._provider_model_cache = cache

    async def clear_cache(self) -> None:
        async with self._cache_lock:
            self._llm_cache.clear()
            self._agent_llm_cache.clear()

    async def build_chains(
        self,
        model_name: str,
        *,
        use_contextualize: bool,
        retriever: Optional[BaseRetriever],
        retrieval_degraded: bool,
        rag_context: str = "",
        emotion_hint: Optional[str] = None,
    ) -> Tuple[Optional[Any], Any, Any]:
        """Return (streaming_chain, conversational_chain, llm_only_safe_chain)."""
        assistant_data = self._get_assistant_entry(model_name)
        system_instructions = assistant_data.get("system_instructions", "")
        system_instructions = render_system_instructions(system_instructions, rag_context=rag_context)
        if rag_context and "{{context}}" not in assistant_data.get("system_instructions", ""):
            system_instructions = system_instructions.rstrip() + "\n\nRelevant context:\n" + rag_context
        if emotion_hint:
            system_instructions = system_instructions.rstrip() + "\n\n" + emotion_hint
        # MessagesPlaceholder("input_messages") takes the real HumanMessage —
        # a ("human","{input}") string template would stringify multimodal
        # content lists into a Python repr and silently destroy image parts.
        # SystemMessage (not a ("system", str) tuple) carries the instructions:
        # string tuples become f-string templates, so "{" / "}" in the prompt
        # (JSON examples, markdown) would be parsed as variables and crash
        # format_messages with KeyError.
        qa_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_instructions),
            MessagesPlaceholder("chat_history"),
            MessagesPlaceholder("input_messages"),
        ])
        # RAG prompt: context is injected into the system message so strict
        # chat templates (Qwen, Llama, Mistral via LM Studio) never see a
        # SystemMessage after HumanMessage.
        rag_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_text}"),
            MessagesPlaceholder("chat_history"),
            MessagesPlaceholder("input_messages"),
        ])

        streaming_llm = await self._get_llm_for_assistant(model_name, streaming=True)
        non_streaming_llm = await self._get_llm_for_assistant(model_name, streaming=False)

        llm_only_chain = qa_prompt | non_streaming_llm
        llm_only_safe_chain = llm_only_chain | RunnableLambda(self._ensure_answer_output)

        streaming_chain = None
        conversational_chain = llm_only_safe_chain

        if retriever is not None and not retrieval_degraded:
            active_retriever = retriever
            if use_contextualize:
                assistant_model, _, assistant_reasoning = self._resolve_model_settings(model_name)
                rewrite_model = (self.config.contextualize_question_model or "").strip() or assistant_model
                rewrite_reasoning = assistant_reasoning
                if self.config.contextualize_question_disable_reasoning:
                    rewrite_reasoning = None

                context_llm = await self._get_llm_direct(
                    model=rewrite_model,
                    temperature=float(self.config.contextualize_question_temperature),
                    streaming=False,
                    reasoning=rewrite_reasoning,
                )
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Formuliere die Frage so um, dass sie auch ohne Kontext verständlich ist."),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                active_retriever = create_history_aware_retriever(
                    llm=context_llm,
                    retriever=retriever,
                    prompt=contextualize_q_prompt,
                )

            def _inject_rag_context(payload):
                docs = payload.get("context", [])
                formatted = "\n\n".join(doc.page_content for doc in docs)
                # Safety: truncate inline-retrieved context if it exceeds a
                # generous token limit.  LLM-based summarization for the
                # pre-computed rag_context already happens upstream in
                # RAGApplication.run/run_streaming.
                assistant_model = self._resolve_model_settings(model_name)[0]
                if formatted and count_text_tokens(formatted, assistant_model) > 4096:
                    formatted = _truncate_summary(formatted, 4096, assistant_model)
                sys_text = system_instructions
                if formatted:
                    sys_text += "\n\nRelevant context:\n" + formatted
                return {
                    "system_text": sys_text,
                    "chat_history": payload.get("chat_history", []),
                    "input_messages": payload.get("input_messages") or [],
                }

            # Defensive: ensure context key exists before formatting
            question_answer_chain = RunnableLambda(
                lambda payload: (
                    (lambda p: (p.setdefault("context", []), p)[1])(dict(payload))
                    if isinstance(payload, dict)
                    else {"context": []}
                )
            ) | RunnableLambda(
                lambda payload: (
                    logger.debug(
                        "RAG context (top=%d): %s",
                        len(payload.get("context") or []),
                        [
                            {
                                "source": (doc.metadata or {}).get("source", ""),
                                "page": (doc.metadata or {}).get("page", (doc.metadata or {}).get("page_number", "")),
                                "chunk_index": (doc.metadata or {}).get("chunk_index", ""),
                                "doc_id": (doc.metadata or {}).get("doc_id", ""),
                                "preview": (getattr(doc, "page_content", "") or "")[:300],
                            }
                            for doc in (payload.get("context") or [])[:10]
                        ],
                    ),
                    payload,
                )[1]
            ) | RunnableLambda(_inject_rag_context) | rag_qa_prompt | streaming_llm

            async def _retrieve_docs(payload):
                if use_contextualize:
                    # The history-aware retriever expects {"input","chat_history"}
                    # and rewrites the latest message into a standalone query —
                    # feeding it a concatenated history blob would crash on
                    # x.get("chat_history") and double-mix history anyway.
                    docs = await active_retriever.ainvoke({
                        "input": payload.get("retrieval_query_primary") or payload["input"],
                        "chat_history": payload.get("chat_history") or [],
                    })
                else:
                    docs = await active_retriever.ainvoke({
                        "primary": payload.get("retrieval_query_primary") or payload["input"],
                        "contextual": payload.get("retrieval_query") or payload["input"],
                    })
                return {**dict(payload), "context": docs}

            rag_chain = RunnableLambda(_retrieve_docs) | question_answer_chain
            streaming_chain = rag_chain
            conversational_chain = rag_chain | RunnableLambda(self._ensure_answer_output)

        return streaming_chain, conversational_chain, llm_only_safe_chain

    async def get_reasoning_text(self, model_name: str) -> str:
        """Return accumulated reasoning text from the streaming LLM and clear it."""
        llm = await self._get_llm_for_assistant(model_name, streaming=True)
        return llm.get_and_clear_reasoning_text()

    async def peek_reasoning_text(self, model_name: str) -> str:
        """Return accumulated reasoning text without clearing it."""
        llm = await self._get_llm_for_assistant(model_name, streaming=True)
        return llm.peek_reasoning_text()

    async def get_metadata_llm(self, model_name: str | None = None, purpose: str | None = None) -> ChatOpenAIWithReasoning:
        """Return a non-streaming LLM for metadata helpers (title/tags)."""
        target = model_name
        if target is None:
            default_assistant = (
                "CyberSecurityAssistant"
                if "CyberSecurityAssistant" in self.instructions
                else next(iter(self.instructions.keys()))
            )
            if purpose == "title":
                target = "title_generator" if "title_generator" in self.instructions else default_assistant
            elif purpose == "tags":
                target = "tag_extractor" if "tag_extractor" in self.instructions else default_assistant
            elif purpose == "summary":
                target = "summary_generator" if "summary_generator" in self.instructions else default_assistant
            elif purpose == "observer":
                target = "observer" if "observer" in self.instructions else default_assistant
            else:
                target = default_assistant
        return await self._get_llm_for_assistant(target, streaming=False)

    @staticmethod
    def _merge_extra_body(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Combine ``extra_body`` JSON with the explicit ``enable_thinking`` flag.

        ``enable_thinking`` maps to ``chat_template_kwargs.enable_thinking``
        (vLLM/Qwen-style thinking switch) and wins over the same key written
        manually inside ``extra_body``. Returns ``None`` when nothing is set.
        """
        raw_extra_body = entry.get("extra_body")
        extra_body = copy.deepcopy(raw_extra_body) if isinstance(raw_extra_body, dict) else {}
        enable_thinking = entry.get("enable_thinking")
        if enable_thinking is not None:
            chat_template_kwargs = extra_body.get("chat_template_kwargs")
            if not isinstance(chat_template_kwargs, dict):
                chat_template_kwargs = {}
                extra_body["chat_template_kwargs"] = chat_template_kwargs
            chat_template_kwargs["enable_thinking"] = enable_thinking
        return extra_body or None

    async def _get_llm_for_assistant(self, assistant_name: str, *, streaming: bool) -> ChatOpenAIWithReasoning:
        model_name, temperature, reasoning = self._resolve_model_settings(assistant_name)
        entry = self._get_assistant_entry(assistant_name)
        provider_id = entry.get("provider_id")
        return await self._get_llm_direct(
            model=model_name,
            temperature=temperature,
            streaming=streaming,
            reasoning=reasoning,
            provider_id=provider_id,
            extra_body=self._merge_extra_body(entry),
            extra_parameters=entry.get("extra_parameters"),
        )

    async def _resolve_provider_endpoint(
        self, model_name: str, provider_id: Optional[str], *, label: str = ""
    ) -> Tuple[str, str]:
        """Resolve ``(base_url, api_key)`` for a model via the provider store.

        Lookup order: explicit ``provider_id`` → per-model provider cache →
        default provider → ``config.llm_api_base``/``llm_api_key`` fallback.
        """
        llm_base_url = f"{self.config.llm_api_base}/v1"
        upstream_key = (self.config.llm_api_key or "").strip() or "none"
        if self._provider_store is None:
            return llm_base_url, upstream_key
        provider_found = False
        tag = f"{label} " if label else ""
        if provider_id:
            try:
                provider = await self._provider_store.get_provider(provider_id)
                if provider:
                    llm_base_url = f"{provider.base_url}/v1"
                    api_key = await self._provider_store.get_api_key(provider.id)
                    if api_key:
                        upstream_key = api_key
                    provider_found = True
                    logger.debug("Routing %smodel '%s' to explicit provider '%s'", tag, model_name, provider_id)
            except Exception:
                logger.debug("Failed to look up explicit provider '%s' for %smodel '%s'", provider_id, tag, model_name, exc_info=True)
        if not provider_found:
            cached = self._provider_model_cache.get(model_name)
            if cached:
                cached_provider_id, base_url, api_key = cached
                llm_base_url = f"{base_url}/v1"
                upstream_key = api_key or upstream_key
                provider_found = True
                logger.debug("Routing %smodel '%s' to cached provider '%s'", tag, model_name, cached_provider_id)
        if not provider_found:
            try:
                default_provider = await self._provider_store.get_default_provider()
                if default_provider:
                    llm_base_url = f"{default_provider.base_url}/v1"
                    api_key = await self._provider_store.get_api_key(default_provider.id)
                    if api_key:
                        upstream_key = api_key
                    logger.debug("Routing %smodel '%s' to default provider '%s'", tag, model_name, default_provider.id)
            except Exception:
                logger.debug("Failed to look up default provider for %smodel '%s'", tag, model_name, exc_info=True)
        return llm_base_url, upstream_key

    async def _resolve_reasoning_rule(
        self, model_name: str, base_url: str, provider_id: Optional[str]
    ) -> Dict[str, Any]:
        """Pick the reasoning wire rule — prefer live-probed capabilities.

        Candidate providers mirror ``_resolve_provider_endpoint``'s order:
        explicit ``provider_id`` → per-model provider cache → default
        provider. A cached probe result (``provider_model_reasoning``)
        overrides regex family matching for families the probe can describe
        (generic, thinking toggle, effort) — richer families (claude,
        gemini, ``thinking`` kwarg, openai object) stay authoritative.
        """
        if self._provider_store is not None:
            candidates: List[str] = [provider_id] if provider_id else []
            cached = self._provider_model_cache.get(model_name)
            if cached and cached[0] and cached[0] not in candidates:
                candidates.append(cached[0])
            for pid in candidates:
                rule = await self._caps_rule(pid, model_name, base_url)
                if rule is not None:
                    return rule
            try:
                default_provider = await self._provider_store.get_default_provider()
            except Exception:
                default_provider = None
            if default_provider and default_provider.id not in candidates:
                rule = await self._caps_rule(default_provider.id, model_name, base_url)
                if rule is not None:
                    return rule
        return match_family(model_name, base_url)

    async def _caps_rule(
        self, provider_id: str, model_name: str, base_url: str
    ) -> Optional[Dict[str, Any]]:
        """Rule built from a cached probe for ``(provider, model)`` — or None.

        Returns a rule only when a persisted probe result exists AND the
        family that name matching produces is one the probe dimensions may
        supersede. An error-only row (``capabilities`` NULL) is skipped.
        """
        try:
            status = await self._provider_store.get_model_reasoning_status(provider_id, model_name)
        except Exception:
            return None
        if not status or not status.get("capabilities"):
            return None
        family_rule = match_family(model_name, base_url)
        if not probe_overrides_family(family_rule):
            return family_rule
        return rule_from_capabilities(status["capabilities"])

    async def _get_agent_llm(self, assistant_name: str, max_tokens: Optional[int] = None) -> ChatOpenAIAgentLLM:
        """Return an LLM suitable for the agent runner.

        Uses the same model / temperature / provider as the assistant. The
        ``reasoning`` configuration is translated model-aware the same way as
        the chat path so e.g. ``enabled: false`` also silences Qwen-style
        thinking inside the agent loop.
        """
        model_name, temperature, reasoning = self._resolve_model_settings(assistant_name)
        entry = self._get_assistant_entry(assistant_name)
        provider_id = entry.get("provider_id")
        extra_body = self._merge_extra_body(entry)
        extra_parameters = entry.get("extra_parameters")
        cache_key = (
            model_name,
            float(temperature),
            provider_id,
            max_tokens,
            self._freeze_json(extra_body),
            self._freeze_json(extra_parameters),
            self._freeze_reasoning(reasoning),
        )
        async with self._cache_lock:
            if cache_key not in self._agent_llm_cache:
                llm_base_url, upstream_key = await self._resolve_provider_endpoint(model_name, provider_id, label="agent")
                if reasoning:
                    rule = await self._resolve_reasoning_rule(model_name, llm_base_url, provider_id)
                    wire = translate_to_wire(reasoning, rule)
                    extra_body = deep_merge_extra_body(extra_body, wire)
                client_kwargs: Dict[str, Any] = {
                    "base_url": llm_base_url,
                    "openai_api_key": upstream_key,
                    "model": model_name,
                    "temperature": float(temperature),
                    "streaming": False,
                }
                if max_tokens is not None:
                    client_kwargs["max_tokens"] = max_tokens
                if extra_body:
                    client_kwargs["extra_body"] = extra_body
                if extra_parameters:
                    client_kwargs["model_kwargs"] = dict(extra_parameters)
                self._agent_llm_cache[cache_key] = ChatOpenAIAgentLLM(**client_kwargs)
            return self._agent_llm_cache[cache_key]

    async def _get_llm_direct(
        self,
        *,
        model: str,
        temperature: float,
        streaming: bool,
        reasoning: Optional[Dict[str, Any]],
        provider_id: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        extra_parameters: Optional[Dict[str, Any]] = None,
    ) -> ChatOpenAIWithReasoning:
        stream_options = {"include_usage": True} if streaming else None
        cache_key = (
            model,
            float(temperature),
            bool(streaming),
            self._freeze_reasoning(reasoning),
            provider_id,
            tuple(sorted(stream_options.items())) if stream_options else None,
            self._freeze_json(extra_body),
            self._freeze_json(extra_parameters),
        )
        async with self._cache_lock:
            if cache_key not in self._llm_cache:
                llm_base_url, upstream_key = await self._resolve_provider_endpoint(model, provider_id)
                client_kwargs: Dict[str, Any] = {
                    "base_url": llm_base_url,
                    "openai_api_key": upstream_key,
                    "model": model,
                    "temperature": float(temperature),
                    "streaming": bool(streaming),
                }
                model_kwargs: Dict[str, Any] = dict(extra_parameters or {})
                if streaming:
                    model_kwargs["stream_options"] = stream_options
                if model_kwargs:
                    client_kwargs["model_kwargs"] = model_kwargs
                merged_extra_body = dict(extra_body) if extra_body else {}
                if reasoning:
                    rule = await self._resolve_reasoning_rule(model, llm_base_url, provider_id)
                    wire = translate_to_wire(reasoning, rule)
                    merged_extra_body = deep_merge_extra_body(merged_extra_body, wire)
                if merged_extra_body:
                    client_kwargs["extra_body"] = merged_extra_body
                self._llm_cache[cache_key] = ChatOpenAIWithReasoning(**client_kwargs)
            return self._llm_cache[cache_key]

    def _resolve_model_settings(self, assistant_name: str) -> Tuple[str, float, Optional[Dict[str, Any]]]:
        entry = self._get_assistant_entry(assistant_name)
        model_name = entry.get("model")
        if not model_name:
            raise ConfigurationError(
                f"Assistant '{assistant_name}' does not define a model in instructions"
            )
        temperature = float(entry.get("temperature", 0.0) or 0.0)
        reasoning = entry.get("reasoning") or None
        return model_name, temperature, reasoning

    def _get_assistant_entry(self, assistant_name: str) -> Dict[str, Any]:
        entry = self.instructions.get(assistant_name)
        if entry is None:
            raise ConfigurationError(f"Assistant '{assistant_name}' not found in instructions")
        return entry

    @staticmethod
    def _ensure_answer_output(output: Any) -> Dict[str, Any]:
        """Normalize LangChain outputs so the ``answer`` key always contains plain text.

        Also preserves ``reasoning`` if present in the output.
        """
        if isinstance(output, dict):
            payload = dict(output)
            text = extract_answer_text(payload.get("answer") or payload)
            if text:
                payload["answer"] = text
            reasoning = extract_reasoning_text(payload)
            if reasoning:
                payload["reasoning"] = reasoning
            return payload
        text = extract_answer_text(output)
        reasoning = extract_reasoning_text(output)
        result: Dict[str, Any] = {"answer": text}
        if reasoning:
            result["reasoning"] = reasoning
        return result

    @staticmethod
    def _freeze_json(value: Optional[Dict[str, Any]]) -> Optional[str]:
        if not value:
            return None
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _freeze_reasoning(reasoning: Optional[Dict[str, Any]]) -> Optional[str]:
        if not reasoning:
            return None
        try:
            return json.dumps(reasoning, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(reasoning)
