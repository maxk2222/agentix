"""Configuration objects used across the FastAPI and RAG layers."""

from __future__ import annotations

import logging
import os
from shutil import copy2
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

from pydantic import AliasChoices, AnyHttpUrl, Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigurationError

# Ensure a sensible default USER_AGENT early so HTTP libraries don't warn during imports
os.environ.setdefault("USER_AGENT", "Agentix-RAG/1.0")

REPO_DEFAULTS_ROOT = Path(__file__).resolve().parents[1] / "defaults"
PACKAGED_DEFAULTS_ROOT = Path("/defaults")


class AppConfig(BaseSettings):
    """Strongly-typed view over the environment variables that drive the service."""

    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    instructions_file: Path = Field(
        default=Path("data/SystemInstructions/instructions.yaml"), alias="INSTRUCTIONS_FILE"
    )
    knowledge_urls_raw: Optional[str] = Field(default=None, alias="KNOWLEDGE_URLS")
    llm_api_url: AnyHttpUrl = Field(default="http://127.0.0.1:5001", alias="LLM_API_URL")
    llm_api_key: Optional[str] = Field(default=None, alias="LLM_API_KEY")
    system_slow: bool = Field(default=True, alias="SYSTEM_SLOW")
    vector_search_k: int = Field(default=5, alias="VECTOR_SEARCH_K")
    vector_batch_size: int = Field(default=200, alias="VECTOR_BATCH_SIZE")
    vector_force_rebuild: bool = Field(default=False, alias="VECTOR_FORCE_REBUILD")
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")
    embeddings_api_url: Optional[AnyHttpUrl] = Field(default=None, alias="EMBEDDINGS_API_URL")
    embeddings_endpoint: Optional[str] = Field(default=None, alias="EMBEDDINGS_ENDPOINT")
    embeddings_api_key: Optional[str] = Field(default=None, alias="EMBEDDINGS_API_KEY")
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_reload: bool = Field(default=False, alias="API_RELOAD")
    api_server: str = Field(default="granian", alias="API_SERVER")
    api_key: Optional[str] = Field(default=None, alias="API_KEY")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_dir: Path = Field(default=Path("logs"), alias="LOG_DIR")
    log_file_max_bytes: int = Field(default=5_242_880, alias="LOG_FILE_MAX_BYTES")

    pg_host: str = Field(default="localhost", alias="PGHOST")
    pg_port: int = Field(default=5432, alias="PGPORT")
    pg_database: str = Field(default="agentix-db", alias="PGDATABASE")
    pg_user: str = Field(default="default", alias="PGUSER")
    pg_password: str = Field(default="mypassword", alias="PGPASSWORD")
    # Optional Redis for run-registry fanout, steering, leader locks and the
    # metadata cache. Unset → everything stays in-process (single node).
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")

    tesseract_cmd: Optional[Path] = Field(default=None, alias="TESSERACT_CMD")
    ocr_language: str = Field(default="deu", alias="OCR_LANG")
    # OCR/Vision — DB-only config (no env aliases)
    ocr_mode: str = Field(default="tesseract")
    ocr_llm_prompt: str = Field(default="Extract all text and relevant information from this image.")
    ocr_llm_concurrency: int = Field(default=4)
    pdf_image_extraction_enabled: bool = Field(default=False)
    pdf_image_extraction_mode: str = Field(default="llm")
    pptx_slide_analysis_enabled: bool = Field(default=False)
    pptx_slide_analysis_mode: str = Field(default="llm")
    # Chat image upload — DB-only config
    chat_image_upload_enabled: bool = Field(default=False)
    chat_image_max_size_bytes: int = Field(default=5_242_880)
    chat_image_max_count: int = Field(default=4)
    chat_image_allowed_types: str = Field(default="image/png,image/jpeg,image/webp,image/gif")
    # Chat file (document) upload — DB-only config
    chat_auto_title_enabled: bool = Field(default=True)
    chat_file_upload_enabled: bool = Field(default=False)
    chat_file_max_size_bytes: int = Field(default=52_428_800)
    chat_file_max_count: int = Field(default=5)
    chat_file_allowed_exts: str = Field(default=".docx,.xlsx,.pptx,.pdf,.odt,.csv,.txt,.md,.doc,.xls,.ppt,.png,.jpg,.jpeg,.webp,.gif,.bmp,.tiff")
    chat_file_max_extract_chars: int = Field(default=50_000)
    chat_file_max_total_chars: int = Field(default=120_000)
    chat_file_extract_timeout_seconds: float = Field(default=120.0)
    # Chat file lifecycle — storage sweeper (DB-only config)
    chat_files_orphan_ttl_days: int = Field(default=7)
    chat_files_retention_days: int = Field(default=0)

    auto_initialize: bool = Field(default=True, alias="AUTO_INITIALIZE")
    user_agent: str = Field(default="Agentix-RAG/1.0", alias="USER_AGENT")
    # LLM token-level watchdog: abort an LLM stream when no token arrives for
    # this long mid-flight. Applies to each model call inside an agent run.
    stream_idle_timeout_seconds: float = Field(
        default=300.0,
        validation_alias=AliasChoices("STREAM_IDLE_TIMEOUT_SECONDS", "STREAMING_TIMEOUT_SECONDS"),
    )
    # Run-level watchdog: abort a chat/agent run after this many seconds
    # without a *real* stream item (token, reasoning, tool or subagent event).
    # Synthetic tool keepalives do not count. No fixed total-run limit — a
    # working run may take as long as it needs. Should stay above the sandbox
    # budgets (install + exec) so a healthy code_execution fits one window.
    run_idle_timeout_seconds: float = Field(
        default=900.0,
        validation_alias=AliasChoices("RUN_IDLE_TIMEOUT_SECONDS"),
    )

    # --- Input validation / abuse prevention (user request payloads) ---
    # Hard limit for incoming HTTP request bodies (JSON) in bytes.
    # Set to 0 to disable the limit (not recommended).
    max_request_bytes: int = Field(default=1_048_576, alias="MAX_REQUEST_BYTES")
    # Maximum number of messages allowed in a single chat/completions request.
    max_messages: int = Field(default=100, alias="MAX_MESSAGES")
    # Maximum characters allowed per message.content.
    max_message_chars: int = Field(default=20_000, alias="MAX_MESSAGE_CHARS")
    # Maximum characters allowed across all messages in a request.
    max_total_chars: int = Field(default=120_000, alias="MAX_TOTAL_CHARS")
    # Require the final message to be a user message (OpenAI-style semantics).
    require_last_user: bool = Field(default=True, alias="REQUIRE_LAST_USER")
    # When false, reject non-string message content (lists/dicts/tool blocks).
    allow_non_string_content: bool = Field(default=False, alias="ALLOW_NON_STRING_CONTENT")
    # Control character policy for message content: allow|reject
    validation_control_chars_mode: str = Field(default="reject", alias="VALIDATION_CONTROL_CHARS_MODE")
    # When false, only logs validation issues but does not block (log-only mode).
    validation_enforce: bool = Field(default=True, alias="VALIDATION_ENFORCE")

    # Chat/runtime controls
    # Maximum number of prior messages (excluding the latest user message) forwarded to the LLM.
    # Set to 0 to disable truncation.
    max_history_messages: int = Field(default=50, alias="MAX_HISTORY_MESSAGES")
    rolling_history_enabled: bool = Field(default=True, alias="ROLLING_HISTORY_ENABLED")
    rolling_history_context_window: int = Field(default=32_768, alias="ROLLING_HISTORY_CONTEXT_WINDOW")
    rolling_history_max_output_tokens: int = Field(default=2_048, alias="ROLLING_HISTORY_MAX_OUTPUT_TOKENS")
    rolling_history_rag_reserved_tokens: int = Field(default=0, alias="ROLLING_HISTORY_RAG_RESERVED_TOKENS")
    rolling_history_min_recent_turns: int = Field(default=4, alias="ROLLING_HISTORY_MIN_RECENT_TURNS")
    rolling_history_safety_margin_ratio: float = Field(default=0.08, alias="ROLLING_HISTORY_SAFETY_MARGIN_RATIO")
    rolling_history_summary_timeout_seconds: float = Field(default=30.0, alias="ROLLING_HISTORY_SUMMARY_TIMEOUT_SECONDS")

    # Loader/runtime tuning
    loader_max_concurrency: int = Field(default=8, alias="LOADER_MAX_CONCURRENCY")
    chunk_size: int = Field(default=1500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # Retrieval tuning
    retrieval_hybrid_enabled: bool = Field(default=True, alias="RETRIEVAL_HYBRID_ENABLED")
    hybrid_vector_candidates: int = Field(default=20, alias="HYBRID_VECTOR_CANDIDATES")
    hybrid_keyword_candidates: int = Field(default=20, alias="HYBRID_KEYWORD_CANDIDATES")
    hybrid_rrf_k: int = Field(default=60, alias="HYBRID_RRF_K")
    hybrid_max_chunks_per_source: int = Field(default=3, alias="HYBRID_MAX_CHUNKS_PER_SOURCE")
    hybrid_query_expansion_enabled: bool = Field(default=True, alias="HYBRID_QUERY_EXPANSION_ENABLED")
    hybrid_max_expanded_queries: int = Field(default=2, alias="HYBRID_MAX_EXPANDED_QUERIES")
    hybrid_rerank_enabled: bool = Field(default=True, alias="HYBRID_RERANK_ENABLED")
    # Minimum cosine similarity for a chunk to be included in RAG context.
    # 0.0 disables the filter. A value like 0.45-0.55 can improve precision.
    hybrid_min_semantic_similarity: float = Field(default=0.4, alias="HYBRID_MIN_SEMANTIC_SIMILARITY")

    # Lightweight history-aware retrieval (no LLM call).
    # When enabled the retriever receives a query built from recent user messages.
    retrieval_history_enabled: bool = Field(default=True, alias="RETRIEVAL_HISTORY_ENABLED")
    retrieval_history_user_messages: int = Field(default=2, alias="RETRIEVAL_HISTORY_USER_MESSAGES")
    retrieval_history_max_chars: int = Field(default=800, alias="RETRIEVAL_HISTORY_MAX_CHARS")
    # Detect standalone definition/explanation queries and skip history mixing.
    retrieval_history_standalone_detection: bool = Field(default=True, alias="RETRIEVAL_HISTORY_STANDALONE_DETECTION")

    # Recency-weighted retrieval: run a primary search on the latest user
    # message and only backfill with the history-mixed query so fresh intent
    # dominates the retrieved documents.
    retrieval_recency_enabled: bool = Field(default=True, alias="RETRIEVAL_RECENCY_ENABLED")
    retrieval_recency_primary_share: float = Field(default=0.6, alias="RETRIEVAL_RECENCY_PRIMARY_SHARE")

    # Question contextualization (history-aware retrieval)
    # LangChain's history-aware retriever does an extra LLM call to rewrite the user question.
    # This improves follow-up questions but can add significant latency.
    contextualize_question_enabled: bool = Field(
        default=False, alias="CONTEXTUALIZE_QUESTION_ENABLED"
    )
    # Optional model override for contextualization (e.g. a smaller/faster model).
    contextualize_question_model: Optional[str] = Field(
        default=None, alias="CONTEXTUALIZE_QUESTION_MODEL"
    )
    # Contextualization should usually be deterministic.
    contextualize_question_temperature: float = Field(
        default=0.0, alias="CONTEXTUALIZE_QUESTION_TEMPERATURE"
    )
    # Many upstreams run slower when reasoning is enabled; disable it for the rewrite step.
    contextualize_question_disable_reasoning: bool = Field(
        default=True, alias="CONTEXTUALIZE_QUESTION_DISABLE_REASONING"
    )

    # Incremental indexing
    indexing_incremental_enabled: bool = Field(default=True, alias="INDEXING_INCREMENTAL_ENABLED")

    # API key enforcement toggle
    api_keys_enabled: bool = Field(default=True, alias="API_KEYS_ENABLED")

    # --- Database backup (encrypted pg_dump uploaded to WebDAV/Sciebo) ---
    # Symmetric key for backup archives (32 bytes, hex or base64). Required:
    # backups fail closed when unset. Never store this key in the database.
    backup_encryption_key: Optional[str] = Field(default=None, alias="BACKUP_ENCRYPTION_KEY")
    # Local scratch space for dumps/encryption (cleaned after every run).
    backup_tmp_dir: Path = Field(default=Path("data/backups/tmp"), alias="BACKUP_TMP_DIR")
    # Hard timeout for pg_dump / pg_restore subprocess calls (seconds).
    backup_dump_timeout: int = Field(default=1800, alias="BACKUP_DUMP_TIMEOUT")
    # Timeout for WebDAV upload/download requests (seconds).
    backup_upload_timeout: int = Field(default=600, alias="BACKUP_UPLOAD_TIMEOUT")
    # Upload retry attempts with exponential backoff.
    backup_upload_retries: int = Field(default=3, alias="BACKUP_UPLOAD_RETRIES")
    # Instance identifier used in backup filenames so multiple deployments can
    # share one remote directory without collisions.
    backup_instance: str = Field(default="", alias="BACKUP_INSTANCE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        populate_by_name=True,
    )

    @field_validator("data_dir", "instructions_file", "log_dir", "audio_tts_voices_dir", "audio_tts_kokoro_model_dir", "backup_tmp_dir", mode="before")
    @classmethod
    def _as_path(cls, value: str | Path) -> Path:
        """Normalize relative paths to absolute ones rooted at the current working dir."""
        path = Path(value)
        return path if path.is_absolute() else Path.cwd() / path

    @field_validator("tesseract_cmd", mode="before")
    @classmethod
    def _empty_string_to_none(cls, value: object) -> Optional[Path]:
        """Treat empty environment variables as unset.

        `.env` templates often contain `TESSERACT_CMD=` which would otherwise parse
        as `Path('')` (i.e. '.') and break pytesseract by trying to execute '.'.
        """
        if value is None:
            return None
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            return Path(stripped) if stripped else None
        return value  # let pydantic handle/raise

    @computed_field
    @property
    def knowledge_urls(self) -> List[str]:
        """Return the configured remote knowledge sources as a sanitized list."""
        if not self.knowledge_urls_raw:
            return []
        return [url.strip() for url in self.knowledge_urls_raw.split(",") if url.strip()]

    @property
    def llm_api_base(self) -> str:
        """Return the LLM API base URL without the ``/v1`` suffix or trailing slash."""
        return str(self.llm_api_url).rstrip("/")

    @property
    def embeddings_api_base(self) -> str:
        """Return the embeddings endpoint base, defaulting to the LLM base when unset."""
        if self.embeddings_api_url:
            return str(self.embeddings_api_url).rstrip("/")
        return self.llm_api_base

    def _default_bundle_root(self) -> Optional[Path]:
        for root in (REPO_DEFAULTS_ROOT, PACKAGED_DEFAULTS_ROOT):
            if (root / "SystemInstructions" / "instructions.yaml").exists():
                return root
        return None

    def bootstrap_default_runtime_data(self) -> bool:
        """Ensure the runtime ``data`` tree exists by copying defaults when needed.

        The startup check is intentionally simple: if both the data directory and the
        instructions file already exist, nothing is changed. Otherwise the complete
        defaults tree is copied into ``self.data_dir`` so the application can keep using
        the normal runtime locations afterwards.
        """
        if self.data_dir.exists() and self.instructions_file.exists():
            return False

        defaults_root = self._default_bundle_root()
        if defaults_root is None:
            raise ConfigurationError(f"Instructions file not found: {self.instructions_file}")

        self.data_dir.parent.mkdir(parents=True, exist_ok=True)
        self._copy_directory_contents(defaults_root, self.data_dir)
        return True

    def _copy_directory_contents(self, source_root: Path, target_root: Path) -> None:
        """Recursively copy missing files from ``source_root`` into ``target_root``."""
        target_root.mkdir(parents=True, exist_ok=True)
        for source_path in source_root.iterdir():
            target_path = target_root / source_path.name
            if source_path.is_dir():
                self._copy_directory_contents(source_path, target_path)
                continue
            if not target_path.exists():
                copy2(source_path, target_path)

    # Embeddings request tuning
    embeddings_request_timeout: int = Field(default=30, alias="EMBEDDINGS_REQUEST_TIMEOUT")
    embeddings_max_retries: int = Field(default=3, alias="EMBEDDINGS_MAX_RETRIES")
    embeddings_backoff_factor: float = Field(default=0.5, alias="EMBEDDINGS_BACKOFF_FACTOR")
    # Fallback safety net: truncate texts to this many characters before sending to the
    # embedding API. Should be comfortably larger than CHUNK_SIZE. Set to 0 to disable.
    embedding_max_chars: int = Field(default=4000, alias="EMBEDDING_MAX_CHARS")
    # Retriever invocation timeout (seconds) for per-collection async calls
    retriever_timeout_seconds: float = Field(default=10.0, alias="RETRIEVER_TIMEOUT_SECONDS")

    # Dashboard / Admin session settings
    dashboard_enabled: bool = Field(default=False, alias="DASHBOARD_ENABLED")
    dashboard_admin_username: str = Field(default="admin", alias="DASHBOARD_ADMIN_USERNAME")
    dashboard_admin_password: str = Field(default="", alias="DASHBOARD_ADMIN_PASSWORD")
    dashboard_session_secret: str = Field(default="", alias="DASHBOARD_SESSION_SECRET")
    dashboard_session_ttl_seconds: int = Field(default=3600, alias="DASHBOARD_SESSION_TTL_SECONDS")

    # Widget defaults endpoint CORS (comma-separated origins, "*" = allow all)
    widget_defaults_cors_origins: str = Field(default="*", alias="WIDGET_DEFAULTS_CORS_ORIGINS")

    # Audio / TTS / STT settings
    audio_stt_enabled: bool = Field(default=False, alias="AUDIO_STT_ENABLED")
    audio_tts_enabled: bool = Field(default=False, alias="AUDIO_TTS_ENABLED")
    audio_tts_voices_dir: Path = Field(default=Path("data/voices/tts_voices"), alias="AUDIO_TTS_VOICES_DIR")
    audio_tts_default_voice: Optional[str] = Field(default=None, alias="AUDIO_TTS_DEFAULT_VOICE")
    audio_tts_piper_binary: str = Field(default="piper", alias="AUDIO_TTS_PIPER_BINARY")
    audio_tts_streaming_enabled: bool = Field(default=True, alias="AUDIO_TTS_STREAMING_ENABLED")
    audio_tts_provider: str = Field(default="kokoro", alias="AUDIO_TTS_PROVIDER")
    audio_tts_kokoro_voice: str = Field(default="martin", alias="AUDIO_TTS_KOKORO_VOICE")
    audio_tts_kokoro_model_dir: Path = Field(default=Path("data/models/voices/kokoro_models"), alias="AUDIO_TTS_KOKORO_MODEL_DIR")
    audio_tts_language: str = Field(default="de", alias="AUDIO_TTS_LANGUAGE")
    audio_stt_model: str = Field(default="small", alias="AUDIO_STT_MODEL")
    audio_stt_language: str = Field(default="de", alias="AUDIO_STT_LANGUAGE")
    audio_stt_provider: str = Field(default="faster_whisper", alias="AUDIO_STT_PROVIDER")
    hf_cache_dir: Optional[str] = Field(default="data/models/HF_CACHE", alias="HF_CACHE_DIR")
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")

    # Emotion Detection
    emotion_detection_enabled: bool = Field(default=False, alias="EMOTION_DETECTION_ENABLED")
    emotion_webcam_enabled: bool = Field(default=False, alias="EMOTION_WEBCAM_ENABLED")
    emotion_prompt_steering_enabled: bool = Field(default=True, alias="EMOTION_PROMPT_STEERING_ENABLED")
    emotion_analytics_enabled: bool = Field(default=True, alias="EMOTION_ANALYTICS_ENABLED")
    emotion_min_confidence: float = Field(default=0.55, alias="EMOTION_MIN_CONFIDENCE")
    emotion_webcam_runtime: str = Field(default="onnxruntime-web", alias="EMOTION_WEBCAM_RUNTIME")
    emotion_webcam_execution_providers: str = Field(default="webgpu,wasm", alias="EMOTION_WEBCAM_EXECUTION_PROVIDERS")
    emotion_webcam_face_detector: str = Field(default="mediapipe", alias="EMOTION_WEBCAM_FACE_DETECTOR")
    emotion_webcam_face_model_url: str = Field(default="/models/emotion/face-detector.onnx", alias="EMOTION_WEBCAM_FACE_MODEL_URL")
    emotion_webcam_face_model_download_url: Optional[str] = Field(default="https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?download=", alias="EMOTION_WEBCAM_FACE_MODEL_DOWNLOAD_URL")
    emotion_webcam_expression_model_url: str = Field(default="/models/emotion/expression-affectnet.onnx", alias="EMOTION_WEBCAM_EXPRESSION_MODEL_URL")
    emotion_webcam_expression_model_download_url: Optional[str] = Field(default="https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx", alias="EMOTION_WEBCAM_EXPRESSION_MODEL_DOWNLOAD_URL")
    emotion_webcam_expression_model_type: str = Field(default="affectnet", alias="EMOTION_WEBCAM_EXPRESSION_MODEL_TYPE")
    emotion_webcam_expression_input_size: int = Field(default=224, alias="EMOTION_WEBCAM_EXPRESSION_INPUT_SIZE")
    emotion_webcam_snippet_window_ms: int = Field(default=3000, alias="EMOTION_WEBCAM_SNIPPET_WINDOW_MS")
    emotion_webcam_sample_fps: int = Field(default=3, alias="EMOTION_WEBCAM_SAMPLE_FPS")
    emotion_webcam_max_frames_per_snippet: int = Field(default=12, alias="EMOTION_WEBCAM_MAX_FRAMES_PER_SNIPPET")
    emotion_webcam_min_valid_face_frames: int = Field(default=3, alias="EMOTION_WEBCAM_MIN_VALID_FACE_FRAMES")
    emotion_webcam_frame_upload_enabled: bool = Field(default=False, alias="EMOTION_WEBCAM_FRAME_UPLOAD_ENABLED")
    emotion_audio_enabled: bool = Field(default=False, alias="EMOTION_AUDIO_ENABLED")
    emotion_audio_model_url: str = Field(default="/models/emotion/audio-emotion.onnx", alias="EMOTION_AUDIO_MODEL_URL")
    emotion_audio_model_download_url: Optional[str] = Field(default="https://huggingface.co/onnx-community/wav2vec2-emotion-recognition-ONNX/resolve/main/onnx/model_q4f16.onnx?download=true", alias="EMOTION_AUDIO_MODEL_DOWNLOAD_URL")

    # Text Emotion (German GoEmotions ONNX)
    emotion_text_model_enabled: bool = Field(default=True, alias="EMOTION_TEXT_MODEL_ENABLED")
    emotion_text_model_url: str = Field(default="/models/emotion/text-emotion.onnx", alias="EMOTION_TEXT_MODEL_URL")
    emotion_text_model_download_url: Optional[str] = Field(default="https://huggingface.co/tojohere/goemotions-de-xlm-roberta-base/resolve/main/goemotions.onnx", alias="EMOTION_TEXT_MODEL_DOWNLOAD_URL")
    emotion_text_tokenizer_url: str = Field(default="/models/emotion/text-emotion-tokenizer.json", alias="EMOTION_TEXT_TOKENIZER_URL")
    emotion_text_tokenizer_download_url: Optional[str] = Field(default="https://huggingface.co/tojohere/goemotions-de-xlm-roberta-base/resolve/main/tokenizer.json", alias="EMOTION_TEXT_TOKENIZER_DOWNLOAD_URL")

    # Emotion Fusion (configurable weights and per-source thresholds)
    emotion_fusion_source_weights: Optional[dict] = Field(default=None, alias="EMOTION_FUSION_SOURCE_WEIGHTS")
    emotion_fusion_min_confidence: Optional[dict] = Field(default=None, alias="EMOTION_FUSION_MIN_CONFIDENCE")

    @property
    def connection_string(self) -> str:
        """Assemble the SQLAlchemy connection URL for the PGVector database."""
        user = quote(self.pg_user, safe="")
        password = quote(self.pg_password, safe="")
        database = quote(self.pg_database, safe="")
        return (
            f"postgresql+psycopg://{user}:{password}"
            f"@{self.pg_host}:{self.pg_port}/{database}"
        )

    @property
    def knowledge_base_root(self) -> Path:
        """Return the absolute path to ``data/KnowledgeDatabase``."""
        return (self.data_dir / "KnowledgeDatabase").resolve()

    @property
    def tts_voices_dir(self) -> Path:
        """Return the absolute path to the TTS voices directory."""
        path = self.audio_tts_voices_dir
        if path.is_absolute():
            return path.resolve()
        return (self.data_dir / path).resolve()

    @property
    def normalized_log_level(self) -> str:
        return self.log_level.upper()

    @property
    def debug_logging_enabled(self) -> bool:
        level = logging.getLevelName(self.normalized_log_level)
        if isinstance(level, str):
            return False
        return level <= logging.DEBUG
