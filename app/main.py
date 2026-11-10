"""FastAPI wiring for the cybersecurity RAG assistant.

The module centralizes environment loading, instantiates :class:`~agent.application.RAGApplication`
exactly once per process, and attaches middleware plus routers required for the
OpenAI-compatible surface.  Keeping the bootstrap logic here ensures ``python -m app.server``
and ``uvicorn app.main:app`` share identical configuration, logging, and lifecycle
behaviour.
"""

# --- Standard Library Imports ---
import asyncio
import os
import subprocess
import sys
import time
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager


if sys.version_info >= (3, 14):
    raise RuntimeError(
        "Python 3.14+ is not supported by the current LangChain/Pydantic stack. "
        "Use Python 3.11.x (recommended) or a validated 3.12/3.13 runtime."
    )

# --- Third-party Library Imports ---
import nest_asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

# --- Local Application Imports ---
from agent.application import RAGApplication
from agent.config import AppConfig
from app.logging_config import configure_logging
from app.logging_utils import colorize_process_time
from app.middleware import APIKeyMiddleware, LoggingMiddleware, SessionRefreshMiddleware
from app.metrics import BlockedRequestRecord, BlockedRequestStore, MetricsStore, VectorstoreMetricsStore
from app.orjson_response import ORJSONResponse
from app.runtime_config import RuntimeConfigStore
from app.llm_providers import LLMProviderStore
from app.api_keys import AppApiKeyStore
from app.dashboard_users import DashboardUserStore
from app.chat_users import ChatUserStore
from app.chat_groups_store import ChatGroupsStore
from app.chat_conversations_store import ChatConversationsStore
from app.chat_role_assistants_store import ChatRoleAssistantsStore
from app.tools_store import ToolsStore
from app.triggers_store import TriggersStore
from app.triggers_scheduler import TriggersScheduler
from app.messages import MessageStore
from app.agent_chains_store import AgentChainsStore
from app.backup_store import BackupConfigStore, BackupJobStore
from app.backup_service import BackupService
from app.backup_scheduler import BackupScheduler
from app.security_middleware import RequestSizeLimitMiddleware
from app.routers import chat, admin, tools, triggers, agent_chains, agent_workspace, audio, admin_audio, admin_emotion, emotions, widget, public_chat, public_chat_auth, backup, branding

load_dotenv()
try:
    nest_asyncio.apply()
except ValueError as exc:
    logging.getLogger(__name__).warning("Skipping nest_asyncio patch: %s", exc)

app_config = AppConfig()
app_config.bootstrap_default_runtime_data()
configure_logging(app_config)


async def _init_core(app: FastAPI, config: AppConfig) -> RAGApplication:
    """Fast core initialization: engine, stores, tables, runtime config overlay.
    Returns the RAGApplication instance WITHOUT calling initialize()."""
    from agent.utils.pg_helpers import _to_async_connection_string
    from sqlalchemy.ext.asyncio import create_async_engine
    async_conn = _to_async_connection_string(config.connection_string)
    engine = create_async_engine(async_conn, pool_size=10, max_overflow=20, pool_pre_ping=True)

    blocked_store = BlockedRequestStore(engine)
    metrics_store = MetricsStore(engine)
    vectorstore_metrics_store = VectorstoreMetricsStore(engine)
    runtime_config_store = RuntimeConfigStore(engine)
    provider_store = LLMProviderStore(engine)
    api_key_store = AppApiKeyStore(engine)
    user_store = DashboardUserStore(engine)
    chat_user_store = ChatUserStore(engine)
    chat_groups_store = ChatGroupsStore(engine)
    chat_conversations_store = ChatConversationsStore(engine)
    chat_role_store = ChatRoleAssistantsStore(engine)
    tools_store = ToolsStore(engine)
    triggers_store = TriggersStore(engine)
    message_store = MessageStore(engine)
    agent_chains_store = AgentChainsStore(engine)
    backup_config_store = BackupConfigStore(engine)
    backup_job_store = BackupJobStore(engine)
    backup_service = BackupService(
        engine, config, backup_config_store, backup_job_store, app_state=app.state
    )
    await blocked_store.ensure_tables()
    await metrics_store.ensure_tables()
    await vectorstore_metrics_store.ensure_tables()
    await runtime_config_store.ensure_tables()
    await provider_store.ensure_tables()
    await api_key_store.ensure_tables()
    await user_store.ensure_tables()
    await chat_user_store.ensure_tables()
    await chat_groups_store.ensure_tables()
    await chat_conversations_store.ensure_tables()
    await chat_role_store.ensure_tables()
    await tools_store.ensure_tables()
    await triggers_store.ensure_tables()
    await agent_chains_store.ensure_tables()
    await backup_config_store.ensure_tables()
    await backup_job_store.ensure_tables()
    await api_key_store.migrate_legacy_key(config.api_key or "")

    # Overlay stored runtime config onto the fresh AppConfig so that DB-stored
    # values (e.g. embeddings_api_url) are honoured even after a hot-reload.
    # Keep a snapshot of the original env-backed values so the dashboard can
    # still show the correct source (env vs db).
    config._env_backed_values = {}
    for key in AppConfig.model_fields:
        config._env_backed_values[key] = getattr(config, key, None)
    stored = await runtime_config_store.get_all()
    for key, value in stored.items():
        if key in AppConfig.model_fields:
            default_val = AppConfig.model_fields[key].default
            current_val = getattr(config, key, None)
            if current_val == default_val:
                try:
                    setattr(config, key, value)
                except Exception:
                    pass

    rag_application = RAGApplication(config=config, provider_store=provider_store, engine=engine)
    rag_application._tools_store = tools_store  # type: ignore[misc]
    await tools_store.seed_builtin_tools()

    app.state.rag_application = rag_application
    app.state.app_config = config
    app.state._blocked_store = blocked_store
    app.state._metrics_store = metrics_store
    app.state._vectorstore_metrics_store = vectorstore_metrics_store
    app.state._runtime_config_store = runtime_config_store
    app.state._provider_store = provider_store
    app.state._api_key_store = api_key_store
    app.state._dashboard_user_store = user_store
    app.state._chat_user_store = chat_user_store
    app.state._chat_groups_store = chat_groups_store
    app.state._chat_conversations_store = chat_conversations_store
    app.state._chat_role_assistants_store = chat_role_store
    app.state._tools_store = tools_store
    app.state._triggers_store = triggers_store
    app.state._message_store = message_store
    app.state._agent_chains_store = agent_chains_store
    app.state._backup_config_store = backup_config_store
    app.state._backup_job_store = backup_job_store
    app.state._backup_service = backup_service
    from app.artifacts import ArtifactStore
    app.state.artifact_store = ArtifactStore(config.data_dir / "artifacts")
    app.state.mcp_manager = getattr(rag_application, "_mcp_manager", None)
    from app.chat_attachments import ChatAttachmentStore
    app.state.chat_attachment_store = ChatAttachmentStore(config.data_dir / "chat_uploads")
    # Periodic orphan cleanup for chat-scoped file dirs (artifacts, uploads).
    # One shot at startup, then daily — cancelled in the lifespan shutdown.
    from app.storage_sweeper import storage_sweeper_loop
    app.state._storage_sweeper_task = asyncio.create_task(
        storage_sweeper_loop(app.state)
    )
    return rag_application


async def _background_rag_init(app: FastAPI, lifespan_start: float) -> None:
    """Background task: finish RAG initialization and start the trigger scheduler."""
    rag_app = getattr(app.state, "rag_application", None)
    if rag_app is None:
        logger.error("Background RAG init: no rag_application found on app.state")
        return

    # Backup scheduler (independent of RAG readiness - only needs the DB)
    backup_service = getattr(app.state, "_backup_service", None)
    backup_config_store = getattr(app.state, "_backup_config_store", None)
    if backup_service is not None and backup_config_store is not None:
        backup_scheduler = getattr(app.state, "_backup_scheduler", None)
        if backup_scheduler is not None:
            backup_scheduler.service = backup_service
            backup_scheduler.config_store = backup_config_store
        else:
            backup_scheduler = BackupScheduler(backup_config_store, backup_service)
            await backup_scheduler.start()
            app.state._backup_scheduler = backup_scheduler

    try:
        await rag_app.initialize(skip_engine_creation=True)
    except Exception as exc:
        logger.error("Background RAG initialization failed: %s", exc)
        return

    init_duration = (time.perf_counter() - lifespan_start) * 1000
    logger.info(
        "RAG initialization complete | Process Time: %s",
        colorize_process_time(init_duration),
        extra={"process_time": init_duration},
    )

    triggers_store = getattr(app.state, "_triggers_store", None)
    message_store = getattr(app.state, "_message_store", None)
    scheduler = getattr(app.state, "_triggers_scheduler", None)
    if triggers_store is not None and rag_app is not None:
        if scheduler is not None:
            scheduler.rag_app = rag_app
            logger.info("Trigger scheduler updated with new RAG application.")
        else:
            scheduler = TriggersScheduler(triggers_store, rag_app, message_store)
            await scheduler.start()
            app.state._triggers_scheduler = scheduler
        # Pause triggers while a restore holds the maintenance flag
        backup_service = getattr(app.state, "_backup_service", None)
        if backup_service is not None:
            scheduler.maintenance_check = lambda: backup_service.in_maintenance

    total_duration = (time.perf_counter() - lifespan_start) * 1000
    logger.info(
        "Lifespan startup: Complete. | Process Time: %s",
        colorize_process_time(total_duration),
        extra={"process_time": total_duration},
    )


async def reload_rag_application(app: FastAPI) -> None:
    """Hot-reload: read a fresh :class:`AppConfig`, shut down the old RAG application
    and rebuild all stores without restarting the uvicorn process.
    """
    logging.info("Hot-reload: shutting down old RAG application...")
    old_rag = getattr(app.state, "rag_application", None)
    if old_rag is not None:
        await old_rag.shutdown()

    old_task = getattr(app.state, "_rag_init_task", None)
    if old_task is not None and not old_task.done():
        old_task.cancel()
        try:
            await old_task
        except asyncio.CancelledError:
            pass

    new_config = AppConfig()
    await _init_core(app, new_config)

    # Re-check emotion model downloads on hot-reload
    try:
        from app.utils.model_download import ensure_emotion_models
        await ensure_emotion_models(new_config)
    except Exception as exc:
        logging.warning("Emotion model download check on reload failed: %s", exc)

    lifespan_start = time.perf_counter()
    task = asyncio.create_task(_background_rag_init(app, lifespan_start))
    app.state._rag_init_task = task
    scheduler = getattr(app.state, "_triggers_scheduler", None)
    if scheduler is not None:
        scheduler.rag_app = getattr(app.state, "rag_application", None)
    logging.info("Hot-reload: RAG application core rebuilt; background init started.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Provision the shared :class:`RAGApplication` for the lifetime of the API process.

    Initialization ingests the document corpus, (re)builds the PGVector collection when
    required, and keeps the ready-to-serve ``RAGApplication`` on ``app.state``.  During
    shutdown the routine ensures embeddings clients and other async resources are closed
    cleanly so restarts remain predictable.
    """
    logging.info("Lifespan startup: Initializing core services...")
    lifespan_start = time.perf_counter()

    await _init_core(app, app_config)

    # Download missing ONNX emotion models if URLs are configured
    try:
        from app.utils.model_download import ensure_emotion_models
        await ensure_emotion_models(app_config)
    except Exception as exc:
        logging.warning("Emotion model download check failed: %s", exc)

    task = asyncio.create_task(_background_rag_init(app, lifespan_start))
    app.state._rag_init_task = task

    core_duration = (time.perf_counter() - lifespan_start) * 1000
    logger.info(
        "Lifespan startup: Core ready. | Process Time: %s",
        colorize_process_time(core_duration),
        extra={"process_time": core_duration},
    )

    yield

    logging.info("Lifespan shutdown: Releasing RAG resources.")
    scheduler = getattr(app.state, "_triggers_scheduler", None)
    if scheduler is not None:
        await scheduler.stop()
    backup_scheduler = getattr(app.state, "_backup_scheduler", None)
    if backup_scheduler is not None:
        await backup_scheduler.stop()
    rag_app = getattr(app.state, "rag_application", None)
    if rag_app is not None:
        try:
            await rag_app.shutdown()
        except Exception as exc:
            logging.warning("RAGApplication shutdown failed: %s", exc)
    task = getattr(app.state, "_rag_init_task", None)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    # Ensure sandbox containers are cleaned up even if RAG shutdown failed
    try:
        from agent.tools.docker_sandbox import ConversationSandboxManager
        await ConversationSandboxManager().destroy_all()
    except Exception as exc:
        logging.warning("Sandbox cleanup during lifespan shutdown failed: %s", exc)
    mcp_manager = getattr(app.state, "mcp_manager", None)
    if mcp_manager is not None:
        try:
            await mcp_manager.aclose()
        except Exception as exc:
            logging.warning("MCP client cleanup during lifespan shutdown failed: %s", exc)
    sweeper_task = getattr(app.state, "_storage_sweeper_task", None)
    if sweeper_task is not None:
        sweeper_task.cancel()
        try:
            await sweeper_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logging.warning("Storage sweeper shutdown failed: %s", exc)

# ==============================================================================
# FASTAPI APPLICATION INITIALIZATION
# ==============================================================================
app = FastAPI(
    title="RAG Cybersecurity Assistant API",
    description="An OpenAI-compatible API for a Retrieval-Augmented Generation chatbot.",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)

# --- Add Middleware ---
async def _on_size_block(path: str, method: str, status_code: int, reason: str) -> None:
    store = getattr(app.state, "_blocked_store", None)
    if store is not None:
        try:
            await store.save(BlockedRequestRecord(path=path, method=method, status_code=status_code, reason=reason))
        except Exception:
            logging.getLogger(__name__).exception("Failed to save size-blocked request for %s", path)


def _scope_request(scope):
    from starlette.requests import Request

    return Request(scope)


async def _chat_media_body_limit(scope) -> int:
    """Body cap for /chat/completions and /chat/runs/*.

    Images travel inside the JSON body as base64 (~1.4x), so the cap must
    cover the image budget plus message text — not just max_request_bytes.
    Reads effective config per request: admin toggles apply immediately.
    """
    from app.config_access import effective_config_bool, effective_config_value

    req = _scope_request(scope)
    base = int(
        await effective_config_value(req, "max_request_bytes", app_config.max_request_bytes)
        or app_config.max_request_bytes
    )
    if not await effective_config_bool(req, "chat_image_upload_enabled"):
        return base
    img_max = int(await effective_config_value(req, "chat_image_max_size_bytes", 5_242_880) or 0)
    img_cnt = int(await effective_config_value(req, "chat_image_max_count", 4) or 0)
    msg_budget = int(await effective_config_value(req, "max_total_chars", 120_000) or 120_000) * 4
    return max(base, msg_budget + (img_max * img_cnt * 3) // 2)


async def _chat_attachment_body_limit(scope) -> int:
    """Body cap for /chat/attachments — the endpoint enforces
    chat_file_max_size_bytes itself; the middleware must let that body
    (plus multipart overhead) through."""
    from app.config_access import effective_config_value

    req = _scope_request(scope)
    base = int(
        await effective_config_value(req, "max_request_bytes", app_config.max_request_bytes)
        or app_config.max_request_bytes
    )
    fmax = int(await effective_config_value(req, "chat_file_max_size_bytes", 20_971_520) or 0)
    return max(base, fmax + 1_048_576)


app.add_middleware(
    RequestSizeLimitMiddleware,
    max_bytes=app_config.max_request_bytes,
    on_block=_on_size_block,
    exempt_paths={
        "/admin/config/instructions/upload",
        "/admin/knowledge/upload",
        "/admin/audio/voices/upload",
        "/admin/audio/voices/kokoro/upload",
        "/audio/stt",
        # Branding can carry base64 logo+favicon (~2.7MB); the router applies
        # its own 3MB payload cap — tighter than this exemption implies.
        "/admin/branding",
    },
    path_limits={
        "/chat/attachments": _chat_attachment_body_limit,
        "/v1/chat/attachments": _chat_attachment_body_limit,
        "/chat/completions": _chat_media_body_limit,
        "/v1/chat/completions": _chat_media_body_limit,
        "/chat/runs": _chat_media_body_limit,
        "/v1/chat/runs": _chat_media_body_limit,
    },
)
app.add_middleware(LoggingMiddleware, app_config=app_config)
app.add_middleware(SessionRefreshMiddleware)
app.add_middleware(APIKeyMiddleware, legacy_key=app_config.api_key)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include Routers ---
app.include_router(chat.router)
app.include_router(admin.router)
app.include_router(admin_audio.router)
app.include_router(admin_emotion.router)
app.include_router(tools.router)
app.include_router(triggers.router)
app.include_router(agent_chains.router)
app.include_router(audio.router)
app.include_router(emotions.router)
app.include_router(widget.router)
app.include_router(public_chat.router)
app.include_router(branding.router)
app.include_router(public_chat_auth.router)
app.include_router(backup.router)
app.include_router(agent_workspace.router)


# --- Widget Static Files ---
# Serviere das Widget unter /widget/ (aus dem Build-Ordner im widget/dist Verzeichnis)
# html=False verhindert die automatische Auslieferung von index.html
# Prüfe mehrere mögliche Pfade, da der Build-Kontext oder Dockerfile die Dateien
# an verschiedenen Orten ablegen kann (z.B. /app/widget/dist oder /widget/dist).
possible_widget_paths = [
    Path(__file__).resolve().parent.parent / "widget" / "dist",
    Path.cwd() / "widget" / "dist",
    Path("/widget/dist"),
]

widget_build_path: Optional[Path] = None
for candidate in possible_widget_paths:
    if candidate.exists():
        widget_build_path = candidate
        break

logger = logging.getLogger(__name__)
if widget_build_path:
    app.mount("/widget", StaticFiles(directory=str(widget_build_path), html=False), name="widget")
    logger.info("Widget static files mounted at /widget from %s", widget_build_path)
else:
    logger.warning(
        "Widget build directory not found in any of the expected locations: %s - widget will not be served",
        possible_widget_paths,
    )


class DashboardStaticFiles(StaticFiles):
    """Serves index.html for any 404 under /dashboard so React Router handles SPA routes."""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


# --- Dashboard Static Files ---
possible_dashboard_paths = [
    Path(__file__).resolve().parent.parent / "dashboard" / "dist",
    Path.cwd() / "dashboard" / "dist",
    Path("/dashboard/dist"),
]

dashboard_build_path: Optional[Path] = None
for candidate in possible_dashboard_paths:
    if candidate.exists():
        dashboard_build_path = candidate
        break

if dashboard_build_path:
    app.mount("/dashboard", DashboardStaticFiles(directory=str(dashboard_build_path), html=True), name="dashboard")
    logger.info("Dashboard static files mounted at /dashboard from %s", dashboard_build_path)
else:
    logger.warning(
        "Dashboard build directory not found in any of the expected locations: %s - dashboard will not be served",
        possible_dashboard_paths,
    )


# --- Emotion Model Static Files ---
emotion_models_path = app_config.data_dir / "models" / "emotion"
if emotion_models_path.exists():
    app.mount("/models/emotion", StaticFiles(directory=str(emotion_models_path)), name="emotion_models")
    logger.info("Emotion model files mounted at /models/emotion from %s", emotion_models_path)
else:
    logger.info("Emotion model directory does not exist yet: %s", emotion_models_path)


# --- Public Chat Static Files ---
possible_chat_paths = [
    Path(__file__).resolve().parent.parent / "dashboard" / "dist" / "public-chat",
    Path.cwd() / "dashboard" / "dist" / "public-chat",
    Path("/dashboard/dist/public-chat"),
]

chat_build_path: Optional[Path] = None
for candidate in possible_chat_paths:
    if candidate.exists():
        chat_build_path = candidate
        break

if chat_build_path:
    # Mount shared assets FIRST so they take priority over the catch-all
    # /chat StaticFiles mount (html=True falls back to index.html for unknown
    # paths, which would swallow /chat/assets/... requests).
    shared_assets = chat_build_path.parent / "assets"
    if shared_assets.exists():
        app.mount("/chat/assets", StaticFiles(directory=str(shared_assets)), name="chat_assets")
        logger.info("Public chat assets mounted at /chat/assets from %s", shared_assets)
    # Then mount the chat HTML entry point
    app.mount("/chat", StaticFiles(directory=str(chat_build_path), html=True), name="chat")
    logger.info("Public chat static files mounted at /chat from %s", chat_build_path)
else:
    logger.warning(
        "Public chat build directory not found in any of the expected locations: %s - public chat will not be served",
        possible_chat_paths,
    )


_APP_COMMIT: Optional[str] = None


def _app_commit() -> str:
    """Return the git commit this build was created from (baked in via
    ``GIT_COMMIT_SHA`` at image build time; falls back to the checkout)."""
    global _APP_COMMIT
    if _APP_COMMIT is not None:
        return _APP_COMMIT
    sha = os.environ.get("GIT_COMMIT_SHA") or os.environ.get("CI_COMMIT_SHORT_SHA") or ""
    if not sha or sha == "unknown":
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parent.parent,
                capture_output=True,
                text=True,
                timeout=2,
            )
            sha = out.stdout.strip() if out.returncode == 0 else ""
        except Exception:
            sha = ""
    _APP_COMMIT = sha or "unknown"
    return _APP_COMMIT


@app.get("/health", include_in_schema=False)
async def health_check(request: Request) -> dict[str, str]:
    """Lightweight health probe that skips API key checks."""
    rag_app = getattr(request.app.state, "rag_application", None)
    status = "ready" if rag_app and getattr(rag_app, "ready", False) else "initializing"
    return {"status": status, "commit": _app_commit()}
