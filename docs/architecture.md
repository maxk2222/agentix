# Architecture

## System topology

```mermaid
flowchart LR
    subgraph Clients
        W["Chat Widget<br/>(widget/)"]
        PC["Public Chat<br/>(dashboard/dist/public-chat)"]
        EXT["External tools<br/>/v1/* API"]
        UI["Admin Dashboard<br/>(React)"]
    end

    subgraph "Agentix Agent Platform (app/)"
        GW["FastAPI<br/>app/main.py"]
        RT["Routers<br/>chat · admin · tools · triggers ·<br/>agent_chains · audio · emotions ·<br/>widget · public_chat · backup ·<br/>agent_workspace"]
        SCHED["Schedulers<br/>TriggersScheduler · BackupScheduler"]
        MCP["MCPClientManager<br/>stdio / streamable-http"]
    end

    subgraph "RAG pipeline (agent/)"
        RAG["RAGApplication<br/>retrieval · assistants · tools"]
        RUN["AgentRunner<br/>iterations · dedup · observer<br/>subagents · steering"]
        EMB["Embeddings client"]
    end

    DB[("PostgreSQL + PGVector<br/>all durable state")]
    FS["data/ filesystem<br/>agent_workspaces · caches<br/>ws_render_cache (not backup scope)"]
    SB["Sandbox containers<br/>code_execution · /agent_ws mount"]
    MCPS["External MCP servers"]
    LLM["LLM endpoint<br/>(OpenAI-compatible)"]
    TTS["TTS / STT providers<br/>(Kokoro · faster-whisper)"]
    DAV["Sciebo / WebDAV<br/>knowledge sources · backup archives"]

    W --> GW
    PC --> GW
    EXT --> GW
    UI --> GW
    GW --> RT --> RAG
    RAG --> RUN
    SCHED --> DB
    RT --> DB
    RAG --> DB
    RUN --> LLM
    RAG --> EMB
    RAG --> TTS
    RUN --> FS
    RUN --> SB
    RUN --> MCP
    MCP --> MCPS
    RT --> DAV
    SCHED --> DAV
```

## Components

| Layer | Path | Responsibility |
|---|---|---|
| API gateway | `app/main.py`, `app/routers/` | HTTP surface, auth, request size limits, static mounts (`/dashboard`, `/widget`, `/chat`) |
| Application config | `agent/config.py` | `AppConfig` (pydantic-settings): env vars → typed config; `connection_string` assembles the asyncpg DSN |
| Persistence | `app/*_store.py`, `agent/utils/pg_helpers.py` | Postgres-backed stores; tables created lazily via `ensure_tables()` |
| RAG core | `agent/` | Document processing, chunking, hybrid retrieval, assistants, tools |
| Agent runtime | `agent/tools/runner.py`, `agent/subagents.py`, `agent/observer.py` | Iteration loop, tool execution, dedup, subagents, observer, steering — see [Agent runtime](agent.md) |
| Agent workspace | `app/agent_workspace.py`, `app/doc_inspect.py` | Scoped persistent folders, quotas, document rendering/inspection |
| Code sandbox | `agent/tools/docker_sandbox.py`, `agent/tools/sandbox_server.py` | Per-run Docker containers for `code_execution`; `/agent_ws` mount |
| MCP | `app/mcp_client.py` | External MCP servers → `mcp__*` tool proxies |
| Schedulers | `app/triggers_scheduler.py`, `app/backup_scheduler.py` | Asyncio loops (30 s tick); DB-driven `next_run_at` due-checks |
| Dashboard | `dashboard/src/` | React SPA served from `/dashboard`; session-cookie auth |
| Widget | `widget/` | Standalone bundle served from `/widget` |

## Persistence model

Everything durable lives in one PostgreSQL database (default `agentix-db`):

| Domain | Tables (prefix `agentix_` unless noted) |
|---|---|
| Vectors | `langchain_pg_collection`, `langchain_pg_embedding` |
| Knowledge | `agentix_raw_files`, `agentix_source_state`, WebDAV source tables |
| Assistants | `agentix_instructions_versions` |
| Conversations | conversations / messages stores |
| Identity | dashboard users, API keys |
| Operations | runtime_config, metrics, tools, triggers, agent chains |
| Backup | `agentix_backup_config`, `agentix_backup_jobs` |

`data/` contains caches, downloaded models, **agent workspaces** (`agent_workspaces/` incl. the sibling `ws_render_cache` render cache) and temporary artefacts — it is **not** part of the backup scope and can be rebuilt. Agent workspaces are the exception to "rebuildable": they hold user-relevant files and survive until quotas or the orphan sweeper remove them.

## Startup lifecycle

```mermaid
sequenceDiagram
    participant U as Uvicorn
    participant M as app/main.py
    participant DB as PostgreSQL
    participant R as RAGApplication
    participant S as Schedulers

    U->>M: lifespan start
    M->>DB: create engine (asyncpg)
    M->>DB: ensure_tables() for all stores
    M->>DB: load runtime_config overlay
    Note over M,R: _background_rag_init()
    M->>S: start BackupScheduler (DB only)
    M->>R: initialize (embeddings, vector stores)
    M->>S: start TriggersScheduler + maintenance guard
    M-->>U: ready (GET /health)
```

The **BackupScheduler starts before RAG initialization**, so scheduled backups keep working even if LLM/embedding backends are unavailable.

## Backup data flow

```mermaid
flowchart TD
    A[Scheduler or manual trigger] --> B["pg_dump -Fc<br/>(custom format)"]
    B --> C["manifest.json<br/>sha256(plain) + sha256(enc) + metadata"]
    B --> D["AES-256-GCM chunked encrypt<br/>BACKUP_ENCRYPTION_KEY"]
    D --> E["PUT archive + manifest<br/>WebDAV → Sciebo"]
    E --> F["PROPFIND verify<br/>+ retention cleanup"]
    F --> G[("agentix_backup_jobs<br/>audit row")]

    R["Restore (typed confirm)"] --> RS["pre-restore safety dump"]
    RS --> RM["maintenance mode:<br/>triggers paused"]
    RM --> RD["download + decrypt +<br/>checksum verify"]
    RD --> RR["pg_restore --clean --if-exists"]
    RR --> RV["post-restore validation"]
    RV --> G
```

See [Backup & Recovery](backup-and-recovery.md) for the full runbook.
