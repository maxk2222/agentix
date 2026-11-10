# Overview

The **Agentix Agent Platform** is an OpenAI-compatible Retrieval-Augmented Generation (RAG) service that supports persona-driven assistants, RAG knowledge bases and tool-using agents. It powers in-game training and standalone advisory experiences alike.

## What it does

- **Persona-driven assistants** — multiple assistants ("agents") with individual system instructions, models, model-aware reasoning settings and tool access, managed through the dashboard or `instructions.yaml`.
- **Agent runtime** — tool-calling iteration loop with native & JSON-fallback modes, background subagents, a run observer, mid-run user steering, persistent conversation memory and content deduplication — see [Agent runtime](agent.md).
- **Agent workspace** — persistent scoped folders (conversation/user/assistant) the agent reads and writes; uploads can be imported, generated files offered as downloads, and `vision_read` lets the agent *see* rendered PDF/Office pages — see [Agent workspace](agent.md#agent-workspace).
- **Tools & MCP** — 25+ built-in tools (web/academic search, sandboxed Python, document generation, e-mail, calendar, SSH) plus custom tools and external MCP servers — see [Tools](tools.md).
- **Knowledge base with RAG** — documents are ingested (with OCR for scans and slides), chunked, embedded and stored in PGVector. Hybrid retrieval (vector + full-text, RRF fusion) feeds assistant answers.
- **OpenAI-compatible API** — clients use the standard `/v1/chat/completions` interface; the platform adds tool calling, run control (status/stream/cancel/input), attachments, artifact downloads, agent chains and streaming.
- **Embeddable chat widget** — a standalone JS widget (`widget/`) that can be dropped into any page and configured from the dashboard.
- **Public chat** — token-gated, rate-limited public chat surface for external participants.
- **Operational tooling** — triggers (scheduled prompts), agent chains (multi-step flows), audio (STT/TTS), emotion analytics, conversations archive, metrics, and a role-based admin dashboard.
- **Backup & recovery** — encrypted PostgreSQL dumps uploaded to Sciebo via WebDAV, with restore drills and retention rules. See [Backup & Recovery](backup-and-recovery.md).

## Project context

| | |
|---|---|
| Backend | Python 3.11, FastAPI, SQLAlchemy (async), LangChain |
| Database | PostgreSQL + PGVector (single canonical store for all state) |
| LLM access | OpenAI-compatible endpoints (`LLM_API_URL`), pluggable providers |
| Dashboard | React 18 + Vite (`dashboard/`) |
| Widget | TypeScript + Vite (`widget/`) |
| Deployment | Docker (`Dockerfile`), GitLab CI (`.gitlab-ci.yml`) |

## Repository layout

```
agent/      RAG pipeline: config, retrieval, processing, tools, agent runtime
app/        FastAPI application: routers, stores, schedulers, auth, MCP client
dashboard/  Admin dashboard (React/TypeScript)
widget/     Embeddable chat widget (TypeScript)
defaults/   Default instructions and seed data
tests/      pytest suite (unit + router-level, fake engines)
docs/       This documentation
```

## Design properties

- **Single canonical datastore.** All durable state — vectors, raw files, conversations, users, API keys, runtime config, triggers, WebDAV sources, backup metadata — lives in PostgreSQL. `data/` holds caches and model artefacts only.
- **Lazy schema.** Stores create their tables on first use (`_ensure_tables()`); there is no Alembic-style migration framework.
- **Env-over-DB config.** `agent/config.py` reads environment variables; selected runtime settings persist in the `runtime_config` table with env always winning.
