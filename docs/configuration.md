# Configuration

All settings are environment variables read by `AppConfig` (`agent/config.py`) via pydantic-settings. `.env` in the working directory is loaded automatically. Two templates ship with the repo: `.env.example.minimum` (required only) and `.env.example.complete` (everything).

Selected settings can also be stored in the `runtime_config` DB table via the dashboard; **environment variables always take precedence** over stored values.

[[_TOC_]]

## Core

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `data` | Local data directory (caches, models, tmp) |
| `INSTRUCTIONS_FILE` | `data/SystemInstructions/instructions.yaml` | Assistant definitions YAML |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Bind address |
| `API_RELOAD` | `false` | Uvicorn auto-reload (dev only) |
| `API_KEY` | — | Legacy single API key (migrated into the API-key store) |
| `API_KEYS_ENABLED` | `true` | Enforce API keys on `/v1/*` |
| `LOG_LEVEL` / `LOG_DIR` / `LOG_FILE_MAX_BYTES` | `INFO` / `logs` / 5 MiB | Logging |
| `USER_AGENT` | `Agentix-RAG/1.0` | HTTP user agent for outbound fetches |
| `AUTO_INITIALIZE` | `true` | Auto-ingest `KNOWLEDGE_URLS` at startup |
| `KNOWLEDGE_URLS` | — | Comma-separated seed URLs |

## PostgreSQL

| Variable | Default |
|---|---|
| `PGHOST` | `localhost` |
| `PGPORT` | `5432` |
| `PGDATABASE` | `agentix-db` |
| `PGUSER` | `default` |
| `PGPASSWORD` | `mypassword` |

## LLM & embeddings

| Variable | Default | Description |
|---|---|---|
| `LLM_API_URL` | `http://127.0.0.1:5001` | OpenAI-compatible base URL |
| `LLM_API_KEY` | — | Key for the LLM endpoint |
| `SYSTEM_SLOW` | `true` | Conservative latency assumptions |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `EMBEDDINGS_API_URL` / `EMBEDDINGS_ENDPOINT` / `EMBEDDINGS_API_KEY` | — | Embeddings endpoint config |
| `EMBEDDINGS_REQUEST_TIMEOUT` / `EMBEDDINGS_MAX_RETRIES` / `EMBEDDINGS_BACKOFF_FACTOR` | `30` / `3` / `0.5` | Resilience |
| `EMBEDDING_MAX_CHARS` | `4000` | Truncation for embedding input |

## Request validation

| Variable | Default |
|---|---|
| `MAX_REQUEST_BYTES` | `1048576` |
| `MAX_MESSAGES` | `100` |
| `MAX_MESSAGE_CHARS` | `20000` |
| `MAX_TOTAL_CHARS` | `120000` |
| `REQUIRE_LAST_USER` | `true` |
| `ALLOW_NON_STRING_CONTENT` | `false` |
| `VALIDATION_CONTROL_CHARS_MODE` | `reject` |
| `VALIDATION_ENFORCE` | `true` |
| `STREAM_IDLE_TIMEOUT_SECONDS` | `300.0` (LLM token-level idle watchdog — aborts an in-flight LLM stream after N seconds without a token; legacy `STREAMING_TIMEOUT_SECONDS` still accepted) |
| `RUN_IDLE_TIMEOUT_SECONDS` | `900.0` (run-level idle watchdog — aborts a run after N seconds without a *real* stream item: token, reasoning, tool or subagent event; synthetic tool keepalives do not count. Should stay above the sandbox `SANDBOX_INSTALL_TIMEOUT` + `SANDBOX_EXEC_MAX_TIMEOUT` budgets so a healthy code_execution fits one silent window) |

## History & retrieval tuning

| Variable | Default |
|---|---|
| `MAX_HISTORY_MESSAGES` | `50` |
| `ROLLING_HISTORY_*` | context window `32768`, output `2048`, min turns `4`, margin `0.08`, summary timeout `30 s` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1500` / `200` |
| `VECTOR_SEARCH_K` / `VECTOR_BATCH_SIZE` / `VECTOR_FORCE_REBUILD` | `5` / `200` / `false` |
| `RETRIEVAL_HYBRID_ENABLED` | `true` |
| `HYBRID_VECTOR_CANDIDATES` / `HYBRID_KEYWORD_CANDIDATES` / `HYBRID_RRF_K` / `HYBRID_MAX_CHUNKS_PER_SOURCE` | `20` / `20` / `60` / `3` |
| `HYBRID_QUERY_EXPANSION_ENABLED` / `HYBRID_MAX_EXPANDED_QUERIES` / `HYBRID_RERANK_ENABLED` / `HYBRID_MIN_SEMANTIC_SIMILARITY` | `true` / `2` / `true` / `0.4` |
| `RETRIEVAL_HISTORY_*` | enabled `true`, user messages `2`, max chars `800`, standalone detection `true` |
| `RETRIEVAL_RECENCY_ENABLED` / `RETRIEVAL_RECENCY_PRIMARY_SHARE` | `true` / `0.6` | Recency-aware retrieval: share of results preferring newer chunks |
| `CONTEXTUALIZE_QUESTION_*` | enabled `false`, model —, temperature `0.0`, disable reasoning `true` |
| `RETRIEVER_TIMEOUT_SECONDS` | `10.0` |
| `INDEXING_INCREMENTAL_ENABLED` | `true` |
| `LOADER_MAX_CONCURRENCY` | `8` |

## Dashboard

| Variable | Default | Notes |
|---|---|---|
| `DASHBOARD_ENABLED` | `false` | Required for `/dashboard` and admin API |
| `DASHBOARD_ADMIN_USERNAME` / `DASHBOARD_ADMIN_PASSWORD` | `admin` / — | Bootstrap admin |
| `DASHBOARD_SESSION_SECRET` | — | Cookie signing secret — set a random value |
| `DASHBOARD_SESSION_TTL_SECONDS` | `3600` | Session lifetime |

## Agent assistants and subagents

Assistant behavior is configured in `instructions.yaml` (or in the dashboard's
Assistant configuration page). Enable background subagents per agent-enabled
assistant:

```yaml
- name: CyberAnalyst
  agent_enabled: true
  subagents:
    enabled: true
    max_concurrent: 2
    max_depth: 1
    max_total: 8
```

Subagents are isolated background clones of the calling assistant: they use
the same model and instructions, but receive no chat history. They inherit the
parent's full tool set minus `subagents.exclude_tools` (an explicit list, even
empty, replaces the defaults — by default only `agent_memory` is excluded).
`max_total` is the total spawn budget for one request
(default `8`, allowed range `1`–`500`); `max_concurrent` still limits how many
run simultaneously. Include all required context in each task. The agent can launch parallel tasks with
`spawn_subagent`, inspect live activity with `subagent_status`, and collect
answers and tool activity reports with `wait_subagent`. A bounded `wait`
tool is available to every agent for polling and retry delays. Any subagents
still outstanding when the main agent is ready to answer are automatically
collected before final synthesis.

Mid-run guidance: `nudge_subagent(subagent_id, message)` injects a text
message into a running child — it lands as a clearly marked `[Message from
parent agent]` entry before the child's next iteration (and wakes a child
that is currently inside `wait`/`wait_subagent`). Nudges are one-way: the
child cannot reply, so the parent collects its answer via `wait_subagent`.
Users can nudge any subagent in the tree through the run-input endpoint
(`POST /v1/chat/runs/{id}/input` with `subagent_id`, delivered as
`[User message]`) or the nudge field in the dashboard's subagent detail
view. Likewise, `wait` and `wait_subagent` return early when new input is
queued — a steer message no longer has to sit out a running sleep.

### Run observer

Each agent-enabled assistant can also enable a run observer — a lightweight
supervisor that watches the running agent at iteration boundaries, keeps its
todo list current, and injects short guidance notes when the run stalls
semantically (repeated identical tool calls, consecutive tool errors, a
missing or stale plan, budget pressure):

```yaml
- name: CyberAnalyst
  agent_enabled: true
  observer:
    enabled: true
    assistant: ObserverModel        # optional; falls back to the run's model
    every_iterations: 4             # periodic check cadence (1–20)
    every_seconds: 180              # periodic time cadence (30–3600)
    max_checks: 8                   # check budget per run (1–50)
    max_nudges: 4                   # guidance-injection budget per run (1–20)
    check_timeout_seconds: 60       # LLM check timeout (10–300)
    include_subagents: true         # give each spawned subagent its own observer
    instructions: ""                # optional extra focus, max 4000 chars
```

Observer notes reach the agent as clearly marked `[Observer note]` messages,
distinct from live user steering, and are shown in the dashboard as observer
events. The observer complements the stream-idle watchdog: the watchdog
detects hard transport stalls, the observer detects semantic stalls. Observer
checks run asynchronously so they never block an agent iteration, and a
failing observer LLM never aborts the run.

### Agent workspace (dedicated folder)

Each agent-enabled assistant can get a persistent folder the agent manages on
its own — it can list, read, write, append, move and delete files there via
the `agent_workspace` tool, and hand files to the user as chat downloads
(`offer` action → regular artifact cards). The code-execution sandbox sees the
same folder bind-mounted at `/agent_ws` (read-write in every sandbox mode).
The agent can also pull files the user uploaded to the chat into the folder:
`list_uploads` shows them, `import_upload` copies one in (by id or filename,
destination optional, default `uploads/<name>`) — imported files then count
against the workspace quota like any other write.

```yaml
- name: CyberAnalyst
  agent_enabled: true
  workspace:
    enabled: true
    scope: conversation        # conversation | user | assistant
    api_enabled: false         # expose files read-only via the API
    vision_read: false         # let 'view' attach rendered page images
    max_file_mb: 50            # per-file limit (1–500)
    max_total_mb: 500          # per-folder quota (10–10000)
    max_files: 1000            # file count limit (1–100000)
```

> [!warning]
> `scope: assistant` shares one folder across **all** users and conversations —
> only for trusted/single-user setups. `user`-scoped API access requires a
> signed-in chat identity (anonymous callers get 401). `vision_read` needs a
> vision-capable model — text-only providers reject requests containing
> `image_url` parts.

`scope` controls sharing: `conversation` gives each conversation its own
folder (default, strongest isolation), `user` shares one folder per signed-in
chat user across that user's conversations (falls back to per-conversation for
anonymous callers), `assistant` shares one folder across ALL users and
conversations — only for trusted/single-user setups, since every caller sees
everyone's files. Subagents inherit the root run's resolved folder.

`api_enabled: true` exposes two read-only endpoints —
`GET /chat/assistants/{name}/workspace/files` (list) and
`GET /chat/assistants/{name}/workspace/file?path=…` (download). Access mirrors
the assistant's own access rules (`api_accessible`, public-chat roles,
dashboard session, API keys); `conversation`-scoped access additionally
requires the conversation id (bearer credential). It can be revoked at any
time without deleting data.

`vision_read: true` unlocks the `view` action: the agent can render PDF
pages, office documents (via LibreOffice → PDF → PyMuPDF) and plain images
into pictures the model actually sees — the rendered images are injected as
a follow-up user message with `image_url` blocks after the tool result.
Enable this only for vision-capable models; with a text-only model the
provider will reject the request. `inspect` works without `vision_read` and
returns document *structure* (slide shapes with positions, heading outlines,
tables, sheet dims) plus — with `extract_media=true` — embedded media files
written to `extracted/<name>/` inside the workspace (quota-enforced, visible
to the sandbox and offerable). Hard caps: 8 images per call, 4 MB per image,
1568 px longest edge; rendered pages and LibreOffice-converted PDFs are
cached on disk (`AGENT_WORKSPACE_RENDER_CACHE`, default
`<agent_workspaces>/../ws_render_cache`, LRU 256 MB / 24 h).

Within a run, identical content is sent to the model exactly once: each
unique image (by data URI) and each identical tool output longer than 1000
chars keeps only its *latest* occurrence — earlier duplicates collapse to
`#img-N`/`#res-N` reference stubs. At most 16 unique images stay live at a
time; older ones are evicted (tool renders before user uploads) with a stub
that tells the model to call `view` again.

Storage lives under `AGENT_WORKSPACE_ROOT` (daemon-visible bind source,
default `<DATA_DIR>/agent_workspaces`) with the same daemon-vs-app split as
the sandbox workspaces — set `AGENT_WORKSPACE_APP_ROOT` when the app runs in a
container and sees the data dir under a different mount path. Layout:
`<assistant>/<scope-dir>/` where scope dirs are `shared`, `u_<user>` or
`c_<conversation>`. Renaming an assistant moves its tree (single added/removed
name heuristic); deleting one leaves an orphan the storage sweeper removes
after the orphan TTL.

### Model-aware reasoning parameters

Each assistant accepts a normalized `reasoning` block with canonical keys.
The backend translates it into the wire format the selected model family
actually understands:

```yaml
- name: CyberAnalyst
  model: Qwen3-32B
  reasoning:
    enabled: false        # thinking on/off (toggle families)
    budget: 4096          # token budget (budget families)
    effort: high          # effort level (effort families)
    level: high           # thinking level (Gemini 3)
```

| Model family (matched by name/URL) | Wire parameter |
|---|---|
| Qwen 3.5/3.8/3.9/Next/4 (`qwen3.8`, `qwen3-next`, `qwen4`) | top-level `reasoning_effort` (`low`/`medium`/`xhigh`) + optional `chat_template_kwargs` |
| Qwen3 / QwQ (`qwen3`, `qwq`) | `chat_template_kwargs.enable_thinking` + `thinking_budget` |
| DeepSeek, Granite, Nemotron, Holo | `chat_template_kwargs.thinking` |
| Gemma | `chat_template_kwargs.enable_thinking` |
| Claude (`claude` / anthropic URL) | `thinking.type` + `budget_tokens` |
| Gemini (`gemini` / googleapis URL) | `thinkingConfig.thinkingBudget` + `thinkingLevel` |
| OpenAI o-series, gpt-5/6 (`oN`, `gpt-5`, `gpt-6`) | `reasoning.effort` |
| Unknown models | verbatim `reasoning` passthrough (historical behavior) |

Only the keys a family supports are sent; the rest is ignored and logged.
Precedence: `reasoning` > `extra_body` JSON > the legacy top-level
`enable_thinking` field (deprecated — use `reasoning.enabled`; still honored
when `reasoning.enabled` is unset). The translation applies everywhere a model
is invoked, including the agent loop — `enabled: false` finally silences
Qwen-style thinking there too.

The dashboard config editor fetches `/admin/config/reasoning-schema?model=…`
to render only the controls the family supports; unknown models get the
generic effort select. When a `provider_id` is given, the schema endpoint
also consults the **live capability probe** (`app/reasoning_detector.py`,
analogous to tool-calling detection): on first access it fires a background
probe that POSTs `reasoning_effort` and `chat_template_kwargs.enable_thinking`
to the provider and persists the result per `(provider, model)` in
`provider_model_reasoning`. Probed results override family matching — both in
the schema descriptor (`probed: true` badge in the UI) and at request time —
so renamed or unknown model deployments still get the wire format their
backend verified. `POST /admin/config/providers/{id}/detect-reasoning`
re-triggers a probe manually.

New families can be added without a redeploy via
`REASONING_PARAM_RULES_JSON` (JSON array or path to a JSON file, array or
`{"rules": [...]}`):

```json
[{"family": "my_effort", "patterns": ["mymodel"], "wire": "openai_effort",
  "controls": [{"key": "effort", "type": "select", "options": ["low", "high"]}]}]
```

Runtime rules are prepended to the built-in table and take precedence.
`wire` must be one of `qwen_thinking`, `qwen3_effort`, `probed_effort`,
`chat_template_thinking`, `chat_template_enable_thinking`, `claude_thinking`,
`gemini_thinking`, `openai_effort`, `generic`, `drop`.

## Agent workspace paths

| Variable | Default | Description |
|---|---|---|
| `AGENT_WORKSPACE_ROOT` | derived from `/proc/self/mountinfo`, else `<DATA_DIR>/agent_workspaces` | Host-visible root for assistant workspaces (daemon-side bind source for sandbox mounts). Containerized app: auto-derived from the `/app/data` bind; set explicitly for non-standard layouts — a wrong value mounts an arbitrary host dir into sandboxes |
| `AGENT_WORKSPACE_APP_ROOT` | `<DATA_DIR>/agent_workspaces` | App-side path when the app runs in a container and sees the data dir under a different mount path |
| `AGENT_WORKSPACE_RENDER_CACHE` | `<AGENT_WORKSPACE_ROOT>/../ws_render_cache` | Disk cache for rendered pages + LibreOffice-converted PDFs (LRU 256 MB / 24 h) |

## Code execution sandbox

The `code_execution` tool runs Python inside a dedicated container per run (created via the Docker daemon socket). The in-sandbox agent server is `agent/tools/sandbox_server.py`.

| Variable | Default | Description |
|---|---|---|
| `SANDBOX_IMAGE` | auto-built from `sandbox.Dockerfile` | Image used for sandbox containers |
| `SANDBOX_BASE_IMAGE` | `python:3.11-slim-bookworm` | Base for the auto-built image |
| `SANDBOX_NETWORK` | `agentix-sandbox-net` | Docker network for sandbox containers |
| `SANDBOX_NETWORK_INTERNAL` | unset (egress policy decides) | `1`/`true` is a **hard operator floor**: every sandbox's egress policy is forced to `internet=off` (newly created bridges are additionally `internal`; embedded-DNS is dropped too so DNS-tunnel exfil fails). pip installs then fail fast with a clear error |
| `SANDBOX_EGRESS_BLOCKED_CIDRS` | unset | Comma-separated IPs/CIDRs dropped for **every** sandbox — unioned with the per-tool `egress_blocked` list |
| `SANDBOX_RUNTIME` | unset (runc) | Container runtime override — `runsc` (gVisor) or a Kata runtime for stronger kernel isolation |
| `SANDBOX_PIP_INDEX_URL` / `SANDBOX_PIP_EXTRA_INDEX_URL` | unset | package mirror for sandbox installs (e.g. internal devpi/Nexus with an allowlisted package set) — forwarded to the installer only (`uv pip` with pip fallback), never to executed code |
| `SANDBOX_PIP_ONLY_BINARY` | unset | `1` adds `--only-binary :all:` — no sdist `setup.py` ever executes inside the sandbox |
| `SANDBOX_NOFILE_LIMIT` | `4096` | `RLIMIT_NOFILE` inside the sandbox |
| `SANDBOX_FSIZE_LIMIT_MB` | `2048` | `RLIMIT_FSIZE` cap in MB inside the sandbox |
| `SANDBOX_TTL_SECONDS` | `300` | Max lifetime of a sandbox container |
| `SANDBOX_MEM_LIMIT` | `512m` | Memory (and swap) cap per container — raise for pip installs of larger trees (pandas & co.) |
| `SANDBOX_INSTALL_TIMEOUT` | `240` | In-container pip install subprocess timeout (seconds) |
| `SANDBOX_EXEC_MAX_TIMEOUT` | `600` | Hard cap per `/exec` call (seconds). The in-container watchdog interrupts overrunning code best-effort; the tool's `timeout` arg is clamped to this |
| `SANDBOX_WORKSPACE_ROOT` / `SANDBOX_WORKSPACE_APP_ROOT` | `<DATA_DIR>/sandbox_workspaces` (daemon-visible / app-side split, like `AGENT_WORKSPACE_*`) | Per-run workspace bind mounts |
| `SANDBOX_TOKEN` | per-container generated | Bearer token between app and sandbox server |
| `SANDBOX_PACKAGES_DIR` | `/var/sandbox-packages` | Preinstalled package dir inside the sandbox image |

**Isolation model:** sandbox containers see only their own tmpfs dirs (`/tmp`,
`/var/sandbox-packages`) plus the explicit bind mounts — the per-conversation
`/workspace` (only in `writable_workspace` mode) and the persistent
`/agent_ws` scope dir (when the assistant workspace is enabled). No host
filesystem, no `docker.sock`, no API secrets are exposed inside the sandbox.
The per-container `SANDBOX_TOKEN` guards the exec endpoint and is scrubbed
from `os.environ` at server start so generated code cannot read it
(residual: `/proc/self/environ` still carries the original env). Containers
run as `sandboxuser` (uid/gid ≠ 0; `writable_root` excepted), with
`cap_drop=ALL`, `no-new-privileges`, `--init` zombie reaping, a 512-process
pid limit, memory/CPU caps, `RLIMIT_NOFILE`/`RLIMIT_FSIZE`, `ip_forward=0`,
`dns_search=["."]` (host search domains are not leaked) and bounded
json-file logs (10m×3).

**Egress policy** is enforced packet-level inside the sandbox netns by a
one-shot `NET_ADMIN` sidecar sharing the container's network namespace —
the sandbox itself runs `cap_drop=ALL` and cannot edit the rules, even in
`writable_root` mode. Policy comes from the `code_execution` tool config
(per-assistant `tools:` overrides apply):

| Key | Default | Effect |
|---|---|---|
| `egress_internet` | `true` | Allow outbound to public IPs. `false` = fully isolated (also drops the embedded-DNS resolver → no DNS-tunnel exfil; pip installs fail fast) |
| `egress_local_network` | `false` | Allow private/link-local/CGNAT ranges (LAN, sibling sandboxes on the bridge). The cloud-metadata IP `169.254.169.254` stays blocked regardless |
| `egress_blocked` | `[]` | Additional IPs/CIDRs always dropped — union with `SANDBOX_EGRESS_BLOCKED_CIDRS` |

The `AGENTIX_EGRESS` OUTPUT chain allows loopback + `ESTABLISHED` traffic
(so the app can always reach the executor), drops the metadata IP and all
blocked/LAN ranges, then drops everything else when `internet=false`.
Changing the policy recreates the container on next use (label-compared).
Residual: with `internet=true`, DNS resolution keeps working, so
DNS-tunnel exfiltration remains theoretically possible — full coverage
there needs a filtering resolver or forward proxy.

Pip installs use `--target /var/sandbox-packages`
(an exec-capable tmpfs) with `--prefer-binary`; `importlib.invalidate_caches()`
runs after every install so freshly installed packages import immediately.
The in-container server is threaded: a hung `/exec` cannot starve `/install`,
`/files` or later `/exec` calls, execs are serialized (shared globals/stdout),
and a per-exec watchdog interrupts overruns best-effort. Long-running tools
emit periodic keepalive progress events that keep the SSE wire and the
dashboard progress display alive — they are flagged as keepalives and do NOT
reset the run idle watchdog, so a stuck tool is still aborted after
`RUN_IDLE_TIMEOUT_SECONDS` without real activity. Workspace files are made world-writable
(`0o666`/`0o777`) since the sandbox runs as an unprivileged uid — the
`/agent_ws` mount is probed for writability, not just visibility, at
container start.
Bind-mount sources are validated: shallow or system-owned roots
(`AGENT_WORKSPACE_ROOT=/` etc.) are refused rather than mounted, and each
fresh container probes that `/agent_ws` really shows the workspace dir —
mismatches surface as a `mount_warning` in the tool output and in logs.
Note that `workspace.scope` chooses sharing granularity: `assistant` shares
one dir across **all** users and conversations, so sandbox code of any run
can read files other users placed there — use `conversation` or `user`
scope to keep data separated.

## Misc tool & runtime variables

| Variable | Default | Description |
|---|---|---|
| `SERPAPI_API_KEY` | — | Key for the `serpapi_search` tool |
| `LIBREOFFICE_PATH` / `SOFFICE_PATH` | auto-detect | LibreOffice binary overrides (`view`, `document_generator`, preview rendering) |
| `HF_TOKEN` / `HF_CACHE_DIR` | — / default HF cache | HuggingFace auth/cache for emotion + Whisper model downloads |
| `WIDGET_DEFAULTS_CORS_ORIGINS` | `*` | Allowed origins for `GET /widget-defaults` |

## WebDAV knowledge sources

| Variable | Notes |
|---|---|
| `WEBDAV_ENCRYPTION_KEY` | Fernet key (32 bytes, urlsafe base64) encrypting stored WebDAV source passwords. Generate: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |

> [!warning]
> Without `WEBDAV_ENCRYPTION_KEY` a **weak development fallback** encrypts stored WebDAV passwords — always set it in production.

## Database backup

| Variable | Default | Notes |
|---|---|---|
| `BACKUP_ENCRYPTION_KEY` | — | **Required for backups.** AES-256-GCM key (64 hex chars or 32-byte base64). Fails closed when unset. Keep it outside the database and outside the backup target. |
| `BACKUP_TMP_DIR` | `data/backups/tmp` | Scratch space for dumps/encryption (auto-cleaned) |
| `BACKUP_DUMP_TIMEOUT` | `1800` | Seconds for `pg_dump`/`pg_restore` |
| `BACKUP_UPLOAD_TIMEOUT` | `600` | Seconds per WebDAV request |
| `BACKUP_UPLOAD_RETRIES` | `3` | Upload attempts with backoff |
| `BACKUP_INSTANCE` | — | Instance label embedded in backup filenames |

Schedule, target URL, credentials and retention are configured **at runtime** in the dashboard (Backups page) — they persist in `agentix_backup_config`. See [Backup & Recovery](backup-and-recovery.md).

## Audio / STT / TTS

`AUDIO_STT_*` (faster-whisper), `AUDIO_TTS_*` (Kokoro/Piper: voices dir, model dir, default voice, language, streaming). Defaults are sensible for German (`de`).

## Emotion detection

`EMOTION_DETECTION_ENABLED`, `EMOTION_WEBCAM_*` (ONNX models, execution providers, sampling), `EMOTION_AUDIO_*`, `EMOTION_TEXT_*`, `EMOTION_FUSION_*` — see `agent/config.py` lines ~294–326 for the full set with defaults.

## OCR

| Variable | Default |
|---|---|
| `TESSERACT_CMD` | auto-detect |
| `OCR_LANG` | `deu` |

## Builtin tool parameters

`rag_search` (actions `search`/`list`/`read`) and `collection_manager` (actions `create`/`delete`/`list`/`read`/`edit`) share the same paging/navigation arguments; all tool output is capped at 15,000 characters per call and ends with a footer naming the actual range plus the next offset/lines_start to continue with.

| Parameter | Applies to | Meaning |
|---|---|---|
| `limit` | search, list, read | Max characters of output (default/max 15000) |
| `offset` | search, list, read | 0-based char offset; output snaps *back* to the containing line's start so pages never split words — the footer reports the real range |
| `lines_start` / `lines_end` | search, list, read | 1-based inclusive line bounds; take priority over offset/limit |
| `pattern` | read, list | Case-insensitive substring. On `read`: numbered match windows ± `context_lines` (locate content instead of paging). On `list`: filter entries by path/filename |
| `context_lines` | read + pattern | Context lines around each match (default 2, max 10) |
| `include_hidden` | rag_search list | Show dotfile paths like `.trash-*/` (default false) |
| `folder`, `depth` | rag_search list | Folder scoping / tree collapsing for large collections |

`read` output is always line-numbered (`NNNN│`); the same numbers feed `collection_manager edit` (`line=`, `lines_start`/`lines_end`). Text-based files stored in the raw file store are returned byte-exact (`type: raw`); binary/other documents are reassembled from index chunks (`type: reconstructed`).
