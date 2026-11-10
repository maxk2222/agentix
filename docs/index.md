<div align="center">
  <img src="assets/logo.png" alt="Agentix Agent Platform" width="180" />
  <h1>Agentix Agent Platform — Documentation</h1>
  <p><em>OpenAI-compatible RAG agent platform for cybersecurity training and advisory scenarios.</em></p>
</div>

---

## Documentation map

| Audience | Start here |
|---|---|
| Operators / DevOps | [Installation](installation.md) → [Configuration](configuration.md) → [Deployment](deployment.md) → [Backup & Recovery](backup-and-recovery.md) |
| Administrators | [Administration](administration.md) → [Security](security.md) → [Troubleshooting](troubleshooting.md) |
| Integrators | [API Reference](api.md) → [Widget Integration](widget.md) |
| Agent builders | [Agent runtime](agent.md) → [Tools](tools.md) → [Configuration](configuration.md) |
| Developers | [Development](development.md) → [Architecture](architecture.md) |

## All documents

| Document | Contents |
|---|---|
| [Overview](overview.md) | Purpose, feature scope, project context |
| [Architecture](architecture.md) | Components, data flow, startup lifecycle, persistence model |
| [Agent runtime](agent.md) | Agent loop, tool modes, dedup, subagents, observer, steering, workspace, artifacts, MCP |
| [Tools](tools.md) | Built-in tool reference, configuration requirements, MCP servers |
| [Installation](installation.md) | Requirements, local setup, Docker Compose |
| [Configuration](configuration.md) | Environment variables verified against `agent/config.py` |
| [Administration](administration.md) | Dashboard, roles & permissions, runtime config |
| [Backup & Recovery](backup-and-recovery.md) | Encrypted Sciebo/WebDAV backups, restore runbook |
| [API Reference](api.md) | `/v1/*` OpenAI-compatible endpoints, run control, files, workspace, admin API |
| [Widget Integration](widget.md) | Embedding the chat widget, configurator |
| [Development](development.md) | Dev setup, tests, builds, commit conventions |
| [Security](security.md) | AuthN/AuthZ, secrets handling, request validation |
| [Troubleshooting](troubleshooting.md) | Common failure modes and diagnostics |
| [Deployment](deployment.md) | Dockerfile, Compose, GitLab CI flow |
| [Roadmap](roadmap.md) | Known limitations and planned work |

## Conventions

- **Product name:** "Agentix Agent Platform" for prose; internal identifiers (`agentix_*` tables, `AGENTIX_*` headers) stay unchanged.
- **Table of contents:** long pages start with `[[_TOC_]]` (rendered natively by GitLab).
- **Alert boxes:** GitLab alert syntax — `> [!note]` neutral info, `> [!tip]` best practice, `> [!important]` must-know, `> [!warning]` data/security risk, `> [!caution]` destructive action. On older renderers they degrade gracefully to blockquotes.
- **Diagrams:** Mermaid inline (rendered natively by GitLab) — `flowchart` for pipelines/topologies, `sequenceDiagram` for call flows.
- **Screenshots:** real captures under `docs/assets/screenshots/<page>-<theme>.png` (2560×1440 viewport, light + dark, dashboard + public chat). Walkthrough GIFs under `docs/assets/walkthroughs/<scenario>.gif` (1920px, 12fps, segment-recorded so no loading splash ships). Regenerate everything with `AGENTIX_PASSWORD='…' docs/update_screenshots.sh` (engine: `tests/e2e/capture_media.py`) against a running instance — never paste fabricated images.
- **Colour palette** (from `dashboard/src/index.css`): primary/accent `#1d4e89` (light) / `#6b9bd4` (dark), success `#059669`, warning `#d97706`, danger `#dc2626`, dark background `#0c1420`.
