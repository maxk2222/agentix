# Widget Integration

The chat widget (`widget/`) is a standalone TypeScript bundle that embeds a chat UI into any website. It talks to the platform's **own OpenAI-compatible API** (`/v1/chat/completions`) — **not** to OpenAI directly.

> **Correction for older docs:** no OpenAI account or `sk-…` key is involved. `apiKey` refers to a **platform API key** created in the dashboard (`/admin/api-keys`), and `apiUrl` points at the platform backend (default: same origin + `/v1`).

## Build

```bash
cd widget
npm ci
npm run build      # → widget/dist/widget.umd.js (+ assets)
```

The backend serves `widget/dist` at `/widget` — when deploying with the Docker image this is already wired.

## Embed

```html
<script src="https://your-host/widget/widget.umd.js"></script>
<script>
  window.initAgentixChat({
    apiKey: '<platform-api-key>',   // from Dashboard → API keys
    apiUrl: 'https://your-host/v1', // optional; defaults to <origin>/v1
    position: 'bottom-right',
    theme: 'dark',                  // 'light' | 'dark' | 'auto'
    language: 'auto',               // 'de' | 'en' | 'auto'
  });
</script>
```

Key config fields (`widget/src/types/index.ts` → `ChatWidgetConfig`):

| Field | Default | Notes |
|---|---|---|
| `apiKey` | — | Platform API key (required for chat) |
| `apiUrl` | `<origin>/v1` | Backend base URL; the widget also fetches `/widget-defaults` from the backend for theme/defaults |
| `position` | `bottom-right` | Button placement |
| `theme`, `language` | `auto` | UI theme / UI language |
| assistant/model | dashboard default | Selectable per widget config |

## Configurator

Dashboard → Configuration → *Widget & Chat* provides a visual configurator that generates the embed snippet and manages `/widget-defaults` (colors, logo, position, greeting) plus the public-chat configuration.

### Server-side defaults

The configurator's **"Konfiguration deployen"** button (roles `admin` + `owner`, permission `widget_defaults:write`) stores the current settings as server-side defaults (`POST /admin/widget-defaults`). Embedded widgets fetch them at runtime from `GET /widget-defaults` (no credentials), so deployed defaults apply on the next page load.

Merge order: **explicit embed config > server defaults > built-in defaults**. Secrets (`apiKey`, callbacks) are stripped on save and never served publicly.

Cross-origin embeds need `WIDGET_DEFAULTS_CORS_ORIGINS` (default `*`) on the backend.

## Security notes

- The API key is embedded in client-side code — use a dedicated, least-privilege key and rotate it from the dashboard.
- XSS is mitigated by input sanitization and Markdown rendering rules inside the widget.
- Prefer serving the widget same-origin (avoids CORS); cross-origin requires the backend CORS setup to allow it.
