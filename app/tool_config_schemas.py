"""Declarative configuration schemas for built-in tools.

Each schema is a list of field descriptors consumed by the dashboard to
render a settings form instead of a raw-JSON editor. Fields:

  key               config key (flat, or dotted for nested e.g. "theme.accent")
  type              text | password | number | select | toggle | textarea |
                    json | string_list | cidr_list | collections | color |
                    mcp_servers
  label             English fallback label (i18n keys live under
                    toolsCfg.<tool>.<key-suffix>)
  default           value assumed when the key is absent from config
  secret            masked in GET responses; the __agentix_secret_set__
                    sentinel keeps the stored value on PUT, "" clears it
  required_for_use  empty value renders a "needs config" badge
  options           list of {value,label} for type=select
  min / max         numeric bounds for type=number
  placeholder       input placeholder
  help              short description rendered under the field
  group             optional section title for visual grouping

Only keys the builtin actually reads at runtime belong here — schemas are
the contract between the tool implementation and the dashboard form.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Sentinel returned for non-empty secret values in GET responses. On PUT the
# router keeps the stored value when it sees the sentinel; an explicit ""
# clears the secret.
SECRET_SENTINEL = "__agentix_secret_set__"

BUILTIN_CONFIG_SCHEMAS: Dict[str, List[Dict[str, Any]]] = {
    "code_execution": [
        {
            "key": "sandbox_mode",
            "type": "select",
            "label": "Sandbox mode",
            "default": "read_only",
            "options": [
                {"value": "read_only", "label": "read_only — read-only fs (recommended)"},
                {"value": "writable_user", "label": "writable_user — writable fs, uid sandboxuser"},
                {"value": "writable_workspace", "label": "writable_workspace — + per-conversation /workspace"},
                {"value": "writable_root", "label": "writable_root — root (least safe)"},
            ],
            "help": "Filesystem/identity mode of the execution container.",
            "group": "Sandbox",
        },
        {
            "key": "egress_internet",
            "type": "toggle",
            "label": "Internet access",
            "default": True,
            "help": "Allow outbound connections to public IPs. Off = fully isolated (pip installs fail).",
            "group": "Egress policy",
        },
        {
            "key": "egress_local_network",
            "type": "toggle",
            "label": "Local network access",
            "default": False,
            "help": "Allow private/link-local/CGNAT ranges (LAN + other sandboxes). Higher risk — enables lateral movement and metadata-style targets.",
            "group": "Egress policy",
        },
        {
            "key": "egress_blocked",
            "type": "cidr_list",
            "label": "Blocked IP ranges",
            "default": [],
            "placeholder": "10.0.0.0/8",
            "help": "Additional IPs/CIDRs the sandbox can never reach, e.g. company-internal ranges. Union with the SANDBOX_EGRESS_BLOCKED_CIDRS env floor.",
            "group": "Egress policy",
        },
    ],
    "web_search": [
        {
            "key": "engine",
            "type": "select",
            "label": "Engine",
            "default": "auto",
            "options": [
                {"value": "auto", "label": "auto"},
                {"value": "ddg", "label": "ddg — DuckDuckGo"},
                {"value": "serpapi", "label": "serpapi — Google via SerpAPI"},
            ],
        },
        {
            "key": "region",
            "type": "text",
            "label": "Region",
            "default": "",
            "placeholder": "de-de",
            "help": "Optional region hint passed to the engine (e.g. de-de, wt-wt).",
        },
        {
            "key": "safe_search",
            "type": "toggle",
            "label": "Safe search",
            "default": True,
        },
    ],
    "serpapi_search": [
        {
            "key": "api_key",
            "type": "password",
            "label": "SerpAPI key",
            "default": "",
            "secret": True,
            "required_for_use": True,
            "help": "Fallback: SERPAPI_API_KEY environment variable.",
        },
    ],
    "openalex_search": [
        {
            "key": "api_key",
            "type": "password",
            "label": "API key",
            "default": "",
            "secret": True,
            "help": "Optional premium key — raises rate limits.",
        },
        {
            "key": "mailto",
            "type": "text",
            "label": "mailto (polite pool)",
            "default": "",
            "placeholder": "you@example.org",
            "help": "OpenAlex asks clients to identify themselves — gets you into the faster 'polite pool'.",
        },
    ],
    "arxiv_search": [],
    "wikipedia_search": [],
    "web_scraper": [],
    "rss_feed_reader": [],
    "long_document_summarizer": [],
    "rag_search": [
        {
            "key": "collection_names",
            "type": "collections",
            "label": "Collections",
            "default": [],
            "help": "Restrict retrieval to these collections. Empty = all readable collections.",
        },
        {
            "key": "max_top_k",
            "type": "number",
            "label": "Max top-k",
            "default": 10,
            "min": 1,
            "max": 50,
        },
    ],
    "collection_manager": [
        {
            "key": "collection_names",
            "type": "collections",
            "label": "Collections",
            "default": [],
            "help": "Collections this tool may manage. Empty = all collections.",
        },
        {
            "key": "index_debounce_seconds",
            "type": "number",
            "label": "Index debounce (s)",
            "default": 60,
            "min": 0,
            "max": 3600,
            "help": "Delay before re-indexing after uploads so batch writes coalesce.",
        },
    ],
    "agent_memory": [],
    "agent_todo": [
        {
            "key": "max_tasks",
            "type": "number",
            "label": "Max tasks",
            "default": 20,
            "min": 1,
            "max": 50,
            "help": "Upper bound on todo-list size for this tool.",
        },
    ],
    "plan_generator": [],
    "wait": [
        {
            "key": "max_seconds",
            "type": "number",
            "label": "Max wait (s)",
            "default": 120,
            "min": 0.1,
            "max": 120,
            "help": "Per-call cap on the wait duration.",
        },
    ],
    "document_generator": [
        {
            "key": "theme.accent",
            "type": "color",
            "label": "Accent",
            "default": "#0A66C2",
            "group": "Theme",
        },
        {
            "key": "theme.accent_dark",
            "type": "color",
            "label": "Accent dark",
            "default": "#084B96",
            "group": "Theme",
        },
        {
            "key": "theme.accent_light",
            "type": "color",
            "label": "Accent light",
            "default": "#D6E8F9",
            "group": "Theme",
        },
        {
            "key": "theme.heading_font",
            "type": "text",
            "label": "Heading font",
            "default": "",
            "placeholder": "Calibri Light",
            "group": "Theme",
        },
        {
            "key": "theme.body_font",
            "type": "text",
            "label": "Body font",
            "default": "",
            "placeholder": "Calibri",
            "group": "Theme",
        },
    ],
    "email_sender": [
        {
            "key": "smtp_host",
            "type": "text",
            "label": "SMTP host",
            "default": "",
            "required_for_use": True,
            "group": "SMTP",
        },
        {
            "key": "smtp_port",
            "type": "number",
            "label": "SMTP port",
            "default": 587,
            "min": 1,
            "max": 65535,
            "group": "SMTP",
        },
        {
            "key": "smtp_use_ssl",
            "type": "toggle",
            "label": "Use SSL",
            "default": False,
            "help": "SSL from connect (465) vs STARTTLS (587).",
            "group": "SMTP",
        },
        {
            "key": "smtp_user",
            "type": "text",
            "label": "Username",
            "default": "",
            "group": "SMTP",
        },
        {
            "key": "smtp_password",
            "type": "password",
            "label": "Password",
            "default": "",
            "secret": True,
            "group": "SMTP",
        },
        {
            "key": "smtp_from",
            "type": "text",
            "label": "From address",
            "default": "",
            "placeholder": "agent@example.org",
            "help": "Defaults to the SMTP username when empty.",
            "group": "SMTP",
        },
        {
            "key": "smtp_timeout",
            "type": "number",
            "label": "Timeout (s)",
            "default": 30,
            "min": 5,
            "max": 300,
            "group": "SMTP",
        },
    ],
    "email_reader": [
        {
            "key": "imap_host",
            "type": "text",
            "label": "IMAP host",
            "default": "",
            "required_for_use": True,
            "group": "IMAP",
        },
        {
            "key": "imap_port",
            "type": "number",
            "label": "IMAP port",
            "default": 993,
            "min": 1,
            "max": 65535,
            "group": "IMAP",
        },
        {
            "key": "imap_use_ssl",
            "type": "toggle",
            "label": "Use SSL",
            "default": True,
            "group": "IMAP",
        },
        {
            "key": "imap_user",
            "type": "text",
            "label": "Username",
            "default": "",
            "group": "IMAP",
        },
        {
            "key": "imap_password",
            "type": "password",
            "label": "Password",
            "default": "",
            "secret": True,
            "group": "IMAP",
        },
        {
            "key": "imap_timeout",
            "type": "number",
            "label": "Timeout (s)",
            "default": 30,
            "min": 5,
            "max": 300,
            "group": "IMAP",
        },
    ],
    "google_calendar": [
        {
            "key": "service_account_json",
            "type": "textarea",
            "label": "Service-account JSON",
            "default": "",
            "secret": True,
            "required_for_use": True,
            "help": "Full JSON key of a Google service account with Calendar access.",
            "group": "Google",
        },
        {
            "key": "calendar_id",
            "type": "text",
            "label": "Calendar ID",
            "default": "",
            "required_for_use": True,
            "placeholder": "primary",
            "group": "Google",
        },
        {
            "key": "default_timezone",
            "type": "text",
            "label": "Default timezone",
            "default": "Europe/Berlin",
            "group": "Google",
        },
    ],
    "ssh_exec": [
        {
            "key": "host",
            "type": "text",
            "label": "Host",
            "default": "",
            "required_for_use": True,
            "group": "Connection",
        },
        {
            "key": "port",
            "type": "number",
            "label": "Port",
            "default": 22,
            "min": 1,
            "max": 65535,
            "group": "Connection",
        },
        {
            "key": "username",
            "type": "text",
            "label": "Username",
            "default": "",
            "required_for_use": True,
            "group": "Connection",
        },
        {
            "key": "auth_type",
            "type": "select",
            "label": "Auth type",
            "default": "password",
            "options": [
                {"value": "password", "label": "password"},
                {"value": "key", "label": "key — private key"},
            ],
            "group": "Connection",
        },
        {
            "key": "password",
            "type": "password",
            "label": "Password",
            "default": "",
            "secret": True,
            "group": "Auth",
        },
        {
            "key": "private_key",
            "type": "textarea",
            "label": "Private key (PEM)",
            "default": "",
            "secret": True,
            "group": "Auth",
        },
        {
            "key": "key_passphrase",
            "type": "password",
            "label": "Key passphrase",
            "default": "",
            "secret": True,
            "group": "Auth",
        },
    ],
    "webhook_post": [
        {
            "key": "allowed_hosts",
            "type": "string_list",
            "label": "Allowed hosts",
            "default": [],
            "placeholder": "*.example.org",
            "help": "fnmatch globs on the URL hostname. Empty = any https host.",
        },
        {
            "key": "allow_private_ips",
            "type": "toggle",
            "label": "Allow private-IP URLs",
            "default": True,
            "help": "When off, literal private/link-local IPs are rejected (SSRF guard; hostname resolution is not checked).",
        },
    ],
    "weather_lookup": [
        {
            "key": "provider",
            "type": "select",
            "label": "Provider",
            "default": "open-meteo",
            "options": [
                {"value": "open-meteo", "label": "open-meteo — free, no key"},
                {"value": "openweathermap", "label": "openweathermap — needs api_key"},
            ],
        },
        {
            "key": "api_key",
            "type": "password",
            "label": "API key",
            "default": "",
            "secret": True,
        },
    ],
    "spawn_subagent": [],
    "wait_subagent": [],
    "subagent_status": [],
    "nudge_subagent": [],
    "assistant_caller": [],
    "agent_workspace": [],
    "mcp_servers": [
        {
            "key": "servers",
            "type": "mcp_servers",
            "label": "MCP servers",
            "default": [],
            "help": "External MCP servers whose tools are materialized as mcp__* builtins.",
        },
    ],
}


def schema_for(tool_name: str) -> List[Dict[str, Any]]:
    """Config schema for a builtin tool (empty list = no structured form)."""
    return BUILTIN_CONFIG_SCHEMAS.get(tool_name, [])


def secret_keys(tool_name: str) -> set:
    """Config keys flagged secret for a tool (dotted paths supported)."""
    return {f["key"] for f in schema_for(tool_name) if f.get("secret")}


def mask_config(tool_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Replace non-empty secret values with the sentinel for API responses."""
    keys = secret_keys(tool_name)
    if not keys or not isinstance(config, dict):
        return config
    masked = dict(config)
    for key in keys:
        parts = key.split(".")
        target = masked
        for part in parts[:-1]:
            nxt = target.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                target[part] = nxt
            else:
                nxt = dict(nxt)
                target[part] = nxt
            target = nxt
        leaf = parts[-1]
        if target.get(leaf):
            target[leaf] = SECRET_SENTINEL
    return masked


def merge_secret_preserves(
    tool_name: str, new_config: Dict[str, Any], old_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve secret sentinels in an update payload against stored config.

    Sentinel → keep the stored value; everything else is stored as sent
    ("" explicitly clears a secret).
    """
    keys = secret_keys(tool_name)
    if not keys or not isinstance(new_config, dict):
        return new_config
    merged = dict(new_config)
    for key in keys:
        parts = key.split(".")
        new_target = merged
        old_target = old_config if isinstance(old_config, dict) else {}
        for part in parts[:-1]:
            new_target = new_target.get(part)
            old_target = old_target.get(part)
            if not isinstance(new_target, dict) or not isinstance(old_target, dict):
                new_target = None
                break
        if new_target is None:
            continue
        leaf = parts[-1]
        if new_target.get(leaf) == SECRET_SENTINEL:
            new_target[leaf] = old_target.get(leaf, "")
    return merged


def validate_schema_config(tool_name: str, config: Dict[str, Any]) -> List[str]:
    """Validate typed fields with strict formats; returns error strings.

    Currently only ``cidr_list`` needs server-side validation — the frontend
    validates the rest, and unknown/legacy keys are always preserved.
    """
    import ipaddress

    errors: List[str] = []
    for field in schema_for(tool_name):
        if field["type"] != "cidr_list":
            continue
        value = config.get(field["key"])
        if value in (None, "", []):
            continue
        if not isinstance(value, list):
            value = [value]
        for item in value:
            try:
                ipaddress.ip_network(str(item).strip(), strict=False)
            except ValueError:
                errors.append(f"{field['key']}: invalid IP/CIDR '{item}'")
    return errors
