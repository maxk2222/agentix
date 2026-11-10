"""Model-aware reasoning parameter mapping.

Translates a normalized assistant ``reasoning`` config into provider- and
model-family-specific wire parameters that are merged into ``extra_body``.

Canonical reasoning keys (what users/admins configure):

    reasoning:
      effort:  low|medium|high|xhigh|minimal|none   # effort-style families
      enabled: true|false                           # thinking on/off switch
      budget:  <int>                                # token budget families
      level:   low|high|...                         # level-style families (Gemini 3)

Families are matched by regex against the model name, with the resolved
provider base URL as a secondary hint. Additional families can be injected at
runtime via the ``REASONING_PARAM_RULES_JSON`` environment variable, which
accepts either a JSON array/object of rules or a path to a JSON file with the
same structure. Runtime rules are prepended and therefore take precedence over
the built-in defaults.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CANONICAL_KEYS = frozenset({"effort", "enabled", "budget", "level"})

ENV_RULES_VAR = "REASONING_PARAM_RULES_JSON"

# ---------------------------------------------------------------------------
# Family rules — order matters: first match wins, generic fallback last.
#
# Each rule:
#   family      unique name, surfaced in logs and the schema endpoint
#   patterns    regex list matched (case-insensitive) against the model name
#   url_hints   optional regex list matched against provider base_url
#   wire        translation strategy key used by translate_to_wire()
#   controls    UI descriptor returned by describe_for_model()
# ---------------------------------------------------------------------------

REASONING_FAMILIES: List[Dict[str, Any]] = [
    {
        # Newer Qwen generations (3.5/3.8/3.9/Next/4) accept top-level
        # ``reasoning_effort`` — their chat templates validate the value, and
        # Qwen3.8 only accepts low|medium|xhigh. Plain ``qwen-flash`` etc.
        # deliberately stay on qwen_thinking (DashScope supports the toggle,
        # not reasoning_effort).
        "family": "qwen3_effort",
        "patterns": [
            r"qwen3\.?[589]",
            r"qwen3[._-]?next",
            r"qwen4",
        ],
        "wire": "qwen3_effort",
        "controls": [
            {"key": "enabled", "type": "toggle", "default": True},
            {"key": "effort", "type": "select", "options": ["low", "medium", "xhigh"]},
            {"key": "budget", "type": "number", "min": 1, "max": 131072},
        ],
    },
    {
        "family": "qwen_thinking",
        "patterns": [r"qwen3", r"qwq"],
        "wire": "qwen_thinking",
        "controls": [
            {"key": "enabled", "type": "toggle", "default": True},
            {"key": "budget", "type": "number", "min": 1, "max": 131072},
        ],
    },
    {
        "family": "chat_template_toggle",
        "patterns": [r"deepseek", r"granite", r"holo", r"nemotron"],
        "wire": "chat_template_thinking",
        "controls": [{"key": "enabled", "type": "toggle", "default": True}],
    },
    {
        "family": "chat_template_enable_thinking",
        "patterns": [r"gemma"],
        "wire": "chat_template_enable_thinking",
        "controls": [{"key": "enabled", "type": "toggle", "default": False}],
    },
    {
        "family": "claude_budget",
        "patterns": [r"claude"],
        "url_hints": [r"anthropic"],
        "wire": "claude_thinking",
        "controls": [
            {"key": "enabled", "type": "toggle", "default": True},
            {"key": "budget", "type": "number", "min": 1024, "max": 200000},
        ],
    },
    {
        "family": "gemini_budget",
        "patterns": [r"gemini"],
        "url_hints": [r"generativelanguage", r"googleapis"],
        "wire": "gemini_thinking",
        "controls": [
            {"key": "level", "type": "select", "options": ["low", "high"]},
            {"key": "budget", "type": "number", "min": -1, "max": 32768},
        ],
    },
    {
        "family": "openai_effort",
        "patterns": [r"(?:^|[/_.-])o[0-9]", r"gpt-5", r"gpt-6"],
        "wire": "openai_effort",
        "controls": [
            {
                "key": "effort",
                "type": "select",
                "options": ["minimal", "low", "medium", "high", "xhigh"],
            },
        ],
    },
    {
        "family": "generic",
        "patterns": [],
        "wire": "generic",
        "assumed": True,
        "controls": [
            {
                "key": "effort",
                "type": "select",
                "options": ["low", "medium", "high"],
            },
        ],
    },
]

GENERIC_RULE = REASONING_FAMILIES[-1]

_OVERRIDE_RULES: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Rule loading / matching
# ---------------------------------------------------------------------------

def _validate_rule(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    family = raw.get("family")
    wire = raw.get("wire")
    if not family or not isinstance(family, str):
        return None
    known_wires = {
        "qwen3_effort",
        "probed_effort",
        "drop",
        "qwen_thinking",
        "chat_template_thinking",
        "chat_template_enable_thinking",
        "claude_thinking",
        "gemini_thinking",
        "openai_effort",
        "generic",
    }
    if wire not in known_wires:
        logger.warning("reasoning rule %r has unknown wire format %r; ignored", family, wire)
        return None
    patterns = [p for p in raw.get("patterns") or [] if isinstance(p, str)]
    url_hints = [p for p in raw.get("url_hints") or [] if isinstance(p, str)]
    controls = raw.get("controls")
    if not isinstance(controls, list):
        controls = []
    return {
        "family": family,
        "patterns": patterns,
        "url_hints": url_hints,
        "wire": wire,
        "controls": controls,
        "assumed": bool(raw.get("assumed", False)),
    }


def load_override_rules() -> List[Dict[str, Any]]:
    """Parse ``REASONING_PARAM_RULES_JSON`` (JSON text or file path)."""
    global _OVERRIDE_RULES
    raw = os.environ.get(ENV_RULES_VAR, "").strip()
    if not raw:
        _OVERRIDE_RULES = []
        return []
    text = raw
    if not raw.startswith("[") and not raw.startswith("{"):
        try:
            text = Path(raw).read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read %s from %r: %s", ENV_RULES_VAR, raw, exc)
            _OVERRIDE_RULES = []
            return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON in %s: %s", ENV_RULES_VAR, exc)
        _OVERRIDE_RULES = []
        return []
    items = data.get("rules") if isinstance(data, dict) else data
    if not isinstance(items, list):
        logger.warning("%s must contain a rules array; ignored", ENV_RULES_VAR)
        _OVERRIDE_RULES = []
        return []
    rules = [r for r in (_validate_rule(item) for item in items) if r]
    logger.info("Loaded %d reasoning parameter override rule(s) from %s", len(rules), ENV_RULES_VAR)
    _OVERRIDE_RULES = rules
    return rules


def _all_rules() -> List[Dict[str, Any]]:
    global _OVERRIDE_RULES
    if _OVERRIDE_RULES is None:
        load_override_rules()
    return (_OVERRIDE_RULES or []) + REASONING_FAMILIES


def reset_override_rules() -> None:
    """Clear the cached runtime rules (used by tests)."""
    global _OVERRIDE_RULES
    _OVERRIDE_RULES = None


def match_family(
    model_name: Optional[str], provider_base_url: Optional[str] = None
) -> Dict[str, Any]:
    """Return the first family rule matching model name or provider URL.

    Model-name patterns take precedence over provider URL hints — a model
    named ``gpt-5`` routed through an Anthropic-compatible gateway is still
    treated as an OpenAI-effort model.
    """
    name = (model_name or "").lower()
    url = (provider_base_url or "").lower()
    rules = _all_rules()
    for rule in rules:
        for pattern in rule.get("patterns", []):
            try:
                if re.search(pattern, name):
                    return rule
            except re.error:
                logger.warning("Bad reasoning rule pattern %r in family %r", pattern, rule.get("family"))
    if url:
        for rule in rules:
            for hint in rule.get("url_hints", []):
                try:
                    if re.search(hint, url):
                        return rule
                except re.error:
                    logger.warning(
                        "Bad reasoning rule url_hint %r in family %r", hint, rule.get("family")
                    )
    return GENERIC_RULE


# ---------------------------------------------------------------------------
# Normalization (config in) and translation (wire out)
# ---------------------------------------------------------------------------

def normalize_reasoning_dict(raw: Any) -> Optional[Dict[str, Any]]:
    """Whitelist canonical keys and coerce value types.

    Unknown keys are stripped with a warning so nothing unexpected reaches the
    wire. Returns ``None`` for empty/invalid input.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        logger.warning("reasoning config must be a mapping; got %r", type(raw).__name__)
        return None
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if key not in CANONICAL_KEYS:
            logger.warning("reasoning: unknown key %r ignored", key)
            continue
        if key in ("effort", "level"):
            if isinstance(value, str) and value:
                out[key] = value
            else:
                logger.warning("reasoning.%s expects a non-empty string; got %r", key, value)
        elif key == "enabled":
            if isinstance(value, bool):
                out[key] = value
            else:
                logger.warning("reasoning.enabled expects a boolean; got %r", value)
        elif key == "budget":
            if isinstance(value, bool):
                logger.warning("reasoning.budget expects an integer; got %r", value)
            else:
                try:
                    out[key] = int(value)
                except (TypeError, ValueError):
                    logger.warning("reasoning.budget expects an integer; got %r", value)
    return out or None


def _warn_ignored(reasoning: Dict[str, Any], supported: set, family: str) -> None:
    for key in reasoning:
        if key not in supported:
            logger.info(
                "reasoning.%s is not supported by family %s; ignored", key, family
            )


def translate_to_wire(
    reasoning: Optional[Dict[str, Any]], rule: Dict[str, Any]
) -> Dict[str, Any]:
    """Map a canonical reasoning dict to provider-specific extra_body keys.

    Returns a dict to be deep-merged into ``extra_body``. Unknown family wire
    formats fall back to the generic verbatim ``reasoning`` passthrough.
    """
    if not reasoning:
        return {}
    wire = rule.get("wire")

    if wire == "drop":
        # A live probe rejected every reasoning control for this model — send
        # nothing rather than parameters known to fail.
        _warn_ignored(reasoning, set(), rule["family"])
        return {}

    if wire == "openai_effort":
        _warn_ignored(reasoning, {"effort"}, rule["family"])
        if reasoning.get("effort"):
            return {"reasoning": {"effort": reasoning["effort"]}}
        return {}

    if wire == "probed_effort":
        # Live probe confirmed ``reasoning_effort`` but NOT the thinking toggle
        # — never send template kwargs a probed model rejected.
        _warn_ignored(reasoning, {"effort"}, rule["family"])
        if reasoning.get("effort"):
            return {"reasoning_effort": reasoning["effort"]}
        return {}

    if wire == "qwen3_effort":
        # Newer Qwen/vLLM-served models: top-level ``reasoning_effort`` plus
        # optional chat-template knobs. An explicit budget bounds thinking the
        # same way as on the plain ``qwen_thinking`` family.
        _warn_ignored(reasoning, {"enabled", "effort", "budget"}, rule["family"])
        out: Dict[str, Any] = {}
        if reasoning.get("effort"):
            out["reasoning_effort"] = reasoning["effort"]
        kwargs: Dict[str, Any] = {}
        if reasoning.get("enabled") is not None:
            kwargs["enable_thinking"] = bool(reasoning["enabled"])
        if reasoning.get("budget") is not None:
            kwargs["thinking_budget"] = int(reasoning["budget"])
        if kwargs:
            out["chat_template_kwargs"] = kwargs
        return out

    if wire == "qwen_thinking":
        _warn_ignored(reasoning, {"enabled", "budget"}, rule["family"])
        kwargs: Dict[str, Any] = {}
        if reasoning.get("enabled") is not None:
            kwargs["enable_thinking"] = bool(reasoning["enabled"])
        if reasoning.get("budget") is not None:
            kwargs["thinking_budget"] = int(reasoning["budget"])
        return {"chat_template_kwargs": kwargs} if kwargs else {}

    if wire in ("chat_template_thinking", "chat_template_enable_thinking"):
        _warn_ignored(reasoning, {"enabled"}, rule["family"])
        if reasoning.get("enabled") is None:
            return {}
        param = "thinking" if wire == "chat_template_thinking" else "enable_thinking"
        return {"chat_template_kwargs": {param: bool(reasoning["enabled"])}}

    if wire == "claude_thinking":
        _warn_ignored(reasoning, {"enabled", "budget"}, rule["family"])
        if reasoning.get("enabled") is False:
            return {"thinking": {"type": "disabled"}}
        thinking: Dict[str, Any] = {"type": "enabled"}
        if reasoning.get("budget") is not None:
            thinking["budget_tokens"] = int(reasoning["budget"])
        elif reasoning.get("enabled") is None:
            # Neither an explicit toggle nor a budget was configured — send nothing.
            return {}
        return {"thinking": thinking}

    if wire == "gemini_thinking":
        _warn_ignored(reasoning, {"enabled", "budget", "level"}, rule["family"])
        cfg: Dict[str, Any] = {}
        if reasoning.get("enabled") is False:
            cfg["thinkingBudget"] = 0
        else:
            if reasoning.get("budget") is not None:
                cfg["thinkingBudget"] = int(reasoning["budget"])
            if reasoning.get("level"):
                cfg["thinkingLevel"] = reasoning["level"]
        return {"thinkingConfig": cfg} if cfg else {}

    # generic / unknown: preserve the historical verbatim passthrough so
    # existing configs on unrecognized models behave exactly as before.
    return {"reasoning": copy.deepcopy(reasoning)}


def deep_merge_extra_body(
    base: Optional[Dict[str, Any]], wire: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Deep-merge ``wire`` into ``base``; wire wins on per-key conflicts."""
    if not wire:
        return dict(base) if base else {}
    result = dict(base) if base else {}
    for key, value in wire.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = deep_merge_extra_body(existing, value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Frontend descriptor
# ---------------------------------------------------------------------------

def describe_for_model(
    model_name: Optional[str], provider_base_url: Optional[str] = None
) -> Dict[str, Any]:
    """Return a serializable capability descriptor for the config UI."""
    rule = match_family(model_name, provider_base_url)
    return {
        "family": rule["family"],
        "wire": rule["wire"],
        "controls": copy.deepcopy(rule["controls"]),
        "assumed": bool(rule.get("assumed", False)),
        "probed": False,
        "canonical_keys": sorted(CANONICAL_KEYS),
        "model": model_name or "",
    }


# Ordered effort ladder used to normalize probed capability reports.
KNOWN_EFFORT_LADDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT_LEVELS = ["low", "medium", "high"]


# Families whose wire format is expressed purely in the two dimensions the
# live probe measures (top-level ``reasoning_effort`` +
# ``chat_template_kwargs.enable_thinking``). Only those — plus the assumed
# generic fallback — may be superseded by a probe result; richer families
# (claude ``thinking.type``, gemini ``thinkingConfig``, the ``thinking``
# kwarg, openai's ``reasoning`` object) stay authoritative.
PROBE_OVERRIDABLE_WIRES = frozenset(
    {
        "generic",
        "qwen_thinking",
        "qwen3_effort",
        "probed_effort",
        "chat_template_enable_thinking",
    }
)


def probe_overrides_family(rule: Dict[str, Any]) -> bool:
    """Whether a live probe result may supersede a name-matched family rule."""
    return bool(rule.get("assumed")) or rule.get("wire") in PROBE_OVERRIDABLE_WIRES


def rule_from_capabilities(
    caps: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build a ``translate_to_wire`` rule dict from live-probe capabilities.

    ``None`` when no probe data exists. A probe that rejected BOTH controls
    yields the ``"drop"`` wire so stale configs cannot send parameters the
    backend is known to refuse.
    """
    if not caps:
        return None
    supports_effort = bool(caps.get("supports_effort"))
    supports_toggle = bool(caps.get("supports_thinking_toggle"))
    if supports_effort:
        wire = "qwen3_effort" if supports_toggle else "probed_effort"
    elif supports_toggle:
        wire = "qwen_thinking"
    else:
        wire = "drop"
    return {"family": "probed", "wire": wire, "controls": []}


def normalize_effort_levels(levels: Any) -> List[str]:
    """Filter/order probed effort levels against the known ladder."""
    if not isinstance(levels, (list, tuple)):
        return []
    seen = {str(level).strip().lower() for level in levels if str(level).strip()}
    return [level for level in KNOWN_EFFORT_LADDER if level in seen]


def descriptor_from_capabilities(
    caps: Optional[Dict[str, Any]], model_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Build a UI descriptor from a live capability probe result.

    Returns ``None`` only when no probe data exists (``caps`` falsy) — the
    caller then falls back to rule matching. A probe that ran and found no
    supported controls yields a descriptor with empty ``controls`` so the UI
    can say "no reasoning options" instead of offering disproved controls.
    """
    if not caps:
        return None
    supports_effort = bool(caps.get("supports_effort"))
    supports_toggle = bool(caps.get("supports_thinking_toggle"))
    controls: List[Dict[str, Any]] = []
    if supports_toggle:
        controls.append({"key": "enabled", "type": "toggle", "default": True})
    if supports_effort:
        levels = normalize_effort_levels(caps.get("effort_levels"))
        controls.append(
            {
                "key": "effort",
                "type": "select",
                "options": levels or list(DEFAULT_EFFORT_LEVELS),
            }
        )
    if supports_toggle:
        controls.append({"key": "budget", "type": "number", "min": 1, "max": 131072})
    rule = rule_from_capabilities(caps)
    return {
        "family": "probed",
        "wire": rule["wire"] if rule else "generic",
        "controls": controls,
        "assumed": False,
        "probed": True,
        "canonical_keys": sorted(CANONICAL_KEYS),
        "model": model_name or "",
    }
