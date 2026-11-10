"""Tool argument validation and normalization for agent tools."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Central limits for tool arguments to prevent abuse or token overflow
_TOOL_ARG_LIMITS: Dict[str, Dict[str, Any]] = {
    "web_search": {"results": 10, "read_pages": 3},
    "serpapi_search": {"num": 10},
    "web_scraper": {"url": 2048},
    "openalex_search": {"per_page": 10},
    "rss_feed_reader": {"entries": 50},
    "code_execution": {"timeout": 180, "code": 100_000, "packages": 20},
    "webhook_post": {"timeout": 60, "url": 2048},
    "email_sender": {"attachments": 3, "body": 200_000, "html_body": 400_000, "subject": 1_000},
    "long_document_summarizer": {"max_sentences": 50, "max_words": 4_000, "text": 250_000, "url": 2048, "collection_file": 500, "read_limit": 1_000_000},
    "plan_generator": {"goal": 4_000, "context": 20_000},
    "arxiv_search": {"max_results": 10},
    "wikipedia_search": {"limit": 10, "query": 2_000, "title": 500, "read_limit": 250_000, "offset": 50_000_000, "lines_start": 5_000_000, "lines_end": 5_000_000, "pattern": 500, "context_lines": 20},
    "spawn_subagent": {"task": 50_000, "label": 80},
    "wait": {"seconds": 120, "reason": 1_000},
    "wait_subagent": {"subagent_ids": 50},
    "nudge_subagent": {"subagent_id": 64, "message": 20_000},
    "agent_todo": {"task_id": 20, "title": 500, "note": 4_000},
}

# Maximum characters returned by any single tool call to the LLM
MAX_TOOL_OUTPUT_CHARS = 250_000


def _coerce_type(value: Any, expected_type: str) -> Any:
    """Attempt to coerce a value to the expected JSON schema type."""
    if value is None:
        return None
    if expected_type == "integer":
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    if expected_type == "number":
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        return value
    if expected_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower = value.lower()
            if lower in ("true", "1", "yes", "on"):
                return True
            if lower in ("false", "0", "no", "off"):
                return False
        return value
    if expected_type == "string":
        if not isinstance(value, str):
            return str(value)
        return value
    return value


def _get_param_schema(schema: Dict[str, Any], param_name: str) -> Optional[Dict[str, Any]]:
    """Extract the parameter schema for a given parameter name."""
    parameters = schema.get("function", {}).get("parameters", {})
    props = parameters.get("properties", {})
    return props.get(param_name)


def _matches_type(value: Any, expected_type: str) -> bool:
    """Check whether a (possibly coerced) value satisfies the JSON schema type."""
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def validate_and_normalize_args(
    tool_name: str,
    raw_args: Dict[str, Any],
    tool_schema: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[str], List[str]]:
    """Validate and normalize tool arguments against the tool schema.

    Returns (normalized_args, error_message, arg_notes).  If error_message is
    not None, the caller should treat it as a validation failure and pass the
    error back to the model as a ToolMessage.  ``arg_notes`` lists arguments
    that were ignored (unknown name or wrong type) so the caller can surface
    them to the model — silently dropping mistyped/misnamed arguments makes
    different calls produce identical results without any feedback.
    """
    if tool_schema is None:
        return raw_args, None, []

    parameters = tool_schema.get("function", {}).get("parameters", {})
    props = parameters.get("properties", {})
    required = set(parameters.get("required", []))

    # 1. Filter unknown parameters and apply defaults/coerce types for known props
    normalized: Dict[str, Any] = {}
    notes: List[str] = []
    unknown: List[str] = []
    for key, value in raw_args.items():
        if key in props:
            expected = props[key].get("type")
            if expected:
                value = _coerce_type(value, expected)
                # Models frequently pass a bare value where an array is
                # declared, or a JSON-encoded string for objects — repair
                # those shapes instead of dropping the argument.  Explicit
                # nulls pass through untouched (tools treat them as unset).
                if value is not None:
                    if expected == "array" and not isinstance(value, list):
                        if isinstance(value, str):
                            try:
                                parsed = json.loads(value)
                            except (ValueError, TypeError):
                                parsed = None
                            if isinstance(parsed, list):
                                value = parsed
                                notes.append(f"argument '{key}' was parsed from a JSON string")
                            else:
                                value = [value]
                                notes.append(f"argument '{key}' was wrapped into an array")
                        else:
                            value = [value]
                            notes.append(f"argument '{key}' was wrapped into an array")
                    elif expected == "object" and isinstance(value, str):
                        try:
                            parsed = json.loads(value)
                        except (ValueError, TypeError):
                            parsed = None
                        if isinstance(parsed, dict):
                            value = parsed
                            notes.append(f"argument '{key}' was parsed from a JSON string")
                    if not _matches_type(value, expected):
                        notes.append(
                            f"argument '{key}' ignored (expected {expected}, "
                            f"got {type(value).__name__})"
                        )
                        continue
            normalized[key] = value
        else:
            unknown.append(key)
            logger.debug("Tool '%s' received unknown argument '%s'", tool_name, key)
    if unknown:
        notes.append(
            f"ignored unknown argument(s): {', '.join(sorted(unknown))}; "
            f"valid parameters for {tool_name}: {', '.join(sorted(props))}"
        )

    # 2. Apply defaults for missing optional props
    for name, param_schema in props.items():
        if name not in normalized:
            default = param_schema.get("default")
            if default is not None:
                normalized[name] = default

    # 3. Check required fields
    missing = required - set(normalized.keys())
    if missing:
        return raw_args, f"Missing required argument(s): {', '.join(sorted(missing))}", notes

    # 4. Apply central limits
    limits = _TOOL_ARG_LIMITS.get(tool_name, {})
    for limit_key, limit_value in limits.items():
        if limit_key in normalized:
            current = normalized[limit_key]
            if isinstance(limit_value, int) and isinstance(current, bool):
                pass  # bools are ints in Python but never meant as counts
            elif isinstance(limit_value, int) and isinstance(current, int):
                if current > limit_value:
                    normalized[limit_key] = limit_value
                    notes.append(f"argument '{limit_key}' capped at {limit_value}")
                    logger.debug("Tool '%s' argument '%s' capped from %d to %d", tool_name, limit_key, current, limit_value)
            elif isinstance(limit_value, int) and isinstance(current, str):
                if len(current) > limit_value:
                    normalized[limit_key] = current[:limit_value]
                    notes.append(f"argument '{limit_key}' truncated to {limit_value} chars")
                    logger.debug("Tool '%s' argument '%s' truncated to %d chars", tool_name, limit_key, limit_value)
            elif isinstance(limit_value, int) and isinstance(current, list):
                if len(current) > limit_value:
                    normalized[limit_key] = current[:limit_value]
                    notes.append(f"argument '{limit_key}' truncated to {limit_value} items")
                    logger.debug("Tool '%s' argument '%s' truncated from %d to %d items", tool_name, limit_key, len(current), limit_value)

    return normalized, None, notes


def truncate_tool_output(content: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Truncate tool output and append a clear marker when truncated."""
    if not content or len(content) <= max_chars:
        return content
    truncated = content[:max_chars]
    marker = f"\n\n[Content truncated: {len(content)} chars total, showing first {max_chars}]"
    return truncated + marker
