"""Tests for model-aware reasoning parameter mapping."""

import pytest

from app.reasoning_params import (
    CANONICAL_KEYS,
    REASONING_FAMILIES,
    deep_merge_extra_body,
    describe_for_model,
    load_override_rules,
    match_family,
    normalize_reasoning_dict,
    reset_override_rules,
    translate_to_wire,
)


# ---------------------------------------------------------------------------
# match_family
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model,expected_family",
    [
        ("Qwen3-8B", "qwen_thinking"),
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen_thinking"),
        ("qwen3-vl-8b-thinking", "qwen_thinking"),
        ("QwQ-32B", "qwen_thinking"),
        ("deepseek-v3.1", "chat_template_toggle"),
        ("deepseek-r1-0528", "chat_template_toggle"),
        ("ibm-granite-4.0", "chat_template_toggle"),
        ("gemma-3-27b", "chat_template_enable_thinking"),
        ("claude-sonnet-4-5", "claude_budget"),
        ("claude-4-opus", "claude_budget"),
        ("gemini-3-pro", "gemini_budget"),
        ("gemini-2.5-flash", "gemini_budget"),
        ("gpt-5.4", "openai_effort"),
        ("openai/gpt-5-mini", "openai_effort"),
        ("o3", "openai_effort"),
        ("azure/o4-mini", "openai_effort"),
        ("gpt-4o", "generic"),
        ("llama-3.3-70b", "generic"),
        ("mistral-large", "generic"),
        ("", "generic"),
        (None, "generic"),
    ],
)
def test_match_family_by_model_name(model, expected_family):
    assert match_family(model)["family"] == expected_family


def test_match_family_url_hint():
    rule = match_family("some-model", "https://api.anthropic.com")
    assert rule["family"] == "claude_budget"
    rule = match_family("some-model", "https://generativelanguage.googleapis.com")
    assert rule["family"] == "gemini_budget"


def test_match_family_model_pattern_beats_url_hint():
    # A qwen model hosted anywhere must match qwen, not the provider URL hint.
    rule = match_family("Qwen3-8B", "https://api.anthropic.com")
    assert rule["family"] == "qwen_thinking"
    # And an OpenAI-effort model behind an anthropic URL still matches its pattern.
    rule = match_family("gpt-5.4", "https://api.anthropic.com")
    assert rule["family"] == "openai_effort"


def test_openai_pattern_does_not_match_o_in_word():
    # "o3" inside a word must not match the o-series pattern.
    assert match_family("modelo3x")["family"] == "generic"
    assert match_family("photo3-renderer")["family"] == "generic"
    # Separated tokens do match.
    assert match_family("openai/o3")["family"] == "openai_effort"
    assert match_family("google-o3-foo")["family"] == "openai_effort"


# ---------------------------------------------------------------------------
# normalize_reasoning_dict
# ---------------------------------------------------------------------------

def test_normalize_none_and_empty():
    assert normalize_reasoning_dict(None) is None
    assert normalize_reasoning_dict({}) is None
    assert normalize_reasoning_dict("low") is None
    assert normalize_reasoning_dict(42) is None


def test_normalize_strips_unknown_keys(caplog):
    out = normalize_reasoning_dict({"effort": "high", "bogus": 1, "also_bad": {}})
    assert out == {"effort": "high"}


def test_normalize_type_checks(caplog):
    out = normalize_reasoning_dict(
        {
            "effort": 123,
            "enabled": "yes",
            "budget": "2048",
            "level": "high",
        }
    )
    # effort/enabled invalid types stripped; budget coerced; level ok
    assert out == {"budget": 2048, "level": "high"}


def test_normalize_bool_not_int_budget():
    assert normalize_reasoning_dict({"budget": True}) is None


# ---------------------------------------------------------------------------
# translate_to_wire — golden outputs per family
# ---------------------------------------------------------------------------

def _rule(family: str):
    return next(r for r in REASONING_FAMILIES if r["family"] == family)


def test_translate_none():
    assert translate_to_wire(None, _rule("qwen_thinking")) == {}
    assert translate_to_wire({}, _rule("openai_effort")) == {}


def test_translate_openai_effort():
    wire = translate_to_wire({"effort": "high"}, _rule("openai_effort"))
    assert wire == {"reasoning": {"effort": "high"}}


def test_translate_openai_effort_ignores_budget(caplog):
    wire = translate_to_wire({"effort": "low", "budget": 500}, _rule("openai_effort"))
    assert wire == {"reasoning": {"effort": "low"}}


def test_translate_qwen():
    wire = translate_to_wire({"enabled": False, "budget": 4096}, _rule("qwen_thinking"))
    assert wire == {"chat_template_kwargs": {"enable_thinking": False, "thinking_budget": 4096}}


def test_translate_qwen_enabled_only():
    wire = translate_to_wire({"enabled": True}, _rule("qwen_thinking"))
    assert wire == {"chat_template_kwargs": {"enable_thinking": True}}


def test_translate_qwen_effort_ignored():
    wire = translate_to_wire({"effort": "high"}, _rule("qwen_thinking"))
    assert wire == {}


def test_translate_chat_template_thinking():
    wire = translate_to_wire({"enabled": False}, _rule("chat_template_toggle"))
    assert wire == {"chat_template_kwargs": {"thinking": False}}


def test_translate_chat_template_enable_thinking():
    wire = translate_to_wire({"enabled": True}, _rule("chat_template_enable_thinking"))
    assert wire == {"chat_template_kwargs": {"enable_thinking": True}}


def test_translate_claude_budget():
    wire = translate_to_wire({"enabled": True, "budget": 8000}, _rule("claude_budget"))
    assert wire == {"thinking": {"type": "enabled", "budget_tokens": 8000}}


def test_translate_claude_disabled():
    wire = translate_to_wire({"enabled": False}, _rule("claude_budget"))
    assert wire == {"thinking": {"type": "disabled"}}


def test_translate_claude_budget_without_toggle():
    wire = translate_to_wire({"budget": 4096}, _rule("claude_budget"))
    assert wire == {"thinking": {"type": "enabled", "budget_tokens": 4096}}


def test_translate_gemini():
    wire = translate_to_wire({"budget": 8192, "level": "high"}, _rule("gemini_budget"))
    assert wire == {"thinkingConfig": {"thinkingBudget": 8192, "thinkingLevel": "high"}}


def test_translate_gemini_disabled():
    wire = translate_to_wire({"enabled": False}, _rule("gemini_budget"))
    assert wire == {"thinkingConfig": {"thinkingBudget": 0}}


def test_translate_generic_passthrough():
    wire = translate_to_wire({"effort": "low"}, _rule("generic"))
    assert wire == {"reasoning": {"effort": "low"}}


# ---------------------------------------------------------------------------
# deep_merge_extra_body
# ---------------------------------------------------------------------------

def test_deep_merge_nested():
    base = {"chat_template_kwargs": {"enable_thinking": True}, "other": 1}
    wire = {"chat_template_kwargs": {"thinking_budget": 100}}
    merged = deep_merge_extra_body(base, wire)
    assert merged == {
        "chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 100},
        "other": 1,
    }
    # Inputs not mutated
    assert base["chat_template_kwargs"] == {"enable_thinking": True}


def test_deep_merge_wire_wins():
    base = {"reasoning": {"effort": "low"}}
    wire = {"reasoning": {"effort": "high"}}
    assert deep_merge_extra_body(base, wire)["reasoning"]["effort"] == "high"


def test_deep_merge_empty():
    assert deep_merge_extra_body(None, None) == {}
    assert deep_merge_extra_body({"a": 1}, None) == {"a": 1}
    assert deep_merge_extra_body(None, {"b": 2}) == {"b": 2}


# ---------------------------------------------------------------------------
# Runtime override rules
# ---------------------------------------------------------------------------

def test_load_override_rules_env(monkeypatch):
    reset_override_rules()
    monkeypatch.setenv(
        "REASONING_PARAM_RULES_JSON",
        '[{"family": "custom_effort", "patterns": ["mymodel"], "wire": "openai_effort",'
        ' "controls": [{"key": "effort", "type": "select", "options": ["a"]}]}]',
    )
    try:
        rules = load_override_rules()
        assert len(rules) == 1
        assert rules[0]["family"] == "custom_effort"
        # Override rule matches before built-ins
        assert match_family("mymodel-9")["family"] == "custom_effort"
    finally:
        monkeypatch.delenv("REASONING_PARAM_RULES_JSON", raising=False)
        reset_override_rules()
        load_override_rules()


def test_load_override_rules_invalid_json(monkeypatch, caplog):
    reset_override_rules()
    monkeypatch.setenv("REASONING_PARAM_RULES_JSON", "{not json")
    try:
        rules = load_override_rules()
        assert rules == []
        assert match_family("gpt-5")["family"] == "openai_effort"
    finally:
        reset_override_rules()


def test_load_override_rules_rejects_unknown_wire(monkeypatch):
    reset_override_rules()
    monkeypatch.setenv(
        "REASONING_PARAM_RULES_JSON",
        '[{"family": "bad", "patterns": ["x"], "wire": "not_a_wire"}]',
    )
    try:
        rules = load_override_rules()
        assert rules == []
    finally:
        monkeypatch.delenv("REASONING_PARAM_RULES_JSON", raising=False)
        reset_override_rules()


def test_load_override_rules_from_file(monkeypatch, tmp_path):
    reset_override_rules()
    rule_file = tmp_path / "rules.json"
    rule_file.write_text(
        '{"rules": [{"family": "file_family", "patterns": ["filemodel"], "wire": "generic"}]}'
    )
    monkeypatch.setenv("REASONING_PARAM_RULES_JSON", str(rule_file))
    try:
        rules = load_override_rules()
        assert len(rules) == 1
        assert match_family("filemodel-2")["family"] == "file_family"
    finally:
        monkeypatch.delenv("REASONING_PARAM_RULES_JSON", raising=False)
        reset_override_rules()


# ---------------------------------------------------------------------------
# describe_for_model
# ---------------------------------------------------------------------------

def test_describe_for_model_serializable():
    import json

    desc = describe_for_model("Qwen3-8B")
    json.dumps(desc)  # must be JSON-serializable
    assert desc["family"] == "qwen_thinking"
    assert desc["assumed"] is False
    keys = {c["key"] for c in desc["controls"]}
    assert keys == {"enabled", "budget"}


def test_describe_for_model_unknown_assumed():
    desc = describe_for_model("some-custom-model")
    assert desc["family"] == "generic"
    assert desc["assumed"] is True
    assert "effort" in {c["key"] for c in desc["controls"]}


def test_canonical_keys_documented():
    assert CANONICAL_KEYS == {"effort", "enabled", "budget", "level"}


# ---------------------------------------------------------------------------
# qwen3_effort family (newer Qwen generations on vLLM)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model",
    [
        "Qwen3.8-27B",
        "Qwen3.8-Next-Flash",
        "qwen3-next-80b-a3b",
        "Qwen3.5-7B-Instruct",
        "qwen39-something",
        "qwen4-72b",
    ],
)
def test_match_family_qwen3_effort(model):
    assert match_family(model)["family"] == "qwen3_effort"


@pytest.mark.parametrize(
    "model",
    [
        "Qwen3-8B",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "qwen3-vl-8b-thinking",
        "qwen3-5b",          # 5-billion param, not 3.5 generation → no match on "-5b"
        "Qwen3-30B-Flash",   # param-size suffix, no gen marker → old family
    ],
)
def test_match_family_qwen_thinking_unchanged(model):
    assert match_family(model)["family"] == "qwen_thinking"


def test_match_family_qwen_flash_not_effort():
    # Plain DashScope qwen-flash has no gen marker — must NOT get the
    # reasoning_effort family (its API only supports enable_thinking).
    assert match_family("qwen-flash")["family"] != "qwen3_effort"


def test_qwen3_effort_controls():
    desc = describe_for_model("Qwen3.8-Next-Flash")
    assert desc["family"] == "qwen3_effort"
    controls = {c["key"]: c for c in desc["controls"]}
    assert controls["effort"]["type"] == "select"
    # Qwen3.8 chat template only accepts these — no 'high'/'minimal'/'none'.
    assert controls["effort"]["options"] == ["low", "medium", "xhigh"]
    assert controls["enabled"]["type"] == "toggle"
    assert controls["budget"]["type"] == "number"


def test_translate_qwen3_effort_full():
    rule = match_family("Qwen3.8-Next-Flash")
    wire = translate_to_wire({"effort": "low", "enabled": True, "budget": 4096}, rule)
    assert wire == {
        "reasoning_effort": "low",
        "chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 4096},
    }


def test_translate_qwen3_effort_only():
    rule = match_family("Qwen3.8-Next-Flash")
    assert translate_to_wire({"effort": "xhigh"}, rule) == {"reasoning_effort": "xhigh"}


def test_translate_qwen3_effort_empty():
    rule = match_family("Qwen3.8-Next-Flash")
    assert translate_to_wire({}, rule) == {}
    assert translate_to_wire(None, rule) == {}


def test_validate_rule_accepts_new_wires():
    from app.reasoning_params import _validate_rule

    for wire in ("qwen3_effort", "probed_effort"):
        rule = _validate_rule({"family": f"f_{wire}", "wire": wire, "patterns": [], "controls": []})
        assert rule is not None
        assert rule["wire"] == wire


# ---------------------------------------------------------------------------
# descriptor_from_capabilities (probe-driven descriptor)
# ---------------------------------------------------------------------------

def test_descriptor_from_capabilities_full():
    from app.reasoning_params import descriptor_from_capabilities

    desc = descriptor_from_capabilities(
        {
            "supports_effort": True,
            "effort_levels": ["low", "medium", "xhigh"],
            "supports_thinking_toggle": True,
        },
        "Qwen3.8-Next-Flash",
    )
    assert desc is not None
    assert desc["family"] == "probed"
    assert desc["probed"] is True
    assert desc["assumed"] is False
    assert desc["wire"] == "qwen3_effort"
    keys = {c["key"]: c for c in desc["controls"]}
    assert keys["effort"]["options"] == ["low", "medium", "xhigh"]
    assert keys["enabled"]["type"] == "toggle"
    assert keys["budget"]["type"] == "number"


def test_descriptor_from_capabilities_effort_only():
    from app.reasoning_params import descriptor_from_capabilities

    desc = descriptor_from_capabilities(
        {"supports_effort": True, "effort_levels": [], "supports_thinking_toggle": False},
        "m",
    )
    assert desc is not None
    assert desc["wire"] == "probed_effort"
    keys = {c["key"] for c in desc["controls"]}
    assert keys == {"effort"}
    assert desc["controls"][0]["options"] == ["low", "medium", "high"]  # default


def test_descriptor_from_capabilities_toggle_only():
    from app.reasoning_params import descriptor_from_capabilities

    desc = descriptor_from_capabilities(
        {"supports_effort": False, "supports_thinking_toggle": True}, "m"
    )
    assert desc is not None
    assert desc["wire"] == "qwen_thinking"
    assert {c["key"] for c in desc["controls"]} == {"enabled", "budget"}


def test_descriptor_from_capabilities_none():
    from app.reasoning_params import descriptor_from_capabilities

    # No probe data → None → caller falls back to rule matching
    assert descriptor_from_capabilities(None, "m") is None
    assert descriptor_from_capabilities({}, "m") is None


def test_descriptor_from_capabilities_all_rejected():
    from app.reasoning_params import descriptor_from_capabilities

    # Probe ran and rejected BOTH controls → explicit empty descriptor so the
    # UI can say "no options" instead of offering controls the probe disproved.
    desc = descriptor_from_capabilities(
        {"supports_effort": False, "supports_thinking_toggle": False}, "m"
    )
    assert desc is not None
    assert desc["controls"] == []
    assert desc["probed"] is True
    assert desc["wire"] == "drop"


# ---------------------------------------------------------------------------
# rule_from_capabilities — runtime wire selection from probe cache
# ---------------------------------------------------------------------------


def test_rule_from_capabilities():
    from app.reasoning_params import rule_from_capabilities

    assert rule_from_capabilities(None) is None
    assert rule_from_capabilities({}) is None
    assert rule_from_capabilities(
        {"supports_effort": True, "supports_thinking_toggle": True}
    )["wire"] == "qwen3_effort"
    assert rule_from_capabilities(
        {"supports_effort": True, "supports_thinking_toggle": False}
    )["wire"] == "probed_effort"
    assert rule_from_capabilities(
        {"supports_effort": False, "supports_thinking_toggle": True}
    )["wire"] == "qwen_thinking"
    assert rule_from_capabilities(
        {"supports_effort": False, "supports_thinking_toggle": False}
    )["wire"] == "drop"


def test_translate_drop_wire_sends_nothing():
    # A probe that disproved all controls must not leak stale config values.
    rule = {"family": "probed", "wire": "drop", "controls": []}
    assert translate_to_wire({"effort": "low", "enabled": True}, rule) == {}


def test_translate_probed_effort_sends_only_effort():
    # Probe confirmed effort but NOT the toggle — template kwargs must never
    # be sent even if a stale config still carries them.
    rule = {"family": "probed", "wire": "probed_effort", "controls": []}
    assert translate_to_wire(
        {"effort": "low", "enabled": False, "budget": 100}, rule
    ) == {"reasoning_effort": "low"}


def test_descriptor_from_capabilities_filters_unknown_levels():
    from app.reasoning_params import descriptor_from_capabilities

    desc = descriptor_from_capabilities(
        {"supports_effort": True, "effort_levels": ["low", "bogus", "HIGH"]}, "m"
    )
    assert desc is not None
    assert desc["controls"][0]["options"] == ["low", "high"]
