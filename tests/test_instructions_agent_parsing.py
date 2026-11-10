"""Tests for agent_enabled and tools parsing in instructions YAML."""

from __future__ import annotations

import pytest
import yaml

from app.chat_utils import load_instructions_from_yaml_text


def test_parse_agent_enabled_true():
    data = {
        "assistants": [
            {
                "name": "AgentAssistant",
                "model": "gpt-4",
                "system_instructions": "You are an agent.",
                "agent_enabled": True,
                "tools": {"web_search": {}, "web_scraper": {}},
            }
        ]
    }
    text = yaml.dump(data)
    result = load_instructions_from_yaml_text(text)

    assert "AgentAssistant" in result
    assert result["AgentAssistant"]["agent_enabled"] is True
    assert result["AgentAssistant"]["tools"] == {"web_search": {}, "web_scraper": {}}


def test_parse_agent_enabled_false_by_default():
    data = {
        "assistants": [
            {
                "name": "NormalAssistant",
                "model": "gpt-4",
                "system_instructions": "You are helpful.",
            }
        ]
    }
    text = yaml.dump(data)
    result = load_instructions_from_yaml_text(text)

    assert "NormalAssistant" in result
    assert result["NormalAssistant"]["agent_enabled"] is False
    assert result["NormalAssistant"]["tools"] == {}


def test_parse_tools_empty_dict():
    data = {
        "assistants": [
            {
                "name": "Assistant",
                "model": "gpt-4",
                "system_instructions": "You are helpful.",
                "agent_enabled": True,
                "tools": {},
            }
        ]
    }
    text = yaml.dump(data)
    result = load_instructions_from_yaml_text(text)

    assert result["Assistant"]["tools"] == {}


def test_parse_tools_invalid_type_raises():
    data = {
        "assistants": [
            {
                "name": "BadAssistant",
                "model": "gpt-4",
                "system_instructions": "You are helpful.",
                "tools": "not_a_dict",
            }
        ]
    }
    text = yaml.dump(data)
    with pytest.raises(ValueError, match="invalid 'tools' field"):
        load_instructions_from_yaml_text(text)


def test_parse_agent_enabled_various_truthy_values():
    """Test that various truthy YAML values for agent_enabled are parsed correctly."""
    for value in [True, "yes", 1]:
        data = {
            "assistants": [
                {
                    "name": "A",
                    "model": "gpt-4",
                    "system_instructions": "You are helpful.",
                    "agent_enabled": value,
                }
            ]
        }
        text = yaml.dump(data)
        result = load_instructions_from_yaml_text(text)
        assert result["A"]["agent_enabled"] == bool(value)


def test_tools_dict_with_config():
    """Test that tool names with whitespace are stripped."""
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "You are helpful.",
                "tools": {"web_search": {"max_results": 5}, "web_scraper": {}},
            }
        ]
    }
    text = yaml.dump(data)
    result = load_instructions_from_yaml_text(text)
    assert result["A"]["tools"] == {"web_search": {"max_results": 5}, "web_scraper": {}}


def test_parse_max_tool_calls():
    """Test that max_tool_calls is correctly parsed and propagated."""
    data = {
        "assistants": [
            {
                "name": "AgentAssistant",
                "model": "gpt-4",
                "system_instructions": "You are an agent.",
                "agent_enabled": True,
                "tools": {"web_search": {}},
                "max_tool_calls": 12,
            }
        ]
    }
    text = yaml.dump(data)
    result = load_instructions_from_yaml_text(text)

    assert result["AgentAssistant"]["max_tool_calls"] == 12


def test_parse_max_tool_calls_omitted():
    """Test that max_tool_calls is None when not present in YAML."""
    data = {
        "assistants": [
            {
                "name": "AgentAssistant",
                "model": "gpt-4",
                "system_instructions": "You are an agent.",
                "agent_enabled": True,
                "tools": {"web_search": {}},
            }
        ]
    }
    text = yaml.dump(data)
    result = load_instructions_from_yaml_text(text)

    assert result["AgentAssistant"]["max_tool_calls"] is None


def test_parse_agent_memory_enabled_true():
    data = {
        "assistants": [
            {
                "name": "AgentAssistant",
                "model": "gpt-4",
                "system_instructions": "You are an agent.",
                "agent_enabled": True,
                "agent_memory_enabled": True,
            }
        ]
    }
    text = yaml.dump(data)
    result = load_instructions_from_yaml_text(text)

    assert result["AgentAssistant"]["agent_memory_enabled"] is True


def test_parse_agent_memory_enabled_false_by_default():
    data = {
        "assistants": [
            {
                "name": "AgentAssistant",
                "model": "gpt-4",
                "system_instructions": "You are an agent.",
                "agent_enabled": True,
            }
        ]
    }
    text = yaml.dump(data)
    result = load_instructions_from_yaml_text(text)

    assert result["AgentAssistant"]["agent_memory_enabled"] is False


def test_parse_extra_body_and_extra_parameters():
    """Regression: extra_body/extra_parameters/enable_thinking must survive the
    runtime loader (they were previously dropped by the field whitelist)."""
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "You are helpful.",
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                "extra_parameters": {"top_p": 0.9},
                "enable_thinking": True,
            }
        ]
    }
    text = yaml.dump(data)
    result = load_instructions_from_yaml_text(text)

    entry = result["A"]
    assert entry["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert entry["extra_parameters"] == {"top_p": 0.9}
    assert entry["enable_thinking"] is True


def test_parse_extra_fields_default_to_none():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "You are helpful.",
            }
        ]
    }
    text = yaml.dump(data)
    result = load_instructions_from_yaml_text(text)

    entry = result["A"]
    assert entry["extra_body"] is None
    assert entry["extra_parameters"] is None
    assert entry["enable_thinking"] is None


def test_parse_extra_body_invalid_type_raises():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "You are helpful.",
                "extra_body": "not_a_dict",
            }
        ]
    }
    text = yaml.dump(data)
    with pytest.raises(ValueError, match="invalid 'extra_body' field"):
        load_instructions_from_yaml_text(text)


def test_parse_extra_parameters_invalid_type_raises():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "You are helpful.",
                "extra_parameters": ["top_p"],
            }
        ]
    }
    text = yaml.dump(data)
    with pytest.raises(ValueError, match="invalid 'extra_parameters' field"):
        load_instructions_from_yaml_text(text)


def test_parse_enable_thinking_invalid_type_raises():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "You are helpful.",
                "enable_thinking": "maybe",
            }
        ]
    }
    text = yaml.dump(data)
    with pytest.raises(ValueError, match="invalid 'enable_thinking' field"):
        load_instructions_from_yaml_text(text)


def test_parse_subagents_config_and_defaults():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "You are helpful.",
                "subagents": {"enabled": True, "max_concurrent": 3, "max_depth": 2},
            }
        ]
    }
    result = load_instructions_from_yaml_text(yaml.dump(data))
    assert result["A"]["subagents"] == {
        "enabled": True,
        "max_concurrent": 3,
        "max_depth": 2,
        "max_total": 8,
    }


def test_parse_invalid_subagents_limits_raises():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "You are helpful.",
                "subagents": {"enabled": True, "max_concurrent": 99},
            }
        ]
    }
    with pytest.raises(ValueError, match="max_concurrent must be between"):
        load_instructions_from_yaml_text(yaml.dump(data))


def test_parse_subagents_max_total_up_to_500():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "You are helpful.",
                "subagents": {"enabled": True, "max_total": 500},
            }
        ]
    }
    result = load_instructions_from_yaml_text(yaml.dump(data))
    assert result["A"]["subagents"]["max_total"] == 500


def test_parse_observer_config_and_defaults():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "You are helpful.",
                "observer": {
                    "enabled": True,
                    "every_iterations": 3,
                    "assistant": "ObsModel",
                },
            }
        ]
    }
    result = load_instructions_from_yaml_text(yaml.dump(data))
    obs = result["A"]["observer"]
    assert obs["enabled"] is True
    assert obs["every_iterations"] == 3
    assert obs["assistant"] == "ObsModel"
    # defaults filled for omitted fields
    assert obs["every_seconds"] == 180.0
    assert obs["max_checks"] == 8
    assert obs["max_nudges"] == 4
    assert obs["check_timeout_seconds"] == 60.0
    assert obs["include_subagents"] is True


def test_parse_observer_absent_gives_none():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "You are helpful.",
            }
        ]
    }
    result = load_instructions_from_yaml_text(yaml.dump(data))
    assert result["A"]["observer"] is None


def test_parse_observer_not_a_mapping_raises():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "x",
                "observer": "enabled",
            }
        ]
    }
    with pytest.raises(ValueError, match="invalid 'observer' field"):
        load_instructions_from_yaml_text(yaml.dump(data))


def test_parse_observer_out_of_range_raises():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "x",
                "observer": {"enabled": True, "every_iterations": 99},
            }
        ]
    }
    with pytest.raises(ValueError, match="observer.every_iterations must be between"):
        load_instructions_from_yaml_text(yaml.dump(data))


def test_parse_observer_empty_assistant_normalizes_to_none():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "x",
                "observer": {"enabled": True, "assistant": "   "},
            }
        ]
    }
    result = load_instructions_from_yaml_text(yaml.dump(data))
    assert result["A"]["observer"]["assistant"] is None


def test_parse_observer_custom_instructions_capped():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "x",
                "observer": {"enabled": True, "instructions": "y" * 5000},
            }
        ]
    }
    result = load_instructions_from_yaml_text(yaml.dump(data))
    assert len(result["A"]["observer"]["instructions"]) == 4000


def test_parse_subagents_max_total_above_500_raises():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "You are helpful.",
                "subagents": {"enabled": True, "max_total": 501},
            }
        ]
    }
    with pytest.raises(ValueError, match="max_total must be between"):
        load_instructions_from_yaml_text(yaml.dump(data))


def test_parse_subagents_exclude_tools_and_report_chars():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "x",
                "subagents": {
                    "enabled": True,
                    "exclude_tools": ["code_execution", "agent_workspace"],
                    "report_answer_chars": 5000,
                },
            }
        ]
    }
    result = load_instructions_from_yaml_text(yaml.dump(data))
    sub = result["A"]["subagents"]
    assert sub["exclude_tools"] == ["code_execution", "agent_workspace"]
    assert sub["report_answer_chars"] == 5000


def test_parse_subagents_exclude_tools_invalid_raises():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "x",
                "subagents": {"enabled": True, "exclude_tools": "code_execution"},
            }
        ]
    }
    with pytest.raises(ValueError, match="exclude_tools must be a list"):
        load_instructions_from_yaml_text(yaml.dump(data))


def test_parse_workspace_defaults_none_when_absent():
    data = {
        "assistants": [
            {"name": "A", "model": "gpt-4", "system_instructions": "x"}
        ]
    }
    result = load_instructions_from_yaml_text(yaml.dump(data))
    assert result["A"]["workspace"] is None


def test_parse_workspace_full():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "x",
                "workspace": {
                    "enabled": True,
                    "scope": "user",
                    "api_enabled": True,
                    "vision_read": True,
                    "max_file_mb": 10,
                    "max_total_mb": 100,
                    "max_files": 50,
                },
            }
        ]
    }
    result = load_instructions_from_yaml_text(yaml.dump(data))
    ws = result["A"]["workspace"]
    assert ws == {
        "enabled": True,
        "scope": "user",
        "api_enabled": True,
        "vision_read": True,
        "max_file_mb": 10,
        "max_total_mb": 100,
        "max_files": 50,
    }


def test_parse_workspace_vision_read_not_bool_raises():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "x",
                "workspace": {"enabled": True, "vision_read": "yes"},
            }
        ]
    }
    with pytest.raises(ValueError, match="workspace.vision_read"):
        load_instructions_from_yaml_text(yaml.dump(data))


def test_parse_workspace_invalid_scope_raises():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "x",
                "workspace": {"enabled": True, "scope": "global"},
            }
        ]
    }
    with pytest.raises(ValueError, match="workspace.scope"):
        load_instructions_from_yaml_text(yaml.dump(data))


def test_parse_workspace_api_without_enabled_raises():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "x",
                "workspace": {"enabled": False, "api_enabled": True},
            }
        ]
    }
    with pytest.raises(ValueError, match="api_enabled requires workspace.enabled"):
        load_instructions_from_yaml_text(yaml.dump(data))


def test_parse_workspace_quota_bounds():
    data = {
        "assistants": [
            {
                "name": "A",
                "model": "gpt-4",
                "system_instructions": "x",
                "workspace": {"enabled": True, "max_file_mb": 0},
            }
        ]
    }
    with pytest.raises(ValueError, match="workspace.max_file_mb must be between"):
        load_instructions_from_yaml_text(yaml.dump(data))
