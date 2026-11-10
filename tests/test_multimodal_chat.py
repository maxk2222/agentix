"""Tests for multimodal chat support — schemas, chat_utils, and image validation."""

import base64
import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.schemas import ChatRequest
from app.chat_utils import (
    build_multimodal_content,
    convert_openai_messages,
    stringify_message_content,
    _has_multimodal_content,
)


class TestBuildMultimodalContent:
    """Tests for build_multimodal_content helper."""

    def test_text_only(self):
        content = build_multimodal_content("Hello", [])
        assert content == [{"type": "text", "text": "Hello"}]

    def test_empty_text_with_images(self):
        content = build_multimodal_content("", ["data:image/png;base64,abc"])
        assert len(content) == 1
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == "data:image/png;base64,abc"

    def test_text_and_images(self):
        content = build_multimodal_content("Describe this", ["data:image/png;base64,abc"])
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Describe this"}
        assert content[1]["type"] == "image_url"

    def test_raw_base64_gets_prefix(self):
        content = build_multimodal_content("Hi", ["abc123"])
        assert content[1]["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_multiple_images(self):
        content = build_multimodal_content("Hi", ["data:image/png;base64,a", "data:image/jpeg;base64,b"])
        assert len(content) == 3  # 1 text + 2 images
        assert content[1]["image_url"]["url"] == "data:image/png;base64,a"
        assert content[2]["image_url"]["url"] == "data:image/jpeg;base64,b"


class TestHasMultimodalContent:
    def test_string_content(self):
        assert not _has_multimodal_content("hello")

    def test_list_without_images(self):
        assert not _has_multimodal_content([{"type": "text", "text": "hi"}])

    def test_list_with_image(self):
        assert _has_multimodal_content([
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ])

    def test_none(self):
        assert not _has_multimodal_content(None)


class TestStringifyMultimodal:
    def test_skips_image_parts(self):
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "World"},
        ]
        result = stringify_message_content(content)
        assert result == "HelloWorld"
        assert "image" not in result
        assert "data:" not in result

    def test_plain_string(self):
        assert stringify_message_content("Hello") == "Hello"

    def test_list_of_strings(self):
        assert stringify_message_content(["Hello", "World"]) == "HelloWorld"


class TestConvertOpenAIMessagesMultimodal:
    def test_user_message_with_multimodal_content_preserved(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]}
        ]
        result = convert_openai_messages(messages)
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[0].content, list)
        assert any(part.get("type") == "image_url" for part in result[0].content)

    def test_user_message_plain_text(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = convert_openai_messages(messages)
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello"

    def test_assistant_message_stringified(self):
        messages = [{"role": "assistant", "content": "Hi there"}]
        result = convert_openai_messages(messages)
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "Hi there"

    def test_system_message_stringified(self):
        messages = [{"role": "system", "content": "You are helpful"}]
        result = convert_openai_messages(messages)
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are helpful"


class TestChatRequestImages:
    def test_images_field_optional(self):
        req = ChatRequest(messages=[{"role": "user", "content": "Hi"}])
        assert req.images is None

    def test_images_field_set(self):
        req = ChatRequest(
            messages=[{"role": "user", "content": "Hi"}],
            images=["data:image/png;base64,abc"],
        )
        assert req.images == ["data:image/png;base64,abc"]

    def test_images_field_empty_list(self):
        req = ChatRequest(
            messages=[{"role": "user", "content": "Hi"}],
            images=[],
        )
        assert req.images == []


class TestAgentRunnerMultimodal:
    """Test that AgentRunner._build_messages handles multimodal_input."""

    def test_build_messages_with_multimodal(self):
        from agent.tools.runner import AgentRunner
        runner = AgentRunner.__new__(AgentRunner)
        runner.max_tool_calls = 5
        runner.max_parallel_tool_calls = 3

        multimodal = HumanMessage(content=[
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ])

        messages = runner._build_messages(
            system_instructions="You are helpful",
            chat_history=[],
            input_text="Describe this image",
            fallback_mode=False,
            tool_schemas=[],
            multimodal_input=multimodal,
        )

        # System + user message
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert isinstance(messages[1].content, list)
        assert any(part.get("type") == "image_url" for part in messages[1].content)

    def test_build_messages_without_multimodal(self):
        from agent.tools.runner import AgentRunner
        runner = AgentRunner.__new__(AgentRunner)
        runner.max_tool_calls = 5
        runner.max_parallel_tool_calls = 3

        messages = runner._build_messages(
            system_instructions="You are helpful",
            chat_history=[],
            input_text="Hello",
            fallback_mode=False,
            tool_schemas=[],
            multimodal_input=None,
        )

        assert len(messages) == 2
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "Hello"
