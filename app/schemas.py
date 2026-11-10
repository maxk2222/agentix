# app/schemas.py

from typing import Any, List, Optional

from pydantic import BaseModel

class Message(BaseModel):
    """Single OpenAI-style chat message used for both requests and responses."""
    role: str
    content: Any

class ChatRequest(BaseModel):
    """Payload accepted by ``POST /v1/chat/completions``."""
    model: str = "CyberSecurityAssistant"
    messages: List[Message]
    # Optional client-provided conversation id. If not provided the server
    # will generate one using the existing `chatcmpl-{uuid4().hex}` format.
    conversation_id: Optional[str] = None
    stream: bool = False
    max_tokens: Optional[int] = None
    emotion_context: Optional[dict] = None
    # Optional inline images as Base64 data-URIs or raw Base64 strings.
    # Only processed when chat_image_upload_enabled is True in runtime config.
    images: Optional[List[str]] = None

class ChatResponseChoice(BaseModel):
    """Single alternative in the OpenAI-compatible chat completion response."""
    index: int
    message: Message
    finish_reason: str = "stop"

class Usage(BaseModel):
    """Token usage statistics for a completion request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0


class ChatResponse(BaseModel):
    """Canonical structure returned for non-streaming chat completions."""
    id: str
    # Echo the conversation id used for the request (server-generated or
    # client-provided). This is optional to remain backwards compatible.
    conversation_id: Optional[str] = None
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatResponseChoice]
    usage: Usage = Usage()


class DeltaMessage(BaseModel):
    """Incremental message fragment used in SSE streaming chunks."""
    role: Optional[str] = None
    content: Optional[str] = None


class ChatResponseChunkChoice(BaseModel):
    """Single alternative in a streaming chat completion chunk."""
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatResponseChunk(BaseModel):
    """Structure of each SSE event emitted during streaming."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatResponseChunkChoice]