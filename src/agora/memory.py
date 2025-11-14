"""Data structures representing per-agent memory turns."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .llm import ChatMessage


@dataclass(frozen=True)
class MemoryTurn:
    """Immutable record of a single turn as experienced by an agent."""

    turn_id: int
    speaker_id: str
    role: str
    public_speech: Optional[str] = None
    private_reflection: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: Optional[str] = None
    status: Optional[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        """Serialize only the public portions of the turn for sharing."""

        return {
            "turn_id": self.turn_id,
            "speaker_id": self.speaker_id,
            "role": self.role,
            "public_speech": self.public_speech,
            "metadata": self.metadata,
            "message_id": self.message_id,
            "status": self.status,
        }

    def to_chat_message(self, *, viewer_id: str, multi_party: bool = False) -> ChatMessage:
        """Render the turn as a standard chat completion message."""

        role = "assistant" if self.speaker_id == viewer_id else "user"
        content = self.public_speech or ""
        if role == "user" and multi_party:
            speaker = self.metadata.get("speaker_name") or self.speaker_id
            content = f"{speaker}: {content}"
        return ChatMessage(role=role, content=content)

    def to_openrouter_response(self, *, viewer_id: str, multi_party: bool = False) -> Dict[str, Any]:
        """Render the turn as an OpenRouter responses API message."""

        role = "assistant" if self.speaker_id == viewer_id else "user"
        content_text = self.public_speech or ""
        if role == "user" and multi_party:
            speaker = self.metadata.get("speaker_name") or self.speaker_id
            content_text = f"{speaker}: {content_text}"
        content_item = {
            "type": "output_text" if role == "assistant" else "input_text",
            "text": content_text,
        }
        message: Dict[str, Any] = {
            "type": "message",
            "role": role,
            "content": [content_item],
        }
        if self.message_id:
            message["id"] = self.message_id
        if role == "assistant":
            message["status"] = self.status or "completed"
        return message


__all__ = ["MemoryTurn"]
