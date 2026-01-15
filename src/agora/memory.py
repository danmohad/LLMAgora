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
    keep: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire turn (public + private) to a JSON-safe dict."""

        return {
            "turn_id": self.turn_id,
            "speaker_id": self.speaker_id,
            "role": self.role,
            "public_speech": self.public_speech,
            "private_reflection": self.private_reflection,
            "metadata": self.metadata,
            "message_id": self.message_id,
            "status": self.status,
            "keep": self.keep,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MemoryTurn":
        """Rehydrate a MemoryTurn from ``to_dict`` output."""

        return cls(
            turn_id=payload["turn_id"],
            speaker_id=payload["speaker_id"],
            role=payload["role"],
            public_speech=payload.get("public_speech"),
            private_reflection=payload.get("private_reflection"),
            metadata=payload.get("metadata", {}),
            message_id=payload.get("message_id"),
            status=payload.get("status"),
            keep=payload.get("keep", True),
        )

    def to_chat_message(
        self, *, viewer_id: str, multi_party: bool = False
    ) -> Optional[ChatMessage]:
        """Render the turn as a standard chat completion message."""

        if not self.keep:
            return None

        role = "assistant" if self.speaker_id == viewer_id else "user"
        if self.public_speech:
            content = self.public_speech
            if role == "user" and multi_party:
                speaker = self.metadata.get("speaker_name") or self.speaker_id
                content = f"{speaker}: {content}"
            return ChatMessage(role=role, content=content)

        if self.private_reflection and self.speaker_id == viewer_id:
            return ChatMessage(role="assistant", content=self.private_reflection)

        return None

__all__ = ["MemoryTurn"]
