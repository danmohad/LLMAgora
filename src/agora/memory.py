"""Data structures representing per-agent memory turns."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .llm import ChatMessage

CURRENT_INSTRUCTION_LABEL = "Current instruction"
_TRANSCRIPT_LABELS = (
    CURRENT_INSTRUCTION_LABEL,
    "Other speaker | public statement",
    "Other speaker | public survey response",
    "You | public statement",
    "You | private note",
    "You | public survey response",
    "You | private survey response",
    "You | pre-interview note",
    "You | post-interview note",
)


def render_labeled_content(*, label: str, content: str) -> str:
    """Render transcript metadata in-band while keeping provider-valid roles."""

    return f"[{label}]\n{content}"


def strip_transcript_label_prefix(text: str) -> str:
    """Remove any leading transcript metadata label echoed by a model."""

    if not text:
        return text

    label_pattern = "|".join(
        re.escape(label) for label in sorted(_TRANSCRIPT_LABELS, key=len, reverse=True)
    )
    prefix_pattern = re.compile(
        rf"^\s*\[(?:{label_pattern})\]\s*(?:\r?\n|\s*[:\-]\s*)?"
    )

    stripped = text
    while True:
        updated = prefix_pattern.sub("", stripped, count=1).lstrip()
        if updated == stripped:
            return stripped
        stripped = updated


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

    def to_chat_message(self, *, viewer_id: str) -> Optional[ChatMessage]:
        """Render the turn as a standard chat completion message."""

        if not self.keep:
            return None

        role = "assistant" if self.speaker_id == viewer_id else "user"
        if self.public_speech:
            return ChatMessage(
                role=role,
                content=render_labeled_content(
                    label=self._transcript_label(viewer_id=viewer_id),
                    content=self.public_speech,
                ),
            )

        if self.private_reflection and self.speaker_id == viewer_id:
            return ChatMessage(
                role="assistant",
                content=render_labeled_content(
                    label=self._transcript_label(viewer_id=viewer_id),
                    content=self.private_reflection,
                ),
            )

        return None

    def _transcript_label(self, *, viewer_id: str) -> str:
        """Describe the message's source and purpose for the model."""

        event_type = self.metadata.get("event_type")
        is_self = self.speaker_id == viewer_id

        if event_type == "public_survey" or self.role == "public_survey":
            return (
                "You | public survey response"
                if is_self
                else "Other speaker | public survey response"
            )
        if event_type == "private_survey" or self.role == "private_survey":
            return "You | private survey response"
        if event_type == "pre_interview" or self.role == "pre_interview":
            return "You | pre-interview note"
        if event_type == "post_interview" or self.role == "post_interview":
            return "You | post-interview note"
        if event_type == "private_utterance" or self.role == "reflection":
            return "You | private note"
        return "You | public statement" if is_self else "Other speaker | public statement"

__all__ = [
    "CURRENT_INSTRUCTION_LABEL",
    "MemoryTurn",
    "render_labeled_content",
    "strip_transcript_label_prefix",
]
