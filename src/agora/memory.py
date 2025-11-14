"""Data structures representing per-agent memory turns."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MemoryTurn:
    """Immutable record of a single turn as experienced by an agent."""

    turn_id: int
    speaker_id: str
    role: str
    public_speech: Optional[str] = None
    private_reflection: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> Dict[str, Any]:
        """Serialize only the public portions of the turn for sharing."""

        return {
            "turn_id": self.turn_id,
            "speaker_id": self.speaker_id,
            "role": self.role,
            "public_speech": self.public_speech,
            "metadata": self.metadata,
        }


__all__ = ["MemoryTurn"]
