"""Agent definitions for the Agora arena."""

import uuid
from typing import TYPE_CHECKING, List, Optional, Sequence

from .llm import ChatMessage, LLMClient
from .memory import MemoryTurn

if TYPE_CHECKING:  # pragma: no cover - only used for type hints.
    from .agora import Agora


class Agent:
    """Simple agent that only produces public speech."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        llm_client: LLMClient,
        system_prompt: str = "",
        agent_id: Optional[str] = None,
    ) -> None:
        """Initialize the agent with identity, backend model, and prompt."""

        self.id = agent_id or str(uuid.uuid4())
        self.name = name
        self.model = model
        self._system_prompt = system_prompt
        self._llm = llm_client
        self._memory: List[MemoryTurn] = []
        self._agora: Optional["Agora"] = None

    def attach_agora(self, agora: "Agora") -> None:
        """Store the Agora reference so the agent can view history later."""

        self._agora = agora

    def view_history(self) -> List[MemoryTurn]:
        """Return the agent's view of the history, enforcing Agora rules."""

        if not self._agora:
            raise RuntimeError("Agent is not yet part of an Agora")
        return self._agora.history_for_agent(self.id)

    @property
    def memory(self) -> Sequence[MemoryTurn]:
        """Expose an immutable snapshot of the agent's memory."""

        return tuple(self._memory)

    def generate_public_speech(self) -> str:
        """Ask the LLM client for this agent's next public response."""

        messages = self._build_messages()
        response = self._llm.complete(messages=messages, model=self.model)
        return response.strip()

    def observe_turn(self, turn: MemoryTurn) -> None:
        """Append a public turn to the agent's personal memory."""

        self._memory.append(turn)

    def _build_messages(self) -> Sequence[ChatMessage]:
        """Convert remembered turns into a transcript for the LLM."""

        transcript_lines = []
        for turn in self._memory:
            if not turn.public_speech:
                continue
            speaker = turn.metadata.get("speaker_name") or turn.speaker_id
            transcript_lines.append(f"{speaker}: {turn.public_speech}")
        conversation = "\n".join(transcript_lines) if transcript_lines else "No prior conversation."
        user_prompt = (
            f"You are agent '{self.name}'.\n"
            f"Conversation so far:\n{conversation}\n"
            "Respond with your next public utterance."
        )
        messages: List[ChatMessage] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages


__all__ = ["Agent"]
