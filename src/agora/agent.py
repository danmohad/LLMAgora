"""Agent definitions for the Agora arena."""

import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

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
        response_instruction: str,
        private_response_instruction: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the agent with identity, backend model, and prompts.

        Args:
            name: Human-friendly label for logging/history.
            model: OpenRouter model identifier.
            llm_client: Backend completion client.
            system_prompt: Optional system message prepended to every call.
            response_instruction: Final user message directing the agent's public reply.
            private_response_instruction: Optional user message directing private reflections.
            agent_id: Override identifier (auto-generated when omitted).
        """

        self.id = agent_id or str(uuid.uuid4())
        self.name = name
        self.model = model
        self._system_prompt = system_prompt
        self._response_instruction = response_instruction
        self._private_instruction = private_response_instruction
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

    @property
    def supports_private_reflection(self) -> bool:
        """Return True when the agent is configured for private reflections."""

        return bool(self._private_instruction)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def response_instruction(self) -> str:
        return self._response_instruction

    @property
    def private_response_instruction(self) -> Optional[str]:
        return self._private_instruction

    def generate_public_speech(self) -> str:
        """Ask the LLM client for this agent's next public response."""

        messages = self._build_messages(final_instruction=self._response_instruction)
        response = self._llm.complete(messages=messages, model=self.model)
        return response.strip()

    def generate_private_reflection(self) -> str:
        """Ask the LLM client for the agent's private reflection."""

        if not self._private_instruction:
            raise RuntimeError("Private reflection requested for agent without instructions")
        messages = self._build_messages(final_instruction=self._private_instruction)
        response = self._llm.complete(messages=messages, model=self.model)
        return response.strip()

    def observe_turn(self, turn: MemoryTurn) -> None:
        """Append a public turn to the agent's personal memory."""

        self._memory.append(turn)

    def reset_memory(self) -> None:
        """Clear the agent's memory (useful before loading history)."""

        self._memory.clear()

    def export_configuration(self) -> Dict[str, Optional[str]]:
        """Return a JSON-serializable representation of the agent's prompts."""

        return {
            "id": self.id,
            "name": self.name,
            "model": self.model,
            "system_prompt": self._system_prompt,
            "response_instruction": self._response_instruction,
            "private_response_instruction": self._private_instruction,
        }

    @classmethod
    def from_configuration(
        cls, config: Dict[str, Optional[str]], llm_client: LLMClient
    ) -> "Agent":
        """Instantiate an agent from ``export_configuration`` output."""

        return cls(
            name=config.get("name") or "",
            model=config.get("model") or "",
            llm_client=llm_client,
            system_prompt=config.get("system_prompt", "") or "",
            response_instruction=config.get("response_instruction", "") or "",
            private_response_instruction=config.get("private_response_instruction"),
            agent_id=config.get("id"),
        )

    def _build_messages(self, *, final_instruction: str) -> Sequence[ChatMessage]:
        """Convert remembered turns into structured chat messages for OpenRouter."""

        messages: List[ChatMessage] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # Use speaker labels only when more than two agents are in the Agora.
        multi_party = bool(self._agora and self._agora.agent_count() > 2)

        for turn in self._memory:
            chat_message = turn.to_chat_message(viewer_id=self.id, multi_party=multi_party)
            if chat_message:
                messages.append(chat_message)

        messages.append({"role": "user", "content": final_instruction})
        return messages


__all__ = ["Agent"]
