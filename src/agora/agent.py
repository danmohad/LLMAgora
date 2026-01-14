"""Agent definitions for the Agora arena."""

import re
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

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
        survey_questions: Optional[list[str]] = None,
        survey_base_prompt: Optional[str] = None,
        private_response_instruction: Optional[str] = None,
        private_response_keep: bool = True,
        pre_interview_instruction: Optional[str] = None,
        pre_interview_keep: bool = True,
        post_interview_instruction: Optional[str] = None,
        post_interview_keep: bool = True,
        agent_id: Optional[str] = None,
        opening_instruction: Optional[str] = None,
    ) -> None:
        """
        Initialize the agent with identity, backend model, and prompts.

        Args:
            name: Human-friendly label for logging/history.
            model: OpenRouter model identifier.
            llm_client: Backend completion client.
            system_prompt: Optional system message prepended to every call.
            response_instruction: Final user message directing the agent's public reply.
            opening_instruction: Optional user message directing the opening public remark.
            survey_base_prompt: Survey instructions prepended to question list.
            private_response_instruction: Optional user message directing private reflections.
            private_response_keep: Whether to keep reflections in local memory.
            pre_interview_instruction: Optional pre-run interview prompt.
            pre_interview_keep: Whether to keep the pre-interview in local memory.
            post_interview_instruction: Optional post-run interview prompt.
            post_interview_keep: Whether to keep the post-interview in local memory.
            agent_id: Override identifier (auto-generated when omitted).
        """

        self.id = agent_id or str(uuid.uuid4())
        self.name = name
        self.model = model
        self._system_prompt = system_prompt
        self._response_instruction = response_instruction
        self._opening_instruction = opening_instruction
        self._survey_questions = survey_questions
        self._survey_prompt = survey_base_prompt or ""
        self._private_instruction = private_response_instruction
        self._private_keep = private_response_keep
        self._pre_instruction = pre_interview_instruction
        self._pre_keep = pre_interview_keep
        self._post_instruction = post_interview_instruction
        self._post_keep = post_interview_keep
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
    def private_keep(self) -> bool:
        return self._private_keep

    @property
    def pre_interview_instruction(self) -> Optional[str]:
        return self._pre_instruction

    @property
    def pre_interview_keep(self) -> bool:
        return self._pre_keep

    @property
    def post_interview_instruction(self) -> Optional[str]:
        return self._post_instruction

    @property
    def post_interview_keep(self) -> bool:
        return self._post_keep

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def response_instruction(self) -> str:
        return self._response_instruction

    @property
    def opening_instruction(self) -> Optional[str]:
        return self._opening_instruction

    @property
    def private_response_instruction(self) -> Optional[str]:
        return self._private_instruction

    def _survey_base_prompt(self) -> str:
        if not self._survey_prompt:
            raise RuntimeError(
                "Survey prompt missing; configure 'survey_base_prompt' in prompts.json."
            )
        return self._survey_prompt

    @property
    def do_survey_eval(self) -> bool:
        return self._survey_questions is not None and self._survey_questions != []

    @property
    def survey_questions(self) -> str:
        return self._survey_questions

    def generate_public_speech(self, *, opening: bool = False) -> str:
        """Ask the LLM client for this agent's next public response."""

        instruction = self._response_instruction
        if opening and self._opening_instruction:
            instruction = self._opening_instruction
        messages = self._build_messages(final_instruction=instruction)
        response = self._llm.complete(messages=messages, model=self.model)
        return self._strip_speaker_prefix(response.strip())

    def generate_private_reflection(self) -> str:
        """Ask the LLM client for the agent's private reflection."""

        if not self._private_instruction:
            raise RuntimeError(
                "Private reflection requested for agent without instructions"
            )
        messages = self._build_messages(final_instruction=self._private_instruction)
        response = self._llm.complete(messages=messages, model=self.model)
        return self._strip_speaker_prefix(response.strip())

    def generate_interview_response(self, instruction: str) -> str:
        """Ask the LLM client for an interview response (pre/post)."""

        messages = self._build_messages(final_instruction=instruction)
        response = self._llm.complete(messages=messages, model=self.model)
        return self._strip_speaker_prefix(response.strip())

    def generate_survey_response(self, survey_questions: list[str]) -> str:
        """Ask the LLM client for a survey response with JSON structured format."""
        survey_prompt = self._survey_base_prompt()
        for i, q in enumerate(survey_questions, start=1):
            survey_prompt += f"Q{i}. {q}\n"

        messages = self._build_messages(final_instruction=survey_prompt)
        response = self._llm.complete(
            messages=messages, model=self.model, survey_questions=survey_questions
        )
        return self._strip_speaker_prefix(response.strip())

    def observe_turn(self, turn: MemoryTurn) -> None:
        """Append a public turn to the agent's personal memory."""

        self._memory.append(turn)

    def reset_memory(self) -> None:
        """Clear the agent's memory (useful before loading history)."""

        self._memory.clear()

    def export_configuration(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the agent's prompts."""

        return {
            "id": self.id,
            "name": self.name,
            "model": self.model,
            "system_prompt": self._system_prompt,
            "response_instruction": self._response_instruction,
            "opening_instruction": self._opening_instruction,
            "private_response_instruction": self._private_instruction,
            "private_response_keep": self._private_keep,
            "pre_interview_instruction": self._pre_instruction,
            "pre_interview_keep": self._pre_keep,
            "post_interview_instruction": self._post_instruction,
            "post_interview_keep": self._post_keep,
            "survey_base_prompt": self._survey_prompt,
        }

    @classmethod
    def from_configuration(
        cls, config: Dict[str, Any], llm_client: LLMClient
    ) -> "Agent":
        """Instantiate an agent from ``export_configuration`` output."""

        return cls(
            name=config.get("name") or "",
            model=config.get("model") or "",
            llm_client=llm_client,
            system_prompt=config.get("system_prompt", "") or "",
            response_instruction=config.get("response_instruction", "") or "",
            opening_instruction=config.get("opening_instruction"),
            private_response_instruction=config.get("private_response_instruction"),
            private_response_keep=bool(config.get("private_response_keep", True)),
            pre_interview_instruction=config.get("pre_interview_instruction"),
            pre_interview_keep=bool(config.get("pre_interview_keep", True)),
            post_interview_instruction=config.get("post_interview_instruction"),
            post_interview_keep=bool(config.get("post_interview_keep", True)),
            survey_base_prompt=config.get("survey_base_prompt"),
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
            chat_message = turn.to_chat_message(
                viewer_id=self.id, multi_party=multi_party
            )
            if chat_message:
                messages.append(chat_message)

        messages.append({"role": "user", "content": final_instruction})
        return messages

    def _strip_speaker_prefix(self, text: str) -> str:
        """
        Remove a leading speaker label (e.g., 'Alpha:') if present.

        This prevents assistants from echoing speaker prefixes in their own replies.
        """

        names = {self.name}
        for turn in self._memory:
            name = turn.metadata.get("speaker_name")
            if name:
                names.add(str(name))

        for name in sorted(names, key=len, reverse=True):
            pattern = rf"^\\s*{re.escape(name)}\\s*:\\s*"
            new_text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).lstrip()
            if new_text != text:
                return new_text
        return text


def build_system_prompt(config: Dict[str, Any], *, total_agents: int) -> str:
    """
    Build a system prompt from either a raw string or structured role config.

    Acceptable keys:
    - system_prompt: raw string used verbatim (takes precedence if non-empty)
    - self_role: description of this agent's role (required when system_prompt is absent)
    - perceived_nonself_roles: list of dicts with keys {"name": str, "role": str},
      one entry per other agent (len must equal total_agents - 1)
    """

    raw = (config.get("system_prompt") or "").strip()
    self_role = (config.get("self_role") or "").strip()

    # Require exactly one of raw system_prompt or structured self_role.
    if (raw and self_role) or not (raw or self_role):
        raise ValueError(
            "Agent config requires exactly one of 'system_prompt' or 'self_role'."
        )

    if raw:
        return raw

    perceived = config.get("perceived_nonself_roles")
    if perceived is None:
        return self_role

    if not isinstance(perceived, list):
        raise ValueError("'perceived_nonself_roles' must be a list of dicts.")

    expected_len = max(total_agents - 1, 0)
    if len(perceived) != expected_len:
        raise ValueError(
            f"'perceived_nonself_roles' must have length {expected_len} "
            f"(one entry for each other agent)."
        )

    role_fragments: List[str] = [self_role]
    for entry in perceived:
        if not isinstance(entry, dict):
            raise ValueError("Each perceived_nonself_roles entry must be a dict.")
        name = (entry.get("name") or "").strip()
        role_text = (entry.get("role") or "").strip()
        if not name or not role_text:
            raise ValueError(
                "Each perceived_nonself_roles entry must include 'name' and 'role'."
            )
        role_fragments.append(role_text)

    return " ".join(role_fragments)


__all__ = ["Agent", "build_system_prompt"]
