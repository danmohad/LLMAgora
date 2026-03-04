import pytest

from typing import List, Sequence

from agora.llm import ChatMessage


class StubLLM:
    """Simple stand-in for LLMClient returning a predefined sequence."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.calls: List[dict] = []

    def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        model: str,
        survey_questions: Sequence[str] = None,
        survey_question_groups: dict[str, str] | None = None,
    ) -> str:
        """Return the next canned response, recording the prompt for inspection."""

        self.calls.append({"messages": list(messages), "model": model})
        if not self._responses:
            raise AssertionError("StubLLM ran out of canned responses")
        return self._responses.pop(0)


@pytest.fixture
def stub_llm_factory():
    """Factory fixture that yields pre-loaded StubLLM instances."""

    def _factory(responses: Sequence[str]) -> StubLLM:
        return StubLLM(responses)

    return _factory
