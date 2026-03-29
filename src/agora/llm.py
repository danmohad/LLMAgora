"""LLM client interfaces for Agora agents."""

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, TypedDict

import httpx

from .survey import (
    SURVEY_GROUP_DELIBERATIVE,
    build_likert_survey_schema,
    build_survey_response_schema,
)


class ChatMessage(TypedDict):
    """Typed representation of an OpenRouter chat message payload."""

    role: str
    content: str


class LLMClient(Protocol):
    """Protocol for LLM clients so agents remain easily testable."""

    def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        model: str,
        survey_questions: Sequence[str] = None,
        survey_question_groups: Mapping[str, str] | None = None,
    ) -> str:
        """Return the chat completion text for the provided conversation."""


def build_completion_payload(
    *,
    messages: Sequence[ChatMessage],
    model: str,
    survey_questions: Sequence[str] = None,
    survey_question_groups: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Build the exact chat-completions payload sent to the provider."""

    payload: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
    }
    if survey_questions is None:
        return payload

    if survey_question_groups:
        full_question_groups = {
            f"Q{i}": survey_question_groups.get(f"Q{i}", SURVEY_GROUP_DELIBERATIVE)
            for i in range(1, len(survey_questions) + 1)
        }
        survey_schema = build_survey_response_schema(full_question_groups)
    else:
        survey_schema = build_likert_survey_schema(num_questions=len(survey_questions))

    payload["response_format"] = {
        "type": "json_schema",
        "json_schema": survey_schema,
    }
    return payload


class OpenRouterClient:
    """Thin wrapper around the OpenRouter chat completion endpoint."""

    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        timeout: float = 30.0,
        referer: str = "https://github.com/danmohad/LLMAgora",
        title: str = "LLM Agora",
    ) -> None:
        """Initialize the HTTP client, grabbing the API key from the environment."""

        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured in the environment"
            )
        self._timeout = timeout
        self._client = httpx.Client(base_url=self._BASE_URL, timeout=timeout)
        self._referer = referer
        self._title = title

    def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        model: str,
        survey_questions: Sequence[str] = None,
        survey_question_groups: Mapping[str, str] | None = None,
    ) -> str:
        """Submit a chat completion request and return the LLM's reply. If 'survey_questions' are supplied, then a structured response is requested."""

        payload = build_completion_payload(
            messages=messages,
            model=model,
            survey_questions=survey_questions,
            survey_question_groups=survey_question_groups,
        )
        response = self._client.post(
            "/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "HTTP-Referer": self._referer,
                "X-Title": self._title,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            content=json.dumps(payload),
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(self._format_error(response)) from exc

        payload = response.json()
        choices: List[dict] = payload.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter response did not include any choices")
        message = choices[0].get("message")
        if not message:
            raise RuntimeError("OpenRouter response missing 'message'")
        content = message.get("content", "").strip()
        if not content:
            raise RuntimeError("OpenRouter returned empty content")
        return content

    def _format_error(self, response: httpx.Response) -> str:
        """Produce a readable error message from a non-2xx response."""

        detail: Any
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        return f"OpenRouter request failed ({response.status_code}): {detail}"

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._client.close()

__all__ = [
    "ChatMessage",
    "LLMClient",
    "OpenRouterClient",
    "build_completion_payload",
]
