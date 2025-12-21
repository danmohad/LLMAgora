"""LLM client interfaces for Agora agents."""

import json
import os
from typing import Any, Dict, List, Optional, Protocol, Sequence, TypedDict

import httpx

from .survey import build_likert_survey_schema

class ChatMessage(TypedDict):
    """Typed representation of an OpenRouter chat message payload."""

    role: str
    content: str


class LLMClient(Protocol):
    """Protocol for LLM clients so agents remain easily testable."""

    def complete(self, *, messages: Sequence[ChatMessage], model: str, survey_questions: Sequence[str] = None) -> str:
        """Return the chat completion text for the provided conversation."""


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
            raise RuntimeError("OPENROUTER_API_KEY is not configured in the environment")
        self._timeout = timeout
        self._client = httpx.Client(base_url=self._BASE_URL, timeout=timeout)
        self._referer = referer
        self._title = title

    def complete(self, *, messages: Sequence[ChatMessage], model: str, survey_questions: Sequence[str] = None) -> str:
        """Submit a chat completion request and return the LLM's reply."""

        if survey_questions is not None:
            
            survey_schema = build_likert_survey_schema(num_questions=len(survey_questions))
            payload: Dict[str, Any] = {
                "model": model,
                "messages": list(messages),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": survey_schema
                }
            }
        else:
            payload: Dict[str, Any] = {
                "model": model,
                "messages": list(messages),
            }
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


__all__ = ["ChatMessage", "LLMClient", "OpenRouterClient"]
