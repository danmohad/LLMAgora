import json

import pytest

from agora import llm


class DummyResponse:
    def __init__(self, payload, *, status_code=200, raise_error=None, text="bad"):
        self._payload = payload
        self.status_code = status_code
        self._raise_error = raise_error
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self):
        if self._raise_error:
            raise self._raise_error


class DummyClient:
    def __init__(self, response):
        self._response = response
        self.last_content = None
        self.closed = False

    def post(self, _path, *, headers, content):
        self.last_content = content
        return self._response

    def close(self):
        self.closed = True


def _install_dummy_client(monkeypatch, response):
    dummy = DummyClient(response)

    def fake_client(*_args, **_kwargs):
        return dummy

    monkeypatch.setattr(llm.httpx, "Client", fake_client)
    return dummy


def test_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        llm.OpenRouterClient(api_key=None)


def test_complete_returns_content(monkeypatch):
    response = DummyResponse({"choices": [{"message": {"content": "Hello"}}]})
    dummy = _install_dummy_client(monkeypatch, response)

    client = llm.OpenRouterClient(api_key="key")
    result = client.complete(messages=[{"role": "user", "content": "hi"}], model="m")

    assert result == "Hello"
    payload = json.loads(dummy.last_content)
    assert "response_format" not in payload


def test_complete_survey_includes_schema(monkeypatch):
    response = DummyResponse({"choices": [{"message": {"content": "{\"Q1\": \"Neutral\"}"}}]})
    dummy = _install_dummy_client(monkeypatch, response)

    client = llm.OpenRouterClient(api_key="key")
    result = client.complete(
        messages=[{"role": "user", "content": "hi"}],
        model="m",
        survey_questions=["q1"],
    )

    assert "Q1" in result
    payload = json.loads(dummy.last_content)
    assert "response_format" in payload


def test_complete_survey_uses_reasoning_when_content_missing(monkeypatch):
    response = DummyResponse(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning": '{"Q1": "Agree"}',
                    }
                }
            ]
        }
    )
    _install_dummy_client(monkeypatch, response)

    client = llm.OpenRouterClient(api_key="key")
    result = client.complete(
        messages=[{"role": "user", "content": "hi"}],
        model="m",
        survey_questions=["q1"],
    )

    assert result == '{"Q1": "Agree"}'


def test_complete_survey_uses_reasoning_details_when_content_missing(monkeypatch):
    response = DummyResponse(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning": None,
                        "reasoning_details": [{"text": '{"Q1": "Neutral"}'}],
                    }
                }
            ]
        }
    )
    _install_dummy_client(monkeypatch, response)

    client = llm.OpenRouterClient(api_key="key")
    result = client.complete(
        messages=[{"role": "user", "content": "hi"}],
        model="m",
        survey_questions=["q1"],
    )

    assert result == '{"Q1": "Neutral"}'


def test_complete_survey_uses_question_groups_for_schema(monkeypatch):
    response = DummyResponse({"choices": [{"message": {"content": "{\"Q1\": \"Agree\"}"}}]})
    dummy = _install_dummy_client(monkeypatch, response)

    client = llm.OpenRouterClient(api_key="key")
    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        model="m",
        survey_questions=["q1"],
        survey_question_groups={"Q1": "evaluative"},
    )

    payload = json.loads(dummy.last_content)
    enum_values = payload["response_format"]["json_schema"]["schema"]["properties"]["Q1"]["enum"]
    assert enum_values == llm.build_likert_survey_schema(1)["schema"]["properties"]["Q1"]["enum"]


def test_complete_handles_http_error(monkeypatch):
    response = DummyResponse(
        {"error": "bad"},
        status_code=400,
        raise_error=llm.httpx.HTTPStatusError("bad", request=None, response=None),
    )
    _install_dummy_client(monkeypatch, response)

    client = llm.OpenRouterClient(api_key="key")
    with pytest.raises(RuntimeError):
        client.complete(messages=[{"role": "user", "content": "hi"}], model="m")


def test_complete_requires_choices(monkeypatch):
    response = DummyResponse({"choices": []})
    _install_dummy_client(monkeypatch, response)

    client = llm.OpenRouterClient(api_key="key")
    with pytest.raises(RuntimeError):
        client.complete(messages=[{"role": "user", "content": "hi"}], model="m")


def test_complete_requires_message(monkeypatch):
    response = DummyResponse({"choices": [{"nope": {}}]})
    _install_dummy_client(monkeypatch, response)

    client = llm.OpenRouterClient(api_key="key")
    with pytest.raises(RuntimeError):
        client.complete(messages=[{"role": "user", "content": "hi"}], model="m")


def test_complete_requires_content(monkeypatch):
    response = DummyResponse({"choices": [{"message": {"content": " "}}]})
    _install_dummy_client(monkeypatch, response)

    client = llm.OpenRouterClient(api_key="key")
    with pytest.raises(RuntimeError):
        client.complete(messages=[{"role": "user", "content": "hi"}], model="m")


def test_complete_requires_text_content(monkeypatch):
    response = DummyResponse({"choices": [{"message": {"content": None}}]})
    _install_dummy_client(monkeypatch, response)

    client = llm.OpenRouterClient(api_key="key")
    with pytest.raises(RuntimeError, match="content was not text"):
        client.complete(messages=[{"role": "user", "content": "hi"}], model="m")


def test_complete_survey_requires_text_when_no_content_fallback(monkeypatch):
    response = DummyResponse(
        {"choices": [{"message": {"content": None, "reasoning_details": [{}]}}]}
    )
    _install_dummy_client(monkeypatch, response)

    client = llm.OpenRouterClient(api_key="key")
    with pytest.raises(RuntimeError, match="content was not text"):
        client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="m",
            survey_questions=["q1"],
        )


def test_format_error_uses_text_on_invalid_json(monkeypatch):
    response = DummyResponse(ValueError("bad"), status_code=500, text="oops")
    dummy = _install_dummy_client(monkeypatch, DummyResponse({"choices": [{"message": {"content": "ok"}}]}))

    client = llm.OpenRouterClient(api_key="key")
    error = client._format_error(response)
    assert "oops" in error
    client.close()
    assert dummy.closed is True
