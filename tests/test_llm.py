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


def test_format_error_uses_text_on_invalid_json(monkeypatch):
    response = DummyResponse(ValueError("bad"), status_code=500, text="oops")
    dummy = _install_dummy_client(monkeypatch, DummyResponse({"choices": [{"message": {"content": "ok"}}]}))

    client = llm.OpenRouterClient(api_key="key")
    error = client._format_error(response)
    assert "oops" in error
    client.close()
    assert dummy.closed is True
