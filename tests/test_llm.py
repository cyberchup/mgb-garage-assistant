import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from mgb_assistant import llm as llm_module
from mgb_assistant.llm import ClaudeLLM, LLMError, OllamaLLM

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}


class FakeClaudeClient:
    """Stands in for anthropic.Anthropic(): records the request, returns a canned response."""

    def __init__(self, text: str = '{"a": 1}', stop_reason: str = "end_turn", error: Exception | None = None):
        self.requests: list[dict] = []
        self.response = SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])
        self.error = error
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def test_claude_request_shape():
    client = FakeClaudeClient()
    assert ClaudeLLM(client=client).complete_json("system text", "user text", SCHEMA) == {"a": 1}
    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["fallbacks"] == "default"
    assert request["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert request["system"] == [{"type": "text", "text": "system text", "cache_control": {"type": "ephemeral"}}]
    assert request["messages"] == [{"role": "user", "content": "user text"}]


def test_claude_model_id_records_model_and_effort():
    assert ClaudeLLM(client=FakeClaudeClient(), effort="high").model_id == "claude:claude-opus-5-5@high"


@pytest.mark.parametrize(
    ("client", "error"),
    [
        (FakeClaudeClient(stop_reason="refusal"), "declined"),
        (FakeClaudeClient(stop_reason="max_tokens"), "ran out of output tokens"),
        (FakeClaudeClient(text="not json"), "isn't JSON"),
        (FakeClaudeClient(error=TypeError("Could not resolve authentication method.")), "ANTHROPIC_API_KEY"),
    ],
)
def test_claude_failures_become_llm_errors(client, error):
    with pytest.raises(LLMError, match=error):
        ClaudeLLM(client=client).complete_json("s", "u", SCHEMA)


def test_claude_does_not_hide_unrelated_type_errors():
    with pytest.raises(TypeError, match="unexpected keyword"):
        ClaudeLLM(client=FakeClaudeClient(error=TypeError("unexpected keyword"))).complete_json("s", "u", SCHEMA)


class FakeHTTPResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ollama_request_shape(monkeypatch):
    sent = {}

    def fake_urlopen(request, timeout):
        sent["url"], sent["body"] = request.full_url, json.loads(request.data)
        return FakeHTTPResponse(json.dumps({"message": {"role": "assistant", "content": '{"a": 2}'}, "done": True}).encode())

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", fake_urlopen)
    result = OllamaLLM("test-model", host="http://box:11434/").complete_json("sys", "usr", SCHEMA)
    assert result == {"a": 2}
    assert sent["url"] == "http://box:11434/api/chat"
    assert sent["body"]["format"] == SCHEMA
    assert sent["body"]["stream"] is False
    assert sent["body"]["options"] == {"temperature": 0, "seed": 0}
    assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user"]


def test_ollama_reports_its_own_error_message(monkeypatch):
    def fake_urlopen(request, timeout):
        body = io.BytesIO(b'{"error": "model \\"x\\" not found, try pulling it first"}')
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, body)

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMError, match="try pulling it first"):
        OllamaLLM("x").complete_json("s", "u", SCHEMA)


def test_ollama_not_running(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMError, match="is it running"):
        OllamaLLM().complete_json("s", "u", SCHEMA)
