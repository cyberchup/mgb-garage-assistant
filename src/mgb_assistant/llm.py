"""LLM providers behind one interface: a system prompt, a user message and a JSON schema in, parsed JSON out.

Claude is the optional hosted provider; Ollama runs a local model, so everything can stay offline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol

CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_EFFORT = "medium"  # Opus 5.5's own default; set explicitly so runs are reproducible
OLLAMA_MODEL = os.environ.get("MGB_OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class LLMError(RuntimeError):
    """The model gave no usable answer: unreachable, refused, cut off, or not the JSON asked for."""


class LLM(Protocol):
    model_id: str  # recorded with every eval run

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]: ...


def _parse(text: str | None, provider: str) -> dict[str, Any]:
    if not text:
        raise LLMError(f"{provider} returned no text")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"{provider} returned text that isn't JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMError(f"{provider} returned JSON that isn't an object")
    return data


class ClaudeLLM:
    """Claude through the Anthropic API. Credentials come from ANTHROPIC_API_KEY or `ant auth login`."""

    def __init__(self, model: str = CLAUDE_MODEL, *, effort: str = CLAUDE_EFFORT, max_tokens: int = 16000, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.model_id = f"claude:{model}@{effort}"

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        import anthropic

        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                # If a safety classifier declines, the API reruns the request on its recommended fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                # The system prompt is the same on every question, so it can be cached.
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            )
        except TypeError as exc:
            if "authentication" not in str(exc):
                raise
            # With no credentials at all, the SDK raises TypeError before any request goes out.
            raise LLMError("no Anthropic credentials; set ANTHROPIC_API_KEY or run `ant auth login`") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("Claude rejected the credentials; set ANTHROPIC_API_KEY or run `ant auth login`") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("can't reach the Anthropic API; check the internet connection") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        if response.stop_reason == "refusal":
            raise LLMError("Claude declined to answer this question")
        if response.stop_reason == "max_tokens":
            raise LLMError("Claude ran out of output tokens before finishing")
        return _parse(next((b.text for b in response.content if b.type == "text"), None), "Claude")


class OllamaLLM:
    """A local model served by Ollama. Temperature 0 and a fixed seed keep eval runs repeatable."""

    def __init__(self, model: str = OLLAMA_MODEL, *, host: str = OLLAMA_HOST, timeout: float = 600.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.model_id = f"ollama:{model}"

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "seed": 0},
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                reply = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except json.JSONDecodeError:
                pass
            raise LLMError(f"Ollama error {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"can't reach Ollama at {self.host}; is it running?") from exc
        return _parse((reply.get("message") or {}).get("content"), "Ollama")
