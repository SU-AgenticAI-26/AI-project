"""
llm/ollama_provider.py — Ollama local model provider.

Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1.
Newer versions support tool calling for capable models (llama3.1, mistral-nemo, etc.).
For models/versions that don't support tools, falls back to ReAct text mode.

Requires: pip install openai   (uses the OpenAI client pointed at Ollama)
Or direct HTTP if openai is not installed.

Usage examples:
  OllamaProvider("llama3.1:8b")
  OllamaProvider("mistral-nemo", host="http://192.168.1.10:11434")
  OllamaProvider("qwen2.5:14b")
"""

from __future__ import annotations
from typing import Optional

from llm.base import LLMProvider, ToolCall


class OllamaProvider(LLMProvider):
    supports_native_tools = False  # detected at runtime

    def __init__(self, model: str = "llama3.1:8b", host: str = "http://localhost:11434"):
        self._model = model
        self._host  = host.rstrip("/")
        self._tools_confirmed = None  # None=untested, True=works, False=fallback

        # Try to use the OpenAI client (most reliable for Ollama)
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key="ollama",   # Ollama ignores the key but openai client requires one
                base_url=f"{self._host}/v1",
            )
            self._use_openai_client = True
        except ImportError:
            self._client = None
            self._use_openai_client = False

    @property
    def name(self) -> str:
        return f"Ollama ({self._model} @ {self._host})"

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, system: str, prompt: str, max_tokens: int = 1500) -> str:
        if self._use_openai_client:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
            )
            return (response.choices[0].message.content or "").strip()
        else:
            return self._direct_complete(system, prompt, max_tokens)

    def _direct_complete(self, system: str, prompt: str, max_tokens: int) -> str:
        """Direct HTTP call to Ollama API (no openai package required)."""
        import requests, json
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system",  "content": system},
                {"role": "user",    "content": prompt},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        resp = requests.post(
            f"{self._host}/api/chat", json=payload, timeout=120
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()

    def complete_with_tools(
        self,
        system:     str,
        messages:   list[dict],
        tools:      list[dict],
        max_tokens: int = 2000,
    ) -> tuple[str, list[ToolCall]]:
        # If we already know tools don't work, go straight to ReAct
        if self._tools_confirmed is False:
            return self._react_complete(system, messages, tools, max_tokens)

        # Try native tool calling via OpenAI-compat endpoint
        if self._use_openai_client:
            from llm.openai_provider import _to_openai_tools, _to_openai_messages
            import json

            oai_tools    = _to_openai_tools(tools)
            oai_messages = [{"role": "system", "content": system}] + _to_openai_messages(messages)
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    messages=oai_messages,
                    tools=oai_tools,
                    tool_choice="auto",
                )
                self._tools_confirmed = True
                choice     = response.choices[0]
                msg        = choice.message
                reasoning  = (msg.content or "").strip()
                tool_calls = []
                for tc in (msg.tool_calls or []):
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                    tool_calls.append(ToolCall(
                        id=tc.id, name=tc.function.name, input=args
                    ))
                return reasoning, tool_calls

            except Exception:
                self._tools_confirmed = False

        # Fall back to ReAct
        return self._react_complete(system, messages, tools, max_tokens)
