"""
llm/openai_provider.py — OpenAI-compatible provider.

Covers:
  - OpenAI (api.openai.com)
  - Azure OpenAI
  - llama.cpp server  (--host 0.0.0.0 --port 8080)  base_url=http://localhost:8080/v1
  - LM Studio         base_url=http://localhost:1234/v1
  - vLLM              base_url=http://localhost:8000/v1
  - Any OpenAI-compatible endpoint

Uses native function calling for orchestrator tool use.
Falls back to ReAct text mode if the model/server doesn't support tools
(detected by exception on first attempt).
"""

from __future__ import annotations
from typing import Optional
import json

from llm.base import LLMProvider, ToolCall


def _to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool schema format to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t.get("description", ""),
                "parameters":  t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in anthropic_tools
    ]


def _block_type(block) -> str:
    """Get type from either a dict or an Anthropic SDK content block object."""
    if isinstance(block, dict):
        return block.get("type", "")
    return getattr(block, "type", "")


def _block_get(block, key: str, default=""):
    """Get a field from either a dict or an Anthropic SDK content block object."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _to_openai_messages(anthropic_messages: list[dict]) -> list[dict]:
    """
    Convert Anthropic-style messages (which may contain tool_result blocks,
    or Anthropic SDK content block objects) to OpenAI-style messages.
    """
    out = []
    for msg in anthropic_messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            if role == "assistant":
                text_parts = []
                tool_calls = []
                for block in content:
                    bt = _block_type(block)
                    if bt == "text":
                        text_parts.append(_block_get(block, "text"))
                    elif bt == "tool_use":
                        raw_input = _block_get(block, "input", {})
                        tool_calls.append({
                            "id":   _block_get(block, "id", ""),
                            "type": "function",
                            "function": {
                                "name":      _block_get(block, "name", ""),
                                "arguments": json.dumps(raw_input if isinstance(raw_input, dict) else {}),
                            },
                        })
                assistant_msg: dict = {"role": "assistant"}
                if text_parts:
                    assistant_msg["content"] = " ".join(text_parts)
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                if assistant_msg.get("content") or assistant_msg.get("tool_calls"):
                    out.append(assistant_msg)

            elif role == "user":
                for block in content:
                    bt = _block_type(block)
                    if bt == "tool_result":
                        result_content = _block_get(block, "content", "")
                        if isinstance(result_content, list):
                            result_content = " ".join(
                                _block_get(b, "text", "") for b in result_content
                            )
                        out.append({
                            "role":         "tool",
                            "tool_call_id": _block_get(block, "tool_use_id", ""),
                            "content":      str(result_content),
                        })
                    elif bt == "text":
                        out.append({"role": "user", "content": _block_get(block, "text", "")})

    return out


class OpenAIProvider(LLMProvider):
    supports_native_tools = True

    def __init__(
        self,
        model:    str            = "gpt-4o",
        api_key:  Optional[str] = None,
        base_url: Optional[str] = None,   # override for local / Azure endpoints
        label:    str           = "",
    ):
        import os
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self._model   = model
        self._label   = label or model
        self._client  = OpenAI(
            api_key  = api_key  or os.getenv("OPENAI_API_KEY", "not-needed"),
            base_url = base_url or os.getenv("OPENAI_BASE_URL") or None,
        )
        self._tools_confirmed = None   # None=untested, True=works, False=fallback

    @property
    def name(self) -> str:
        return f"OpenAI-compat ({self._label})"

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, system: str, prompt: str, max_tokens: int = 1500) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def complete_with_tools(
        self,
        system:     str,
        messages:   list[dict],
        tools:      list[dict],
        max_tokens: int = 2000,
    ) -> tuple[str, list[ToolCall]]:
        # Fall back to ReAct if we already know this model doesn't support tools
        if self._tools_confirmed is False:
            return self._react_complete(system, messages, tools, max_tokens)

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
        except Exception:
            # Tool calling not supported — switch to ReAct permanently
            self._tools_confirmed = False
            return self._react_complete(system, messages, tools, max_tokens)

        choice      = response.choices[0]
        msg         = choice.message
        reasoning   = (msg.content or "").strip()
        tool_calls  = []

        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            tool_calls.append(ToolCall(
                id=tc.id, name=tc.function.name, input=args
            ))

        return reasoning, tool_calls
