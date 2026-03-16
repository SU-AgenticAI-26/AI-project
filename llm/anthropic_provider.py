"""
llm/anthropic_provider.py — Anthropic Claude provider.

Uses native tool_use for the orchestrator's complete_with_tools().

"""

from __future__ import annotations
from typing import Optional
import anthropic as _anthropic

from llm.base import LLMProvider, ToolCall


class AnthropicProvider(LLMProvider):
    supports_native_tools = True

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        import os
        self._model  = model
        self._client = _anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    @property
    def name(self) -> str:
        return f"Anthropic ({self._model})"

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, system: str, prompt: str, max_tokens: int = 1500) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def complete_with_tools(
        self,
        system:     str,
        messages:   list[dict],
        tools:      list[dict],
        max_tokens: int = 2000,
    ) -> tuple[str, list[ToolCall]]:
        """Native Anthropic tool_use."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )

        reasoning_text = ""
        tool_calls     = []
        for block in response.content:
            if not hasattr(block, "type"):
                continue
            if block.type == "text":
                reasoning_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id, name=block.name, input=block.input or {}
                ))

        return reasoning_text.strip(), tool_calls

    def raw_response_content(
        self,
        system:     str,
        messages:   list[dict],
        tools:      list[dict],
        max_tokens: int = 2000,
    ):
        """
        Return the raw Anthropic response object.
        Used by the orchestrator to build proper tool_result messages.
        """
        return self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )
