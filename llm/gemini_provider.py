"""
llm/gemini_provider.py — Google Gemini provider.

Supports both SDK generations:
  - google-genai    (new, >=1.0):  pip install google-genai
  - google-generativeai (legacy):  pip install google-generativeai

The provider detects which is installed at import time and uses the
appropriate API. If neither is installed, an ImportError is raised with
install instructions.

Native function calling is used for the orchestrator's tool loop.
Falls back to the base-class ReAct text mode if tool calling fails.
"""

from __future__ import annotations
from typing import Optional
import json

from llm.base import LLMProvider, ToolCall

# ── SDK detection ─────────────────────────────────────────────────────────────

def _detect_sdk():
    """
    Returns ("new", module) for google-genai,
            ("legacy", module) for google-generativeai,
            raises ImportError if neither found.
    """
    try:
        import google.genai as genai
        return "new", genai
    except ImportError:
        pass
    try:
        import google.generativeai as genai
        return "legacy", genai
    except ImportError:
        pass
    raise ImportError(
        "No Gemini SDK found. Install one of:\n"
        "  pip install google-genai          # recommended (new SDK)\n"
        "  pip install google-generativeai   # legacy SDK"
    )


# ── Tool schema conversion ────────────────────────────────────────────────────

def _to_gemini_tools_new(anthropic_tools: list[dict], genai) -> list:
    """Build tool list for google-genai SDK."""
    from google.genai import types as gtypes

    declarations = []
    for t in anthropic_tools:
        schema = t.get("input_schema", {"type": "object", "properties": {}})
        declarations.append(gtypes.FunctionDeclaration(
            name=t["name"],
            description=t.get("description", ""),
            parameters=schema,
        ))
    return [gtypes.Tool(function_declarations=declarations)]


def _to_gemini_tools_legacy(anthropic_tools: list[dict], genai) -> list:
    """Build tool list for google-generativeai SDK."""
    try:
        from google.generativeai.types import FunctionDeclaration, Tool
    except ImportError:
        return []

    declarations = []
    for t in anthropic_tools:
        schema = t.get("input_schema", {"type": "object", "properties": {}})
        declarations.append(FunctionDeclaration(
            name=t["name"],
            description=t.get("description", ""),
            parameters=schema,
        ))
    return [Tool(function_declarations=declarations)]


# ── Provider ──────────────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    supports_native_tools = True

    def __init__(self, model: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        import os
        sdk_ver, genai = _detect_sdk()
        self._sdk     = sdk_ver
        self._genai   = genai
        self._model   = model
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        self._tools_confirmed = None   # None=untested, True=works, False=fallback

        if sdk_ver == "new":
            self._client = genai.Client(api_key=self._api_key)
        else:
            genai.configure(api_key=self._api_key)
            self._client = None   # legacy SDK uses module-level calls

    @property
    def name(self) -> str:
        return f"Google Gemini ({self._model}, {self._sdk} SDK)"

    @property
    def model_id(self) -> str:
        return self._model

    # ── complete ──────────────────────────────────────────────────────────

    def complete(self, system: str, prompt: str, max_tokens: int = 1500) -> str:
        if self._sdk == "new":
            return self._complete_new(system, prompt, max_tokens)
        else:
            return self._complete_legacy(system, prompt, max_tokens)

    def _complete_new(self, system: str, prompt: str, max_tokens: int) -> str:
        from google.genai import types as gtypes
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
        )
        return (response.text or "").strip()

    def _complete_legacy(self, system: str, prompt: str, max_tokens: int) -> str:
        model = self._genai.GenerativeModel(
            model_name=self._model,
            system_instruction=system,
        )
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens},
        )
        return (response.text or "").strip()

    # ── complete_with_tools ───────────────────────────────────────────────

    def complete_with_tools(
        self,
        system:     str,
        messages:   list[dict],
        tools:      list[dict],
        max_tokens: int = 2000,
    ) -> tuple[str, list[ToolCall]]:
        if self._tools_confirmed is False:
            return self._react_complete(system, messages, tools, max_tokens)

        # Flatten conversation history to a single prompt
        # (Gemini's multi-turn API differs between SDK versions; single-turn is reliable)
        prompt = self._flatten_messages(messages)

        try:
            if self._sdk == "new":
                reasoning, tool_calls = self._tools_new(system, prompt, tools, max_tokens)
            else:
                reasoning, tool_calls = self._tools_legacy(system, prompt, tools, max_tokens)
            self._tools_confirmed = True
            return reasoning, tool_calls
        except Exception:
            self._tools_confirmed = False
            return self._react_complete(system, messages, tools, max_tokens)

    def _tools_new(self, system: str, prompt: str, tools: list[dict],
                   max_tokens: int) -> tuple[str, list[ToolCall]]:
        from google.genai import types as gtypes

        gemini_tools = _to_gemini_tools_new(tools, self._genai)
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=system,
                tools=gemini_tools,
                max_output_tokens=max_tokens,
            ),
        )
        return self._parse_response_new(response)

    def _tools_legacy(self, system: str, prompt: str, tools: list[dict],
                      max_tokens: int) -> tuple[str, list[ToolCall]]:
        gemini_tools = _to_gemini_tools_legacy(tools, self._genai)
        model = self._genai.GenerativeModel(
            model_name=self._model,
            system_instruction=system,
            tools=gemini_tools,
        )
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens},
        )
        return self._parse_response_legacy(response)

    # ── Response parsers ──────────────────────────────────────────────────

    def _parse_response_new(self, response) -> tuple[str, list[ToolCall]]:
        reasoning  = ""
        tool_calls = []
        for part in (response.candidates[0].content.parts if response.candidates else []):
            if hasattr(part, "text") and part.text:
                reasoning += part.text
            elif hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                try:
                    args = dict(fc.args) if fc.args else {}
                except Exception:
                    args = {}
                tool_calls.append(ToolCall.make(name=fc.name, input_dict=args))
        return reasoning.strip(), tool_calls

    def _parse_response_legacy(self, response) -> tuple[str, list[ToolCall]]:
        reasoning  = ""
        tool_calls = []
        for part in response.parts:
            if hasattr(part, "text") and part.text:
                reasoning += part.text
            elif hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                try:
                    args = dict(fc.args)
                except Exception:
                    args = {}
                tool_calls.append(ToolCall.make(name=fc.name, input_dict=args))
        return reasoning.strip(), tool_calls

    # ── Helpers ───────────────────────────────────────────────────────────

    def _flatten_messages(self, messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                text_bits = []
                for block in content:
                    if isinstance(block, dict):
                        t = block.get("type", "")
                        if t == "text":
                            text_bits.append(block.get("text", ""))
                        elif t == "tool_result":
                            text_bits.append(f"OBSERVATION: {block.get('content', '')}")
                    elif isinstance(block, str):
                        text_bits.append(block)
                    elif hasattr(block, "type"):
                        if block.type == "text":
                            text_bits.append(block.text)
                        elif block.type == "tool_result":
                            text_bits.append(f"OBSERVATION: {block.content}")
                content = "\n".join(text_bits)
            parts.append(str(content))
        return "\n\n".join(p for p in parts if p)
