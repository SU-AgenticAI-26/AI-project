"""
llm/base.py — Abstract LLMProvider interface.

Every provider (Anthropic, OpenAI, Gemini, Ollama, llama.cpp) implements
this interface. Agents only depend on this interface — they never import
a specific provider directly.

Two methods:
  complete(system, prompt, max_tokens) -> str
    Standard text completion. Used by all specialist agents.

  complete_with_tools(system, messages, tools) -> (text, tool_calls)
    Tool-use / function-calling completion. Used by the orchestrator.
    Providers that support native tool calling implement it natively.
    Providers that don't (Ollama, llama.cpp without tool support) fall
    back to a text-based ReAct loop defined in this base class.

ToolCall dataclass is provider-agnostic: {id, name, input_dict}.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import json
import re
import uuid


@dataclass
class ToolCall:
    """Provider-agnostic tool call result."""
    id:    str
    name:  str
    input: dict   # parsed JSON arguments

    @staticmethod
    def make(name: str, input_dict: dict) -> "ToolCall":
        return ToolCall(id=str(uuid.uuid4())[:8], name=name, input=input_dict)


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    Subclasses must implement complete() and, optionally, complete_with_tools().
    If complete_with_tools() is not overridden, the ReAct text fallback is used.
    """

    # Set to True in providers that implement native tool/function calling
    supports_native_tools: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name for display."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model identifier string."""

    @abstractmethod
    def complete(
        self,
        system:     str,
        prompt:     str,
        max_tokens: int = 1500,
    ) -> str:
        """
        Single-turn text completion.
        Returns the model's response as a plain string.
        Raises on hard failure.
        """

    def complete_with_tools(
        self,
        system:     str,
        messages:   list[dict],   # OpenAI-style [{role, content}]
        tools:      list[dict],   # Anthropic-style tool schemas
        max_tokens: int = 2000,
    ) -> tuple[str, list[ToolCall]]:
        """
        Tool-calling completion.
        Returns (reasoning_text, list[ToolCall]).

        Default implementation: text-based ReAct loop.
        Override in providers that support native function calling.
        """
        return self._react_complete(system, messages, tools, max_tokens)

    # ── ReAct text-based fallback ─────────────────────────────────────────

    _REACT_SYSTEM_SUFFIX = """
You must respond using this EXACT format — no deviations:

THOUGHT: <your reasoning about what to do next>
ACTION: <exact tool name from the list>
ACTION_INPUT: <valid JSON object with the tool's arguments>

Available tools:
{tool_list}

Rules:
- Always include THOUGHT, ACTION, and ACTION_INPUT.
- ACTION must be exactly one of the tool names listed.
- ACTION_INPUT must be valid JSON.
- Do not add any text after ACTION_INPUT.
- If you want to finish, use ACTION: finish and ACTION_INPUT: {{}}
"""

    _REACT_OBSERVATION_TEMPLATE = "OBSERVATION: {result}\n\nContinue with THOUGHT:"

    _ACTION_RE   = re.compile(r"ACTION:\s*(\w+)", re.IGNORECASE)
    _INPUT_RE    = re.compile(r"ACTION_INPUT:\s*(\{.*?\}|\[.*?\])", re.DOTALL | re.IGNORECASE)
    _THOUGHT_RE  = re.compile(r"THOUGHT:\s*(.*?)(?=ACTION:|$)", re.DOTALL | re.IGNORECASE)

    def _build_tool_list(self, tools: list[dict]) -> str:
        lines = []
        for t in tools:
            props = t.get("input_schema", {}).get("properties", {})
            param_desc = ", ".join(
                f"{k} ({v.get('type','any')}): {v.get('description','')[:60]}"
                for k, v in props.items()
            )
            lines.append(f"  {t['name']}: {t.get('description','')[:100]}")
            if param_desc:
                lines.append(f"    params: {param_desc}")
        return "\n".join(lines)

    def _react_complete(
        self,
        system:     str,
        messages:   list[dict],
        tools:      list[dict],
        max_tokens: int,
    ) -> tuple[str, list[ToolCall]]:
        """
        Run a single ReAct step: given current conversation history,
        produce one THOUGHT/ACTION/ACTION_INPUT block.
        Returns (thought_text, [ToolCall]) or ("", []) if parsing fails.
        """
        tool_list   = self._build_tool_list(tools)
        react_system = system + self._REACT_SYSTEM_SUFFIX.format(tool_list=tool_list)

        # Convert messages to a single prompt string for providers that
        # only support system+prompt (not multi-turn history)
        prompt_parts = []
        for msg in messages:
            role    = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle structured content blocks (tool results etc.)
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            parts.append(f"OBSERVATION: {block.get('content','')}")
                    elif isinstance(block, str):
                        parts.append(block)
                content = "\n".join(parts)
            prefix = "Assistant:" if role == "assistant" else "User:"
            prompt_parts.append(f"{prefix}\n{content}")

        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"

        try:
            raw = self.complete(react_system, prompt, max_tokens=max_tokens)
        except Exception as exc:
            return f"(ReAct completion error: {exc})", []

        # Parse THOUGHT
        thought_match = self._THOUGHT_RE.search(raw)
        thought       = thought_match.group(1).strip() if thought_match else ""

        # Parse ACTION
        action_match = self._ACTION_RE.search(raw)
        if not action_match:
            return thought, []
        action = action_match.group(1).strip()

        # Parse ACTION_INPUT
        input_match = self._INPUT_RE.search(raw)
        try:
            action_input = json.loads(input_match.group(1)) if input_match else {}
        except json.JSONDecodeError:
            action_input = {}

        tool_call = ToolCall.make(name=action, input_dict=action_input)
        return thought, [tool_call]
