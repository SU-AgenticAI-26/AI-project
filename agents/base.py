"""
agents/base.py — BaseAgent (provider-agnostic).

Every agent accepts an LLMProvider at construction time.
Agents never import a specific provider — they call self._provider.complete().
"""

import json
import re
from typing import Optional

from llm.base import LLMProvider, ToolCall
from message_bus import Message, MessageBus


class BaseAgent:
    name:          str = "base_agent"
    system_prompt: str = "You are a helpful assistant."

    # ── Orchestrator tool registration ────────────────────────────────────────
    # Subclasses set these three attributes to participate in auto-registration.
    # The Orchestrator reads them to build TOOL_DEFINITIONS, _make_agents(),
    # and the route map — no changes to orchestrator.py needed for new agents.
    tool_name:        str  = ""   # CLI name, e.g. "send_to_my_agent"
    tool_description: str  = ""   # shown to the Orchestrator LLM
    tool_schema:      dict = {"type": "object", "properties": {}, "required": []}

    def __init__(self, provider: LLMProvider):
        self._provider = provider

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def _llm(self, prompt: str, max_tokens: int = 1500, extra_system: str = "") -> str:
        system = self.system_prompt
        if extra_system:
            system = system + "\n\n" + extra_system
        return self._provider.complete(system, prompt, max_tokens=max_tokens)

    def _parse_json(self, text: str) -> Optional[dict]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def _reply(self, to_message: Message, msg_type: str,
               content: dict, bus: MessageBus) -> Message:
        reply = Message(
            sender=self.name, recipient=to_message.sender,
            msg_type=msg_type, content=content, in_reply_to=to_message.id,
        )
        bus.send(reply)
        return reply

    def run(self, message: Message, state: dict, bus: MessageBus) -> Message:
        raise NotImplementedError
