"""
message_bus.py — Typed inter-agent message passing.

Every communication between agents is a Message on this bus.
The bus is the single source of truth for all inter-agent coordination —
the orchestrator reads it to understand what agents have said to each other,
and the app surfaces it as the "Agent Communications" tab.

Message types
─────────────
  task      — orchestrator assigns work to an agent
  result    — agent returns completed work
  feedback  — ValidationAgent sends directed critique to a specific agent
  request   — an agent requests something from another agent
  ack       — an agent acknowledges feedback and states how it will address it
"""

from dataclasses import dataclass, field
from typing import Optional
import uuid
import datetime


@dataclass
class Message:
    sender:       str           # agent name, e.g. "orchestrator", "synthesis_agent"
    recipient:    str           # agent name or "bus" for broadcast
    msg_type:     str           # "task" | "result" | "feedback" | "request" | "ack"
    content:      dict          # payload — structure depends on msg_type
    id:           str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:    str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    in_reply_to:  Optional[str] = None   # id of the message this is a reply to

    def summary(self) -> str:
        """One-line human-readable summary for the UI."""
        ctype = self.content.get("type", "")
        detail = ""
        if self.msg_type == "task":
            detail = self.content.get("instruction", "")[:80]
        elif self.msg_type == "result":
            detail = self.content.get("summary", "")[:80]
        elif self.msg_type == "feedback":
            issues = self.content.get("issues", [])
            detail = f"{len(issues)} issue(s): " + "; ".join(issues[:2])
        elif self.msg_type == "request":
            detail = self.content.get("reason", "")[:80]
        elif self.msg_type == "ack":
            detail = self.content.get("plan", "")[:80]
        return (
            f"[{self.id}] {self.sender} → {self.recipient} "
            f"({self.msg_type}): {detail}"
        )


class MessageBus:
    """
    In-memory message bus. Every agent holds a reference to this bus
    and posts all outbound messages through it.
    """

    def __init__(self):
        self._messages: list[Message] = []

    def send(self, msg: Message) -> Message:
        """Post a message. Returns the message (with its assigned id)."""
        self._messages.append(msg)
        return msg

    def all(self) -> list[Message]:
        return list(self._messages)

    def for_recipient(self, recipient: str) -> list[Message]:
        return [m for m in self._messages if m.recipient == recipient]

    def by_sender(self, sender: str) -> list[Message]:
        return [m for m in self._messages if m.sender == sender]

    def thread(self, root_id: str) -> list[Message]:
        """All messages that are replies to root_id (direct + transitive)."""
        result, queue = [], [root_id]
        seen = set()
        while queue:
            mid = queue.pop()
            for m in self._messages:
                if m.in_reply_to == mid and m.id not in seen:
                    result.append(m)
                    seen.add(m.id)
                    queue.append(m.id)
        return result

    def last_result_from(self, sender: str) -> Optional[Message]:
        results = [m for m in self._messages
                   if m.sender == sender and m.msg_type == "result"]
        return results[-1] if results else None

    def feedback_for(self, recipient: str) -> list[Message]:
        return [m for m in self._messages
                if m.recipient == recipient and m.msg_type == "feedback"]

    def log_lines(self) -> list[str]:
        return [m.summary() for m in self._messages]
