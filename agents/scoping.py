"""
agents/scoping.py — ScopingAgent

Responsible for decomposing a broad research query into 3-5 focused,
independently searchable sub-questions. Uses its own Claude call with
a system prompt tuned for academic query analysis.

Receives:  task message with {"query": str}
Sends:     result message with {"sub_questions": list, "rationale": str, "summary": str}
"""

from message_bus import Message, MessageBus
from agents.base import BaseAgent


SYSTEM_PROMPT = """\
You are a specialist in academic research scoping. Your role is to analyse a \
broad research question and decompose it into focused, independently searchable \
sub-questions for a graduate-level literature review.

A good decomposition:
- Covers background theory, key methods, applications, known limitations, and open problems
- Produces 3-5 sub-questions that are specific enough to yield targeted search results
- Ensures the sub-questions together cover the full scope of the original query
- Avoids overlap between sub-questions
- Phrases each sub-question as a searchable academic query (not conversational)

Return only valid JSON — no preamble, no markdown fences."""

PROMPT = """\
Research query: "{query}"

Decompose this into 3-5 focused sub-questions for a systematic literature review.

Return exactly this JSON:
{{
  "sub_questions": [
    "sub-question 1",
    "sub-question 2",
    "sub-question 3"
  ],
  "rationale": "One sentence explaining how you split the topic and why."
}}"""


class ScopingAgent(BaseAgent):
    name          = "scoping_agent"
    system_prompt = SYSTEM_PROMPT

    def __init__(self, provider):
        super().__init__(provider)

    def run(self, message: Message, state: dict, bus: MessageBus) -> Message:
        query = message.content.get("query", state.get("query", ""))

        try:
            raw  = self._llm(PROMPT.format(query=query), max_tokens=600)
            data = self._parse_json(raw)

            if data and data.get("sub_questions"):
                sub_questions = data["sub_questions"]
                rationale     = data.get("rationale", "")
            else:
                sub_questions = [query]
                rationale     = "Fallback: could not decompose query."

        except Exception as exc:
            sub_questions = [query]
            rationale     = f"Error during decomposition: {exc}"

        state["sub_questions"] = sub_questions

        return self._reply(
            message,
            msg_type="result",
            content={
                "sub_questions": sub_questions,
                "rationale":     rationale,
                "summary":       f"Decomposed into {len(sub_questions)} sub-questions",
            },
            bus=bus,
        )
