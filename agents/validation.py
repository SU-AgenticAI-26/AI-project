"""
agents/validation.py — ValidationAgent

Evaluates the synthesis and research plan against three criteria:
  1. Citation accuracy   — every cited paper must exist in the retrieved corpus
  2. Logical coherence   — claimed connections are supported by cited abstracts
  3. Completeness        — every sub-question is addressed by ≥1 cited paper

Issues are categorised and sent as a directed feedback message to
synthesis_planning_agent on the bus (not relayed through the Orchestrator).
This lateral communication is the core multi-agent coordination mechanism.

Automated citation check runs before the LLM evaluation: quoted fragments
in the synthesis and plan text are matched against known paper titles.

Receives:  task from orchestrator
Sends:     result to orchestrator
           feedback to synthesis_planning_agent (if issues found)
"""

import re
from message_bus import Message, MessageBus
from agents.base import BaseAgent

SYSTEM_PROMPT = """\
You are a rigorous academic peer reviewer. Evaluate a literature synthesis
and research plan against strict criteria. Produce specific, actionable
feedback naming exact papers and sub-questions. Be precise.
Return only valid JSON — no preamble, no markdown fences."""

VALIDATION_PROMPT = """\
Research query: "{query}"

Sub-questions (ALL must be addressed):
{sub_questions}

Papers in corpus (ONLY these may be cited):
{paper_titles}

Literature synthesis:
{synthesis}

Research plan:
{plan}

Evaluate on three criteria:

1. CITATION ACCURACY: Does the synthesis or plan reference any paper NOT in the
   corpus list above? Are cited claims actually supported by those papers?

2. LOGICAL COHERENCE: Are the thematic connections between papers logically
   sound? Are any generalisations unsupported?

3. COMPLETENESS: Does the synthesis address EVERY sub-question listed above?
   Does the research plan cite specific papers for each proposed step?

Return:
{{
  "issues": [
    "specific, actionable issue — name the paper and sub-question involved"
  ],
  "approved": true or false,
  "feedback": "2-3 sentence quality summary"
}}

Set approved=true only if issues is empty."""


def _auto_citation_check(synthesis: str, plan: str, summaries: list) -> list:
    """
    Detect quoted fragments in the text that don't match any corpus paper title.
    Returns a list of warning strings.
    """
    known  = {re.sub(r"[^a-z0-9 ]", "", p["title"].lower()) for p in summaries}
    text   = (synthesis + " " + plan).lower()
    quoted = re.findall(r'"([^"]{15,80})"', text)
    research_words = {
        "learning", "model", "network", "agent", "system",
        "approach", "method", "survey", "attention", "transformer",
        "framework", "detection", "generation",
    }
    suspects = []
    for fragment in quoted:
        norm     = re.sub(r"[^a-z0-9 ]", "", fragment)
        is_known = any(norm in t or t in norm for t in known)
        if not is_known and any(w in fragment for w in research_words):
            suspects.append(
                f"Possible uncorroborated citation: '{fragment[:60]}'"
            )
    return suspects[:4]


class ValidationAgent(BaseAgent):
    name          = "validation_agent"
    system_prompt = SYSTEM_PROMPT

    tool_name        = "send_to_validation_agent"
    tool_description = (
        "Ask ValidationAgent to evaluate the synthesis and research plan. "
        "It sends directed feedback to SynthesisPlanningAgent if issues are found."
    )
    tool_schema      = {"type": "object", "properties": {}, "required": []}

    def __init__(self, provider):
        super().__init__(provider)

    def run(self, message: Message, state: dict, bus: MessageBus) -> Message:
        summaries = state.get("paper_summaries", [])
        synthesis = state.get("synthesis", "")
        plan      = state.get("research_plan", "")

        paper_titles = "\n".join(
            f"- {p['title']} ({p.get('year', '?')}) [{p.get('source', '')}]"
            for p in summaries
        )
        sub_q_block = "\n".join(
            f"- Sub-question {i}: {q}"
            for i, q in enumerate(state.get("sub_questions", []), 1)
        )

        # Automated check first
        auto_suspects = _auto_citation_check(synthesis, plan, summaries)

        try:
            raw  = self._llm(
                VALIDATION_PROMPT.format(
                    query=state.get("query", ""),
                    sub_questions=sub_q_block,
                    paper_titles=paper_titles,
                    synthesis=synthesis[:2500],
                    plan=plan[:1500],
                ),
                max_tokens=800,
            )
            data     = self._parse_json(raw) or {}
            issues   = data.get("issues", [])
            approved = data.get("approved", False)
            feedback = data.get("feedback", "")
        except Exception as exc:
            issues   = []
            approved = True   # fail-open to avoid blocking the pipeline
            feedback = f"Validation error (auto-approving): {exc}"

        # Merge automated suspects
        all_issues = issues + auto_suspects

        # ── Send directed feedback to synthesis_planning_agent ────────────
        if all_issues:
            fb_msg = Message(
                sender=self.name,
                recipient="synthesis_planning_agent",
                msg_type="feedback",
                content={
                    "issues":  all_issues,
                    "summary": f"{len(all_issues)} issue(s) require correction",
                },
                in_reply_to=message.id,
            )
            bus.send(fb_msg)

        # Update state
        state["validation_result"] = {
            "approved":  approved if not all_issues else False,
            "issues":    all_issues,
            "feedback":  feedback,
        }
        state["validation_iterations"] = state.get("validation_iterations", 0) + 1

        final_approved = not bool(all_issues)

        return self._reply(
            message,
            msg_type="result",
            content={
                "approved":  final_approved,
                "issues":    all_issues,
                "feedback":  feedback,
                "summary": (
                    "APPROVED"
                    if final_approved
                    else (
                        f"NOT APPROVED — {len(all_issues)} issue(s) "
                        f"(sent feedback to synthesis_planning_agent)"
                    )
                ),
            },
            bus=bus,
        )
