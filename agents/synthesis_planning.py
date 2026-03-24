"""
agents/synthesis_planning.py — SynthesisPlanningAgent (merged Synthesis + Planning)

Produces the literature review and research plan in a single agent,
matching the proposal's "Synthesis & Planning" specialist role.

What makes this genuinely agentic
──────────────────────────────────
  - Reads directed feedback from ValidationAgent off the bus
  - Makes a dedicated LLM call to produce an acknowledgement (ack) message
    describing exactly how each issue will be addressed
  - Posts the ack directly to validation_agent on the bus (lateral communication,
    not relayed through the Orchestrator)
  - Revises both synthesis and plan incorporating those corrections

Synthesis methodology
──────────────────────
  Groups papers into 2-4 thematic clusters.
  Identifies ≥1 direct contradiction between papers and flags it.
  Identifies sub-questions not covered by ≥2 papers.

Research plan
──────────────
  3-5 research directions each citing ≥2 specific papers.
  Links each direction to a stated future-work item from the corpus.
  Prose narrative + structured steps.

Receives:  task from orchestrator; feedback from validation_agent
Sends:     result to orchestrator; ack to validation_agent
"""

from message_bus import Message, MessageBus
from agents.base import BaseAgent

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an academic literature synthesis and research planning specialist.
You produce coherent, thematic literature reviews and concrete research plans
grounded in a provided set of paper summaries. You are rigorous about citation
accuracy — only reference papers that appear in the provided summaries.
You identify genuine themes, note real contradictions, and characterise research
gaps with precision. Plans must cite specific papers by exact title.
Return only valid JSON — no preamble, no markdown fences."""

SYNTHESIS_PLANNING_PROMPT = """\
Research query: "{query}"

Sub-questions to address:
{sub_questions}

{feedback_block}
Paper summaries ({n_papers} papers):
{paper_block}

Produce a thematic synthesis AND a research plan. Return:
{{
  "themes": [
    {{
      "theme_name":   "short label",
      "description":  "2-3 sentences — which papers fall here and why",
      "paper_titles": ["exact title from the list above"]
    }}
  ],
  "synthesis": "3-5 paragraph prose literature review. Use inline citations in the form [N] where N is the paper number from the list above (e.g. 'attention mechanisms [3] have been shown to...'. Only reference papers listed above. Address every sub-question explicitly.",
  "contradictions": ["specific contradiction between two named papers — or empty list"],
  "gaps": "2-3 paragraph discussion of what the literature is missing",
  "coverage_check": {{
    "covered":   ["sub-question text"],
    "uncovered": ["sub-question text — fewer than 2 papers address this"]
  }},
  "proposed_direction": "1-2 sentence statement of the most promising research direction",
  "research_steps": [
    {{
      "step":             1,
      "title":            "step title",
      "description":      "what to do and why",
      "grounding_papers": ["exact paper title from the list above"],
      "future_work_link": "future-work item from one of the grounding papers"
    }}
  ],
  "risks_and_mitigations": "brief discussion of methodological risks",
  "research_plan_prose": "3-4 paragraph narrative of the full research plan"
}}"""

ACK_PROMPT = """\
You are about to revise a literature review and research plan based on
feedback from a validation agent.

Feedback issues:
{issues}

State concisely how you will address each issue. Return only JSON:
{{
  "will_fix":   ["issue — how you will fix it"],
  "cannot_fix": ["issue — reason (e.g. insufficient corpus coverage)"],
  "plan":       "one sentence summary of your revision approach"
}}"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _paper_block(papers: list) -> str:
    """
    Build a numbered evidence block for the synthesis prompt.
    Accepts either extracted_papers format (from ReadingExtractionAgent,
    preferred) or the legacy paper_summaries format (from SearchReadingAgent).
    Detected by presence of "research_problem" key.
    """
    lines = []
    for i, p in enumerate(papers, 1):
        if "research_problem" in p:
            # extracted_papers format — richer schema
            lines.append(
                f"[{i}] {p['title']} ({p.get('year', '?')}) "
                f"[{p.get('source', '?')}] — {p.get('citation_count', 0)} citations\n"
                f"    Research problem: {p.get('research_problem', '')}\n"
                f"    Key claims:       {'; '.join(p.get('key_claims', []))}\n"
                f"    Methods:          {p.get('methodology', '')}\n"
                f"    Findings:         {p.get('findings', '')}\n"
                f"    Limitations:      {p.get('limitations', 'not stated')}\n"
                f"    Future work:      {p.get('future_work', 'not stated')}"
            )
        else:
            # paper_summaries format — legacy fallback
            s = p.get("summary", {})
            lines.append(
                f"[{i}] {p['title']} ({p.get('year', '?')}) "
                f"[{p.get('source', '?')}] — {p.get('citation_count', 0)} citations\n"
                f"    Key claims:  {'; '.join(s.get('key_claims', []))}\n"
                f"    Methods:     {s.get('methods', '')}\n"
                f"    Findings:    {s.get('findings', '')}\n"
                f"    Future work: {s.get('future_work', 'not stated')}"
            )
    return "\n\n".join(lines)


class SynthesisPlanningAgent(BaseAgent):
    name          = "synthesis_planning_agent"
    system_prompt = SYSTEM_PROMPT

    tool_name        = "send_to_synthesis_planning_agent"
    tool_description = (
        "Ask SynthesisPlanningAgent to produce (or revise) the literature "
        "review and research plan. The agent automatically reads any pending "
        "feedback from ValidationAgent and acknowledges it before revising."
    )
    tool_schema      = {"type": "object", "properties": {}, "required": []}

    def __init__(self, provider):
        super().__init__(provider)

    def run(self, message: Message, state: dict, bus: MessageBus) -> Message:
        # Prefer extracted_papers (richer 6-field schema with provenance) when
        # ReadingExtractionAgent has run; fall back to paper_summaries otherwise.
        extracted = state.get("extracted_papers", [])
        summaries = extracted if extracted else state.get("paper_summaries", [])

        # ── Check for directed feedback from ValidationAgent ──────────────
        feedback_block = ""
        feedback_msgs  = bus.feedback_for(self.name)
        if feedback_msgs:
            latest = feedback_msgs[-1]
            issues = latest.content.get("issues", [])
            if issues:
                # Acknowledge: dedicated LLM call to reason about each issue
                try:
                    ack_raw  = self._llm(
                        ACK_PROMPT.format(
                            issues="\n".join(f"- {iss}" for iss in issues)
                        ),
                        max_tokens=400,
                    )
                    ack_data = self._parse_json(ack_raw) or {}
                except Exception:
                    ack_data = {}

                # Post ack directly to validation_agent (lateral communication)
                ack_msg = Message(
                    sender=self.name,
                    recipient="validation_agent",
                    msg_type="ack",
                    content={
                        **ack_data,
                        "summary": f"Acknowledged {len(issues)} issue(s); revising",
                    },
                    in_reply_to=latest.id,
                )
                bus.send(ack_msg)

                feedback_block = (
                    "CORRECTION INSTRUCTIONS (from ValidationAgent — must be applied):\n"
                    + "\n".join(f"- {iss}" for iss in issues)
                    + "\n\n"
                )

        # ── Main synthesis + planning LLM call ────────────────────────────
        sub_q_block = "\n".join(
            f"- Sub-question {i}: {q}"
            for i, q in enumerate(state.get("sub_questions", []), 1)
        )

        try:
            raw  = self._llm(
                SYNTHESIS_PLANNING_PROMPT.format(
                    query=state.get("query", ""),
                    sub_questions=sub_q_block,
                    feedback_block=feedback_block,
                    n_papers=len(summaries),
                    paper_block=_paper_block(summaries),
                ),
                max_tokens=3500,
            )
            data = self._parse_json(raw) or {}

            synthesis      = data.get("synthesis", "")
            gaps           = data.get("gaps", "")
            themes         = data.get("themes", [])
            contradictions = data.get("contradictions", [])
            coverage_check = data.get("coverage_check", {})
            uncovered      = coverage_check.get("uncovered", [])
            direction      = data.get("proposed_direction", "")
            steps          = data.get("research_steps", [])
            plan_prose     = data.get("research_plan_prose", "")
            risks          = data.get("risks_and_mitigations", "")

            steps_txt = "\n".join(
                f"{s['step']}. **{s['title']}** — {s['description']}"
                + (f" (grounded in: {', '.join(s.get('grounding_papers', [])[:2])})" if s.get('grounding_papers') else "")
                for s in steps
            )
            research_plan = (
                f"**Proposed Direction:** {direction}\n\n"
                f"{plan_prose}\n\n"
                f"**Research Steps:**\n{steps_txt}\n\n"
                f"**Risks and Mitigations:** {risks}"
            )

        except Exception as exc:
            synthesis = gaps = research_plan = ""
            themes = contradictions = []
            uncovered = []

        state["synthesis"]                = synthesis
        state["gaps"]                     = gaps
        state["research_plan"]            = research_plan
        state["themes"]                   = themes
        state["contradictions"]           = contradictions
        state["uncovered_sub_questions"]  = uncovered
        state["research_steps"]           = steps
        state["risks_and_mitigations"]    = risks

        return self._reply(
            message,
            msg_type="result",
            content={
                "synthesis":      synthesis,
                "gaps":           gaps,
                "research_plan":  research_plan,
                "themes":         [t.get("theme_name", "") for t in themes],
                "contradictions": contradictions,
                "uncovered":      uncovered,
                "summary": (
                    f"Synthesis + plan complete — {len(themes)} theme(s), "
                    f"{len(contradictions)} contradiction(s)"
                    + (f", {len(uncovered)} sub-question(s) uncovered" if uncovered else "")
                ),
            },
            bus=bus,
        )
