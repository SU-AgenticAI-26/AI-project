from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def _default_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def experiment_design_agent(state: dict[str, Any], model: Any, stamp_fn=None) -> dict[str, Any]:
    stamp_fn = stamp_fn or _default_stamp
    system = SystemMessage(content=(
        "You are a Planning Agent in a multi-agent research assistant system. "
        "Your role is to translate a completed literature analysis into an actionable, "
        "citation-grounded structured research plan.\n\n"
        "You have access to: (1) the user's research question, (2) a synthesized literature "
        "summary, (3) per-paper extraction records containing each paper's research problem, "
        "methods, findings, limitations, and stated future work, and (4) a merged evidence "
        "context from all retrieval channels.\n\n"
        "Produce a structured research plan in markdown with EXACTLY these sections, in order:\n\n"
        "## Research Landscape Overview\n"
        "One concise paragraph: what is well-established, which methods dominate, and where "
        "the field currently stands relative to the research question.\n\n"
        "## Identified Research Gaps\n"
        "List 3–5 specific, concrete gaps derived from reported limitations and future-work "
        "statements in the literature. Each gap must cite the paper(s) that reveal it. "
        "Format each as:\n"
        "**Gap N: <short title>** — <description of what is missing or unresolved>  \n"
        "*Grounded in: <paper title / author shorthand>*\n\n"
        "## Proposed Hypotheses\n"
        "One falsifiable, testable hypothesis per gap. Each must be explicitly grounded in the "
        "evidence and directly address its corresponding gap. Format each as:\n"
        "**H-N** *(addresses Gap N)*: <hypothesis statement>\n\n"
        "## Recommended Methodologies\n"
        "For each hypothesis, specify: study design, experimental protocol, key procedures, "
        "and evaluation approach. Where the literature already validates a method, reference it "
        "by name and source. Note which methodologies are novel vs. established.\n\n"
        "## Datasets & Domains\n"
        "For each hypothesis, identify: concrete public datasets or benchmarks (with names), "
        "data collection approaches if no public dataset exists, domain scope and inclusion "
        "criteria, and approximate scale needed. Reference datasets already used in the "
        "literature where applicable.\n\n"
        "## Anticipated Challenges & Risks\n"
        "For each major risk (technical, logistical, or validity-related), briefly describe "
        "the risk and a concrete mitigation strategy. Include risks around reproducibility, "
        "data access, computational cost, and evaluation validity.\n\n"
        "## Short-term Next Steps (0–3 months)\n"
        "A numbered list of immediate, concrete actions a researcher could begin today. "
        "Be specific: name tools, datasets, baselines, or collaborators where relevant.\n\n"
        "## Medium-term Next Steps (3–12 months)\n"
        "Milestones that build on short-term work toward full experimental execution and "
        "publication. Include checkpoints for evaluating progress.\n\n"
        "IMPORTANT: Every claim must be grounded in the provided evidence. "
        "Do not fabricate paper titles, authors, or dataset names. "
        "If evidence is sparse for a section, say so explicitly rather than hallucinating."
    ))

    extraction = state.get("extraction_findings", "") or ""
    context = state.get("merged_context", "") or ""
    km_nodes = [n.get("label", "") for n in state.get("knowledge_map", {}).get("nodes", []) if isinstance(n, dict)]

    resp = model.invoke([system, HumanMessage(content=(
        f"Research question: {state['query']}\n\n"
        f"Synthesized literature summary:\n{state['summary']}\n\n"
        "Per-paper extraction records (problems · methods · findings · limitations · future work):\n"
        f"{extraction[:2500]}\n\n"
        f"Merged evidence context:\n{context[:2000]}\n\n"
        f"Key concepts from knowledge graph: {', '.join(km_nodes[:30])}"
    ))])

    plan = str(resp.content)
    n_gaps = plan.count("**Gap ")
    n_hyp = plan.count("**H-")
    n_steps = plan.count("## Short-term") + plan.count("## Medium-term")

    return {
        "experiment_plan": plan,
        "messages": [AIMessage(content=f"[ExperimentDesign] {plan[:120]}…")],
        "activity_log": [{
            "agent": "experiment_design",
            "icon": "🧪",
            "title": "Planning Agent — structured research plan generated",
            "detail": f"{n_gaps} gaps identified · {n_hyp} hypotheses · {n_steps} step horizons · {len(plan)} characters",
            "ts": stamp_fn(),
        }],
        "current_agent": "experiment_design",
    }
