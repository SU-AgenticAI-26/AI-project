from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def _default_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_citations(summary: str, merged_context: str, extraction_findings: str) -> tuple[dict[str, Any], float]:
    grounding_map: dict[str, Any] = {}
    citation_patterns = [
        r'"([^"]{20,150})"',
        r"\(([^)]{20,150})\)",
    ]
    citations: list[str] = []
    for pattern in citation_patterns:
        citations.extend(re.findall(pattern, summary))

    for match in re.finditer(r"\[(\d+)\]", summary):
        idx = match.start()
        sent_start = summary.rfind(".", 0, idx) + 1
        sent_end = summary.find(".", idx)
        if sent_end == -1:
            sent_end = len(summary)
        sentence = summary[sent_start:sent_end].strip()[:150]
        if len(sentence.split()) >= 3:
            citations.append(sentence)

    citations = list(set(citations))[:15]
    context_text = merged_context + "\n" + extraction_findings
    context_lower = context_text.lower()
    grounded_count = 0

    for cit in citations:
        cit_lower = cit.lower()
        words = cit_lower.split()
        if len(words) < 3:
            continue

        found_in_merged = cit_lower in merged_context.lower()
        found_in_extraction = cit_lower in extraction_findings.lower()

        if not (found_in_merged or found_in_extraction):
            content_words = [w for w in words if len(w) > 3]
            if content_words:
                matching = sum(1 for w in content_words if w in context_lower)
                if matching >= len(content_words) * 0.6:
                    found_in_merged = True

        is_grounded = found_in_merged or found_in_extraction
        if is_grounded:
            grounded_count += 1
            evidence = next(
                (
                    line.strip()[:150]
                    for line in context_text.split("\n")
                    if any(w in line.lower() for w in words[:3])
                ),
                "",
            )
        else:
            evidence = ""

        grounding_map[cit[:100]] = {
            "grounded": is_grounded,
            "source": "merged" if found_in_merged else ("extraction" if found_in_extraction else "none"),
            "evidence": evidence,
        }

    score = grounded_count / len(citations) if citations else 1.0
    return grounding_map, score


def summarizer_agent(state: dict[str, Any], model: Any, stamp_fn=None) -> dict[str, Any]:
    stamp_fn = stamp_fn or _default_stamp
    system = SystemMessage(content=(
        "You are given a research query and multi-source findings.\n"
        "Write a response that:\n"
        "- Answers ONLY what the query asks -- do not add background the user didn't request\n"
        "- Is <=400 words unless the query explicitly requires exhaustive coverage\n"
        "- Uses prose paragraphs, not bullet lists\n"
        "- Identifies at least one tension or disagreement across sources\n"
        "- Cites which source supports each distinct claim\n\n"
        "Accepted source labels: Vector DB, SQL DB, Web, Extraction."
    ))
    synthesis = state.get("synthesis_report", "")
    context_src = synthesis if synthesis and synthesis != "(no content to synthesise)" else state.get("merged_context", "")
    key_concepts = [n.get("label", "") for n in state.get("knowledge_map", {}).get("nodes", []) if isinstance(n, dict)]
    resp = model.invoke([system, HumanMessage(content=(
        f"Query: {state['query']}\n\n"
        f"Findings: {context_src}\n\n"
        f"Key concepts: {key_concepts}"
    ))])

    citation_grounding, grounding_score = validate_citations(
        str(resp.content),
        state.get("merged_context", ""),
        state.get("extraction_findings", ""),
    )

    grounded_count = sum(1 for v in citation_grounding.values() if v.get("grounded"))
    total_citations = len(citation_grounding)

    return {
        "summary": str(resp.content),
        "citation_grounding": citation_grounding,
        "grounding_score": grounding_score,
        "messages": [AIMessage(content=f"[Summarizer] {str(resp.content)[:120]}…")],
        "activity_log": [{
            "agent": "summarizer",
            "icon": "✍️",
            "title": "Summarizer — final answer",
            "detail": f"{len(str(resp.content))} chars · {grounded_count}/{total_citations} citations grounded ({int(grounding_score * 100)}%)",
            "ts": stamp_fn(),
        }],
        "current_agent": "summarizer",
    }
