from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


_EMPTY_PLACEHOLDER = {"", "(none)", "(not activated)", "(no SQL results)"}


def _default_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _has_content(s: str) -> bool:
    return s.strip() not in _EMPTY_PLACEHOLDER


def reading_extraction_agent(state: dict[str, Any], model: Any, stamp_fn=None) -> dict[str, Any]:
    stamp_fn = stamp_fn or _default_stamp
    vf = state.get("vector_findings", "")
    sf = state.get("sql_findings", "")
    wf = state.get("web_findings", "")

    if not any(_has_content(f) for f in (vf, sf, wf)):
        return {
            "extraction_findings": "(none)",
            "messages": [AIMessage(content="[ReadingExtraction] skipped — no retrieval content")],
            "activity_log": [{
                "agent": "reading_extraction",
                "icon": "📖",
                "title": "Reading / Extraction — skipped",
                "detail": "No content from any retrieval channel.",
                "ts": stamp_fn(),
            }],
            "current_agent": "reading_extraction",
        }

    sections = []
    if _has_content(vf):
        sections.append(f"=== Vector DB findings ===\n{vf}")
    if _has_content(sf):
        sections.append(f"=== SQL / DB findings ===\n{sf}")
    if _has_content(wf):
        sections.append(f"=== Web / API findings ===\n{wf}")
    combined = "\n\n".join(sections)

    system = SystemMessage(content=(
        "You are a Reading and Extraction Agent for academic research.\n"
        "Given synthesised retrieval findings from multiple sources, extract a structured record "
        "for every distinct paper, study, or source you can identify.\n\n"
        "For each paper output a block in this exact format:\n"
        "---\n"
        "**Title / Topic:** <title or identifier>\n"
        "**Provenance:** <abstract-only | full-text | structured-db>\n"
        "**Research Problem:** <one sentence>\n"
        "**Methodology:** <one sentence>\n"
        "**Key Findings:**\n- <bullet points>\n"
        "**Limitations:** <one sentence>\n"
        "**Future Work:** <one sentence>\n"
        "---\n\n"
        "Provenance rules:\n"
        "- 'structured-db' if the source is from the SQL/DB findings section\n"
        "- 'full-text' if large paragraphs of text were available (VectorDB)\n"
        "- 'abstract-only' if only title/abstract/year was available (Web API results)\n\n"
        "If no distinct papers can be identified, output exactly: NO_PAPERS_EXTRACTED\n\n"
        "Be precise. Do not hallucinate citations."
    ))
    resp = model.invoke([system, HumanMessage(
        content=f"Query: {state['query']}\n\nRetrieval findings:\n{combined}"
    )])

    extraction = str(resp.content).strip()
    title_marker = "**Title / Topic:**"
    paper_count = extraction.count(title_marker)
    if paper_count == 0:
        delimiter_lines = sum(1 for line in extraction.splitlines() if line.strip() == "---")
        paper_count = max(1, delimiter_lines // 2) if delimiter_lines > 0 else 0

    return {
        "extraction_findings": extraction,
        "messages": [AIMessage(content=f"[ReadingExtraction] {paper_count} paper(s) structured")],
        "activity_log": [{
            "agent": "reading_extraction",
            "icon": "📖",
            "title": "Reading / Extraction agent",
            "detail": (
                f"{paper_count} paper(s) extracted — fields: research problem, methodology, "
                f"findings, limitations, future work. Provenance tagged per source."
            ),
            "ts": stamp_fn(),
        }],
        "current_agent": "reading_extraction",
    }
