from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def _default_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def orchestrator_agent(state: dict[str, Any], model: Any, stamp_fn=None) -> dict[str, Any]:
    stamp_fn = stamp_fn or _default_stamp
    block = "\n\n".join([
        f"=== Vector DB ===\n{state.get('vector_findings', '')}",
        f"=== SQL / DB ===\n{state.get('sql_findings', '')}",
        f"=== Web / APIs ===\n{state.get('web_findings', '')}",
        f"=== Structured Extraction ===\n{state.get('extraction_findings', '')}",
    ])
    system = SystemMessage(content=(
        "You are an Orchestrator Agent. Merge findings from four specialised agents "
        "(VectorDB, SQL/DB, Web, and Extraction):\n"
        "1. Deduplicate overlapping information\n"
        "2. Resolve contradictions, preferring higher-confidence structured sources\n"
        "3. Label each claim: [VectorDB] / [SQL] / [Web] / [Extraction]\n"
        "4. Produce one coherent merged context for downstream agents."
    ))
    resp = model.invoke([system, HumanMessage(content=f"Query: {state['query']}\n\n{block}")])
    active = [
        src for src, key in [
            ("Vector DB", "vector_findings"),
            ("SQL DB", "sql_findings"),
            ("Web", "web_findings"),
            ("Extraction", "extraction_findings"),
        ]
        if state.get(key, "") not in ("", "(none)", "NO_PAPERS_EXTRACTED", "(not activated)")
        and "not activated" not in state.get(key, "")
    ]
    return {
        "merged_context": str(resp.content),
        "messages": [AIMessage(content=f"[Orchestrator] {str(resp.content)[:120]}…")],
        "activity_log": [{
            "agent": "orchestrator",
            "icon": "🤝",
            "title": "Orchestrator merged findings",
            "detail": f"Sources merged: {', '.join(active) or 'none'}",
            "ts": stamp_fn(),
        }],
        "current_agent": "orchestrator",
    }
