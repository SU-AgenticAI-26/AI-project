from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


CREDIBILITY_TIERS = {
    "sql_db": {"label": "peer-reviewed corpus", "score": 0.9},
    "vector_db": {"label": "indexed papers", "score": 0.8},
    "web": {"label": "web / preprints", "score": 0.6},
    "merged": {"label": "consensus", "score": 0.7},
}


def _default_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def conflict_agent(state: dict[str, Any], model: Any, stamp_fn=None) -> dict[str, Any]:
    stamp_fn = stamp_fn or _default_stamp
    if not state.get("merged_context") or len(state.get("merged_context", "")) < 100:
        return {
            "conflicts": [],
            "credibility_map": CREDIBILITY_TIERS,
            "messages": [AIMessage(content="[Conflict] No conflicts (insufficient context)")],
            "activity_log": [{
                "agent": "conflict_detector",
                "icon": "⚡",
                "title": "Conflict Detection",
                "detail": "Insufficient context to identify conflicts",
                "ts": stamp_fn(),
            }],
            "current_agent": "conflict_detector",
        }

    system = SystemMessage(content=(
        "You are a Conflict Detection Agent. Read the research findings and identify any direct "
        "contradictions or disagreements between sources.\n\n"
        "Examples of conflicts:\n"
        "- Source A claims 'X improves performance', Source B claims 'X has no effect'\n"
        "- Source A says 'method requires Y parameter', Source B says 'Y parameter is optional'\n"
        "- Source A: 'Dataset size is critical', Source B: 'Architecture matters more'\n\n"
        "Return ONLY valid JSON (no markdown, no code blocks):\n"
        '{"conflicts": [{"topic": "short phrase", "claim_a": "what source A says", '
        '"source_a": "vector_db|sql_db|web", "claim_b": "what source B says", '
        '"source_b": "vector_db|sql_db|web", "resolution": "which is more likely correct (one sentence)"}]}\n\n'
        "If no real contradictions exist, return {\"conflicts\": []}. "
        "Do NOT invent conflicts or over-interpret differences in terminology."
    ))

    resp = model.invoke([system, HumanMessage(content=(
        f"Research findings (first 4000 chars):\n\n{state['merged_context'][:4000]}"
    ))])

    raw = str(resp.content).strip().lstrip("```json").rstrip("```").strip()
    try:
        result = json.loads(raw)
        conflicts = result.get("conflicts", [])[:10]
    except Exception:
        conflicts = []

    conflict_count = len(conflicts)
    return {
        "conflicts": conflicts,
        "credibility_map": CREDIBILITY_TIERS,
        "messages": [AIMessage(content=f"[Conflict] {conflict_count} conflicts identified")],
        "activity_log": [{
            "agent": "conflict_detector",
            "icon": "⚡",
            "title": f"Conflict Detection — {conflict_count} issues",
            "detail": "Identified potential inconsistencies between sources",
            "ts": stamp_fn(),
        }],
        "current_agent": "conflict_detector",
    }
