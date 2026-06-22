from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def _default_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def scoping_agent(state: dict[str, Any], model: Any, stamp_fn=None) -> dict[str, Any]:
    stamp_fn = stamp_fn or _default_stamp
    system = SystemMessage(content=(
        "You are a Query Scoping Agent. Decompose the user's query into:\n"
        "1. 3-5 focused sub-questions that break down the research\n"
        "2. 4-8 key terms/themes for retrieval\n"
        "Return ONLY JSON:\n"
        '{"sub_questions": ["q1", "q2", ...], "keywords": ["k1", "k2", ...], "reasoning": "brief explain"}\n'
        "No other text or markdown."
    ))
    resp = model.invoke([system, HumanMessage(content=f"Query: {state['query']}")])
    raw = str(resp.content).strip().lstrip("```json").rstrip("```").strip()
    try:
        parsed = json.loads(raw)
        subs = parsed.get("sub_questions", [])[:5]
        keys = parsed.get("keywords", [])[:8]
        reason = parsed.get("reasoning", "")
    except Exception:
        subs = [state.get("query", "")]
        keys = str(state.get("query", "")).split()[:5]
        reason = "scoping_failed"

    return {
        "sub_questions": subs,
        "keywords": keys,
        "scoping_reasoning": reason,
        "messages": [AIMessage(content=f"[Scoping] {len(subs)} sub-questions, {len(keys)} keywords")],
        "activity_log": [{
            "agent": "scoping",
            "icon": "🔍",
            "title": "Query Scoped",
            "detail": f"→ {len(subs)} sub-questions · {len(keys)} keywords",
            "ts": stamp_fn(),
        }],
        "current_agent": "scoping",
    }
