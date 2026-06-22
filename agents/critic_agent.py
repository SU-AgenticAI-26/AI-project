from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage


def _default_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def critic_agent(state: dict[str, Any], model: Any, stamp_fn=None) -> dict[str, Any]:
    stamp_fn = stamp_fn or _default_stamp
    nodes = state.get("knowledge_map", {}).get("nodes", [])
    edges = state.get("knowledge_map", {}).get("edges", [])

    prompt = (
        "You are a Critic Agent evaluating whether the current knowledge graph is sufficient "
        "to generate a reliable final answer.\n"
        "Criteria:\n"
        "1) At least 8 nodes\n"
        "2) At least 4 edges\n"
        "3) At least 2 distinct node types\n\n"
        "Return ONLY JSON: {\"needs_more\": true/false, \"feedback\": \"short reason\"}"
    )
    resp = model.invoke([AIMessage(content=prompt), AIMessage(content=json.dumps({"nodes": nodes, "edges": edges}))])
    raw = str(resp.content).strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("```").strip()

    try:
        parsed = json.loads(raw)
        needs = bool(parsed.get("needs_more", False))
        feedback = str(parsed.get("feedback", ""))
    except Exception:
        needs = False
        feedback = ""

    return {
        "critique": feedback,
        "_needs_more": needs,
        "loop_count": state.get("loop_count", 0) + 1,
        "messages": [AIMessage(content=f"[Critic] needs_more={needs}")],
        "activity_log": [{
            "agent": "critic",
            "icon": "🧐",
            "title": f"Critic — {'needs enrichment' if needs else 'approved'}",
            "detail": feedback,
            "ts": stamp_fn(),
        }],
        "current_agent": "critic",
    }
