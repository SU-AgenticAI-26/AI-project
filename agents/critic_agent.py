from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def _default_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _graph_metrics(nodes: list[Any], edges: list[Any]) -> dict[str, int]:
    node_types = {
        str(n.get("type", "")).strip().lower()
        for n in nodes
        if isinstance(n, dict) and n.get("type")
    }
    sources = {
        str(n.get("source", "")).strip().lower()
        for n in nodes
        if isinstance(n, dict) and n.get("source")
    }
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "type_count": len(node_types),
        "source_count": len(sources),
    }


def critic_agent(state: dict[str, Any], model: Any, stamp_fn=None) -> dict[str, Any]:
    stamp_fn = stamp_fn or _default_stamp
    nodes = state.get("knowledge_map", {}).get("nodes", [])
    edges = state.get("knowledge_map", {}).get("edges", [])
    metrics = _graph_metrics(nodes, edges)
    query = str(state.get("query", "") or "")
    loop_count = int(state.get("loop_count", 0) or 0)

    prompt = (
        "You are a Critic Agent deciding whether retrieval/graph coverage is sufficient for final synthesis.\n"
        "Assess BOTH structure and coverage quality.\n"
        "Guidelines:\n"
        "1) Structural minimums: >=8 nodes, >=4 edges, >=2 distinct node types.\n"
        "2) Source breadth: prefer >=2 distinct node sources when the question is broad/comparative.\n"
        "3) If key mechanisms/challenges/approaches likely missing, request another retrieval pass.\n"
        "4) Keep feedback concrete and actionable for router/retrievers.\n\n"
        "Return ONLY JSON: {\"needs_more\": true/false, \"feedback\": \"short reason\"}"
    )
    payload = {
        "query": query,
        "loop_count": loop_count,
        "graph_metrics": metrics,
        "nodes": nodes,
        "edges": edges,
    }
    resp = model.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=json.dumps(payload)),
    ])
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

    if not feedback:
        feedback = (
            f"nodes={metrics['node_count']}, edges={metrics['edge_count']}, "
            f"types={metrics['type_count']}, sources={metrics['source_count']}"
        )

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
