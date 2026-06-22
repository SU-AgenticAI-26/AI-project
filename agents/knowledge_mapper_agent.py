from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def _default_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def knowledge_mapper_agent(state: dict[str, Any], model: Any, stamp_fn=None) -> dict[str, Any]:
    stamp_fn = stamp_fn or _default_stamp
    system = SystemMessage(content=(
        "You are a Knowledge Mapping Agent. Extract a knowledge graph from the merged context.\n"
        "The context labels each claim with its origin: [VectorDB], [SQL], [Web], or [Extraction]. "
        "Use those labels to set each node's \"source\" field:\n"
        "  [VectorDB]   → \"vector_db\"\n"
        "  [SQL]        → \"sql_db\"\n"
        "  [Web]        → \"web\"\n"
        "  [Extraction] → \"merged\"\n"
        "  no label     → \"merged\"\n\n"
        "Return ONLY valid JSON (no markdown, no extra text):\n"
        '{"nodes": [{"id":"str","label":"str","type":"concept","source":"vector_db"}],'
        '"edges": [{"source":"str","target":"str","relation":"str","weight":0.5}]}\n'
        '"type" must be exactly one of: concept, entity, fact, process.\n'
        '"source" must be exactly one of: vector_db, sql_db, web, merged.\n'
        "Include 12–20 nodes."
    ))
    resp = model.invoke([system, HumanMessage(
        content=f"Query: {state['query']}\n\nMerged context:\n{state['merged_context']}"
    )])
    raw = str(resp.content).strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("```").strip()
    try:
        km = json.loads(raw)
    except Exception:
        km = {"nodes": [], "edges": [], "error": "parse_failed"}

    return {
        "knowledge_map": km,
        "messages": [AIMessage(content=f"[KnowledgeMapper] {len(km.get('nodes', []))} nodes")],
        "activity_log": [{
            "agent": "knowledge_mapper",
            "icon": "🗺️",
            "title": "Knowledge Mapper",
            "detail": f"{len(km.get('nodes', []))} nodes, {len(km.get('edges', []))} edges extracted",
            "ts": stamp_fn(),
        }],
        "current_agent": "knowledge_mapper",
    }
