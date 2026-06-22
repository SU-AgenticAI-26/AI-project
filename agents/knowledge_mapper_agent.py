from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


_ALLOWED_SOURCES = {"vector_db", "sql_db", "web", "merged"}


def _default_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_tagged_findings(raw_items: Any) -> list[dict[str, str]]:
    if not isinstance(raw_items, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        source = str(item.get("source", "merged") or "merged").strip().lower()
        if not text:
            continue
        if source not in _ALLOWED_SOURCES:
            source = "merged"
        out.append({"text": text, "source": source})
    return out


def _compact_tagged_findings(tagged: list[dict[str, str]], max_items: int = 40, max_text: int = 900) -> list[dict[str, str]]:
    compact: list[dict[str, str]] = []
    for item in tagged[:max_items]:
        text = item["text"]
        compact.append({
            "source": item["source"],
            "text": text if len(text) <= max_text else f"{text[:max_text]}...",
        })
    return compact


def _repair_node_sources(km: dict[str, Any], tagged: list[dict[str, str]]) -> dict[str, Any]:
    nodes = km.get("nodes", [])
    if not isinstance(nodes, list):
        return km
    if not tagged:
        for node in nodes:
            if isinstance(node, dict) and node.get("source") not in _ALLOWED_SOURCES:
                node["source"] = "merged"
        return km

    lower_tagged = [
        {"source": item["source"], "text": item["text"].lower()}
        for item in tagged
    ]

    for node in nodes:
        if not isinstance(node, dict):
            continue
        label = str(node.get("label", "") or "").strip().lower()
        current_source = str(node.get("source", "") or "").strip().lower()
        if current_source not in _ALLOWED_SOURCES:
            current_source = "merged"

        if label:
            matched_sources = {
                item["source"] for item in lower_tagged if label in item["text"]
            }
            if len(matched_sources) == 1:
                node["source"] = next(iter(matched_sources))
                continue

        node["source"] = current_source or "merged"

    return km


def knowledge_mapper_agent(state: dict[str, Any], model: Any, stamp_fn=None) -> dict[str, Any]:
    stamp_fn = stamp_fn or _default_stamp
    tagged_findings = _sanitize_tagged_findings(state.get("tagged_findings", []))
    compact_tagged = _compact_tagged_findings(tagged_findings)

    system = SystemMessage(content=(
        "You are a Knowledge Mapping Agent. Extract a knowledge graph from the merged context.\n"
        "Use source provenance carefully. You will receive tagged_findings JSON where each item has {text, source}. "
        "Treat tagged_findings as authoritative provenance and preserve source labels in nodes whenever possible.\n"
        "Valid node source labels: vector_db, sql_db, web, merged.\n"
        "Return ONLY valid JSON (no markdown, no extra text):\n"
        '{"nodes": [{"id":"str","label":"str","type":"concept","source":"vector_db"}],'
        '"edges": [{"source":"str","target":"str","relation":"str","weight":0.5}]}\n'
        '"type" must be exactly one of: concept, entity, fact, process.\n'
        '"source" must be exactly one of: vector_db, sql_db, web, merged.\n'
        "Include 12–20 nodes."
    ))
    resp = model.invoke([system, HumanMessage(
        content=(
            f"Query: {state['query']}\n\n"
            f"Merged context:\n{state['merged_context']}\n\n"
            f"tagged_findings JSON (authoritative provenance):\n"
            f"{json.dumps(compact_tagged, ensure_ascii=True)}"
        )
    )])
    raw = str(resp.content).strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("```").strip()
    try:
        km = json.loads(raw)
    except Exception:
        km = {"nodes": [], "edges": [], "error": "parse_failed"}

    km = _repair_node_sources(km, tagged_findings)

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
