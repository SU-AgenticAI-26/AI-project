from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# Thresholds — tune these to match your pass/fail targets
_STRUCTURAL_MIN = {"node_count": 8, "edge_count": 4, "type_count": 2}
_MAX_AUTO_APPROVE_LOOP = 1   # never loop more than once without explicit quality failure
_NEEDS_MORE_MIN_CONFIDENCE = 0.75


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


def _structural_gaps(metrics: dict[str, int]) -> list[str]:
    """Return a list of concrete structural failures, empty if none."""
    gaps = []
    for key, minimum in _STRUCTURAL_MIN.items():
        if metrics[key] < minimum:
            gaps.append(f"{key}={metrics[key]} (min {minimum})")
    return gaps


def _has_specific_missing_concept(missing_concept: str, feedback: str) -> bool:
    concept = (missing_concept or "").strip()
    if len(concept) >= 3:
        return True
    # Backward-compatible fallback for older LLM outputs that don't include missing_concept.
    fb = (feedback or "").strip()
    return len(fb.split()) >= 6


def critic_agent(state: dict[str, Any], model: Any, stamp_fn=None) -> dict[str, Any]:
    stamp_fn = stamp_fn or _default_stamp
    nodes = state.get("knowledge_map", {}).get("nodes", [])
    edges = state.get("knowledge_map", {}).get("edges", [])
    metrics = _graph_metrics(nodes, edges)
    query = str(state.get("query", "") or "")
    loop_count = int(state.get("loop_count", 0) or 0)

    # --- Fast path: auto-approve if already looped enough and structure is fine ---
    structural_gaps = _structural_gaps(metrics)
    if loop_count >= _MAX_AUTO_APPROVE_LOOP and not structural_gaps:
        feedback = (
            f"Auto-approved after {loop_count} loop(s): "
            f"nodes={metrics['node_count']}, edges={metrics['edge_count']}, "
            f"types={metrics['type_count']}, sources={metrics['source_count']}"
        )
        return _build_result(
            needs=False,
            feedback=feedback,
            state=state,
            loop_count=loop_count,
            stamp_fn=stamp_fn,
        )

    # --- If structural minimums aren't met, skip the LLM and just say so ---
    if structural_gaps:
        feedback = f"Structural gaps: {'; '.join(structural_gaps)}. Retrieve more."
        return _build_result(
            needs=True,
            feedback=feedback,
            state=state,
            loop_count=loop_count,
            stamp_fn=stamp_fn,
        )

    # --- LLM quality check: only reaches here if structure passes and loop budget remains ---
    prompt = (
        "You are a Critic Agent. Structure checks have already passed.\n"
        "Your only job: decide if COVERAGE is sufficient to answer the query well.\n\n"
        "Bias strongly toward stopping.\n"
        "Return needs_more=true ONLY if you can name a specific concept or mechanism "
        "the query requires that is clearly absent from the graph nodes below.\n"
        "Do NOT return needs_more=true based on vague concerns like 'could be more comprehensive'.\n"
        "If in doubt, return needs_more=false.\n\n"
        "Return ONLY JSON with this exact schema: "
        "{\"needs_more\": true/false, \"missing_concept\": \"string\", "
        "\"confidence\": 0.0-1.0, \"feedback\": \"one concrete sentence\"}.\n"
        "If needs_more=true, missing_concept must be specific (not empty) and confidence must be >= 0.75."
    )
    payload = {
        "query": query,
        "loop_count": loop_count,
        "graph_metrics": metrics,
        "node_summaries": [
            {"type": n.get("type"), "label": n.get("label"), "source": n.get("source")}
            for n in nodes
            if isinstance(n, dict)
        ],
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
        missing_concept = str(parsed.get("missing_concept", "")).strip()
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        feedback = str(parsed.get("feedback", "")).strip()
    except Exception:
        needs = False
        missing_concept = ""
        confidence = 0.0
        feedback = "Parse error — defaulting to approved."

    # Conservative gate: only allow enrichment when the model gives a concrete,
    # specific gap with sufficient confidence.
    if needs:
        if not _has_specific_missing_concept(missing_concept, feedback):
            needs = False
            feedback = "Approval: needs_more rejected because no specific missing concept was identified."
        elif confidence < _NEEDS_MORE_MIN_CONFIDENCE:
            needs = False
            feedback = (
                f"Approval: needs_more confidence {confidence:.2f} below "
                f"threshold {_NEEDS_MORE_MIN_CONFIDENCE:.2f}."
            )

    if not feedback:
        feedback = f"Coverage approved: {metrics}"

    return _build_result(
        needs=needs,
        feedback=feedback,
        state=state,
        loop_count=loop_count,
        stamp_fn=stamp_fn,
    )


def _build_result(
    needs: bool,
    feedback: str,
    state: dict[str, Any],
    loop_count: int,
    stamp_fn,
) -> dict[str, Any]:
    next_loop_count = state.get("loop_count", 0)
    if needs:
        next_loop_count += 1

    return {
        "critique": feedback,
        "_needs_more": needs,
        # Approvals must not consume loop budget.
        "loop_count": next_loop_count,
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