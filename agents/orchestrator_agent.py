from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


_EMPTY_PLACEHOLDERS = {
    "",
    "(none)",
    "(not activated)",
    "(no SQL results)",
    "NO_PAPERS_EXTRACTED",
}


def _default_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _has_content(value: str) -> bool:
    text = (value or "").strip()
    if text in _EMPTY_PLACEHOLDERS:
        return False
    return "not activated" not in text.lower()


def _chunk_text(text: str, max_chars: int = 700) -> list[str]:
    """Split long findings into moderate chunks while preserving sentence boundaries."""
    clean = (text or "").strip()
    if not clean:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean) if p.strip()]
    chunks: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= max_chars:
            buffer = candidate
            continue

        if buffer:
            chunks.append(buffer)
            buffer = ""

        if len(paragraph) <= max_chars:
            buffer = paragraph
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        sentence_buffer = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_candidate = f"{sentence_buffer} {sentence}" if sentence_buffer else sentence
            if len(sentence_candidate) <= max_chars:
                sentence_buffer = sentence_candidate
            else:
                if sentence_buffer:
                    chunks.append(sentence_buffer)
                sentence_buffer = sentence
        if sentence_buffer:
            chunks.append(sentence_buffer)

    if buffer:
        chunks.append(buffer)

    return chunks


def _build_tagged_findings(state: dict[str, Any]) -> list[dict[str, str]]:
    tagged: list[dict[str, str]] = []
    source_fields = [
        ("vector_db", "vector_findings"),
        ("sql_db", "sql_findings"),
        ("web", "web_findings"),
        ("merged", "extraction_findings"),
    ]
    for source, key in source_fields:
        text = str(state.get(key, "") or "")
        if not _has_content(text):
            continue
        for chunk in _chunk_text(text):
            tagged.append({"text": chunk, "source": source})
    return tagged


def orchestrator_agent(state: dict[str, Any], model: Any, stamp_fn=None) -> dict[str, Any]:
    stamp_fn = stamp_fn or _default_stamp
    tagged_findings = _build_tagged_findings(state)
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
        "tagged_findings": tagged_findings,
        "messages": [AIMessage(content=f"[Orchestrator] {str(resp.content)[:120]}…")],
        "activity_log": [{
            "agent": "orchestrator",
            "icon": "🤝",
            "title": "Orchestrator merged findings",
            "detail": f"Sources merged: {', '.join(active) or 'none'}; tagged chunks: {len(tagged_findings)}",
            "ts": stamp_fn(),
        }],
        "current_agent": "orchestrator",
    }
