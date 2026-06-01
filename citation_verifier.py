"""
citation_verifier.py — Semantic citation grounding using sentence-transformers.

Replaces the lexical string-matching in validate_citations() with cosine-similarity-based
grounding. Uses all-MiniLM-L6-v2 (already in requirements.txt via sentence-transformers).

Public API:
    extract_citations(summary)         -> list[dict]
    extract_paper_chunks(extraction_findings) -> list[dict]
    semantic_grounded(claim, candidates, threshold) -> (bool, float, str)
    compute_citation_metrics(...)      -> (grounding_map, score)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Lazy model singleton — never imported at module level so import is instant.
# Tests patch _get_model() directly.
# ---------------------------------------------------------------------------

_MODEL: "SentenceTransformer | None" = None


def _get_model() -> "SentenceTransformer":
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


# ---------------------------------------------------------------------------
# Source tag mapping
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"\[(VectorDB|SQL|Web|Extraction)\]", re.IGNORECASE)
_TAG_TO_SOURCE = {
    "vectordb":  "vector_db",
    "sql":       "sql_db",
    "web":       "web",
    "extraction": "web",   # treat Extraction-tagged claims like web (no separate bucket)
}


def _find_source_tag(text: str) -> str | None:
    """Return source key for the first [VectorDB/SQL/Web/Extraction] tag found, or None."""
    m = _TAG_RE.search(text)
    if m:
        return _TAG_TO_SOURCE.get(m.group(1).lower())
    return None


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

def extract_citations(summary: str) -> list[dict]:
    """
    Extract cited claims from a summary string.

    Returns list of dicts: {text, type, source_hint}
      type:        "quoted" | "parenthetical" | "numeric"
      source_hint: "vector_db" | "sql_db" | "web" | None
    """
    seen: set[str] = set()
    results: list[dict] = []

    def _add(text: str, ctype: str, hint: str | None) -> None:
        text = text.strip()[:150]
        norm = text.lower()
        if len(text.split()) < 3 or norm in seen:
            return
        seen.add(norm)
        results.append({"text": text, "type": ctype, "source_hint": hint})

    # 1. Quoted claims: "..." optionally followed by [Tag]
    for m in re.finditer(r'"([^"]{20,150})"', summary):
        claim = m.group(1)
        # scan up to 40 chars after closing quote for a source tag
        tail = summary[m.end(): m.end() + 40]
        hint = _find_source_tag(claim + " " + tail)
        _add(claim, "quoted", hint)

    # 2. Parenthetical claims: (...) with 20-150 chars
    for m in re.finditer(r'\(([^)]{20,150})\)', summary):
        claim = m.group(1)
        tail = summary[m.end(): m.end() + 40]
        hint = _find_source_tag(claim + " " + tail)
        _add(claim, "parenthetical", hint)

    # 3. Numeric [N] citations — extract enclosing sentence
    for m in re.finditer(r'\[(\d+)\]', summary):
        idx = m.start()
        sent_start = summary.rfind('.', 0, idx) + 1
        sent_end = summary.find('.', idx)
        if sent_end == -1:
            sent_end = len(summary)
        sentence = summary[sent_start:sent_end].strip()
        if not sentence:
            continue
        # look backward in the sentence for a source tag
        hint = _find_source_tag(summary[max(0, sent_start - 60): m.end()])
        _add(sentence[:150], "numeric", hint)

    return results[:15]


# ---------------------------------------------------------------------------
# Extraction-findings parser
# ---------------------------------------------------------------------------

_FIELD_RE = re.compile(r'^\*\*([^:*]+?):\*\*\s*(.*)', re.DOTALL)

_PROVENANCE_TO_SOURCE = {
    "full-text":      "vector_db",
    "full_text":      "vector_db",
    "structured-db":  "sql_db",
    "structured_db":  "sql_db",
    "abstract-only":  "web",
    "abstract_only":  "web",
}


def extract_paper_chunks(extraction_findings: str) -> list[dict]:
    """
    Parse the ---delimited block format produced by reading_extraction_agent.

    Returns list of dicts: {title, provenance, source, text}
    """
    if not extraction_findings or extraction_findings.strip() in (
        "", "NO_PAPERS_EXTRACTED", "(none)", "(not activated)"
    ):
        return []

    # Normalize line endings and split on --- boundaries
    raw = extraction_findings.replace("\r\n", "\n").replace("\r", "\n")
    segments = re.split(r'\n---+\n', raw)

    chunks: list[dict] = []
    for seg in segments:
        seg = seg.strip()
        if len(seg) < 20:
            continue

        title = ""
        provenance = "abstract-only"
        field_texts: list[str] = []
        current_field: list[str] = []
        in_named_field = False

        for line in seg.splitlines():
            m = _FIELD_RE.match(line)
            if m:
                if current_field and in_named_field:
                    field_texts.append(" ".join(current_field).strip())
                field_name = m.group(1).strip().lower()
                field_val = m.group(2).strip()
                if "title" in field_name or "topic" in field_name:
                    title = field_val
                    current_field = []
                    in_named_field = False
                elif "provenance" in field_name:
                    provenance = field_val.lower()
                    current_field = []
                    in_named_field = False
                else:
                    current_field = [field_val] if field_val else []
                    in_named_field = True
            else:
                stripped = line.strip().lstrip("- ").strip()
                if stripped and in_named_field:
                    current_field.append(stripped)

        if current_field and in_named_field:
            field_texts.append(" ".join(current_field).strip())

        text = " ".join(field_texts).strip()
        if not text:
            # Fall back to the whole segment if no fields parsed
            text = seg

        source = _PROVENANCE_TO_SOURCE.get(provenance, "web")
        chunks.append({"title": title, "provenance": provenance, "source": source, "text": text})

    return chunks


# ---------------------------------------------------------------------------
# Sliding-window chunker for raw findings strings
# ---------------------------------------------------------------------------

def _sliding_windows(text: str, window: int = 300, step: int = 150) -> list[str]:
    """Split text into overlapping word windows; skip very short fragments."""
    if not text or text.strip() in ("(not activated)", "(none)", ""):
        return []
    # Split at paragraph breaks first, then window
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    windows: list[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) < 10:
            continue
        if len(words) <= window:
            windows.append(para)
        else:
            for start in range(0, len(words) - window + 1, step):
                windows.append(" ".join(words[start: start + window]))
    return windows


# ---------------------------------------------------------------------------
# Source bucket builder
# ---------------------------------------------------------------------------

def _build_source_chunks(
    extraction_findings: str,
    merged_context: str,
    vector_findings: str,
    sql_findings: str,
    web_findings: str,
) -> dict[str, list[str]]:
    """
    Return source-keyed text buckets for candidate matching.

    Keys: "vector_db", "sql_db", "web", "all"
    Falls back to "all" when a source bucket has fewer than 3 candidates.
    """
    buckets: dict[str, list[str]] = {"vector_db": [], "sql_db": [], "web": []}

    # Paper chunks from extraction_findings, grouped by inferred source
    for chunk in extract_paper_chunks(extraction_findings):
        src = chunk["source"]
        if src in buckets:
            buckets[src].append(chunk["text"])

    # Sliding-window chunks from raw per-source findings
    buckets["vector_db"].extend(_sliding_windows(vector_findings))
    buckets["sql_db"].extend(_sliding_windows(sql_findings))
    buckets["web"].extend(_sliding_windows(web_findings))

    # "all" = union of all three + merged_context windows
    all_texts: list[str] = []
    for texts in buckets.values():
        all_texts.extend(texts)
    all_texts.extend(_sliding_windows(merged_context))
    buckets["all"] = all_texts

    return buckets


# ---------------------------------------------------------------------------
# Cosine similarity grounding
# ---------------------------------------------------------------------------

def semantic_grounded(
    claim: str,
    candidate_texts: list[str],
    threshold: float = 0.75,
) -> tuple[bool, float, str]:
    """
    Check whether a claim is semantically grounded in any of the candidate texts.

    Returns (is_grounded, max_cosine_similarity, evidence_snippet).
    """
    if not candidate_texts:
        return False, 0.0, ""

    model = _get_model()
    claim_vec = model.encode(claim, convert_to_numpy=True)
    cand_vecs = model.encode(candidate_texts, convert_to_numpy=True)

    # Cosine similarity: dot / (norm * norm)
    claim_norm = np.linalg.norm(claim_vec)
    if claim_norm == 0:
        return False, 0.0, ""

    claim_unit = claim_vec / claim_norm
    cand_norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
    # Avoid div-by-zero for zero-norm candidates
    safe_norms = np.where(cand_norms == 0, 1.0, cand_norms)
    cand_units = cand_vecs / safe_norms

    scores = cand_units @ claim_unit
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    evidence = candidate_texts[best_idx][:150]
    return best_score >= threshold, best_score, evidence


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_citation_metrics(
    summary: str,
    extraction_findings: str,
    merged_context: str,
    vector_findings: str = "",
    sql_findings: str = "",
    web_findings: str = "",
    threshold: float = 0.75,
) -> tuple[dict, float]:
    """
    Compute semantic citation grounding metrics.

    Returns (grounding_map, citation_accuracy) with the same shape as the
    original validate_citations() for backward compatibility with the Streamlit UI.

    grounding_map: {citation_text[:100]: {grounded, source, evidence, similarity, type}}
    citation_accuracy: grounded_count / total_citations (1.0 when no citations found)
    """
    citations = extract_citations(summary)
    if not citations:
        return {}, 1.0

    buckets = _build_source_chunks(
        extraction_findings, merged_context,
        vector_findings, sql_findings, web_findings,
    )

    grounding_map: dict = {}
    grounded_count = 0

    for cit in citations:
        hint = cit["source_hint"]
        candidates = buckets.get(hint, []) if hint else []
        actual_source = hint or "all"

        # Fall back to "all" if the hinted bucket is too sparse
        if len(candidates) < 3:
            candidates = buckets["all"]
            actual_source = "all"

        is_grounded, score, evidence = semantic_grounded(claim=cit["text"],
                                                         candidate_texts=candidates,
                                                         threshold=threshold)
        if is_grounded:
            grounded_count += 1

        base_key = cit["text"][:100]
        key = base_key
        suffix = 1
        while key in grounding_map:
            key = f"{base_key}#{suffix}"
            suffix += 1
        grounding_map[key] = {
            "grounded":   is_grounded,
            "source":     actual_source,
            "evidence":   evidence,
            "similarity": round(score, 4),
            "type":       cit["type"],
        }

    n = len(citations)
    citation_accuracy = grounded_count / n if n else 1.0
    return grounding_map, citation_accuracy
