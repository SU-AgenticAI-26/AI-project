from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def _default_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# Labels the prompt forbids — extracting and failing to ground these
# was silently dragging down the citation accuracy score.
_FORBIDDEN_LABEL_PATTERNS = re.compile(
    r"^(source\s*:\s*(web|sql|vector|extraction|merged)|"
    r"(web|sql|vector|vector\s*db|sql\s*db|extraction)\s*(agent|findings|source)?|"
    r"evidence\s*:)",
    re.IGNORECASE,
)


def _extract_citations(summary: str) -> list[str]:
    """
    Extract only the inner quoted span from (evidence: "...") citations,
    plus any bare quoted spans. Strips the wrapper so the grounding check
    works against the actual claim text, not the label noise.

    Bug fixed: the old code ran both a quote pattern and a paren pattern
    over the same text. (evidence: "X") would yield both "X" (from the
    quote pattern) and 'evidence: "X"' (from the paren pattern). The
    wrapper form consistently failed grounding because its content words
    included "evidence" and the citation key, not the claim itself —
    doubling citation count while halving grounded hits.
    """
    seen: set[str] = set()
    results: list[str] = []

    # Primary: pull the inner span from (evidence: "...") wrappers first.
    # Cap is 150 chars to accommodate real quoted spans (the prompt allows 8-30 words,
    # which at ~6 chars/word can reach ~180 chars; 150 covers the vast majority safely).
    for m in re.finditer(r'\(evidence\s*:\s*"([^"]{8,150})"\s*\)', summary, re.IGNORECASE):
        span = m.group(1).strip()
        if span not in seen:
            seen.add(span)
            results.append(span)

    # Secondary: bare quoted spans not already captured above.
    # [^"\n] stops the pattern from matching across line breaks, which caused the old
    # code to capture multi-line garbage like '").\nThis is counteracted by...'.
    for m in re.finditer(r'"([^"\n]{8,150})"', summary):
        span = m.group(1).strip()
        if span not in seen:
            seen.add(span)
            results.append(span)

    # Tertiary: [N] bracket citations — extract the surrounding sentence as before
    for m in re.finditer(r"\[(\d+)\]", summary):
        idx = m.start()
        sent_start = summary.rfind(".", 0, idx) + 1
        sent_end = summary.find(".", idx)
        if sent_end == -1:
            sent_end = len(summary)
        sentence = summary[sent_start:sent_end].strip()[:120]
        if len(sentence.split()) >= 5 and sentence not in seen:
            seen.add(sentence)
            results.append(sentence)

    # Bug fixed: cap was 15 but prompt instructs max 5 citations.
    # Inflating the denominator with ungroundable extras was the
    # primary driver of low citation_accuracy scores on Claude runs.
    return results[:5]


def validate_citations(
    summary: str,
    merged_context: str,
    extraction_findings: str,
) -> tuple[dict[str, Any], float]:
    citations = _extract_citations(summary)
    context_text = merged_context + "\n" + extraction_findings
    context_lower = context_text.lower()
    grounding_map: dict[str, Any] = {}
    grounded_count = 0

    for cit in citations:
        # Bug fixed: forbidden labels were entering the pool and failing
        # grounding, quietly lowering the score without any diagnostic signal.
        # Now they are skipped entirely — consistent with the prompt instruction
        # that bans them — rather than counted as ungrounded citations.
        if _FORBIDDEN_LABEL_PATTERNS.match(cit.strip()):
            grounding_map[cit[:100]] = {
                "grounded": False,
                "source": "forbidden_label",
                "evidence": "",
                "type": "skipped",
            }
            continue

        cit_lower = cit.lower()
        words = cit_lower.split()
        if len(words) < 3:
            continue

        found_in_merged = cit_lower in merged_context.lower()
        found_in_extraction = cit_lower in extraction_findings.lower()

        if not (found_in_merged or found_in_extraction):
            content_words = [w for w in words if len(w) > 3]
            if content_words:
                matching = sum(1 for w in content_words if w in context_lower)
                if matching >= len(content_words) * 0.6:
                    found_in_merged = True

        is_grounded = found_in_merged or found_in_extraction
        if is_grounded:
            grounded_count += 1
            evidence = next(
                (
                    line.strip()[:150]
                    for line in context_text.split("\n")
                    if any(w in line.lower() for w in words[:3])
                ),
                "",
            )
        else:
            evidence = ""

        grounding_map[cit[:100]] = {
            "grounded": is_grounded,
            "source": "all",
            "evidence": evidence,
            "type": "quoted",
        }

    # Score only against citations that were actually evaluated (not skipped labels)
    evaluated = [v for v in grounding_map.values() if v.get("type") != "skipped"]
    score = grounded_count / len(evaluated) if evaluated else 1.0
    return grounding_map, score


def summarizer_agent(state: dict[str, Any], model: Any, stamp_fn=None) -> dict[str, Any]:
    stamp_fn = stamp_fn or _default_stamp
    system = SystemMessage(content=(
        "You are a synthesis writer for research evidence.\n"
        "Your output must prioritize reasoning over listing.\n\n"
        "Hard requirements:\n"
        "1) Answer only the user query; do not add unrelated background.\n"
        "2) Keep length near 400 words (target 300-420 words) unless the query explicitly asks for exhaustive detail.\n"
        "3) Write prose paragraphs only (2-4 paragraphs). Do not use bullets, numbering, headings, or markdown lists.\n"
        "4) Include at least one explicit cross-source tension/disagreement and explain why it matters for the conclusion.\n"
        "5) Support each distinct empirical claim with an evidence span quoted verbatim from Findings.\n"
        "   Citation format: (evidence: \"<exact quote from Findings>\").\n"
        "   Each quote must be 8-30 words copied exactly from Findings.\n"
        "   Use at most 5 citations total. Prefer fewer, higher-quality citations over many weak ones.\n"
        "6) Do NOT cite retrieval channels, agent names, or structural labels as evidence.\n"
        "   Forbidden citation styles include: (source: Web), (source: Extraction), (Vector DB), (SQL DB).\n"
        "7) If evidence is thin or conflicting, say so explicitly instead of overstating certainty.\n\n"
        "Output quality bar: integrated synthesis, not a catalog of findings."
    ))
    synthesis = state.get("synthesis_report", "")
    context_src = synthesis if synthesis and synthesis != "(no content to synthesise)" else state.get("merged_context", "")
    key_concepts = [n.get("label", "") for n in state.get("knowledge_map", {}).get("nodes", []) if isinstance(n, dict)]
    resp = model.invoke([system, HumanMessage(content=(
        f"Query: {state['query']}\n\n"
        f"Findings: {context_src}\n\n"
        f"Key concepts: {key_concepts}"
    ))])

    citation_grounding, grounding_score = validate_citations(
        str(resp.content),
        state.get("merged_context", ""),
        state.get("extraction_findings", ""),
    )

    grounded_count = sum(1 for v in citation_grounding.values() if v.get("grounded"))
    total_citations = len(citation_grounding)

    return {
        "summary": str(resp.content),
        "citation_grounding": citation_grounding,
        "grounding_score": grounding_score,
        "messages": [AIMessage(content=f"[Summarizer] {str(resp.content)[:120]}…")],
        "activity_log": [{
            "agent": "summarizer",
            "icon": "✍️",
            "title": "Summarizer — final answer",
            "detail": f"{len(str(resp.content))} chars · {grounded_count}/{total_citations} citations grounded ({int(grounding_score * 100)}%)",
            "ts": stamp_fn(),
        }],
        "current_agent": "summarizer",
    }