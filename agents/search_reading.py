"""
agents/search_reading.py — SearchReadingAgent (merged Search + Reading)

Handles source selection, paper retrieval, and structured summarisation.

Source selection
────────────────
Before searching, the agent makes one LLM call to inspect the source
registry and select 2-4 sources appropriate for the research domain.
Selection is stored in state so the Orchestrator and UI can observe it.
Selections are cached for the session — targeted searches reuse them.

Search behaviour
────────────────
Queries each selected source for every sub-question.
Scoring: relevance_rank × log(1 + citation_count).
Deduplication: by normalised title; source priority order used when the
same paper appears in multiple sources (semantic_scholar > pubmed >
europe_pmc > openalex > arxiv > crossref > dblp).
Three-tier error handling:
  Tier 1: HTTP 429/503 → exponential backoff (2, 4, 8 s), up to 3 retries
  Tier 2: zero results → LLM query reformulation, one retry per source
  Tier 3: persistent failure → logged to state, sub-question flagged

Summarisation
─────────────
Papers with has_abstract=True get a 5-field LLM summary.
Papers with has_abstract=False (Crossref, DBLP) are kept in the corpus
for citation tracking but receive a metadata-only stub — no LLM call.
Schema validated; one JSON-only retry on parse failure.

Receives:  task from orchestrator
Sends:     result to orchestrator
"""

import re
import io
import math
import time
import requests
from message_bus import Message, MessageBus
from agents.base import BaseAgent
from tools.sources import search_source, list_available_sources, build_citation_edges

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an academic research specialist responsible for selecting appropriate
databases, retrieving papers, and extracting structured summaries from them.
You understand each database's domain coverage and limitations.
Return only valid JSON — no preamble, no markdown fences."""

SOURCE_SELECTION_PROMPT = """\
Research query: "{query}"

Sub-questions to cover:
{sub_questions}

Available sources:
{registry_block}

Select 2-4 sources appropriate for this research domain. Rank by priority
(most relevant first). Consider:
  - Domain fit (best_for)
  - Whether abstracts are available (has_abstracts: true strongly preferred)
  - Coverage complementarity

Return:
{{
  "selected_sources": ["source_id_1", "source_id_2"],
  "rationale": {{
    "source_id_1": "one sentence why this source is ranked first",
    "source_id_2": "one sentence"
  }}
}}"""

REFORMULATION_PROMPT = """\
Original query: "{query}"

This query returned no results from academic databases. Reformulate it as a
shorter, more general search query (3-6 words, no domain jargon).
Return only JSON:
{{"query": "the reformulated query", "reasoning": "one sentence"}}"""

SUMMARY_PROMPT = """\
Paper title:    {title}
Authors:        {authors}
Year:           {year}
Citation count: {citations}
Source:         {source}

Text:
\"\"\"
{text}
\"\"\"

Return exactly this JSON (no other text):
{{
  "key_claims":     ["specific claim 1", "specific claim 2"],
  "methods":        "concise description of the approach or methodology",
  "findings":       "main findings or contributions, 1-2 sentences",
  "future_work":    "what the authors suggest for future work, or 'not stated'",
  "relevance_note": "one sentence on relevance to the research query"
}}"""

# Source priority for deduplication (lower = preferred)
SOURCE_PRIORITY = {
    "semantic_scholar": 1,
    "pubmed":           2,
    "europe_pmc":       3,
    "openalex":         4,
    "arxiv":            5,
    "crossref":         6,
    "dblp":             7,
}

PDF_CHAR_LIMIT = 3000
TOP_N_PAPERS   = 15
READ_CAP       = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def _score(paper: dict, rank: int) -> float:
    citations = paper.get("citation_count", 0) or 0
    return (1.0 / (rank + 1)) * math.log1p(citations + 1)


def _try_pdf(url: str) -> str:
    try:
        import fitz
        resp = requests.get(url, timeout=20,
                            headers={"User-Agent": "research-assistant/1.0"})
        resp.raise_for_status()
        doc  = fitz.open(stream=io.BytesIO(resp.content), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) >= PDF_CHAR_LIMIT:
                break
        doc.close()
        return text[:PDF_CHAR_LIMIT]
    except Exception:
        return ""


# ── Agent ─────────────────────────────────────────────────────────────────────

class SearchReadingAgent(BaseAgent):
    name          = "search_reading_agent"
    system_prompt = SYSTEM_PROMPT

    def __init__(self, provider):
        super().__init__(provider)

    def run(self, message: Message, state: dict, bus: MessageBus) -> Message:
        targeted_query = message.content.get("targeted_query")
        targeted_label = message.content.get("targeted_label", "")

        if targeted_query:
            return self._targeted_search(targeted_query, targeted_label,
                                         message, state, bus)
        else:
            return self._full_pipeline(message, state, bus)

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def _full_pipeline(self, message: Message, state: dict,
                       bus: MessageBus) -> Message:
        sub_questions = state.get("sub_questions", []) or [state.get("query", "")]

        # Select sources (cached after first call)
        sources = self._get_or_select_sources(state)

        all_new_papers: list = []
        failed_subqs:   list = []

        for sub_q in sub_questions:
            papers = self._search_sources(sub_q, sources, state)
            if not papers:
                failed_subqs.append(sub_q)
            all_new_papers.extend(papers)

        all_new_papers = self._dedup(all_new_papers, state)
        all_new_papers = sorted(all_new_papers,
                                key=lambda p: p.get("_score", 0), reverse=True)[:TOP_N_PAPERS]
        for p in all_new_papers:
            p.pop("_score", None)

        state.setdefault("all_papers", []).extend(all_new_papers)

        # Summarise — skip no-abstract papers
        to_summarise = sorted(
            [p for p in all_new_papers if p.get("has_abstract", True)],
            key=lambda p: p.get("citation_count", 0), reverse=True,
        )[:READ_CAP]
        summaries = self._summarise_papers(to_summarise)

        # Add metadata stubs for no-abstract papers so they appear in the UI
        no_abstract = [p for p in all_new_papers if not p.get("has_abstract", True)]
        for p in no_abstract:
            summaries.append({**p, "summary": {
                "key_claims": [],
                "methods": "",
                "findings": "(No abstract available — metadata only)",
                "future_work": "not stated",
                "relevance_note": "",
            }})

        state.setdefault("paper_summaries", []).extend(summaries)

        # Build citation graph from Semantic Scholar references
        try:
            edges = build_citation_edges(state["paper_summaries"], max_refs_per_paper=8)
            state["citation_edges"] = edges
        except Exception:
            state.setdefault("citation_edges", [])

        src_counts = {}
        for p in all_new_papers:
            s = p.get("source", "?")
            src_counts[s] = src_counts.get(s, 0) + 1

        return self._reply(message, msg_type="result", content={
            "papers_found":     len(all_new_papers),
            "papers_read":      len([s for s in summaries if s.get("summary", {}).get("findings") != "(No abstract available — metadata only)"]),
            "sources_used":     sources,
            "source_breakdown": src_counts,
            "failed_subqs":     failed_subqs,
            "total_corpus":     len(state["all_papers"]),
            "total_summarised": len(state["paper_summaries"]),
            "summary": (
                f"Retrieved {len(all_new_papers)} papers from {sources} "
                f"(corpus: {len(state['all_papers'])})"
                + (f" — {len(failed_subqs)} sub-question(s) returned no results"
                   if failed_subqs else "")
            ),
        }, bus=bus)

    # ── Targeted search ───────────────────────────────────────────────────────

    def _targeted_search(self, query: str, label: str,
                         message: Message, state: dict, bus: MessageBus) -> Message:
        sources = self._get_or_select_sources(state)
        papers  = self._search_sources(query, sources, state)
        papers  = self._dedup(papers, state)[:5]
        for p in papers:
            p.pop("_score", None)

        state.setdefault("all_papers", []).extend(papers)
        summaries = self._summarise_papers(
            [p for p in papers if p.get("has_abstract", True)]
        )
        state.setdefault("paper_summaries", []).extend(summaries)

        return self._reply(message, msg_type="result", content={
            "papers_found":     len(papers),
            "papers_read":      len(summaries),
            "total_corpus":     len(state["all_papers"]),
            "total_summarised": len(state["paper_summaries"]),
            "summary": (
                f"Targeted '{query}' ({label}): "
                f"{len(papers)} new papers, {len(summaries)} summarised"
            ),
        }, bus=bus)

    # ── Source selection ──────────────────────────────────────────────────────

    def _get_or_select_sources(self, state: dict) -> list:
        """Return previously selected sources, or run the selection LLM call."""
        cached = state.get("sources_selected")
        if cached:
            return cached

        registry = list_available_sources()
        sub_q_block = "\n".join(f"- {q}" for q in state.get("sub_questions", []))

        registry_block = "\n".join(
            f"  {sid}:\n"
            f"    best_for: {', '.join(info['best_for'][:5])}\n"
            f"    has_abstracts: {info['has_abstracts']}\n"
            f"    has_citations: {info['has_citations']}\n"
            f"    description: {info['description']}"
            for sid, info in registry.items()
        )

        try:
            raw  = self._llm(
                SOURCE_SELECTION_PROMPT.format(
                    query=state.get("query", ""),
                    sub_questions=sub_q_block,
                    registry_block=registry_block,
                ),
                max_tokens=500,
            )
            data = self._parse_json(raw)
            if data and data.get("selected_sources"):
                selected  = [s for s in data["selected_sources"] if s in registry]
                rationale = data.get("rationale", {})
                if selected:
                    state["sources_selected"] = selected
                    state["source_rationale"] = rationale
                    return selected
        except Exception:
            pass

        # Fallback: semantic_scholar + arxiv
        fallback = ["semantic_scholar", "arxiv"]
        state["sources_selected"] = fallback
        state["source_rationale"] = {}
        return fallback

    # ── Multi-source search for one sub-question ──────────────────────────────

    def _search_sources(self, sub_q: str, sources: list, state: dict) -> list:
        results: list = []
        for source in sources:
            dedup_key = f"{sub_q}|||{source}"
            used      = state.setdefault("search_queries_used", [])
            if dedup_key in used:
                continue
            used.append(dedup_key)

            papers = search_source(source, sub_q, max_results=10)
            time.sleep(0.35)

            # Tier 2: empty → reformulate once
            if not papers:
                reformulated = self._reformulate_query(sub_q)
                if reformulated and reformulated != sub_q:
                    retry_key = f"{reformulated}|||{source}"
                    if retry_key not in used:
                        used.append(retry_key)
                        papers = search_source(source, reformulated, max_results=10)
                        time.sleep(0.35)

            for rank, p in enumerate(papers):
                p["_score"] = _score(p, rank)
            results.extend(papers)

        return results

    # ── Deduplication ─────────────────────────────────────────────────────────

    def _dedup(self, papers: list, state: dict) -> list:
        """
        Remove papers already in corpus.
        When multiple sources return the same paper, keep the one with the
        highest priority (lowest SOURCE_PRIORITY value).
        """
        existing = {_norm(p["title"]) for p in state.get("all_papers", [])}
        seen: dict[str, dict] = {}   # norm_title → paper dict

        for p in papers:
            nt = _norm(p["title"])
            if nt in existing:
                continue
            if nt not in seen:
                seen[nt] = p
            else:
                # Keep whichever source has higher priority (lower number)
                cur_prio = SOURCE_PRIORITY.get(seen[nt].get("source", ""), 99)
                new_prio = SOURCE_PRIORITY.get(p.get("source", ""), 99)
                if new_prio < cur_prio:
                    seen[nt] = p

        return list(seen.values())

    # ── Summarisation ─────────────────────────────────────────────────────────

    def _summarise_papers(self, papers: list) -> list:
        results = []
        for paper in papers:
            results.append({**paper, "summary": self._summarise_one(paper)})
        return results

    def _summarise_one(self, paper: dict) -> dict:
        text = paper.get("abstract", "")
        if not text and paper.get("pdf_url"):
            text = _try_pdf(paper["pdf_url"])
        if not text:
            text = "(No abstract or full text available.)"

        authors_str = ", ".join(paper.get("authors", [])[:4])
        if len(paper.get("authors", [])) > 4:
            authors_str += " et al."

        prompt = SUMMARY_PROMPT.format(
            title=paper["title"],
            authors=authors_str or "Unknown",
            year=paper.get("year") or "Unknown",
            citations=paper.get("citation_count", 0),
            source=paper.get("source", "unknown"),
            text=text[:2500],
        )

        try:
            raw  = self._llm(prompt, max_tokens=600)
            data = self._parse_json(raw)
            if data:
                return data
            raw2  = self._llm(prompt + "\n\nRespond with ONLY the JSON object.", max_tokens=600)
            data2 = self._parse_json(raw2)
            if data2:
                return data2
        except Exception:
            pass

        return {
            "key_claims":     [],
            "methods":        "",
            "findings":       (paper.get("abstract") or "")[:300],
            "future_work":    "not stated",
            "relevance_note": "",
        }

    def _reformulate_query(self, query: str) -> str:
        try:
            raw  = self._llm(REFORMULATION_PROMPT.format(query=query), max_tokens=120)
            data = self._parse_json(raw)
            if data and data.get("query"):
                return data["query"]
        except Exception:
            pass
        return " ".join(w for w in query.split() if len(w) > 3)[:4]
