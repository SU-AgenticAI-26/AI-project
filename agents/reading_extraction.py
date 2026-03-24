"""
agents/reading_extraction.py — ReadingExtractionAgent

Converts raw retrieved papers into structured evidence records with
per-paper field extraction and explicit provenance metadata.

For each paper in the retrieved corpus (state["all_papers"]), this agent:
  1. Resolves the best available text:
       - Attempts full-text extraction via PDF (PyMuPDF) if pdf_url is present
       - Falls back to abstract if PDF is unavailable or fails to parse
       - Falls back to a metadata stub if no abstract exists
  2. Runs a 6-field LLM extraction:
       research_problem · methodology · findings · limitations ·
       future_work · key_claims
  3. Attaches provenance metadata that explicitly flags the text source,
     confidence level, and whether the record is abstract-only — enabling
     downstream agents and the UI to communicate data completeness.

Output is written to state["extracted_papers"]. SynthesisPlanningAgent
reads this field in preference to paper_summaries when it is populated.

Architecture note
─────────────────
This agent sits between the retrieval layer (SearchReadingAgent, VectorDB,
Web/API) and the synthesis layer (SynthesisPlanningAgent). It runs once
per pipeline after all retrieval is complete. The Orchestrator calls it
via send_to_reading_extraction_agent.

Integration note for new agents
────────────────────────────────
To add a new agent to the pipeline, you only need to:
  1. Create agents/your_agent.py (set tool_name, tool_description, tool_schema)
  2. Add ("your_agent_key", YourAgentClass) to AGENT_REGISTRY in orchestrator.py
  3. Add "your_agent_key" to AGENT_NAMES in llm/registry.py
  4. Add a state field to state.py
  5. Add one import to agents/__init__.py
See agents/base.py for the tool descriptor protocol.

Receives:  task from orchestrator
Sends:     result to orchestrator
"""

import io
import re
import time
import requests

from message_bus import Message, MessageBus
from agents.base import BaseAgent

# ── Constants ─────────────────────────────────────────────────────────────────

EXTRACT_CAP               = 10    # max papers to fully extract (top by citation count)
PDF_CHAR_LIMIT            = 4000  # characters to read from a fetched PDF
ABSTRACT_CHAR_LIMIT       = 2500  # characters to pass from abstract
CONFIDENCE_HIGH_THRESHOLD = 500   # chars required for "high" confidence
CONFIDENCE_MEDIUM_THRESHOLD = 50  # chars required for "medium" confidence

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a rigorous academic literature analyst. Extract structured information
from research papers with precision. Use only what is explicitly stated in the
provided text — do not infer or hallucinate content. When a field is not
addressed in the text, return exactly the string "not stated".
Return only valid JSON — no preamble, no markdown fences."""

EXTRACTION_PROMPT = """\
Paper title:   {title}
Authors:       {authors}
Year:          {year}
Source:        {source}
Citations:     {citations}
Text source:   {text_source_label}

Text:
\"\"\"{text}
\"\"\"

Extract the following fields from the text above.
Return "not stated" for any field not addressed in the text.
Return exactly this JSON:
{{
  "research_problem": "the specific problem or gap the paper addresses",
  "methodology":      "how they studied it (methods, datasets, evaluation approach)",
  "findings":         "main results or contributions, 2-3 sentences",
  "limitations":      "acknowledged constraints, weaknesses, or scope limits",
  "future_work":      "what the authors suggest as next steps",
  "key_claims":       ["most important claim 1", "most important claim 2"]
}}"""

# ── PDF fetcher ───────────────────────────────────────────────────────────────

def _fetch_pdf(url: str) -> str:
    """
    Download and extract plain text from a PDF URL using PyMuPDF (fitz).
    Returns an empty string on any failure (missing library, network error,
    parse error) so callers can fall back to abstract.
    """
    try:
        import fitz  # PyMuPDF — optional dependency
        resp = requests.get(
            url, timeout=20,
            headers={"User-Agent": "research-assistant/1.0"},
        )
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

# ── Provenance helpers ────────────────────────────────────────────────────────

def _confidence(text_chars: int) -> str:
    if text_chars >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if text_chars >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def _resolve_text(paper: dict) -> tuple:
    """
    Determine the best available text for a paper.

    Returns (text: str, text_source: str, pdf_attempted: bool) where
    text_source is one of:
      "full_text"     — text extracted from PDF
      "abstract_only" — abstract from the API response
      "metadata_only" — no text available; only title/authors/year
    """
    pdf_attempted = False

    # 1. Try PDF
    pdf_url = paper.get("pdf_url", "")
    if pdf_url:
        pdf_attempted = True
        pdf_text = _fetch_pdf(pdf_url)
        if pdf_text.strip():
            return pdf_text, "full_text", True

    # 2. Fall back to abstract
    abstract = (paper.get("abstract") or "").strip()
    if abstract:
        return abstract[:ABSTRACT_CHAR_LIMIT], "abstract_only", pdf_attempted

    # 3. Metadata stub — title + authors + year
    stub = " ".join(filter(None, [
        paper.get("title", ""),
        ", ".join(paper.get("authors", [])[:3]),
        str(paper.get("year", "")),
    ]))
    return stub, "metadata_only", pdf_attempted


# ── Agent ─────────────────────────────────────────────────────────────────────

class ReadingExtractionAgent(BaseAgent):
    name          = "reading_extraction_agent"
    system_prompt = SYSTEM_PROMPT

    # ── Orchestrator tool registration ────────────────────────────────────
    tool_name        = "send_to_reading_extraction_agent"
    tool_description = (
        "Process all retrieved papers into structured 6-field records "
        "(research_problem, methodology, findings, limitations, future_work, "
        "key_claims) with provenance metadata flagging abstract-only vs "
        "full-text sources. Call after send_to_search_reading_agent and "
        "before send_to_synthesis_planning_agent."
    )
    tool_schema      = {"type": "object", "properties": {}, "required": []}

    def __init__(self, provider):
        super().__init__(provider)

    def run(self, message: Message, state: dict, bus: MessageBus) -> Message:
        all_papers = state.get("all_papers", [])

        if not all_papers:
            return self._reply(message, msg_type="result", content={
                "extracted":          0,
                "full_text":          0,
                "abstract_only":      0,
                "metadata_only":      0,
                "extraction_errors":  0,
                "summary":            "No papers in corpus — extraction skipped",
            }, bus=bus)

        # Top papers by citation count, capped at EXTRACT_CAP
        candidates = sorted(
            all_papers,
            key=lambda p: p.get("citation_count", 0),
            reverse=True,
        )[:EXTRACT_CAP]

        extracted_papers = []
        n_full    = 0
        n_abstract = 0
        n_metadata = 0
        n_errors  = 0

        for paper in candidates:
            text, text_source, pdf_attempted = _resolve_text(paper)

            provenance = {
                "text_source":        text_source,
                "abstract_only_flag": text_source != "full_text",
                "pdf_attempted":      pdf_attempted,
                "text_chars":         len(text),
                "confidence":         _confidence(len(text)),
            }

            if text_source == "metadata_only":
                record = self._make_stub(paper, provenance)
                n_metadata += 1
            else:
                record = self._extract(paper, text, text_source, provenance)
                if record is None:
                    record = self._make_stub(paper, provenance)
                    n_errors += 1
                    state.setdefault("status_log", []).append(
                        f"[ReadingExtraction] extraction failed for: {paper['title'][:60]}"
                    )
                elif text_source == "full_text":
                    n_full += 1
                else:
                    n_abstract += 1

            extracted_papers.append(record)
            time.sleep(0.1)  # courtesy pause between LLM calls

        state["extracted_papers"] = extracted_papers

        return self._reply(message, msg_type="result", content={
            "extracted":          len(extracted_papers),
            "full_text":          n_full,
            "abstract_only":      n_abstract,
            "metadata_only":      n_metadata,
            "extraction_errors":  n_errors,
            "summary": (
                f"Extracted {len(extracted_papers)} papers — "
                f"{n_full} full-text, {n_abstract} abstract-only, "
                f"{n_metadata} metadata-only"
                + (f" ({n_errors} extraction error(s))" if n_errors else "")
            ),
        }, bus=bus)

    # ── Extraction helpers ────────────────────────────────────────────────

    def _extract(self, paper: dict, text: str, text_source: str,
                 provenance: dict) -> "dict | None":
        """
        Run the LLM extraction prompt. Returns a full record dict on
        success, or None if JSON parsing fails after one retry.
        """
        authors_str = ", ".join(paper.get("authors", [])[:4])
        if len(paper.get("authors", [])) > 4:
            authors_str += " et al."

        text_source_label = {
            "full_text":     "full text (PDF)",
            "abstract_only": "abstract only",
        }.get(text_source, text_source)

        prompt = EXTRACTION_PROMPT.format(
            title=paper["title"],
            authors=authors_str or "Unknown",
            year=paper.get("year") or "Unknown",
            source=paper.get("source", "unknown"),
            citations=paper.get("citation_count", 0),
            text_source_label=text_source_label,
            text=text,
        )

        try:
            raw  = self._llm(prompt, max_tokens=700)
            data = self._parse_json(raw)
            if not data:
                # One retry with an explicit JSON reminder
                raw2 = self._llm(
                    prompt + "\n\nRespond with ONLY the JSON object.",
                    max_tokens=700,
                )
                data = self._parse_json(raw2)
            if data:
                return self._build_record(paper, data, provenance)
        except Exception:
            pass
        return None

    def _build_record(self, paper: dict, extraction: dict,
                      provenance: dict) -> dict:
        return {
            # ── Identity ─────────────────────────────────────────────
            "title":          paper["title"],
            "authors":        paper.get("authors", []),
            "year":           paper.get("year"),
            "source":         paper.get("source", ""),
            "doi":            paper.get("doi"),
            "url":            paper.get("url"),
            "pdf_url":        paper.get("pdf_url"),
            "citation_count": paper.get("citation_count", 0),
            # ── Structured extraction ─────────────────────────────────
            "research_problem": extraction.get("research_problem", "not stated"),
            "methodology":      extraction.get("methodology",      "not stated"),
            "findings":         extraction.get("findings",         "not stated"),
            "limitations":      extraction.get("limitations",      "not stated"),
            "future_work":      extraction.get("future_work",      "not stated"),
            "key_claims":       extraction.get("key_claims",       []),
            # ── Provenance ────────────────────────────────────────────
            "provenance":       provenance,
        }

    def _make_stub(self, paper: dict, provenance: dict) -> dict:
        """Minimal record for papers with no extractable text (no LLM call)."""
        return self._build_record(paper, {
            "research_problem": "not stated",
            "methodology":      "not stated",
            "findings":         (paper.get("abstract") or "")[:200] or "not stated",
            "limitations":      "not stated",
            "future_work":      "not stated",
            "key_claims":       [],
        }, provenance)
