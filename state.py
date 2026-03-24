"""state.py — ResearchState TypedDict (5-agent architecture)."""
from typing import TypedDict, Optional, Annotated
import operator


class ResearchState(TypedDict):
    # Input
    query:              str

    # Scoping
    sub_questions:      list           # list[str]

    # Search & Reading
    all_papers:         list           # list[paper dict]
    search_queries_used: list          # list[str] dedup keys
    paper_summaries:    list           # list[paper + summary dict]
    sources_selected:   list           # list[str] source IDs chosen by agent
    source_rationale:   dict           # source_id → rationale string

    # Reading/Extraction Agent output
    # Each record: {title, authors, year, source, doi, url, pdf_url,
    #   citation_count, research_problem, methodology, findings, limitations,
    #   future_work, key_claims, provenance: {text_source, abstract_only_flag,
    #   pdf_attempted, text_chars, confidence}}
    extracted_papers:   list           # list[ExtractedPaper dict]

    # ── Add new agent output fields below this line ───────────────────────────
    # Follow the pattern above: one field per agent, with a comment block
    # explaining the schema. Keep fields grouped by the agent that writes them.

    # Synthesis & Planning
    synthesis:          str
    gaps:               str
    research_plan:      str
    themes:             list           # list[{theme_name, description, paper_titles}]
    contradictions:     list           # list[str]
    uncovered_sub_questions: list      # list[str] sub-questions with <2 papers
    research_steps:     list           # list[{step, title, description, grounding_papers, future_work_link}]
    risks_and_mitigations: str

    # Citation graph (built from Semantic Scholar references)
    citation_edges:     list           # list[{source: title, target: title}]

    # Validation
    validation_result:  dict
    validation_iterations: int

    # Observability
    orchestrator_trace: Annotated[list, operator.add]
    status_log:         Annotated[list, operator.add]
    message_log:        list           # bus summaries for UI
    bus_messages:       list           # full bus message dicts for UI
    provider_summary:   dict           # agent → provider name

    # Control
    is_complete:        bool
    error:              Optional[str]
