# Reading / Extraction Agent — Implementation Plan

## Context

This agent is specified in both the AI Proposal and the Progress Report as a **Layer 2** component sitting between the three retrieval agents (VectorDB, SQL/DB, Web/API) and the Orchestrator. Its deadline is **3/29 – 4/5**.

**Problem it solves:** Without this agent, the downstream Orchestrator and Summarizer receive only raw LLM-synthesised notes from each retrieval agent. There is no structured per-paper record, no explicit provenance tagging (abstract-only vs. full-text vs. structured-db), and no extraction of standardised fields (research problem, methodology, findings, limitations, future work). The Planning Agent (not yet started) also requires this agent's output.

---

## What the Agent Does (per proposal)

- Receives the pooled output of all three retrieval agents (`vector_findings`, `sql_findings`, `web_findings`).
- Parses the combined content and extracts, for **each distinct paper or source** referenced:
  - **Research problem** — what gap or question the paper addresses
  - **Methodology** — how it was studied (e.g. dataset, model, experiment design)
  - **Findings** — key results and contributions
  - **Limitations** — stated or inferred weaknesses
  - **Future work** — directions proposed by the authors
  - **Provenance** — one of: `abstract-only`, `full-text`, `structured-db`
- Returns a structured string (`extraction_findings`) ready for the Orchestrator to merge.

---

## Changes Required to `streamlit_app.py`

### 1. `AgentState` — add new field
```python
extraction_findings: str   # written by reading_extraction_agent, read by orchestrator
```

### 2. New agent function `reading_extraction_agent(state, model)`
- LLM temperature: **0.1** (precision extraction, not creativity)
- Prompt instructs the model to parse the combined retrieval findings and return one structured block per paper.
- Provenance heuristic:
  - If the finding came from `web_findings` and has a URL/DOI → `abstract-only`
  - If from `vector_findings` with large chunk → `full-text`
  - If from `sql_findings` → `structured-db`
- Graceful fallback: if no papers found in any channel, return a minimal "no structured papers extracted" string so downstream agents still proceed.
- Activity log entry with icon `📖`, agent key `reading_extraction`, CSS class `reading_extraction`, card color `#27AE60` (green).

### 3. `build_graph()` — wire new node
```
web → reading_extraction → orchestrator
```
- Use `lm_e = _llm(api_key, 0.1)` (dedicated low-temp LLM for extraction).
- Replace existing `g.add_edge("web", "orchestrator")` with two edges through `reading_extraction`.

### 4. `orchestrator_agent()` — consume `extraction_findings` as 4th block
```python
block = "\n\n".join([
    f"=== Vector DB ===\n{state.get('vector_findings','')}",
    f"=== SQL / DB ===\n{state.get('sql_findings','')}",
    f"=== Web / APIs ===\n{state.get('web_findings','')}",
    f"=== Structured Extraction ===\n{state.get('extraction_findings','')}",  # NEW
])
```
Update the "Sources merged" activity log to also check `extraction_findings`.

### 5. CSS — add new card color
```css
.agent-card.reading_extraction { border-color: #27AE60 }
```

### 6. Progress bar `pct_map` — add entry
```python
"reading_extraction": 62,   # between web (55) and orchestrator (68)
```
Shift `orchestrator` from 68 → 70 if needed.

### 7. `full_state` initialisation — add new key
```python
"extraction_findings": "",
```

### 8. Per-Agent Findings tab — add new expander
```python
("📖 Reading / Extraction", "extraction_findings", "be"),   # new CSS badge class
```
Add `.be { background:#0f3d20; color:#27AE60 }` badge CSS.

### 9. Message log `av` dict — add entry
```python
"[ReadingExtraction]": "📖",
```

### 10. Header docstring — update architecture diagram
Add `[Reading/Extraction Agent]` between `[Web/API Agent]` and `[Orchestrator Agent]`.

---

## Prompt Design for `reading_extraction_agent`

```
System:
You are a Reading and Extraction Agent for academic research.
Given synthesised retrieval findings from multiple sources, extract a structured record
for every distinct paper, study, or source you can identify.

For each paper output a block in this exact format:
---
PAPER: <title or identifier>
PROVENANCE: <abstract-only | full-text | structured-db>
RESEARCH PROBLEM: <one sentence>
METHODOLOGY: <one sentence>
FINDINGS: <bullet points>
LIMITATIONS: <one sentence>
FUTURE WORK: <one sentence>
---

Provenance rules:
- "structured-db" if the source is from the SQL/DB findings
- "full-text" if large chunks of text were available (VectorDB with full paragraphs)
- "abstract-only" if only title/abstract/year was available (Web API results)

If no distinct papers can be identified, output:
NO_PAPERS_EXTRACTED

Be precise. Do not hallucinate citations.
```

---

## Test File

`test_reading_extraction.py` already exists in the repo root. It should:
- Mock a minimal `AgentState` with sample `vector_findings`, `sql_findings`, `web_findings`
- Call `reading_extraction_agent()` with a stub LLM or real call
- Assert that `extraction_findings` is a non-empty string
- Assert that provenance tags appear in the output

---

## Non-goals for this branch

- MMR-based VectorDB diversification (separate task)
- Scoping Agent (separate task, Layer 1)
- Planning Agent (depends on this agent, but separate task)
- Evaluation / baseline comparison

---

## File Touch-list

| File | Change |
|---|---|
| `streamlit_app.py` | All 10 changes above |
| `reading_extraction_plan.md` | This document (reference only) |
| `test_reading_extraction.py` | Implement/verify tests |
