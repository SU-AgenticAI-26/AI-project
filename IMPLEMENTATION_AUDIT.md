# Implementation Audit: Proposal vs. Current System

**Date**: April 11, 2026  
**Status**: Most core features implemented; evaluation framework and advanced features missing

---

## Executive Summary

| Category | Status | Coverage |
|----------|--------|----------|
| **Agent Architecture** | ✅ **COMPLETE** | All 10 agents implemented with LLM tool calling |
| **Retrieval Pipeline** | ✅ **COMPLETE** | Vector DB + SQL DB + Web APIs fully integrated |
| **Data Extraction** | ✅ **MOSTLY COMPLETE** | Structured per-paper records (problem/methods/findings/limits/future work) |
| **Knowledge Representation** | ✅ **COMPLETE** | PyVis-based knowledge graph visualization |
| **Synthesis & Planning** | ✅ **COMPLETE** | Full synthesis + experiment design suggestions |
| **Validation Loops** | ✅ **PARTIAL** | Critic agent loops up to 2x; missing conflict resolution |
| **Evaluation Framework** | ❌ **NOT IMPLEMENTED** | No metrics, baselines, or user study infrastructure |
| **Query Intelligence** | ⚠️ **PARTIAL** | Conference paper detection; no scoping agent or query rewriting |
| **PDF Processing** | ❌ **NOT IMPLEMENTED** | No PyMuPDF integration |
| **UI/Export** | ✅ **PARTIAL** | Streamlit UI complete; no report export feature |

---

## Detailed Implementation Status

### ✅ 1. Core System Architecture

**Implemented**:
- ✅ LangGraph StateGraph with 10 sequential agents
- ✅ Conditional graph edges (agents skip if not in active_agents list)  
- ✅ Shared AgentState with all required fields:
  - `messages`, `activity_log`, `query`
  - `vector_findings`, `sql_findings`, `web_findings`
  - `merged_context`, `knowledge_map`, `summary`
  - `critique`, `experiment_plan`, `loop_count`
- ✅ Orchestrator agent for merging/deduplication
- ✅ Session state persistence

**Missing**:
- ❌ Parallel agent execution (currently sequential only)

---

### ✅ 2. Agent Modules

#### Layer 1: Query Understanding

| Agent | Implemented | Details |
|-------|-------------|---------|
| **Router Agent** | ✅ **YES** | Decides which agents to activate (vector/sql/web) based on query |
| **Scoping Agent** | ❌ **NO** | Could extract 3–5 sub-questions and 4–8 keywords per query |

**Router Implementation**:
```python
def router_agent(state, model):
    # Decides: active_agents = ["vector_db", "sql_db", "web"] or subsets
    # Based on query analysis
```

**Gap**: No explicit scoping agent to decompose queries. Router currently makes binary decisions, not structured decomposition.

---

#### Layer 2: Retrieval + Processing

| Component | Status | Details |
|-----------|--------|---------|
| **Vector DB Agent** | ✅ **COMPLETE** | FAISS-based semantic search with LLM tool calling |
| **SQL DB Agent** | ✅ **COMPLETE** | SQLite structured queries with LLM tool calling |
| **Web Agent** | ✅ **COMPLETE** | Conference papers + arXiv + Semantic Scholar + OpenAlex + Crossref |
| **Tool Calling** | ✅ **COMPLETE** | All 3 agents use `tool_choice="auto"` for LLM decision-making |
| **Fallback Logic** | ✅ **YES** | Non-OpenAI models fall back to direct search |
| **Query Rewriting** | ❌ **NO** | No query expansion or reformulation |

**Web Agent APIs Integrated**:
```
✅ Conference Papers (OpenReview + ACL Anthology)
✅ arXiv (direct API)
✅ Semantic Scholar (via requests)
✅ OpenAlex (via requests)
✅ Crossref (via requests)
```

**Gap**: No query rewriting step. Some sources might miss relevant papers if query syntax doesn't match.

---

#### Reading/Extraction Agent

| Field | Status | Details |
|-------|--------|---------|
| **Problem/Research Question** | ✅ | Extracted per-paper |
| **Methods/Methodology** | ✅ | Extracted per-paper |
| **Findings/Results** | ✅ | Extracted per-paper |
| **Limitations** | ✅ | Extracted per-paper |
| **Future Work** | ✅ | Extracted per-paper |
| **Provenance Tracking** | ✅ | Source tagged (vector_db/sql_db/web/merged) |
| **PDF Parsing (PyMuPDF)** | ❌ | Not integrated |
| **Abstract Fallback** | ✅ | Uses abstract when full-text unavailable |

**Current Implementation**:
- Extracts structured records from retrieved text
- Tags each field with source
- Does NOT parse PDFs directly (would need PyMuPDF)

**Gap**: PDF processing not implemented. System work with API-returned metadata/abstracts only.

---

#### Orchestrator Agent

| Feature | Status | Details |
|---------|--------|---------|
| **Merge Results** | ✅ | Combines vector/sql/web findings |
| **Deduplication** | ✅ | Removes duplicate insights |
| **Weighting** | ⚠️ | Mentioned in findings but not explicitly weighted |
| **Provenance Tracking** | ✅ | Maintains source attribution |
| **Conflict Resolution** | ❌ | No credibility weighting or conflict handling |

**Gap**: No explicit conflict resolution for contradicting findings. No credibility scoring for sources.

---

### ✅ 3. Knowledge Representation

| Feature | Status | Details |
|---------|--------|---------|
| **Graph Construction** | ✅ | LLM extracts 12–20 nodes per query |
| **Node Types** | ✅ | concept / entity / fact / process |
| **Source Tagging** | ✅ | Nodes tagged by source (vector_db/sql_db/web/merged) |
| **Edge Relations** | ✅ | Directed edges with relation labels and weights |
| **PyVis Visualization** | ✅ | Interactive HTML graph with physics simulation |
| **NetworkX Backend** | ✅ | PyVis uses NetworkX internally |
| **Persistent Storage** | ✅ | Knowledge maps saved to JSON |

**Visualization Features**:
- ✅ Color coding by source
- ✅ Shape coding by node type (dot/diamond/square/triangleDown)
- ✅ Physics simulation (forceAtlas2Based)
- ✅ Interactive hover tooltips
- ✅ Directed arrows with relation labels

**Gap**: No advanced filtering (e.g., "show only high-weight edges") or community detection.

---

### ✅ 4. Synthesis & Planning

| Agent | Status | Details |
|-------|--------|---------|
| **Summarizer** | ✅ | Generates citation-grounded answer |
| **Experiment Design** | ✅ | Proposes research plan with gaps/hypotheses/methods/datasets |
| **Thematic Grouping** | ⚠️ | Implicit in synthesis; not explicit categorization |

**Experiment Design Output** (Suggested Research Plan):
- ✅ Research gaps (gap analysis)
- ✅ Hypotheses (testable predictions)
- ✅ Proposed methods (techniques to test)
- ✅ Datasets/benchmarks (what to evaluate on)
- ✅ Challenges (known blockers)
- ✅ Next steps (sequencing)

**Gap**: No explicit thematic grouping step. Synthesis happens in final summarizer, but findings not categorized by theme (methods vs. findings vs. theoretical insights).

---

### ✅ 5. Validation & Feedback Loop

| Feature | Status | Details |
|---------|--------|---------|
| **Critic Agent** | ✅ | Evaluates knowledge map completeness |
| **Feedback Generation** | ✅ | Structured JSON feedback |
| **Loop Control** | ✅ | Max 2 iterations (loop_count checks) |
| **Revision Trigger** | ✅ | Re-triggers retrieval if nodes < 8 |
| **Citation Grounding Check** | ⚠️ | Done in summarizer; not formal validation |
| **Coherence Evaluation** | ❌ | No 1–4 scale evaluation |
| **Consistency Checking** | ❌ | No conflict detection between sources |
| **Credibility Weighting** | ❌ | No source credibility scores |

**Critic Logic**:
```python
if num_nodes < 8 or low_source_diversity:
    needs_more = True  # Trigger loop
else:
    needs_more = False  # Accept result
```

**Gaps**:
- No formal coherence scoring
- No consistency check (conflicting findings not detected)
- No credibility weighting for sources

---

### ❌ 6. Evaluation Framework (NOT IMPLEMENTED)

This is the **major gap** in the proposal.

#### A. Baseline Comparison

**Proposed**:
- Single LLM (GPT-4o-mini) with NO retrieval
- Compare results directly

**Current**:
- ❌ No baseline implementation
- ❌ No comparative results

---

#### B. Quantitative Metrics

| Metric | Proposed Target | Status |
|--------|-----------------|--------|
| **Precision@10** | ≥10% improvement over baseline | ❌ Not implemented |
| **Source Diversity Score** | [0–1] aggregation | ❌ Not implemented |
| **Citation Accuracy** | ≥80% supported | ❌ Not implemented |
| - Supported | Correct citation to source | ❌ |
| - Hallucinated | No source match | ❌ |
| - Partially supported | Partial correctness | ❌ |

---

#### C. Qualitative Metrics

| Metric | Scale | Status |
|--------|-------|--------|
| **Thematic Coherence** | 1–4 | ❌ Not implemented |
| **Completeness** | [as part of coherence] | ❌ |
| **Clarity** | [user study only] | ❌ |

---

#### D. Ground Truth Construction

**Proposed**:
1. Each reviewer selects top-10 papers per query
2. Keep papers selected by ≥2 reviewers
3. Validate with 2 additional reviewers

**Current**:
- ❌ No ground truth collection infrastructure
- ❌ No reviewer workflow

---

#### E. User Study

**Proposed**:
- ~3 users
- Blind evaluation (baseline vs. system)
- Metrics: relevance, completeness, clarity, usefulness

**Current**:
- ❌ No user study infrastructure

---

### ⚠️ 7. Query Intelligence & Optimization

| Feature | Status | Details |
|---------|--------|---------|
| **Conference Detection** | ✅ | Recognizes "NeurIPS", "ICML", etc. → activates web agent |
| **Query Rewriting** | ❌ | No query expansion or multi-hop reformulation |
| **Scoping / Sub-questions** | ❌ | No decomposition into sub-problems |
| **Keyword Extraction** | ✅ | Implicit in router; not explicit output |
| **Query Expansion** | ❌ | No synonym detection or related terms |

---

### ⚠️ 8. PDF Processing

| Feature | Status | Details |
|---------|--------|---------|
| **PyMuPDF Integration** | ❌ | Not implemented |
| **PDF Parsing** | ❌ | None |
| **Structured Extraction from PDFs** | ❌ | Would require PyMuPDF |
| **Text Chunking** | ✅ | Using RecursiveCharacterTextSplitter (on already-retrieved text) |
| **Fallback to Abstracts** | ✅ | Uses abstract when full-text unavailable |

**Gap**: System assumes text is already extracted (from APIs). No direct PDF file ingestion.

---

### ✅ 9. Output & Interface

| Feature | Status | Details |
|---------|--------|---------|
| **Streamlit UI** | ✅ | Full interface implemented |
| **Literature Review Display** | ✅ | Shows merged context + per-paper records |
| **Knowledge Graph Visualization** | ✅ | Interactive PyVis graph |
| **Research Plan Display** | ✅ | Shows gaps, hypotheses, methods, datasets, challenges |
| **Agent Activity Log** | ✅ | Real-time activity cards |
| **Session Persistence** | ✅ | Saves to JSON sessions |
| **Report Export** | ❌ | No PDF/Markdown export |
| **Citation Export** | ❌ | No BibTeX or CSL-JSON export |

**UI Components Implemented**:
- ✅ Collaborative tab (manual web search + RAG)
- ✅ Agent Research tab (automated pipeline)
- ✅ Settings panel (model selection, API keys)
- ✅ Knowledge tab (history + saved knowledge maps)
- ✅ VectorDB management

---

## Gap Analysis: What's Missing

### High Priority (Breaks Proposal Completeness)

1. **Evaluation Framework** (CRITICAL)
   - No baseline comparison (single LLM)
   - No metrics implementation (Precision@10, citation accuracy, diversity)
   - No ground truth collection
   - No user study infrastructure
   - **Impact**: Cannot validate claim of "≥10% improvement"

2. **Citation Grounding Validation**
   - Summarizer mentions citations but doesn't validate them
   - No formal check: "Does this citation index actually appear in retrieved text?"
   - **Impact**: Cannot guarantee ≥80% citation accuracy

3. **Conflict Resolution**
   - No detection of contradicting findings across sources
   - No credibility weighting or source ranking
   - **Impact**: System may present conflicting information without acknowledgment

---

### Medium Priority (Reduces Proposal Scope)

4. **Query Decomposition (Scoping Agent)**
   - No explicit sub-question generation
   - No keyword extraction output
   - **Impact**: Harder for users to see query understanding

5. **PDF Direct Processing**
   - No PyMuPDF integration
   - **Impact**: Limited to API-provided metadata/abstracts

6. **Report Export**
   - No PDF/Markdown export
   - No BibTeX export
   - **Impact**: Users must copy-paste results

---

### Low Priority (Enhancements)

7. **Query Rewriting**
   - No query expansion for better API coverage
   - **Impact**: Some relevant papers might be missed

8. **Thematic Grouping**
   - No explicit categorization of findings by theme
   - **Impact**: Less structured synthesis

9. **Parallel Retrieval**
   - Agents run sequentially
   - **Impact**: Slower execution

---

## Recommendations

### If the goal is **proposal validation** (academic paper):
1. **MUST implement** evaluation framework (high priority #1–#3)
2. Implement ground truth with 3–5 reviewers
3. Run baseline comparisons
4. Collect quantitative metrics
5. Optional: pilot user study

### If the goal is **production system** (user-facing):
1. Implement report export (high priority #6)
2. Add citation grounding checks
3. Implement conflict detection
4. Add query decomposition (scoping agent)
5. Optional: PDF processing

### If the goal is **research showcase**:
1. Focus on evaluation metrics
2. Add advanced visualization (filtering, community detection)
3. Implement thematic grouping in synthesis
4. Create demo comparison videos (baseline vs. system)

---

## Code Locations

| Feature | File | Lines |
|---------|------|-------|
| Agent definitions | streamlit_app.py | 632–1650 |
| Knowledge graph rendering | streamlit_app.py | 1814–1850 |
| Web agent APIs | streamlit_app.py | 1173–1300 |
| Reading extraction | streamlit_app.py | 1382–1450 |
| Critic/feedback loop | streamlit_app.py | 1534–1566 |
| UI main loop | streamlit_app.py | 1900–2300 |
| Vector DB module | streamlit_app.py | (integrated inline) |
| SQL DB queries | streamlit_app.py | 360–430 |

---

## Summary Table

| Component | Status | Completeness |
|-----------|--------|--------------|
| LangGraph Architecture | ✅ | 100% |
| 10 Agents | ✅ | 100% |
| Retrieval (Vector/SQL/Web) | ✅ | 100% |
| Knowledge Graph | ✅ | 90% (no filtering) |
| Synthesis & Planning | ✅ | 95% (no thematic grouping) |
| Validation Loop | ⚠️ | 70% (no conflict resolution) |
| **Evaluation Framework** | ❌ | **0%** |
| Query Intelligence | ⚠️ | 40% (no scoping) |
| PDF Processing | ❌ | 0% |
| Export Features | ⚠️ | 20% (JSON only) |
| **OVERALL** | ⚠️ | **~70%** |

---

## Next Steps

**To achieve ~95% proposal coverage**:
1. Implement evaluation framework (2–3 hours)
2. Add scoping agent (1 hour)
3. Add citation grounding check (1 hour)
4. Add conflict resolution (2 hours)
5. Add report export (1–2 hours)

**Estimated effort**: ~8 hours of additional implementation.

