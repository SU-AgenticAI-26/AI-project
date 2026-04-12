# Multi-Agent Research System Overview

## Executive Summary

This is a sophisticated **9-agent reasoning and retrieval system** designed for academic research support. The system orchestrates multiple specialized AI agents to:
1. Understand research queries
2. Retrieve information from multiple sources
3. Structure and synthesize findings
4. Generate grounded research plans

---

## Agent Architecture

### Graph Flow

```
START
  ↓
[1. Scoping Agent] ← Decomposes query into sub-questions & keywords
  ↓
[2. Router Agent] ← Decides which retrieval sources to activate
  ↓
┌─────────────────────────────────────────────────────────┐
│ PARALLEL RETRIEVAL TIER (fan-out, then fan-in)         │
│ [✓ OPTIMIZED: All run simultaneously, not cascading]   │
│                                                         │
│   ↙ [3a. Vector DB Agent]    (2-3s)                   │
│  /   [3b. SQL DB Agent]       (1-2s)  parallel exec    │
│ \    [3c. Web Agent]          (2-4s)                   │
│   ↘ Total: max(3-4s) not sum(9-15s)                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
  ↓ (fan-in: all three complete before extraction)
[4. Reading Extraction Agent] ← Structures papers & findings
  ↓
[5. Orchestrator Agent] ← Merges & deduplicates sources
  ↓
[6. Knowledge Mapper Agent] ← Extracts knowledge graph
  ↓
[7. Critic Agent] ← Quality control (may loop back)
  │
  ├─→ needs_more=true → [2. Router] (LOOP: ✓ FIXED to trigger fresh retrieval)
  └─→ needs_more=false ↓
  ↓
[8. Summarizer Agent] ← Final answer with citations
  ↓ (including citation grounding validation)
  ↓
[9. Experiment Design Agent] ← Research plan generation
  ↓
[? Export Agent] ← [PLANNED: Markdown/BibTeX/JSON export]
  ↓
END
```

---

## Detailed Agent Descriptions

### 1. **Scoping Agent** 🔍
**Purpose:** Parse user query into actionable components  
**Input:** Raw user query  
**Output:**
- `sub_questions`: 3–5 focused research questions
- `keywords`: 4–8 search terms/themes
- `scoping_reasoning`: explanation of decomposition

**Example:**
- User: *"How do diffusion models compare to GANs in image generation?"*
- Output: 
  - Sub-questions: What are diffusion models? What are GANs? Performance metrics? Training efficiency? Why choose one over another?
  - Keywords: diffusion models, generative adversarial networks, image synthesis, training convergence

---

### 2. **Router Agent** 🔀
**Purpose:** Decide which retrieval channels to activate  
**Input:** User query  
**Output:**
- `active_agents`: list of ["vector_db", "sql_db", "web", ...] to activate
- `router_reasoning`: rationale for selection

**Decision Logic:**
- `vector_db`: Best for semantic document search (papers, articles in indexed DB)
- `sql_db`: Best for structured facts, topics, relationships
- `web`: Best for recent papers, external knowledge, conference proceedings

---

### 3. Retrieval Tier (Cascading)

#### **3a. Vector DB Agent** 🗂️
**Purpose:** Search indexed document database using semantic embeddings  
**Input:** Query, active_agents flag  
**Output:** `vector_findings` — synthesized research notes from top-k documents

**Process:**
1. LLM decides: Should we search? What query? How many results? Any filters?
2. Uses tool calling (for ChatOpenAI models) or direct search fallback
3. Retrieves semantically similar documents
4. Synthesizes findings into research notes

---

#### **3b. SQL DB Agent** 🗄️
**Purpose:** Query structured database for facts and relationships  
**Input:** Query, active_agents flag  
**Output:** `sql_findings` — extracted structured information

**Process:**
1. LLM decides: Is SQL search helpful? What to query? How many results?
2. Uses tool calling (ChatOpenAI) or direct SQL fallback
3. Returns structured results
4. LLM synthesizes into relevant facts

---

#### **3c. Web Agent** 🌐
**Purpose:** Live web search + conference paper search  
**Input:** Query, active_agents flag  
**Output:** `web_findings` — synthesis of web results and papers

**Features:**
- DuckDuckGo web search (title, snippet, URL)
- Conference paper search (when query mentions: NeurIPS, ICML, ICLR, ACL, EMNLP, etc.)
- Results indexed into vector DB for future searches
- Fallback for non-OpenAI models

**Conference Support:** NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, EACL

---

### 4. **Reading Extraction Agent** 📖
**Purpose:** Structure retrieval findings into per-paper records  
**Input:** Merged findings from all retrieval agents + original query  
**Output:** `extraction_findings` — structured records, one per source

**Extraction Fields (per paper/source):**
```markdown
---
**Title / Topic:** <name>
**Provenance:** <abstract-only | full-text | structured-db>
**Research Problem:** <one sentence>
**Methodology:** <one sentence>
**Key Findings:**
  - <bullet 1>
  - <bullet 2>
**Limitations:** <one sentence>
**Future Work:** <one sentence>
---
```

**Provenance Labels:**
- `structured-db`: From SQL/database findings
- `full-text`: From vector DB (full paragraphs available)
- `abstract-only`: From web API (title/abstract/year only)

---

### 5. **Orchestrator Agent** 🤝
**Purpose:** Merge findings from all sources into one coherent context  
**Input:** vector_findings, sql_findings, web_findings, extraction_findings  
**Output:** `merged_context` — deduplicated, labeled synthesis

**Process:**
1. Combines all findings
2. Deduplicates overlapping information
3. Resolves contradictions (preferring structured sources)
4. Labels each claim: [VectorDB] / [SQL] / [Web] / [Extraction]
5. Produces one coherent merged context

---

### 6. **Knowledge Mapper Agent** 🗺️
**Purpose:** Extract knowledge graph structure from merged context  
**Input:** merged_context, query  
**Output:** `knowledge_map` — JSON graph with nodes and edges

**Graph Structure:**
```json
{
  "nodes": [
    {
      "id": "unique_id",
      "label": "concept/entity name",
      "type": "concept | entity | fact | process",
      "source": "vector_db | sql_db | web | merged"
    }
  ],
  "edges": [
    {
      "source": "node_id_1",
      "target": "node_id_2",
      "relation": "description of relationship",
      "weight": 0.1  // confidence/strength
    }
  ]
}
```

**Graph Size:** 12–20 nodes, proportional edges

---

### 7. **Critic Agent** 🧐
**Purpose:** Quality control & loop back if insufficient information  
**Input:** knowledge_map structure  
**Output:**
- `_needs_more`: true/false (should we loop?)
- `critique`: feedback on gaps
- `loop_count`: iteration counter

**Exit Criteria (proceed to summarize if):**
- ✓ Knowledge map has ≥8 nodes
- ✓ Multiple source types represented (not just one source)

**Loop Back Criteria (if true):**
- Graph too small (<8 nodes)
- Key source diversity missing
- Loop count <2 (prevent infinite loops)

**Loop Back Target:** → Orchestrator (re-merge with more context)

---

### 8. **Summarizer Agent** ✍️
**Purpose:** Generate final answer grounded in retrieved sources  
**Input:** merged_context, knowledge_map, query  
**Output:**
- `summary`: final answer with source citations
- `citation_grounding`: validation of each citation
- `grounding_score`: 0.0–1.0 fraction of grounded claims

**Citation Grounding (BLOCK 2):**
1. Summarizer wraps key claims in double quotes: `"claim text"` (source)
2. System extracts citations from summary
3. Validates each citation against merged_context + extraction_findings
4. Returns grounding map: `{citation: {grounded: bool, source: "merged|extraction", evidence: text}}`
5. Calculates grounding_score = grounded_count / total_citations

**Citation Patterns Validated:**
- Quoted text: `"claim"`
- Parenthetical: `(claim)`
- Numeric references: `[1]`, `[2]`, etc.

---

### 9. **Experiment Design Agent** 📋
**Purpose:** Translate literature findings into actionable research plan  
**Input:** summary, extraction_findings, query, knowledge_map  
**Output:** `research_plan` — structured markdown plan

**Output Sections:**

1. **Research Landscape Overview**
   - What's well-established
   - Which methods dominate
   - Current state relative to research question

2. **Identified Research Gaps** (3–5 gaps, each grounded in literature)
   - Gap description
   - Grounding: which papers reveal this gap

3. **Proposed Hypotheses** (one per gap, falsifiable)
   - Links to corresponding gap
   - Grounded in evidence

4. **Recommended Methodologies**
   - Study design
   - Experimental protocol
   - Procedures
   - Evaluation method
   - References to validated methods in literature

5. **Datasets & Domains**
   - Public datasets (with names)
   - Data collection approach
   - Domain scope
   - Approximate scale

6. **Anticipated Challenges & Risks**
   - Technical/logistical/validity risks
   - Mitigation strategy per risk

7. **Short-term Next Steps** (0–3 months)
   - Numbered actionable steps

8. **Long-term Research Roadmap**
   - 3–6, 6–12 month milestones

9. **Evaluation & Success Criteria**
   - Metrics
   - Baselines
   - Publication targets

10. **Citation & Grounding**
    - All claims grounded to literature
    - Every gap grounded to specific papers

---

## State Management (AgentState Schema)

The system maintains a shared state dictionary throughout execution:

```python
class AgentState(TypedDict):
    # User input
    query: str
    
    # Scoping outputs
    sub_questions: list[str]
    keywords: list[str]
    scoping_reasoning: str
    
    # Router output
    active_agents: list[str]
    router_reasoning: str
    
    # Retrieval outputs
    vector_findings: str
    sql_findings: str
    web_findings: str
    
    # Synthesis outputs
    extraction_findings: str
    merged_context: str
    
    # Knowledge graph
    knowledge_map: dict  # {nodes: [...], edges: [...]}
    
    # Critic loop
    critique: str
    _needs_more: bool
    loop_count: int
    
    # Final outputs
    summary: str
    citation_grounding: dict
    grounding_score: float
    research_plan: str
    
    # Metadata
    messages: list[AIMessage]
    activity_log: list[dict]
    current_agent: str
```

---

## Data Flow & Integration Points

### Vector DB Integration
- **Module:** `VectorDBModule`
- **Methods Used:**
  - `vdb.search(query, k=5)`: Semantic search
  - `vdb.index(docs, source)`: Index new documents
- **Embedding Model:** OpenAI text-embedding-ada-002
- **Storage:** FAISS index at `collab_rag_data/vectorstore/`

### SQL DB Integration
- **Method:** `sql_search(query, k=8)` function
- **Purpose:** Structured topic/fact lookup
- **Returns:** Formatted results table

### Tool Calling (OpenAI Only)
- **Models:** ChatOpenAI with tool_choice="auto"
- **Tools:**
  - `search_vectordb`: LLM-driven vector search
  - `search_sqldb`: LLM-driven SQL queries
  - `search_conference_papers`: Conference paper lookup
- **Fallback:** Non-OpenAI models use direct search

---

## Configuration & Parameters

### Temperature (Creativity) by Agent
- `scoping`: 0.5 (moderate creativity for decomposition)
- `router`: 0.1 (deterministic routing)
- `vector_db`: 0.1 (precise retrieval synthesis)
- `sql_db`: 0.1 (precise structured search)
- `web`: 0.2 (moderate for web synthesis)
- `reading_extraction`: 0.1 (precise extraction)
- `orchestrator`: 0.2 (careful merging)
- `knowledge_mapper`: 0.1 (precise graph)
- `critic`: 0.0 (fully deterministic quality check)
- `summarizer`: 0.5 (balanced summary quality)
- `experiment_design`: 0.4 (creative planning)

### Limits & Constraints
- **Scoping:** Max 5 sub-questions, max 8 keywords
- **Knowledge mapper:** 12–20 nodes per graph
- **Citation validation:** Max 15 unique citations checked
- **Loop limit:** Critic can loop max 2 times before forcing summarization
- **Web search:** Max 5 results per DuckDuckGo query

---

## Key Features & Innovations

### 1. **Parallelism Optimization** ✓ COMPLETED
- Retrieval tier (vector_db, sql_db, web) now runs in parallel, not sequentially
- **Latency improvement:** ~9-15 seconds (sequential) → ~3-5 seconds (parallel)
- All three retrievers complete simultaneously, then fan-in to extraction
- LangGraph automatically handles fanin: extraction waits for all incoming edges
- See [ARCHITECTURE_IMPROVEMENTS.md](ARCHITECTURE_IMPROVEMENTS.md) for technical details

### 2. **Critic Loop Fix** ✓ COMPLETED
- Critic now loops back to **router** (not orchestrator) when needs_more=true
- Triggers **fresh retrieval pass** instead of re-merging same data
- Actually improves coverage (vs decorative loop in prior version)
- See [ARCHITECTURE_IMPROVEMENTS.md](ARCHITECTURE_IMPROVEMENTS.md) for details

### 3. **Multi-Source Convergence**
- Combines vector search (semantic), SQL (structured), and web (live) in one pipeline
- Orchestrator deduplicates and resolves contradictions
- Each claim labeled with source for traceability

### 4. **Citation Grounding (BLOCK 2)** ✓ IMPLEMENTED
- Summarizer produces grounded citations in structured formats
- Automatic validation of each citation against retrieved sources
- Grounding score = percentage of citations verified in source text
- Supports quoted text, parentheticals, and numeric references

### 5. **Reasoning Loop (Critic)**
- Critic reviews knowledge graph quality
- Loops back to router if insufficient coverage (now with fresh retrieval)
- Prevents hallucination by requiring multiple sources
- Terminates when diversity + size criteria met

### 6. **LLM Tool Calling**
- Enables LLMs to *decide* whether to search (not just search blindly)
- Supports dynamic query refinement
- Fallback to direct search for non-OpenAI models

### 7. **Structured Extraction**
- Per-paper records with provenance labels
- Standardized format (problem, methods, findings, limitations, future work)
- Enables downstream traceback to original sources

### 8. **Research Plan Generation**
- Translates literature into 9-section actionable research plan
- Every gap grounded to specific papers
- Includes hypotheses, methodologies, datasets, risks, and milestones

---

## Outstanding Gaps (Planned Enhancements)

See [ARCHITECTURE_IMPROVEMENTS.md](ARCHITECTURE_IMPROVEMENTS.md) for details. Summary:

| Gap | Issue | Priority | Status |
|-----|-------|----------|--------|
| 1 | Scoping keywords not injected into retrieval queries | HIGH | 🚨 TODO |
| 2 | No conflict detection between contradictory sources | HIGH | 🚨 TODO |
| 3 | Research plan exports (BibTeX, JSON) not implemented | MEDIUM | 🚨 TODO |
| 4 | Critic doesn't pass feedback context to router | LOW | 🚨 TODO |

---

## Example Execution Flow

### User Query
*"What are the recent advances in efficient neural networks and how can I apply them to mobile devices?"*

### Execution Trace

1. **Scoping Agent Output:**
   ```
   Sub-questions:
   - What are the main efficiency metrics for neural networks?
   - Which recent architectures achieve high performance with low parameters?
   - How do quantization and pruning techniques work?
   - What are deployment strategies for mobile devices?
   - How is mobile deployment performance measured?
   
   Keywords: efficient networks, MobileNets, quantization, pruning, parameter efficiency
   ```

2. **Router Decision:**
   ```
   Active agents: [vector_db, web]
   Reasoning: Vector DB has indexed papers on efficient architectures.
   Web search will find recent conference papers (NeurIPS 2024, ICLR 2024).
   SQL DB not needed (not looking for structured facts, more literature).
   ```

3. **Parallel Retrieval:**
   - Vector DB: Returns 8 papers on efficient architectures
   - Web: Returns recent NeurIPS/ICML papers on mobile efficiency

4. **Reading Extraction:**
   ```
   Extracts 10 papers with structured records:
   - MobileNetV3 (full-text from VectorDB)
   - EfficientNet (abstract-only from Web)
   - ...
   ```

5. **Orchestrator:**
   ```
   Merges 10 sources, resolves conflicting accuracy claims,
   labels each with [VectorDB] or [Web].
   ```

6. **Knowledge Mapper:**
   ```
   Creates graph:
   - Nodes: MobileNet, EfficientNet, Quantization, Pruning, ...
   - Edges: "MobileNet uses" → Depthwise Convolution
   - Graph: 15 nodes, 18 edges
   ```

7. **Critic:**
   ```
   Knowledge map looks sufficient.
   needs_more = false
   → Proceed to summarizer
   ```

8. **Summarizer:**
   ```
   Outputs final answer:
   "Recent advances in efficient networks include MobileNetV3,
   which achieved 75% ImageNet accuracy with 5.4M parameters
   [VectorDB source: Han et al. 2019]. Quantization techniques
   reduce model size by 4–8x [Web source: ICLR 2024]."
   
   Citation validation: 8/10 citations grounded (80%)
   ```

9. **Experiment Design Agent:**
   ```
   Produces research plan for deploying efficient networks:
   - Gap 1: Limited understanding of quantization + hardware interaction
   - Gap 2: Benchmark datasets incomplete for edge devices
   - H1: Propose hardware-aware quantization method
   - H2: Create mobile-specific benchmark suite
   - Methodologies: comparison study using 5 mobile devices
   - Datasets: ImageNet + custom 10k mobile photos
   - Risks: Hardware variation across phone models (mitigation: standardize)
   - Next steps: 1) Literature review, 2) Benchmarking setup, 3) Prototyping
   ```

---

## Error Handling & Robustness

### Graceful Failures

| Component | Failure Mode | Handling |
|-----------|--------------|----------|
| JSON parsing (LLM output) | Invalid JSON from agent | Fallback to defaults (empty list, false, etc.) |
| Vector DB empty | No indexed docs | Skip vector_db, continue with other sources |
| Web search timeout | DuckDuckGo unreachable | Return empty web_findings, continue |
| Conference paper tool | Tool call fails | Fallback to direct text search |
| SQL DB unreachable | DB connection error | Skip sql_db, continue with available sources |
| LLM timeout | Model hangs | Timeout + retry with simpler prompt |
| Citation not in source | Grounding fails | Mark as ungrounded, decrease grounding_score |

### Loop Prevention
- Critic loop count capped at 2
- Prevents infinite loops if quality criteria never met
- Falls back to summarization on max iterations

---

## Performance & Scalability

### Token Usage (Typical Query)
- Scoping: ~500 tokens
- Router: ~300 tokens
- Retrieval synthesis (3 agents): ~1500 tokens
- Reading extraction: ~1000 tokens
- Orchestrator: ~800 tokens
- Knowledge mapper: ~600 tokens
- Critic: ~300 tokens
- Summarizer: ~1200 tokens
- Experiment design: ~1500 tokens
- **Total per query:** ~8,000–10,000 tokens (≈ $0.08–$0.10)

### Latency (Typical Query)
- Scoping: ~1–2 sec
- Retrieval (parallel): ~3–5 sec
- Synthesis: ~8–12 sec
- Final agents: ~5–8 sec
- **Total end-to-end:** ~20–30 seconds

### Scalability
- **Indexed documents:** Currently ~100s in vector DB (FAISS)
- **Web search results:** Limited by DuckDuckGo (≈5–20 results)
- **Concurrent queries:** LangGraph can handle serial pipeline; parallel agent execution within one query
- **Critique loops:** Max 2 prevents runaway re-computation

---

## Future Enhancements

1. **Adaptive routing:** Router learns which sources are productive for patterns
2. **Parallel critic loop:** Multiple refinement attempts in parallel
3. **Interactive feedback:** User can steer during execution
4. **Multi-language support:** Extend to non-English queries
5. **Custom knowledge bases:** User-uploaded document integration
6. **Streaming output:** Live agent progress to UI
7. **Caching:** Reuse scoping/routing/extraction for similar queries
8. **Hybrid search:** Combine vector search with BM25 for better coverage

---

## References

- **Framework:** LangGraph (StateGraph, compiled into runnable pipeline)
- **LLM Provider:** OpenAI (ChatOpenAI with tool calling)
- **Vector Store:** FAISS with OpenAI embeddings (ada-002)
- **Web Search:** DuckDuckGo (free, no API key)
- **Conference Papers:** Custom tool integration
- **UI:** Streamlit with real-time activity logging

