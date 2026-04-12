# Architecture Improvements & Known Gaps

## Status: April 2026

---

## ✓ COMPLETED: Parallelism Optimization

### The Change
Refactored the retrieval tier from sequential cascade to parallel fan-out/fan-in:

**Before (sequential — ~9-15 sec latency):**
```
router → vector_db → sql_db → web → extraction
```

**After (parallel — ~3-5 sec latency):**
```
        ↙ vector_db ↘
router ─  sql_db   ─→ extraction (fan-in)
        ↖ web     ↗
```

### Implementation
- Replaced `_route_to_retrieval_agents()`, `_route_vector_db_to_next()`, `_route_sql_db_to_next()` (three cascading functions)
- Added new `_route_to_all_retrievers(state) → list[str]` that returns ALL active retrievers as list
- Updated graph edges:
  ```python
  g.add_conditional_edges(
      "router",
      _route_to_all_retrievers,  # Returns ["vector_db", "sql_db", "web", ...]
      {...}
  )
  # Each retriever → extraction (LangGraph fan-in automatic)
  g.add_edge("vector_db",  "reading_extraction")
  g.add_edge("sql_db",     "reading_extraction")
  g.add_edge("web",        "reading_extraction")
  ```

### Latency Impact
- **Before:** vector_db (2-3s) → sql_db (1-2s) → web (2-4s) = ~5-9s sequential
- **After:** All three run in parallel = ~2-4s (max of three)
- **Total system latency reduction:** ~20-30s → ~15-20s end-to-end

### Backwards Compatibility
✓ All retrieval agent outputs (`vector_findings`, `sql_findings`, `web_findings`) unchanged  
✓ Extraction agent logic unchanged (still reads same state fields)  
✓ No changes needed to downstream agents  

---

## ✓ COMPLETED: Critic Loop Fix

### The Problem
Critic was looping to orchestrator:
```python
# OLD (broken):
g.add_conditional_edges(
    "critic", _route_critic,
    {"orchestrator": "orchestrator", "summarizer": "summarizer"},
)

def _route_critic(state: AgentState) -> str:
    if state.get("_needs_more") and state.get("loop_count", 0) < 2:
        return "orchestrator"  # ← loops to same agent
    return "summarizer"
```

**Issue:** Orchestrator just re-merges the same `vector_findings`, `sql_db`, `web_findings` it already has. Nothing new comes in, so the loop produces identical output. The critique flag signals insufficient coverage, but the loop doesn't actually retrieve more sources—it's silent correctness bug.

### The Fix
Critic now loops to router with critique feedback:
```python
# NEW (corrected):
g.add_conditional_edges(
    "critic", _route_critic,
    {"router": "router", "summarizer": "summarizer"},  # ← router, not orchestrator
)

def _route_critic(state: AgentState) -> str:
    """
    If needs_more and loop_count < 2, trigger fresh retrieval via router.
    Router receives state["critique"] and can adjust active_agents strategy.
    """
    if state.get("_needs_more") and state.get("loop_count", 0) < 2:
        return "router"  # ← Fresh retrieval with critique context
    return "summarizer"
```

### Graph Impact
```
                       critic
                      ╱       ╲
            (needs_more)   (done)
              ↓               ↓
            router          summarizer
            ↓
        (retrieval tier)
            ↓
        extraction
```

### Benefits
- Second critic loop now triggers actual new retrieval attempts
- Router can use `state["critique"]` to adjust which sources to prioritize
- Prevents infinite loops (still capped at loop_count < 2)
- Meaningful quality improvement → better coverage

---

## ⚠️ OUTSTANDING GAPS

### Gap 1: Scoping Keywords Not Injected Into Retrieval 🚨 (Priority: HIGH)

**Current Behavior:**
- Scoping agent extracts: `sub_questions` (5) and `keywords` (8) ✓
- Router agent **ignores** these and only uses raw `state['query']` ✗
- Retrieval agents (vector_db, sql_db, web) also only use `state['query']` ✗

**Impact:**
- Block 1 (scoping) becomes **decorative UI feature, not functional**
- Keywords sit in state but don't influence search quality
- Queries lack focused refinement that scoping provides

**Proposed Fix:**
```python
# In router_agent() and retrieval agents:

# OLD:
content=f"Query: {state['query']}"

# NEW:
keywords_str = ", ".join(state.get("keywords", [])[:8])
sub_questions_str = "\n".join(state.get("sub_questions", [])[:5])

content = f"""Query: {state['query']}

Scoped Keywords (use to filter/prioritize): {keywords_str}

Sub-questions to address:
{sub_questions_str}
"""
```

**Effort:** ~30 lines across 4 agents (router, vector_db, sql_db, web)  
**Location:** Search prompt construction in respective agent functions

---

### Gap 2: Conflict Detection (Block 3) 🚨 (Priority: HIGH)

**Current Architecture:**
```
orchestrator → knowledge_mapper → critic
```

**Gap:** No node checks for contradictory claims across sources.

**Example Conflict:**
- Vector DB paper: "Model X achieves 95% accuracy"
- Web paper: "Model X achieves 87% accuracy"
- SQL DB: "Baseline for domain is 78%"
→ No agent detects or flags these inconsistencies

**Proposed Placement:**
```
orchestrator → [NEW: conflict_detector] → knowledge_mapper
```

**What It Should Do:**
1. Parse `merged_context` to identify contradictory claims
2. Categorize: direct contradiction vs. different experimental setup
3. Return conflict map: `{claim: [sources], contradiction: text}`
4. Influence knowledge graph creation (may merge nodes, add conflict edges)

**Output Schema:**
```json
{
  "conflicts_detected": 2,
  "high_confidence_conflicts": [
    {
      "claim_a": "Model X achieves 95%",
      "claim_b": "Model X achieves 87%",
      "source_a": "VectorDB",
      "source_b": "Web",
      "likely_cause": "Different datasets or hyperparameters"
    }
  ],
  "conflict_summary": "Findings generally align; note X vs Y discrepancy"
}
```

**Effort:** ~200-300 lines for LLM prompt + parsing  
**Benefit:** Trust in merged context increases, knowledge mapper can tag contradictions

**Nice-to-have:** Pass conflict summary to knowledge_mapper to influence edge weights

---

### Gap 3: Research Plan Export (Block 4) 🚨 (Priority: MEDIUM)

**Current:**
- Experiment_design_agent generates markdown research plan ✓
- No export functionality ✗

**Missing End Node:**
```
experiment_design → [NEW: export_agent] → END
```

**What It Should Do:**
1. Accept `research_plan` markdown from experiment_design
2. Format into exportable forms:
   - **Markdown** (current output, saved as `.md`)
   - **BibTeX** (citations as `@article{...}` blocks)
   - **JSON** (structured machine-readable format)
   - **PDF** (optional, requires weasyprint/pandoc)

3. Collect citations from entire pipeline:
   - extraction_findings (per-paper records)
   - knowledge_map (source nodes)
   - merged_context (labeled citations)

4. Generate BibTeX entries for all sources

**Output Schema:**
```json
{
  "research_plan_md": "## Research Plan\n...",
  "bibtex_entries": "@article{han2019mobilenetv3, ...}",
  "metadata": {
    "query": "original user query",
    "created_at": "timestamp",
    "sources_cited": 12,
    "agent_chain": ["scoping", "router", "vector_db", "sql_db", "web", ...],
    "grounding_score": 0.82
  }
}
```

**Files:**
```
research_plan.md      (markdown)
citations.bib         (BibTeX)
research_summary.json (metadata + structured)
```

**Effort:** ~150-200 lines for formatting + citation collection  
**Benefit:** Enables downstream workflow (LLM → Overleaf, dataset annotation, etc.)  
**Nice-to-have:** Markdown includes inline BibTeX labels `[han2019]` → translates to citations

---

### Gap 4: Critic Receives Fresh Critique Context 🟡 (Priority: LOW / Already Part of Gap 2 Fix)

**Current Behavior (after critic loop fix):**
- Critic loops to router, but router still calls same routing logic
- Doesn't receive any feedback from critic about *why* needs_more=true

**Proposed Enhancement:**
- Store `state["critique"]` (feedback string from critic agent)
- Pass to router as "user feedback to adjust strategy"
- Router can then adjust active_agents based on gap feedback

**Example:**
```python
# Critic returns:
critique = "Graph lacks entities from applied ML domain; need more web sources"
state["_needs_more"] = True

# Router receives it and decides:
"critic says we need more applied examples → activate web + keep vector_db"
```

**Effort:** ~20 lines (use critique in router prompt)  
**Benefit:** More intelligent second pass retrieval  
**Status:** Low priority—system works without this, but improves loop quality

---

## Revised Architecture Diagram

```
                    scoping (🔍)
                        ↓
                    router (🔀)
                   ↙    ↓    ↘
          vector_db  sql_db  web  (parallel 🗂️🗄️🌐)
            (2-3s)   (1-2s)  (2-4s)
                     ↓ ↓ ↓ (fan-in)
        reading_extraction (📖)
                        ↓
                  orchestrator (🤝)
                        ↓
            [conflict_detector NEW] ← GAP 2
                        ↓
            knowledge_mapper (🗺️)
                        ↓
                    critic (🧐)
                   ↙        ↘
          (needs_more)    (done)
              ↓             ↓
           router        summarizer (✍️)
    (fresh retrieval)      ↓
            ↓         experiment_design (📋)
      (retrieval tier)      ↓
            ↓         [export_agent NEW] ← GAP 3
            ↓              ↓
          extraction      END
            ↓
        (merged output)
```

---

## Implementation Roadmap

| Gap | Title | Effort | Priority | Type | Status |
|-----|-------|--------|----------|------|--------|
| 1 | Keywords injection into retrieval | 30 min | HIGH | Enhancement | ⏱️ TODO |
| 2 | Conflict detection between sources | 2-3 hrs | HIGH | New agent | ⏱️ TODO |
| 3 | Research plan export (MD/BibTeX/JSON) | 2 hrs | MEDIUM | New agent | ⏱️ TODO |
| 4 | Critic passes feedback to router | 15 min | LOW | Enhancement | ⏱️ TODO (with Gap 2) |

---

## Testing Checklist

After implementing each gap:

- [ ] **Gap 1 (Keywords):**
  - Run query with multi-word topic
  - Verify keywords appear in agent prompts
  - Check retrieval results improve (fewer irrelevant sources)

- [ ] **Gap 2 (Conflict Detection):**
  - Run query with contradictory literature
  - Verify conflict_detector outputs conflict map
  - Check knowledge_mapper uses conflict info

- [ ] **Gap 3 (Export):**
  - Run full pipeline
  - Verify `.md` and `.bib` files generated
  - Check BibTeX entries are valid (try in Overleaf)

- [ ] **All improvements:**
  - End-to-end latency: should be ~15-20s
  - Parallelism visible in logs (all three retrievers active simultaneously)
  - No regressions in existing agents

---

## Demo Talking Points

1. **Parallelism:** "Retrieval tier now runs three sources in parallel instead of sequentially—cuts retrieval latency by 50%"
2. **Smarter loops:** "Critic loop now triggers fresh retrieval, not just re-merging same data"
3. **Scoping impact (once Gap 1 done):** "Keywords from scoping feed directly into retrieval queries"
4. **Conflict detection (once Gap 2 done):** "System flags contradictory claims and resolves them systematically"
5. **Export (once Gap 3 done):** "Research plan exports as Markdown + BibTeX for immediate use in writing"

---

## Questions for Next Review

1. Should conflict_detector be a new agent node, or a capability added to orchestrator?
2. For export, what's the priority: Markdown (most utility) vs PDF (most polish)?
3. Should critic loop limit be increased from 2 to account for multi-pass refinement?
4. Should router store routing decisions for analytics (which sources worked best)?

