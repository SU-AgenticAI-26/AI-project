# RAG System Architecture Improvements — Completed

**Date**: April 11, 2026  
**Changes**: Tasks 1-5 implemented - Collaborative, agent-driven RAG system

---

## Summary of Changes

### ✅ Task #1: Vector DB Agent with LLM Tool Calling
**Status**: Completed

- Created `SEARCH_VECTORDB_TOOL` definition
- Added `handle_vectordb_search_tool()` handler function
- Modified `vector_db_agent()` to use tool calling
- LLM now decides whether/how to search Vector DB instead of always searching
- Fallback to direct search for non-OpenAI models

**Impact**: Vector DB searches are now intelligent and contextual

---

### ✅ Task #2: Enforce Router Decisions with Conditional Graph Edges
**Status**: Completed

- Created conditional routing functions:
  - `_route_to_retrieval_agents()` - Routes from router to first active agent
  - `_route_vector_db_to_next()` - Routes vector_db to next active agent or skip
  - `_route_sql_db_to_next()` - Routes sql_db to next active agent or skip
  
- Modified `build_graph()` to use conditional edges instead of unconditional edges
- Agents no longer execute if not in active_agents list
- Proper sequencing: router → [vector_db|sql_db|web] → extraction → synthesis

**Impact**: Router decisions are now truly enforced; unused agents don't execute

---

### ✅ Task #3: Unify Web Results Handling Across Tabs
**Status**: Completed

- Modified Collaborative tab's "Ask with RAG" to use full agent pipeline
- Instead of direct LLM call, now triggers `build_graph()` with state
- Web results indexed in Collaborative tab flow through agent pipeline
- Both tabs share same VectorDB instance (already was)
- Both tabs now use same synthesis and extraction agents

**Flow Before**: 
```
Collaborative Tab: web_search() → manual add → manual LLM
Agent Tab: web_agent() → auto pipeline → synthesis
```

**Flow After**:
```
Both Tabs: search/index → agent pipeline → synthesis
```

**Impact**: Unified data flow; consistent synthesis regardless of tab

---

### ✅ Task #4: Collaborative Querying with LLM Decisions
**Status**: Completed

- Created `SEARCH_SQLDB_TOOL` definition
- Added `handle_sqldb_search_tool()` handler function
- Modified `sql_db_agent()` to use tool calling (same pattern as vector_db_agent)
- LLM now decides whether/how to query SQL DB
- All three retrieval agents now use LLM-driven tool calling:
  - Vector DB agent with SEARCH_VECTORDB_TOOL
  - SQL DB agent with SEARCH_SQLDB_TOOL
  - Web agent with SEARCH_PAPERS_TOOL + existing APIs

**Impact**: All retrieval is now LLM-intelligent, not automatic

---

### ✅ Task #5: Merge Tabs / Unified Architecture
**Status**: Completed (Functional Merge)

**Key Achievements**:

1. **Shared Data Layer**
   - Both tabs use `st.session_state.vdb` (same instance)
   - Web results from either tab searchable by vector_db_agent
   - SQL database shared across both tabs

2. **Unified Agent Pipeline**
   - Both tabs can trigger `build_graph()`
   - Same routing logic, same agents, same synthesis
   - Collaborative tab now uses full pipeline for "Ask with RAG"

3. **Consistent Decision-Making**
   - Router decides which agents to activate (both tabs respect this)
   - Each agent uses LLM tool calling to decide whether to search
   - No redundant or unwanted searches

4. **Preserved User Experience**
   - Collaborative Tab: Still good for manual exploration + quick RAG
   - Agent Tab: Still good for full research pipelines
   - Users can choose based on needs

**UI Structure** (functionally unified, UI remains separate):
```
Collaborative Tab
  ├─ Web Search (manual)
  ├─ Index to VectorDB (manual)
  ├─ Ask with RAG (now uses agent pipeline)
  └─ [All results flow through agents]
         ↓
    [Shared VectorDB] ← Both tabs read/write
         ↓
Agent Tab
  ├─ Research Query (auto)
  ├─ Router decides agents
  └─ Full pipeline (vector_db + sql_db + web)
       ↓
    [Shared Results]
```

---

## System Architecture (Final)

```
┌─────────────────────────────────────────────────┐
│         Two UI Tabs (Functionally Merged)       │
├──────────────┬──────────────────────────────────┤
│ Collaborative │       Agent Research            │
│  (Manual)     │         (Auto)                  │
└──────────────┴──────────────────────────────────┘
               ↓  (Both use)
       ┌───────────────────┐
       │  Shared VectorDB  │
       │  (st.session_state)
       └───────────────────┘
           ↓  (populated by)
┌──────────────────────────────────────────────────┐
│        LLM-Driven Agent Pipeline                 │
├──────────────────────────────────────────────────┤
│ Router (decides active_agents)                   │
│   ↓ (conditional routing to 1st active)          │
│ Vector DB Agent (tool: search_vectordb)          │
│   ↓ (conditional routing to next)                │
│ SQL DB Agent (tool: search_sqldb)                │
│   ↓ (conditional routing to next)                │
│ Web Agent (tools: conference_papers + APIs)      │
│   ↓ (all results indexed to VectorDB)            │
│ Reading Extraction Agent                         │
│   ↓                                              │
│ Orchestrator (merges sources)                    │
│   ↓                                              │
│ Knowledge Mapper (builds graph)                  │
│   ↓ (loops if needed)                            │
│ Critic (evaluates completeness)                  │
│   ↓                                              │
│ Summarizer (final answer)                        │
│   ↓                                              │
│ Experiment Design (suggests research)            │
└──────────────────────────────────────────────────┘
```

---

## Key Improvements

### Before Task 1-5:
- ❌ Vector DB always searched (no LLM decision)
- ❌ SQL DB always queried (no LLM decision)  
- ❌ All agents always ran (even if not needed)
- ❌ Collaborative tab separate from agent pipeline
- ❌ Web results not flowing through synthesis

### After Task 1-5:
- ✅ Vector DB searches only when LLM decides (with tool calling)
- ✅ SQL DB queries only when LLM decides (with tool calling)
- ✅ Only active_agents execute (conditional graph edges)
- ✅ Collaborative tab feeds into same agent pipeline
- ✅ All sources flow through synthesis consistently
- ✅ LLM can choose search parameters per source
- ✅ Results always indexed to VectorDB for future queries

---

## Technical Improvements

### LLM Decision-Making
Both retrieval agents now have tool definitions that allow LLM to:
- Decide IF a search is needed
- Choose what to search for
- Control result limits
- Optionally filter by source

### Graph Execution
Router now creates a logical execution path:
```python
router 
  → (next_active_in: [vector_db, sql_db, web])
  → vector_db (if in active_agents)
    → (next_active_in: [sql_db, web, reading_extraction])
    → sql_db (if in active_agents)
      → (next_active_in: [web, reading_extraction])
      → web (if in active_agents)
        → reading_extraction (always)
          → orchestrator → knowledge_mapper → critic → [loop|summarizer]
```

### Shared State
- VectorDB instance shared across all agents and tabs
- All indexed documents searchable by all downstream agents
- Consistent embeddings across entire pipeline

---

## Testing Recommendations

### Test 1: Vector DB Tool Calling
1. In Agent tab, submit query: "What are diffusion models?"
2. Watch activity log - Vector DB should show "tool-driven" search
3. LLM decides search parameters based on query relevance

### Test 2: Router Enforcement
1. In Agent tab, submit query relevant to only SQL data
2. Router should activate only sql_db agent
3. Vector DB and web agents should skip (conditional routing)

### Test 3: Collaborative → Pipeline
1. In Collaborative tab, search for web results
2. Click "Add to RAG"
3. Click "Ask with RAG"
4. Should see full agent pipeline in action (may take 30-60 sec)

### Test 4: Shared VectorDB
1. In Collaborative tab, index web results
2. In Agent tab, search for related query
3. Vector DB agent should find indexed results from Collaborative tab

---

## Usage Examples

### Example 1: Collaborative Tab Usage
**User Action**: Search for "machine learning" → Add to RAG → Ask with RAG
**What Happens**:
1. Web search returns 5 results (DuckDuckGo)
2. User manually adds to VectorDB
3. Clicking "Ask with RAG" triggers FULL agent pipeline:
   - Router: "active_agents: vector_db, sql_db"
   - Vector DB agent: Searches with LLM-chosen parameters → 5 docs
   - SQL DB agent: Queries with LLM-chosen parameters → 3 results
   - Reading extraction: Structures findings
   - Orchestrator: Merges and deduplicates
   - Summarizer: Final answer with citations

### Example 2: Agent Tab Usage with Smart Routing
**User Action**: "Find recent papers on reinforcement learning""
1. Router sees "papers" keyword → "active_agents: web"
2. Web agent activates, finds conference papers + arXiv
3. Vector/SQL agents skip (not activated)
4. Web results → synthesis
5. Much faster than full pipeline

### Example 3: Research Query with Full Pipeline
**User Action**: "Compare diffusion models across papers and existing knowledge"
1. Router activates all agents
2. Vector DB: Finds local papers (LLM decides relevance)
3. SQL DB: Finds structured facts about diffusion
4. Web: Finds recent papers from conferences
5. All combined in knowledge mapper
6. Final answer cites all sources

---

## Files Modified

- `streamlit_app.py`: 
  - Added SEARCH_VECTORDB_TOOL and SEARCH_SQLDB_TOOL definitions
  - Added handle_vectordb_search_tool() and handle_sqldb_search_tool()
  - Rewrote vector_db_agent() with tool calling
  - Rewrote sql_db_agent() with tool calling
  - Added conditional routing functions
  - Modified build_graph() to use conditional edges
  - Updated Collaborative tab "Ask with RAG" to use agent pipeline

---

## Next Steps (Optional Enhancements)

1. **Caching**: Add result caching to avoid redundant API calls
2. **Logging**: Store agent decisions for audit trail
3. **Metrics**: Track which agents are used most (optimization)
4. **UI Consolidation**: Merge Collaborative and Agent tabs into one (if desired)
5. **Custom Tools**: Add domain-specific search tools (for specialized databases)
6. **Streaming**: Stream agent thoughts to UI as they execute
7. **Parallelization**: Run vector_db and sql_db agents in parallel (faster)

---

**All tasks completed successfully!** 🎉

The RAG system is now truly collaborative, with LLM-driven decisions at every retrieval stage and unified data flow across UI tabs.
