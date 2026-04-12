# Technical Reference: Conference Paper Search Integration

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Streamlit UI (User Query)                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              streamlit_app.py: build_graphs() / stream()            │
│            - Router Agent decides which agents to activate          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                  "web" in active_agents?
                    yes ↓
┌─────────────────────────────────────────────────────────────────────┐
│              streamlit_app.py: web_agent(state, model, vdb)         │
│                                                                      │
│  1. Check query for conference keywords (case-insensitive)          │
│  2. If mention detected & model is ChatOpenAI:                      │
│     - Prepare message with tool definition                          │
│     - Call: model.invoke(messages, tools=[SEARCH_PAPERS_TOOL])      │
│  3. If LLM calls tool:                                              │
│     - handle_conference_paper_tool_call(tool_args)                  │
│  4. Combine with existing sources:                                  │
│     - OpenAlex API   (5 results)                                    │
│     - Crossref API   (5 results)                                    │
│     - Semantic Scholar (5 results)                                  │
│     - arXiv API      (5 results)                                    │
│     - Conference Papers (up to 10 results)                          │
│  5. All indexed into VectorDB: vdb.add_text(...)                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│         conference_paper_search.py: search_conference_papers()      │
│                                                                      │
│  Entry point for LLM tool calls                                     │
│  Dispatches to appropriate backend                                  │
└──────────────────────┬──────────────────────────┬──────────────────┘
                       │                          │
    OpenReview Venues  │                          │  ACL Anthology
    (requires auth)    │                          │  (no auth)
                       ▼                          ▼
┌──────────────────────────────┐  ┌────────────────────────────────┐
│  search_openreview()         │  │  search_acl_anthology()        │
│                              │  │                                │
│  • get_or_clients()          │  │  • Load Anthology              │
│  • fetch_api2()              │  │  • Query papers by venue/year  │
│  • fetch_api1()              │  │  • Pattern match keywords      │
│  • Pattern matching regex    │  │  • Format results              │
│  • Fetch NeurIPS/ICML/ICLR  │  │  • Handle pagination           │
└──────────────────────────────┘  └────────────────────────────────┘
                │                              │
                └──────────────┬───────────────┘
                               │
                               ▼
                        JSON results:
                    {
                      "papers": [
                        {
                          "title": "...",
                          "authors": "...",
                          "abstract": "...",
                          "conference": "NeurIPS",
                          "year": "2024",
                          "url": "https://...",
                          "pdf": "https://..."
                        },
                        ...
                      ]
                    }
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              VectorDB: vdb.add_text(metadata={...})                 │
│              All papers indexed for downstream extraction           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│           reading_extraction_agent() → orchestrator_agent()         │
│           knowledge_mapper_agent() → summarizer_agent()             │
└─────────────────────────────────────────────────────────────────────┘
```

## Code Flow

### 1. Imports in `streamlit_app.py`

```python
# Line ~73: Conditional import with error handling
try:
    from conference_paper_search import (
        SEARCH_PAPERS_TOOL,
        handle_conference_paper_tool_call,
        OPENREVIEW_CONFERENCES,
        ACL_CONFERENCES,
    )
    HAS_CONFERENCE_SEARCH = True
except ImportError:
    HAS_CONFERENCE_SEARCH = False
```

### 2. Web Agent Detection Logic

```python
# streamlit_app.py: web_agent() function starting line ~768

def web_agent(state: AgentState, model: BaseChatModel, vdb: VectorDBModule) -> dict:
    # ... validation checks ...
    
    query = state["query"]
    
    # NEW: Conference paper search with tool calling
    if HAS_CONFERENCE_SEARCH:
        try:
            # Check if query mentions any conference keywords
            query_lower = query.lower()
            conference_keywords = {"neurips", "icml", "iclr", "acl", "emnlp", ...}
            mentions_conference = any(kw in query_lower for kw in conference_keywords)
            
            if mentions_conference and isinstance(model, ChatOpenAI):
                # LLM-based tool invocation
                messages = [
                    SystemMessage(content="You are a research assistant..."),
                    HumanMessage(content=f"Search for papers related to: {query}")
                ]
                
                response = model.invoke(
                    messages,
                    tools=[SEARCH_PAPERS_TOOL],
                    tool_choice="auto"
                )
                
                # Process tool calls if any
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    for tool_call in response.tool_calls:
                        if tool_call.function.name == "search_conference_papers":
                            tool_args = json.loads(tool_call.function.arguments)
                            tool_result = handle_conference_paper_tool_call(tool_args)
                            # ... process results and add to VectorDB ...
        except Exception as e:
            pass  # Graceful degradation
    
    # Continue with existing APIs (OpenAlex, Crossref, Semantic Scholar, arXiv)
    # ...
```

### 3. Tool Definition

```python
# conference_paper_search.py: SEARCH_PAPERS_TOOL

SEARCH_PAPERS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_conference_papers",
        "description": "Search for academic papers by keywords across top ML and NLP conferences...",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords":    {"type": "array", "items": {"type": "string"}},
                "years":       {"type": "array", "items": {"type": "integer"}},
                "conferences": {"type": "array", "items": {"type": "string", "enum": [...]}},
                "search_in":   {"type": "array", "items": {"type": "string", "enum": ["title", "abstract"]}},
                "max_results": {"type": "integer"}
            },
            "required": ["keywords", "years", "conferences"],
            "additionalProperties": False
        }
    }
}
```

### 4. Tool Invocation Path

```python
# conference_paper_search.py

def search_conference_papers(
    keywords: list[str],
    years: list[int],
    conferences: list[str],
    search_in: list[str] | None = None,
    max_results: int = 10,
) -> dict:
    """Entry point when LLM calls the tool"""
    
    or_confs = [c for c in conferences if c in OPENREVIEW_CONFERENCES]
    acl_confs = [c for c in conferences if c.upper() in ACL_CONFERENCES]
    
    all_results = []
    
    if or_confs:
        all_results.extend(search_openreview(or_confs, years, keywords, search_in))
    
    if acl_confs:
        acl_venues = [c.lower() for c in acl_confs]
        all_results.extend(search_acl_anthology(acl_venues, years, keywords, search_in))
    
    return {
        "papers": all_results[:max_results],
        "returned": len(all_results),
        "truncated": len(all_results) > max_results,
    }
```

### 5. OpenReview Backend

```python
# conference_paper_search.py: search_openreview()

def search_openreview(conferences, years, keywords, search_in):
    results = []
    pattern = make_pattern(keywords)
    
    v1, v2 = get_or_clients()  # Login to both API v1 and v2
    
    for conf in conferences:
        for year in years:
            info = OR_VENUES.get(conf, {}).get(year)  # Look up venue ID
            
            papers = (
                fetch_api2(v2, info["id"])
                if info["api"] == 2
                else fetch_api1(v1, info["id"])
            )
            
            # Pattern match on title + abstract
            for p in papers:
                texts = [get_field(p.content, f) for f in search_in]
                if any(matches(t, pattern) for t in texts):
                    results.append({
                        "conference": conf,
                        "year": str(year),
                        "title": get_field(p.content, "title"),
                        "abstract": get_field(p.content, "abstract"),
                        "authors": get_field(p.content, "authors"),
                        "pdf": f"https://openreview.net/pdf?id={p.id}",
                        "url": f"https://openreview.net/forum?id={p.id}",
                    })
    
    return results
```

### 6. ACL Anthology Backend

```python
# conference_paper_search.py: search_acl_anthology()

def search_acl_anthology(venues, years, keywords, search_in):
    anthology = Anthology.from_repo()  # Load local cache or download
    pattern = make_pattern(keywords)
    results = []
    
    for paper in anthology.papers():
        full_id = paper.full_id  # "2024.acl-long.42"
        parts = full_id.split(".")
        
        paper_year = parts[0]  # "2024"
        venue_slug = parts[1].split("-")[0].lower()  # "acl"
        
        if paper_year in year_set and venue_slug in venue_set:
            # Pattern match on title + abstract
            if any(matches(text, pattern) for text in texts):
                results.append({
                    "conference": venue_slug.upper(),
                    "year": paper_year,
                    "title": str(paper.title),
                    "abstract": str(paper.abstract),
                    "authors": ", ".join(...),
                    "url": f"https://aclanthology.org/{full_id}",
                    "pdf": f"https://aclanthology.org/{full_id}.pdf",
                })
    
    return results
```

## Configuration Hierarchy

```
Environment Variables (highest priority)
    ├─ OPENREVIEW_USERNAME
    ├─ OPENREVIEW_PASSWORD
    └─ SEMANTIC_SCHOLAR_API_KEY

↓

.env file (local)
    ├─ OPENREVIEW_USERNAME=...
    └─ OPENREVIEW_PASSWORD=...

↓

conference_paper_search.py defaults (lowest)
    ├─ OR_USERNAME = ""  (disable if empty)
    ├─ OR_PASSWORD = ""  (disable if empty)
    └─ Graceful degradation if not set
```

## Error Handling Strategy

```python
# Layered error handling ensures robustness

1. Import level
   ├─ try: from conference_paper_search import ...
   └─ except ImportError: HAS_CONFERENCE_SEARCH = False
     → Streamlit app still runs without conference search

2. Credential level
   ├─ if not OR_USERNAME or not OR_PASSWORD:
   └─ return None, None → Skip OpenReview
     → ACL Anthology search still works

3. API call level
   ├─ try: get_or_clients()
   └─ except: print("Login failed") → continue
     → Fall back to other sources

4. Parsing level
   ├─ try: json.loads(tool_result)
   └─ except: return {} with empty papers
     → No crash, downstream agents handle empty results

5. VectorDB integration
   ├─ try: vdb.add_text(...)
   └─ except: log error → continue
     → Extraction still works with what's available
```

## Data Flow: Query to Answer

```
User Query: "Find NeurIPS 2024 papers on diffusion models"
    │
    ├─→ Router.invoke() → {"active_agents": ["web", ...]}
    │
    ├─→ web_agent.invoke()
    │   │
    │   ├─→ Keyword detection: "NeurIPS" ✓, "2024" ✓, "diffusion" ✓
    │   │
    │   ├─→ ChatOpenAI.invoke(messages, tools=[SEARCH_PAPERS_TOOL])
    │   │   └─→ LLM: "I should call search_conference_papers"
    │   │       Arguments: {
    │   │         "keywords": ["diffusion models"],
    │   │         "years": [2024],
    │   │         "conferences": ["NeurIPS"]
    │   │       }
    │   │
    │   ├─→ handle_conference_paper_tool_call(args)
    │   │   │
    │   │   └─→ search_conference_papers(...)
    │   │       │
    │   │       ├─→ search_openreview(["NeurIPS"], [2024], ...)
    │   │       │   ├─→ get_or_clients() (login)
    │   │       │   ├─→ fetch_api2(v2, "NeurIPS.cc/2024/Conference")
    │   │       │   ├─→ Pattern match papers
    │   │       │   └─→ [8 papers found]
    │   │       │
    │   │       └─→ return {"papers": [...], "returned": 8}
    │   │
    │   ├─→ For each paper:
    │   │   └─→ vdb.add_text(paper_info, metadata={})
    │   │
    │   ├─→ Combine with OpenAlex/Crossref/Semantic/arXiv
    │   │
    │   └─→ LLM summarize all results
    │
    ├─→ reading_extraction_agent.invoke()
    │   └─→ Extract structured data from papers
    │
    ├─→ orchestrator_agent.invoke()
    │   └─→ Merge + deduplicate + weight evidence
    │
    ├─→ knowledge_mapper_agent.invoke()
    │   └─→ Build knowledge graph
    │
    └─→ summarizer_agent.invoke()
        └─→ Generate final answer with citations
```

## Performance Characteristics

```
Component              | Time (sec) | Notes
─────────────────────────────────────────────────────
Conference Detection   | <0.1       | String matching
OpenReview Auth        | 2-5        | API handshake only once
OpenReview Search      | 5-10       | Per venue per year
ACL Anthology Download | 30-60      | First run only; cached after
ACL Anthology Search   | <1         | Local cache query
VectorDB Indexing      | 0.1-0.5    | Per paper
LLM Summarization      | 5-10       | gpt-4o-mini inference
Total web_agent        | ~30-60     | Typical range
─────────────────────────────────────────────────────
Full Pipeline          | ~60-120    | Including extraction + synthesis
```

## Debugging

Enable printing for diagnostics:

```python
# In streamlit_app.py or direct script:
import logging
logging.basicConfig(level=logging.DEBUG)

# In conference_paper_search.py:
print(f"[DEBUG] Searching {conf} {year}...")
print(f"[DEBUG] Pattern: {pattern.pattern}")
print(f"[DEBUG] Matched: {len(results)} papers")
```

## Extension Points

### Add a new venue

```python
# conference_paper_search.py

OR_VENUES["MyConf"] = {
    2024: {"id": "MyConf.cc/2024/Conference", "api": 2},
}

OPENREVIEW_CONFERENCES.add("MyConf")
```

### Custom ranking

```python
def rank_papers(papers, query):
    # Implement custom scoring
    for p in papers:
        p["score"] = calculate_relevance(p, query)
    return sorted(papers, key=lambda x: x["score"], reverse=True)
```

### Custom field extraction

```python
def extract_custom_field(paper, field):
    # Override get_field() for your venue
    pass
```

---

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `conference_paper_search.py` | ~400 | Core search logic |
| `streamlit_app.py` | ~50 (modified) | web_agent integration |
| `requirements.txt` | 2 (added) | Dependencies |
| `CONFERENCE_SEARCH_SETUP.md` | ~300 | User documentation |
| `verify_conference_search.py` | ~150 | Testing/diagnostics |

---

**Last Updated**: April 2026
**Status**: Production Ready
