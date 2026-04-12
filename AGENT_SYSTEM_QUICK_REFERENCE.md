# Agent System: Quick Reference Guide

## Agent Inventory (9 total)

| # | Agent | Icon | Input | Output | Temperature | Key Role |
|---|-------|------|-------|--------|-------------|----------|
| 1 | Scoping | 🔍 | query | sub_questions, keywords | 0.5 | Parse user intent |
| 2 | Router | 🔀 | query | active_agents list | 0.1 | Select sources |
| 3a | Vector DB | 🗂️ | query | vector_findings | 0.1 | Semantic search |
| 3b | SQL DB | 🗄️ | query | sql_findings | 0.1 | Structured lookup |
| 3c | Web | 🌐 | query | web_findings | 0.2 | Live search + papers |
| 4 | Reading Extraction | 📖 | all findings | extraction_findings | 0.1 | Structure papers |
| 5 | Orchestrator | 🤝 | merged inputs | merged_context | 0.2 | Deduplicate & merge |
| 6 | Knowledge Mapper | 🗺️ | merged_context | knowledge_map JSON | 0.1 | Extract graph |
| 7 | Critic | 🧐 | knowledge_map | needs_more flag | 0.0 | Quality control |
| 8 | Summarizer | ✍️ | merged_context, graph | summary + grounding | 0.5 | Generate answer |
| 9 | Experiment Design | 📋 | summary, extraction | research_plan | 0.4 | Plan research |

## Agent Routing Decisions

### Router Agent Decisions
```
Input: User query
↓
Decide activation:
├─ vector_db? → Yes if: literature search, academic papers, document analysis
├─ sql_db? → Yes if: structured facts, topics, relationships, factual lookup
├─ web? → Yes if: recent papers, external knowledge, conference proceedings
└─ (none) → Yes if: query too vague or no search needed
```

### Retrieval Cascade
```
Vector DB Agent activated?
├─ Yes: Execute search → outputs to cascade router
│  └─ Next: SQL DB or Web or skip to extraction
└─ No: Skip to cascade 2

SQL DB Agent activated?  
├─ Yes: Execute query → outputs to cascade router
│  └─ Next: Web or skip to extraction
└─ No: Skip to cascade 3

Web Agent activated?
├─ Yes: Execute search → goes to extraction
└─ No: Skip to extraction
```

### Critic Loop Decision
```
Knowledge map quality?
├─ ≥8 nodes + diverse sources? → needs_more = False → Summarize
└─ <8 nodes OR single source? → needs_more = True
   └─ loop_count < 2? → LOOP to Router for fresh retrieval
   └─ loop_count ≥ 2? → Force summarization (even if incomplete)
```

## State Variables by Agent

### Written by Each Agent
```
Agent 1 (Scoping):  sub_questions, keywords, scoping_reasoning
Agent 2 (Router):   active_agents, router_reasoning
Agent 3a (VDB):     vector_findings
Agent 3b (SQL):     sql_findings
Agent 3c (Web):     web_findings
Agent 4 (Extract):  extraction_findings
Agent 5 (Orch):     merged_context
Agent 6 (KMapper):  knowledge_map
Agent 7 (Critic):   critique, _needs_more, loop_count
Agent 8 (Summ):     summary, citation_grounding, grounding_score
Agent 9 (ExpDes):   research_plan
```

### Read by Each Agent
```
Agent 1: query (only new input)
Agent 2: query, sub_questions, keywords
Agent 3a-c: query, active_agents, [other findings]
Agent 4: query, vector_findings, sql_findings, web_findings
Agent 5: query, extraction_findings, vector_findings, sql_findings, web_findings
Agent 6: query, merged_context
Agent 7: query, knowledge_map
Agent 8: query, merged_context, knowledge_map, extraction_findings
Agent 9: query, summary, knowledge_map, extraction_findings, merged_context
```

## Common Patterns

### Pattern 1: Multi-Source Convergence
```
Vector DB finds paper A
Web finds paper B (same topic, different source)
Orchestrator deduplicates + labels both
Knowledge Mapper creates single node with [VectorDB, Web] sources
→ Increased confidence through triangulation
```

### Pattern 2: Citation Grounding
```
Summarizer: "Model X achieves 95% accuracy" [source]
Validation: grep "95%" in merged_context?
├─ Found → grounded = true
├─ Not found → grounded = false
└─ Similar claim found → grounded = true (semantic match)
grounding_score = grounded_count / total_claims
```

### Pattern 3: Loop Back Decision
```
Critic checks knowledge map:
├─ nodes count < 8?
├─ sources = ["vector_db"] only?
├─ edges = 0?
└─ Any 2 of above true?
   → needs_more = true
   → Increment loop_count
   → Retry orchestrator with modified prompts
```

### Pattern 4: Cascading Retrieval
```
Vector DB finds docs
  ↓
Router decides: should SQL search too?
  ↓
SQL DB finds structured facts
  ↓
Router decides: should Web search too?
  ↓
Web finds recent papers
  ↓
All results → Reading Extraction → Orchestrator
```

## Configuration Reference

### LLM Model Selection
```python
# Supported models:
- ChatOpenAI (gpt-4, gpt-3.5-turbo)
  └─ Tool calling: YES
  └─ Conference search: YES
  
- Other LLMs (Claude, Llama, etc.)
  └─ Tool calling: NO
  └─ Fallback: Direct search
  └─ Conference search: NO
```

### Temperature Settings (Creativity vs Precision)
```
High (0.4–0.5):  Scoping, Summarizer, Experiment Design
                 (need diverse/creative output)

Medium (0.2):    Orchestrator, Web Agent
                 (moderate balance)

Low (0.0–0.1):   Router, Vector DB, SQL DB, Extraction, 
                 Knowledge Mapper, Critic
                 (need deterministic/precise output)
```

### Limits & Constraints
```
sub_questions:     Max 5
keywords:          Max 8
knowledge_nodes:   12–20 (target)
citation_checks:   Max 15
web_results:       Max 5 per query
sql_results:        8 rows
loop_iterations:   Max 2 (Critic)
token_budget:      ~10k per query
timeout_web:       15 sec
timeout_llm:       30–60 sec
```

## Error Modes & Recovery

### Scoping Fails
```
Symptom: sub_questions = [original query], keywords = query.split()
Recovery: Continues with defaults, Router makes routing decision
Impact: Slightly less focused search, but system continues
```

### Vector DB Empty
```
Symptom: vector_findings = "(no documents indexed)"
Decision: Router skips vector_db, tries SQL/Web
Recovery: Continues with remaining sources
Impact: May miss relevant papers, but completes
```

### JSON Parse Fails
```
Symptom: LLM returns non-JSON text
Recovery: Try lstrip/rstrip, then fallback to defaults
Examples: 
  - Knowledge Mapper fails → {"nodes": [], "edges": []}
  - Router fails → active_agents = ["vector_db", "sql_db"]
Impact: Reduced quality, but prevents hard failure
```

### Citation Grounding Fails
```
Symptom: quoted_text "claim" not found in merged_context
Recovery: Mark as grounded=false, continue
Impact: Lower grounding_score, but doesn't block summarizer
```

### Loop Never Converges
```
Symptom: Critic always needs_more = true
Recovery: Terminate on loop_count ≥ 2, force summarization
Impact: Lower quality summary, but prevents infinite loop
```

## Performance Metrics

### Token Usage (Typical 5-source Query)
```
Agent          Tokens  Cost
─────────────────────────────
Scoping        500     0.5c
Router         300     0.3c
Vector DB      600     0.6c
SQL DB         400     0.4c
Web            600     0.6c
Reading Extr   1000    1.0c
Orchestrator   800     0.8c
Knowledge Map  600     0.6c
Critic         300     0.3c
Summarizer     1200    1.2c
Experiment     1500    1.5c
─────────────────────────────
TOTAL          ~8000   ~8.0c
```

### Latency Profile
```
Stage                          Latency
─────────────────────────────────────
Scoping                        1–2 sec
Router                         1 sec
Vector DB                      2–3 sec (parallel)
SQL DB                         1–2 sec (parallel)
Web                            2–4 sec (parallel)
Reading Extraction             2–3 sec
Orchestrator                   1–2 sec
Knowledge Mapper               1–2 sec
Critic                         1 sec
Summarizer                     2–3 sec
Experiment Design              2–3 sec
─────────────────────────────────────
TOTAL (serial path)            ~20–30 sec
```

## How to Read Output

### Activity Log Entry Format
```json
{
  "agent": "vector_db",           // which agent
  "icon": "🗂️",                   // visual identifier
  "title": "Vector DB agent",     // human-readable title
  "detail": "Retrieved 8 chunks", // what happened
  "ts": "2024-01-15 14:25:30"    // timestamp
}
```

### Reading Extraction Format
```markdown
---
**Title / Topic:** <name>
**Provenance:** full-text (indicates data quality/completeness)
**Research Problem:** One sentence describing what the paper addresses
**Methodology:** One sentence describing how they approached it
**Key Findings:**
  - Bullet 1: Main finding
  - Bullet 2: Supporting finding
**Limitations:** One sentence on what wasn't covered
**Future Work:** One sentence on suggested next steps
---
```

### Knowledge Map Format
```json
{
  "nodes": [
    {"id": "n1", "label": "Concept", "type": "concept", "source": "web"},
    {"id": "n2", "label": "Person", "type": "entity", "source": "vector_db"}
  ],
  "edges": [
    {"source": "n1", "target": "n2", "relation": "developed", "weight": 0.8}
  ]
}
```

### Citation Grounding Format
```json
{
  "\"claim text\"": {
    "grounded": true,
    "source": "merged",
    "evidence": "relevant text excerpt from source"
  },
  "[1]": {
    "grounded": false,
    "source": "extraction",
    "evidence": "no matching text found"
  }
}
```

## Debugging Checklist

- [ ] Is the query clear and research-focused?
- [ ] Are vector DB and SQL DB indexed (check `collab_rag_data/`)?
- [ ] Is Internet available (for web search)?
- [ ] Are OpenAI API keys configured?
- [ ] Is loop_count increasing? (indicates repeated Critic loops)
- [ ] Is grounding_score < 50%? (indicates poor citation quality)
- [ ] Are all 9 agents logging activity?
- [ ] Does knowledge_map have 0 nodes? (indicates mapping failed)
- [ ] Is extraction_findings = "NO_PAPERS_EXTRACTED"? (retrieval may have failed)

