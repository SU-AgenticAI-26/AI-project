# Evaluation Plan: Multi-Agent Research Assistant

This document is the authoritative reference for implementing automated benchmarking.
It describes what to measure, why each metric matters, which script implements it,
and how to interpret results. Claude Code should use this document alongside the
scripts in this directory to set up and extend the evaluation pipeline.

---

## System overview (relevant to eval)

The pipeline is a LangGraph state machine with 8 nodes in this order:

```
router -> vector_db -> sql_db -> web -> orchestrator -> knowledge_mapper -> critic -> summarizer
                                                                           |
                                                            (loop back to orchestrator if sparse, max 2x)
```

The shared state dict (`AgentState`) has these fields populated after a full run:

| Field | Type | Contents |
|---|---|---|
| `query` | str | Original user query |
| `active_agents` | list[str] | Channels activated by Router |
| `router_reasoning` | str | Router's one-sentence rationale |
| `vector_findings` | str | VectorDB agent synthesis |
| `sql_findings` | str | SQL agent synthesis |
| `web_findings` | str | Web/arXiv agent synthesis |
| `merged_context` | str | Orchestrator's merged output |
| `knowledge_map` | dict | `{nodes: [...], edges: [...]}` |
| `critique` | str | Critic's feedback (if loop triggered) |
| `loop_count` | int | Number of Critic enrichment loops (0-2) |
| `summary` | str | Final Summarizer output |
| `activity_log` | list[dict] | Per-agent log entries |

---

## Test queries

Four queries cover different retrieval difficulty levels. Use these consistently
across all eval runs so results are comparable.

```python
TEST_QUERIES = [
    {
        "id": "q1_rag",
        "query": "How does retrieval-augmented generation reduce hallucination in large language models?",
        "expected_channels": ["vector_db", "web"],
        "difficulty": "easy",  # well-covered ML topic
    },
    {
        "id": "q2_federated",
        "query": "What are the main approaches and challenges in federated learning for healthcare applications?",
        "expected_channels": ["vector_db", "sql_db", "web"],
        "difficulty": "medium",  # cross-disciplinary
    },
    {
        "id": "q3_agents_experiment",
        "query": "How are LLM agents being used to assist with scientific experiment design?",
        "expected_channels": ["web"],
        "difficulty": "hard",  # sparse, emerging topic
    },
    {
        "id": "q4_multiagent",
        "query": "What collaboration mechanisms are used in multi-agent LLM systems?",
        "expected_channels": ["vector_db", "sql_db", "web"],
        "difficulty": "medium",  # broad survey topic
    },
]
```

---

## Evaluation dimensions

### 1. Retrieval quality (RAGAS)

**What it measures:** Is the retrieved context relevant to the query? Does the
final summary stay grounded in what was retrieved?

**Metrics:**
- `Faithfulness` (0-1): Fraction of claims in the summary that can be inferred
  from the merged context. A score below 0.7 indicates hallucination.
- `ResponseRelevancy` (0-1): Does the summary directly address the query?
  Low scores indicate the pipeline retrieved off-topic material.
- `LLMContextPrecisionWithoutReference` (0-1): Is the retrieved context
  focused (relevant chunks ranked high)?

**Inputs needed:** `query`, `merged_context` (split into chunks), `summary`

**Script:** `eval_ragas.py`

**Target:** All three metrics >= 0.75 for the system to be considered
minimally viable. Compare against baseline (single LLM call, no retrieval).

---

### 2. Router correctness (DeepEval ToolCorrectnessMetric)

**What it measures:** Did the Router activate the right retrieval channels
for each query type? This is the only agent whose decision can be checked
deterministically against a predefined expected set.

**Metric:** `ToolCorrectnessMetric` (0-1): Fraction of expected channels
that were actually activated. Note: activating extra channels is not penalised
(the metric checks recall, not precision, by default).

**Inputs needed:** `active_agents` from state, `expected_channels` from test
query definition above.

**Script:** `eval_deepeval.py`

**Target:** >= 0.8 across all test queries.

---

### 3. Knowledge graph quality (custom + RAGAS ContextEntitiesRecall)

**What it measures:**
- Node count: does the graph meet the 12-20 node target from the KnowledgeMapper prompt?
- Entity recall: do the graph nodes cover the key concepts a domain expert would expect?
- Edge validity: are the stated relationships semantically accurate?

**Metrics:**
- `node_count` (int): direct read from `knowledge_map["nodes"]`
- `source_diversity` (float): fraction of nodes whose `source` field is not
  all the same value (measures whether multi-source retrieval actually contributed)
- `ContextEntitiesRecall` (RAGAS, 0-1): checks whether named entities from
  the summary appear in the merged context (proxy for entity grounding)

**Script:** `eval_graph.py`

**Target:** node_count in [8, 25]; source_diversity >= 0.3; entity recall >= 0.6.

---

### 4. Synthesis thematic coherence (DeepEval GEval)

**What it measures:** Is the summary a cross-paper thematic analysis, or
just a list of what each source said? This is the hardest quality dimension
to measure automatically but GEval with a clear rubric is tractable.

**Metric:** Custom `GEval` with four evaluation steps (see `eval_deepeval.py`).
Score 0-1. The LLM judge is GPT-4o-mini with chain-of-thought reasoning.

**Script:** `eval_deepeval.py`

**Target:** >= 0.65. Compare against baseline single-LLM output on same queries.

---

### 5. Critic loop effectiveness (automatic from state)

**What it measures:** Is the Critic Agent adding value? If it always hits the
2-loop cap without approving, the feedback is not actionable. If it never
triggers, the Knowledge Mapper is always producing sufficient graphs.

**Metrics (computed directly from state, no LLM calls needed):**
- `loop_count` per run
- `cap_hit_rate`: fraction of runs where `loop_count == 2` (Critic never approved)
- `mean_node_count_delta`: average increase in node count after enrichment loops
  (requires running with and without Critic, see ablation below)

**Script:** `eval_perf.py` (computed from `state["loop_count"]` and
`state["activity_log"]`)

**Target:** `cap_hit_rate < 0.3`; enrichment loops should add nodes on average.

---

### 6. Performance (latency and tokens)

**What it measures:** Per-agent wall time and approximate token consumption.
Identifies bottlenecks and computes cost-per-query for the full system vs.
each ablation configuration.

**Metrics:**
- `wall_s` per agent (from timing wrapper)
- `total_wall_s` (end-to-end)
- `prompt_tokens_approx` per agent (from tiktoken)
- `estimated_cost_usd` (tokens * GPT-4o-mini pricing)

**Script:** `eval_perf.py`

**GPT-4o-mini pricing (as of April 2026):** $0.15/1M input tokens,
$0.60/1M output tokens. Update these constants in `eval_perf.py` if pricing changes.

---

### 7. Ablation configurations

Run all 4 test queries under each of these configurations and report
all metrics above for each:

| Label | Description |
|---|---|
| `full` | Full system, Critic active, all 3 retrieval channels |
| `no_critic` | Critic bypassed (force `loop_count` cap to 0) |
| `vector_only` | Router forced to activate only `vector_db` |
| `web_only` | Router forced to activate only `web` |
| `baseline` | Single GPT-4o-mini call, no retrieval, no agents |

**Script:** `eval_ablation.py`

---

## Running the full eval suite

```bash
# Install dependencies
pip install ragas deepeval tiktoken openai

# Set API key
export OPENAI_API_KEY="sk-..."

# Run all evaluations (writes results to eval_results/)
python run_eval.py

# Or run individual components:
python eval_ragas.py        # Faithfulness, relevancy, context precision
python eval_deepeval.py     # Router correctness, thematic coherence
python eval_graph.py        # Knowledge graph quality
python eval_perf.py         # Latency and token profiling
python eval_ablation.py     # All configs x all queries
```

Results are written to `eval_results/` as JSON and a summary CSV.

---

## Interpreting results

A result is considered **passing** if:
- Faithfulness >= 0.75
- ResponseRelevancy >= 0.75
- ContextPrecision >= 0.70
- RouterCorrectness >= 0.80
- ThematicCoherence >= 0.65
- NodeCount in [8, 25]
- CapHitRate < 0.30

The system demonstrates value over the baseline if it beats the baseline
on Faithfulness and ThematicCoherence by >= 0.10 on average across all 4 queries.

---

## Files in this directory

```
eval_suite/
    EVAL_PLAN.md         <- this file (read first)
    run_eval.py          <- master runner, calls all scripts
    eval_ragas.py        <- RAGAS metrics (faithfulness, relevancy, precision)
    eval_deepeval.py     <- DeepEval metrics (router, coherence, task completion)
    eval_graph.py        <- Knowledge graph quality checks
    eval_perf.py         <- Latency and token profiling wrapper
    eval_ablation.py     <- Multi-configuration ablation runner
    eval_baseline.py     <- Single-LLM baseline comparison
    test_queries.py      <- Shared query definitions and helpers
    eval_results/        <- Output directory (created on first run)
```
