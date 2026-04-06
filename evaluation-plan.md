This is worth thinking through carefully because the system has multiple distinct outputs — the retrieved evidence pool, the knowledge graph, and the final synthesis — and each needs different evaluation approaches. Quality and performance also interact: the Critic's enrichment loop improves quality at a cost that needs to be quantified.

---

## Quality Evaluation

### Retrieval layer

The most rigorous approach is **pooled relevance assessment**, the same method used in TREC benchmarks. For each test query, run the system plus two or three alternative retrieval configurations (single-source only, no Router, different top-k values). Pool all retrieved papers across all configurations, remove duplicates, and have two reviewers independently judge each paper for relevance on a three-point scale: highly relevant, marginally relevant, irrelevant. This pool becomes the reference set. You then compute Precision@k (what fraction of the top-k papers are relevant), Recall@k (what fraction of all known relevant papers were found), and Mean Average Precision (MAP), which rewards systems that rank relevant papers higher.

The key metric to add on top of standard IR metrics is **source complementarity**: for each query, what fraction of the finally-merged relevant papers came exclusively from one channel (i.e., would have been missed if that channel were absent)? This directly tests whether the multi-source architecture earns its complexity. If VectorDB and arXiv return almost identical papers, the Web Agent's contribution is marginal and the Router's activation decisions are not adding value.

**Hallucination rate in retrieval metadata** can be measured automatically: for every paper in the merged pool, fetch its true metadata from CrossRef or Semantic Scholar by DOI and compare title, authors, year, and venue against what the agent recorded. Any mismatch is a metadata error. This catches the case where the system retrieves a real paper but corrupts its provenance in the merge step.

### Knowledge graph

The knowledge graph is harder to evaluate objectively because "correct" structure is partially subjective. Two tractable approaches:

**Entity recall against a reference ontology.** For well-studied test domains (RAG, federated learning for healthcare), domain ontologies and survey papers provide a canonical list of key concepts. Compute the fraction of these reference concepts that appear as nodes in the generated graph. This is automatable once you have the reference list.

**Edge precision via entailment checking.** Sample 20 edges per test query. For each edge (A —[relation]→ B), automatically construct the claim as a sentence ("RAG uses dense passage retrieval") and run it through an NLI model (e.g. a DeBERTa model fine-tuned on NLI) against the merged context text. Classify each edge as entailed, neutral, or contradicted by the source text. This gives an objective edge accuracy score without manual review. Edges that are neutral or contradicted indicate the Knowledge Mapper is hallucinating relationships not supported by the retrieved evidence.

**Graph structural consistency** can be checked automatically: does the graph contain contradictory edges (A contradicts B and A extends B simultaneously)? Are node labels consistent with their stated type (a node typed "process" should describe an action, not an entity)? These are rule-based checks that require no human judgment.

### Synthesis quality

This is the hardest output to evaluate objectively, but several approaches reduce the human burden:

**Citation verifiability** is the most automatable quality signal. For every bracketed citation in the synthesis, check whether the cited claim can be located in the abstract or retrieved text of the cited paper using sentence-level dense retrieval (embed the claim, retrieve from the paper's chunked text, check cosine similarity against a threshold). A claim that scores below threshold against its cited paper is either hallucinated or misattributed. This is fully automatic and produces a per-query citation accuracy score.

**Sub-question coverage** requires the Scoping Agent to be implemented, but once it is: check whether the synthesis contains a paragraph addressing each of the sub-questions the Scoping Agent generated. An LLM-as-judge call (asking a separate model "does this paragraph address this question?") is a reasonable automated proxy, and the prompt can be templated and run at scale.

**Lexical diversity relative to source text** detects near-copy synthesis. Compute n-gram overlap (ROUGE-1 and ROUGE-2) between the synthesis and the concatenated retrieved abstracts. High overlap indicates the synthesizer is paraphrasing rather than integrating. A good synthesis should have *low* ROUGE scores against individual sources but *high* thematic coherence — this is the inverse of what ROUGE was designed to reward, but the metric is still informative as a red flag.

**Factual consistency** can be checked with a dedicated NLI pass: decompose the synthesis into individual atomic claims (using an LLM), then for each claim, retrieve the most similar passage from the evidence pool and run NLI to check whether the passage entails, is neutral toward, or contradicts the claim. The fraction of claims that are entailed gives an automated factual grounding score.

### Critic loop effectiveness

The Critic's contribution can be isolated by running every query twice: once with the Critic active and once with `loop_count` hard-capped at 0 (no enrichment). Compare the Knowledge Mapper's output under both conditions on node count, source diversity, and entity recall. This gives a direct, objective measure of what the Critic loop adds. If the enrichment pass consistently adds fewer than 2 nodes with no improvement in entity recall, the Critic is not providing actionable feedback.

### Baseline comparison

The single-LLM baseline (same query, no retrieval, no agents) should be evaluated on the same metrics wherever possible. For citation verifiability and factual grounding, the baseline will have no retrieved text to check against — so you compute those metrics against whatever the baseline cites, fetching those papers independently. For knowledge graph metrics, you can prompt the baseline to output a JSON graph in the same format and run the same structural checks. This creates a clean apples-to-apples comparison on every automated metric.

---

## Performance Evaluation

### Latency decomposition

Instrument each agent with a timing wrapper that records wall-clock time from entry to exit and stores it in the LangGraph state. This gives you a per-agent latency profile for every run: Router, VectorDB, SQL, Web, Orchestrator, KnowledgeMapper, Critic (per iteration), Summarizer. Aggregate across multiple runs to get mean and standard deviation per agent and identify the bottleneck. The Web Agent (live arXiv fetches) and the Orchestrator (largest LLM context window) are the most likely bottlenecks.

Report latency at three granularities: **time to first token of final answer** (relevant for interactive use), **time to knowledge graph available** (the point at which the visual output is ready), and **total pipeline wall time** (end-to-end).

### Token consumption

Every LLM call should log prompt tokens and completion tokens separately. The Orchestrator is likely the highest-cost agent because it receives the full concatenated output of three retrieval agents as context. Compute cost-per-query as `sum(prompt_tokens * input_price + completion_tokens * output_price)` across all agents using the GPT-4o-mini pricing at time of evaluation. Report this alongside quality metrics so you can compute a **quality-per-dollar curve** across configurations.

Token consumption also tells you something structurally: if the Orchestrator's prompt consistently approaches the model's context limit, that's a scalability problem that quality metrics alone won't surface.

### Critic loop overhead

Since the Critic can trigger up to two enrichment passes, measure the marginal cost of each pass: additional wall time, additional tokens, and quality delta (entity recall improvement, new nodes added). Plot quality improvement against cumulative cost for loop iterations 0, 1, and 2. If the second iteration adds less than 5% additional entity recall at significant token cost, the cap should be lowered to one iteration for production use.

### Ablation configurations

Run the same four test queries under these configurations and report latency and token cost for each:

- Full system (all agents, Critic active)
- No Critic (single pass only)
- Single-source only (VectorDB only, no SQL or Web)
- Web only (no local index)
- Baseline (single LLM call)

This produces a table mapping configuration to quality score and cost, which is the most useful output for deciding which configuration to deploy. It also lets you compute the **efficiency frontier**: the configuration that achieves the best quality per dollar.

### Scalability

Test with three query complexity levels: a narrow single-concept query, a moderate multi-aspect query, and a broad survey-style query. Measure whether token consumption and latency scale linearly or super-linearly with query complexity. Super-linear scaling in the Orchestrator would indicate the merge prompt is growing faster than the evidence pool, which suggests a chunking or truncation strategy is needed.

---

The key principle running through all of this is that each metric should be tied to a specific agent or stage rather than evaluated only at the system level. A low citation accuracy score is not actionable unless you know whether the problem is in retrieval (wrong papers), the Orchestrator (misattribution in the merge step), or the Summarizer (hallucination at synthesis time). The agent-level instrumentation is what makes the evaluation diagnostically useful rather than just a report card.

Two mature, pip-installable frameworks cover the main evaluation needs of this system. RAGAS provides broad coverage for RAG pipelines, including retrieval quality, faithfulness to retrieved context, and generation quality without requiring fully labeled ground-truth datasets. DeepEval complements that by focusing on agent evaluation, separating reasoning behavior such as planning and decision-making from action behavior such as tool selection and argument correctness. Together, they provide a practical foundation for evaluating both the pipeline outputs and the behavior of the individual agents.

Here is a concrete plan for what to install, what to run, and what each script tests.

---

## What to install

```bash
pip install ragas deepeval tiktoken openai
```

Both frameworks use GPT-4o-mini as the judge by default, so your existing `OPENAI_API_KEY` is sufficient. No additional infrastructure is needed.

---

## RAGAS: retrieval and synthesis quality

RAGAS evaluates three core dimensions: Faithfulness (whether all claims in the response can be inferred from the retrieved context), Answer Relevance (whether the response directly addresses the question), and Context Precision (whether the retrieved context is focused and contains only relevant information).

The integration with the existing system is straightforward. After a pipeline run, the `full_state` object already contains everything RAGAS needs: `query` (the user input), `merged_context` (the retrieved context), and `summary` (the final answer). A minimal evaluation script looks like this:

```python
# eval_ragas.py
import asyncio
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference
from ragas.llms import llm_factory

TEST_QUERIES = [
    "How does retrieval-augmented generation reduce hallucination in LLMs?",
    "What are the main approaches to federated learning for healthcare applications?",
    "How do multi-agent systems coordinate in LLM-based research assistants?",
    "What are the limitations of current systematic review automation tools?",
]

async def run_ragas_eval(pipeline_results: list[dict]) -> dict:
    """
    pipeline_results: list of dicts with keys:
        query, retrieved_contexts (list of strings), response
    Each entry is one completed pipeline run.
    """
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    samples = [
        SingleTurnSample(
            user_input=r["query"],
            retrieved_contexts=r["retrieved_contexts"],  # chunked text from merged_context
            response=r["response"],
        )
        for r in pipeline_results
    ]

    dataset = EvaluationDataset(samples=samples)
    result = await evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=llm),
            ResponseRelevancy(llm=llm),
            LLMContextPrecisionWithoutReference(llm=llm),
        ],
    )
    return result
```

The `retrieved_contexts` field should be a list of the individual chunk strings that reached the Orchestrator — you can extract these from `full_state["vector_findings"]`, `full_state["sql_findings"]`, and `full_state["web_findings"]` split into sentences or paragraphs. Running this across the four test queries gives you three scores between 0 and 1 for each run and an aggregate mean. Scores above 0.8 on faithfulness, context recall, and context precision typically indicate strong performance, though this varies by domain and use case.

RAGAS also has a `ContextEntitiesRecall` metric that checks whether named entities from a reference answer appear in the retrieved context. This maps directly onto the knowledge graph evaluation: you can use the node labels from `full_state["knowledge_map"]["nodes"]` as a proxy for the expected entity set, then check how many appear in the merged context.

---

## DeepEval: agent-specific metrics

DeepEval's agent metrics analyze the entire execution trace, capturing every reasoning step, tool call, and intermediate decision — which gives you the granularity to pinpoint exactly where things go wrong. The three most relevant metrics for this system are:

**ToolCorrectnessMetric** — verifies that the Router activated the right retrieval channels. You define the expected tool set per query type (e.g., a narrow ML query should activate `vector_db` and `web`; a broad survey query should activate all three) and check whether the Router's JSON decision matched.

**TaskCompletionMetric** — evaluates end-to-end whether the pipeline accomplished what the query asked for, judged against the final summary.

**GEval with custom criteria** — GEval uses LLM-as-a-judge with chain-of-thought reasoning to evaluate outputs on any custom criteria, and is the most versatile metric for use-case-specific evaluation. For this system, you can define a `ThematicCoherence` GEval criterion: "Does the summary group findings by research theme rather than summarising papers one by one? Does it identify contradictions across the literature?"

```python
# eval_deepeval.py
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric, GEval, TaskCompletionMetric
from deepeval.test_case import LLMTestCaseParams

# Router correctness: did the right channels activate?
router_test = LLMTestCase(
    input="How does RAG reduce hallucination in LLMs?",
    actual_output=full_state["summary"],
    tools_called=[
        ToolCall(name=agent) for agent in full_state["active_agents"]
    ],
    expected_tools=[
        ToolCall(name="vector_db"),
        ToolCall(name="web"),
    ],
)

# Thematic coherence via GEval
coherence_metric = GEval(
    name="ThematicCoherence",
    evaluation_steps=[
        "Check whether the summary organises findings by research theme rather than paper by paper.",
        "Check whether the summary identifies at least one point of disagreement or contradiction across sources.",
        "Penalise summaries that simply list what each paper says without cross-paper analysis.",
        "Check whether each thematic claim is attributed to a source channel (Vector DB / SQL / Web).",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
)

coherence_test = LLMTestCase(
    input=query,
    actual_output=full_state["summary"],
    retrieval_context=[full_state["merged_context"]],
)

evaluate(
    test_cases=[router_test, coherence_test],
    metrics=[ToolCorrectnessMetric(), coherence_metric],
)
```

---

## Performance profiling: no extra packages needed

Agent-level latency and token consumption can be captured with a lightweight wrapper around each agent function. The most practical approach given the existing architecture is to wrap the LangGraph node lambdas at graph construction time:

```python
# perf_wrapper.py
import time, tiktoken
from functools import wraps

_enc = tiktoken.encoding_for_model("gpt-4o-mini")
perf_log: list[dict] = []

def timed_agent(name: str, fn):
    def wrapper(state):
        t0 = time.perf_counter()
        result = fn(state)
        elapsed = time.perf_counter() - t0
        # Count tokens in this agent's LLM input (merged context going in)
        ctx_text = state.get("merged_context", "") + state.get("query", "")
        prompt_tokens = len(_enc.encode(ctx_text))
        perf_log.append({
            "agent": name,
            "wall_s": round(elapsed, 3),
            "prompt_tokens_approx": prompt_tokens,
            "loop_count": state.get("loop_count", 0),
        })
        return result
    return wrapper

# In build_graph(), wrap each node:
g.add_node("orchestrator", timed_agent("orchestrator",
    lambda s: orchestrator_agent(s, lm_o)))
```

After a full run, `perf_log` gives you a per-agent breakdown of wall time and token load that you can print as a table or export to CSV. Running this across the four test queries and across the ablation configurations (full system, no Critic, single-source only, baseline) produces the efficiency data needed for the paper.

---

## What each tool covers

| What you want to measure | Tool | Key metric |
|---|---|---|
| Does the summary stay grounded in retrieved text? | RAGAS `Faithfulness` | 0–1, no ground truth needed |
| Is the retrieved context actually relevant to the query? | RAGAS `ContextPrecision` | 0–1, no ground truth needed |
| Did the Router activate the right agents? | DeepEval `ToolCorrectnessMetric` | 0–1, requires expected tool list |
| Is the summary thematically integrated rather than paper-by-paper? | DeepEval `GEval` (custom) | 0–1, LLM judge |
| Did the pipeline complete the stated research task? | DeepEval `TaskCompletionMetric` | 0–1, trace-based |
| Per-agent wall time and bottleneck identification | `time` + `perf_log` | seconds |
| Token consumption per agent and per run | `tiktoken` + `perf_log` | tokens, estimated cost |

The RAGAS metrics require no labelled data and can be run on every query automatically. The DeepEval tool correctness metric requires you to specify the expected tool activation per query type, which is a one-time setup. The GEval metric runs without ground truth. None of these require human reviewers for the automated passes — the human evaluation described in the proposal is additive on top, not a substitute.
