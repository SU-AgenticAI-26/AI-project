"""
eval_ragas.py — RAGAS-based evaluation of retrieval and synthesis quality.

Metrics:
  - Faithfulness:          Are all claims in the summary grounded in the retrieved context?
  - ResponseRelevancy:     Does the summary directly address the query?
  - ContextPrecision:      Is the retrieved context focused on the query (no noise)?
  - ContextEntitiesRecall: Do the KG node labels appear in the merged context?

Usage:
    # OpenAI (default)
    python eval_ragas.py

    # Local llama.cpp pipeline, judge also local
    python eval_ragas.py --provider local --model mistral --base-url http://localhost:8080/v1

    # Local pipeline, OpenAI judge (recommended for highest-quality RAGAS scores)
    python eval_ragas.py --provider local --model mistral \\
                         --judge-provider openai --judge-model gpt-4o-mini

    # Subset of queries
    python eval_ragas.py --query-ids q1_rag q2_federated

Requires:
    pip install ragas
    # Plus provider packages; see streamlit_app.py install notes.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

from test_queries import TEST_QUERIES, run_pipeline, split_context_into_chunks
from eval_provider import EvalConfig, add_provider_args, cfg_from_args

try:
    from ragas import evaluate, RunConfig
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithoutReference,
    )
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}\nRun: pip install ragas")


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

async def evaluate_single(
    query: str,
    retrieved_contexts: list[str],
    response: str,
    cfg: EvalConfig,
) -> dict:
    """Run RAGAS metrics on one pipeline result. Returns a dict of scores."""
    ragas_llm = cfg.ragas_llm()
    ragas_emb = cfg.ragas_embeddings()

    sample = SingleTurnSample(
        user_input=query,
        retrieved_contexts=retrieved_contexts,
        response=response,
    )
    dataset = EvaluationDataset(samples=[sample])

    # Generous timeout for local/slow judge models; avoids silent nan scores.
    run_cfg = RunConfig(timeout=300, max_retries=1, max_wait=10)

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb),
            LLMContextPrecisionWithoutReference(llm=ragas_llm),
        ],
        run_config=run_cfg,
    )
    df = result.to_pandas()
    row = df.iloc[0]
    return {
        "faithfulness":       round(float(row.get("faithfulness", 0)), 4),
        "response_relevancy": round(float(row.get("response_relevancy", 0)), 4),
        "context_precision":  round(float(row.get("llm_context_precision_without_reference", 0)), 4),
    }


def evaluate_entity_recall(
    knowledge_map: dict,
    merged_context: str,
) -> float:
    """
    Proxy for entity recall: check what fraction of knowledge graph node
    labels appear (case-insensitive) in the merged context.

    This is a lightweight string-match proxy. For higher accuracy, replace
    with RAGAS ContextEntitiesRecall (requires a reference answer).
    """
    nodes = knowledge_map.get("nodes", [])
    if not nodes:
        return 0.0
    ctx_lower = merged_context.lower()
    found = sum(
        1 for n in nodes
        if n.get("label", "")
        and re.search(
            r'\b' + re.escape(n["label"].lower()) + r'\b',
            ctx_lower,
        )
    )
    return round(found / len(nodes), 4)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_ragas_eval(
    query_ids: list[str] | None = None,
    output_dir: str = "eval_results",
    cfg: EvalConfig | None = None,
    # Pre-computed pipeline results: maps query_id -> (state, wall_time).
    # When provided, pipeline runs are skipped (reuse results from run_eval.py).
    pipeline_cache: dict | None = None,
    # Legacy kwarg kept for backwards compatibility
    api_key: str | None = None,
) -> list[dict]:
    if cfg is None:
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("Pass an EvalConfig or set OPENAI_API_KEY.")
        cfg = EvalConfig(provider="openai", api_key=key)

    queries = [q for q in TEST_QUERIES if query_ids is None or q["id"] in query_ids]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Run (or reuse) pipelines for all queries
    # ------------------------------------------------------------------
    pipeline_results: list[tuple[dict, dict, float]] = []  # (tq, state, wall_time)
    for tq in queries:
        if pipeline_cache and tq["id"] in pipeline_cache:
            state, wall_time = pipeline_cache[tq["id"]]
            print(f"\n[RAGAS] Reusing cached pipeline: {tq['id']}")
        else:
            print(f"\n[RAGAS] Running: {tq['id']} — {tq['query'][:60]}...")
            state, wall_time = run_pipeline(tq["query"], cfg=cfg)
        pipeline_results.append((tq, state, wall_time))

    # ------------------------------------------------------------------
    # Phase 2: Batch all RAGAS evaluations concurrently
    # ------------------------------------------------------------------
    # Build (index, coroutine) pairs only for queries with non-empty context.
    async def _batch_evaluate():
        tasks = {}
        for i, (tq, state, _) in enumerate(pipeline_results):
            contexts = split_context_into_chunks(state.get("merged_context", ""))
            if contexts:
                tasks[i] = evaluate_single(
                    query=tq["query"],
                    retrieved_contexts=contexts,
                    response=state.get("summary", ""),
                    cfg=cfg,
                )
        if not tasks:
            return {}
        indices = list(tasks.keys())
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return dict(zip(indices, results))

    print(f"\n[RAGAS] Evaluating {len(pipeline_results)} queries concurrently...")
    score_by_index = asyncio.run(_batch_evaluate())

    # ------------------------------------------------------------------
    # Phase 3: Collate results
    # ------------------------------------------------------------------
    all_results = []
    for i, (tq, state, wall_time) in enumerate(pipeline_results):
        contexts = split_context_into_chunks(state.get("merged_context", ""))

        if not contexts:
            print(f"  WARNING: empty merged_context for {tq['id']}, skipping RAGAS.")
            scores = {
                "faithfulness":       None,
                "response_relevancy": None,
                "context_precision":  None,
            }
        else:
            raw = score_by_index.get(i)
            if isinstance(raw, Exception):
                print(f"  WARNING: RAGAS evaluation failed for {tq['id']}: {raw}")
                scores = {
                    "faithfulness":       None,
                    "response_relevancy": None,
                    "context_precision":  None,
                }
            else:
                scores = raw

        entity_recall = evaluate_entity_recall(
            state.get("knowledge_map", {}),
            state.get("merged_context", ""),
        )

        result = {
            "query_id":       tq["id"],
            "query":          tq["query"],
            "difficulty":     tq["difficulty"],
            "provider":       cfg.provider,
            "model":          cfg.model,
            "judge_provider": cfg._jp(),
            "judge_model":    cfg._jm(),
            "wall_time_s":    round(wall_time, 2),
            "loop_count":     state.get("loop_count", 0),
            "active_agents":  state.get("active_agents", []),
            "node_count":     len(state.get("knowledge_map", {}).get("nodes", [])),
            "entity_recall":  entity_recall,
            "context_chunks": len(contexts),
            **scores,
        }

        # Pass/fail flags
        result["faithfulness_pass"]      = (result["faithfulness"] or 0) >= 0.75
        result["relevancy_pass"]         = (result["response_relevancy"] or 0) >= 0.75
        result["context_precision_pass"] = (result["context_precision"] or 0) >= 0.70
        result["entity_recall_pass"]     = result["entity_recall"] >= 0.60

        all_results.append(result)

        print(f"  Provider:        {cfg.provider} / {cfg.model}")
        print(f"  Judge:           {cfg._jp()} / {cfg._jm()}")
        print(f"  Faithfulness:    {result['faithfulness']}")
        print(f"  Resp. relevancy: {result['response_relevancy']}")
        print(f"  Context prec.:   {result['context_precision']}")
        print(f"  Entity recall:   {result['entity_recall']}")
        print(f"  Nodes:           {result['node_count']}")
        print(f"  Wall time:       {result['wall_time_s']}s")

    # Write output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"ragas_{ts}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[RAGAS] Results written to {out_path}")

    # Print aggregate summary
    print("\n=== RAGAS AGGREGATE ===")
    metrics = ["faithfulness", "response_relevancy", "context_precision", "entity_recall"]
    for m in metrics:
        vals = [r[m] for r in all_results if r.get(m) is not None]
        if vals:
            mean = round(sum(vals) / len(vals), 4)
            passes = sum(1 for r in all_results if r.get(f"{m}_pass"))
            print(f"  {m:30s} mean={mean}  pass={passes}/{len(all_results)}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--query-ids", nargs="*", help="Subset of query IDs to run")
    parser.add_argument("--output-dir", default="eval_results")
    add_provider_args(parser)
    args = parser.parse_args()
    run_ragas_eval(
        query_ids=args.query_ids,
        output_dir=args.output_dir,
        cfg=cfg_from_args(args),
    )
