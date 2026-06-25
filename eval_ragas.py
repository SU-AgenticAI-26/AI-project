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
    python eval_ragas.py --query-ids q1_rag q2_continual

Requires:
    pip install ragas langchain-huggingface
"""

# ── Must be set before any HuggingFace / tokenizer import ──────────────────
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import argparse
import asyncio
import json
import math
import re
import sys
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

def _safe_score(val, default=None):
    """Return float or None; guards against NaN from failed LLM calls."""
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return default


def evaluate_single_sync(
    query: str,
    retrieved_contexts: list[str],
    response: str,
    cfg: EvalConfig,
) -> dict:
    """
    Run RAGAS metrics synchronously for one pipeline result.
    Called in a thread via asyncio.to_thread to avoid blocking the event loop,
    but the call itself is fully synchronous — no nested asyncio.run().
    """
    response = (response or "").strip()
    if not response:
        return {
            "faithfulness": None,
            "response_relevancy": None,
            "context_precision": None,
            "_diag": "empty_response",
        }

    ragas_llm = cfg.ragas_llm()
    ragas_emb = cfg.ragas_embeddings()

    sample = SingleTurnSample(
        user_input=query,
        retrieved_contexts=retrieved_contexts,
        response=response,
    )
    dataset = EvaluationDataset(samples=[sample])

    # Generous timeout; max_retries=3 handles occasional judge parse failures.
    run_cfg = RunConfig(timeout=300, max_retries=3, max_wait=30)

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb),
            LLMContextPrecisionWithoutReference(llm=ragas_llm),
        ],
        run_config=run_cfg,
        raise_exceptions=False,
        show_progress=False,
    )
    df = result.to_pandas()
    row = df.iloc[0]

    response_relevancy = row.get("response_relevancy")
    if response_relevancy is None:
        response_relevancy = row.get("answer_relevancy")

    return {
        "faithfulness":       _safe_score(row.get("faithfulness")),
        "response_relevancy": _safe_score(response_relevancy),
        "context_precision":  _safe_score(row.get("llm_context_precision_without_reference")),
        "_diag": None,
    }


def evaluate_entity_recall(
    knowledge_map: dict,
    merged_context: str,
) -> float:
    """
    Proxy for entity recall: fraction of KG node labels found
    (case-insensitive word-boundary match) in the merged context.
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
# Async batch runner — sequential to avoid Windows tokenizer deadlock
# ---------------------------------------------------------------------------

async def _batch_evaluate_sequential(
    pipeline_results: list[tuple[dict, dict, float]],
    cfg: EvalConfig,
) -> dict[int, dict]:
    """
    Evaluate each query one at a time in a background thread.

    Running them all concurrently via asyncio.gather + asyncio.to_thread
    causes a tokenizer deadlock on Windows (TOKENIZERS_PARALLELISM=false
    alone is not always sufficient when multiple threads initialise the
    model simultaneously). Sequential execution is ~same wall time because
    each RAGAS call is I/O-bound (waiting on the judge LLM).
    """
    results: dict[int, dict] = {}
    for i, (tq, state, _) in enumerate(pipeline_results):
        contexts = split_context_into_chunks(state.get("merged_context", ""))
        if not contexts:
            continue

        print(f"  [RAGAS] Evaluating {tq['id']} ({i + 1}/{len(pipeline_results)})...")
        try:
            scores = await asyncio.to_thread(
                evaluate_single_sync,
                tq["query"],
                contexts,
                state.get("summary", ""),
                cfg,
            )
        except Exception as exc:
            print(f"  WARNING: RAGAS evaluation failed for {tq['id']}: {exc}")
            scores = {
                "faithfulness": None,
                "response_relevancy": None,
                "context_precision": None,
                "_diag": str(exc),
            }
        results[i] = scores
    return results


def _run_async(coro):
    """
    Cross-platform asyncio.run() wrapper.

    On Windows with Python 3.10+ the default ProactorEventLoop works fine,
    but if an event loop is already running (e.g. inside Jupyter / Streamlit)
    asyncio.run() raises RuntimeError. This wrapper handles both cases.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop (Streamlit, Jupyter) — use nest_asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except ImportError:
            raise RuntimeError(
                "Running inside an existing event loop. "
                "Install nest_asyncio: pip install nest_asyncio"
            )
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _print_header() -> None:
    print("""\
╔══════════════════════════════════════════════════════════════════╗
║                      RAGAS Evaluation Framework                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Metrics (all scored 0.0 – 1.0, higher is better):              ║
║    faithfulness        Are all claims grounded in context?       ║
║    response_relevancy  Does the summary address the query?       ║
║    context_precision   Is retrieved context focused/not noisy?   ║
║    entity_recall       KG node labels found in merged context.   ║
║                                                                  ║
║  PASS thresholds:                                                ║
║    faithfulness ≥ 0.75  |  response_relevancy ≥ 0.75            ║
║    context_precision ≥ 0.70  |  entity_recall ≥ 0.60            ║
╚══════════════════════════════════════════════════════════════════╝
""")


def run_ragas_eval(
    query_ids: list[str] | None = None,
    output_dir: str = "eval_results",
    cfg: EvalConfig | None = None,
    pipeline_cache: dict | None = None,
    api_key: str | None = None,   # legacy kwarg
) -> list[dict]:
    _print_header()

    if cfg is None:
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("Pass an EvalConfig or set OPENAI_API_KEY.")
        cfg = EvalConfig(provider="openai", api_key=key)

    queries = [q for q in TEST_QUERIES if query_ids is None or q["id"] in query_ids]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Run (or reuse) pipelines
    # ------------------------------------------------------------------
    pipeline_results: list[tuple[dict, dict, float]] = []
    for tq in queries:
        if pipeline_cache and tq["id"] in pipeline_cache:
            state, wall_time = pipeline_cache[tq["id"]]
            print(f"[RAGAS] Reusing cached pipeline: {tq['id']}")
        else:
            print(f"[RAGAS] Running pipeline: {tq['id']} — {tq['query'][:60]}...")
            state, wall_time = run_pipeline(tq["query"], cfg=cfg)
        pipeline_results.append((tq, state, wall_time))

    # ------------------------------------------------------------------
    # Phase 2: RAGAS evaluation — sequential to avoid tokenizer deadlock
    # ------------------------------------------------------------------
    print(f"\n[RAGAS] Evaluating {len(pipeline_results)} queries (sequential, Windows-safe)...")
    score_by_index = _run_async(
        _batch_evaluate_sequential(pipeline_results, cfg)
    )

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
            if raw is None or isinstance(raw, Exception):
                print(f"  WARNING: no scores for {tq['id']}: {raw}")
                scores = {
                    "faithfulness":       None,
                    "response_relevancy": None,
                    "context_precision":  None,
                }
            else:
                scores = {k: v for k, v in raw.items() if k != "_diag"}

        if scores.get("response_relevancy") is None and contexts:
            print(
                f"  WARNING: response_relevancy is None for {tq['id']} despite "
                "non-empty response. Check judge/embedding provider configuration."
            )

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
        result["faithfulness_pass"]       = (result.get("faithfulness") or 0) >= 0.75
        result["response_relevancy_pass"] = (result.get("response_relevancy") or 0) >= 0.75
        result["context_precision_pass"]  = (result.get("context_precision") or 0) >= 0.70
        result["entity_recall_pass"]      = result["entity_recall"] >= 0.60

        all_results.append(result)

        print(f"\n  [{tq['id']}]")
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

    # Aggregate summary
    print("\n=== RAGAS AGGREGATE ===")
    metrics = ["faithfulness", "response_relevancy", "context_precision", "entity_recall"]
    for m in metrics:
        vals = [r[m] for r in all_results if r.get(m) is not None]
        if vals:
            mean_val = round(sum(vals) / len(vals), 4)
            passes = sum(1 for r in all_results if r.get(f"{m}_pass"))
            print(f"  {m:30s} mean={mean_val}  pass={passes}/{len(all_results)}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Windows ProactorEventLoop fix for Python 3.8+
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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