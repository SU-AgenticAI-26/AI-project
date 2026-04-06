"""
run_eval.py — Master evaluation runner.

Runs the full evaluation suite and writes a consolidated summary report.

Usage:
    # Full suite with OpenAI (default)
    python run_eval.py

    # Local llama.cpp pipeline, OpenAI judge for RAGAS/DeepEval
    python run_eval.py --provider local --model mistral \\
                       --base-url http://localhost:8080/v1 \\
                       --judge-provider openai

    # Fully local (pipeline + judge) — quality of RAGAS/DeepEval scores may vary
    python run_eval.py --provider local --model mistral --base-url http://localhost:8080/v1

    # Quick sanity check (1 query, skip ablation)
    python run_eval.py --quick

    # Skip specific modules
    python run_eval.py --skip ragas ablation

    # Only specific queries
    python run_eval.py --query-ids q1_rag q2_federated

Available modules: ragas, deepeval, graph, perf, baseline, ablation
"""

from __future__ import annotations
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

# Suppress "missing ScriptRunContext" warnings Streamlit emits when its
# modules are imported outside a running Streamlit server.
# Streamlit's child loggers set their own levels, so a filter on the root
# streamlit logger is more reliable than just setting the level there.
def _suppress_streamlit_bare_mode_warnings():
    class _Drop(logging.Filter):
        def filter(self, record):
            return "ScriptRunContext" not in record.getMessage()
    for name in (
        "streamlit",
        "streamlit.runtime.scriptrunner",
        "streamlit.runtime.scriptrunner.script_run_context",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.addFilter(_Drop())

_suppress_streamlit_bare_mode_warnings()

from eval_provider import EvalConfig, add_provider_args, cfg_from_args


PASS_THRESHOLDS = {
    "faithfulness":       0.75,
    "response_relevancy": 0.75,
    "context_precision":  0.70,
    "entity_recall":      0.60,
    "router_score":       0.80,
    "coherence_score":    0.65,
    "node_count_min":     8,
    "node_count_max":     25,
    "source_diversity":   0.30,
    "cap_hit_rate_max":   0.30,
}


def check_pass(results: list[dict], metric: str, threshold: float) -> tuple[float | None, bool | None]:
    vals = [r[metric] for r in results if r.get(metric) is not None]
    if not vals:
        return None, None
    mean = round(sum(vals) / len(vals), 4)
    return mean, mean >= threshold


def run_full_eval(
    modules: list[str],
    query_ids: list[str] | None,
    output_dir: str,
    cfg: EvalConfig,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary = {
        "run_at":         datetime.now().isoformat(),
        "modules":        modules,
        "query_ids":      query_ids,
        "provider":       cfg.provider,
        "model":          cfg.model,
        "judge_provider": cfg._jp(),
        "judge_model":    cfg._jm(),
        "results":        {},
        "pass_fail":      {},
    }

    kwargs = dict(output_dir=output_dir, cfg=cfg)
    if query_ids:
        kwargs["query_ids"] = query_ids

    # ------------------------------------------------------------------
    # Shared pipeline cache — ragas, deepeval, and graph all need the same
    # full pipeline results. Pre-run them once here and pass the cache so
    # each module skips redundant pipeline invocations.
    # ------------------------------------------------------------------
    _cache_consumers = {"ragas", "deepeval", "graph"}
    pipeline_cache: dict = {}
    if len(set(modules) & _cache_consumers) > 1:
        from test_queries import TEST_QUERIES, run_pipeline
        _queries_to_cache = [
            q for q in TEST_QUERIES
            if query_ids is None or q["id"] in query_ids
        ]
        print("\n" + "="*60)
        print("PRE-RUNNING PIPELINES (shared across ragas / deepeval / graph)")
        print("="*60)
        for tq in _queries_to_cache:
            print(f"  [{tq['id']}] {tq['query'][:70]}...")
            state, wall_time = run_pipeline(tq["query"], cfg=cfg)
            pipeline_cache[tq["id"]] = (state, wall_time)
            print(f"  Done in {wall_time:.1f}s  "
                  f"nodes={len(state.get('knowledge_map', {}).get('nodes', []))}")

    # --- Baseline ---
    if "baseline" in modules:
        print("\n" + "="*60)
        print("MODULE: Baseline")
        print("="*60)
        from eval_baseline import run_baseline_eval
        baseline_results = run_baseline_eval(**kwargs)
        summary["results"]["baseline"] = baseline_results

    # --- RAGAS ---
    if "ragas" in modules:
        print("\n" + "="*60)
        print("MODULE: RAGAS")
        print("="*60)
        from eval_ragas import run_ragas_eval
        ragas_results = run_ragas_eval(**kwargs, pipeline_cache=pipeline_cache or None)
        summary["results"]["ragas"] = ragas_results

        for metric in ["faithfulness", "response_relevancy", "context_precision", "entity_recall"]:
            mean, passed = check_pass(ragas_results, metric, PASS_THRESHOLDS.get(metric, 0))
            summary["pass_fail"][f"ragas_{metric}"] = {
                "mean": mean, "threshold": PASS_THRESHOLDS.get(metric), "pass": passed
            }

    # --- DeepEval ---
    if "deepeval" in modules:
        print("\n" + "="*60)
        print("MODULE: DeepEval")
        print("="*60)
        from eval_deepeval import run_deepeval_eval
        de_results = run_deepeval_eval(**kwargs, pipeline_cache=pipeline_cache or None)
        summary["results"]["deepeval"] = de_results

        for metric in ["router_score", "coherence_score"]:
            mean, passed = check_pass(de_results, metric, PASS_THRESHOLDS.get(metric, 0))
            summary["pass_fail"][f"deepeval_{metric}"] = {
                "mean": mean, "threshold": PASS_THRESHOLDS.get(metric), "pass": passed
            }

    # --- Graph ---
    if "graph" in modules:
        print("\n" + "="*60)
        print("MODULE: Knowledge Graph")
        print("="*60)
        from eval_graph import run_graph_eval
        graph_results = run_graph_eval(**kwargs, pipeline_cache=pipeline_cache or None)
        summary["results"]["graph"] = graph_results

        for metric, thr in [("source_diversity", PASS_THRESHOLDS["source_diversity"]),
                             ("entity_recall",    PASS_THRESHOLDS["entity_recall"])]:
            mean, passed = check_pass(graph_results, metric, thr)
            summary["pass_fail"][f"graph_{metric}"] = {
                "mean": mean, "threshold": thr, "pass": passed
            }
        node_means = [r["node_count"] for r in graph_results]
        if node_means:
            mean_nodes = round(sum(node_means) / len(node_means), 1)
            summary["pass_fail"]["graph_node_count"] = {
                "mean": mean_nodes,
                "threshold": f"[{PASS_THRESHOLDS['node_count_min']}, {PASS_THRESHOLDS['node_count_max']}]",
                "pass": all(
                    PASS_THRESHOLDS["node_count_min"] <= r["node_count"] <= PASS_THRESHOLDS["node_count_max"]
                    for r in graph_results
                ),
            }

    # --- Perf ---
    if "perf" in modules:
        print("\n" + "="*60)
        print("MODULE: Performance")
        print("="*60)
        from eval_perf import run_perf_eval
        perf_results = run_perf_eval(**kwargs)
        summary["results"]["perf"] = perf_results

        cap_hits = [r for r in perf_results if r.get("cap_hit")]
        cap_rate = round(len(cap_hits) / len(perf_results), 3) if perf_results else None
        summary["pass_fail"]["critic_cap_hit_rate"] = {
            "rate": cap_rate,
            "threshold_max": PASS_THRESHOLDS["cap_hit_rate_max"],
            "pass": cap_rate is not None and cap_rate <= PASS_THRESHOLDS["cap_hit_rate_max"],
        }

    # --- Ablation ---
    if "ablation" in modules:
        print("\n" + "="*60)
        print("MODULE: Ablation")
        print("="*60)
        from eval_ablation import run_ablation
        ablation_results = run_ablation(output_dir=output_dir, cfg=cfg)
        summary["results"]["ablation"] = ablation_results

    # --- Write consolidated summary ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(output_dir) / f"summary_{ts}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    # Print final pass/fail table
    print("\n" + "="*60)
    print("FINAL PASS/FAIL SUMMARY")
    print("="*60)
    print(f"  Provider: {cfg.provider} / {cfg.model or '(default)'}  "
          f"Judge: {cfg._jp()} / {cfg._jm() or '(default)'}")
    all_pass = True
    for check, result in summary["pass_fail"].items():
        passed  = result.get("pass")
        symbol  = "PASS" if passed else ("FAIL" if passed is False else "N/A ")
        mean    = result.get("mean") or result.get("rate", "")
        thr     = result.get("threshold") or result.get("threshold_max", "")
        print(f"  [{symbol}] {check:40s} mean={mean}  threshold={thr}")
        if passed is False:
            all_pass = False

    overall = "ALL PASS" if all_pass else "SOME FAILURES — see details above"
    print(f"\n  Overall: {overall}")
    print(f"  Summary written to: {summary_path}")
    return summary


if __name__ == "__main__":
    ALL_MODULES = ["ragas", "deepeval", "graph", "perf", "baseline", "ablation"]

    parser = argparse.ArgumentParser(description="Run full evaluation suite")
    parser.add_argument("--quick",      action="store_true",
                        help="Run only graph + perf on q1_rag (fast sanity check)")
    parser.add_argument("--skip",       nargs="*", default=[],
                        help="Modules to skip")
    parser.add_argument("--query-ids",  nargs="*",
                        help="Subset of query IDs")
    parser.add_argument("--output-dir", default="eval_results")
    add_provider_args(parser)
    args = parser.parse_args()

    cfg = cfg_from_args(args)

    # Require an API key unless using a local provider
    if cfg.provider != "local" and not cfg.api_key:
        raise SystemExit(
            f"No API key for provider '{cfg.provider}'. "
            f"Set the appropriate environment variable or pass --api-key."
        )

    if args.quick:
        modules   = ["graph", "perf"]
        query_ids = ["q1_rag"]
    else:
        modules   = [m for m in ALL_MODULES if m not in (args.skip or [])]
        query_ids = args.query_ids

    run_full_eval(
        modules=modules,
        query_ids=query_ids,
        output_dir=args.output_dir,
        cfg=cfg,
    )
