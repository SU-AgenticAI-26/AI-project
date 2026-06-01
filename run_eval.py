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

    # Reset all cached state before running (clean slate)
    python run_eval.py --clean

    # Inspect what --clean would remove without deleting anything
    python run_eval.py --clean --dry-run

Available modules: ragas, deepeval, uptrain, agentbench, graph, perf, baseline, ablation, citation
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
    "faithfulness":          0.75,
    "response_relevancy":    0.75,
    "context_precision":     0.70,
    "entity_recall":         0.60,
    "router_score":          0.80,
    "coherence_score":       0.65,
    "node_count_min":        8,
    "node_count_max":        25,
    "source_diversity":      0.30,
    "cap_hit_rate_max":      0.30,
    # UpTrain
    "response_relevance":    0.75,
    "response_completeness": 0.70,
    "context_relevance":     0.65,
    "response_conciseness":  0.60,
    # AgentBench
    "task_success":          0.70,
    "channel_f1":            0.75,
    "iteration_efficiency":  0.70,
    "kg_density":            0.60,
    # Citation
    "citation_accuracy":     0.75,
}


# ---------------------------------------------------------------------------
# Clean / reset helpers
# ---------------------------------------------------------------------------

_APP_DATA_ROOT = Path("collab_rag_data")

def clean_eval_state(output_dir: str, dry_run: bool = False) -> None:
    """
    Delete all state that can influence eval results between sessions:

      1. Pipeline cache JSON(s) — stale LLM outputs reused by --rerun-pipelines
      2. App query cache        — collab_rag_data/cache/*.json (20-day TTL)
      3. FAISS vector indices   — collab_rag_data/vectorstore/**/index.{faiss,pkl}
                                  (grows with interactive app use; biases vector_db_agent)

    Timestamped eval output files (ragas_*.json, summary_*.json, …) and the SQL
    knowledge.db are left untouched — they don't affect future pipeline runs.
    """
    targets: list[Path] = []

    # 1. Pipeline cache files in output_dir and repo root
    for pattern in (
        Path(output_dir).glob("pipeline_cache_*.json"),
        Path(".").glob("pipeline_cache_*.json"),
    ):
        targets.extend(pattern)

    # 2. App query cache
    targets.extend((_APP_DATA_ROOT / "cache").glob("*.json"))

    # 3. FAISS vector indices (all embedding-backend subdirectories)
    for suffix in ("index.faiss", "index.pkl"):
        targets.extend((_APP_DATA_ROOT / "vectorstore").rglob(suffix))

    if not targets:
        print("  [clean] Nothing to remove — already clean.")
        return

    label = "Would remove" if dry_run else "Removing"
    total = 0
    for p in sorted(targets):
        size = p.stat().st_size if p.exists() else 0
        print(f"  {label}: {p}  ({size:,} bytes)")
        if not dry_run:
            p.unlink(missing_ok=True)
            total += size

    if dry_run:
        print(f"  [dry-run] {len(targets)} file(s) would be removed.")
    else:
        print(f"  [clean] Removed {len(targets)} file(s), freed {total:,} bytes.")


# ---------------------------------------------------------------------------
# Pipeline disk-cache helpers
# ---------------------------------------------------------------------------

def _pipeline_cache_path(cfg: EvalConfig, output_dir: str) -> Path:
    """Return the auto-named pipeline cache file for this provider/model combo."""
    model_slug = (cfg.model or "default").replace("/", "-").replace(":", "-")
    return Path(output_dir) / f"pipeline_cache_{cfg.provider}_{model_slug}.json"


def _load_pipeline_cache(path: Path) -> dict:
    """
    Load pipeline results from a previously saved JSON cache.

    Returns a dict mapping query_id -> (state, wall_time), or {} if the
    file doesn't exist or can't be parsed.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        return {qid: (entry["state"], entry["wall_time"]) for qid, entry in raw.items()}
    except Exception as exc:
        print(f"  WARNING: Could not load pipeline cache from {path}: {exc}")
        return {}


def _save_pipeline_cache(path: Path, cache: dict) -> None:
    """
    Write the current pipeline cache to disk.

    LangChain message objects are excluded (they aren't JSON-serialisable and
    aren't needed by any eval module).
    """
    serialisable = {}
    for qid, (state, wall_time) in cache.items():
        safe_state = {k: v for k, v in state.items() if k != "messages"}
        serialisable[qid] = {"state": safe_state, "wall_time": wall_time}
    path.write_text(json.dumps(serialisable, indent=2, default=str))


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
    rerun_pipelines: bool = False,
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
    # Shared pipeline cache — ragas, deepeval, graph, uptrain, and agentbench
    # all need the same full pipeline results.  Pre-run them once here and
    # pass the cache so each module skips redundant pipeline invocations.
    #
    # Results are also persisted to a JSON file so that interrupted runs can
    # be resumed without re-running the expensive LangGraph pipeline steps.
    # ------------------------------------------------------------------
    _cache_consumers = {"ragas", "deepeval", "graph", "uptrain", "agentbench", "citation"}
    pipeline_cache: dict = {}
    if set(modules) & _cache_consumers:
        from test_queries import TEST_QUERIES, run_pipeline
        _queries_to_cache = [
            q for q in TEST_QUERIES
            if query_ids is None or q["id"] in query_ids
        ]

        disk_cache_path = _pipeline_cache_path(cfg, output_dir)

        # Load whatever has already been computed in a previous run.
        if not rerun_pipelines:
            pipeline_cache = _load_pipeline_cache(disk_cache_path)
            if pipeline_cache:
                cached_ids = sorted(pipeline_cache.keys())
                print(f"\n[Pipeline cache] Loaded {len(pipeline_cache)} cached "
                      f"result(s) from {disk_cache_path}: {cached_ids}")

        missing = [q for q in _queries_to_cache if q["id"] not in pipeline_cache]
        if missing:
            print("\n" + "="*60)
            print("PRE-RUNNING PIPELINES (shared across eval modules)")
            print("="*60)
            for tq in missing:
                print(f"  [{tq['id']}] {tq['query'][:70]}...")
                state, wall_time = run_pipeline(tq["query"], cfg=cfg)
                pipeline_cache[tq["id"]] = (state, wall_time)
                print(f"  Done in {wall_time:.1f}s  "
                      f"nodes={len(state.get('knowledge_map', {}).get('nodes', []))}")
                # Save incrementally so a crash mid-run doesn't lose earlier results.
                _save_pipeline_cache(disk_cache_path, pipeline_cache)
                print(f"  Saved to {disk_cache_path}")
        else:
            print(f"\n[Pipeline cache] All {len(_queries_to_cache)} queries already "
                  f"cached — skipping pipeline runs.")

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

    # --- UpTrain ---
    if "uptrain" in modules:
        print("\n" + "="*60)
        print("MODULE: UpTrain")
        print("="*60)
        from eval_uptrain import run_uptrain_eval
        uptrain_results = run_uptrain_eval(**kwargs, pipeline_cache=pipeline_cache or None)
        summary["results"]["uptrain"] = uptrain_results

        for metric in ["response_relevance", "response_completeness",
                        "context_relevance", "response_conciseness"]:
            mean, passed = check_pass(uptrain_results, metric, PASS_THRESHOLDS.get(metric, 0))
            summary["pass_fail"][f"uptrain_{metric}"] = {
                "mean": mean, "threshold": PASS_THRESHOLDS.get(metric), "pass": passed
            }

    # --- AgentBench ---
    if "agentbench" in modules:
        print("\n" + "="*60)
        print("MODULE: AgentBench")
        print("="*60)
        from eval_agentbench import run_agentbench_eval
        ab_kwargs = {k: v for k, v in kwargs.items()}
        ab_results = run_agentbench_eval(**ab_kwargs, pipeline_cache=pipeline_cache or None)
        summary["results"]["agentbench"] = ab_results

        for metric in ["task_success", "channel_f1", "iteration_efficiency", "kg_density"]:
            mean, passed = check_pass(ab_results, metric, PASS_THRESHOLDS.get(metric, 0))
            summary["pass_fail"][f"agentbench_{metric}"] = {
                "mean": mean, "threshold": PASS_THRESHOLDS.get(metric), "pass": passed
            }

    # --- Citation ---
    if "citation" in modules:
        print("\n" + "="*60)
        print("MODULE: Citation Verifier")
        print("="*60)
        from eval_citation import run_citation_eval
        cit_results = run_citation_eval(**kwargs, pipeline_cache=pipeline_cache or None)
        summary["results"]["citation"] = cit_results

        mean, passed = check_pass(cit_results, "citation_accuracy",
                                   PASS_THRESHOLDS["citation_accuracy"])
        summary["pass_fail"]["citation_accuracy"] = {
            "mean": mean, "threshold": PASS_THRESHOLDS["citation_accuracy"], "pass": passed
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
    ALL_MODULES = ["ragas", "deepeval", "uptrain", "agentbench", "graph", "perf", "baseline", "ablation", "citation"]

    parser = argparse.ArgumentParser(description="Run full evaluation suite")
    parser.add_argument("--quick",      action="store_true",
                        help="Run only graph + perf on q1_rag (fast sanity check)")
    parser.add_argument("--skip",       nargs="*", default=[],
                        help="Modules to skip")
    parser.add_argument("--query-ids",  nargs="*",
                        help="Subset of query IDs")
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--rerun-pipelines", action="store_true",
                        dest="rerun_pipelines",
                        help="Ignore disk pipeline cache and re-run all pipelines")
    parser.add_argument("--clean", action="store_true",
                        help="Delete pipeline cache, app query cache, and FAISS indices "
                             "before running, ensuring a clean-slate eval")
    parser.add_argument("--dry-run", action="store_true",
                        dest="dry_run",
                        help="With --clean: print what would be deleted without removing anything")
    add_provider_args(parser)
    args = parser.parse_args()

    if args.clean or args.dry_run:
        label = "DRY RUN — " if args.dry_run else ""
        print(f"\n{'='*60}")
        print(f"{label}CLEANING EVAL STATE")
        print(f"{'='*60}")
        clean_eval_state(args.output_dir, dry_run=args.dry_run)
        if args.dry_run:
            raise SystemExit(0)
        print()

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
        rerun_pipelines=args.rerun_pipelines,
    )
