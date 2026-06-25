"""
run_eval.py — Master evaluation runner (parallelised).

All eval modules that share the pipeline cache (ragas, deepeval, graph,
uptrain, agentbench, citation) now run concurrently via a thread pool.
Baseline, perf, and ablation run in parallel with the main group.

Usage:
    python run_eval.py                          # full suite
    python run_eval.py --quick                  # graph + perf on q1_rag only
    python run_eval.py --skip ragas ablation    # skip specific modules
    python run_eval.py --query-ids q1_rag q2_continual
    python run_eval.py --clean                  # wipe cache first
    python run_eval.py --clean --dry-run        # inspect what --clean removes
    python run_eval.py --rerun-pipelines        # ignore disk pipeline cache
"""

from __future__ import annotations
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Suppress Streamlit bare-mode noise
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Pass/fail thresholds
# ---------------------------------------------------------------------------
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
    "response_relevance":    0.75,
    "response_completeness": 0.70,
    "context_relevance":     0.65,
    "response_conciseness":  0.60,
    "task_success":          0.70,
    "channel_f1":            0.75,
    "iteration_efficiency":  0.70,
    "kg_density":            0.60,
    "citation_accuracy":     0.75,
}


# ---------------------------------------------------------------------------
# Clean / reset helpers
# ---------------------------------------------------------------------------
_APP_DATA_ROOT = Path("collab_rag_data")

def clean_eval_state(output_dir: str, dry_run: bool = False) -> None:
    candidates: list[Path] = []
    for pattern in (
        Path(output_dir).glob("pipeline_cache_*.json"),
        Path(".").glob("pipeline_cache_*.json"),
    ):
        candidates.extend(pattern)
    candidates.extend((_APP_DATA_ROOT / "cache").glob("*.json"))
    for suffix in ("index.faiss", "index.pkl"):
        candidates.extend((_APP_DATA_ROOT / "vectorstore").rglob(suffix))

    seen: set[Path] = set()
    targets: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen and p.exists():
            seen.add(rp)
            targets.append(p)

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
    model_slug = (cfg.model or "default").replace("/", "-").replace(":", "-")
    return Path(output_dir) / f"pipeline_cache_{cfg.provider}_{model_slug}.json"

def _load_pipeline_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        return {qid: (entry["state"], entry["wall_time"]) for qid, entry in raw.items()}
    except Exception as exc:
        print(f"  WARNING: Could not load pipeline cache from {path}: {exc}")
        return {}

def _save_pipeline_cache(path: Path, cache: dict) -> None:
    serialisable = {}
    for qid, (state, wall_time) in cache.items():
        safe_state = {k: v for k, v in state.items() if k != "messages"}
        serialisable[qid] = {"state": safe_state, "wall_time": wall_time}
    path.write_text(json.dumps(serialisable, indent=2, default=str))

def check_pass(results: list[dict], metric: str, threshold: float):
    vals = [r[metric] for r in results if r.get(metric) is not None]
    if not vals:
        return None, None
    mean = round(sum(vals) / len(vals), 4)
    return mean, mean >= threshold


# ---------------------------------------------------------------------------
# Per-module runners — each returns (module_name, results_list)
# These are called from threads so they must be thread-safe.
# Each module gets its own copy of kwargs to avoid shared-state issues.
# ---------------------------------------------------------------------------

def _run_baseline(kwargs: dict) -> tuple[str, list]:
    from eval_baseline import run_baseline_eval
    return "baseline", run_baseline_eval(**kwargs)

def _run_ragas(kwargs: dict, pipeline_cache: dict) -> tuple[str, list]:
    from eval_ragas import run_ragas_eval
    return "ragas", run_ragas_eval(**kwargs, pipeline_cache=pipeline_cache or None)

def _run_deepeval(kwargs: dict, pipeline_cache: dict) -> tuple[str, list]:
    from eval_deepeval import run_deepeval_eval
    return "deepeval", run_deepeval_eval(**kwargs, pipeline_cache=pipeline_cache or None)

def _run_graph(kwargs: dict, pipeline_cache: dict) -> tuple[str, list]:
    from eval_graph import run_graph_eval
    return "graph", run_graph_eval(**kwargs, pipeline_cache=pipeline_cache or None)

def _run_perf(kwargs: dict) -> tuple[str, list]:
    from eval_perf import run_perf_eval
    return "perf", run_perf_eval(**kwargs)

def _run_uptrain(kwargs: dict, pipeline_cache: dict) -> tuple[str, list]:
    from eval_uptrain import run_uptrain_eval
    return "uptrain", run_uptrain_eval(**kwargs, pipeline_cache=pipeline_cache or None)

def _run_agentbench(kwargs: dict, pipeline_cache: dict) -> tuple[str, list]:
    from eval_agentbench import run_agentbench_eval
    return "agentbench", run_agentbench_eval(**kwargs, pipeline_cache=pipeline_cache or None)

def _run_citation(kwargs: dict, pipeline_cache: dict) -> tuple[str, list]:
    from eval_citation import run_citation_eval
    return "citation", run_citation_eval(**kwargs, pipeline_cache=pipeline_cache or None)

def _run_ablation(kwargs: dict) -> tuple[str, list]:
    from eval_ablation import run_ablation
    # ablation only takes output_dir and cfg
    return "ablation", run_ablation(output_dir=kwargs["output_dir"], cfg=kwargs["cfg"])


# ---------------------------------------------------------------------------
# Pass/fail collation — called after all modules complete
# ---------------------------------------------------------------------------

def _collate_pass_fail(summary: dict) -> None:
    results = summary["results"]
    pf = summary["pass_fail"]

    if "ragas" in results:
        for m in ["faithfulness", "response_relevancy", "context_precision", "entity_recall"]:
            mean, passed = check_pass(results["ragas"], m, PASS_THRESHOLDS.get(m, 0))
            pf[f"ragas_{m}"] = {"mean": mean, "threshold": PASS_THRESHOLDS.get(m), "pass": passed}

    if "deepeval" in results:
        for m in ["router_score", "coherence_score"]:
            mean, passed = check_pass(results["deepeval"], m, PASS_THRESHOLDS.get(m, 0))
            pf[f"deepeval_{m}"] = {"mean": mean, "threshold": PASS_THRESHOLDS.get(m), "pass": passed}

    if "graph" in results:
        for m, thr in [("source_diversity", PASS_THRESHOLDS["source_diversity"]),
                       ("entity_recall",    PASS_THRESHOLDS["entity_recall"])]:
            mean, passed = check_pass(results["graph"], m, thr)
            pf[f"graph_{m}"] = {"mean": mean, "threshold": thr, "pass": passed}
        node_means = [r["node_count"] for r in results["graph"]]
        if node_means:
            mn = round(sum(node_means) / len(node_means), 1)
            pf["graph_node_count"] = {
                "mean": mn,
                "threshold": f"[{PASS_THRESHOLDS['node_count_min']}, {PASS_THRESHOLDS['node_count_max']}]",
                "pass": all(
                    PASS_THRESHOLDS["node_count_min"] <= r["node_count"] <= PASS_THRESHOLDS["node_count_max"]
                    for r in results["graph"]
                ),
            }

    if "perf" in results:
        pr = results["perf"]
        cap_hits = [r for r in pr if r.get("cap_hit")]
        cap_rate = round(len(cap_hits) / len(pr), 3) if pr else None
        pf["critic_cap_hit_rate"] = {
            "rate": cap_rate,
            "threshold_max": PASS_THRESHOLDS["cap_hit_rate_max"],
            "pass": cap_rate is not None and cap_rate <= PASS_THRESHOLDS["cap_hit_rate_max"],
        }

    if "uptrain" in results:
        for m in ["response_relevance", "response_completeness",
                  "context_relevance", "response_conciseness"]:
            mean, passed = check_pass(results["uptrain"], m, PASS_THRESHOLDS.get(m, 0))
            pf[f"uptrain_{m}"] = {"mean": mean, "threshold": PASS_THRESHOLDS.get(m), "pass": passed}

    if "agentbench" in results:
        for m in ["task_success", "channel_f1", "iteration_efficiency", "kg_density"]:
            mean, passed = check_pass(results["agentbench"], m, PASS_THRESHOLDS.get(m, 0))
            pf[f"agentbench_{m}"] = {"mean": mean, "threshold": PASS_THRESHOLDS.get(m), "pass": passed}

    if "citation" in results:
        mean, passed = check_pass(results["citation"], "citation_accuracy",
                                  PASS_THRESHOLDS["citation_accuracy"])
        pf["citation_accuracy"] = {
            "mean": mean, "threshold": PASS_THRESHOLDS["citation_accuracy"], "pass": passed
        }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

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

    base_kwargs = dict(output_dir=output_dir, cfg=cfg)
    if query_ids:
        base_kwargs["query_ids"] = query_ids

    # ------------------------------------------------------------------
    # Step 1: Build shared pipeline cache (serial — must finish first)
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

        if not rerun_pipelines:
            pipeline_cache = _load_pipeline_cache(disk_cache_path)
            if pipeline_cache:
                print(f"\n[Pipeline cache] Loaded {len(pipeline_cache)} cached "
                      f"result(s) from {disk_cache_path}: {sorted(pipeline_cache.keys())}")

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
                _save_pipeline_cache(disk_cache_path, pipeline_cache)
        else:
            print(f"\n[Pipeline cache] All {len(_queries_to_cache)} queries cached "
                  f"— skipping pipeline runs.")

    # ------------------------------------------------------------------
    # Step 2: Run all modules concurrently
    #
    # Groups:
    #   A — share pipeline_cache, are I/O bound (LLM judge calls):
    #       ragas, deepeval, graph, uptrain, agentbench, citation
    #   B — independent, run in parallel with group A:
    #       baseline, perf, ablation
    #
    # RAGAS uses asyncio.to_thread internally (sequential, Windows-safe).
    # All other modules are synchronous. ThreadPoolExecutor handles both.
    # ------------------------------------------------------------------

    # Map module name -> callable(base_kwargs, pipeline_cache) or callable(base_kwargs)
    module_fns = {
        "baseline":   lambda kw, pc: _run_baseline(kw),
        "ragas":      lambda kw, pc: _run_ragas(kw, pc),
        "deepeval":   lambda kw, pc: _run_deepeval(kw, pc),
        "graph":      lambda kw, pc: _run_graph(kw, pc),
        "perf":       lambda kw, pc: _run_perf(kw),
        "uptrain":    lambda kw, pc: _run_uptrain(kw, pc),
        "agentbench": lambda kw, pc: _run_agentbench(kw, pc),
        "citation":   lambda kw, pc: _run_citation(kw, pc),
        "ablation":   lambda kw, pc: _run_ablation(kw),
    }

    active = [m for m in modules if m in module_fns]

    print("\n" + "="*60)
    print(f"RUNNING {len(active)} MODULE(S) CONCURRENTLY: {', '.join(active)}")
    print("="*60)

    t_start = datetime.now()
    errors: dict[str, Exception] = {}

    # One thread per module. RAGAS is the slowest (~judge LLM × 4 queries);
    # all others finish well before it. Max workers capped at module count.
    with ThreadPoolExecutor(max_workers=len(active)) as pool:
        futures = {
            pool.submit(module_fns[m], dict(base_kwargs), pipeline_cache): m
            for m in active
        }
        for future in as_completed(futures):
            module_name = futures[future]
            try:
                name, result = future.result()
                summary["results"][name] = result
                elapsed = (datetime.now() - t_start).seconds
                print(f"  ✓ [{elapsed:>3}s] {name} complete")
            except Exception as exc:
                errors[module_name] = exc
                print(f"  ✗ {module_name} FAILED: {exc}")

    if errors:
        print(f"\n  WARNING: {len(errors)} module(s) failed: {list(errors.keys())}")

    total_elapsed = (datetime.now() - t_start).seconds
    print(f"\n  All modules done in {total_elapsed}s total.")

    # ------------------------------------------------------------------
    # Step 3: Collate pass/fail, write summary
    # ------------------------------------------------------------------
    _collate_pass_fail(summary)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(output_dir) / f"summary_{ts}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "="*60)
    print("FINAL PASS/FAIL SUMMARY")
    print("="*60)
    print(f"  Provider: {cfg.provider} / {cfg.model or '(default)'}  "
          f"Judge: {cfg._jp()} / {cfg._jm() or '(default)'}")

    all_pass = True
    for check, result in summary["pass_fail"].items():
        passed = result.get("pass")
        symbol = "PASS" if passed else ("FAIL" if passed is False else "N/A ")
        mean   = result.get("mean") or result.get("rate", "")
        thr    = result.get("threshold") or result.get("threshold_max", "")
        print(f"  [{symbol}] {check:40s} mean={mean}  threshold={thr}")
        if passed is False:
            all_pass = False

    overall = "ALL PASS" if all_pass else "SOME FAILURES — see details above"
    print(f"\n  Overall: {overall}")
    print(f"  Summary written to: {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if sys.platform == "win32":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    ALL_MODULES = [
        "ragas", "deepeval", "uptrain", "agentbench",
        "graph", "perf", "baseline", "ablation", "citation",
    ]

    parser = argparse.ArgumentParser(description="Run full evaluation suite (parallelised)")
    parser.add_argument("--quick",           action="store_true",
                        help="Run only graph + perf on q1_rag (fast sanity check)")
    parser.add_argument("--skip",            nargs="*", default=[])
    parser.add_argument("--query-ids",       nargs="*")
    parser.add_argument("--output-dir",      default="eval_results")
    parser.add_argument("--rerun-pipelines", action="store_true", dest="rerun_pipelines",
                        help="Ignore disk pipeline cache and re-run all pipelines")
    parser.add_argument("--clean",           action="store_true",
                        help="Delete pipeline cache, app query cache, and FAISS indices before running")
    parser.add_argument("--dry-run",         action="store_true", dest="dry_run",
                        help="With --clean: show what would be deleted without removing anything")
    add_provider_args(parser)
    args = parser.parse_args()

    if args.dry_run:
        args.clean = True

    if args.clean:
        label = "DRY RUN — " if args.dry_run else ""
        print(f"\n{'='*60}")
        print(f"{label}CLEANING EVAL STATE")
        print(f"{'='*60}")
        clean_eval_state(args.output_dir, dry_run=args.dry_run)
        if args.dry_run:
            raise SystemExit(0)
        print()

    cfg = cfg_from_args(args)

    if cfg.provider != "local" and not cfg.api_key:
        raise SystemExit(
            f"No API key for provider '{cfg.provider}'. "
            "Set the appropriate environment variable or pass --api-key."
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