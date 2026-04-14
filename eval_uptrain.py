"""
eval_uptrain.py — UpTrain-based evaluation of response and retrieval quality.

Metrics:
  - ResponseRelevance:      How well does the summary answer the query?
  - ResponseCompleteness:   Does the summary address all aspects of the query?
  - ContextRelevance:       Is the retrieved context on-topic for the query?
  - ResponseConciseness:    Is the summary free of irrelevant padding?

UpTrain scores each metric 0–1 (higher is better).  Thresholds mirror those
used by the RAGAS module so results are directly comparable.

Usage:
    # OpenAI judge (default)
    python eval_uptrain.py

    # Anthropic Claude judge
    python eval_uptrain.py --judge-provider claude

    # Local judge (requires a running OpenAI-compatible server)
    python eval_uptrain.py --judge-provider local --judge-base-url http://localhost:8080/v1

    # Subset of queries
    python eval_uptrain.py --query-ids q1_rag q2_federated

Requires:
    pip install uptrain
    # Plus provider packages; see requirements-eval.txt.
"""

from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from test_queries import TEST_QUERIES, run_pipeline
from eval_provider import EvalConfig, add_provider_args, cfg_from_args

try:
    import uptrain
    from uptrain import EvalLLM, Settings
    from uptrain import Evals
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}\nRun: pip install uptrain rouge_score")


# ---------------------------------------------------------------------------
# Thresholds (aligned with RAGAS module for cross-framework comparability)
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "response_relevance":    0.75,
    "response_completeness": 0.70,
    "context_relevance":     0.65,
    "response_conciseness":  0.60,
}


# ---------------------------------------------------------------------------
# UpTrain Settings factory
# ---------------------------------------------------------------------------

def _make_uptrain_settings(cfg: EvalConfig) -> Settings:
    """
    Build an UpTrain Settings object from an EvalConfig.

    evaluate_locally=True tells UpTrain to call the LLM provider directly
    using the supplied API key, rather than routing traffic through UpTrain's
    hosted cloud service (which requires a separate account and times out
    without one).

    For --judge-provider local, UpTrain's OpenAI client is pointed at the
    base URL supplied via --judge-base-url so that evaluation calls go to the
    local OpenAI-compatible server instead of api.openai.com.
    """
    p = cfg._jp()
    k = cfg._jk()
    m = cfg._jm()
    base_url = cfg._jb()   # None unless --judge-base-url was set

    if p == "local":
        # Local OpenAI-compatible server — key is ignored by most servers.
        # Pass openai_api_base so UpTrain routes requests to the local endpoint.
        kwargs: dict = dict(openai_api_key="EMPTY", evaluate_locally=True, model=m)
        if base_url:
            kwargs["openai_api_base"] = base_url
        return Settings(**kwargs)

    # openai / gemini / claude — all evaluated locally via the OpenAI client
    # that UpTrain bundles.  Non-OpenAI providers work if their API is
    # OpenAI-compatible; otherwise scores will be None (caught in _evaluate_query).
    return Settings(openai_api_key=k, evaluate_locally=True, model=m)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _normalise_score(raw) -> float | None:
    """Normalise an UpTrain score to the 0–1 range."""
    try:
        f = float(raw)
        # UpTrain scores are already 0–1, but some versions return 0–10.
        if f > 1.0:
            f = f / 10.0
        return round(min(max(f, 0.0), 1.0), 4)
    except (TypeError, ValueError):
        return None


def _evaluate_query(
    query: str,
    context: str,
    summary: str,
    eval_llm: "EvalLLM",
) -> dict:
    """
    Run all four UpTrain metrics for a single query.

    Returns a dict with keys matching THRESHOLDS.
    """
    # UpTrain expects context as a plain string.  We pass the full merged_context
    # (up to UpTrain's token limit) so the evaluation matches what the pipeline saw.
    data = [{
        "question": query,
        "context":  context[:8000],   # cap to avoid token-limit errors
        "response": summary,
    }]

    checks = [
        Evals.RESPONSE_RELEVANCE,
        Evals.RESPONSE_COMPLETENESS,
        Evals.CONTEXT_RELEVANCE,
        Evals.RESPONSE_CONCISENESS,
    ]

    try:
        results = eval_llm.evaluate(data=data, checks=checks)
        row = results[0] if results else {}
    except Exception as exc:
        print(f"    WARNING: UpTrain evaluation failed: {exc}")
        return {k: None for k in THRESHOLDS}

    # UpTrain key names vary slightly across versions; try both forms.
    def _get(keys: list[str]):
        for key in keys:
            if key in row:
                return _normalise_score(row[key])
        return None

    return {
        "response_relevance":    _get(["score_response_relevance",    "response_relevance"]),
        "response_completeness": _get(["score_response_completeness", "response_completeness"]),
        "context_relevance":     _get(["score_context_relevance",     "context_relevance"]),
        "response_conciseness":  _get(["score_response_conciseness",  "response_conciseness"]),
    }


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def _print_header() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    UpTrain Evaluation Framework                  ║
╠══════════════════════════════════════════════════════════════════╣
║  What it is:                                                     ║
║    UpTrain is an open-source LLM evaluation framework that uses  ║
║    a judge LLM to score response and retrieval quality. All      ║
║    evaluation is done locally (evaluate_locally=True) — traffic  ║
║    is not routed through UpTrain's cloud service.                ║
║                                                                  ║
║  Metrics (all scored 0.0 – 1.0, higher is better):              ║
║    response_relevance    Does the summary directly address the   ║
║                          research query?                         ║
║    response_completeness Does the summary cover all aspects of   ║
║                          the query, or leave gaps?               ║
║    context_relevance     Is the retrieved context on-topic for   ║
║                          the query (low noise)?                  ║
║    response_conciseness  Is the summary free of irrelevant       ║
║                          padding or off-topic content?           ║
║                                                                  ║
║  PASS thresholds (aligned with RAGAS for cross-framework         ║
║  comparability):                                                 ║
║    response_relevance ≥ 0.75  |  response_completeness ≥ 0.70   ║
║    context_relevance ≥ 0.65   |  response_conciseness ≥ 0.60    ║
║                                                                  ║
║  How to read the output:                                         ║
║    Each query prints a score and PASS/FAIL for every metric.     ║
║    The aggregate section shows the mean score and pass rate      ║
║    across all queries, with the threshold listed for reference.  ║
║    A score of None means the pipeline returned an empty summary  ║
║    or context for that query and evaluation was skipped.         ║
╚══════════════════════════════════════════════════════════════════╝
""")


def run_uptrain_eval(
    query_ids: list[str] | None = None,
    output_dir: str = "eval_results",
    cfg: EvalConfig | None = None,
    # Pre-computed pipeline results: maps query_id -> (state, wall_time).
    pipeline_cache: dict | None = None,
    # Legacy kwarg kept for backwards compatibility
    api_key: str | None = None,
) -> list[dict]:
    _print_header()

    if cfg is None:
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("Pass an EvalConfig or set OPENAI_API_KEY.")
        cfg = EvalConfig(provider="openai", api_key=key)

    queries = [q for q in TEST_QUERIES if query_ids is None or q["id"] in query_ids]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    settings  = _make_uptrain_settings(cfg)
    eval_llm  = EvalLLM(settings)
    all_results: list[dict] = []

    for tq in queries:
        if pipeline_cache and tq["id"] in pipeline_cache:
            state, wall_time = pipeline_cache[tq["id"]]
            print(f"\n[UpTrain] Reusing cached pipeline: {tq['id']}")
        else:
            print(f"\n[UpTrain] Running: {tq['id']} — {tq['query'][:60]}...")
            state, wall_time = run_pipeline(tq["query"], cfg=cfg)

        summary  = state.get("summary", "")
        context  = state.get("merged_context", "")

        if summary and context:
            scores = _evaluate_query(tq["query"], context, summary, eval_llm)
        else:
            print(f"    WARNING: empty summary or context for {tq['id']}, skipping UpTrain.")
            scores = {k: None for k in THRESHOLDS}

        result = {
            "query_id":          tq["id"],
            "query":             tq["query"],
            "difficulty":        tq["difficulty"],
            "provider":          cfg.provider,
            "model":             cfg.model,
            "judge_provider":    cfg._jp(),
            "judge_model":       cfg._jm(),
            "wall_time_s":       round(wall_time, 2),
            "active_agents":     state.get("active_agents", []),
            "loop_count":        state.get("loop_count", 0),
            **scores,
        }

        # Pass / fail flags
        for metric, threshold in THRESHOLDS.items():
            result[f"{metric}_pass"] = (result.get(metric) or 0) >= threshold

        all_results.append(result)

        print(f"  Provider:              {cfg.provider} / {cfg.model}")
        print(f"  Judge:                 {cfg._jp()} / {cfg._jm()}")
        for metric in THRESHOLDS:
            score  = result.get(metric)
            passed = result.get(f"{metric}_pass")
            print(f"  {metric:30s} {score}  ({'PASS' if passed else 'FAIL'})")

    # Write output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"uptrain_{ts}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[UpTrain] Results written to {out_path}")

    # Aggregate summary
    print("\n=== UpTrain AGGREGATE ===")
    for metric, threshold in THRESHOLDS.items():
        vals   = [r[metric] for r in all_results if r.get(metric) is not None]
        passes = sum(1 for r in all_results if r.get(f"{metric}_pass"))
        if vals:
            mean = round(sum(vals) / len(vals), 4)
            print(f"  {metric:30s} mean={mean}  pass={passes}/{len(all_results)}"
                  f"  threshold={threshold}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UpTrain agent evaluation")
    parser.add_argument("--query-ids",  nargs="*")
    parser.add_argument("--output-dir", default="eval_results")
    add_provider_args(parser)
    args = parser.parse_args()
    run_uptrain_eval(
        query_ids=args.query_ids,
        output_dir=args.output_dir,
        cfg=cfg_from_args(args),
    )
