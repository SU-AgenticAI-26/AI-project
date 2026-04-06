"""
eval_deepeval.py — Agent-level evaluation using DeepEval.

Metrics:
  - ToolCorrectnessMetric:  Did the Router activate the expected retrieval channels?
  - GEval (ThematicCoherence): Is the summary a cross-paper thematic analysis
                               rather than a paper-by-paper list?

Usage:
    # OpenAI (default)
    python eval_deepeval.py

    # Local llama.cpp pipeline, judge also local
    python eval_deepeval.py --provider local --model mistral --base-url http://localhost:8080/v1

    # Local pipeline, OpenAI judge (recommended: GEval quality depends heavily on judge)
    python eval_deepeval.py --provider local --model mistral \\
                            --judge-provider openai --judge-model gpt-4o-mini

Requires:
    pip install deepeval
    # Plus provider packages; see streamlit_app.py install notes.
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
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall
    from deepeval.metrics import ToolCorrectnessMetric, GEval
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}\nRun: pip install deepeval")


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

def make_thematic_coherence_metric(cfg: EvalConfig) -> GEval:
    """
    Custom GEval metric for thematic synthesis quality.

    Evaluates whether the summary integrates findings across papers
    into themes rather than summarising sources one at a time.
    """
    return GEval(
        name="ThematicCoherence",
        evaluation_steps=[
            "Check whether the summary organises findings by research theme "
            "or concept (e.g., 'approaches', 'limitations', 'open questions') "
            "rather than by source or paper.",
            "Check whether the summary identifies at least one point of "
            "disagreement, contradiction, or tension between the sources.",
            "Penalise summaries that simply list what each source (Vector DB / "
            "SQL / Web) said without synthesising across them.",
            "Check whether each thematic claim cites or references a source "
            "channel or specific finding.",
            "A score of 1.0 means fully thematic with identified contradictions. "
            "A score of 0.0 means pure listing with no integration.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.65,
        model=cfg.deepeval_model(),
        include_reason=True,
    )


def make_router_metric() -> ToolCorrectnessMetric:
    return ToolCorrectnessMetric(
        threshold=0.8,
        include_reason=True,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_deepeval_eval(
    query_ids: list[str] | None = None,
    output_dir: str = "eval_results",
    cfg: EvalConfig | None = None,
    # Pre-computed pipeline results: maps query_id -> (state, wall_time).
    pipeline_cache: dict | None = None,
    # Legacy kwarg kept for backwards compatibility
    api_key: str | None = None,
) -> list[dict]:
    if cfg is None:
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("Pass an EvalConfig or set OPENAI_API_KEY.")
        cfg = EvalConfig(provider="openai", api_key=key)

    # DeepEval reads OPENAI_API_KEY from env when using OpenAI models directly.
    # For non-OpenAI judges we pass the model object, so no env var needed.
    if cfg._jp() == "openai":
        os.environ["OPENAI_API_KEY"] = cfg._jk()

    queries = [q for q in TEST_QUERIES if query_ids is None or q["id"] in query_ids]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    coherence_metric = make_thematic_coherence_metric(cfg)
    router_metric    = make_router_metric()

    for tq in queries:
        if pipeline_cache and tq["id"] in pipeline_cache:
            state, wall_time = pipeline_cache[tq["id"]]
            print(f"\n[DeepEval] Reusing cached pipeline: {tq['id']}")
        else:
            print(f"\n[DeepEval] Running: {tq['id']} — {tq['query'][:60]}...")
            state, wall_time = run_pipeline(tq["query"], cfg=cfg)

        summary        = state.get("summary", "")
        merged_context = state.get("merged_context", "")
        active_agents  = state.get("active_agents", [])

        # --- Router correctness test ---
        router_test = LLMTestCase(
            input=tq["query"],
            actual_output=state.get("router_reasoning", ""),
            tools_called=[ToolCall(name=a) for a in active_agents],
            expected_tools=[ToolCall(name=c) for c in tq["expected_channels"]],
        )
        router_metric.measure(router_test)
        router_score  = round(router_metric.score, 4)
        router_reason = router_metric.reason

        # --- Thematic coherence test ---
        if summary and merged_context:
            coherence_test = LLMTestCase(
                input=tq["query"],
                actual_output=summary,
                retrieval_context=[merged_context],
            )
            coherence_metric.measure(coherence_test)
            coherence_score  = round(coherence_metric.score, 4)
            coherence_reason = coherence_metric.reason
        else:
            coherence_score  = None
            coherence_reason = "No summary or context available."

        result = {
            "query_id":          tq["id"],
            "query":             tq["query"],
            "difficulty":        tq["difficulty"],
            "provider":          cfg.provider,
            "model":             cfg.model,
            "judge_provider":    cfg._jp(),
            "judge_model":       cfg._jm(),
            "wall_time_s":       round(wall_time, 2),
            "active_agents":     active_agents,
            "expected_channels": tq["expected_channels"],
            "router_score":      router_score,
            "router_reason":     router_reason,
            "router_pass":       router_score >= 0.8,
            "coherence_score":   coherence_score,
            "coherence_reason":  coherence_reason,
            "coherence_pass":    (coherence_score or 0) >= 0.65,
            "loop_count":        state.get("loop_count", 0),
        }

        all_results.append(result)

        print(f"  Provider:        {cfg.provider} / {cfg.model}")
        print(f"  Judge:           {cfg._jp()} / {cfg._jm()}")
        print(f"  Router score:    {result['router_score']} ({'PASS' if result['router_pass'] else 'FAIL'})")
        print(f"  Router reason:   {result['router_reason']}")
        print(f"  Coherence score: {result['coherence_score']} ({'PASS' if result['coherence_pass'] else 'FAIL'})")
        print(f"  Coherence why:   {(result['coherence_reason'] or '')[:120]}")

    # Write output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"deepeval_{ts}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[DeepEval] Results written to {out_path}")

    # Aggregate summary
    print("\n=== DeepEval AGGREGATE ===")
    for metric, key_score, key_pass in [
        ("Router correctness", "router_score",    "router_pass"),
        ("Thematic coherence", "coherence_score", "coherence_pass"),
    ]:
        vals = [r[key_score] for r in all_results if r.get(key_score) is not None]
        if vals:
            mean   = round(sum(vals) / len(vals), 4)
            passes = sum(1 for r in all_results if r.get(key_pass))
            print(f"  {metric:30s} mean={mean}  pass={passes}/{len(all_results)}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepEval agent evaluation")
    parser.add_argument("--query-ids", nargs="*")
    parser.add_argument("--output-dir", default="eval_results")
    add_provider_args(parser)
    args = parser.parse_args()
    run_deepeval_eval(
        query_ids=args.query_ids,
        output_dir=args.output_dir,
        cfg=cfg_from_args(args),
    )
