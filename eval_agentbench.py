"""
eval_agentbench.py — AgentBench-inspired structured task evaluation.

Adapts the AgentBench evaluation philosophy (task success rate, capability
across task categories, and interaction efficiency) to the multi-agent
research pipeline.  No external AgentBench installation is required; the
metrics are implemented here using the judge LLM already available via
eval_provider.py.

Metrics (all 0–1, higher is better):
  - task_success:      Did the agent produce a substantive, on-topic answer?
                       Judged by the judge LLM using a structured rubric.
  - channel_precision: Fraction of activated channels that were expected.
  - channel_recall:    Fraction of expected channels that were activated.
  - channel_f1:        Harmonic mean of precision and recall.
  - iteration_efficiency: 1 - (loop_count / max_loops).  Agents that approve
                           on the first pass score 1.0; agents that always hit
                           the loop cap score 0.5.
  - kg_density:        Normalised knowledge-graph node count.
                       Score = 1.0 when node_count is in the ideal range
                       [8, 25]; tapers linearly outside it.

The task_success score is the primary metric; the rest are structural metrics
that do not require LLM calls and are always available.

Usage:
    # OpenAI judge (default — used only for task_success scoring)
    python eval_agentbench.py

    # Anthropic Claude judge
    python eval_agentbench.py --judge-provider claude

    # Skip LLM judging (structural metrics only, fast, no API cost)
    python eval_agentbench.py --no-llm-judge

    # Subset of queries
    python eval_agentbench.py --query-ids q1_rag q2_federated

Requires:
    No additional packages beyond the core requirements.
"""

from __future__ import annotations
import argparse
import json
import os
import textwrap
from datetime import datetime
from pathlib import Path

from test_queries import TEST_QUERIES, run_pipeline
from eval_provider import EvalConfig, add_provider_args, cfg_from_args


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LOOPS        = 2      # pipeline loop cap (matches graph.py setting)
IDEAL_NODES_MIN  = 8      # from run_eval.py PASS_THRESHOLDS
IDEAL_NODES_MAX  = 25

THRESHOLDS = {
    "task_success":          0.70,
    "channel_f1":            0.75,
    "iteration_efficiency":  0.70,
    "kg_density":            0.60,
}

# Structured rubric prompt for task_success judging
_JUDGE_PROMPT = textwrap.dedent("""\
You are evaluating the output of a multi-agent research assistant.

QUERY:
{query}

AGENT RESPONSE:
{response}

Score the response on the following criteria, then output a single JSON object.

Criteria (score each 0.0 – 1.0):
1. relevance:    Does the response directly address the query?
2. depth:        Does the response go beyond surface-level facts and
                 synthesise information across sources or themes?
3. accuracy:     Are the claims plausible and internally consistent?
                 (You cannot verify external facts; penalise obvious errors.)
4. organisation: Is the response clearly structured (e.g. by theme)?

Output format — return ONLY valid JSON, no commentary:
{{"relevance": <float>, "depth": <float>, "accuracy": <float>, "organisation": <float>}}
""")


# ---------------------------------------------------------------------------
# Structural metrics (no LLM required)
# ---------------------------------------------------------------------------

def _channel_scores(active: list[str], expected: list[str]) -> dict:
    if not expected:
        return {"channel_precision": None, "channel_recall": None, "channel_f1": None}

    active_set   = set(active)
    expected_set = set(expected)

    tp        = len(active_set & expected_set)
    precision = tp / len(active_set)   if active_set   else 0.0
    recall    = tp / len(expected_set) if expected_set else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if precision + recall > 0 else 0.0)

    return {
        "channel_precision": round(precision, 4),
        "channel_recall":    round(recall, 4),
        "channel_f1":        round(f1, 4),
    }


def _iteration_efficiency(loop_count: int, max_loops: int = MAX_LOOPS) -> float:
    """
    Score agent efficiency based on number of critic-triggered enrichment loops.

    loop_count=0  → 1.00  (approved on first pass; ideal)
    loop_count=1  → 0.75
    loop_count=2+ → 0.50  (hit or approached the cap)
    """
    if loop_count == 0:
        return 1.00
    if loop_count >= max_loops:
        return 0.50
    return round(1.0 - 0.25 * loop_count, 4)


def _kg_density(node_count: int,
                ideal_min: int = IDEAL_NODES_MIN,
                ideal_max: int = IDEAL_NODES_MAX) -> float:
    """
    Normalise node count to [0, 1].

    Score = 1.0 when node_count is in [ideal_min, ideal_max].
    Tapers linearly to 0.0 below ideal_min or above 2×ideal_max.
    """
    if node_count == 0:
        return 0.0
    if ideal_min <= node_count <= ideal_max:
        return 1.0
    if node_count < ideal_min:
        return round(node_count / ideal_min, 4)
    # above ideal_max: taper to 0 at 2×ideal_max
    ceiling = 2 * ideal_max
    if node_count >= ceiling:
        return 0.0
    return round(1.0 - (node_count - ideal_max) / ideal_max, 4)


# ---------------------------------------------------------------------------
# LLM-based task_success judging
# ---------------------------------------------------------------------------

def _judge_task_success(
    query: str,
    response: str,
    cfg: EvalConfig,
) -> float | None:
    """
    Ask the judge LLM to score the agent's response using the structured rubric.

    Returns the arithmetic mean of the four rubric dimensions, or None on failure.
    """
    import json as _json
    import re

    prompt = _JUDGE_PROMPT.format(query=query, response=response[:4000])
    lm = cfg.judge_langchain_llm(temperature=0.0)

    try:
        from langchain_core.messages import HumanMessage
        raw = lm.invoke([HumanMessage(content=prompt)]).content or ""
    except Exception as exc:
        print(f"    WARNING: judge LLM call failed: {exc}")
        return None

    # Extract JSON from the response; the LLM may wrap it in markdown code fences.
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if not match:
        print(f"    WARNING: judge returned no JSON.  Raw: {raw[:200]}")
        return None

    try:
        scores = _json.loads(match.group())
    except _json.JSONDecodeError as exc:
        print(f"    WARNING: JSON parse error: {exc}  Raw: {match.group()[:200]}")
        return None

    dims = ["relevance", "depth", "accuracy", "organisation"]
    vals = []
    for d in dims:
        try:
            v = float(scores.get(d, 0))
            vals.append(min(max(v, 0.0), 1.0))
        except (TypeError, ValueError):
            pass

    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def _print_header() -> None:
    print("""\
╔══════════════════════════════════════════════════════════════════╗
║                  AgentBench Evaluation Framework                 ║
╠══════════════════════════════════════════════════════════════════╣
║  What it is:                                                     ║
║    An adaptation of the AgentBench benchmark (Liu et al., 2023)  ║
║    that measures task success rate, routing capability, and      ║
║    interaction efficiency for multi-agent research pipelines.    ║
║    No external AgentBench installation is required; metrics are  ║
║    computed locally using the judge LLM from eval_provider.py.   ║
║                                                                  ║
║  Metrics (all scored 0.0 – 1.0, higher is better):              ║
║    task_success         LLM judge rates relevance, depth,        ║
║                         accuracy, and organisation of the        ║
║                         summary (mean of four 0–1 rubric dims).  ║
║    channel_precision    Fraction of activated channels that were ║
║                         expected for this query type.            ║
║    channel_recall       Fraction of expected channels that were  ║
║                         actually activated by the Router.        ║
║    channel_f1           Harmonic mean of precision and recall.   ║
║    iteration_efficiency 1.0 = approved on first critic pass;     ║
║                         0.75 = one enrichment loop; 0.5 = hit    ║
║                         the loop cap (penalises over-iteration). ║
║    kg_density           Normalised knowledge-graph node count:   ║
║                         1.0 in ideal range [8, 25]; tapers to    ║
║                         0.0 below 8 or above 50 nodes.           ║
║                                                                  ║
║  How to read the output:                                         ║
║    Each query shows per-metric scores with a PASS/FAIL flag.     ║
║    The aggregate table breaks results down by difficulty bucket  ║
║    (easy / medium / hard), mirroring the original AgentBench     ║
║    per-task-category reporting style.                            ║
║    task_success is None when --no-llm-judge is used.            ║
╚══════════════════════════════════════════════════════════════════╝""")
    print("  PASS thresholds (≥ to PASS):")
    for metric, threshold in THRESHOLDS.items():
        print(f"    {metric}: {threshold}")


def run_agentbench_eval(
    query_ids: list[str] | None = None,
    output_dir: str = "eval_results",
    cfg: EvalConfig | None = None,
    no_llm_judge: bool = False,
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
    all_results: list[dict] = []

    for tq in queries:
        if pipeline_cache and tq["id"] in pipeline_cache:
            state, wall_time = pipeline_cache[tq["id"]]
            print(f"\n[AgentBench] Reusing cached pipeline: {tq['id']}")
        else:
            print(f"\n[AgentBench] Running: {tq['id']} — {tq['query'][:60]}...")
            state, wall_time = run_pipeline(tq["query"], cfg=cfg)

        summary    = state.get("summary", "")
        loop_count = state.get("loop_count", 0)
        active     = state.get("active_agents", [])
        node_count = len(state.get("knowledge_map", {}).get("nodes", []))

        # Structural metrics (no LLM)
        ch_scores    = _channel_scores(active, tq["expected_channels"])
        iter_eff     = _iteration_efficiency(loop_count)
        kg_dens      = _kg_density(node_count)

        # LLM task_success (skippable)
        if no_llm_judge or not summary:
            task_success = None
        else:
            print(f"  Judging task_success with {cfg._jp()} / {cfg._jm()}...")
            task_success = _judge_task_success(tq["query"], summary, cfg)

        result = {
            "query_id":             tq["id"],
            "query":                tq["query"],
            "difficulty":           tq["difficulty"],
            "provider":             cfg.provider,
            "model":                cfg.model,
            "judge_provider":       cfg._jp(),
            "judge_model":          cfg._jm(),
            "wall_time_s":          round(wall_time, 2),
            "active_agents":        active,
            "expected_channels":    tq["expected_channels"],
            "loop_count":           loop_count,
            "node_count":           node_count,
            # Core metrics
            "task_success":         task_success,
            "iteration_efficiency": iter_eff,
            "kg_density":           kg_dens,
            **ch_scores,
        }

        # Pass / fail flags
        for metric, threshold in THRESHOLDS.items():
            val = result.get(metric)
            result[f"{metric}_pass"] = None if val is None else val >= threshold

        all_results.append(result)

        print(f"  Provider:             {cfg.provider} / {cfg.model}")
        print(f"  Active channels:      {active}  (expected: {tq['expected_channels']})")
        print(f"  Channel F1:           {ch_scores['channel_f1']}")
        print(f"  Iteration efficiency: {iter_eff}  (loop_count={loop_count})")
        print(f"  KG density:           {kg_dens}  (nodes={node_count})")
        if task_success is not None:
            ts_pass = result["task_success_pass"]
            print(f"  Task success:         {task_success}  "
                  f"({'PASS' if ts_pass else 'FAIL'})")

    # Write output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"agentbench_{ts}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[AgentBench] Results written to {out_path}")

    # Aggregate summary (per-difficulty breakdown like the original AgentBench paper)
    print("\n=== AgentBench AGGREGATE ===")
    difficulties = sorted({r["difficulty"] for r in all_results})
    print(f"  {'Metric':30s} {'Overall':>10}", end="")
    for d in difficulties:
        print(f"  {d:>10}", end="")
    print()

    for metric, threshold in THRESHOLDS.items():
        overall_vals = [r[metric] for r in all_results if r.get(metric) is not None]
        overall_mean = (round(sum(overall_vals) / len(overall_vals), 4)
                        if overall_vals else None)
        print(f"  {metric:30s} {str(overall_mean):>10}", end="")
        for d in difficulties:
            d_vals = [r[metric] for r in all_results
                      if r["difficulty"] == d and r.get(metric) is not None]
            d_mean = round(sum(d_vals) / len(d_vals), 4) if d_vals else None
            print(f"  {str(d_mean):>10}", end="")
        print()

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AgentBench-inspired evaluation")
    parser.add_argument("--query-ids",    nargs="*")
    parser.add_argument("--output-dir",   default="eval_results")
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip LLM-based task_success scoring (structural metrics only, no API cost)",
    )
    add_provider_args(parser)
    args = parser.parse_args()
    run_agentbench_eval(
        query_ids=args.query_ids,
        output_dir=args.output_dir,
        cfg=cfg_from_args(args),
        no_llm_judge=args.no_llm_judge,
    )
