"""
eval_ablation.py — Multi-configuration ablation study.

Runs all 4 test queries under 5 configurations and writes a comparison CSV:
  - full:         Full system (Critic active, all 3 channels)
  - no_critic:    Critic bypassed (loop_count always 0)
  - vector_only:  Router forced to activate only vector_db
  - web_only:     Router forced to activate only web
  - baseline:     Single LLM call, no retrieval

For each configuration x query, reports:
  - node_count, source_diversity, entity_recall (from eval_graph)
  - loop_count, cap_hit
  - wall_time_s measured by this script, plus total_tokens and cost_usd
    estimated via count_tokens() over concatenated text and estimate_cost()

RAGAS metrics are NOT run in ablation (too slow / expensive for 20 runs).
Run eval_ragas.py separately on full and baseline only.

Usage:
    # OpenAI (default)
    python eval_ablation.py

    # Local llama.cpp
    python eval_ablation.py --provider local --model mistral --base-url http://localhost:8080/v1
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

from test_queries import TEST_QUERIES, run_pipeline
from eval_graph import compute_graph_metrics
from eval_baseline import run_baseline_query
from eval_provider import EvalConfig, add_provider_args, cfg_from_args, count_tokens, estimate_cost


ABLATION_CONFIGS = [
    {
        "label":          "full",
        "description":    "Full system: Critic active, all 3 channels",
        "force_channels": None,
        "disable_critic": False,
        "is_baseline":    False,
    },
    {
        "label":          "no_critic",
        "description":    "Critic bypassed (no enrichment loops)",
        "force_channels": None,
        "disable_critic": True,
        "is_baseline":    False,
    },
    {
        "label":          "vector_only",
        "description":    "Router forced: vector_db only",
        "force_channels": ["vector_db"],
        "disable_critic": False,
        "is_baseline":    False,
    },
    {
        "label":          "web_only",
        "description":    "Router forced: web only",
        "force_channels": ["web"],
        "disable_critic": False,
        "is_baseline":    False,
    },
    {
        "label":          "baseline",
        "description":    "Single LLM call, no retrieval",
        "force_channels": None,
        "disable_critic": False,
        "is_baseline":    True,
    },
]


def run_ablation(
    output_dir: str = "eval_results",
    cfg: EvalConfig | None = None,
    # Legacy kwarg kept for backwards compatibility
    api_key: str | None = None,
) -> list[dict]:
    if cfg is None:
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("Pass an EvalConfig or set OPENAI_API_KEY.")
        cfg = EvalConfig(provider="openai", api_key=key)

    pricing = cfg.pricing()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for config in ABLATION_CONFIGS:
        print(f"\n{'='*60}")
        print(f"[Ablation] Config: {config['label']} — {config['description']}")
        print(f"{'='*60}")

        for tq in TEST_QUERIES:
            print(f"  Query: {tq['id']}...")

            if config["is_baseline"]:
                t0    = time.perf_counter()
                state = run_baseline_query(tq["query"], cfg=cfg)
                wall  = time.perf_counter() - t0
                tokens = state.get("_in_tokens", 0) + state.get("_out_tokens", 0)
                cost   = state.get("_cost_usd", 0.0)
                loop_count = 0
                cap_hit    = False
            else:
                state, wall = run_pipeline(
                    tq["query"],
                    cfg=cfg,
                    force_channels=config["force_channels"],
                    disable_critic=config["disable_critic"],
                )
                ctx_text = (
                    state.get("merged_context", "") +
                    state.get("summary", "") +
                    state.get("query", "")
                )
                tokens     = count_tokens(ctx_text)
                cost       = estimate_cost(tokens, pricing)
                loop_count = state.get("loop_count", 0)
                cap_hit    = loop_count >= 2

            graph_metrics = compute_graph_metrics(
                state.get("knowledge_map", {}),
                state.get("merged_context", ""),
            )

            row = {
                "config":           config["label"],
                "query_id":         tq["id"],
                "difficulty":       tq["difficulty"],
                "provider":         cfg.provider,
                "model":            cfg.model,
                "wall_time_s":      round(wall, 2),
                "tokens_approx":    tokens,
                "cost_usd":         cost,
                "loop_count":       loop_count,
                "cap_hit":          cap_hit,
                "node_count":       graph_metrics["node_count"],
                "edge_count":       graph_metrics["edge_count"],
                "source_diversity": graph_metrics["source_diversity"],
                "entity_recall":    graph_metrics["entity_recall"],
                "orphan_rate":      graph_metrics["orphan_rate"],
                "type_coverage":    graph_metrics["type_coverage"],
            }
            rows.append(row)

            print(f"    nodes={row['node_count']}  diversity={row['source_diversity']}  "
                  f"recall={row['entity_recall']}  wall={row['wall_time_s']}s  "
                  f"cost=${row['cost_usd']}")

    # Write JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = Path(output_dir) / f"ablation_{ts}.json"
    json_path.write_text(json.dumps(rows, indent=2))

    # Write CSV for easy comparison
    csv_path = Path(output_dir) / f"ablation_{ts}.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n[Ablation] JSON: {json_path}")
    print(f"[Ablation] CSV:  {csv_path}")

    # Print summary table
    print("\n=== Ablation Summary (means across 4 queries) ===")
    print(f"{'Config':15s} {'Nodes':>7} {'Diversity':>10} {'Recall':>8} "
          f"{'Wall(s)':>9} {'Cost($)':>10} {'Cap%':>7}")
    for config in ABLATION_CONFIGS:
        cfg_rows = [r for r in rows if r["config"] == config["label"]]
        if not cfg_rows:
            continue
        def mean(key): return round(sum(r[key] for r in cfg_rows) / len(cfg_rows), 3)
        cap_pct = round(sum(1 for r in cfg_rows if r["cap_hit"]) / len(cfg_rows) * 100)
        print(f"  {config['label']:13s} {mean('node_count'):>7.1f} {mean('source_diversity'):>10.3f} "
              f"{mean('entity_recall'):>8.3f} {mean('wall_time_s'):>9.1f} "
              f"{mean('cost_usd'):>10.5f} {cap_pct:>6}%")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--output-dir", default="eval_results")
    add_provider_args(parser)
    args = parser.parse_args()
    run_ablation(
        output_dir=args.output_dir,
        cfg=cfg_from_args(args),
    )
