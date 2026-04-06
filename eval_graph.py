"""
eval_graph.py — Knowledge graph quality evaluation.

Metrics (all computed without external API calls):
  - node_count:       Number of nodes. Target: 8-25.
  - edge_count:       Number of edges.
  - source_diversity: Fraction of nodes from more than one source channel.
  - type_coverage:    Fraction of node types present (concept/entity/fact/process).
  - edge_density:     edges / (nodes * (nodes-1)) — measures graph connectivity.
  - entity_recall:    Fraction of KG node labels appearing in merged_context.
  - orphan_rate:      Fraction of nodes with no edges (isolated nodes).
  - contradiction_edges: Count of pairs where A->B and B->A with different relations.

Usage:
    python eval_graph.py [--query-ids q1_rag] [--output-dir eval_results]

No additional pip installs needed beyond the main project requirements.
"""

from __future__ import annotations
import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

from test_queries import TEST_QUERIES, run_pipeline
from eval_provider import EvalConfig, add_provider_args, cfg_from_args


# ---------------------------------------------------------------------------
# Graph metrics (no LLM calls)
# ---------------------------------------------------------------------------

def compute_graph_metrics(knowledge_map: dict, merged_context: str) -> dict:
    nodes = knowledge_map.get("nodes", [])
    edges = knowledge_map.get("edges", [])
    n = len(nodes)
    e = len(edges)

    if n == 0:
        return {
            "node_count":          0,
            "edge_count":          0,
            "source_diversity":    0.0,
            "type_coverage":       0.0,
            "edge_density":        0.0,
            "entity_recall":       0.0,
            "orphan_rate":         0.0,
            "contradiction_edges": 0,
            "node_count_pass":     False,
            "source_diversity_pass": False,
            "entity_recall_pass":  False,
        }

    # Source diversity: fraction of nodes not all from the same source
    sources = [nd.get("source", "unknown") for nd in nodes]
    unique_sources = set(sources)
    source_diversity = round(1.0 - (max(sources.count(s) for s in unique_sources) / n), 4)

    # Type coverage: how many of the 4 expected types are represented
    expected_types = {"concept", "entity", "fact", "process"}
    present_types  = {nd.get("type", "").lower() for nd in nodes}
    type_coverage  = round(len(present_types & expected_types) / len(expected_types), 4)

    # Edge density
    max_edges     = n * (n - 1) if n > 1 else 1
    edge_density  = round(e / max_edges, 4)

    # Entity recall: fraction of node labels appearing in context.
    # Use whole-word matching to avoid substring false positives
    # (e.g. label "tent" matching inside "attention").
    ctx_lower = merged_context.lower()
    found = sum(
        1 for nd in nodes
        if nd.get("label", "")
        and re.search(
            r'\b' + re.escape(nd["label"].lower()) + r'\b',
            ctx_lower,
        )
    )
    entity_recall = round(found / n, 4)

    # Orphan rate: nodes not referenced in any edge
    connected_ids = set()
    for ed in edges:
        connected_ids.add(ed.get("source", ""))
        connected_ids.add(ed.get("target", ""))
    node_ids    = {nd.get("id", nd.get("label", "")) for nd in nodes}
    orphans     = node_ids - connected_ids
    orphan_rate = round(len(orphans) / n, 4)

    # Contradiction edges: A->B and B->A with different relation labels
    edge_pairs: dict[tuple, set] = {}
    for ed in edges:
        pair = (ed.get("source", ""), ed.get("target", ""))
        rev  = (pair[1], pair[0])
        if rev in edge_pairs:
            edge_pairs[rev].add(ed.get("relation", ""))
    contradictions = sum(1 for _, rels in edge_pairs.items() if len(rels) > 1)

    return {
        "node_count":             n,
        "edge_count":             e,
        "source_diversity":       source_diversity,
        "type_coverage":          type_coverage,
        "edge_density":           edge_density,
        "entity_recall":          entity_recall,
        "orphan_rate":            orphan_rate,
        "contradiction_edges":    contradictions,
        "unique_sources":         sorted(unique_sources),
        # Pass/fail
        "node_count_pass":        8 <= n <= 25,
        "source_diversity_pass":  source_diversity >= 0.3,
        "entity_recall_pass":     entity_recall >= 0.6,
        "orphan_rate_pass":       orphan_rate <= 0.25,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_graph_eval(
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

    queries = [q for q in TEST_QUERIES if query_ids is None or q["id"] in query_ids]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    for tq in queries:
        if pipeline_cache and tq["id"] in pipeline_cache:
            state, wall_time = pipeline_cache[tq["id"]]
            print(f"\n[Graph] Reusing cached pipeline: {tq['id']}")
        else:
            print(f"\n[Graph] Running: {tq['id']} — {tq['query'][:60]}...")
            state, wall_time = run_pipeline(tq["query"], cfg=cfg)

        metrics = compute_graph_metrics(
            state.get("knowledge_map", {}),
            state.get("merged_context", ""),
        )

        result = {
            "query_id":    tq["id"],
            "query":       tq["query"],
            "difficulty":  tq["difficulty"],
            "provider":    cfg.provider,
            "model":       cfg.model,
            "wall_time_s": round(wall_time, 2),
            "loop_count":  state.get("loop_count", 0),
            **metrics,
        }
        all_results.append(result)

        print(f"  Provider:          {cfg.provider} / {cfg.model}")
        print(f"  Nodes:             {result['node_count']} ({'PASS' if result['node_count_pass'] else 'FAIL'})")
        print(f"  Edges:             {result['edge_count']}")
        print(f"  Source diversity:  {result['source_diversity']} ({'PASS' if result['source_diversity_pass'] else 'FAIL'})")
        print(f"  Entity recall:     {result['entity_recall']} ({'PASS' if result['entity_recall_pass'] else 'FAIL'})")
        print(f"  Orphan rate:       {result['orphan_rate']} ({'PASS' if result['orphan_rate_pass'] else 'FAIL'})")
        print(f"  Sources used:      {result['unique_sources']}")
        print(f"  Type coverage:     {result['type_coverage']}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"graph_{ts}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[Graph] Results written to {out_path}")

    print("\n=== Graph AGGREGATE ===")
    for m, label in [
        ("node_count",       "Avg node count"),
        ("source_diversity", "Avg source diversity"),
        ("entity_recall",    "Avg entity recall"),
        ("orphan_rate",      "Avg orphan rate"),
        ("type_coverage",    "Avg type coverage"),
    ]:
        vals = [r[m] for r in all_results if isinstance(r.get(m), (int, float))]
        if vals:
            print(f"  {label:30s} {round(sum(vals)/len(vals), 4)}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run knowledge graph evaluation")
    parser.add_argument("--query-ids", nargs="*")
    parser.add_argument("--output-dir", default="eval_results")
    add_provider_args(parser)
    args = parser.parse_args()
    run_graph_eval(
        query_ids=args.query_ids,
        output_dir=args.output_dir,
        cfg=cfg_from_args(args),
    )
