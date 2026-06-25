"""
eval_citation.py — Semantic citation grounding evaluation.

Metrics:
  - citation_accuracy:  Fraction of cited claims that are semantically grounded
                        in the retrieved context. Pass: >= 0.75.
  - grounded_count:     Absolute count of grounded citations.
  - citation_count:     Total citations extracted from the summary.
  - mean_similarity:    Mean max cosine similarity across all citations.

The citation *scoring* is fully local — it uses sentence-transformers/all-MiniLM-L6-v2
(no external API calls). The pipeline run that produces the summary still calls
the configured LLM provider (OpenAI, Groq, etc.) via run_pipeline().

Uses sentence-transformers/all-MiniLM-L6-v2 (already in requirements.txt).
First run downloads ~80MB model from HuggingFace; subsequent runs use disk cache.

Usage:
    # OpenAI pipeline (default judge)
    python eval_citation.py

    # Local llama.cpp pipeline
    python eval_citation.py --provider local --model mistral --base-url http://localhost:8080/v1

    # Subset of queries
    python eval_citation.py --query-ids q1_rag q2_continual

    # Custom similarity threshold
    python eval_citation.py --threshold 0.70

No additional pip installs needed beyond the main project requirements.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from test_queries import TEST_QUERIES, run_pipeline
from eval_provider import EvalConfig, add_provider_args, cfg_from_args

_CITATION_ACCURACY_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Per-state metrics
# ---------------------------------------------------------------------------

def compute_citation_eval_metrics(state: dict, threshold: float = _CITATION_ACCURACY_THRESHOLD) -> dict:
    """
    Compute citation grounding metrics from a completed pipeline state.

    Returns a flat dict with: citation_accuracy, grounded_count, citation_count,
    mean_similarity, per_citation list, and citation_accuracy_pass bool.
    """
    from citation_verifier import compute_citation_metrics

    grounding_map, citation_accuracy = compute_citation_metrics(
        summary=state.get("summary", ""),
        extraction_findings=state.get("extraction_findings", ""),
        merged_context=state.get("merged_context", ""),
        vector_findings=state.get("vector_findings", ""),
        sql_findings=state.get("sql_findings", ""),
        web_findings=state.get("web_findings", ""),
        threshold=threshold,
    )

    citation_count  = len(grounding_map)
    grounded_count  = sum(1 for v in grounding_map.values() if v.get("grounded"))
    similarities    = [v.get("similarity", 0.0) for v in grounding_map.values()]
    mean_similarity = round(sum(similarities) / len(similarities), 4) if similarities else 0.0
    per_citation    = [{"citation": k, **v} for k, v in grounding_map.items()]

    return {
        "citation_accuracy":      round(citation_accuracy, 4),
        "grounded_count":         grounded_count,
        "citation_count":         citation_count,
        "mean_similarity":        mean_similarity,
        "per_citation":           per_citation,
        "citation_accuracy_pass": citation_accuracy >= threshold,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_citation_eval(
    query_ids: list[str] | None = None,
    output_dir: str = "eval_results",
    cfg: EvalConfig | None = None,
    pipeline_cache: dict | None = None,
    threshold: float = _CITATION_ACCURACY_THRESHOLD,
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
    all_results: list[dict] = []

    for tq in queries:
        if pipeline_cache and tq["id"] in pipeline_cache:
            state, wall_time = pipeline_cache[tq["id"]]
            print(f"\n[Citation] Reusing cached pipeline: {tq['id']}")
        else:
            print(f"\n[Citation] Running: {tq['id']} — {tq['query'][:60]}...")
            state, wall_time = run_pipeline(tq["query"], cfg=cfg)

        metrics = compute_citation_eval_metrics(state, threshold=threshold)

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

        status = "PASS" if metrics["citation_accuracy_pass"] else "FAIL"
        print(f"  Provider:          {cfg.provider} / {cfg.model}")
        print(f"  Citation accuracy: {metrics['citation_accuracy']} ({status}, threshold={threshold})")
        print(f"  Grounded:          {metrics['grounded_count']} / {metrics['citation_count']}")
        print(f"  Mean similarity:   {metrics['mean_similarity']}")
        if metrics["per_citation"]:
            ungrounded = [c["citation"][:60] for c in metrics["per_citation"] if not c["grounded"]]
            if ungrounded:
                print(f"  Ungrounded ({len(ungrounded)}):")
                for u in ungrounded[:3]:
                    print(f"    - {u}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"citation_{ts}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[Citation] Results written to {out_path}")

    print("\n=== Citation AGGREGATE ===")
    for m, label in [
        ("citation_accuracy", "Avg citation accuracy"),
        ("mean_similarity",   "Avg mean similarity"),
        ("citation_count",    "Avg citations/query"),
        ("grounded_count",    "Avg grounded/query"),
    ]:
        vals = [r[m] for r in all_results if isinstance(r.get(m), (int, float))]
        if vals:
            print(f"  {label:30s} {round(sum(vals) / len(vals), 4)}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run semantic citation grounding evaluation")
    parser.add_argument("--query-ids",  nargs="*")
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--threshold",  type=float, default=_CITATION_ACCURACY_THRESHOLD,
                        help="Cosine similarity threshold for grounding (default: 0.75)")
    add_provider_args(parser)
    args = parser.parse_args()
    run_citation_eval(
        query_ids=args.query_ids,
        output_dir=args.output_dir,
        cfg=cfg_from_args(args),
        threshold=args.threshold,
    )
