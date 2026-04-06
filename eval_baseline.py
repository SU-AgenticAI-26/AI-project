"""
eval_baseline.py — Single-LLM baseline for comparison against the multi-agent system.

The baseline is a single LLM call with the raw query and no retrieval,
no agent decomposition, and no knowledge graph. This is the condition described
in the proposal's Section 5.9 evaluation methodology.

The baseline output is stored in the same format as the full pipeline state so
it can be passed to eval_ragas.py and eval_deepeval.py for direct comparison.

Usage:
    # OpenAI (default)
    python eval_baseline.py

    # Local llama.cpp
    python eval_baseline.py --provider local --model mistral --base-url http://localhost:8080/v1

    # Gemini
    python eval_baseline.py --provider gemini --model gemini-2.0-flash

    # Subset of queries
    python eval_baseline.py --query-ids q1_rag

Requires:
    pip install langchain-openai tiktoken
    # Plus provider-specific packages; see streamlit_app.py install notes.
"""

from __future__ import annotations
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from test_queries import TEST_QUERIES
from eval_provider import EvalConfig, add_provider_args, cfg_from_args, count_tokens as _count_tokens


BASELINE_SYSTEM_PROMPT = (
    "You are a research assistant. Given a research question, write a clear and "
    "informative summary of what is known about the topic based on your training knowledge. "
    "Organise your answer by theme where possible and note any important disagreements "
    "or open questions in the literature."
)


def run_baseline_query(query: str, cfg: EvalConfig) -> dict:
    """Run single-LLM baseline. Returns a state-compatible dict."""
    from langchain_core.messages import HumanMessage, SystemMessage

    lm = cfg.pipeline_llm(temperature=0.3)
    pricing = cfg.pricing()

    t0 = time.perf_counter()
    response = lm.invoke([
        SystemMessage(content=BASELINE_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ])
    wall_time = time.perf_counter() - t0

    summary = response.content or ""

    # Use token counts from response metadata when available (OpenAI/Gemini provide them),
    # fall back to tiktoken estimation.
    raw_usage = getattr(response, "usage_metadata", None)
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    in_tokens  = usage.get("input_tokens")  or _count_tokens(query + BASELINE_SYSTEM_PROMPT)
    out_tokens = usage.get("output_tokens") or _count_tokens(summary)
    cost = (in_tokens  * pricing["input"]  / 1_000_000 +
            out_tokens * pricing["output"] / 1_000_000)

    return {
        # Mirror the structure of AgentState so eval scripts can process it identically
        "query":            query,
        "active_agents":    [],
        "router_reasoning": "BASELINE: no routing",
        "vector_findings":  "",
        "sql_findings":     "",
        "web_findings":     "",
        "merged_context":   "",  # no retrieval
        "knowledge_map":    {"nodes": [], "edges": []},
        "critique":         "",
        "loop_count":       0,
        "summary":          summary,
        # Perf extras
        "_wall_time_s":  round(wall_time, 3),
        "_in_tokens":    in_tokens,
        "_out_tokens":   out_tokens,
        "_cost_usd":     round(cost, 6),
        "_provider":     cfg.provider,
        "_model":        cfg.model,
        "_is_baseline":  True,
    }


def run_baseline_eval(
    query_ids: list[str] | None = None,
    output_dir: str = "eval_results",
    cfg: EvalConfig | None = None,
    # Legacy kwarg kept for backwards compatibility
    api_key: str | None = None,
) -> list[dict]:
    if cfg is None:
        from eval_provider import EvalConfig as EC
        import os
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("Pass an EvalConfig or set OPENAI_API_KEY.")
        cfg = EC(provider="openai", api_key=key)

    queries = [q for q in TEST_QUERIES if query_ids is None or q["id"] in query_ids]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Baseline queries are independent single-LLM calls; run them in parallel.
    # Worker count is capped at 4 to avoid overwhelming a local inference server.
    workers = min(len(queries), 4)
    states: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_tq = {
            pool.submit(run_baseline_query, tq["query"], cfg): tq
            for tq in queries
        }
        for future in as_completed(future_to_tq):
            tq = future_to_tq[future]
            state = future.result()
            states[tq["id"]] = state
            print(f"\n[Baseline] Done: {tq['id']} — {tq['query'][:60]}...")
            print(f"  Provider:  {cfg.provider} / {cfg.model}")
            print(f"  Wall time: {state['_wall_time_s']}s")
            print(f"  Tokens:    {state['_in_tokens']} in / {state['_out_tokens']} out")
            print(f"  Cost:      ${state['_cost_usd']}")
            print(f"  Summary:   {state['summary'][:200]}...")

    # Restore original query order for deterministic output files.
    all_results = []
    for tq in queries:
        state = states[tq["id"]]
        result = {
            "query_id":    tq["id"],
            "query":       tq["query"],
            "difficulty":  tq["difficulty"],
            "summary_len": len(state["summary"]),
            **{k: v for k, v in state.items() if k.startswith("_")},
        }
        all_results.append({**result, "state": state})

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"baseline_{ts}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[Baseline] Results written to {out_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-LLM baseline")
    parser.add_argument("--query-ids", nargs="*")
    parser.add_argument("--output-dir", default="eval_results")
    add_provider_args(parser)
    args = parser.parse_args()
    run_baseline_eval(
        query_ids=args.query_ids,
        output_dir=args.output_dir,
        cfg=cfg_from_args(args),
    )
