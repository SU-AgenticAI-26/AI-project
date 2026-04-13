"""
eval_perf.py — Performance profiling: per-agent latency and token consumption.

Wraps the LangGraph agent functions with timing and token-counting decorators,
then runs each test query and reports a per-agent breakdown.

Requires the eval dependencies (includes tiktoken):
  pip install -r requirements-eval.txt   OR   uv sync --extra eval

Metrics reported:
  - wall_s:           Wall-clock time per agent call
  - prompt_tokens:    Approximate input token count (via tiktoken)
  - total_wall_s:     End-to-end pipeline time
  - estimated_cost:   USD cost at the configured provider's pricing
  - critic_loop_overhead: Extra time and tokens spent in enrichment loops
  - cap_hit_rate:     Fraction of runs where loop_count == 2

Usage:
    # OpenAI (default)
    python eval_perf.py

    # Local llama.cpp
    python eval_perf.py --provider local --model mistral --base-url http://localhost:8080/v1

    # Subset of queries
    python eval_perf.py --query-ids q1_rag
"""

from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from test_queries import TEST_QUERIES, INITIAL_STATE_TEMPLATE
from eval_provider import EvalConfig, add_provider_args, cfg_from_args, count_tokens, estimate_cost


# ---------------------------------------------------------------------------
# Instrumented graph builder
# ---------------------------------------------------------------------------

def build_instrumented_graph(cfg: EvalConfig, vdb, perf_log: list[dict]):
    """
    Build the LangGraph with timing wrappers on every agent node.
    Each agent call appends an entry to perf_log.
    """
    from test_queries import _add_project_to_path
    _add_project_to_path()

    import streamlit_app as _sa
    from langgraph.graph import END, StateGraph

    app_cfg = cfg.pipeline_app_cfg()
    pricing = cfg.pricing()

    lm_sc = _sa._llm(app_cfg, 0.2)   # scoping
    lm_r  = _sa._llm(app_cfg, 0.0)   # router
    lm_s  = _sa._llm(app_cfg, 0.3)   # retrieval agents
    lm_e  = _sa._llm(app_cfg, 0.1)   # reading_extraction
    lm_o  = _sa._llm(app_cfg, 0.2)   # orchestrator
    lm_cf = _sa._llm(app_cfg, 0.0)   # conflict_detector
    lm_m  = _sa._llm(app_cfg, 0.1)   # knowledge_mapper
    lm_c  = _sa._llm(app_cfg, 0.0)   # critic
    lm_z  = _sa._llm(app_cfg, 0.5)   # summarizer
    lm_x  = _sa._llm(app_cfg, 0.3)   # experiment_design

    agent_fns = {
        "scoping":            lambda s: _sa.scoping_agent(s, lm_sc),
        "router":             lambda s: _sa.router_agent(s, lm_r),
        "vector_db":          lambda s: _sa.vector_db_agent(s, lm_s, vdb),
        "sql_db":             lambda s: _sa.sql_db_agent(s, lm_s),
        "web":                lambda s: _sa.web_agent(s, lm_s, vdb),
        "reading_extraction": lambda s: _sa.reading_extraction_agent(s, lm_e),
        "orchestrator":       lambda s: _sa.orchestrator_agent(s, lm_o),
        "conflict_detector":  lambda s: _sa.conflict_agent(s, lm_cf),
        "knowledge_mapper":   lambda s: _sa.knowledge_mapper_agent(s, lm_m),
        "critic":             lambda s: _sa.critic_agent(s, lm_c),
        "summarizer":         lambda s: _sa.summarizer_agent(s, lm_z),
        "experiment_design":  lambda s: _sa.experiment_design_agent(s, lm_x),
    }

    def _make_timed(name: str, fn: Callable) -> Callable:
        def wrapper(state):
            ctx = (
                state.get("query", "") +
                state.get("merged_context", "") +
                state.get("vector_findings", "") +
                state.get("sql_findings", "") +
                state.get("web_findings", "") +
                state.get("extraction_findings", "")
            )
            prompt_tokens = count_tokens(ctx)
            t0 = time.perf_counter()
            result = fn(state)
            elapsed = time.perf_counter() - t0
            perf_log.append({
                "agent":         name,
                "wall_s":        round(elapsed, 3),
                "prompt_tokens": prompt_tokens,
                "cost_usd":      estimate_cost(prompt_tokens, pricing),
                "loop_count":    state.get("loop_count", 0),
                "ts":            time.time(),
            })
            return result
        return wrapper

    g = StateGraph(_sa.AgentState)
    for name, fn in agent_fns.items():
        g.add_node(name, _make_timed(name, fn))

    # Mirror the real build_graph topology exactly so perf numbers reflect
    # production behaviour (parallel retrieval fan-out, all agent nodes).
    g.set_entry_point("scoping")
    g.add_edge("scoping",            "router")
    g.add_edge("router",             "vector_db")
    g.add_edge("router",             "sql_db")
    g.add_edge("router",             "web")
    g.add_edge("vector_db",          "reading_extraction")
    g.add_edge("sql_db",             "reading_extraction")
    g.add_edge("web",                "reading_extraction")
    g.add_edge("reading_extraction", "orchestrator")
    g.add_edge("orchestrator",       "conflict_detector")
    g.add_edge("conflict_detector",  "knowledge_mapper")
    g.add_edge("knowledge_mapper",   "critic")
    g.add_conditional_edges(
        "critic", _sa._route_critic,
        {"router": "router", "summarizer": "summarizer"},
    )
    g.add_edge("summarizer",         "experiment_design")
    g.add_edge("experiment_design",  END)
    return g.compile()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_perf_eval(
    query_ids: list[str] | None = None,
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

    from test_queries import _add_project_to_path
    _add_project_to_path()

    app_cfg = cfg.pipeline_app_cfg()
    from streamlit_app import VectorDBModule, _embeddings, _embedding_key, init_sql_db
    from pathlib import Path
    init_sql_db()
    vdb_dir = Path("collab_rag_data") / "vectorstore" / _embedding_key(app_cfg)
    vdb = VectorDBModule(_embeddings(app_cfg), vector_dir=vdb_dir)

    queries = [q for q in TEST_QUERIES if query_ids is None or q["id"] in query_ids]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    # Build the instrumented graph once; reuse it across all queries.
    # The perf_log list is captured by reference in the timing closures — clearing
    # it between queries is sufficient to reset per-query tracking.
    perf_log: list[dict] = []
    app = build_instrumented_graph(cfg, vdb, perf_log)

    for tq in queries:
        print(f"\n[Perf] Running: {tq['id']} — {tq['query'][:60]}...")

        perf_log.clear()

        initial_state = {**INITIAL_STATE_TEMPLATE, "query": tq["query"]}
        t0 = time.perf_counter()
        final_state = app.invoke(initial_state)
        total_wall = round(time.perf_counter() - t0, 3)

        # Aggregate per-agent (agents may appear more than once if Critic looped)
        by_agent: dict[str, dict] = {}
        for entry in perf_log:
            name = entry["agent"]
            if name not in by_agent:
                by_agent[name] = {"calls": 0, "wall_s": 0.0, "prompt_tokens": 0, "cost_usd": 0.0}
            by_agent[name]["calls"]         += 1
            by_agent[name]["wall_s"]        += entry["wall_s"]
            by_agent[name]["prompt_tokens"] += entry["prompt_tokens"]
            by_agent[name]["cost_usd"]      += entry["cost_usd"]

        total_tokens = sum(e["prompt_tokens"] for e in perf_log)
        total_cost   = round(sum(e["cost_usd"] for e in perf_log), 6)
        loop_count   = final_state.get("loop_count", 0)

        result = {
            "query_id":       tq["id"],
            "query":          tq["query"],
            "difficulty":     tq["difficulty"],
            "provider":       cfg.provider,
            "model":          cfg.model,
            "total_wall_s":   total_wall,
            "total_tokens":   total_tokens,
            "total_cost_usd": total_cost,
            "loop_count":     loop_count,
            "cap_hit":        loop_count >= 2,
            "per_agent":      {
                name: {
                    "calls":         v["calls"],
                    "wall_s":        round(v["wall_s"], 3),
                    "prompt_tokens": v["prompt_tokens"],
                    "cost_usd":      round(v["cost_usd"], 6),
                    "pct_wall":      round(v["wall_s"] / total_wall * 100, 1) if total_wall else 0,
                }
                for name, v in by_agent.items()
            },
        }
        all_results.append(result)

        print(f"  Provider:        {cfg.provider} / {cfg.model}")
        print(f"  Total wall time: {total_wall}s")
        print(f"  Total tokens:    {total_tokens:,}")
        print(f"  Estimated cost:  ${total_cost}")
        print(f"  Loop count:      {loop_count}")
        print(f"  Per-agent breakdown:")
        for agent, v in sorted(result["per_agent"].items(), key=lambda x: -x[1]["wall_s"]):
            calls_str = f"x{v['calls']}" if v["calls"] > 1 else "    "
            print(f"    {agent:20s} {calls_str}  {v['wall_s']:6.2f}s  "
                  f"{v['pct_wall']:5.1f}%  {v['prompt_tokens']:6,} tok  ${v['cost_usd']:.5f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"perf_{ts}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[Perf] Results written to {out_path}")

    # Aggregate
    print("\n=== Perf AGGREGATE ===")
    print(f"  {'Query':40s} {'Wall(s)':>8} {'Tokens':>8} {'Cost':>10} {'Loops':>6}")
    for r in all_results:
        print(f"  {r['query_id']:40s} {r['total_wall_s']:>8.1f} "
              f"{r['total_tokens']:>8,} ${r['total_cost_usd']:>9.5f} {r['loop_count']:>6}")
    if all_results:
        cap_rate = round(sum(1 for r in all_results if r["cap_hit"]) / len(all_results), 3)
        print(f"\n  Cap-hit rate (loop_count == 2): {cap_rate} (target: < 0.30)")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run performance profiling")
    parser.add_argument("--query-ids", nargs="*")
    parser.add_argument("--output-dir", default="eval_results")
    add_provider_args(parser)
    args = parser.parse_args()
    run_perf_eval(
        query_ids=args.query_ids,
        output_dir=args.output_dir,
        cfg=cfg_from_args(args),
    )
