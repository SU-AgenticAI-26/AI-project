"""
test_queries.py — Shared test query definitions and pipeline invocation helper.

Import this in every eval script to ensure all scripts use the same queries
and the same method of running the pipeline.
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from eval_provider import EvalConfig

# Imported lazily so we don't require streamlit_app on import; set after _add_project_to_path()
VECTOR_DIR = Path("collab_rag_data") / "vectorstore"

# ---------------------------------------------------------------------------
# Test query definitions
# ---------------------------------------------------------------------------

TEST_QUERIES: list[dict] = [
    {
        "id": "q1_rag",
        "query": (
            "How does retrieval-augmented generation reduce hallucination "
            "in large language models?"
        ),
        "expected_channels": ["vector_db", "web"],
        "difficulty": "easy",
    },
    {
        "id": "q2_federated",
        "query": (
            "What are the main approaches and challenges in federated learning "
            "for healthcare applications?"
        ),
        "expected_channels": ["vector_db", "sql_db", "web"],
        "difficulty": "medium",
    },
    {
        "id": "q3_agents_experiment",
        "query": (
            "How are LLM agents being used to assist with scientific "
            "experiment design?"
        ),
        "expected_channels": ["web"],
        "difficulty": "hard",
    },
    {
        "id": "q4_multiagent",
        "query": (
            "What collaboration mechanisms are used in multi-agent LLM systems?"
        ),
        "expected_channels": ["vector_db", "sql_db", "web"],
        "difficulty": "medium",
    },
]

# ---------------------------------------------------------------------------
# Initial state template
# ---------------------------------------------------------------------------

INITIAL_STATE_TEMPLATE: dict = {
    "messages":           [],
    "query":              "",
    # Scoping
    "sub_questions":      [],
    "keywords":           [],
    "scoping_reasoning":  "",
    # Router
    "active_agents":      [],
    "router_reasoning":   "",
    # Retrieval
    "vector_findings":    "",
    "sql_findings":       "",
    "web_findings":       "",
    # Extraction + merge
    "extraction_findings": "",
    "merged_context":     "",
    "synthesis_report":   "",
    # Knowledge graph
    "knowledge_map":      {},
    # Critic loop
    "critique":           "",
    "_needs_more":        False,
    "loop_count":         0,
    # Conflict detection
    "conflicts":          [],
    "credibility_map":    {},
    # Final outputs
    "summary":            "",
    "citation_grounding": {},
    "grounding_score":    0.0,
    "experiment_plan":    "",
    # Metadata
    "activity_log":       [],
    "current_agent":      "",
}


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _add_project_to_path():
    """Add the AI-project directory to sys.path so streamlit_app can be imported."""
    # Suppress the flood of "missing ScriptRunContext" warnings that Streamlit
    # emits for every st.* call made outside of a Streamlit server context.
    import logging
    logging.getLogger("streamlit").setLevel(logging.ERROR)

    candidates = []
    if os.environ.get("PROJECT_DIR"):
        candidates.append(Path(os.environ["PROJECT_DIR"]))
    candidates += [
        Path(__file__).parent.parent / "AI-project-main",
        Path(__file__).parent.parent / "project_src" / "AI-project-main",
        Path.cwd() / "AI-project-main",
        Path.cwd(),
    ]
    for p in candidates:
        if (p / "streamlit_app.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return str(p)
    raise FileNotFoundError(
        "Cannot find streamlit_app.py. Run eval scripts from the project root "
        "or set the PROJECT_DIR environment variable to the directory containing "
        "streamlit_app.py."
    )


def _get_cached_vdb(cfg: "EvalConfig"):
    """
    Build the VectorDB once per EvalConfig instance and cache it.

    Subsequent calls return the cached object, avoiding repeated embedding-model
    loads (which are the most expensive part of VectorDB initialisation).
    """
    if hasattr(cfg, "_vdb_cache"):
        return cfg._vdb_cache

    _add_project_to_path()
    from streamlit_app import VectorDBModule, _embeddings, _embedding_key, init_sql_db

    init_sql_db()
    app_cfg = cfg.pipeline_app_cfg()
    vdb_dir = VECTOR_DIR / _embedding_key(app_cfg)
    vdb = VectorDBModule(_embeddings(app_cfg), vector_dir=vdb_dir)
    object.__setattr__(cfg, "_vdb_cache", vdb)
    return vdb


def _get_cached_app(cfg: "EvalConfig", vdb):
    """
    Compile the LangGraph application once per EvalConfig instance and cache it.

    LangGraph compiled apps are stateless between invocations — each call to
    app.invoke() creates its own isolated state — so the same compiled graph
    can safely be reused across multiple queries.
    """
    if hasattr(cfg, "_app_cache"):
        return cfg._app_cache

    _add_project_to_path()
    from streamlit_app import build_graph

    app_cfg = cfg.pipeline_app_cfg()
    app = build_graph(app_cfg, vdb)
    object.__setattr__(cfg, "_app_cache", app)
    return app


def run_pipeline(
    query: str,
    cfg: "EvalConfig | None" = None,
    force_channels: Optional[list[str]] = None,
    disable_critic: bool = False,
    # Legacy kwarg kept for backwards compatibility; ignored when cfg is given
    api_key: Optional[str] = None,
) -> tuple[dict, float]:
    """
    Run the full LangGraph pipeline for a single query.

    Parameters
    ----------
    query : str
        The research query.
    cfg : EvalConfig, optional
        Provider configuration. When omitted, falls back to a default OpenAI
        config built from OPENAI_API_KEY (legacy behaviour).
    force_channels : list[str], optional
        If provided, bypass the Router and force these channels active.
        Used for ablation configs (e.g. ["vector_db"] for vector-only).
    disable_critic : bool
        If True, the Critic will always route to summarizer (no enrichment loop).

    Returns
    -------
    state : dict
        The final AgentState after pipeline completion.
    wall_time : float
        Total wall-clock time in seconds.
    """
    _add_project_to_path()

    if cfg is None:
        from eval_provider import EvalConfig
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("Pass an EvalConfig or set OPENAI_API_KEY.")
        cfg = EvalConfig(provider="openai", api_key=key)

    # VectorDB is always reused across calls on the same cfg.
    vdb = _get_cached_vdb(cfg)
    app_cfg = cfg.pipeline_app_cfg()

    if force_channels is not None or disable_critic:
        # Ablation path: need a freshly patched graph; cannot reuse the cached app.
        import streamlit_app as _sa

        patched_router = None
        patched_critic = None

        if force_channels is not None:
            _orig_router = _sa.router_agent
            lm_r = _sa._llm(app_cfg, 0.0)

            def _patched_router(state):
                result = _orig_router(state, lm_r)
                result["active_agents"] = force_channels
                result["router_reasoning"] = f"[FORCED for ablation: {force_channels}]"
                return result

            patched_router = _patched_router

        if disable_critic:
            _orig_critic = _sa.critic_agent
            lm_c = _sa._llm(app_cfg, 0.0)

            def _patched_critic(state):
                result = _orig_critic(state, lm_c)
                result["_needs_more"] = False  # always approve
                return result

            patched_critic = _patched_critic

        app = _rebuild_graph(app_cfg, vdb,
                             router_fn=patched_router,
                             critic_fn=patched_critic)
    else:
        # Normal (unpatched) path: reuse the cached compiled app.
        app = _get_cached_app(cfg, vdb)

    initial_state = {**INITIAL_STATE_TEMPLATE, "query": query}
    t0 = time.perf_counter()
    final_state = app.invoke(initial_state)
    wall_time = time.perf_counter() - t0
    return dict(final_state), wall_time


def _rebuild_graph(app_cfg, vdb, router_fn=None, critic_fn=None):
    """Rebuild the LangGraph with patched agent functions for ablation.

    Mirrors the full build_graph topology so ablation results are comparable
    to production runs (parallel retrieval fan-out, all agent nodes present).
    """
    import streamlit_app as _sa
    from langgraph.graph import END, StateGraph

    lm_sc = _sa._llm(app_cfg, 0.2)
    lm_r  = _sa._llm(app_cfg, 0.0)
    lm_s  = _sa._llm(app_cfg, 0.3)
    lm_e  = _sa._llm(app_cfg, 0.1)
    lm_o  = _sa._llm(app_cfg, 0.2)
    lm_cf = _sa._llm(app_cfg, 0.0)
    lm_m  = _sa._llm(app_cfg, 0.1)
    lm_c  = _sa._llm(app_cfg, 0.0)
    lm_z  = _sa._llm(app_cfg, 0.5)
    lm_x  = _sa._llm(app_cfg, 0.4)

    router = router_fn if router_fn else lambda s: _sa.router_agent(s, lm_r)
    critic = critic_fn if critic_fn else lambda s: _sa.critic_agent(s, lm_c)

    g = StateGraph(_sa.AgentState)
    g.add_node("scoping",            lambda s: _sa.scoping_agent(s, lm_sc))
    g.add_node("router",             router)
    g.add_node("vector_db",          lambda s: _sa.vector_db_agent(s, lm_s, vdb))
    g.add_node("sql_db",             lambda s: _sa.sql_db_agent(s, lm_s))
    g.add_node("web",                lambda s: _sa.web_agent(s, lm_s, vdb))
    g.add_node("reading_extraction", lambda s: _sa.reading_extraction_agent(s, lm_e))
    g.add_node("orchestrator",       lambda s: _sa.orchestrator_agent(s, lm_o))
    g.add_node("conflict_detector",  lambda s: _sa.conflict_agent(s, lm_cf))
    g.add_node("knowledge_mapper",   lambda s: _sa.knowledge_mapper_agent(s, lm_m))
    g.add_node("critic",             critic)
    g.add_node("summarizer",         lambda s: _sa.summarizer_agent(s, lm_z))
    g.add_node("experiment_design",  lambda s: _sa.experiment_design_agent(s, lm_x))

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


def split_context_into_chunks(merged_context: str, chunk_size: int = 500) -> list[str]:
    """Split merged_context into non-overlapping chunks for RAGAS evaluation."""
    if not merged_context:
        return []
    words = merged_context.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
