"""
app.py — Streamlit frontend for the Multi-Agent Research Assistant (5-agent).

Provider configuration is in the sidebar: choose any of Anthropic, OpenAI,
OpenAI-compatible, Google Gemini, Ollama, or llama.cpp — including per-agent
overrides for mixed configurations. All settings take effect before the pipeline runs.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""<style>
/* ── Adaptive block colours ── */
/* Light mode defaults */
.reasoning-block  { --bg:#e8edff; --bd:#4a6cf7; }
.tool-call-block  { --bg:#fff3e0; --bd:#f97316; }
.tool-result-block{ --bg:#e8f5e9; --bd:#22c55e; }
.msg-task         { --bg:#ede9fe; --bd:#6366f1; }
.msg-result       { --bg:#e8f5e9; --bd:#22c55e; }
.msg-feedback     { --bg:#fde8e8; --bd:#ef4444; }
.msg-ack          { --bg:#fef9e0; --bd:#f59e0b; }
.msg-request      { --bg:#f0eaff; --bd:#8b5cf6; }

/* Dark mode overrides */
@media (prefers-color-scheme: dark) {
  .reasoning-block  { --bg:#1e2347; --bd:#818cf8; }
  .tool-call-block  { --bg:#2a1f0e; --bd:#fb923c; }
  .tool-result-block{ --bg:#0d2318; --bd:#4ade80; }
  .msg-task         { --bg:#1e1b38; --bd:#818cf8; }
  .msg-result       { --bg:#0d2318; --bd:#4ade80; }
  .msg-feedback     { --bg:#2d1111; --bd:#f87171; }
  .msg-ack          { --bg:#2a1f00; --bd:#fbbf24; }
  .msg-request      { --bg:#1e1230; --bd:#a78bfa; }
}

.reasoning-block, .tool-call-block, .tool-result-block,
.msg-task, .msg-result, .msg-feedback, .msg-ack, .msg-request {
  background: var(--bg);
  border-left: 3px solid var(--bd);
  padding: 6px 12px;
  margin: 4px 0;
  border-radius: 0 4px 4px 0;
  font-size: .88em;
  color: inherit;
}
.reasoning-block   { font-size:.9em; padding:8px 12px; }
.tool-call-block   { font-family: monospace; }
.msg-feedback      { font-weight: 500; }

/* Ensure markdown inside these divs inherits colour */
.reasoning-block *, .tool-call-block *, .tool-result-block *,
.msg-task *, .msg-result *, .msg-feedback *, .msg-ack *, .msg-request * {
  color: inherit;
}
</style>""", unsafe_allow_html=True)

# ── Sidebar: provider configuration ──────────────────────────────────────────

PROVIDERS = ["anthropic", "openai", "gemini", "ollama", "llamacpp"]
PROVIDER_LABELS = {
    "anthropic": "Anthropic Claude",
    "openai":    "OpenAI / OpenAI-compatible",
    "gemini":    "Google Gemini",
    "ollama":    "Ollama (local)",
    "llamacpp":  "llama.cpp (local)",
}
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai":    "gpt-4o",
    "gemini":    "gemini-1.5-pro",
    "ollama":    "llama3.1:8b",
    "llamacpp":  "local-model",
}

# ── Model-list helpers ────────────────────────────────────────────────────────

from llm.model_fetcher import (
    fetch_anthropic_models, fetch_openai_models,
    fetch_gemini_models,    fetch_ollama_models,
)

def _model_selector(
    label: str,
    provider: str,
    key_suffix: str,
    api_key: str = "",
    base_url: str = "",
    host:    str = "",
) -> str:
    """
    Render a model selector for `provider`. When models can be fetched, shows
    a selectbox; otherwise falls back to a text_input. Caches fetched lists in
    st.session_state under 'models_{provider}_{key_suffix}'.

    Returns the selected/typed model ID.
    """
    cache_key  = f"models_{provider}_{key_suffix}"
    status_key = f"model_fetch_status_{provider}_{key_suffix}"

    cached = st.session_state.get(cache_key, [])

    # Fetch button (not shown for llamacpp which has no list endpoint)
    if provider != "llamacpp":
        btn_label = "🔄 Fetch models" if not cached else "🔄 Refresh models"
        if st.button(btn_label, key=f"fetch_btn_{provider}_{key_suffix}"):
            with st.spinner("Fetching available models…"):
                if provider == "anthropic":
                    fetched = fetch_anthropic_models(api_key)
                elif provider == "openai":
                    fetched = fetch_openai_models(api_key, base_url)
                elif provider == "gemini":
                    fetched = fetch_gemini_models(api_key)
                elif provider == "ollama":
                    fetched = fetch_ollama_models(host)
                else:
                    fetched = []

            if fetched:
                st.session_state[cache_key]  = fetched
                st.session_state[status_key] = f"✅ {len(fetched)} models available"
                cached = fetched
            else:
                st.session_state[status_key] = (
                    "⚠️ No models returned — check your API key and connection."
                )

        if st.session_state.get(status_key):
            st.caption(st.session_state[status_key])

    default = DEFAULT_MODELS.get(provider, "")

    if cached:
        # Pick a sensible default index
        idx = 0
        if default in cached:
            idx = cached.index(default)
        return st.selectbox(
            label, cached, index=idx, key=f"sel_{provider}_{key_suffix}"
        )
    else:
        return st.text_input(
            label, value=default, key=f"txt_{provider}_{key_suffix}",
            placeholder="Enter model ID or click Fetch models"
        )


with st.sidebar:
    st.title("⚙️ Configuration")

    # ── User identity ─────────────────────────────────────────────────────
    st.subheader("👤 User")
    username = st.text_input(
        "Username",
        value=st.session_state.get("username", ""),
        placeholder="Enter a username to save history",
        help="Each username has its own query and result history stored as markdown files.",
    )
    if username:
        st.session_state["username"] = username.strip().lower()
    else:
        st.session_state.setdefault("username", "")

    st.divider()

    # ── API Keys ──────────────────────────────────────────────────────────
    st.subheader("API Keys")
    anthropic_key = st.text_input(
        "Anthropic API Key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Required when using Anthropic provider",
    )
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    ss_key = st.text_input(
        "Semantic Scholar API Key (optional)",
        value=os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""),
        type="password",
        help="Free at semanticscholar.org/product/api — raises rate limits",
    )
    if ss_key:
        os.environ["SEMANTIC_SCHOLAR_API_KEY"] = ss_key

    st.divider()

    # ── Default provider ──────────────────────────────────────────────────
    st.subheader("🤖 LLM Provider")
    st.caption(
        "All agents use the default provider. "
        "Expand **Per-agent overrides** to mix providers."
    )

    default_provider = st.selectbox(
        "Default provider (all agents)",
        PROVIDERS,
        format_func=lambda x: PROVIDER_LABELS[x],
        index=0,
    )

    extra_cfg: dict = {}

    if default_provider == "llamacpp":
        llamacpp_host = st.text_input(
            "llama.cpp server URL",
            value=os.getenv("LLAMACPP_HOST", "http://localhost:8080"),
            help="Start server: ./llama-server -m model.gguf --port 8080",
        )
        extra_cfg["host"] = llamacpp_host
        default_model = st.text_input(
            "Model ID", value=DEFAULT_MODELS["llamacpp"],
            key="txt_llamacpp_default",
            help="Enter the model name used by your llama.cpp server",
        )
        st.caption("llama.cpp: `./llama-server -m llama3.1-8b.gguf --port 8080 --n-gpu-layers 99`")

    elif default_provider == "ollama":
        ollama_host = st.text_input(
            "Ollama host",
            value=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
        extra_cfg["host"] = ollama_host
        default_model = _model_selector(
            "Model", "ollama", "default", host=ollama_host
        )

    elif default_provider == "openai":
        oai_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
        if oai_key:
            os.environ["OPENAI_API_KEY"] = oai_key
            extra_cfg["api_key"] = oai_key
        base_url = st.text_input(
            "Base URL (blank = official OpenAI)",
            value="",
            help="Use for Azure, LM Studio, vLLM, etc.",
        )
        if base_url:
            extra_cfg["base_url"] = base_url
        default_model = _model_selector(
            "Model", "openai", "default",
            api_key=oai_key, base_url=base_url,
        )

    elif default_provider == "gemini":
        gemini_key = st.text_input(
            "Google API Key",
            value=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
            type="password",
        )
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
            extra_cfg["api_key"] = gemini_key
        default_model = _model_selector(
            "Model", "gemini", "default", api_key=gemini_key
        )

    else:  # anthropic
        default_model = _model_selector(
            "Model", "anthropic", "default", api_key=anthropic_key
        )

    # ── Per-agent overrides ───────────────────────────────────────────────
    from llm.registry import AGENT_NAMES

    with st.expander("Per-agent overrides (optional)", expanded=False):
        st.caption(
            "Leave at '(default)' to use the setting above. "
            "Useful for: Claude for Validation only, local model for everything else."
        )
        agent_overrides_ui: dict = {}
        for agent_name in AGENT_NAMES:
            col_a, col_b = st.columns([1, 1])
            with col_a:
                p = st.selectbox(
                    agent_name,
                    ["(default)"] + PROVIDERS,
                    format_func=lambda x: x if x == "(default)" else PROVIDER_LABELS[x],
                    key=f"prov_{agent_name}",
                )
            if p != "(default)":
                override: dict = {"provider": p}

                if p == "llamacpp":
                    override["host"] = st.text_input(
                        f"llama.cpp host ({agent_name})",
                        value="http://localhost:8080",
                        key=f"host_{agent_name}",
                    )
                    override["model"] = st.text_input(
                        f"Model ({agent_name})",
                        value=DEFAULT_MODELS["llamacpp"],
                        key=f"txt_llamacpp_{agent_name}",
                    )
                elif p == "ollama":
                    _oh = st.text_input(
                        f"Ollama host ({agent_name})",
                        value="http://localhost:11434",
                        key=f"host_{agent_name}",
                    )
                    override["host"]  = _oh
                    override["model"] = _model_selector(
                        f"Model ({agent_name})", "ollama", agent_name, host=_oh
                    )
                elif p == "openai":
                    _oai_key = st.text_input(
                        f"OpenAI API key ({agent_name})",
                        type="password",
                        value=os.getenv("OPENAI_API_KEY", ""),
                        key=f"oai_key_{agent_name}",
                    )
                    _base_url = st.text_input(
                        f"Base URL ({agent_name})",
                        value="",
                        key=f"base_url_{agent_name}",
                        placeholder="blank = official OpenAI",
                    )
                    if _oai_key:
                        override["api_key"] = _oai_key
                    if _base_url:
                        override["base_url"] = _base_url
                    override["model"] = _model_selector(
                        f"Model ({agent_name})", "openai", agent_name,
                        api_key=_oai_key, base_url=_base_url,
                    )
                elif p == "gemini":
                    _gem_key = st.text_input(
                        f"Google API key ({agent_name})",
                        type="password",
                        value=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
                        key=f"gem_key_{agent_name}",
                    )
                    if _gem_key:
                        override["api_key"] = _gem_key
                    override["model"] = _model_selector(
                        f"Model ({agent_name})", "gemini", agent_name,
                        api_key=_gem_key,
                    )
                else:  # anthropic
                    override["model"] = _model_selector(
                        f"Model ({agent_name})", "anthropic", agent_name,
                        api_key=anthropic_key,
                    )

                agent_overrides_ui[agent_name] = override

    st.divider()

    # ── Architecture summary ──────────────────────────────────────────────
    st.markdown("""**5-Agent Architecture**

| Agent | Role |
|-------|------|
| **Orchestrator** | ReAct loop, 15 turns max |
| **ScopingAgent** | Decomposes query into 3-5 sub-questions |
| **SearchReadingAgent** | Selects sources from 7-source registry; retrieves & summarises |
| **SynthesisPlanningAgent** | Literature review + research plan; reads feedback |
| **ValidationAgent** | Validates; sends directed feedback laterally |

ValidationAgent → SynthesisPlanningAgent feedback is **lateral** (bypasses Orchestrator). Visible in **Agent Comms** tab.""")
    st.caption("Multi-agent · Multi-provider · Streamlit")


# ── History rendering helper ──────────────────────────────────────────────────

def _render_history_tab(username: str):
    """Render the History tab content for `username`."""
    from history_store import HistoryStore
    import re as _re

    if not username:
        st.info("Enter a username in the sidebar to view and save your history.")
        return

    store = HistoryStore(username)
    runs  = store.list_runs()

    if not runs:
        st.info(f"No saved runs for **{username}** yet. Results are saved automatically after each successful pipeline run.")
        return

    st.markdown(f"**{len(runs)} saved run(s)** for user `{username}`")

    # Search within history
    hist_search = st.text_input(
        "🔍 Search history",
        placeholder="Keywords to filter runs…",
        key="hist_search_input",
    )
    if hist_search:
        runs = store.search_runs(hist_search, n=50)
        if not runs:
            st.info("No matching runs.")
            return

    for run in runs:
        ts_fmt = run.timestamp[:16].replace("T", " ") if run.timestamp else "unknown time"
        approved_badge = "✅" if run.approved else "⚠️"
        label = f"{approved_badge} **{run.query[:70]}{'…' if len(run.query) > 70 else ''}** — {ts_fmt} · {run.n_papers} papers"
        with st.expander(label, expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Query:** {run.query}")
                if run.themes:
                    st.markdown(f"**Themes:** {', '.join(run.themes)}")
                st.caption(
                    f"Papers: {run.n_papers}  ·  Steps: {run.n_steps}  ·  "
                    f"Validation: {'approved' if run.approved else 'not fully approved'}  ·  "
                    f"File: `{run.filename}`"
                )
            with col2:
                # Load and offer download
                loaded = store.load_run(run.filename)
                if loaded:
                    st.download_button(
                        "⬇️ Download .md",
                        data=loaded["raw"],
                        file_name=run.filename,
                        mime="text/markdown",
                        key=f"dl_{run.filename}",
                    )

            # Inline body preview (synthesis section only)
            loaded = store.load_run(run.filename)
            if loaded and loaded.get("body"):
                body = loaded["body"]
                # Extract just the synthesis paragraph
                synth_m = _re.search(
                    r"## Thematic Synthesis\n+(.*?)(?=\n## |\Z)", body, _re.DOTALL
                )
                if synth_m:
                    preview = synth_m.group(1).strip()[:800]
                    st.markdown("**Synthesis preview:**")
                    st.markdown(preview + ("…" if len(synth_m.group(1).strip()) > 800 else ""))

                # Full body toggle
                if st.checkbox("Show full document", key=f"full_{run.filename}"):
                    st.markdown(body)


# ── Main panel ────────────────────────────────────────────────────────────────

st.title("🔬 Multi-Agent Research Assistant")
st.markdown(
    "Five specialist agents collaborate through a typed message bus. "
    "ValidationAgent sends directed feedback **directly** to SynthesisPlanningAgent — "
    "lateral agent-to-agent communication, not relayed through the Orchestrator."
)

query = st.text_area(
    "Research question",
    placeholder=(
        "e.g. How can retrieval-augmented generation reduce hallucination in LLMs?"
    ),
    height=90,
)

provider_ready = (
    (default_provider == "anthropic" and bool(os.getenv("ANTHROPIC_API_KEY")))
    or default_provider in ("ollama", "llamacpp")
    or (default_provider == "openai"  and bool(os.getenv("OPENAI_API_KEY")))
    or (default_provider == "gemini"  and bool(
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    ))
)

# ── History browser (always visible) ─────────────────────────────────────────
_sidebar_uname = st.session_state.get("username", "")
if _sidebar_uname:
    with st.expander(f"🕘 History for **{_sidebar_uname}**", expanded=False):
        _render_history_tab(_sidebar_uname)

run_btn = st.button(
    "▶ Run Research Pipeline",
    type="primary",
    use_container_width=True,
    disabled=not bool(query and provider_ready),
)

if not provider_ready:
    if default_provider == "anthropic":
        st.warning("⚠️ Enter your Anthropic API key in the sidebar.")
    elif default_provider == "openai":
        st.warning("⚠️ Enter your OpenAI API key in the sidebar.")
    elif default_provider == "gemini":
        st.warning("⚠️ Enter your Google API key in the sidebar.")
    else:
        st.info(
            f"ℹ️ Using local provider ({PROVIDER_LABELS[default_provider]}). "
            "Make sure the server is running before clicking Run."
        )

# ── Run pipeline ──────────────────────────────────────────────────────────────

if run_btn and query:
    from agents.orchestrator import run_orchestrator
    from llm.registry import build_config

    agent_cfg_dict = {
        "default": {"provider": default_provider, "model": default_model, **extra_cfg},
        "agents":  agent_overrides_ui,
    }
    try:
        agent_config = build_config(agent_cfg_dict)
    except Exception as cfg_err:
        st.error(f"Configuration error: {cfg_err}")
        st.stop()

    working_state = {
        "query": query, "sub_questions": [], "all_papers": [],
        "search_queries_used": [], "paper_summaries": [],
        "sources_selected": [], "source_rationale": {},
        "synthesis": "", "gaps": "", "research_plan": "",
        "themes": [], "contradictions": [], "uncovered_sub_questions": [],
        "research_steps": [], "risks_and_mitigations": "",
        "citation_edges": [],
        "validation_result": {}, "validation_iterations": 0,
        "orchestrator_trace": [], "status_log": [],
        "message_log": [], "bus_messages": [],
        "provider_summary": {},
        "is_complete": False, "error": None,
    }

    # Inject relevant prior history as context for the Orchestrator
    _uname = st.session_state.get("username", "")
    if _uname:
        try:
            from history_store import HistoryStore
            _ctx = HistoryStore(_uname).prior_context(query, n=2)
            if _ctx:
                working_state["prior_context"] = _ctx
        except Exception:
            pass

    st.subheader("🧠 Live Orchestrator Trace")
    st.caption(
        "Blue = orchestrator reasoning  ·  "
        "Orange = message sent to agent  ·  "
        "Green = agent reply"
    )
    trace_placeholder = st.empty()
    trace_events: list = []

    def render_trace():
        with trace_placeholder.container():
            for evt in trace_events[-30:]:
                t, data = evt["type"], evt["data"]
                if t == "reasoning":
                    text = data[:500] + ("…" if len(data) > 500 else "")
                    st.markdown(
                        f'<div class="reasoning-block">💭 <b>Orchestrator:</b> {text}</div>',
                        unsafe_allow_html=True,
                    )
                elif t == "tool_call":
                    agent  = data.get("agent", "")
                    inputs = data.get("inputs", {})
                    detail = ""
                    if inputs.get("targeted_query"):
                        detail += f' targeted="{inputs["targeted_query"]}"'
                    if inputs.get("targeted_label"):
                        detail += f' label="{inputs["targeted_label"]}"'
                    st.markdown(
                        f'<div class="tool-call-block">📨 <b>{data["name"]}</b>'
                        f'{"  →  " + agent if agent else ""}{detail}</div>',
                        unsafe_allow_html=True,
                    )
                elif t == "tool_result":
                    st.markdown(
                        f'<div class="tool-result-block">✅ <b>{data.get("agent","")}</b>: '
                        f'{data.get("result","")}</div>',
                        unsafe_allow_html=True,
                    )
                elif t == "error":
                    msg = str(data)
                    if any(k in msg.lower() for k in ("401", "403", "authentication", "api key", "invalid_api_key")):
                        label = "🔑 Authentication error"
                    elif any(k in msg.lower() for k in ("connection", "timeout", "refused", "network")):
                        label = "🔌 Connection error"
                    else:
                        label = "⚠️ Error"
                    st.error(f"**{label}:** {msg}")

    def callback(event_type, data):
        if event_type != "status":
            trace_events.append({"type": event_type, "data": data})
            render_trace()

    # ── Provider connectivity check ───────────────────────────────────────
    with st.spinner("Checking LLM provider connection…"):
        try:
            orch_provider = agent_config["orchestrator"]
            ping_reply = orch_provider.complete(
                "You are a connectivity test.",
                "Reply with the single word: ready",
                max_tokens=10,
            )
            if not ping_reply.strip():
                raise ValueError("Empty response from provider")
        except Exception as ping_err:
            err_msg = str(ping_err)
            # Classify the error for a more actionable message
            if any(k in err_msg.lower() for k in ("401", "403", "authentication", "api key", "invalid_api_key", "permission")):
                friendly = (
                    f"**Authentication failed** — the API key for "
                    f"**{orch_provider.name}** was rejected.\n\n"
                    f"Check that the key is correct and has not expired.\n\n"
                    f"```\n{err_msg}\n```"
                )
            elif any(k in err_msg.lower() for k in ("connection", "timeout", "refused", "network", "unreachable", "resolve")):
                friendly = (
                    f"**Cannot reach {orch_provider.name}** — connection failed.\n\n"
                    f"If using a local provider (Ollama / llama.cpp), make sure the server "
                    f"is running. Otherwise check your internet connection.\n\n"
                    f"```\n{err_msg}\n```"
                )
            elif "import" in err_msg.lower() or "no module" in err_msg.lower():
                friendly = (
                    f"**Missing package** for **{orch_provider.name}**.\n\n"
                    f"Run `pip install -r requirements.txt` and restart the app.\n\n"
                    f"```\n{err_msg}\n```"
                )
            else:
                friendly = (
                    f"**Could not connect to {orch_provider.name}**.\n\n"
                    f"```\n{err_msg}\n```"
                )
            st.error(friendly)
            st.stop()

    with st.spinner("Pipeline running — typically 2-4 minutes…"):
        final_state = run_orchestrator(
            working_state, callback=callback, agent_config=agent_config
        )

    trace_placeholder.empty()
    st.divider()

    # ── Metrics bar ───────────────────────────────────────────────────────
    val_result   = final_state.get("validation_result", {})
    val_approved = val_result.get("approved", True)
    n_papers     = len(final_state.get("paper_summaries", []))
    n_searches   = len(final_state.get("search_queries_used", []))
    bus_messages = final_state.get("bus_messages", [])

    src_counts: dict = {}
    for p in final_state.get("all_papers", []):
        src = p.get("source", "?")
        src_counts[src] = src_counts.get(src, 0) + 1

    sources_selected = final_state.get("sources_selected", [])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Papers analysed",  n_papers)
    c2.metric("Search queries",   n_searches)
    c3.metric("Sources used",     len(sources_selected) or len(src_counts))
    c4.metric("Agent messages",   len(bus_messages))
    c5.metric("Validation",       "✅ Approved" if val_approved else "⚠️ Issues")

    if sources_selected:
        src_detail = "  ·  ".join(
            f"**{s}** ({src_counts.get(s, 0)})" for s in sources_selected
        )
        st.caption(f"Sources searched: {src_detail}")

    # Provider summary
    prov_summary = final_state.get("provider_summary", {})
    if prov_summary:
        unique = set(prov_summary.values())
        if len(unique) > 1:
            st.info(
                "🤖 Mixed providers: "
                + "  ·  ".join(f"**{a}** → {p}" for a, p in prov_summary.items())
            )
        else:
            st.caption(f"🤖 All agents: {next(iter(unique))}")

    # Feedback/ack callout
    fb_msgs  = [m for m in bus_messages if m["msg_type"] == "feedback"]
    ack_msgs = [m for m in bus_messages if m["msg_type"] == "ack"]
    if fb_msgs:
        st.info(
            f"💬 ValidationAgent sent **{len(fb_msgs)} directed feedback message(s)** "
            f"to SynthesisPlanningAgent. Agent sent **{len(ack_msgs)} acknowledgement(s)**. "
            "See the **Agent Communications** tab."
        )

    # ── Issue suggestion helper (used in validation expander and tab_lit) ─
    def _suggest(issue: str) -> str:
        il = issue.lower()
        if any(k in il for k in ("contradict", "conflict", "disagree", "inconsist")):
            return ("Compare methodologies and sample sizes of the conflicting papers. "
                    "Note the discrepancy explicitly in the review and flag it as an open question.")
        if any(k in il for k in ("uncorroborated", "citation", "not found", "hallucin", "unverified")):
            return ("Remove or rephrase the claim so it only asserts what the corpus supports. "
                    "If the paper exists, add it via a targeted search.")
        if any(k in il for k in ("coverage", "sub-question", "fewer than", "not address", "missing")):
            return ("Run a targeted search on this sub-question with more specific keywords, "
                    "or acknowledge the gap explicitly in the Research Gaps section.")
        if any(k in il for k in ("coherence", "flow", "structure", "organis", "organiz")):
            return ("Restructure the affected paragraph around a single theme. "
                    "Begin with the broadest claim and narrow to specifics.")
        if any(k in il for k in ("plan", "step", "direction", "research")):
            return ("Ensure each research step cites at least two corpus papers and links to "
                    "a stated future-work item from one of them.")
        return ("Review the flagged section against the original paper summaries "
                "and revise to remove unsupported assertions.")

    if not val_approved:
        issues = val_result.get("issues", [])
        with st.expander("⚠️ Remaining validation issues", expanded=False):
            for iss in issues:
                st.markdown(f"- {iss}")
                st.caption(f"Suggestion: {_suggest(iss)}")
            st.caption(val_result.get("feedback", ""))

    # ── Output tabs ───────────────────────────────────────────────────────
    tab_lit, tab_plan, tab_papers, tab_comms, tab_trace, tab_log, tab_hist = st.tabs([
        "📚 Literature Review",
        "📝 Research Plan",
        "🗂️ Papers",
        "💬 Agent Communications",
        "🧠 Orchestrator Trace",
        "📋 Log",
        "🕘 History",
    ])

    with tab_lit:
        sub_questions    = final_state.get("sub_questions", [])
        summaries        = final_state.get("paper_summaries", [])
        synthesis        = final_state.get("synthesis", "")
        gaps             = final_state.get("gaps", "")
        contradictions   = final_state.get("contradictions", [])
        uncovered        = final_state.get("uncovered_sub_questions", [])
        themes           = final_state.get("themes", [])
        research_steps   = final_state.get("research_steps", [])
        plan_prose       = final_state.get("research_plan", "")
        risks            = final_state.get("risks_and_mitigations", "")
        citation_edges   = final_state.get("citation_edges", [])

        # ── Sub-questions ────────────────────────────────────────────────
        if sub_questions:
            with st.expander("Sub-questions (ScopingAgent)", expanded=False):
                for i, q in enumerate(sub_questions, 1):
                    st.markdown(f"**{i}.** {q}")

        # ── Citation helpers (used by synthesis, gaps, and plan) ─────────
        import re as _re

        ref_map = {}
        for i, p in enumerate(summaries, 1):
            ref_map[i] = {"title": p.get("title", ""), "url": p.get("url", "")}

        def _render_citations(text: str) -> str:
            """Replace [N] with markdown links to paper URLs."""
            def _sub(m):
                n = int(m.group(1))
                if n in ref_map:
                    u = ref_map[n]["url"]
                    return f"[\\[{n}\\]]({u})" if u else f"**[{n}]** *{ref_map[n]['title']}*"
                return m.group(0)
            return _re.sub(r"\[(\d+)\]", _sub, text)

        # ── Thematic Synthesis ───────────────────────────────────────────
        if synthesis:
            st.subheader("Thematic Synthesis")

            st.markdown(_render_citations(synthesis))

            # Citation reference list
            cited_nums = sorted({int(m) for m in _re.findall(r"\[(\d+)\]", synthesis)
                                  if m.isdigit() and int(m) in ref_map})
            if cited_nums:
                with st.expander("Citation index", expanded=False):
                    for n in cited_nums:
                        p = ref_map[n]
                        url = p["url"]
                        title = p["title"]
                        if url:
                            st.markdown(f"**[{n}]** [{title}]({url})")
                        else:
                            st.markdown(f"**[{n}]** {title}")

        else:
            st.info("No synthesis generated.")

        # ── Issues and warnings (bold, with suggestions) ──────────────────
        issues_found = contradictions or uncovered
        if issues_found:
            st.markdown("---")
            st.markdown("**Issues and potential problems**")
            if contradictions:
                for c in contradictions:
                    st.markdown(f"**⚠ Contradiction:** {_render_citations(c)}")
                    st.caption(f"Suggestion: {_suggest(c)}")
            if uncovered:
                for u in uncovered:
                    st.markdown(f"**⚠ Insufficient coverage:** sub-question '{u}' is supported by fewer than 2 papers.")
                    st.caption(f"Suggestion: {_suggest('coverage ' + u)}")

        # ── Research Gaps ────────────────────────────────────────────────
        if gaps:
            st.subheader("Research Gaps")
            st.markdown(_render_citations(gaps))

        # ── Citation Chain Graph ─────────────────────────────────────────
        if citation_edges:
            st.subheader("Citation Chain")
            st.caption(
                "Solid edges: corpus paper → reference. "
                "Dashed edges: two corpus papers share a common reference (node shown in italics)."
            )

            # Build node sets
            corpus_titles = {p.get("title", "") for p in summaries}
            nodes_map = {}   # title → {id, label, group}
            edge_rows  = []

            def _short(t, n=32):
                return t if len(t) <= n else t[:n - 1] + "…"

            _nid = [0]
            def _get_node(title, in_corpus):
                if title not in nodes_map:
                    nodes_map[title] = {
                        "id":    _nid[0],
                        "label": _short(title),
                        "group": "corpus" if in_corpus else "external",
                    }
                    _nid[0] += 1
                return nodes_map[title]["id"]

            for e in citation_edges:
                src = e["source"]
                tgt = e.get("target", e.get("shared_via", ""))
                via = e.get("shared_via")

                src_in = src in corpus_titles
                tgt_in = tgt in corpus_titles

                sid = _get_node(src, src_in)
                tid = _get_node(tgt, tgt_in)
                edge_rows.append({
                    "from": sid, "to": tid,
                    "dashes": bool(via),
                    "title": f"shared ref: {_short(via, 48)}" if via else "",
                })

            nodes_js  = str(list(nodes_map.values())).replace("'", '"')
            edges_js  = str(edge_rows).replace("'", '"').replace("True", "true").replace("False", "false")

            graph_html = f"""
<div id="cg" style="height:420px;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css"/>
<script>
var raw = {nodes_js};
var nodeData = raw.map(function(n) {{
  return {{
    id: n.id,
    label: n.label,
    color: n.group === 'corpus'
      ? {{background:'#CECBF6',border:'#534AB7',highlight:{{background:'#AFA9EC',border:'#3C3489'}}}}
      : {{background:'#F1EFE8',border:'#888780',highlight:{{background:'#D3D1C7',border:'#5F5E5A'}}}},
    font: {{ size: 12, face: 'sans-serif',
             color: n.group === 'corpus' ? '#26215C' : '#2C2C2A' }},
    shape: 'box',
    margin: 6,
    borderWidth: n.group === 'corpus' ? 2 : 1,
  }};
}});
var edgeData = {edges_js};
var edgeFormatted = edgeData.map(function(e) {{
  return {{
    from: e.from, to: e.to,
    dashes: e.dashes,
    title: e.title,
    arrows: 'to',
    color: {{ color: e.dashes ? '#888780' : '#7F77DD', opacity: 0.7 }},
    width: e.dashes ? 1 : 1.5,
  }};
}});
var container = document.getElementById('cg');
var data = {{
  nodes: new vis.DataSet(nodeData),
  edges: new vis.DataSet(edgeFormatted),
}};
var options = {{
  layout: {{ improvedLayout: true }},
  physics: {{ stabilization: {{ iterations: 120 }}, barnesHut: {{ gravitationalConstant: -3000 }} }},
  interaction: {{ tooltipDelay: 100, hover: true }},
  edges: {{ smooth: {{ type: 'cubicBezier', roundness: 0.4 }} }},
}};
new vis.Network(container, data, options);
</script>
"""
            import streamlit.components.v1 as components
            components.html(graph_html, height=440, scrolling=False)

        elif final_state.get("paper_summaries"):
            st.caption("Citation graph not available — no Semantic Scholar papers in corpus, or references could not be fetched.")

        # ── Research Plan (below synthesis) ──────────────────────────────
        st.markdown("---")
        st.subheader("Research Plan")

        if plan_prose or research_steps:
            if plan_prose:
                st.markdown(_render_citations(plan_prose))

            if research_steps:
                st.markdown("**Step-by-step research plan**")
                for step in research_steps:
                    n     = step.get("step", "")
                    title = step.get("title", "")
                    desc  = step.get("description", "")
                    gps   = step.get("grounding_papers", [])
                    fw    = step.get("future_work_link", "")

                    with st.expander(f"Step {n}: {title}", expanded=True):
                        st.markdown(_render_citations(desc))
                        if gps:
                            st.markdown("*Grounded in:* " + " · ".join(f"*{g}*" for g in gps))
                        if fw:
                            st.markdown(f"*Links to future work:* {_render_citations(fw)}")

            if risks:
                st.markdown(f"**⚠ Methodological risks:** {risks}")

        else:
            st.info("No research plan generated.")

        # ── Download ──────────────────────────────────────────────────────
        st.markdown("---")
        lit_md = (
            f"# Literature Review\n\n**Query:** {query}\n\n"
            f"## Sub-questions\n\n"
            + "\n".join(f"{i}. {q}" for i, q in enumerate(sub_questions, 1))
            + f"\n\n## Thematic Synthesis\n\n{synthesis}\n\n"
            + (f"## Issues\n\n" + "\n".join(f"- {c}" for c in contradictions + uncovered) + "\n\n" if issues_found else "")
            + f"## Research Gaps\n\n{gaps}\n\n"
            + f"## Research Plan\n\n{plan_prose}\n\n"
            + (f"## Methodological Risks\n\n{risks}" if risks else "")
        )
        st.download_button(
            "⬇️ Download literature review (.md)", data=lit_md,
            file_name="literature_review.md", mime="text/markdown",
        )

    with tab_plan:
        # Keep as a standalone tab for focused access / download
        plan = final_state.get("research_plan", "")
        research_steps = final_state.get("research_steps", [])
        risks = final_state.get("risks_and_mitigations", "")
        if plan or research_steps:
            st.markdown(_render_citations(plan) if plan else "")
            if research_steps:
                st.markdown("---")
                st.markdown("**Step-by-step plan**")
                for step in research_steps:
                    st.markdown(
                        f"**Step {step.get('step','')}: {step.get('title','')}** — "
                        f"{_render_citations(step.get('description',''))}"
                    )
                    gps = step.get("grounding_papers", [])
                    if gps:
                        st.caption("Grounded in: " + ", ".join(gps))
            if risks:
                st.markdown(f"**⚠ Methodological risks:** {_render_citations(risks)}")
        else:
            st.info("No research plan generated.")
        st.download_button(
            "⬇️ Download (.md)",
            data=f"# Research Plan\n\n**Query:** {query}\n\n{plan}",
            file_name="research_plan.md", mime="text/markdown",
        )


    with tab_papers:
        summaries  = final_state.get("paper_summaries", [])
        all_papers = final_state.get("all_papers", [])

        st.markdown(
            f"**{len(summaries)} summarised** from **{len(all_papers)} retrieved**  "
            + "  |  ".join(f"**{s}**: {c}" for s, c in src_counts.items())
        )

        rationale = final_state.get("source_rationale", {})
        if rationale:
            with st.expander("🔍 SearchReadingAgent source selection rationale", expanded=False):
                for sid, reason in rationale.items():
                    st.markdown(f"**{sid}**: {reason}")

        for i, p in enumerate(summaries, 1):
            s = p.get("summary", {})
            with st.expander(
                f"{i}. {p['title']} ({p.get('year','?')}) — "
                f"{p.get('citation_count',0)} citations [{p.get('source','')}]"
            ):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Authors:** {', '.join(p.get('authors', [])[:4])}")
                    if s.get("key_claims"):
                        st.markdown(
                            "**Key claims:** "
                            + " · ".join(f"`{c}`" for c in s["key_claims"])
                        )
                    st.markdown(f"**Methods:** {s.get('methods', '—')}")
                    st.markdown(f"**Findings:** {s.get('findings', '—')}")
                    st.markdown(f"**Future work:** {s.get('future_work', 'not stated')}")
                    if s.get("relevance_note"):
                        st.caption(f"Relevance: {s['relevance_note']}")
                with col2:
                    if p.get("url"):
                        st.markdown(f"[🔗 View]({p['url']})")
                    if p.get("pdf_url"):
                        st.markdown(f"[📄 PDF]({p['pdf_url']})")

    with tab_comms:
        st.markdown("""Every message exchanged between agents on the bus.

🟣 **task** (Orchestrator → agent)  ·  🟢 **result** (agent → Orchestrator)  ·  🔴 **feedback** (ValidationAgent → SynthesisPlanningAgent — **lateral**)  ·  🟡 **ack** (SynthesisPlanningAgent acknowledging feedback)

The **feedback → ack** exchange is the core multi-agent coordination: ValidationAgent critiques SynthesisPlanningAgent directly by name; that agent reads the critique, reasons about each issue, posts an acknowledgement, then revises its outputs — without Orchestrator mediation.""")

        MSG_STYLE = {
            "task":     ("msg-task",     "📋"),
            "result":   ("msg-result",   "✅"),
            "feedback": ("msg-feedback", "🔴"),
            "ack":      ("msg-ack",      "🤝"),
            "request":  ("msg-request",  "🔔"),
        }

        if not bus_messages:
            st.info("No messages recorded.")
        else:
            for msg in bus_messages:
                css, icon = MSG_STYLE.get(msg["msg_type"], ("msg-task", "📨"))
                content = msg["content"]
                detail  = ""

                if msg["msg_type"] == "feedback":
                    issues = content.get("issues", [])
                    detail = "<br>".join(f"  • {iss[:100]}" for iss in issues[:5])
                    if len(issues) > 5:
                        detail += f"<br>  … +{len(issues) - 5} more"
                elif msg["msg_type"] == "ack":
                    plan_txt = content.get("plan", "")
                    will_fix = content.get("will_fix", [])
                    detail   = f"Plan: {plan_txt}"
                    if will_fix:
                        detail += "<br>" + "<br>".join(
                            f"  ✓ {w[:80]}" for w in will_fix[:3]
                        )

                reply_note = (
                    f" ↩ reply to [{msg['in_reply_to']}]"
                    if msg.get("in_reply_to") else ""
                )
                st.markdown(
                    f'<div class="{css}">'
                    f'{icon} <b>[{msg["id"]}]</b> '
                    f'<b>{msg["sender"]}</b> → <b>{msg["recipient"]}</b> '
                    f'<span style="color:#666">({msg["msg_type"]}){reply_note}</span><br>'
                    f'{content.get("summary", "")}'
                    + (f"<br><small>{detail}</small>" if detail else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )

    with tab_trace:
        st.markdown("Turn-by-turn Orchestrator reasoning and tool calls.")
        for evt in final_state.get("orchestrator_trace", []):
            turn = evt.get("turn", "?")
            if evt["type"] == "reasoning":
                with st.expander(f"Turn {turn} — 💭 Reasoning", expanded=False):
                    st.markdown(evt["content"])
            elif evt["type"] == "tool_call":
                tool   = evt.get("tool", "")
                inputs = evt.get("inputs", {})
                detail = f"**{tool}**"
                if inputs.get("targeted_query"):
                    detail += f" · `{inputs['targeted_query']}`"
                if inputs.get("targeted_label"):
                    detail += f" · label=`{inputs['targeted_label']}`"
                st.markdown(f"Turn {turn} — 📨 {detail}")

    with tab_log:
        log = final_state.get("status_log", [])
        if log:
            st.code("\n".join(log), language=None)
        else:
            st.info("No log entries.")
        if final_state.get("error"):
            st.error(f"Last error: {final_state['error']}")

    with tab_hist:
        _render_history_tab(st.session_state.get("username", ""))

    st.session_state["last_result"] = final_state

    # ── Persist to history ────────────────────────────────────────────────
    _uname = st.session_state.get("username", "")
    if _uname and final_state.get("synthesis"):
        try:
            from history_store import HistoryStore
            _store    = HistoryStore(_uname)
            _saved_fn = _store.save(final_state)
            st.session_state["last_saved_file"] = _saved_fn
            st.toast(f"✅ Run saved to history as `{_saved_fn}`", icon="💾")
        except Exception as _he:
            st.caption(f"History save failed: {_he}")
