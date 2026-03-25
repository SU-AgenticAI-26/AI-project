"""
SCOPING AGENT:
query and produces                                  
• 3–5 sub-questions covering distinct aspects of the topic              
• 4–8 search keywords for academic API queries                          

Both are written into the shared LangGraph state so every downstream agent (Router, VectorDB, SQL, Web) can use them.                          ║
Pipeline position:  User Query → [Scoping] → [Router] → retrieval agents
Run:   python -m streamlit run scoping_agent.py                                
"""

from __future__ import annotations

import json
import operator
import os
import streamlit as st

from typing import Annotated, List, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


# ══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# Extends the existing AgentState with two new fields:
#   sub_questions: list of 3–5 focused sub-questions
#   search_keywords: list of 4–8 academic search keywords
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages:         Annotated[List, operator.add]
    query:            str
  
    # ── NEW fields added by Scoping Agent ─────────────────────────────────────
    sub_questions:    List[str]   # 3–5 sub-questions for downstream agents
    search_keywords:  List[str]   # 4–8 keywords for API searches
    scoping_summary:  str         # one-sentence description of query scope
  
    # ── Existing fields (Router reads these) ──────────────────────────────────
    active_agents:    List[str]
    router_reasoning: str
  
    # ── Activity log ──────────────────────────────────────────────────────────
    activity_log:     Annotated[List, operator.add]
    current_agent:    str


# ══════════════════════════════════════════════════════════════════════════════
# SCOPING AGENT
# ══════════════════════════════════════════════════════════════════════════════

def scoping_agent(state: AgentState, model: ChatOpenAI) -> dict:
    """
    Layer 1 — first agent in the pipeline.

    Reads:  state['query']
    Writes: state['sub_questions'], state['search_keywords'], state['scoping_summary']

    The Router and all retrieval agents can then read sub_questions and
    search_keywords from the shared state instead of receiving only the raw query.
    """
    system = SystemMessage(content=(
        "You are a Scoping Agent. Your job is to analyse a research query and "
        "decompose it into structured components that downstream search agents can use.\n\n"
        "Return ONLY valid JSON with exactly these fields:\n"
        "{\n"
        '  "sub_questions": [<3 to 5 strings — distinct aspects of the topic>],\n'
        '  "search_keywords": [<4 to 8 strings — concise academic search terms>],\n'
        '  "scoping_summary": "<one sentence describing what this query is really asking>"\n'
        "}\n\n"
        "Rules:\n"
        "- sub_questions must each address a DIFFERENT facet of the topic\n"
        "- search_keywords should be short (1-3 words each), suitable for API queries\n"
        "- Do not include any text outside the JSON\n"
    ))

    resp = model.invoke([
        system,
        HumanMessage(content=f"Research query: {state['query']}")
    ])

    raw = resp.content.strip()
    # Strip markdown code fences if model wraps output
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("```").strip()

    try:
        parsed = json.loads(raw)
        sub_questions   = parsed.get("sub_questions", [])[:5]
        search_keywords = parsed.get("search_keywords", [])[:8]
        scoping_summary = parsed.get("scoping_summary", "")
    except Exception:
        # Graceful fallback — pipeline can still continue with raw query
        sub_questions   = [state["query"]]
        search_keywords = state["query"].lower().split()[:8]
        scoping_summary = f"Could not parse scope for: {state['query']}"

    log_entry = {
        "agent":  "scoping",
        "icon":   "🔭",
        "title":  "Scoping Agent decomposed the query",
        "detail": (
            f"Summary: {scoping_summary}\n"
            f"Sub-questions ({len(sub_questions)}): "
            + " | ".join(f"({i+1}) {q}" for i, q in enumerate(sub_questions))
            + f"\nKeywords ({len(search_keywords)}): "
            + ", ".join(search_keywords)
        ),
        "sub_questions":   sub_questions,
        "search_keywords": search_keywords,
    }

    return {
        "sub_questions":   sub_questions,
        "search_keywords": search_keywords,
        "scoping_summary": scoping_summary,
        "messages":        [AIMessage(content=f"[Scoping] {scoping_summary}")],
        "activity_log":    [log_entry],
        "current_agent":   "scoping",
    }


# ══════════════════════════════════════════════════════════════════════════════
# MOCK ROUTER: simulates what the real Router will do with Scoping output
# ══════════════════════════════════════════════════════════════════════════════

def mock_router_agent(state: AgentState, model: ChatOpenAI) -> dict:
    """
    Simulates the Router agent receiving the Scoping Agent's output.
    In the real pipeline this is replaced by the full router_agent().
    Shows how the Router uses sub_questions + search_keywords from Scoping.
    """
    system = SystemMessage(content=(
        "You are a Router Agent. Given a research query AND its scoped breakdown, "
        "decide which search agents to activate.\n"
        "Available: 'vector_db', 'sql_db', 'web'\n"
        "Return ONLY JSON: {\"agents\": [...], \"reasoning\": \"one sentence\"}."
    ))

    context = (
        f"Original query: {state['query']}\n"
        f"Scope summary: {state.get('scoping_summary', '')}\n"
        f"Sub-questions: {state.get('sub_questions', [])}\n"
        f"Keywords: {state.get('search_keywords', [])}"
    )

    resp  = model.invoke([system, HumanMessage(content=context)])
    raw   = resp.content.strip().lstrip("```json").rstrip("```").strip()

    try:
        parsed = json.loads(raw)
        agents = parsed.get("agents", ["vector_db", "sql_db"])
        reason = parsed.get("reasoning", "")
    except Exception:
        agents = ["vector_db", "sql_db"]
        reason = "defaulted"

    return {
        "active_agents":    agents,
        "router_reasoning": reason,
        "messages":         [AIMessage(content=f"[Router] {reason} → {agents}")],
        "activity_log":     [{
            "agent":  "router",
            "icon":   "🔀",
            "title":  "Router decided (informed by Scoping output)",
            "detail": f"Activating: {', '.join(agents)} — {reason}",
        }],
        "current_agent": "router",
    }


# ══════════════════════════════════════════════════════════════════════════════
# MINI GRAPH: Scoping → Router → END
# ══════════════════════════════════════════════════════════════════════════════

def build_test_graph(api_key: str):
    model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.2)

    g = StateGraph(AgentState)
    g.add_node("scoping", lambda s: scoping_agent(s, model))
    g.add_node("router",  lambda s: mock_router_agent(s, model))

    g.set_entry_point("scoping")
    g.add_edge("scoping", "router")
    g.add_edge("router",  END)

    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT TEST UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Scoping Agent Test", page_icon="🔭", layout="wide")
st.title("🔭 Scoping Agent — Standalone Test")
st.caption("Tests the Scoping Agent in isolation before integrating into the full pipeline.")

with st.sidebar:
    st.header("⚙️ Config")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-…")
    if not api_key:
        st.warning("Enter your OpenAI API key.")
        st.stop()
    st.success("✅ Ready")
    st.divider()
    st.markdown("### What this tests")
    st.markdown("""
    **Input:** raw user query  
    **Output:**
    - 3–5 sub-questions
    - 4–8 search keywords  
    - Scope summary  
    - Router decision (using scoping output)
    
    **Pipeline position:**  
    `User Query → Scoping → Router → ...`
    """)

st.divider()

query = st.text_area(
    "Enter a research query to test",
    height=100,
    placeholder="e.g. How does RAG relate to transformer architecture?",
    value="What are the key differences between RLHF and traditional supervised fine-tuning for large language models?"
)

col1, col2 = st.columns([1, 3])
with col1:
    run = st.button("🔭 Run Scoping Agent", type="primary", disabled=not query)

if run:
    with st.spinner("Running Scoping Agent → Router…"):
        app = build_test_graph(api_key)

        # Accumulate state across both agents
        full_state = {
            "messages": [], "query": query,
            "sub_questions": [], "search_keywords": [],
            "scoping_summary": "", "active_agents": [],
            "router_reasoning": "", "activity_log": [],
            "current_agent": "",
        }
        for event in app.stream(full_state.copy()):
            for node, state_update in event.items():
                for key, val in state_update.items():
                    if key in ("messages", "activity_log") and isinstance(val, list):
                        full_state[key] = full_state.get(key, []) + val
                    else:
                        full_state[key] = val

    st.success("✅ Done!")
    st.divider()

    # ── Results ───────────────────────────────────────────────────────────────
    tab_scope, tab_router, tab_integration = st.tabs([
        "🔭 Scoping Output", "🔀 Router Decision", "🔗 Integration Guide"
    ])

    with tab_scope:
        st.subheader("Scope Summary")
        st.info(full_state.get("scoping_summary", ""))

        st.subheader("Sub-Questions")
        sub_qs = full_state.get("sub_questions", [])
        if sub_qs:
            for i, q in enumerate(sub_qs, 1):
                st.markdown(f"**{i}.** {q}")
        else:
            st.warning("No sub-questions generated.")

        st.subheader("Search Keywords")
        keywords = full_state.get("search_keywords", [])
        if keywords:
            cols = st.columns(min(len(keywords), 4))
            for i, kw in enumerate(keywords):
                cols[i % 4].markdown(
                    f'<span style="background:#1a3a5c;color:#4A90D9;padding:4px 10px;'
                    f'border-radius:12px;font-size:0.85rem;display:inline-block;margin:3px">'
                    f'{kw}</span>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("No keywords generated.")

        st.divider()
        st.subheader("Raw JSON Output")
        st.json({
            "sub_questions":   sub_qs,
            "search_keywords": keywords,
            "scoping_summary": full_state.get("scoping_summary", ""),
        })

    with tab_router:
        st.subheader("Router Decision (informed by Scoping)")
        st.markdown(f"**Agents activated:** `{full_state.get('active_agents', [])}`")
        st.markdown(f"**Reasoning:** {full_state.get('router_reasoning', '')}")

        st.divider()
        st.subheader("How Scoping helped the Router")
        st.markdown("""
        Without Scoping, the Router only sees:
        ```
        Query: "What are the key differences between RLHF and..."
        ```
        With Scoping, the Router sees:
        ```
        Query: original query
        Scope summary: ...
        Sub-questions: [1. ..., 2. ..., 3. ...]
        Keywords: [RLHF, fine-tuning, reward model, ...]
        ```
        This means the Router can make a **more informed decision** about
        which agents to activate and why.
        """)

    with tab_integration:
        st.subheader("How to integrate into streamlit_app.py")
        st.markdown("""
        Once tested, adding the Scoping Agent to the full pipeline requires
        **4 small changes** to `streamlit_app.py`:

        **1. Add new fields to `AgentState`:**
        ```python
        sub_questions:    List[str]
        search_keywords:  List[str]
        scoping_summary:  str
        ```

        **2. Copy the `scoping_agent()` function** from this file into `streamlit_app.py`

        **3. Register the node and connect it in `build_graph()`:**
        ```python
        g.add_node("scoping", lambda s: scoping_agent(s, lm_r))
        g.set_entry_point("scoping")        # scoping runs first
        g.add_edge("scoping", "router")     # then router
        # rest of edges unchanged
        ```

        **4. Update the initial state dict** when calling `app.stream()`:
        ```python
        full_state = {
            ...existing fields...,
            "sub_questions": [],
            "search_keywords": [],
            "scoping_summary": "",
        }
        ```

        **5. Update `pct_map`** to include scoping in the progress bar:
        ```python
        pct_map = {
            "scoping": 8,       # add this
            "router": 15,       # bump existing ones down slightly
            "vector_db": 28,
            ...
        }
        ```
        """)

        st.divider()
        st.subheader("How retrieval agents use the keywords")
        st.markdown("""
        After integration, each retrieval agent can use `search_keywords`
        instead of the raw query for better results:

        ```python
        # In vector_db_agent:
        keywords = state.get("search_keywords", [])
        search_query = " ".join(keywords) if keywords else state["query"]
        docs = vdb.search(search_query, k=6)

        # In web_agent (OpenAlex, Crossref, arXiv):
        search_query = " ".join(state.get("search_keywords", [state["query"]])[:5])
        ```
        """)
