"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Modular RAG · Persistent Knowledge Maps · Conversation Explorer     ║
║                                                                              ║
║  Install:                                                                    ║
║    pip install streamlit langgraph langchain-openai langchain-core           ║
║               networkx pyvis faiss-cpu langchain-community                  ║
║               sentence-transformers tiktoken                                 ║
║                                                                              ║
║  Run:   streamlit run rag_knowledge_app.py                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict

import networkx as nx
import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from pyvis.network import Network

# ══════════════════════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR       = Path("rag_data")
MAPS_DIR       = DATA_DIR / "knowledge_maps"
CACHE_DIR      = DATA_DIR / "query_cache"
SESSIONS_DIR   = DATA_DIR / "sessions"
VECTOR_DIR     = DATA_DIR / "vectorstore"
DOCS_DIR       = DATA_DIR / "documents"

for d in [MAPS_DIR, CACHE_DIR, SESSIONS_DIR, VECTOR_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CACHE_TTL_DAYS = 20
MAX_CACHE_ENTRIES = 200

TYPE_COLORS = {
    "concept":   "#4A90D9",
    "entity":    "#E67E22",
    "fact":      "#2ECC71",
    "process":   "#E74C3C",
    "default":   "#9B59B6",
}

# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _stamp() -> str:
    return datetime.utcnow().isoformat()


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


# ─── Cache ────────────────────────────────────────────────────────────────────

def cache_save(query: str, payload: dict) -> None:
    """Save a query result to the 20-day rolling cache."""
    key  = _query_hash(query)
    path = CACHE_DIR / f"{key}.json"
    data = {"query": query, "timestamp": _stamp(), "payload": payload}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    _cache_evict()


def cache_load(query: str) -> Optional[dict]:
    """Return cached payload if it exists and is ≤ CACHE_TTL_DAYS old."""
    key  = _query_hash(query)
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    ts   = datetime.fromisoformat(data["timestamp"])
    if datetime.utcnow() - ts > timedelta(days=CACHE_TTL_DAYS):
        path.unlink(missing_ok=True)
        return None
    return data["payload"]


def _cache_evict() -> None:
    """Remove entries older than TTL and keep under MAX_CACHE_ENTRIES."""
    files = sorted(CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    cutoff = datetime.utcnow() - timedelta(days=CACHE_TTL_DAYS)
    for p in files:
        try:
            ts = datetime.fromisoformat(json.loads(p.read_text())["timestamp"])
            if ts < cutoff:
                p.unlink()
        except Exception:
            pass
    # enforce max count
    remaining = sorted(CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    while len(remaining) > MAX_CACHE_ENTRIES:
        remaining.pop(0).unlink(missing_ok=True)


def cache_list_recent(days: int = CACHE_TTL_DAYS) -> list[dict]:
    """Return all cache entries within the last `days` days, newest first."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    entries = []
    for p in CACHE_DIR.glob("*.json"):
        try:
            raw = json.loads(p.read_text())
            ts  = datetime.fromisoformat(raw["timestamp"])
            if ts >= cutoff:
                entries.append({
                    "query":     raw["query"],
                    "timestamp": ts,
                    "file":      p.name,
                })
        except Exception:
            pass
    return sorted(entries, key=lambda x: x["timestamp"], reverse=True)


# ─── Knowledge Maps ───────────────────────────────────────────────────────────

def map_save(query: str, knowledge_map: dict) -> Path:
    """Persist a knowledge map to disk, return its path."""
    key  = _query_hash(query)
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{key}.json"
    path = MAPS_DIR / name
    payload = {
        "query":      query,
        "saved_at":   _stamp(),
        "map":        knowledge_map,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def map_list() -> list[dict]:
    """Return all saved maps sorted newest-first."""
    maps = []
    for p in MAPS_DIR.glob("*.json"):
        try:
            raw = json.loads(p.read_text())
            maps.append({
                "query":    raw.get("query", ""),
                "saved_at": raw.get("saved_at", ""),
                "nodes":    len(raw.get("map", {}).get("nodes", [])),
                "edges":    len(raw.get("map", {}).get("edges", [])),
                "file":     p.name,
                "path":     p,
            })
        except Exception:
            pass
    return sorted(maps, key=lambda x: x["saved_at"], reverse=True)


def map_load(filename: str) -> Optional[dict]:
    p = MAPS_DIR / filename
    if p.exists():
        return json.loads(p.read_text())
    return None


def map_delete(filename: str) -> None:
    (MAPS_DIR / filename).unlink(missing_ok=True)


# ─── Sessions (Conversation History) ─────────────────────────────────────────

def session_save(session_id: str, turns: list[dict]) -> None:
    path = SESSIONS_DIR / f"{session_id}.json"
    path.write_text(json.dumps({"session_id": session_id,
                                "updated_at": _stamp(),
                                "turns": turns}, indent=2))


def session_load(session_id: str) -> list[dict]:
    path = SESSIONS_DIR / f"{session_id}.json"
    if path.exists():
        return json.loads(path.read_text()).get("turns", [])
    return []


def session_list() -> list[dict]:
    sessions = []
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            raw = json.loads(p.read_text())
            turns = raw.get("turns", [])
            sessions.append({
                "session_id": raw.get("session_id", p.stem),
                "updated_at": raw.get("updated_at", ""),
                "num_turns":  len(turns),
                "preview":    turns[0]["query"][:60] + "…" if turns else "(empty)",
                "file":       p.name,
            })
        except Exception:
            pass
    return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)


def session_delete(session_id: str) -> None:
    (SESSIONS_DIR / f"{session_id}.json").unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# RAG MODULE — Documents + FAISS Vector Store
# ══════════════════════════════════════════════════════════════════════════════

class RAGModule:
    """
    Modular RAG layer.
    • Indexes text documents into a FAISS vector store.
    • Retrieves top-k relevant chunks for a query.
    • Persists the index to disk; reloads automatically.
    """

    def __init__(self, api_key: str, index_path: Path = VECTOR_DIR):
        self.api_key    = api_key
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.splitter   = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=80,
        )
        self._store: Optional[FAISS] = None
        self._load_index()

    # ── Index management ────────────────────────────────────────────────────

    def _load_index(self) -> None:
        idx_file = self.index_path / "index.faiss"
        if idx_file.exists():
            try:
                self._store = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                self._store = None

    def _save_index(self) -> None:
        if self._store:
            self._store.save_local(str(self.index_path))

    def add_document(self, text: str, metadata: dict | None = None) -> int:
        """Chunk and index a text string. Returns number of chunks added."""
        chunks = self.splitter.create_documents(
            [text], metadatas=[metadata or {}]
        )
        if self._store is None:
            self._store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self._store.add_documents(chunks)
        self._save_index()
        return len(chunks)

    def add_file(self, uploaded_file) -> int:
        """Index an uploaded Streamlit file (txt or md)."""
        raw  = uploaded_file.read().decode("utf-8", errors="ignore")
        meta = {"source": uploaded_file.name, "indexed_at": _stamp()}
        # Save raw doc
        (DOCS_DIR / uploaded_file.name).write_text(raw)
        return self.add_document(raw, meta)

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """Return top-k most relevant chunks."""
        if self._store is None:
            return []
        return self._store.similarity_search(query, k=k)

    def has_documents(self) -> bool:
        return self._store is not None

    def doc_count(self) -> int:
        if self._store is None:
            return 0
        return self._store.index.ntotal

    def list_sources(self) -> list[str]:
        return [p.name for p in DOCS_DIR.iterdir() if p.is_file()]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages:       Annotated[List, operator.add]
    query:          str
    rag_context:    str          # retrieved chunks concatenated
    research_notes: str
    summary:        str
    knowledge_map:  dict
    current_agent:  str
    critique:       str          # critic feedback
    loop_count:     int


# ══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════════════════════

def make_llm(api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.3):
    return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)


# ─── Researcher ───────────────────────────────────────────────────────────────

def researcher_agent(state: AgentState, llm) -> AgentState:
    rag_block = (
        f"\n\n---\nRelevant document excerpts:\n{state['rag_context']}"
        if state.get("rag_context") else ""
    )
    critique_block = (
        f"\n\nPrevious critique (go deeper on these points): {state['critique']}"
        if state.get("critique") else ""
    )
    system = SystemMessage(content=(
        "You are a Research Agent. Given a user query (and optionally relevant "
        "document excerpts), produce comprehensive research notes covering key "
        "concepts, facts, entities, and relationships. Be thorough and precise."
    ))
    human = HumanMessage(content=(
        f"Research this topic thoroughly:\n\n{state['query']}"
        f"{rag_block}{critique_block}"
    ))
    response = llm.invoke([system, human])
    return {
        "messages":       [AIMessage(content=f"[Researcher] {response.content}")],
        "research_notes": response.content,
        "current_agent":  "researcher",
    }


# ─── Knowledge Mapper ─────────────────────────────────────────────────────────

def knowledge_mapper_agent(state: AgentState, llm) -> AgentState:
    system = SystemMessage(content=(
        "You are a Knowledge Mapping Agent. Given research notes, extract a knowledge "
        "graph. Return ONLY valid JSON with this exact schema:\n"
        '{"nodes": [{"id": "string", "label": "string", '
        '"type": "concept|entity|fact|process"}], '
        '"edges": [{"source": "string", "target": "string", "relation": "string"}]}\n'
        "Include 10–20 of the most important nodes. No text outside JSON."
    ))
    human = HumanMessage(content=f"Research notes:\n\n{state['research_notes']}")
    response = llm.invoke([system, human])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw[:-3]
    try:
        km = json.loads(raw)
    except json.JSONDecodeError:
        km = {"nodes": [], "edges": [], "error": "parse_failed"}

    return {
        "messages":      [AIMessage(content=f"[KnowledgeMapper] {len(km.get('nodes',[]))} nodes, "
                                            f"{len(km.get('edges',[]))} edges.")],
        "knowledge_map": km,
        "current_agent": "knowledge_mapper",
    }


# ─── Critic ───────────────────────────────────────────────────────────────────

def critic_agent(state: AgentState, llm) -> AgentState:
    system = SystemMessage(content=(
        "You are a Critic Agent. Review the knowledge map and research notes. "
        "If the knowledge map has fewer than 8 nodes OR is missing key relationships, "
        "return a JSON object: {\"needs_more\": true, \"feedback\": \"specific gaps\"}. "
        "Otherwise return: {\"needs_more\": false, \"feedback\": \"\"}. "
        "Only JSON, no other text."
    ))
    human = HumanMessage(content=(
        f"Nodes: {[n['label'] for n in state['knowledge_map'].get('nodes', [])]}\n"
        f"Edges: {len(state['knowledge_map'].get('edges', []))}\n"
        f"Notes length: {len(state['research_notes'])} chars"
    ))
    response = llm.invoke([system, human])
    raw = response.content.strip().lstrip("```json").rstrip("```").strip()
    try:
        result = json.loads(raw)
    except Exception:
        result = {"needs_more": False, "feedback": ""}
    return {
        "messages":      [AIMessage(content=f"[Critic] needs_more={result.get('needs_more')} | "
                                            f"{result.get('feedback','')}")],
        "critique":      result.get("feedback", ""),
        "current_agent": "critic",
        "loop_count":    state.get("loop_count", 0) + 1,
        # Embed decision in critique for the router
        "_needs_more":   result.get("needs_more", False),
    }


# ─── Summarizer ───────────────────────────────────────────────────────────────

def summarizer_agent(state: AgentState, llm) -> AgentState:
    system = SystemMessage(content=(
        "You are a Summarizer Agent. Using research notes and the knowledge map, "
        "write a clear, well-structured answer to the original query."
    ))
    human = HumanMessage(content=(
        f"Query: {state['query']}\n\n"
        f"Research notes:\n{state['research_notes']}\n\n"
        f"Knowledge map concepts: {[n['label'] for n in state['knowledge_map'].get('nodes',[])]}"
    ))
    response = llm.invoke([system, human])
    return {
        "messages":      [AIMessage(content=f"[Summarizer] {response.content}")],
        "summary":       response.content,
        "current_agent": "summarizer",
    }


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH — Build Graph with Critic Loop
# ══════════════════════════════════════════════════════════════════════════════

def _route_critic(state: AgentState) -> str:
    """Loop back to researcher if critic says needs_more (max 2 loops)."""
    if state.get("_needs_more") and state.get("loop_count", 0) < 2:
        return "researcher"
    return "summarizer"


def build_graph(api_key: str):
    llm_research = make_llm(api_key, model="gpt-4o-mini", temperature=0.4)
    llm_map      = make_llm(api_key, model="gpt-4o-mini", temperature=0.1)
    llm_critic   = make_llm(api_key, model="gpt-4o-mini", temperature=0.0)
    llm_summary  = make_llm(api_key, model="gpt-4o-mini", temperature=0.5)

    graph = StateGraph(AgentState)

    graph.add_node("researcher",       lambda s: researcher_agent(s, llm_research))
    graph.add_node("knowledge_mapper", lambda s: knowledge_mapper_agent(s, llm_map))
    graph.add_node("critic",           lambda s: critic_agent(s, llm_critic))
    graph.add_node("summarizer",       lambda s: summarizer_agent(s, llm_summary))

    graph.set_entry_point("researcher")
    graph.add_edge("researcher",       "knowledge_mapper")
    graph.add_edge("knowledge_mapper", "critic")
    graph.add_conditional_edges("critic", _route_critic,
                                 {"researcher": "researcher", "summarizer": "summarizer"})
    graph.add_edge("summarizer", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE MAP VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def render_knowledge_map(km: dict, height: int = 500) -> str:
    """Build a pyvis graph and return the rendered HTML string."""
    net = Network(
        height=f"{height}px", width="100%",
        bgcolor="#0d1117", font_color="white",
        directed=True,
    )
    net.set_options(json.dumps({
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
            "color":  {"color": "#555", "highlight": "#fff"},
            "font":   {"size": 10, "color": "#aaa"},
            "smooth": {"type": "curvedCW", "roundness": 0.2},
        },
        "nodes": {
            "font": {"size": 13, "bold": True},
            "borderWidth": 2,
            "shadow":      True,
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -60,
                "centralGravity":        0.01,
                "springLength":          200,
                "springConstant":        0.08,
            },
            "solver":        "forceAtlas2Based",
            "stabilization": {"iterations": 200},
        },
        "interaction": {"hover": True, "tooltipDelay": 150},
    }))

    for node in km.get("nodes", []):
        color = TYPE_COLORS.get(node.get("type", "default"), TYPE_COLORS["default"])
        net.add_node(
            node["id"], label=node["label"], color=color,
            title=f"<b>{node['label']}</b><br>Type: {node.get('type','?')}",
            size=25,
        )
    for edge in km.get("edges", []):
        net.add_edge(
            edge["source"], edge["target"],
            title=edge.get("relation", ""),
            label=edge.get("relation", ""),
        )

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        html = Path(f.name).read_text()
        Path(f.name).unlink(missing_ok=True)
    return html


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RAG Knowledge Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
code, pre, .stCode { font-family: 'DM Mono', monospace !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0d1117 !important; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
}

/* Tab styling */
[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Info / success boxes */
.stAlert { border-radius: 6px !important; }

/* Header */
h1, h2, h3 { font-weight: 800 !important; letter-spacing: -0.02em; }

/* Cache badge */
.cache-badge {
    display: inline-block;
    background: #1f6feb;
    color: #fff;
    font-size: 0.68rem;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.06em;
    margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Session bootstrap ─────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.utcnow().strftime("sess_%Y%m%d_%H%M%S")
if "turns" not in st.session_state:
    st.session_state.turns = []
if "rag_module" not in st.session_state:
    st.session_state.rag_module = None

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔬 RAG Knowledge Explorer")
    st.divider()

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-…")

    if api_key and (st.session_state.rag_module is None
                    or st.session_state.get("_api_key") != api_key):
        st.session_state.rag_module = RAGModule(api_key)
        st.session_state._api_key   = api_key

    rag: Optional[RAGModule] = st.session_state.rag_module

    st.divider()
    st.markdown("### 📄 Index Documents")
    uploaded = st.file_uploader("Upload .txt / .md files", type=["txt", "md"],
                                accept_multiple_files=True)
    if uploaded and rag:
        for f in uploaded:
            n = rag.add_file(f)
            st.success(f"Indexed **{f.name}** → {n} chunks")

    if rag:
        sources = rag.list_sources()
        if sources:
            with st.expander(f"📚 {len(sources)} indexed document(s)"):
                for s in sources:
                    st.markdown(f"• `{s}`")
        st.metric("Vector chunks", rag.doc_count())

    st.divider()
    st.markdown("### ⚙️ Pipeline")
    st.markdown("1. 🔍 **Researcher** — deep-dives + RAG context")
    st.markdown("2. 🗺️ **Knowledge Mapper** — extracts nodes & edges")
    st.markdown("3. 🧐 **Critic** — checks depth; loops if sparse")
    st.markdown("4. ✍️ **Summarizer** — crafts final answer")
    st.divider()
    st.caption("Node colour legend")
    for label, color in TYPE_COLORS.items():
        st.markdown(
            f'<span style="color:{color}">■</span> {label.capitalize()}',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "🚀 Research",
    "🗺️ Saved Maps",
    "💬 Conversations",
    "🕐 Cache Browser",
    "⚙️ Manage",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RESEARCH
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Research Query")

    col_q, col_opts = st.columns([3, 1])
    with col_q:
        query = st.text_area("Enter your query", height=100,
                             placeholder="e.g. How does RLHF training work?")
    with col_opts:
        use_rag      = st.checkbox("Use indexed documents (RAG)", value=True)
        use_cache    = st.checkbox("Use 20-day query cache", value=True)
        auto_save_map = st.checkbox("Auto-save knowledge map", value=True)
        rag_k        = st.slider("RAG top-k chunks", 2, 10, 5)

    run_btn = st.button(
        "🚀 Run Pipeline",
        type="primary",
        disabled=not (api_key and query),
    )

    if run_btn:
        if not api_key.startswith("sk-"):
            st.error("Invalid API key.")
            st.stop()

        # Check cache
        cached = cache_load(query) if use_cache else None
        if cached:
            st.info("⚡ Loaded from cache (≤20 days old)")
            full_state = cached
        else:
            # RAG retrieval
            rag_context = ""
            if use_rag and rag and rag.has_documents():
                docs = rag.retrieve(query, k=rag_k)
                rag_context = "\n\n---\n".join(
                    f"[Source: {d.metadata.get('source','?')}]\n{d.page_content}"
                    for d in docs
                )

            app = build_graph(api_key)

            progress = st.progress(0, text="Starting…")
            agent_pct = {
                "researcher":       ("🔍 Researcher working…",         25),
                "knowledge_mapper": ("🗺️ Mapping knowledge…",          50),
                "critic":           ("🧐 Critic reviewing…",            70),
                "summarizer":       ("✍️ Summarizer writing answer…",   90),
            }

            for event in app.stream({
                "messages":       [],
                "query":          query,
                "rag_context":    rag_context,
                "research_notes": "",
                "summary":        "",
                "knowledge_map":  {},
                "current_agent":  "",
                "critique":       "",
                "loop_count":     0,
            }):
                for node_name in event:
                    lbl, pct = agent_pct.get(node_name, ("Processing…", 50))
                    progress.progress(pct, text=lbl)

            full_state = app.invoke({
                "messages":       [],
                "query":          query,
                "rag_context":    rag_context,
                "research_notes": "",
                "summary":        "",
                "knowledge_map":  {},
                "current_agent":  "",
                "critique":       "",
                "loop_count":     0,
            })
            progress.progress(100, text="✅ Done!")

            # Save to cache
            if use_cache:
                cache_save(query, full_state)

        # Save knowledge map
        if auto_save_map and full_state.get("knowledge_map", {}).get("nodes"):
            saved_path = map_save(query, full_state["knowledge_map"])
            st.success(f"🗺️ Knowledge map saved → `{saved_path.name}`")

        # Save to conversation session
        turn = {
            "query":     query,
            "summary":   full_state.get("summary", ""),
            "timestamp": _stamp(),
            "nodes":     len(full_state.get("knowledge_map", {}).get("nodes", [])),
            "cached":    cached is not None,
        }
        st.session_state.turns.append(turn)
        session_save(st.session_state.session_id, st.session_state.turns)

        # Results
        r1, r2, r3, r4 = st.tabs(
            ["💡 Summary", "🗺️ Knowledge Map", "📝 Research Notes", "💬 Agent Log"])

        with r1:
            st.subheader("Final Answer")
            st.markdown(full_state.get("summary", ""))

        with r2:
            km = full_state.get("knowledge_map", {})
            if km.get("nodes"):
                html = render_knowledge_map(km, height=500)
                st.components.v1.html(html, height=520)
                with st.expander("📊 Raw Graph Data"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Nodes**")
                        st.dataframe(km["nodes"])
                    with c2:
                        st.markdown("**Edges**")
                        st.dataframe(km["edges"])
            else:
                st.warning("Knowledge map unavailable.")
                st.json(km)

        with r3:
            st.markdown(full_state.get("research_notes", ""))

        with r4:
            for msg in full_state.get("messages", []):
                c = msg.content
                if "[Researcher]" in c:
                    st.chat_message("assistant", avatar="🔍").write(c)
                elif "[KnowledgeMapper]" in c:
                    st.chat_message("assistant", avatar="🗺️").write(c)
                elif "[Critic]" in c:
                    st.chat_message("assistant", avatar="🧐").write(c)
                elif "[Summarizer]" in c:
                    st.chat_message("assistant", avatar="✍️").write(c)
                else:
                    st.chat_message("assistant").write(c)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SAVED MAPS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Saved Knowledge Maps")

    all_maps = map_list()
    if not all_maps:
        st.info("No knowledge maps saved yet. Run a research query first.")
    else:
        search = st.text_input("🔎 Filter maps by keyword")
        filtered = [m for m in all_maps
                    if not search or search.lower() in m["query"].lower()]

        st.caption(f"{len(filtered)} map(s) found")

        for m in filtered:
            with st.expander(
                f"🗺️ {m['query'][:70]}…  ·  "
                f"**{m['nodes']} nodes**  ·  **{m['edges']} edges**  ·  "
                f"{m['saved_at'][:16].replace('T',' ')}"
            ):
                col_view, col_del = st.columns([5, 1])
                with col_view:
                    if st.button("Load & Visualize", key=f"load_{m['file']}"):
                        raw = map_load(m["file"])
                        if raw:
                            km = raw["map"]
                            html = render_knowledge_map(km, height=450)
                            st.components.v1.html(html, height=470)
                            with st.expander("Raw JSON"):
                                st.json(km)
                with col_del:
                    if st.button("🗑️", key=f"del_{m['file']}"):
                        map_delete(m["file"])
                        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — CONVERSATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Conversation Explorer")

    col_a, col_b = st.columns([2, 3])

    with col_a:
        st.subheader("Sessions")
        all_sessions = session_list()

        # Current session
        st.markdown(f"**Current session:** `{st.session_state.session_id}`")
        st.markdown(f"Turns this session: **{len(st.session_state.turns)}**")

        if st.button("💾 Save current session"):
            session_save(st.session_state.session_id, st.session_state.turns)
            st.success("Saved!")

        if st.button("🆕 New session"):
            session_save(st.session_state.session_id, st.session_state.turns)
            st.session_state.session_id = datetime.utcnow().strftime("sess_%Y%m%d_%H%M%S")
            st.session_state.turns = []
            st.rerun()

        st.divider()
        st.markdown(f"**All saved sessions ({len(all_sessions)})**")

        selected_sid = None
        for s in all_sessions:
            label = (f"📅 {s['updated_at'][:16].replace('T',' ')}  "
                     f"·  {s['num_turns']} turn(s)\n{s['preview']}")
            if st.button(label, key=f"sess_{s['session_id']}"):
                selected_sid = s["session_id"]

    with col_b:
        st.subheader("Turns")
        display_turns = (
            session_load(selected_sid)
            if selected_sid
            else st.session_state.turns
        )

        if not display_turns:
            st.info("No turns in this session yet.")
        else:
            for i, turn in enumerate(reversed(display_turns), 1):
                cached_badge = (
                    '<span class="cache-badge">CACHED</span>'
                    if turn.get("cached") else ""
                )
                st.markdown(
                    f"**Turn {len(display_turns)-i+1}** · "
                    f"{turn['timestamp'][:16].replace('T',' ')} "
                    f"{cached_badge}",
                    unsafe_allow_html=True,
                )
                st.markdown(f"> 🔍 **Query:** {turn['query']}")
                with st.expander("📄 Summary", expanded=(i == 1)):
                    st.markdown(turn.get("summary", "*(no summary)*"))
                st.markdown(f"🗺️ Knowledge map: **{turn.get('nodes', 0)} nodes**")
                st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — CACHE BROWSER
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("20-Day Query Cache")

    recent = cache_list_recent(days=CACHE_TTL_DAYS)

    if not recent:
        st.info("Cache is empty. Run a query with caching enabled.")
    else:
        st.metric("Cached queries", len(recent))

        search_c = st.text_input("🔎 Filter cache by keyword", key="cache_search")
        show = [e for e in recent
                if not search_c or search_c.lower() in e["query"].lower()]

        for entry in show:
            age = datetime.utcnow() - entry["timestamp"]
            age_str = (
                f"{age.days}d {age.seconds//3600}h ago"
                if age.days > 0
                else f"{age.seconds//3600}h {(age.seconds%3600)//60}m ago"
            )
            with st.expander(f"🗃️ {entry['query'][:80]}…  ·  {age_str}"):
                cached_payload = cache_load(entry["query"])
                if cached_payload:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Summary length",
                              f"{len(cached_payload.get('summary',''))} chars")
                    c2.metric("Map nodes",
                              len(cached_payload.get("knowledge_map",{}).get("nodes",[])))
                    c3.metric("Research notes",
                              f"{len(cached_payload.get('research_notes',''))} chars")
                    if st.button("▶️ Reload this query result", key=f"rc_{entry['file']}"):
                        km = cached_payload.get("knowledge_map", {})
                        st.subheader("Summary")
                        st.markdown(cached_payload.get("summary",""))
                        if km.get("nodes"):
                            st.subheader("Knowledge Map")
                            html = render_knowledge_map(km, height=400)
                            st.components.v1.html(html, height=420)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — MANAGE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("Storage Management")

    col1, col2, col3 = st.columns(3)
    maps     = map_list()
    sessions = session_list()
    cached   = cache_list_recent()

    col1.metric("Saved maps",         len(maps))
    col2.metric("Saved sessions",     len(sessions))
    col3.metric("Cached queries",     len(cached))

    st.divider()

    # Delete old caches
    st.subheader("🗑️ Bulk Delete")
    delete_days = st.slider("Delete cache entries older than N days", 1, 20, 7)
    if st.button(f"Clear cache older than {delete_days} days", type="secondary"):
        cutoff = datetime.utcnow() - timedelta(days=delete_days)
        removed = 0
        for e in cache_list_recent():
            if e["timestamp"] < cutoff:
                (CACHE_DIR / e["file"]).unlink(missing_ok=True)
                removed += 1
        st.success(f"Removed {removed} cache entries.")

    st.divider()
    if st.button("🧹 Clear ALL cache", type="secondary"):
        for p in CACHE_DIR.glob("*.json"):
            p.unlink()
        st.success("Cache cleared.")

    if st.button("🗺️ Delete ALL saved maps", type="secondary"):
        for p in MAPS_DIR.glob("*.json"):
            p.unlink()
        st.success("All maps deleted.")

    if st.button("💬 Delete ALL sessions", type="secondary"):
        for p in SESSIONS_DIR.glob("*.json"):
            p.unlink()
        st.session_state.turns = []
        st.success("All sessions deleted.")

    st.divider()
    st.subheader("📂 Data Directories")
    st.code(f"""
Knowledge maps : {MAPS_DIR.resolve()}
Query cache    : {CACHE_DIR.resolve()}
Sessions       : {SESSIONS_DIR.resolve()}
Vector store   : {VECTOR_DIR.resolve()}
Documents      : {DOCS_DIR.resolve()}
""", language="text")