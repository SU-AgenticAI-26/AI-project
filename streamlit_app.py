"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        Collaborative Multi-Agent RAG  —  Single-File Edition                 ║
║                                                                              ║
║  FIXES APPLIED:                                                              ║
║  1. Double pipeline run removed — stream now captures final state directly   ║
║  2. VectorDB persistence — reindex_saved_docs() rebuilds index on startup    ║
║  3. Web agent now queries OpenAlex + Crossref + Semantic Scholar + arXiv     ║
║                                                                              ║
║  Install (core):                                                             ║
║    pip install streamlit langgraph langchain-openai langchain-core           ║
║               langchain-community faiss-cpu langchain-text-splitters         ║
║               networkx pyvis tiktoken arxiv requests                         ║
║                                                                              ║
║  Optional providers:                                                         ║
║    Gemini:  pip install langchain-google-genai                               ║
║    Claude:  pip install langchain-anthropic                                  ║
║    Local embeddings (Claude/Local provider):                                 ║
║             pip install sentence-transformers                                ║
║                                                                              ║
║  Run:   streamlit run streamlit_app.py                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture
────────────
  User Query
      │
  [Router Agent]  ← decides which agents to activate (all / subset)
      │
  ┌───┼────────────────┐
  ▼   ▼                ▼
[VectorDB Agent] [SQL/DB Agent] [Web/API Agent]
  │   │                │
  └───┴────────────────┘
          │
    [Orchestrator Agent]  ← merges, deduplicates, weights evidence
          │
    [Knowledge Mapper]  ← builds graph; loops back if sparse
          │
    [Summarizer]  ← final grounded answer
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
import sqlite3
import tempfile
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict

import requests
import streamlit as st
from pyvis.network import Network

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

ROOT        = Path("collab_rag_data")
VECTOR_DIR  = ROOT / "vectorstore"
DOCS_DIR    = ROOT / "documents"
MAPS_DIR    = ROOT / "knowledge_maps"
CACHE_DIR   = ROOT / "cache"
SESSIONS_DIR = ROOT / "sessions"
SQL_DB_PATH = ROOT / "knowledge.db"

for _d in [VECTOR_DIR, DOCS_DIR, MAPS_DIR, CACHE_DIR, SESSIONS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

CACHE_TTL_DAYS = 20

# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER CONFIG
# ══════════════════════════════════════════════════════════════════════════════

PROVIDER_OPENAI = "OpenAI"
PROVIDER_GEMINI = "Google Gemini"
PROVIDER_CLAUDE = "Anthropic Claude"
PROVIDER_LOCAL  = "Local (llama.cpp / Ollama / LMStudio)"

_DEFAULT_MODELS = {
    PROVIDER_OPENAI: "gpt-4o-mini",
    PROVIDER_GEMINI: "gemini-2.0-flash",
    PROVIDER_CLAUDE: "claude-haiku-4-5-20251001",
    PROVIDER_LOCAL:  "",
}

_DEFAULT_BASE_URL = "http://localhost:8080/v1"  # llama.cpp server default


@dataclass
class ProviderConfig:
    provider: str
    api_key:  str
    model:    str
    base_url: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages:           Annotated[List, operator.add]
    query:              str
    active_agents:      List[str]
    router_reasoning:   str
    vector_findings:    str
    sql_findings:       str
    web_findings:       str
    activity_log:       Annotated[List, operator.add]
    merged_context:     str
    knowledge_map:      dict
    critique:           str
    loop_count:         int
    summary:            str
    current_agent:      str


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _stamp() -> str:
    return datetime.utcnow().isoformat()

def _hash(q: str) -> str:
    return hashlib.sha256(q.strip().lower().encode()).hexdigest()[:16]

# ── Cache ─────────────────────────────────────────────────────────────────────

def cache_save(query: str, payload: dict) -> None:
    p = CACHE_DIR / f"{_hash(query)}.json"

    # Strip LangChain message objects — they're not needed in cache
    safe = {k: v for k, v in payload.items() if k != "messages"}
    p.write_text(json.dumps({"query": query, "ts": _stamp(), "payload": safe}, indent=2))

def cache_load(query: str) -> Optional[dict]:
    p = CACHE_DIR / f"{_hash(query)}.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    if datetime.utcnow() - datetime.fromisoformat(raw["ts"]) > timedelta(days=CACHE_TTL_DAYS):
        p.unlink(missing_ok=True)
        return None
    return raw["payload"]

def cache_list() -> list[dict]:
    out = []
    for p in CACHE_DIR.glob("*.json"):
        try:
            raw = json.loads(p.read_text())
            out.append({"query": raw["query"], "ts": raw["ts"], "file": p.name})
        except Exception:
            pass
    return sorted(out, key=lambda x: x["ts"], reverse=True)

# ── Knowledge Maps ─────────────────────────────────────────────────────────────

def map_save(query: str, km: dict) -> str:
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{_hash(query)}.json"
    (MAPS_DIR / name).write_text(
        json.dumps({"query": query, "saved_at": _stamp(), "map": km}, indent=2)
    )
    return name

def map_list() -> list[dict]:
    out = []
    for p in MAPS_DIR.glob("*.json"):
        try:
            raw = json.loads(p.read_text())
            out.append({
                "query":    raw.get("query", ""),
                "saved_at": raw.get("saved_at", ""),
                "nodes":    len(raw.get("map", {}).get("nodes", [])),
                "edges":    len(raw.get("map", {}).get("edges", [])),
                "file":     p.name,
            })
        except Exception:
            pass
    return sorted(out, key=lambda x: x["saved_at"], reverse=True)

def map_load(filename: str) -> Optional[dict]:
    p = MAPS_DIR / filename
    return json.loads(p.read_text()) if p.exists() else None

# ── Sessions ───────────────────────────────────────────────────────────────────

def session_save(sid: str, turns: list) -> None:
    (SESSIONS_DIR / f"{sid}.json").write_text(
        json.dumps({"sid": sid, "updated": _stamp(), "turns": turns}, indent=2)
    )

def session_load(sid: str) -> list:
    p = SESSIONS_DIR / f"{sid}.json"
    return json.loads(p.read_text()).get("turns", []) if p.exists() else []

def session_list() -> list[dict]:
    out = []
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            raw   = json.loads(p.read_text())
            turns = raw.get("turns", [])
            out.append({
                "sid":     raw.get("sid", p.stem),
                "updated": raw.get("updated", ""),
                "n":       len(turns),
                "preview": turns[0]["query"][:55] + "…" if turns else "(empty)",
                "file":    p.name,
            })
        except Exception:
            pass
    return sorted(out, key=lambda x: x["updated"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# SQL MODULE
# ══════════════════════════════════════════════════════════════════════════════

def init_sql_db() -> None:
    con = sqlite3.connect(SQL_DB_PATH)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS topics (
        id INTEGER PRIMARY KEY,
        title TEXT, category TEXT, summary TEXT, keywords TEXT, created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS relationships (
        id INTEGER PRIMARY KEY,
        from_topic TEXT, to_topic TEXT, relation_type TEXT
    );
    CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY,
        subject TEXT, predicate TEXT, object TEXT, confidence REAL, source TEXT
    );
    """)
    if cur.execute("SELECT COUNT(*) FROM topics").fetchone()[0] == 0:
        cur.executemany(
            "INSERT INTO topics (title,category,summary,keywords,created_at) VALUES (?,?,?,?,?)",
            [
                ("Transformer Architecture", "ML",
                 "Self-attention-based sequence model with encoder-decoder structure.",
                 "transformer,attention,neural network,NLP", _stamp()),
                ("RLHF", "ML",
                 "Reinforcement Learning from Human Feedback — aligns LLMs via preference data.",
                 "RLHF,alignment,reward model,fine-tuning", _stamp()),
                ("RAG", "ML",
                 "Retrieval-Augmented Generation combines retrieval with generation.",
                 "RAG,retrieval,generation,vector search", _stamp()),
                ("Gradient Descent", "Math",
                 "Iterative optimisation algorithm minimising a loss function.",
                 "gradient,optimisation,loss,backprop", _stamp()),
                ("Attention Mechanism", "ML",
                 "Soft-alignment computing weighted sums over key-value pairs given a query.",
                 "attention,query,key,value,softmax", _stamp()),
                ("Knowledge Graph", "Data",
                 "Graph structure representing entities as nodes and relationships as edges.",
                 "graph,entity,relation,knowledge", _stamp()),
            ],
        )
        cur.executemany(
            "INSERT INTO relationships (from_topic,to_topic,relation_type) VALUES (?,?,?)",
            [
                ("RAG",                "Transformer Architecture", "uses"),
                ("RLHF",               "Transformer Architecture", "fine-tunes"),
                ("Attention Mechanism", "Transformer Architecture", "core component of"),
                ("Gradient Descent",   "RLHF",                    "optimises reward model in"),
            ],
        )
        cur.executemany(
            "INSERT INTO facts (subject,predicate,object,confidence,source) VALUES (?,?,?,?,?)",
            [
                ("Transformer", "introduced in", "Attention Is All You Need (2017)", 0.99, "Vaswani et al."),
                ("RLHF",        "used by",       "InstructGPT, ChatGPT, Claude",    0.99, "OpenAI/Anthropic"),
                ("RAG",         "retriever",     "dense passage retrieval or BM25",  0.95, "Lewis et al. 2020"),
                ("Attention",   "complexity",    "O(n²) in sequence length",         0.99, "Transformer paper"),
            ],
        )
    con.commit()
    con.close()


def sql_search(query: str, k: int = 8) -> str:
    con   = sqlite3.connect(SQL_DB_PATH)
    cur   = con.cursor()
    words = [w.strip("?.!,") for w in query.lower().split() if len(w) > 3]
    results = []
    for word in words[:5]:
        for row in cur.execute(
            "SELECT title,category,summary FROM topics WHERE "
            "LOWER(title) LIKE ? OR LOWER(keywords) LIKE ? OR LOWER(summary) LIKE ?",
            (f"%{word}%", f"%{word}%", f"%{word}%"),
        ).fetchall():
            results.append(f"[TOPIC] {row[0]} ({row[1]}): {row[2]}")
        for row in cur.execute(
            "SELECT from_topic,relation_type,to_topic FROM relationships WHERE "
            "LOWER(from_topic) LIKE ? OR LOWER(to_topic) LIKE ?",
            (f"%{word}%", f"%{word}%"),
        ).fetchall():
            results.append(f"[REL] {row[0]} —[{row[1]}]→ {row[2]}")
        for row in cur.execute(
            "SELECT subject,predicate,object,source FROM facts WHERE "
            "LOWER(subject) LIKE ? OR LOWER(object) LIKE ?",
            (f"%{word}%", f"%{word}%"),
        ).fetchall():
            results.append(f"[FACT] {row[0]} {row[1]} '{row[2]}' (source: {row[3]})")
    con.close()
    seen, unique = set(), []
    for r in results:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return "\n".join(unique[:k]) if unique else "(no SQL results)"


def sql_insert_topic(title: str, category: str, summary: str, keywords: str) -> str:
    con = sqlite3.connect(SQL_DB_PATH)
    con.execute(
        "INSERT INTO topics (title,category,summary,keywords,created_at) VALUES (?,?,?,?,?)",
        (title, category, summary, keywords, _stamp()),
    )
    con.commit()
    con.close()
    return f"Inserted: {title}"

def sql_list_topics() -> list[dict]:
    con  = sqlite3.connect(SQL_DB_PATH)
    rows = con.execute(
        "SELECT id,title,category,keywords FROM topics ORDER BY id DESC"
    ).fetchall()
    con.close()
    return [{"id": r[0], "title": r[1], "category": r[2], "keywords": r[3]} for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR DB MODULE
# ══════════════════════════════════════════════════════════════════════════════

class VectorDBModule:
    def __init__(self, embeddings, vector_dir: Path = VECTOR_DIR):
        self.embeddings = embeddings
        self.vector_dir = vector_dir
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.splitter   = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        self._store: Optional[FAISS] = None
        self._load()

    def _load(self) -> None:
        if (self.vector_dir / "index.faiss").exists():
            try:
                self._store = FAISS.load_local(
                    str(self.vector_dir), self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                self._store = None

    def _save(self) -> None:
        if self._store:
            self._store.save_local(str(self.vector_dir))

    def add_text(self, text: str, meta: dict | None = None) -> int:
        chunks = self.splitter.create_documents([text], metadatas=[meta or {}])
        if self._store is None:
            self._store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self._store.add_documents(chunks)
        self._save()
        return len(chunks)

    def add_file(self, f) -> int:
        raw = f.read().decode("utf-8", errors="ignore")
        (DOCS_DIR / f.name).write_text(raw)
        return self.add_text(raw, {"source": f.name, "indexed_at": _stamp()})

    def search(self, query: str, k: int = 5) -> list[Document]:
        if self._store is None:
            return []
        return self._store.similarity_search(query, k=k)

    def count(self) -> int:
        return self._store.index.ntotal if self._store else 0

    def sources(self) -> list[str]:
        return [p.name for p in DOCS_DIR.iterdir() if p.is_file()]

    # FIX 1: Re-index saved docs on startup 
    # Rebuild FAISS index from docs already saved to DOCS_DIR.
    # Fixes the Codespace restart problem where the FAISS index gets wiped,
    # but the raw document files survive because they are tracked by git.
    def reindex_saved_docs(self) -> int:
        if self.count() > 0:
            return 0  # index already has data so skip
          
        count = 0
        for p in DOCS_DIR.iterdir():
            if p.suffix in (".txt", ".md"):
                try:
                    text = p.read_text(errors="ignore")
                    self.add_text(text, {"source": p.name, "indexed_at": _stamp()})
                    count += 1
                except Exception:
                    pass
                  
        return count


# ══════════════════════════════════════════════════════════════════════════════
# ARXIV INDEXER  (stdlib only — no extra deps)
# ══════════════════════════════════════════════════════════════════════════════

def index_arxiv_documents(query: str, vdb: VectorDBModule, limit: int = 10) -> int:
    """Search arXiv via its public API and index abstracts into the vector DB."""
  
    encoded = urllib.parse.quote(query)
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query=all:{encoded}&start=0&max_results={limit}"
        f"&sortBy=relevance&sortOrder=descending"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            xml = resp.read().decode("utf-8")
    except Exception:
        return 0

    import xml.etree.ElementTree as ET
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root    = ET.fromstring(xml)
        entries = root.findall("atom:entry", ns)
    except Exception:
        return 0

    indexed = 0
    for entry in entries:
        try:
            title    = (entry.findtext("atom:title",   "", ns) or "").strip().replace("\n", " ")
            summary  = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
            entry_id = (entry.findtext("atom:id",      "", ns) or "").strip()
            authors  = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
            if not title:
                continue
            doc = f"Title: {title}\nAuthors: {', '.join(authors)}\nAbstract: {summary}"
            vdb.add_text(doc, {
              "source": "arXiv", 
              "title": title, 
              "url": entry_id, 
              "indexed_at": _stamp()
            })
            indexed += 1
        except Exception:
            continue
    return indexed


# ══════════════════════════════════════════════════════════════════════════════
# LLM + EMBEDDINGS FACTORIES
# ══════════════════════════════════════════════════════════════════════════════

def _llm(cfg: ProviderConfig, temperature: float = 0.3) -> BaseChatModel:
    """Return a LangChain chat model for the configured provider."""
    if cfg.provider == PROVIDER_OPENAI:
        return ChatOpenAI(api_key=cfg.api_key, model=cfg.model, temperature=temperature)
    elif cfg.provider == PROVIDER_GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        return ChatGoogleGenerativeAI(
            google_api_key=cfg.api_key, model=cfg.model, temperature=temperature
        )
    elif cfg.provider == PROVIDER_CLAUDE:
        from langchain_anthropic import ChatAnthropic  # type: ignore
        return ChatAnthropic(api_key=cfg.api_key, model=cfg.model, temperature=temperature)
    else:  # PROVIDER_LOCAL — llama.cpp / Ollama / LMStudio all expose OpenAI-compatible API
        return ChatOpenAI(
            base_url=cfg.base_url or _DEFAULT_BASE_URL,
            api_key=cfg.api_key or "no-key",
            model=cfg.model or "local-model",
            temperature=temperature,
        )


def _embeddings(cfg: ProviderConfig):
    """Return a LangChain embeddings object for the configured provider."""
    if cfg.provider == PROVIDER_OPENAI:
        return OpenAIEmbeddings(api_key=cfg.api_key)
    elif cfg.provider == PROVIDER_GEMINI:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
        return GoogleGenerativeAIEmbeddings(
            google_api_key=cfg.api_key, model="models/text-embedding-004"
        )
    else:  # Claude or Local — use a free local model via sentence-transformers
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _embedding_key(cfg: ProviderConfig) -> str:
    """Return a stable, filesystem-safe identifier for the embedding backend.

    Used to namespace the FAISS vector store directory so that indices built
    with different embedding models (and therefore different vector dimensions)
    are stored separately and never mixed up.
    """
    if cfg.provider == PROVIDER_OPENAI:
        return "openai-text-embedding-ada-002"
    elif cfg.provider == PROVIDER_GEMINI:
        return "gemini-text-embedding-004"
    else:  # Claude or Local — HuggingFace sentence-transformers
        return "huggingface-all-MiniLM-L6-v2"


# ══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════════════════════

def router_agent(state: AgentState, model: BaseChatModel) -> dict:
    system = SystemMessage(content=(
        "You are a Router Agent. Given a user query, decide which search agents to activate.\n"
        "Available: 'vector_db' (semantic doc search), 'sql_db' (structured facts/topics), "
        "'web' (live scholarly search — use when query needs recent papers or external knowledge).\n"
        "Return ONLY JSON: {\"agents\": [...], \"reasoning\": \"one sentence\"}. No other text."
    ))
    resp = model.invoke([system, HumanMessage(content=f"Query: {state['query']}")])
    raw  = resp.content.strip().lstrip("```json").rstrip("```").strip()
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
          "agent": "router", 
          "icon": "🔀", 
          "title": "Router decided",
          "detail": f"Activating: {', '.join(agents)} — {reason}", 
          "ts": _stamp(),
        }],
        "current_agent": "router",
    }

# ── 2. Vector DB Agent ────────────────────────────────────────────────────────
def vector_db_agent(state: AgentState, model: BaseChatModel, vdb: VectorDBModule) -> dict:
    if "vector_db" not in state.get("active_agents", []):
        return {
            "vector_findings": "(not activated)",
            "messages":        [AIMessage(content="[VectorDB] skipped")],
            "activity_log":    [{
              "agent": "vector_db", 
              "icon": "🗂️",
              "title": "Vector DB — skipped",
              "detail": "Router did not activate.", 
              "ts": _stamp()}],
            "current_agent": "vector_db",
        }

    docs = vdb.search(state["query"], k=6)
    if not docs:
        raw_ctx = "(no documents indexed)"
        sources = []
    else:
        raw_ctx = "\n\n---\n".join(
            f"[{d.metadata.get('source','?')}]\n{d.page_content}" for d in docs
        )
        sources = list({d.metadata.get("source", "?") for d in docs})

    system = SystemMessage(content=(
        "You are a Vector DB Search Agent. Synthesise the retrieved document chunks into "
        "structured research notes relevant to the query. Include facts, definitions, relationships."
    ))
    resp = model.invoke([system, HumanMessage(content=f"Query: {state['query']}\n\nChunks:\n{raw_ctx}")])

    return {
        "vector_findings": resp.content,
        "messages":        [AIMessage(content=f"[VectorDB] {resp.content[:120]}…")],
        "activity_log":    [{
            "agent":  "vector_db", "icon": "🗂️", 
            "title": "Vector DB agent",
            "detail": f"Retrieved {len(docs)} chunks from {len(sources)} source(s): {', '.join(sources) or 'none'}",
            "chunks": [{"source": d.metadata.get("source","?"), "text": d.page_content[:200]+"…"} for d in docs],
            "ts": _stamp(),
        }],
        "current_agent": "vector_db",
    }


def sql_db_agent(state: AgentState, model: BaseChatModel) -> dict:
    if "sql_db" not in state.get("active_agents", []):
        return {
            "sql_findings": "(not activated)",
            "messages":     [AIMessage(content="[SQLDB] skipped")],
            "activity_log": [{
              "agent": "sql_db", "icon": "🗄️",
              "title": "SQL DB — skipped","detail": "Router did not activate.", 
              "ts": _stamp()}],
            "current_agent": "sql_db",
        }

    raw  = sql_search(state["query"], k=10)
    rows = [l for l in raw.split("\n") if l.strip()]
    system = SystemMessage(content=(
        "You are a SQL Database Agent. Given a query and raw SQL results "
        "(topics, relationships, facts), extract the most relevant structured information."
    ))
    resp = model.invoke([system, HumanMessage(content=f"Query: {state['query']}\n\nSQL results:\n{raw}")])

    return {
        "sql_findings": resp.content,
        "messages":     [AIMessage(content=f"[SQLDB] {resp.content[:120]}…")],
        "activity_log": [{
            "agent": "sql_db", "icon": "🗄️", 
            "title": "SQL / DB agent",
            "detail": f"{len(rows)} row(s) matched across topics, relationships, facts tables",
            "rows": rows[:12], "ts": _stamp(),
        }],
        "current_agent": "sql_db",
    }

# ── 4. Web / arXiv Agent ──────────────────────────────────────────────────────
# FIX 3: Web agent now queries OpenAlex + Crossref + Semantic Scholar + arXiv
def web_agent(state: AgentState, model: BaseChatModel, vdb: VectorDBModule) -> dict:
    if "web" not in state.get("active_agents", []):
        return {
            "web_findings": "(not activated)",
            "messages":     [AIMessage(content="[Web] skipped")],
            "activity_log": [{"agent": "web", "icon": "🌐",
                               "title": "Web agent — skipped",
                               "detail": "Router did not activate.", "ts": _stamp()}],
            "current_agent": "web",
        }

    query        = state["query"]
    results_text = []
    indexed      = 0
    sources_used = []

    # OpenAlex ──────────────────────────────────────────────────────────────
    try:
        r = requests.get("https://api.openalex.org/works",
                         params={"search": query, "per-page": 5, "mailto": "research@example.com"},
                         timeout=10)
        if r.ok:
            for item in r.json().get("results", []):
                title   = item.get("title", "No title")
                year    = item.get("publication_year", "")
                authors = [a.get("author", {}).get("display_name", "")
                           for a in item.get("authorships", [])[:3]]
                results_text.append(f"[OpenAlex] {title} ({year}) — {', '.join(authors)}")
                vdb.add_text(f"Title: {title}\nAuthors: {', '.join(authors)}\nYear: {year}",
                             {"source": "openalex", "title": title})
                indexed += 1
            sources_used.append("OpenAlex")
    except Exception as e:
        results_text.append(f"[OpenAlex error] {e}")

    # ── Crossref ──────────────────────────────────────────────────────────────
    try:
        r = requests.get("https://api.crossref.org/works",
                         params={"query": query, "rows": 5, "mailto": "research@example.com"},
                         timeout=10)
        if r.ok:
            for item in r.json().get("message", {}).get("items", []):
                title   = (item.get("title") or ["No title"])[0]
                doi     = item.get("DOI", "")
                authors = [f"{a.get('given','')} {a.get('family','')}".strip()
                           for a in item.get("author", [])[:3]]
                results_text.append(f"[Crossref] {title} — {', '.join(authors)} | doi:{doi}")
                vdb.add_text(f"Title: {title}\nAuthors: {', '.join(authors)}\nDOI: {doi}",
                             {"source": "crossref", "title": title})
                indexed += 1
            sources_used.append("Crossref")
    except Exception as e:
        results_text.append(f"[Crossref error] {e}")

    # Semantic Scholar ──────────────────────────────────────────────────────
    try:
        ss_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        headers = {"x-api-key": ss_key} if ss_key else {}
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, 
                    "limit": 5,
                    "fields": "title,year,citationCount,authors"},
            headers=headers, timeout=10
        )
      
        if r.ok:
            for paper in r.json().get("data", []):
                title   = paper.get("title", "No title")
                year    = paper.get("year", "")
                cites   = paper.get("citationCount", 0)
                authors = [a.get("name","") for a in (paper.get("authors") or [])[:3]]
                results_text.append(f"[Semantic Scholar] {title} ({year}) — {', '.join(authors)} — {cites} citations")
                vdb.add_text(
                    f"Title: {title}\nAuthors: {', '.join(authors)}\nYear: {year}\nCitations: {cites}",
                    {"source": "semantic_scholar", "title": title}
                )
                indexed += 1
            sources_used.append("Semantic Scholar")
    except Exception as e:
        results_text.append(f"[Semantic Scholar error] {e}")

    # arXiv ─────────────────────────────────────────────────────────────────
    try:
        encoded = urllib.parse.quote(query)
        url = (f"http://export.arxiv.org/api/query"
               f"?search_query=all:{encoded}&start=0&max_results=5&sortBy=relevance")
      
        with urllib.request.urlopen(url, timeout=15) as resp:
            xml = resp.read().decode("utf-8")
          
        ns      = {"atom": "http://www.w3.org/2005/Atom"}
        root    = ET.fromstring(xml)
        entries = root.findall("atom:entry", ns)
      
        for entry in entries:
            title   = (entry.findtext("atom:title",   "", ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", "", ns) or "").strip()[:400]
            eid     = (entry.findtext("atom:id",      "", ns) or "").strip()
            authors = [a.findtext("atom:name","",ns) for a in entry.findall("atom:author",ns)]
            results_text.append(f"[arXiv] {title} — {', '.join(authors[:3])}\n{eid}")
          
            # Index into vector DB
            vdb.add_text(
                f"Title: {title}\nAuthors: {', '.join(authors)}\nAbstract: {summary}",
                {"source": "arXiv", "title": title, "url": eid, "indexed_at": _stamp()},
            )
            indexed += 1
        sources_used.append("arXiv")
      
    except Exception as e:
        results_text.append(f"[arXiv error] {e}")

    combined = "\n\n---\n".join(results_text) if results_text else "(no results)"
    system = SystemMessage(content=(
        "You are a Web Research Agent. Summarise the scholarly search results below into "
        "structured research notes relevant to the query. Cite the source for each finding "
        "(OpenAlex, Crossref, Semantic Scholar, or arXiv)."
    ))
    resp = model.invoke([system, HumanMessage(
        content=f"Query: {query}\n\nResults:\n{combined}"
    )])

    return {
        "web_findings": resp.content,
        "messages":     [AIMessage(content=f"[Web] {resp.content[:120]}…")],
        "activity_log": [{
            "agent": "web", "icon": "🌐", "title": "Web / API agent",
            "detail": (f"Sources queried: {', '.join(sources_used) or 'none'} — "
                       f"{indexed} papers indexed into VectorDB"),
            "ts": _stamp(),
        }],
        "current_agent": "web",
    }


def orchestrator_agent(state: AgentState, model: BaseChatModel) -> dict:
    block = "\n\n".join([
        f"=== Vector DB ===\n{state.get('vector_findings','')}",
        f"=== SQL / DB ===\n{state.get('sql_findings','')}",
        f"=== Web / APIs ===\n{state.get('web_findings','')}",
    ])
    system = SystemMessage(content=(
        "You are an Orchestrator Agent. Merge findings from three specialised agents:\n"
        "1. Deduplicate overlapping information\n"
        "2. Resolve contradictions, preferring higher-confidence structured sources\n"
        "3. Label each claim: [VectorDB] / [SQL] / [Web]\n"
        "4. Produce one coherent merged context for downstream agents."
    ))
    resp = model.invoke([system, HumanMessage(content=f"Query: {state['query']}\n\n{block}")])
    active = [
        src for src, key in [
            ("Vector DB", "vector_findings"),
            ("SQL DB",    "sql_findings"),
            ("Web",       "web_findings"),
        ]
        if "not activated" not in state.get(key, "not activated")
    ]
    return {
        "merged_context": resp.content,
        "messages":       [AIMessage(content=f"[Orchestrator] {resp.content[:120]}…")],
        "activity_log":   [{
            "agent": "orchestrator", "icon": "🤝", "title": "Orchestrator merged findings",
            "detail": f"Sources merged: {', '.join(active) or 'none'}", "ts": _stamp(),
        }],
        "current_agent": "orchestrator",
    }

# ── 6. Knowledge Mapper ───────────────────────────────────────────────────────
def knowledge_mapper_agent(state: AgentState, model: BaseChatModel) -> dict:
    system = SystemMessage(content=(
        "You are a Knowledge Mapping Agent. Extract a knowledge graph from the merged context.\n"
        "Return ONLY valid JSON:\n"
        '{"nodes": [{"id":"str","label":"str","type":"concept|entity|fact|process",'
        '"source":"vector_db|sql_db|web|merged"}],'
        '"edges": [{"source":"str","target":"str","relation":"str","weight":0.1}]}\n'
        "Include 12–20 nodes. No text outside JSON."
    ))
    resp = model.invoke([system, HumanMessage(
        content=f"Query: {state['query']}\n\nMerged context:\n{state['merged_context']}"
    )])
    raw = resp.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("```").strip()
    try:
        km = json.loads(raw)
    except Exception:
        km = {"nodes": [], "edges": [], "error": "parse_failed"}

    return {
        "knowledge_map": km,
        "messages":      [AIMessage(content=f"[KnowledgeMapper] {len(km.get('nodes',[]))} nodes")],
        "activity_log":  [{
            "agent": "knowledge_mapper", "icon": "🗺️", "title": "Knowledge Mapper",
            "detail": f"{len(km.get('nodes',[]))} nodes, {len(km.get('edges',[]))} edges extracted",
            "ts": _stamp(),
        }],
        "current_agent": "knowledge_mapper",
    }

# ── 7. Critic ─────────────────────────────────────────────────────────────────
def critic_agent(state: AgentState, model: BaseChatModel) -> dict:
    system = SystemMessage(content=(
        "You are a Critic Agent. If the knowledge map has fewer than 8 nodes OR "
        "key source diversity is missing, respond: "
        '{\"needs_more\": true, \"feedback\": \"specific gaps\"}. '
        'Otherwise: {\"needs_more\": false, \"feedback\": \"\"}. Only JSON.'
    ))
    resp = model.invoke([system, HumanMessage(content=(
        f"Nodes: {[n['label'] for n in state['knowledge_map'].get('nodes',[])]}\n"
        f"Sources: {list({n.get('source','?') for n in state['knowledge_map'].get('nodes',[])})}\n"
        f"Edges: {len(state['knowledge_map'].get('edges',[]))}"
    ))])
    raw = resp.content.strip().lstrip("```json").rstrip("```").strip()
    try:
        result = json.loads(raw)
    except Exception:
        result = {"needs_more": False, "feedback": ""}

    needs = result.get("needs_more", False)
    return {
        "critique":    result.get("feedback", ""),
        "_needs_more": needs,
        "loop_count":  state.get("loop_count", 0) + 1,
        "messages":    [AIMessage(content=f"[Critic] needs_more={needs}")],
        "activity_log": [{
            "agent": "critic", "icon": "🧐",
            "title": f"Critic — {'needs enrichment' if needs else 'approved'}",
            "detail": result.get("feedback", "Graph looks sufficient."), "ts": _stamp(),
        }],
        "current_agent": "critic",
    }

# ── 8. Summarizer ─────────────────────────────────────────────────────────────
def summarizer_agent(state: AgentState, model: BaseChatModel) -> dict:
    system = SystemMessage(content=(
        "You are a Summarizer Agent. Write a clear, well-structured answer grounded in the "
        "merged context. Cite which source (Vector DB / SQL DB / Web) each key claim comes from."
    ))
    resp = model.invoke([system, HumanMessage(content=(
        f"Query: {state['query']}\n\n"
        f"Merged context:\n{state['merged_context']}\n\n"
        f"Key concepts: {[n['label'] for n in state['knowledge_map'].get('nodes',[])]}"
    ))])
    return {
        "summary":      resp.content,
        "messages":     [AIMessage(content=f"[Summarizer] {resp.content[:120]}…")],
        "activity_log": [{
            "agent": "summarizer", "icon": "✍️", "title": "Summarizer — final answer",
            "detail": f"{len(resp.content)} characters", "ts": _stamp(),
        }],
        "current_agent": "summarizer",
    }


# ── Routing ───────────────────────────────────────────────────────────────────
def _route_critic(state: AgentState) -> str:
    if state.get("_needs_more") and state.get("loop_count", 0) < 2:
        return "orchestrator"
    return "summarizer"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(cfg: ProviderConfig, vdb: VectorDBModule):
    lm_r = _llm(cfg, 0.0)
    lm_s = _llm(cfg, 0.3)
    lm_o = _llm(cfg, 0.2)
    lm_m = _llm(cfg, 0.1)
    lm_c = _llm(cfg, 0.0)
    lm_z = _llm(cfg, 0.5)

    g = StateGraph(AgentState)
    g.add_node("router",           lambda s: router_agent(s, lm_r))
    g.add_node("vector_db",        lambda s: vector_db_agent(s, lm_s, vdb))
    g.add_node("sql_db",           lambda s: sql_db_agent(s, lm_s))
    g.add_node("web",              lambda s: web_agent(s, lm_s, vdb))
    g.add_node("orchestrator",     lambda s: orchestrator_agent(s, lm_o))
    g.add_node("knowledge_mapper", lambda s: knowledge_mapper_agent(s, lm_m))
    g.add_node("critic",           lambda s: critic_agent(s, lm_c))
    g.add_node("summarizer",       lambda s: summarizer_agent(s, lm_z))

    g.set_entry_point("router")
    g.add_edge("router",           "vector_db")
    g.add_edge("vector_db",        "sql_db")
    g.add_edge("sql_db",           "web")
    g.add_edge("web",              "orchestrator")
    g.add_edge("orchestrator",     "knowledge_mapper")
    g.add_edge("knowledge_mapper", "critic")
    g.add_conditional_edges(
        "critic", _route_critic,
        {"orchestrator": "orchestrator", "summarizer": "summarizer"},
    )
    g.add_edge("summarizer", END)
    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

_SRC_COLOR = {
    "vector_db": "#4A90D9",
    "sql_db":    "#E67E22",
    "web":       "#2ECC71",
    "merged":    "#9B59B6",
}
_TYPE_SHAPE = {
    "concept": "dot",
    "entity":  "diamond",
    "fact":    "square",
    "process": "triangleDown",
}

def render_knowledge_map(km: dict, height: int = 500) -> str:
    net = Network(height=f"{height}px", width="100%",
                  bgcolor="#0d1117", font_color="white", directed=True)
    net.set_options(json.dumps({
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}},
            "color":  {"color": "#444", "highlight": "#fff"},
            "font":   {"size": 10, "color": "#aaa"},
            "smooth": {"type": "curvedCW", "roundness": 0.2},
        },
        "nodes": {"font": {"size": 12, "bold": True}, "borderWidth": 2, "shadow": True},
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.01,
                "springLength": 220,
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 200},
        },
        "interaction": {"hover": True, "tooltipDelay": 100},
    }))
    for node in km.get("nodes", []):
        color = _SRC_COLOR.get(node.get("source", "merged"), _SRC_COLOR["merged"])
        shape = _TYPE_SHAPE.get(node.get("type", "concept"), "dot")
        net.add_node(node["id"], label=node["label"], color=color, shape=shape,
                     title=f"<b>{node['label']}</b><br>Type: {node.get('type','?')}<br>Source: {node.get('source','?')}",
                     size=22)
    for edge in km.get("edges", []):
        try:
            net.add_edge(edge["source"], edge["target"],
                         title=edge.get("relation", ""), label=edge.get("relation", ""),
                         width=max(1, edge.get("weight", 0.5) * 4))
        except Exception:
            pass
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        html = Path(f.name).read_text()
        Path(f.name).unlink(missing_ok=True)
    return html


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Collaborative RAG",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Outfit:wght@400;600;800&display=swap');
html,[class*="css"]{font-family:'Outfit',sans-serif}
code,pre{font-family:'JetBrains Mono',monospace!important}
[data-testid="stSidebar"]{background:#0d1117!important}
[data-testid="stSidebar"] *{color:#c9d1d9!important}
h1,h2,h3{font-weight:800!important;letter-spacing:-0.02em}
[data-testid="stMetric"]{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 16px}
.agent-card{border-left:3px solid;padding:10px 14px;margin-bottom:10px;border-radius:0 6px 6px 0;background:#161b22}
.agent-card.router{border-color:#9B59B6}
.agent-card.vector_db{border-color:#4A90D9}
.agent-card.sql_db{border-color:#E67E22}
.agent-card.web{border-color:#2ECC71}
.agent-card.orchestrator{border-color:#E74C3C}
.agent-card.knowledge_mapper{border-color:#1ABC9C}
.agent-card.critic{border-color:#F39C12}
.agent-card.summarizer{border-color:#95A5A6}
.agent-title{font-weight:600;font-size:0.88rem;margin-bottom:3px}
.agent-detail{font-size:0.78rem;color:#8b949e}
.src-badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.68rem;font-family:'JetBrains Mono',monospace;letter-spacing:.04em;margin:0 3px}
.bv{background:#1a3a5c;color:#4A90D9}
.bs{background:#4a2010;color:#E67E22}
.bw{background:#0f3d20;color:#2ECC71}
.bm{background:#2d1a4a;color:#9B59B6}
</style>
""", unsafe_allow_html=True)

# ── Init ──────────────────────────────────────────────────────────────────────
init_sql_db()

if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.utcnow().strftime("sess_%Y%m%d_%H%M%S")
if "turns" not in st.session_state:
    st.session_state.turns = []
if "vdb" not in st.session_state:
    st.session_state.vdb = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤝 Collaborative RAG")
    st.divider()

    # ── Provider selection ────────────────────────────────────────────────────
    provider = st.selectbox(
        "LLM Provider",
        [PROVIDER_OPENAI, PROVIDER_GEMINI, PROVIDER_CLAUDE, PROVIDER_LOCAL],
    )

    _env_keys = {
        PROVIDER_OPENAI: os.environ.get("OPENAI_API_KEY", ""),
        PROVIDER_GEMINI: os.environ.get("GOOGLE_API_KEY", ""),
        PROVIDER_CLAUDE: os.environ.get("ANTHROPIC_API_KEY", ""),
        PROVIDER_LOCAL:  "",
    }
    _key_labels = {
        PROVIDER_OPENAI: "OpenAI API Key",
        PROVIDER_GEMINI: "Google API Key",
        PROVIDER_CLAUDE: "Anthropic API Key",
        PROVIDER_LOCAL:  "API Key (leave blank if none)",
    }
    _key_placeholders = {
        PROVIDER_OPENAI: "sk-…",
        PROVIDER_GEMINI: "AIza…",
        PROVIDER_CLAUDE: "sk-ant-…",
        PROVIDER_LOCAL:  "optional",
    }

    api_key = st.text_input(
        _key_labels.get(provider, "API Key"),
        value=_env_keys.get(provider, ""),
        type="password",
        placeholder=_key_placeholders.get(provider, ""),
    )

    default_model = _DEFAULT_MODELS.get(provider, "")
    llm_model = st.text_input("Model", value=default_model, placeholder=default_model or "model name")

    base_url = None
    if provider == PROVIDER_LOCAL:
        base_url = st.text_input("Base URL", value=_DEFAULT_BASE_URL,
                                 placeholder=_DEFAULT_BASE_URL)

    if provider != PROVIDER_LOCAL and not api_key:
        st.warning(f"Enter your {_key_labels.get(provider, 'API Key')} to continue.")
        st.stop()

    cfg = ProviderConfig(
        provider=provider,
        api_key=api_key,
        model=llm_model or default_model,
        base_url=base_url,
    )

    # Rebuild VDB when provider/key/model changes (embeddings depend on these).
    # Each embedding backend gets its own subdirectory under VECTOR_DIR so that
    # indices built with different dimensionalities (e.g. OpenAI 1536-d vs
    # sentence-transformers 384-d) never collide and cause runtime errors.
    _vdb_cache_key = (provider, api_key, llm_model)
    if st.session_state.vdb is None or st.session_state.get("_vdb_cfg") != _vdb_cache_key:
        with st.spinner("Initialising embeddings…"):
            vdb_dir = VECTOR_DIR / _embedding_key(cfg)
            st.session_state.vdb      = VectorDBModule(_embeddings(cfg), vector_dir=vdb_dir)
            st.session_state._vdb_cfg = _vdb_cache_key

        # FIX 1: auto re-index saved docs on startup
        reindexed = st.session_state.vdb.reindex_saved_docs()
        if reindexed:
            st.info(f"Re-indexed {reindexed} saved document(s) from disk.")

    vdb: VectorDBModule = st.session_state.vdb
    st.success("✅ Ready")

    st.divider()
    st.markdown("### Index Documents")
    uploaded = st.file_uploader("Upload .txt / .md", type=["txt", "md"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            n = vdb.add_file(f)
            st.success(f"{f.name} → {n} chunks")

    with st.expander("Index arXiv papers"):
        aq = st.text_input("arXiv query", placeholder="transformer attention")
        al = st.number_input("Max papers", 1, 30, 5)
        if st.button("Index", key="arxiv_btn"):
            if aq:
                with st.spinner("Fetching from arXiv…"):
                    n = index_arxiv_documents(aq, vdb, al)
                st.success(f"Indexed {n} papers")
            else:
                st.error("Enter a query.")

    st.metric("Vector chunks", vdb.count())
    srcs = vdb.sources()
    if srcs:
        with st.expander(f"{len(srcs)} document(s)"):
            for s in srcs:
                st.markdown(f"• `{s}`")

    st.divider()
    st.markdown("### Add SQL Topic")
    with st.form("add_topic"):
        t_title = st.text_input("Title")
        t_cat   = st.text_input("Category")
        t_sum   = st.text_area("Summary", height=60)
        t_kw    = st.text_input("Keywords (comma-separated)")
        if st.form_submit_button("Add"):
            st.success(sql_insert_topic(t_title, t_cat, t_sum, t_kw))

    st.divider()
    st.markdown("### Legend")
    for label, css in [("Vector DB","bv"),("SQL / DB","bs"),("Web","bw"),("Merged","bm")]:
        st.markdown(f'<span class="src-badge {css}">{label}</span>', unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_research, tab_sql, tab_maps, tab_sessions, tab_cache = st.tabs([
    "🚀 Research", "🗄️ SQL DB", "🗺️ Saved Maps", "💬 Sessions", "🕐 Cache",
])

# ════════════ TAB 1 — RESEARCH ════════════════════════════════════════════════
with tab_research:
    st.header("Collaborative Research Query")

    col_q, col_opts = st.columns([3, 1])
    with col_q:
        query = st.text_area("Query", height=90,
                             placeholder="e.g. How does RAG relate to transformer architecture?")
    with col_opts:
        use_cache     = st.checkbox("Use 20-day cache", value=True)
        auto_save_map = st.checkbox("Auto-save map",    value=True)

    run = st.button("🚀 Run Pipeline", type="primary",
                    disabled=not ((api_key or provider == PROVIDER_LOCAL) and query))

    if run:
        cached = cache_load(query) if use_cache else None
        if cached:
            st.info("⚡ Loaded from cache")
            full_state = cached
        else:
            app      = build_graph(cfg, vdb)
            progress = st.progress(0, "Starting…")
            pct_map  = {
                "router": 10, "vector_db": 25, "sql_db": 40, "web": 55,
                "orchestrator": 68, "knowledge_mapper": 80, "critic": 90, "summarizer": 97,
            }

            # Accumulate state across ALL agents properly
            # Each agent returns a partial update - we merge them so nothing is lost
            full_state = {
                "messages":[], "query": query,
                "active_agents":[], "router_reasoning":"",
                "vector_findings":"", "sql_findings":"", "web_findings":"",
                "activity_log":[], "merged_context":"",
                "knowledge_map":{}, "critique":"", "loop_count":0,
                "summary":"", "current_agent":"",
            }
            for event in app.stream(full_state.copy()):
                for node, state_update in event.items():
                    progress.progress(pct_map.get(node, 50),
                                      f"{node.replace('_',' ').title()} running...")
                    for key, val in state_update.items():
                        if key in ('messages', 'activity_log') and isinstance(val, list):
                            full_state[key] = full_state.get(key, []) + val
                        else:
                            full_state[key] = val

            progress.progress(100, "✅ Done!")
            if full_state and use_cache:
                cache_save(query, full_state)

        if full_state is None:
            st.error("Pipeline returned no state. Please try again.")
            st.stop()

        if auto_save_map and full_state.get("knowledge_map", {}).get("nodes"):
            fname = map_save(query, full_state["knowledge_map"])
            st.success(f"🗺️ Map saved → `{fname}`")

        st.session_state.turns.append({
            "query":   query,
            "summary": full_state.get("summary", ""),
            "agents":  full_state.get("active_agents", []),
            "nodes":   len(full_state.get("knowledge_map", {}).get("nodes", [])),
            "ts":      _stamp(),
            "cached":  cached is not None,
        })
        session_save(st.session_state.session_id, st.session_state.turns)

        r_act, r_ans, r_map, r_ctx, r_find, r_log = st.tabs([
            "🤝 Agent Activity", "💡 Final Answer", "🗺️ Knowledge Map",
            "🔀 Merged Context", "🔍 Per-Agent Findings", "💬 Message Log",
        ])

        with r_act:
            st.subheader("What each agent did")
            badge = {
                "vector_db":    '<span class="src-badge bv">Vector DB</span>',
                "sql_db":       '<span class="src-badge bs">SQL DB</span>',
                "web":          '<span class="src-badge bw">Web</span>',
                "orchestrator": '<span class="src-badge bm">Orchestrator</span>',
            }
            for entry in full_state.get("activity_log", []):
                cls    = entry.get("agent", "")
                icon   = entry.get("icon", "•")
                title  = entry.get("title", "")
                detail = entry.get("detail", "")
                bdg    = badge.get(cls, "")
                st.markdown(
                    f'<div class="agent-card {cls}">'
                    f'<div class="agent-title">{icon} {title} {bdg}</div>'
                    f'<div class="agent-detail">{detail}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if cls == "vector_db" and entry.get("chunks"):
                    with st.expander(f"📄 {len(entry['chunks'])} retrieved chunks"):
                        for ch in entry["chunks"]:
                            st.markdown(f"**`{ch['source']}`**")
                            st.caption(ch["text"])
                if cls == "sql_db" and entry.get("rows"):
                    with st.expander(f"🗄️ {len(entry['rows'])} matched rows"):
                        for row in entry["rows"]:
                            st.code(row, language="text")

        with r_ans:
            st.markdown(full_state.get("summary", ""))

        with r_map:
            km = full_state.get("knowledge_map", {})
            if km.get("nodes"):
                src_counts: dict[str, int] = {}
                for n in km["nodes"]:
                    s = n.get("source", "merged")
                    src_counts[s] = src_counts.get(s, 0) + 1
                cols = st.columns(len(src_counts))
                css  = {"vector_db":"bv","sql_db":"bs","web":"bw","merged":"bm"}
                for i, (s, c) in enumerate(src_counts.items()):
                    cols[i].markdown(f'<span class="src-badge {css.get(s,"bm")}">{s}: {c}</span>',
                                     unsafe_allow_html=True)
                st.components.v1.html(render_knowledge_map(km), height=520)
                with st.expander("📊 Raw graph data"):
                    c1, c2 = st.columns(2)
                    c1.markdown("**Nodes**"); c1.dataframe(km["nodes"])
                    c2.markdown("**Edges**"); c2.dataframe(km["edges"])
            else:
                st.warning("No map generated.")

        with r_ctx:
            st.markdown(full_state.get("merged_context", ""))

        with r_find:
            for label, key, bcls in [
                ("🗂️ Vector DB", "vector_findings", "bv"),
                ("🗄️ SQL / DB",  "sql_findings",    "bs"),
                ("🌐 Web",       "web_findings",     "bw"),
            ]:
                with st.expander(f"{label} findings"):
                    st.markdown(full_state.get(key, ""))

        with r_log:
            av = {
                "[Router]":"🔀","[VectorDB]":"🗂️","[SQLDB]":"🗄️","[Web]":"🌐",
                "[Orchestrator]":"🤝","[KnowledgeMapper]":"🗺️",
                "[Critic]":"🧐","[Summarizer]":"✍️",
            }
            for msg in full_state.get("messages", []):
                icon = next((v for k, v in av.items() if k in msg.content), "🤖")
                st.chat_message("assistant", avatar=icon).write(msg.content)


# ════════════ TAB 2 — SQL DB ══════════════════════════════════════════════════
with tab_sql:
    st.header("SQL / DB Browser")
    topics = sql_list_topics()
    st.metric("Topics in DB", len(topics))
    st.dataframe(topics, use_container_width=True)
    st.divider()
    st.subheader("Test keyword search")
    test_q = st.text_input("Keyword")
    if test_q:
        st.code(sql_search(test_q), language="text")


# ════════════ TAB 3 — SAVED MAPS ══════════════════════════════════════════════
with tab_maps:
    st.header("Saved Knowledge Maps")
    all_maps = map_list()
    if not all_maps:
        st.info("No maps saved yet.")
    else:
        filt = st.text_input("Filter")
        shown = [m for m in all_maps if not filt or filt.lower() in m["query"].lower()]
        for m in shown:
            with st.expander(
                f"🗺️ {m['query'][:65]}…  ·  {m['nodes']}n / {m['edges']}e  ·  "
                f"{m['saved_at'][:16].replace('T',' ')}"
            ):
                if st.button("Visualise", key=f"vis_{m['file']}"):
                    raw = map_load(m["file"])
                    if raw:
                        st.components.v1.html(render_knowledge_map(raw["map"]), height=470)


# ════════════ TAB 4 — SESSIONS ════════════════════════════════════════════════
with tab_sessions:
    st.header("Conversation History")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f"**Session:** `{st.session_state.session_id}`")
        if st.button("💾 Save"):
            session_save(st.session_state.session_id, st.session_state.turns)
            st.success("Saved!")
        if st.button("🆕 New session"):
            session_save(st.session_state.session_id, st.session_state.turns)
            st.session_state.session_id = datetime.utcnow().strftime("sess_%Y%m%d_%H%M%S")
            st.session_state.turns = []
            st.rerun()
        st.divider()
        sel = None
        for s in session_list():
            if st.button(
                f"📅 {s['updated'][:16].replace('T',' ')}  ·  {s['n']} turns\n{s['preview']}",
                key=f"sess_{s['sid']}",
            ):
                sel = s["sid"]
    with c2:
        turns = session_load(sel) if sel else st.session_state.turns
        _agent_css = {"vector_db": "bv", "sql_db": "bs", "web": "bw"}
        for i, t in enumerate(reversed(turns), 1):
            badges = " ".join(
                f'<span class="src-badge {_agent_css.get(a, "bm")}">{a}</span>'
                for a in t.get("agents", [])
            )
            st.markdown(f"**Turn {len(turns)-i+1}** · {t['ts'][:16].replace('T',' ')} {badges}",
                        unsafe_allow_html=True)
            st.markdown(f"> 🔍 **{t['query']}**")
            with st.expander("Summary", expanded=(i == 1)):
                st.markdown(t.get("summary", ""))
            st.markdown(f"🗺️ {t.get('nodes', 0)} nodes")
            st.divider()


# ════════════ TAB 5 — CACHE ═══════════════════════════════════════════════════
with tab_cache:
    st.header("20-Day Query Cache")
    entries = cache_list()
    st.metric("Cached entries", len(entries))
    filt_c = st.text_input("Filter", key="cache_filt")
    shown_c = [e for e in entries if not filt_c or filt_c.lower() in e["query"].lower()]
    for e in shown_c:
        age   = datetime.utcnow() - datetime.fromisoformat(e["ts"])
        age_s = f"{age.days}d {age.seconds//3600}h ago" if age.days else f"{age.seconds//3600}h {(age.seconds%3600)//60}m ago"
        with st.expander(f"🗃️ {e['query'][:75]}…  ·  {age_s}"):
            pl = cache_load(e["query"])
            if pl:
                c1, c2, c3 = st.columns(3)
                c1.metric("Summary",  f"{len(pl.get('summary',''))} chars")
                c2.metric("Nodes",    len(pl.get("knowledge_map",{}).get("nodes",[])))
                c3.metric("Agents",   ", ".join(pl.get("active_agents",[])))
                if st.button("▶️ Reload", key=f"rc_{e['file']}"):
                    st.markdown(pl.get("summary",""))
                    km = pl.get("knowledge_map",{})
                    if km.get("nodes"):
                        st.components.v1.html(render_knowledge_map(km, 400), height=420)
    st.divider()
    if st.button("🧹 Clear ALL cache"):
        for p in CACHE_DIR.glob("*.json"):
            p.unlink()
        st.success("Cache cleared.")
