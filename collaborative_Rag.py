"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     Collaborative Multi-Agent RAG  —  Vector DB · SQL · Web · Orchestrator  ║
║                                                                              ║
║  Install:                                                                    ║
║    pip install streamlit langgraph langchain-openai langchain-core           ║
║               langchain-community faiss-cpu langchain-text-splitters         ║
║               networkx pyvis tiktoken sqlalchemy                             ║
║                                                                              ║
║  Run:   streamlit run collaborative_rag.py                                   ║
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
    [Knowledge Mapper]  ← enriches graph; loops back if sparse
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, List, Optional, TypedDict

import streamlit as st
import networkx as nx
from pyvis.network import Network

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

from research_apis import ResearchAPIs
from secrets_manager import get_api_key, has_api_key

# Import basic search functions
import sys
import os
sys.path.append(os.path.dirname(__file__))
from basic_search_example import search_scientific_apis, query_arxiv

# Import additional libraries
import arxiv
from nasapy import Nasa

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

for d in [VECTOR_DIR, DOCS_DIR, MAPS_DIR, CACHE_DIR, SESSIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CACHE_TTL_DAYS = 20

# ══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    # Core
    messages:           Annotated[List, operator.add]
    query:              str
    # Router decision
    active_agents:      List[str]          # ["vector_db", "sql_db", "web"]
    router_reasoning:   str
    # Per-agent findings
    vector_findings:    str
    sql_findings:       str
    web_findings:       str
    # Activity log (shown in UI)
    activity_log:       Annotated[List, operator.add]
    # Orchestrator output
    merged_context:     str
    # Knowledge graph
    knowledge_map:      dict
    critique:           str
    loop_count:         int
    # Final
    summary:            str
    current_agent:      str


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def _stamp() -> str:
    return datetime.utcnow().isoformat()

def _hash(q: str) -> str:
    return hashlib.sha256(q.strip().lower().encode()).hexdigest()[:16]

# ── Cache ─────────────────────────────────────────────────────────────────────

def cache_save(query: str, payload: dict) -> None:
    p = CACHE_DIR / f"{_hash(query)}.json"
    p.write_text(json.dumps({"query": query, "ts": _stamp(), "payload": payload}, indent=2))

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
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{_hash(query)}.json"
    (MAPS_DIR / name).write_text(json.dumps({"query": query, "saved_at": _stamp(), "map": km}, indent=2))
    return name

def map_list() -> list[dict]:
    out = []
    for p in MAPS_DIR.glob("*.json"):
        try:
            raw = json.loads(p.read_text())
            out.append({
                "query": raw.get("query",""),
                "saved_at": raw.get("saved_at",""),
                "nodes": len(raw.get("map",{}).get("nodes",[])),
                "edges": len(raw.get("map",{}).get("edges",[])),
                "file": p.name,
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
        json.dumps({"sid": sid, "updated": _stamp(), "turns": turns}, indent=2))

def session_load(sid: str) -> list:
    p = SESSIONS_DIR / f"{sid}.json"
    return json.loads(p.read_text()).get("turns", []) if p.exists() else []

def session_list() -> list[dict]:
    out = []
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            raw = json.loads(p.read_text())
            turns = raw.get("turns", [])
            out.append({
                "sid": raw.get("sid", p.stem),
                "updated": raw.get("updated",""),
                "n": len(turns),
                "preview": turns[0]["query"][:55]+"…" if turns else "(empty)",
                "file": p.name,
            })
        except Exception:
            pass
    return sorted(out, key=lambda x: x["updated"], reverse=True)

# ══════════════════════════════════════════════════════════════════════════════
# SQL DB MODULE  —  sample schema + queries
# ══════════════════════════════════════════════════════════════════════════════

def init_sql_db() -> None:
    """Create a sample SQLite database with some knowledge records."""
    con = sqlite3.connect(SQL_DB_PATH)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS topics (
        id INTEGER PRIMARY KEY,
        title TEXT,
        category TEXT,
        summary TEXT,
        keywords TEXT,
        created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS relationships (
        id INTEGER PRIMARY KEY,
        from_topic TEXT,
        to_topic TEXT,
        relation_type TEXT
    );
    CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY,
        subject TEXT,
        predicate TEXT,
        object TEXT,
        confidence REAL,
        source TEXT
    );
    """)
    # Insert sample data if table is empty
    count = cur.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    if count == 0:
        sample_topics = [
            ("Transformer Architecture", "ML", "Self-attention-based sequence model with encoder-decoder structure.", "transformer,attention,neural network,NLP", _stamp()),
            ("RLHF", "ML", "Reinforcement Learning from Human Feedback — aligns LLMs via preference data.", "RLHF,alignment,reward model,fine-tuning", _stamp()),
            ("RAG", "ML", "Retrieval-Augmented Generation combines retrieval with generation for grounded outputs.", "RAG,retrieval,generation,vector search", _stamp()),
            ("Gradient Descent", "Math", "Iterative optimisation algorithm minimising a loss function by parameter updates.", "gradient,optimisation,loss,backprop", _stamp()),
            ("Attention Mechanism", "ML", "Soft-alignment mechanism computing weighted sums over key-value pairs given a query.", "attention,query,key,value,softmax", _stamp()),
            ("Knowledge Graph", "Data", "Graph structure representing entities as nodes and relationships as typed edges.", "graph,entity,relation,knowledge", _stamp()),
        ]
        cur.executemany(
            "INSERT INTO topics (title,category,summary,keywords,created_at) VALUES (?,?,?,?,?)",
            sample_topics,
        )
        sample_rels = [
            ("RAG", "Transformer Architecture", "uses"),
            ("RLHF", "Transformer Architecture", "fine-tunes"),
            ("Attention Mechanism", "Transformer Architecture", "core component of"),
            ("Gradient Descent", "RLHF", "optimises reward model in"),
        ]
        cur.executemany(
            "INSERT INTO relationships (from_topic,to_topic,relation_type) VALUES (?,?,?)",
            sample_rels,
        )
        sample_facts = [
            ("Transformer", "introduced in", "Attention Is All You Need (2017)", 0.99, "Vaswani et al."),
            ("RLHF", "used by", "InstructGPT, ChatGPT, Claude", 0.99, "OpenAI/Anthropic papers"),
            ("RAG", "retriever", "dense passage retrieval or BM25", 0.95, "Lewis et al. 2020"),
            ("Attention", "complexity", "O(n²) in sequence length", 0.99, "Transformer paper"),
        ]
        cur.executemany(
            "INSERT INTO facts (subject,predicate,object,confidence,source) VALUES (?,?,?,?,?)",
            sample_facts,
        )
    con.commit()
    con.close()


def sql_search(query: str, k: int = 8) -> str:
    """Keyword search over topics, relationships, and facts tables."""
    con = sqlite3.connect(SQL_DB_PATH)
    cur = con.cursor()
    # Build a very simple keyword matcher
    words = [w.strip("?.!,") for w in query.lower().split() if len(w) > 3]
    results = []

    for word in words[:5]:
        rows = cur.execute(
            "SELECT title, category, summary, keywords FROM topics WHERE "
            "LOWER(title) LIKE ? OR LOWER(keywords) LIKE ? OR LOWER(summary) LIKE ?",
            (f"%{word}%", f"%{word}%", f"%{word}%"),
        ).fetchall()
        for r in rows:
            results.append(f"[TOPIC] {r[0]} ({r[1]}): {r[2]}")

        rows = cur.execute(
            "SELECT from_topic, relation_type, to_topic FROM relationships WHERE "
            "LOWER(from_topic) LIKE ? OR LOWER(to_topic) LIKE ?",
            (f"%{word}%", f"%{word}%"),
        ).fetchall()
        for r in rows:
            results.append(f"[REL] {r[0]} —[{r[1]}]→ {r[2]}")

        rows = cur.execute(
            "SELECT subject, predicate, object, source FROM facts WHERE "
            "LOWER(subject) LIKE ? OR LOWER(object) LIKE ?",
            (f"%{word}%", f"%{word}%"),
        ).fetchall()
        for r in rows:
            results.append(f"[FACT] {r[0]} {r[1]} '{r[2]}' (source: {r[3]})")

    con.close()
    seen, unique = set(), []
    for r in results:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return "\n".join(unique[:k]) if unique else "(no SQL results)"


def sql_insert_topic(title: str, category: str, summary: str, keywords: str) -> str:
    """Allow the UI to add new records to the SQL DB."""
    con = sqlite3.connect(SQL_DB_PATH)
    con.execute(
        "INSERT INTO topics (title,category,summary,keywords,created_at) VALUES (?,?,?,?,?)",
        (title, category, summary, keywords, _stamp()),
    )
    con.commit()
    con.close()
    return f"Inserted topic: {title}"


def sql_list_topics() -> list[dict]:
    con = sqlite3.connect(SQL_DB_PATH)
    rows = con.execute("SELECT id,title,category,keywords FROM topics ORDER BY id DESC").fetchall()
    con.close()
    return [{"id": r[0], "title": r[1], "category": r[2], "keywords": r[3]} for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR DB MODULE
# ══════════════════════════════════════════════════════════════════════════════

class VectorDBModule:
    def __init__(self, api_key: str):
        self.api_key    = api_key
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.splitter   = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        self._store: Optional[FAISS] = None
        self._load()

    def _load(self) -> None:
        if (VECTOR_DIR / "index.faiss").exists():
            try:
                self._store = FAISS.load_local(
                    str(VECTOR_DIR), self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                self._store = None

    def _save(self) -> None:
        if self._store:
            self._store.save_local(str(VECTOR_DIR))

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


# ══════════════════════════════════════════════════════════════════════════════
# LLM FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def llm(api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(api_key=api_key, model=model, temperature=temperature) #CHANGE IF NEEDED


# ══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Router ─────────────────────────────────────────────────────────────────

def router_agent(state: AgentState, model: ChatOpenAI) -> dict:
    system = SystemMessage(content=(
        "You are a Router Agent. Given a user query, decide which search agents to activate. "
        "Available agents: 'vector_db' (semantic document search), 'sql_db' (structured facts/topics), "
        "'web' (live web search — only if very recent events needed). "
        "Return ONLY JSON: {\"agents\": [...], \"reasoning\": \"one sentence\"}. No other text."
    ))
    human = HumanMessage(content=f"Query: {state['query']}")
    resp = model.invoke([system, human])
    raw = resp.content.strip().lstrip("```json").rstrip("```").strip()
    try:
        parsed = json.loads(raw)
        agents = parsed.get("agents", ["vector_db", "sql_db"])
        reason = parsed.get("reasoning", "")
    except Exception:
        agents = ["vector_db", "sql_db"]
        reason = "defaulted"

    log_entry = {
        "agent": "router",
        "icon": "🔀",
        "title": "Router decided",
        "detail": f"Activating: {', '.join(agents)} — {reason}",
        "ts": _stamp(),
    }
    return {
        "active_agents":    agents,
        "router_reasoning": reason,
        "messages":         [AIMessage(content=f"[Router] {reason} → {agents}")],
        "activity_log":     [log_entry],
        "current_agent":    "router",
    }


# ── 2. Vector DB Agent ────────────────────────────────────────────────────────

def vector_db_agent(state: AgentState, model: ChatOpenAI, vdb: VectorDBModule) -> dict:
    if "vector_db" not in state.get("active_agents", []):
        return {
            "vector_findings": "(vector_db agent not activated)",
            "activity_log": [{
                "agent": "vector_db", "icon": "🗂️", "title": "Vector DB agent — skipped",
                "detail": "Router did not activate this agent.", "ts": _stamp(),
            }],
            "messages": [AIMessage(content="[VectorDB] skipped")],
            "current_agent": "vector_db",
        }

    docs = vdb.search(state["query"], k=6)
    if not docs:
        raw_context = "(no documents in vector store)"
        sources_used = []
    else:
        raw_context = "\n\n---\n".join(
            f"[{d.metadata.get('source','?')}]\n{d.page_content}" for d in docs
        )
        sources_used = list({d.metadata.get("source", "?") for d in docs})

    system = SystemMessage(content=(
        "You are a Vector DB Search Agent. Given a user query and retrieved document "
        "chunks, synthesise the most relevant information into structured research notes. "
        "Include any specific facts, definitions, and relationships you find."
    ))
    human = HumanMessage(content=f"Query: {state['query']}\n\nRetrieved chunks:\n{raw_context}")
    resp = model.invoke([system, human])

    log_entry = {
        "agent":  "vector_db",
        "icon":   "🗂️",
        "title":  "Vector DB agent",
        "detail": f"Searched {len(docs)} chunks from {len(sources_used)} source(s): {', '.join(sources_used) if sources_used else 'none'}",
        "chunks": [{"source": d.metadata.get("source","?"), "text": d.page_content[:200]+"…"} for d in docs],
        "ts": _stamp(),
    }
    return {
        "vector_findings": resp.content,
        "messages":        [AIMessage(content=f"[VectorDB] {resp.content[:120]}…")],
        "activity_log":    [log_entry],
        "current_agent":   "vector_db",
    }


# ── 3. SQL / DB Agent ─────────────────────────────────────────────────────────

def sql_db_agent(state: AgentState, model: ChatOpenAI) -> dict:
    if "sql_db" not in state.get("active_agents", []):
        return {
            "sql_findings": "(sql_db agent not activated)",
            "activity_log": [{
                "agent": "sql_db", "icon": "🗄️", "title": "SQL DB agent — skipped",
                "detail": "Router did not activate this agent.", "ts": _stamp(),
            }],
            "messages": [AIMessage(content="[SQLDB] skipped")],
            "current_agent": "sql_db",
        }

    raw_results = sql_search(state["query"], k=10)
    system = SystemMessage(content=(
        "You are a SQL Database Search Agent. Given a user query and raw SQL query results "
        "(topics, relationships, facts), extract and summarise the most relevant structured "
        "information. Note specific facts, entities, and typed relationships."
    ))
    human = HumanMessage(content=f"Query: {state['query']}\n\nSQL results:\n{raw_results}")
    resp = model.invoke([system, human])

    rows_found = [l for l in raw_results.split("\n") if l.strip()]
    log_entry = {
        "agent":  "sql_db",
        "icon":   "🗄️",
        "title":  "SQL / DB agent",
        "detail": f"Queried topics, relationships, facts tables — {len(rows_found)} row(s) matched",
        "rows":   rows_found[:12],
        "ts": _stamp(),
    }
    return {
        "sql_findings": resp.content,
        "messages":     [AIMessage(content=f"[SQLDB] {resp.content[:120]}…")],
        "activity_log": [log_entry],
        "current_agent": "sql_db",
    }


# ── 4. Web / API Agent ────────────────────────────────────────────────────────

def web_agent(state: AgentState, model: ChatOpenAI, vdb: VectorDBModule = None) -> dict:
    if "web" not in state.get("active_agents", []):
        return {
            "web_findings": "(web agent not activated)",
            "activity_log": [{
                "agent": "web", "icon": "🌐", "title": "Web agent — skipped",
                "detail": "Router did not activate this agent.", "ts": _stamp(),
            }],
            "messages": [AIMessage(content="[Web] skipped")],
            "current_agent": "web",
        }

    # Use basic search example for comprehensive scholarly API search
    api_results = search_scientific_apis(state['query'], limit_per_api=5)
    
    findings = []
    indexed_count = 0
    
    for api_name, results in api_results.items():
        if api_name.endswith("_error"):
            findings.append(f"{api_name}: {results[0].get('error', 'Unknown error')}")
            continue
            
        findings.append(f"=== {api_name.upper()} ===")
        for result in results:
            title = result.get("title", "No title")
            url = result.get("best_url", result.get("abstract_url", result.get("doi_url", "")))
            authors = result.get("authors", [])
            year = result.get("year", result.get("published", ""))
            
            # Format the result
            result_text = f"**{title}**\n"
            if authors:
                result_text += f"Authors: {', '.join(authors)}\n"
            if year:
                result_text += f"Year: {year}\n"
            if url:
                result_text += f"URL: {url}\n"
            
            # For arXiv results, try to fetch and index the abstract
            if api_name == "arxiv" and vdb and url:
                try:
                    # Fetch the abstract page content (simplified - in practice you'd want to parse HTML)
                    import requests
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        # Extract abstract from HTML (basic approach)
                        content = response.text
                        # Look for abstract in the HTML
                        if "Abstract:" in content:
                            abstract_start = content.find("Abstract:")
                            abstract_end = content.find("</blockquote>", abstract_start) if "</blockquote>" in content[abstract_start:] else len(content)
                            abstract = content[abstract_start:abstract_end].replace("Abstract:", "").strip()
                            
                            # Index the document
                            doc_content = f"Title: {title}\nAuthors: {', '.join(authors)}\nAbstract: {abstract}"
                            vdb.add_text(doc_content, {"source": f"arXiv:{url}", "title": title, "authors": authors})
                            indexed_count += 1
                            result_text += f"[Indexed in vector DB]\n"
                except Exception as e:
                    result_text += f"[Indexing failed: {str(e)}]\n"
            
            findings.append(result_text)

    combined_findings = "\n\n".join(findings)
    
    log_entry = {
        "agent":  "web",
        "icon":   "🌐",
        "title":  "Web / API agent",
        "detail": f"Queried {len([k for k in api_results.keys() if not k.endswith('_error')])} APIs, indexed {indexed_count} documents",
        "ts": _stamp(),
    }
    return {
        "web_findings": combined_findings,
        "messages":     [AIMessage(content=f"[Web] Found results from {len(api_results)} sources, indexed {indexed_count} docs")],
        "activity_log": [log_entry],
        "current_agent": "web",
    }


# ── 5. Orchestrator ───────────────────────────────────────────────────────────

def orchestrator_agent(state: AgentState, model: ChatOpenAI) -> dict:
    findings_block = "\n\n".join([
        f"=== Vector DB Findings ===\n{state.get('vector_findings','')}",
        f"=== SQL / DB Findings ===\n{state.get('sql_findings','')}",
        f"=== Web / API Findings ===\n{state.get('web_findings','')}",
    ])
    system = SystemMessage(content=(
        "You are an Orchestrator Agent. You receive findings from multiple specialised search "
        "agents (Vector DB, SQL DB, Web). Your job is to:\n"
        "1. Merge and deduplicate the findings\n"
        "2. Resolve any contradictions, preferring higher-confidence sources\n"
        "3. Weight evidence: structured DB facts > vector chunks > web (unless web is very recent)\n"
        "4. Produce a single coherent merged context for downstream agents\n"
        "Be explicit about which source contributed each piece of information."
    ))
    human = HumanMessage(content=f"Query: {state['query']}\n\n{findings_block}")
    resp = model.invoke([system, human])

    log_entry = {
        "agent":  "orchestrator",
        "icon":   "🤝",
        "title":  "Orchestrator merged findings",
        "detail": (
            f"Sources merged: "
            + (", ".join([
                a for a in ["Vector DB","SQL DB","Web"]
                if "not activated" not in state.get({
                    "Vector DB": "vector_findings",
                    "SQL DB": "sql_findings",
                    "Web": "web_findings",
                }[a], "not activated")
            ]) or "none")
        ),
        "ts": _stamp(),
    }
    return {
        "merged_context": resp.content,
        "messages":       [AIMessage(content=f"[Orchestrator] {resp.content[:120]}…")],
        "activity_log":   [log_entry],
        "current_agent":  "orchestrator",
    }


# ── 6. Knowledge Mapper ───────────────────────────────────────────────────────

def knowledge_mapper_agent(state: AgentState, model: ChatOpenAI) -> dict:
    system = SystemMessage(content=(
        "You are a Knowledge Mapping Agent. Given merged research context, extract a rich "
        "knowledge graph. Return ONLY valid JSON:\n"
        '{"nodes": [{"id": "string", "label": "string", "type": "concept|entity|fact|process", '
        '"source": "vector_db|sql_db|web|merged"}], '
        '"edges": [{"source": "string", "target": "string", "relation": "string", '
        '"weight": 0.1..1.0}]}\n'
        "Include 12–20 nodes. No text outside JSON."
    ))
    human = HumanMessage(content=f"Query: {state['query']}\n\nMerged context:\n{state['merged_context']}")
    resp = model.invoke([system, human])

    raw = resp.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("```").strip()
    try:
        km = json.loads(raw)
    except Exception:
        km = {"nodes": [], "edges": [], "error": "parse_failed"}

    log_entry = {
        "agent":  "knowledge_mapper",
        "icon":   "🗺️",
        "title":  "Knowledge mapper built graph",
        "detail": f"{len(km.get('nodes',[]))} nodes, {len(km.get('edges',[]))} edges extracted",
        "ts": _stamp(),
    }
    return {
        "knowledge_map": km,
        "messages":      [AIMessage(content=f"[KnowledgeMapper] {len(km.get('nodes',[]))} nodes built.")],
        "activity_log":  [log_entry],
        "current_agent": "knowledge_mapper",
    }


# ── 7. Critic (enrichment loop) ───────────────────────────────────────────────

def critic_agent(state: AgentState, model: ChatOpenAI) -> dict:
    system = SystemMessage(content=(
        "You are a Critic Agent. Review the knowledge map. "
        "If it has fewer than 8 nodes OR key source-type diversity is missing, "
        "respond: {\"needs_more\": true, \"feedback\": \"specific gaps\"}. "
        "Otherwise: {\"needs_more\": false, \"feedback\": \"\"}. Only JSON."
    ))
    human = HumanMessage(content=(
        f"Nodes: {[n['label'] for n in state['knowledge_map'].get('nodes', [])]}\n"
        f"Source types: {list({n.get('source','?') for n in state['knowledge_map'].get('nodes',[])})} \n"
        f"Edges: {len(state['knowledge_map'].get('edges', []))}"
    ))
    resp = model.invoke([system, human])
    raw = resp.content.strip().lstrip("```json").rstrip("```").strip()
    try:
        result = json.loads(raw)
    except Exception:
        result = {"needs_more": False, "feedback": ""}

    log_entry = {
        "agent":  "critic",
        "icon":   "🧐",
        "title":  f"Critic — {'needs enrichment' if result.get('needs_more') else 'approved'}",
        "detail": result.get("feedback","Graph is sufficient."),
        "ts": _stamp(),
    }
    return {
        "critique":      result.get("feedback", ""),
        "_needs_more":   result.get("needs_more", False),
        "loop_count":    state.get("loop_count", 0) + 1,
        "messages":      [AIMessage(content=f"[Critic] needs_more={result.get('needs_more')} | {result.get('feedback','')}")],
        "activity_log":  [log_entry],
        "current_agent": "critic",
    }


# ── 8. Summarizer ─────────────────────────────────────────────────────────────

def summarizer_agent(state: AgentState, model: ChatOpenAI) -> dict:
    system = SystemMessage(content=(
        "You are a Summarizer Agent. Using the merged context and knowledge map, "
        "write a clear, well-structured, grounded answer. Cite which data source "
        "(Vector DB / SQL DB / Web) each key claim comes from."
    ))
    human = HumanMessage(content=(
        f"Query: {state['query']}\n\n"
        f"Merged context:\n{state['merged_context']}\n\n"
        f"Key concepts: {[n['label'] for n in state['knowledge_map'].get('nodes', [])]}"
    ))
    resp = model.invoke([system, human])
    log_entry = {
        "agent":  "summarizer",
        "icon":   "✍️",
        "title":  "Summarizer wrote final answer",
        "detail": f"{len(resp.content)} characters",
        "ts": _stamp(),
    }
    return {
        "summary":       resp.content,
        "messages":      [AIMessage(content=f"[Summarizer] {resp.content[:120]}…")],
        "activity_log":  [log_entry],
        "current_agent": "summarizer",
    }


# ── Routing function ──────────────────────────────────────────────────────────

def _route_critic(state: AgentState) -> str:
    if state.get("_needs_more") and state.get("loop_count", 0) < 2:
        return "orchestrator"  # re-merge with critique hint
    return "summarizer"


def index_arxiv_documents(query: str, vdb: VectorDBModule, limit: int = 10) -> int:
    """
    Search arXiv for documents and index them in the vector database.
    Uses the arxiv.py library for better data access.
    Returns the number of documents indexed.
    """
    print(f"[ArXiv Indexer] Searching for '{query}' using arxiv.py library...")
    
    try:
        import arxiv
        client = arxiv.Client()
        
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = list(client.results(search))
        print(f"[ArXiv Indexer] Found {len(results)} results from arXiv")
        
    except Exception as e:
        print(f"[ArXiv Indexer] Failed to query arXiv: {str(e)}")
        return 0
    
    indexed_count = 0
    for result in results:
        try:
            title = result.title.strip()
            authors = [author.name for author in result.authors]
            abstract = result.summary.strip()
            pdf_url = result.pdf_url
            entry_id = result.entry_id
            
            if not title:
                print(f"[ArXiv Indexer] Skipping result with no title")
                continue
            
            print(f"[ArXiv Indexer] Processing: {title[:50]}...")
            
            # Create document content
            doc_content = f"Title: {title}\nAuthors: {', '.join(authors)}\nAbstract: {abstract}"
            
            # Index in vector DB
            vdb.add_text(doc_content, {
                "source": f"arXiv:{entry_id}",
                "title": title,
                "authors": authors,
                "url": entry_id,
                "pdf_url": pdf_url,
                "indexed_at": _stamp()
            })
            indexed_count += 1
            print(f"[ArXiv Indexer] Successfully indexed: {title[:50]}...")
                    
        except Exception as e:
            print(f"[ArXiv Indexer] Failed to index '{result.title[:30] if hasattr(result, 'title') else 'unknown'}...': {str(e)}")
            continue
    
    print(f"[ArXiv Indexer] Completed: {indexed_count} documents indexed out of {len(results)} found")
    return indexed_count


def index_nasa_documents(vdb: VectorDBModule, limit: int = 10) -> int:
    """
    Index NASA Astronomy Picture of the Day and other NASA data.
    Returns the number of documents indexed.
    """
    print(f"[NASA Indexer] Fetching NASA data...")
    
    try:
        from secrets_manager import get_api_key
        nasa_key = get_api_key('nasa') or 'DEMO_KEY'  # Use DEMO_KEY as fallback
            
        nasa = Nasa(nasa_key)
        
        # Get Astronomy Picture of the Day
        apod = nasa.picture_of_the_day()
        if apod and 'title' in apod and 'explanation' in apod:
            title = apod['title']
            explanation = apod['explanation']
            url = apod.get('url', '')
            
            doc_content = f"NASA APOD: {title}\n\n{explanation}"
            
            vdb.add_text(doc_content, {
                "source": f"NASA:APOD",
                "title": title,
                "url": url,
                "indexed_at": _stamp()
            })
            
            print(f"[NASA Indexer] Indexed APOD: {title}")
            return 1
        else:
            print("[NASA Indexer] Failed to fetch APOD")
            return 0
            
    except Exception as e:
        print(f"[NASA Indexer] Failed to index NASA data: {str(e)}")
        return 0


def build_graph(api_key: str, vdb: VectorDBModule):
    lm_route  = llm(api_key, temperature=0.0)
    lm_search = llm(api_key, temperature=0.3)
    lm_orch   = llm(api_key, temperature=0.2)
    lm_map    = llm(api_key, temperature=0.1)
    lm_critic = llm(api_key, temperature=0.0)
    lm_sum    = llm(api_key, temperature=0.5)

    g = StateGraph(AgentState)

    g.add_node("router",           lambda s: router_agent(s, lm_route))
    g.add_node("vector_db",        lambda s: vector_db_agent(s, lm_search, vdb))
    g.add_node("sql_db",           lambda s: sql_db_agent(s, lm_search))
    g.add_node("web",              lambda s: web_agent(s, lm_search, vdb))
    g.add_node("orchestrator",     lambda s: orchestrator_agent(s, lm_orch))
    g.add_node("knowledge_mapper", lambda s: knowledge_mapper_agent(s, lm_map))
    g.add_node("critic",           lambda s: critic_agent(s, lm_critic))
    g.add_node("summarizer",       lambda s: summarizer_agent(s, lm_sum))

    g.set_entry_point("router")
    g.add_edge("router",           "vector_db")
    g.add_edge("vector_db",        "sql_db")
    g.add_edge("sql_db",           "web")
    g.add_edge("web",              "orchestrator")
    g.add_edge("orchestrator",     "knowledge_mapper")
    g.add_edge("knowledge_mapper", "critic")
    g.add_conditional_edges("critic", _route_critic,
                             {"orchestrator": "orchestrator", "summarizer": "summarizer"})
    g.add_edge("summarizer", END)

    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

SOURCE_COLORS = {
    "vector_db": "#4A90D9",
    "sql_db":    "#E67E22",
    "web":       "#2ECC71",
    "merged":    "#9B59B6",
}
TYPE_SHAPES = {
    "concept":  "dot",
    "entity":   "diamond",
    "fact":     "square",
    "process":  "triangleDown",
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
        "nodes": {
            "font": {"size": 12, "bold": True},
            "borderWidth": 2,
            "shadow": True,
        },
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
        color = SOURCE_COLORS.get(node.get("source", "merged"), SOURCE_COLORS["merged"])
        shape = TYPE_SHAPES.get(node.get("type", "concept"), "dot")
        net.add_node(
            node["id"], label=node["label"], color=color, shape=shape,
            title=(
                f"<b>{node['label']}</b><br>"
                f"Type: {node.get('type','?')}<br>"
                f"Source: {node.get('source','?')}"
            ),
            size=22,
        )

    for edge in km.get("edges", []):
        weight = edge.get("weight", 0.5)
        net.add_edge(
            edge["source"], edge["target"],
            title=edge.get("relation",""),
            label=edge.get("relation",""),
            width=max(1, weight * 4),
        )

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        html = Path(f.name).read_text()
        Path(f.name).unlink(missing_ok=True)
    return html


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT  UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Collaborative RAG Pipeline",
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

/* Activity log card */
.agent-card{
  border-left:3px solid;
  padding:10px 14px;
  margin-bottom:10px;
  border-radius:0 6px 6px 0;
  background:#161b22;
}
.agent-card.router       {border-color:#9B59B6}
.agent-card.vector_db    {border-color:#4A90D9}
.agent-card.sql_db       {border-color:#E67E22}
.agent-card.web          {border-color:#2ECC71}
.agent-card.orchestrator {border-color:#E74C3C}
.agent-card.knowledge_mapper{border-color:#1ABC9C}
.agent-card.critic       {border-color:#F39C12}
.agent-card.summarizer   {border-color:#95A5A6}
.agent-title{font-weight:600;font-size:0.88rem;margin-bottom:3px}
.agent-detail{font-size:0.78rem;color:#8b949e}
.source-badge{
  display:inline-block;padding:2px 8px;border-radius:12px;
  font-size:0.68rem;font-family:'JetBrains Mono',monospace;
  letter-spacing:.04em;margin:0 3px;
}
.badge-vector{background:#1a3a5c;color:#4A90D9}
.badge-sql   {background:#4a2010;color:#E67E22}
.badge-web   {background:#0f3d20;color:#2ECC71}
.badge-merged{background:#2d1a4a;color:#9B59B6}
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

    api_key = get_api_key('openai')
    if not api_key:
        st.error("❌ OpenAI API key not configured")
        st.info("**To configure API keys:**")
        st.code("python setup_env.py", language="bash")
        st.markdown("Or run: `python secrets_manager.py`")
        st.stop()  # Prevent the app from running without API key
    else:
        st.success("✅ OpenAI API key configured")
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        st.caption(f"Key: {masked_key}")
        
        # Option to override
        if st.checkbox("Use different key"):
            api_key = st.text_input("Override API Key", type="password", placeholder="sk-…")

    if api_key and (st.session_state.vdb is None
                    or st.session_state.get("_api_key") != api_key):
        st.session_state.vdb = VectorDBModule(api_key)
        st.session_state._api_key = api_key

    vdb: Optional[VectorDBModule] = st.session_state.vdb

    st.divider()
    st.markdown("### 📄 Vector DB — Index Documents")
    uploaded = st.file_uploader("Upload .txt / .md", type=["txt","md"],
                                accept_multiple_files=True)
    if uploaded and vdb:
        for f in uploaded:
            n = vdb.add_file(f)
            st.success(f"{f.name} → {n} chunks")

    if vdb:
        st.metric("Vector chunks", vdb.count())
        srcs = vdb.sources()
        if srcs:
            with st.expander(f"📚 {len(srcs)} document(s)"):
                for s in srcs:
                    st.markdown(f"• `{s}`")

    st.divider()
    st.markdown("### 🗄️ SQL DB — Add Topic")
    with st.form("add_topic"):
        t_title = st.text_input("Title")
        t_cat   = st.text_input("Category")
        t_sum   = st.text_area("Summary", height=60)
        t_kw    = st.text_input("Keywords (comma-separated)")
        if st.form_submit_button("Add"):
            msg = sql_insert_topic(t_title, t_cat, t_sum, t_kw)
            st.success(msg)

    st.divider()
    st.markdown("### Legend")
    for label, color, cls in [
        ("Vector DB","#4A90D9","badge-vector"),
        ("SQL / DB", "#E67E22","badge-sql"),
        ("Web",      "#2ECC71","badge-web"),
        ("Merged",   "#9B59B6","badge-merged"),
    ]:
        st.markdown(
            f'<span class="source-badge {cls}">{label}</span>',
            unsafe_allow_html=True,
        )

# ── Main tabs ─────────────────────────────────────────────────────────────────

main_tabs = st.tabs([
    "🚀 Research",
    "🗄️ SQL DB",
    "🗺️ Saved Maps",
    "💬 Conversations",
    "🕐 Cache",
])

# ════════════════ TAB 1 — RESEARCH ════════════════════════════════════════════
with main_tabs[0]:
    st.header("Collaborative Research Query")

    col_q, col_o = st.columns([3,1])
    with col_q:
        query = st.text_area("Query", height=90,
                             placeholder="e.g. How does RAG relate to transformer architecture?")
    with col_o:
        use_cache     = st.checkbox("Use cache (20 days)", value=True)
        auto_save_map = st.checkbox("Auto-save map",       value=True)

    run = st.button("🚀 Run Collaborative Pipeline", type="primary",
                    disabled=not (api_key and query))

    if run:
        if not api_key.startswith("sk-"):
            st.error("Invalid API key.")
            st.stop()

        # Cache check
        cached = cache_load(query) if use_cache else None
        if cached:
            st.info("⚡ Loaded from 20-day cache")
            full_state = cached
        else:
            app = build_graph(api_key, st.session_state.vdb)
            progress = st.progress(0, "Starting…")
            step_pct = {"router":20,"vector_db":35,"sql_db":50,"web":60,
                        "orchestrator":70,"knowledge_mapper":82,"critic":90,"summarizer":98}
            for event in app.stream({
                "messages":[], "query":query,
                "active_agents":[], "router_reasoning":"",
                "vector_findings":"", "sql_findings":"", "web_findings":"",
                "activity_log":[], "merged_context":"",
                "knowledge_map":{}, "critique":"", "loop_count":0, "summary":"",
                "current_agent":"",
            }):
                for node in event:
                    lbl = node.replace("_"," ").title()
                    progress.progress(step_pct.get(node,50), f"{lbl} running…")

            full_state = app.invoke({
                "messages":[], "query":query,
                "active_agents":[], "router_reasoning":"",
                "vector_findings":"", "sql_findings":"", "web_findings":"",
                "activity_log":[], "merged_context":"",
                "knowledge_map":{}, "critique":"", "loop_count":0, "summary":"",
                "current_agent":"",
            })
            progress.progress(100,"✅ Done!")

            if use_cache:
                cache_save(query, full_state)

        # Save map
        if auto_save_map and full_state.get("knowledge_map",{}).get("nodes"):
            fname = map_save(query, full_state["knowledge_map"])
            st.success(f"🗺️ Map saved → `{fname}`")

        # Save turn
        turn = {
            "query":     query,
            "summary":   full_state.get("summary",""),
            "agents":    full_state.get("active_agents",[]),
            "nodes":     len(full_state.get("knowledge_map",{}).get("nodes",[])),
            "ts":        _stamp(),
            "cached":    cached is not None,
        }
        st.session_state.turns.append(turn)
        session_save(st.session_state.session_id, st.session_state.turns)

        # ── Results ──────────────────────────────────────────────────────────
        r1, r2, r3, r4, r5, r6 = st.tabs([
            "🤝 Agent Activity",
            "💡 Final Answer",
            "🗺️ Knowledge Map",
            "🔀 Merged Context",
            "🔍 Per-Agent Findings",
            "💬 Message Log",
        ])

        # ─── Agent Activity (the cool real-time panel) ────────────────────
        with r1:
            st.subheader("What each agent did")
            activity_log = full_state.get("activity_log", [])
            for entry in activity_log:
                agent_cls = entry.get("agent","")
                icon      = entry.get("icon","•")
                title     = entry.get("title","")
                detail    = entry.get("detail","")

                # Source badge
                badge_map = {
                    "vector_db": '<span class="source-badge badge-vector">Vector DB</span>',
                    "sql_db":    '<span class="source-badge badge-sql">SQL DB</span>',
                    "web":       '<span class="source-badge badge-web">Web</span>',
                    "orchestrator": '<span class="source-badge badge-merged">Orchestrator</span>',
                }
                badge = badge_map.get(agent_cls,"")

                st.markdown(
                    f"""<div class="agent-card {agent_cls}">
                    <div class="agent-title">{icon} {title} {badge}</div>
                    <div class="agent-detail">{detail}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Show retrieved chunks for vector_db
                if agent_cls == "vector_db" and entry.get("chunks"):
                    with st.expander(f"📄 {len(entry['chunks'])} retrieved chunks"):
                        for ch in entry["chunks"]:
                            st.markdown(f"**`{ch['source']}`**")
                            st.caption(ch["text"])

                # Show SQL rows
                if agent_cls == "sql_db" and entry.get("rows"):
                    with st.expander(f"🗄️ {len(entry['rows'])} matched rows"):
                        for row in entry["rows"]:
                            st.code(row, language="text")

        # ─── Final answer ─────────────────────────────────────────────────
        with r2:
            st.markdown(full_state.get("summary",""))

        # ─── Knowledge Map ────────────────────────────────────────────────
        with r3:
            km = full_state.get("knowledge_map",{})
            if km.get("nodes"):
                # Source distribution
                source_counts: dict[str, int] = {}
                for n in km["nodes"]:
                    s = n.get("source","merged")
                    source_counts[s] = source_counts.get(s,0) + 1
                cols = st.columns(len(source_counts))
                for i,(src,cnt) in enumerate(source_counts.items()):
                    color_cls = {"vector_db":"badge-vector","sql_db":"badge-sql",
                                 "web":"badge-web","merged":"badge-merged"}.get(src,"badge-merged")
                    cols[i].markdown(
                        f'<span class="source-badge {color_cls}">{src}: {cnt} nodes</span>',
                        unsafe_allow_html=True,
                    )

                html = render_knowledge_map(km)
                st.components.v1.html(html, height=520)

                with st.expander("📊 Raw Graph"):
                    c1,c2 = st.columns(2)
                    with c1:
                        st.markdown("**Nodes**")
                        st.dataframe(km["nodes"])
                    with c2:
                        st.markdown("**Edges**")
                        st.dataframe(km["edges"])
            else:
                st.warning("No map generated.")

        # ─── Merged context ───────────────────────────────────────────────
        with r4:
            st.markdown(full_state.get("merged_context",""))

        # ─── Per-agent findings ───────────────────────────────────────────
        with r5:
            for label, key, badge_cls in [
                ("🗂️ Vector DB","vector_findings","badge-vector"),
                ("🗄️  SQL / DB", "sql_findings",   "badge-sql"),
                ("🌐  Web",       "web_findings",    "badge-web"),
            ]:
                with st.expander(f"{label} findings"):
                    st.markdown(full_state.get(key,""))

        # ─── Message log ──────────────────────────────────────────────────
        with r6:
            avatar_map = {
                "[Router]":          "🔀",
                "[VectorDB]":        "🗂️",
                "[SQLDB]":           "🗄️",
                "[Web]":             "🌐",
                "[Orchestrator]":    "🤝",
                "[KnowledgeMapper]": "🗺️",
                "[Critic]":          "🧐",
                "[Summarizer]":      "✍️",
            }
            for msg in full_state.get("messages",[]):
                av = next((v for k,v in avatar_map.items() if k in msg.content), "🤖")
                st.chat_message("assistant", avatar=av).write(msg.content)

# ════════════════ TAB 2 — SQL DB ══════════════════════════════════════════════
with main_tabs[1]:
    st.header("SQL / DB Browser")
    topics = sql_list_topics()
    st.metric("Topics in DB", len(topics))
    st.dataframe(topics, use_container_width=True)

    st.divider()
    st.subheader("Run a test search")
    test_q = st.text_input("Test keyword search")
    if test_q:
        results = sql_search(test_q)
        st.code(results, language="text")

# ════════════════ TAB 3 — SAVED MAPS ═════════════════════════════════════════
with main_tabs[2]:
    st.header("Saved Knowledge Maps")
    all_maps = map_list()
    if not all_maps:
        st.info("No maps saved yet.")
    else:
        filt = st.text_input("Filter maps")
        shown = [m for m in all_maps if not filt or filt.lower() in m["query"].lower()]
        for m in shown:
            with st.expander(
                f"🗺️ {m['query'][:65]}…  ·  {m['nodes']}n / {m['edges']}e  ·  {m['saved_at'][:16].replace('T',' ')}"
            ):
                if st.button("Visualise", key=f"vis_{m['file']}"):
                    raw = map_load(m["file"])
                    if raw:
                        html = render_knowledge_map(raw["map"])
                        st.components.v1.html(html, height=470)

# ════════════════ TAB 4 — CONVERSATIONS ══════════════════════════════════════
with main_tabs[3]:
    st.header("Conversation History")
    c1, c2 = st.columns([2,3])
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
        all_sessions = session_list()
        sel = None
        for s in all_sessions:
            if st.button(f"📅 {s['updated'][:16].replace('T',' ')}  ·  {s['n']} turns\n{s['preview']}",
                         key=f"sess_{s['sid']}"):
                sel = s["sid"]
    with c2:
        turns = session_load(sel) if sel else st.session_state.turns
        for i,t in enumerate(reversed(turns),1):
            agents_html = " ".join([
                f'<span class="source-badge badge-{a}">{a}</span>'
                for a in t.get("agents",[])
            ])
            st.markdown(
                f"**Turn {len(turns)-i+1}** · {t['ts'][:16].replace('T',' ')} {agents_html}",
                unsafe_allow_html=True,
            )
            st.markdown(f"> 🔍 **{t['query']}**")
            with st.expander("Summary", expanded=(i==1)):
                st.markdown(t.get("summary",""))
            st.markdown(f"🗺️ {t.get('nodes',0)} nodes")
            st.divider()

# ════════════════ TAB 5 — CACHE ══════════════════════════════════════════════
with main_tabs[4]:
    st.header("20-Day Query Cache")
    cache_entries = cache_list()
    st.metric("Cached entries", len(cache_entries))
    filt_c = st.text_input("Filter cache")
    shown_c = [e for e in cache_entries if not filt_c or filt_c.lower() in e["query"].lower()]
    for e in shown_c:
        age = datetime.utcnow() - datetime.fromisoformat(e["ts"])
        age_s = f"{age.days}d {age.seconds//3600}h ago" if age.days else f"{age.seconds//3600}h {(age.seconds%3600)//60}m ago"
        with st.expander(f"🗃️ {e['query'][:75]}…  ·  {age_s}"):
            pl = cache_load(e["query"])
            if pl:
                cols = st.columns(3)
                cols[0].metric("Summary", f"{len(pl.get('summary',''))} chars")
                cols[1].metric("Nodes",   len(pl.get("knowledge_map",{}).get("nodes",[])))
                cols[2].metric("Agents",  ", ".join(pl.get("active_agents",[])))
                if st.button("▶️ Reload", key=f"rc_{e['file']}"):
                    st.markdown(pl.get("summary",""))
                    km = pl.get("knowledge_map",{})
                    if km.get("nodes"):
                        st.components.v1.html(render_knowledge_map(km,400), height=420)

    st.divider()
    if st.button("🧹 Clear ALL cache"):
        for p in CACHE_DIR.glob("*.json"):
            p.unlink()
        st.success("Cache cleared.")