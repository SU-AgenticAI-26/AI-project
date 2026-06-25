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
  [Reading/Extraction Agent]  ← per-paper structured records + provenance
          │
    [Orchestrator Agent]  ← merges, deduplicates, weights evidence
          │
    [Knowledge Mapper]  ← builds graph; loops back if sparse
          │
    [Summarizer]  ← final grounded answer
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import operator
import os
import re
import sqlite3
import tempfile
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
import io

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict

import requests
import streamlit as st
from pyvis.network import Network

# Conference paper search integration
try:
    from conference_paper_search import (
        SEARCH_PAPERS_TOOL,
        handle_conference_paper_tool_call,
        OPENREVIEW_CONFERENCES,
        ACL_CONFERENCES,
    )
    HAS_CONFERENCE_SEARCH = True
except ImportError:
    HAS_CONFERENCE_SEARCH = False

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

# new code
from concurrent.futures import ThreadPoolExecutor, as_completed


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


CACHE_TTL_DAYS = 20


def _load_env_from_project_root() -> None:
    """Load .env from project root and let file values override stale process env."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    parsed: dict[str, str] = {}
    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                parsed[key] = value

    for key, value in parsed.items():
        os.environ[key] = value

    if "OPENAI_API_KEY" not in parsed:
        for alias in ("key", "openai_key", "OPENAI_KEY"):
            if alias in parsed:
                os.environ["OPENAI_API_KEY"] = parsed[alias]
                break


_load_env_from_project_root()

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
# TOOL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# VectorDB Search Tool — used by vector_db_agent for LLM-driven decisions
SEARCH_VECTORDB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_vectordb",
        "description": (
            "Search the Vector Database for relevant documents, papers, or indexed content. "
            "Use this when you need to find information from previously indexed documents, "
            "web search results, or uploaded files. Returns matching documents with source metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents. Keep it concise (1-10 words)."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to retrieve. Default: 5. Range: 1-20."
                },
                "filter_source": {
                    "type": "string",
                    "enum": ["web_search", "arxiv", "openalex", "crossref", "semantic_scholar", "conference", "uploaded"],
                    "description": "Optional: filter results by source type. Omit to search all sources."
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

# SQL DB Search Tool — used by sql_db_agent for LLM-driven decisions
SEARCH_SQLDB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_sqldb",
        "description": (
            "Search the SQL database for structured facts, topics, and relationships. "
            "Use this to find structured knowledge about specific topics, entities, and their relationships. "
            "Returns matching topics, relationships, and facts with categories and sources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find topics, relationships, or facts. Use short, focused terms."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to retrieve. Default: 10. Range: 1-20."
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


# Router guardrails: categorical/enumeration queries should include SQL.
SQL_TRIGGER_PATTERNS = [
    "what are the main",
    "what approaches",
    "what mechanisms",
    "approaches",
    "mechanisms",
    "challenges",
    "list the",
    "compare",
    "how many",
]


def _needs_sql_for_query(query: str) -> bool:
    q = (query or "").lower()
    return any(pattern in q for pattern in SQL_TRIGGER_PATTERNS)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages:              Annotated[List, operator.add]
    query:                 str

    # ── Scoping ───────────────────────────────────────────────────────────────
    sub_questions:         List[str]      # 3-5 decomposed questions
    keywords:              List[str]      # 4-8 key themes
    scoping_reasoning:     str

    # ── Router ────────────────────────────────────────────────────────────────
    active_agents:         List[str]
    router_reasoning:      str

    # ── Retrieval ─────────────────────────────────────────────────────────────
    vector_findings:       str
    sql_findings:          str
    web_findings:          str

    # ── Extraction + merge ────────────────────────────────────────────────────
    extraction_findings:   str
    tagged_findings:       List[dict]     # [{text, source}] chunk-level provenance
    merged_context:        str
    synthesis_report:      str            # optional thematic synthesis

    # ── Knowledge graph ───────────────────────────────────────────────────────
    knowledge_map:         dict

    # ── Critic loop ───────────────────────────────────────────────────────────
    critique:              str
    _needs_more:           bool
    loop_count:            int

    # ── Conflict detection (Block 3) ──────────────────────────────────────────
    conflicts:             List[dict]     # [{topic, claim_a, source_a, claim_b, source_b, resolution}]
    credibility_map:       dict           # {source: {label, score}}

    # ── Final outputs ─────────────────────────────────────────────────────────
    summary:               str
    citation_grounding:    dict           # {citation: {grounded, source, evidence}}
    grounding_score:       float          # 0.0-1.0
    experiment_plan:       str

    # ── Metadata ──────────────────────────────────────────────────────────────
    activity_log:          Annotated[List, operator.add]
    current_agent:         Annotated[str, lambda _old, new: new]


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()

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
    if datetime.now(timezone.utc) - datetime.fromisoformat(raw["ts"]).replace(tzinfo=timezone.utc) > timedelta(days=CACHE_TTL_DAYS):
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
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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

# ╭─ BLOCK 4: EXPORT (MARKDOWN + BIBTEX) ─────────────────────────────────────

def extract_bibtex_entries(extraction_findings: str, web_findings: str) -> tuple[str, list[dict]]:
    """
    Extract paper metadata from extraction findings and generate BibTeX entries.
    Returns (bibtex_str, papers_list).
    """
    bibtex_entries = []
    papers = []
    
    # Parse extraction findings for paper records
    # Expected format from reading_extraction_agent: structured per-paper records
    lines = extraction_findings.split('\n')
    
    current_paper = {}
    counter = 1
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try to extract title (usually first substantive line)
        if line.startswith('Title:') or line.startswith('**Title'):
            title = re.sub(r'\*\*|Title:\s*', '', line).strip()
            if title and len(title) > 5:
                current_paper['title'] = title
        
        # Extract authors/source info
        if 'author' in line.lower() or 'source:' in line.lower():
            current_paper['author'] = re.sub(r'[*_]|Author[s]*:|Source:', '', line).strip()[:100]
        
        # Try to get year from extraction or default to current
        if re.search(r'\b(20\d{2})\b', line):
            year = re.search(r'\b(20\d{2})\b', line).group(1)
            current_paper['year'] = year
    
    # Also extract from web_findings if available
    for match in re.finditer(r'Title:\s*([^(]+)\(([^)]+)\)', web_findings):
        title = match.group(1).strip()
        source_info = match.group(2).strip()
        
        paper = {
            'title': title,
            'author': source_info.split(',')[0] if ',' in source_info else 'Unknown',
            'year': re.search(r'20\d{2}', source_info).group(0) if re.search(r'20\d{2}', source_info) else '2024',
            'url': '',
        }
        papers.append(paper)
    
    # Generate BibTeX
    bibtex_str = "% Bibliography generated by Collaborative RAG\n"
    bibtex_str += f"% Generated: {_stamp()}\n\n"
    
    if not papers:
        papers = [
            {
                'title': 'Literature Review Results',
                'author': 'Collaborative RAG System',
                'year': '2024',
                'url': '',
            }
        ]
    
    for i, paper in enumerate(papers[:50], 1):  # Max 50 entries
        title = paper.get('title', f'Paper {i}').replace('"', '\\"')
        author = paper.get('author', 'Anonymous').replace('"', '\\"')
        year = paper.get('year', '2024')
        
        # Create sanitized key
        key = f"ref{i:03d}"
        
        bibtex_str += (
            f"@article{{{key},\n"
            f'  title="{title}",\n'
            f'  author="{author}",\n'
            f'  year="{year}"\n'
            f"}}\n\n"
        )
    
    return bibtex_str, papers

def generate_markdown_report(
    query: str,
    sub_questions: list[str],
    keywords: list[str],
    summary: str,
    experiment_plan: str,
    extraction_findings: str,
    knowledge_map: dict,
) -> str:
    """Generate a complete markdown literature review document."""
    
    md = f"# Literature Review: {query}\n\n"
    md += f"**Generated**: {_stamp()}\n\n"
    
    # Query understanding
    md += "## Research Question Decomposition\n\n"
    md += f"**Primary Query**: {query}\n\n"
    
    if sub_questions:
        md += "### Sub-Questions\n"
        for i, q in enumerate(sub_questions[:5], 1):
            md += f"{i}. {q}\n"
        md += "\n"
    
    if keywords:
        md += "### Key Themes\n"
        md += ", ".join([f"`{k}`" for k in keywords[:8]]) + "\n\n"
    
    # Literature summary
    md += "## Literature Summary\n\n"
    md += summary.strip() + "\n\n"
    
    # Structured findings
    md += "## Structured Findings\n\n"
    if extraction_findings.strip():
        md += extraction_findings.strip() + "\n\n"
    
    # Knowledge map overview
    if knowledge_map.get('nodes'):
        md += "## Knowledge Structure\n\n"
        md += f"**Nodes Identified**: {len(knowledge_map.get('nodes', []))}\n"
        md += f"**Relationships**: {len(knowledge_map.get('edges', []))}\n\n"
        
        md += "### Key Concepts\n"
        for node in knowledge_map.get('nodes', [])[:15]:
            label = node.get('label', 'Unknown')
            node_type = node.get('type', 'concept')
            md += f"- **{label}** ({node_type})\n"
        md += "\n"
    
    # Research plan
    md += "## Proposed Research Direction\n\n"
    if experiment_plan.strip():
        md += experiment_plan.strip() + "\n\n"
    
    # Footer
    md += "---\n"
    md += "_This literature review was generated using a multi-agent retrieval system._\n"
    md += "_Powered by: Vector DB + SQL DB + Web APIs + LLM synthesis_\n"
    
    return md

def create_export_zip(
    query: str,
    full_state: dict,
) -> tuple[bytes, str]:
    """
    Create a .zip file containing markdown report + bibtex bibliography.
    Returns (zip_bytes, filename).
    """
    import io
    import zipfile
    
    # Generate content
    markdown = generate_markdown_report(
        query=query,
        sub_questions=full_state.get("sub_questions", []),
        keywords=full_state.get("keywords", []),
        summary=full_state.get("summary", ""),
        experiment_plan=full_state.get("experiment_plan", ""),
        extraction_findings=full_state.get("extraction_findings", ""),
        knowledge_map=full_state.get("knowledge_map", {}),
    )
    
    bibtex, papers = extract_bibtex_entries(
        extraction_findings=full_state.get("extraction_findings", ""),
        web_findings=full_state.get("web_findings", ""),
    )
    
    # Create zip in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add markdown
        clean_query = re.sub(r'[^a-z0-9\s]', '', query.lower())[:50]
        md_filename = f"literature_review_{clean_query[:30]}.md"
        zipf.writestr(md_filename, markdown)
        
        # Add bibtex
        bib_filename = f"bibliography_{clean_query[:30]}.bib"
        zipf.writestr(bib_filename, bibtex)
        
        # Add metadata
        metadata = {
            "query": query,
            "generated_at": _stamp(),
            "papers_count": len(papers),
            "files": [md_filename, bib_filename],
        }
        zipf.writestr("metadata.json", json.dumps(metadata, indent=2))
    
    zip_buffer.seek(0)
    filename = f"literature_review_{_hash(query)}.zip"
    
    return zip_buffer.getvalue(), filename


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
    words = [w.strip("?.!,") for w in query.lower().split() if len(w) > 3][:5]

    if not words:
        con.close()
        return "(no SQL results)"

    results = []
    like = [f"%{w}%" for w in words]

    # One query per table, all keywords combined with OR — avoids N×3 round-trips.
    topic_clause = " OR ".join(
        ["LOWER(title) LIKE ? OR LOWER(keywords) LIKE ? OR LOWER(summary) LIKE ?"] * len(words)
    )
    topic_params = [p for w in like for p in (w, w, w)]
    for row in cur.execute(
        f"SELECT title,category,summary FROM topics WHERE {topic_clause}",
        topic_params,
    ).fetchall():
        results.append(f"[TOPIC] {row[0]} ({row[1]}): {row[2]}")

    rel_clause = " OR ".join(
        ["LOWER(from_topic) LIKE ? OR LOWER(to_topic) LIKE ?"] * len(words)
    )
    rel_params = [p for w in like for p in (w, w)]
    for row in cur.execute(
        f"SELECT from_topic,relation_type,to_topic FROM relationships WHERE {rel_clause}",
        rel_params,
    ).fetchall():
        results.append(f"[REL] {row[0]} —[{row[1]}]→ {row[2]}")

    fact_clause = " OR ".join(
        ["LOWER(subject) LIKE ? OR LOWER(object) LIKE ?"] * len(words)
    )
    fact_params = [p for w in like for p in (w, w)]
    for row in cur.execute(
        f"SELECT subject,predicate,object,source FROM facts WHERE {fact_clause}",
        fact_params,
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
        if not DOCS_DIR.exists():
            return []
        return [p.name for p in DOCS_DIR.iterdir() if p.is_file()]

    # FIX 1: Re-index saved docs on startup 
    # Rebuild FAISS index from docs already saved to DOCS_DIR.
    # Fixes the Codespace restart problem where the FAISS index gets wiped,
    # but the raw document files survive because they are tracked by git.
    def reindex_saved_docs(self) -> int:
        if self.count() > 0:
            return 0  # index already has data so skip
        if not DOCS_DIR.exists():
            return 0
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


def _fetch_local_models(base_url: str) -> list[str]:
    """Query a local OpenAI-compatible server for its available models."""
    import urllib.request, json as _json
    # Strip trailing /v1 or /v1/ so we can reconstruct the path cleanly
    root = base_url.rstrip("/")
    if not root.endswith("/v1"):
        root = root + "/v1"
    try:
        with urllib.request.urlopen(f"{root}/models", timeout=2) as resp:
            data = _json.loads(resp.read())
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []


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
        from langchain_community.embeddings import HuggingFaceEmbeddings
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

# ╭─ BLOCK 1: QUERY SCOPING (UI EXPLICITNESS) ──────────────────────────────────
def scoping_agent(state: AgentState, model: BaseChatModel) -> dict:
    """
    Extract sub-questions and keywords from user query.
    Makes query understanding visible to users before results appear.
    """
    system = SystemMessage(content=(
        "You are a Query Scoping Agent. Decompose the user's query into:\n"
        "1. 3–5 focused sub-questions that break down the research\n"
        "2. 4–8 key terms/themes for retrieval\n"
        "Return ONLY JSON:\n"
        '{"sub_questions": ["q1", "q2", ...], "keywords": ["k1", "k2", ...], "reasoning": "brief explain"}\n'
        "No other text or markdown."
    ))
    resp = model.invoke([system, HumanMessage(content=f"Query: {state['query']}")])
    raw = resp.content.strip().lstrip("```json").rstrip("```").strip()
    try:
        parsed = json.loads(raw)
        subs = parsed.get("sub_questions", [])[:5]  # Max 5
        keys = parsed.get("keywords", [])[:8]       # Max 8
        reason = parsed.get("reasoning", "")
    except Exception:
        subs = [state["query"]]
        keys = state["query"].split()[:5]
        reason = "scoping_failed"

    return {
        "sub_questions": subs,
        "keywords": keys,
        "scoping_reasoning": reason,
        "messages": [AIMessage(content=f"[Scoping] {len(subs)} sub-questions, {len(keys)} keywords")],
        "activity_log": [{
            "agent": "scoping",
            "icon": "🔍",
            "title": "Query Scoped",
            "detail": f"→ {len(subs)} sub-questions · {len(keys)} keywords",
            "ts": _stamp(),
        }],
        "current_agent": "scoping",
    }

# ╭─ BLOCK 2: CITATION GROUNDING VALIDATOR ──────────────────────────────────
def validate_citations(summary: str, merged_context: str, extraction_findings: str) -> tuple[dict, float]:
    """
    Validate that citations in summary actually appear in retrieved context.
    Returns (grounding_map, score) where score is 0.0-1.0.
    """
    grounding_map = {}

    # Extract potential citations from three patterns
    citation_patterns = [
        r'"([^"]{20,150})"',      # "quoted claims"
        r'\(([^)]{20,150})\)',    # (parenthetical claims)
    ]
    citations = []
    for pattern in citation_patterns:
        citations.extend(re.findall(pattern, summary))

    # Numeric citations [1], [2] — extract surrounding sentence as the claim
    for match in re.finditer(r'\[(\d+)\]', summary):
        idx        = match.start()
        sent_start = summary.rfind('.', 0, idx) + 1
        sent_end   = summary.find('.', idx)
        if sent_end == -1:
            sent_end = len(summary)
        sentence = summary[sent_start:sent_end].strip()[:150]
        if len(sentence.split()) >= 3:
            citations.append(sentence)

    citations      = list(set(citations))[:15]   # deduplicate, cap at 15
    context_text   = merged_context + "\n" + extraction_findings
    context_lower  = context_text.lower()
    grounded_count = 0

    for cit in citations:
        cit_lower = cit.lower()
        words     = cit_lower.split()
        if len(words) < 3:
            continue

        found_in_merged     = cit_lower in merged_context.lower()
        found_in_extraction = cit_lower in extraction_findings.lower()

        # Fallback: >60% content-word overlap
        if not (found_in_merged or found_in_extraction):
            content_words = [w for w in words if len(w) > 3]
            if content_words:
                matching = sum(1 for w in content_words if w in context_lower)
                if matching >= len(content_words) * 0.6:
                    found_in_merged = True

        is_grounded = found_in_merged or found_in_extraction
        if is_grounded:
            grounded_count += 1
            evidence = next(
                (line.strip()[:150] for line in context_text.split('\n')
                 if any(w in line.lower() for w in words[:3])),
                ""
            )
        else:
            evidence = ""

        grounding_map[cit[:100]] = {
            "grounded": is_grounded,
            "source":   "merged" if found_in_merged else (
                        "extraction" if found_in_extraction else "none"),
            "evidence": evidence,
        }

    score = grounded_count / len(citations) if citations else 1.0
    return grounding_map, score

# ╭─ CONFLICT DETECTION ────────────────────────────────────────────────────────

CREDIBILITY_TIERS = {
    "sql_db":   {"label": "peer-reviewed corpus", "score": 0.9},
    "vector_db": {"label": "indexed papers",       "score": 0.8},
    "web":      {"label": "web / preprints",       "score": 0.6},
    "merged":   {"label": "consensus",             "score": 0.7},
}

def conflict_agent(state: AgentState, model: BaseChatModel) -> dict:
    """
    Identify conflicts/disagreements between sources in the merged context.
    Uses LLM to read findings and spot contradictions.
    """
    if not state.get("merged_context") or len(state.get("merged_context", "")) < 100:
        return {
            "conflicts": [],
            "credibility_map": CREDIBILITY_TIERS,
            "messages": [AIMessage(content="[Conflict] No conflicts (insufficient context)")],
            "activity_log": [{
                "agent": "conflict_detector",
                "icon": "⚡",
                "title": "Conflict Detection",
                "detail": "Insufficient context to identify conflicts",
                "ts": _stamp(),
            }],
            "current_agent": "conflict_detector",
        }
    
    system = SystemMessage(content=(
        "You are a Conflict Detection Agent. Read the research findings and identify any direct "
        "contradictions or disagreements between sources.\n\n"
        "Examples of conflicts:\n"
        "- Source A claims 'X improves performance', Source B claims 'X has no effect'\n"
        "- Source A says 'method requires Y parameter', Source B says 'Y parameter is optional'\n"
        "- Source A: 'Dataset size is critical', Source B: 'Architecture matters more'\n\n"
        "Return ONLY valid JSON (no markdown, no code blocks):\n"
        '{"conflicts": [{"topic": "short phrase", "claim_a": "what source A says", '
        '"source_a": "vector_db|sql_db|web", "claim_b": "what source B says", '
        '"source_b": "vector_db|sql_db|web", "resolution": "which is more likely correct (one sentence)"}]}\n\n'
        "If no real contradictions exist, return {\"conflicts\": []}. "
        "Do NOT invent conflicts or over-interpret differences in terminology."
    ))
    
    resp = model.invoke([system, HumanMessage(content=(
        f"Research findings (first 4000 chars):\n\n{state['merged_context'][:4000]}"
    ))])
    
    raw = resp.content.strip().lstrip("```json").rstrip("```").strip()
    try:
        result = json.loads(raw)
        conflicts = result.get("conflicts", [])[:10]  # Max 10 conflicts
    except Exception:
        conflicts = []
    
    conflict_count = len(conflicts)
    
    return {
        "conflicts": conflicts,
        "credibility_map": CREDIBILITY_TIERS,
        "messages": [AIMessage(content=f"[Conflict] {conflict_count} conflicts identified")],
        "activity_log": [{
            "agent": "conflict_detector",
            "icon": "⚡",
            "title": f"Conflict Detection — {conflict_count} issues",
            "detail": f"Identified potential inconsistencies between sources",
            "ts": _stamp(),
        }],
        "current_agent": "conflict_detector",
    }

# ────────────────────────────────────────────────────────────────────────────────
def router_agent(state: AgentState, model: BaseChatModel) -> dict:
    # ─ GAP 4 FIX: Include critic feedback if looping (enables intelligent retry)
    critique_context = ""
    if state.get("_needs_more"):
        critique_context = f"\n\nPREVIOUS ATTEMPT FEEDBACK:\n{state.get('critique', 'Insufficient coverage detected.')}\nAdjust source selection strategy to address gaps."
    
    # ─ GAP 1 FIX: Include scoping keywords to guide routing
    keywords_str = ", ".join(state.get("keywords", [])[:8]) or "(none extracted)"
    sub_questions_count = len(state.get("sub_questions", []))
    
    system = SystemMessage(content=(
        "You are a Router Agent. Given a user query and refined scoping context, decide which search agents to activate.\n"
        "Available: 'vector_db' (semantic doc search), 'sql_db' (structured facts/topics), "
        "'web' (live scholarly search — use when query needs recent papers or external knowledge).\n"
        "Routing examples:\n"
        "- Query: 'What are the main approaches and challenges in federated learning for healthcare?' "
        "-> include 'sql_db' for categorical/structured coverage.\n"
        "- Query: 'What collaboration mechanisms are used in multi-agent LLM systems?' "
        "-> include 'sql_db' for mechanism enumeration.\n"
        "Return ONLY JSON: {\"agents\": [...], \"reasoning\": \"one sentence\"}. No other text."
    ))
    
    query_context = f"""Query: {state['query']}

Scoping Context:
- Key themes/keywords to prioritize: {keywords_str}
- Research angles: {sub_questions_count} sub-questions identified

Decide which sources would best address these angles.{critique_context}"""
    
    resp = model.invoke([system, HumanMessage(content=query_context)])
    raw  = resp.content.strip().lstrip("```json").rstrip("```").strip()
    try:
        parsed = json.loads(raw)
        agents = parsed.get("agents", ["vector_db", "sql_db"])
        reason = parsed.get("reasoning", "")
    except Exception:
        agents = ["vector_db", "sql_db"]
        reason = "defaulted"

    # Hard guardrail for structured/categorical prompts the LLM may miss.
    if _needs_sql_for_query(state.get("query", "")) and "sql_db" not in agents:
        agents.append("sql_db")
        if reason:
            reason = f"{reason}; sql_db forced by SQL trigger pattern"
        else:
            reason = "sql_db forced by SQL trigger pattern"

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
# ══════════════════════════════════════════════════════════════════════════════
# MINIMAL WEB SEARCH + RAG HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """Very rough token estimate without extra dependencies."""
    if not text:
        return 0
    return math.ceil(len(text.split()) * 1.3)

def web_search(query: str, limit: int = 5) -> list[dict]:
    """
    Minimal web search using DuckDuckGo HTML.
    Returns a list of dicts: title, url, snippet, content.
    """
    try:
        q = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={q}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html_content = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return [{"title": "Search error", "url": "", "snippet": str(e), "content": str(e)}]

    # crude but minimal parsing
    results = []
    blocks = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="(.*?)"[^>]*>(.*?)</a>.*?(?:<a[^>]*class="result__snippet"|<div[^>]*class="result__snippet")[^>]*>(.*?)</',
        html_content,
        flags=re.S
    )

    for href, title_html, snippet_html in blocks[:limit]:
        title = re.sub(r"<.*?>", "", html.unescape(title_html)).strip()
        snippet = re.sub(r"<.*?>", "", html.unescape(snippet_html)).strip()
        href = html.unescape(href)

        # DuckDuckGo wrapped links often contain uddg=
        parsed = urllib.parse.urlparse(href)
        qs = urllib.parse.parse_qs(parsed.query)
        clean_url = qs.get("uddg", [href])[0]

        content = f"Title: {title}\nURL: {clean_url}\nSnippet: {snippet}"
        results.append({
            "title": title,
            "url": clean_url,
            "snippet": snippet,
            "content": content,
        })

    if not results:
        return [{
            "title": "No results",
            "url": "",
            "snippet": "No search results parsed.",
            "content": "No search results parsed."
        }]

    return results

def add_web_results_to_rag(vdb: VectorDBModule, query: str, results: list[dict]) -> int:
    """Index web results into the same FAISS store used by your app."""
    total_chunks = 0
    for i, r in enumerate(results, start=1):
        text = (
            f"Web Search Query: {query}\n"
            f"Rank: {i}\n"
            f"Title: {r.get('title','')}\n"
            f"URL: {r.get('url','')}\n"
            f"Snippet: {r.get('snippet','')}\n"
        )
        total_chunks += vdb.add_text(
            text,
            {
                "source": "web_search",
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "query": query,
                "indexed_at": _stamp(),
            },
        )
    return total_chunks

def handle_vectordb_search_tool(vdb: VectorDBModule, tool_args: dict) -> str:
    """
    Handle VectorDB search tool calls from LLM.
    
    Args:
        vdb: VectorDBModule instance
        tool_args: dict with keys: query, top_k (optional), filter_source (optional)
    
    Returns:
        JSON string with search results or error
    """
    try:
        query = tool_args.get("query", "").strip()
        top_k = tool_args.get("top_k", 5)
        filter_source = tool_args.get("filter_source")
        
        if not query:
            return json.dumps({"error": "Empty query", "results": []})
        
        # Execute search
        docs = vdb.search(query, k=max(1, min(top_k, 20)))  # Clamp k to 1-20
        
        # Filter by source if specified
        if filter_source and docs:
            docs = [d for d in docs if d.metadata.get("source") == filter_source]
        
        # Format results
        results = []
        for doc in docs:
            results.append({
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content[:500],  # First 500 chars
                "metadata": {
                    "title": doc.metadata.get("title", ""),
                    "url": doc.metadata.get("url", ""),
                    "indexed_at": doc.metadata.get("indexed_at", ""),
                }
            })
        
        return json.dumps({
            "query": query,
            "returned": len(results),
            "filtered_source": filter_source,
            "results": results,
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "results": []
        })


def handle_sqldb_search_tool(tool_args: dict) -> str:
    """
    Handle SQL DB search tool calls from LLM.
    
    Args:
        tool_args: dict with keys: query, max_results (optional)
    
    Returns:
        JSON string with search results or error
    """
    try:
        query = tool_args.get("query", "").strip()
        max_results = tool_args.get("max_results", 10)
        
        if not query:
            return json.dumps({"error": "Empty query", "results": []})
        
        # Execute search
        raw_results = sql_search(query, k=max(1, min(max_results, 20)))
        
        # Parse results into structured format
        results = []
        for line in raw_results.split("\n"):
            if line.strip():
                results.append({"type": "text", "content": line})
        
        return json.dumps({
            "query": query,
            "returned": len(results),
            "results": results,
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "results": []
        })

def build_rag_context(vdb: VectorDBModule, query: str, k: int = 5) -> str:
    docs = vdb.search(query, k=k)
    if not docs:
        return "(no RAG context found)"
    return "\n\n---\n".join(
        f"[source={d.metadata.get('source','?')}] {d.page_content}"
        for d in docs
    )

def answer_with_rag(llm_or_cfg: ProviderConfig | BaseChatModel, query: str, rag_context: str) -> str:
    model = llm_or_cfg if isinstance(llm_or_cfg, BaseChatModel) else _llm(llm_or_cfg, temperature=0.2)
    msgs = [
        SystemMessage(content=(
            "Answer the user's question using the provided RAG context when relevant. "
            "If the context is insufficient, say so briefly and then answer as best you can."
        )),
        HumanMessage(content=f"Question:\n{query}\n\nRAG Context:\n{rag_context}")
    ]
    return model.invoke(msgs).content

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

    # Build scoping context once — used by BOTH paths
    keywords_str = ", ".join(state.get("keywords", [])) or "(none)"
    sub_q_str    = "\n  ".join(state.get("sub_questions", [])[:3]) or "(none)"

    # ── FALLBACK PATH (non-OpenAI models) ────────────────────────────────────
    if not isinstance(model, ChatOpenAI):
        docs = vdb.search(state["query"], k=5)
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
            "structured research notes relevant to the query and scoping keywords."
        ))
        resp = model.invoke([system, HumanMessage(content=(
            f"Query: {state['query']}\n"
            f"Focus keywords: {keywords_str}\n"
            f"Research angles:\n  {sub_q_str}\n\n"
            f"Chunks:\n{raw_ctx}"
        ))])

        return {
            "vector_findings": resp.content,
            "messages":        [AIMessage(content=f"[VectorDB] {resp.content[:120]}…")],
            "activity_log":    [{
                "agent": "vector_db", "icon": "🗂️",
                "title": "Vector DB agent (direct search)",
                "detail": f"Retrieved {len(docs)} chunks from {len(sources)} source(s)",
                "ts": _stamp(),
            }],
            "current_agent": "vector_db",
        }

    # ── TOOL-CALLING PATH (ChatOpenAI) ────────────────────────────────────────
    # FIX: keywords and sub-questions now injected here too
    system_prompt = SystemMessage(content=(
        "You are a Vector DB Search Agent. You have access to a database of indexed documents, "
        "papers, and web search results. Based on the user's query and scoping context, decide:\n"
        "1. Whether searching the Vector DB would be helpful\n"
        "2. What search query to use (prioritize scoped keywords if present)\n"
        "3. How many results to retrieve (1-20)\n"
        "4. Whether to filter by a specific source type (web_search, arxiv, conference, etc.)\n\n"
        "Use the search_vectordb tool if you think the Vector DB has relevant information. "
        "Otherwise, respond explaining why a search is not needed."
    ))

    # FIX: was using raw state["query"] only — now includes scoping context
    query_message = HumanMessage(content=(
        f"User Query: {state['query']}\n\n"
        f"Scoping keywords (use to refine search): {keywords_str}\n\n"
        f"Research angles to address:\n  {sub_q_str}\n\n"
        f"Decide whether and how to search Vector DB for relevant documents."
    ))

    try:
        response = model.invoke(
            [system_prompt, query_message],
            tools=[SEARCH_VECTORDB_TOOL],
            tool_choice="auto"
        )

        if hasattr(response, 'tool_calls') and response.tool_calls:
            docs_found     = []
            tool_reasoning = []

            for tool_call in response.tool_calls:
                if tool_call.function.name == "search_vectordb":
                    try:
                        tool_args   = json.loads(tool_call.function.arguments)
                        tool_result = handle_vectordb_search_tool(vdb, tool_args)
                        result_data = json.loads(tool_result)

                        if result_data.get("results"):
                            docs_found.extend(result_data["results"])
                            tool_reasoning.append(
                                f"Searched '{result_data.get('query')}' "
                                f"→ {result_data.get('returned')} results"
                            )
                    except Exception as e:
                        tool_reasoning.append(f"Tool error: {e}")

            if docs_found:
                formatted_docs = "\n\n---\n".join(
                    f"[{d['source']}] {d['content']}" for d in docs_found[:6]
                )
                synthesis_resp = model.invoke([
                    SystemMessage(content=(
                        "Synthesise the retrieved VectorDB documents into structured research notes. "
                        "Preserve source information and organise findings clearly."
                    )),
                    HumanMessage(content=(
                        f"Query: {state['query']}\n"
                        f"Focus keywords: {keywords_str}\n\n"
                        f"Documents:\n{formatted_docs}"
                    ))
                ])
                vector_findings = synthesis_resp.content
            else:
                vector_findings = "(No relevant documents found in Vector DB)"

            sources = list({d['source'] for d in docs_found})

            return {
                "vector_findings": vector_findings,
                "messages":        [AIMessage(content=f"[VectorDB] {len(sources)} source(s), {len(docs_found)} docs")],
                "activity_log":    [{
                    "agent": "vector_db", "icon": "🗂️",
                    "title": "Vector DB agent (tool-driven)",
                    "detail": f"{'; '.join(tool_reasoning) or 'no results'}",
                    "docs_found": len(docs_found),
                    "sources": sources,
                    "ts": _stamp(),
                }],
                "current_agent": "vector_db",
            }

        else:
            # LLM decided not to search
            llm_decision = getattr(response, 'content', str(response))
            return {
                "vector_findings": f"(Vector DB search not needed: {llm_decision[:200]})",
                "messages":        [AIMessage(content="[VectorDB] Skipped (LLM decision)")],
                "activity_log":    [{
                    "agent": "vector_db", "icon": "🗂️",
                    "title": "Vector DB agent (LLM decision)",
                    "detail": f"No search needed: {llm_decision[:100]}",
                    "ts": _stamp(),
                }],
                "current_agent": "vector_db",
            }

    except Exception as e:
        return {
            "vector_findings": f"(Vector DB error: {str(e)[:100]})",
            "messages":        [AIMessage(content=f"[VectorDB] Error: {str(e)[:50]}")],
            "activity_log":    [{
                "agent": "vector_db", "icon": "🗂️",
                "title": "Vector DB agent (error)",
                "detail": f"Tool execution failed: {str(e)[:100]}",
                "ts": _stamp(),
            }],
            "current_agent": "vector_db",
        }

def sql_db_agent(state: AgentState, model: BaseChatModel) -> dict:
    """
    SQL Database Search Agent with LLM-driven decision making.
    
    Uses tool calling to allow LLM to decide:
    - Whether to query the SQL database
    - What query to execute
    - How many results to retrieve
    """
    if "sql_db" not in state.get("active_agents", []):
        return {
            "sql_findings": "(not activated)",
            "messages":     [AIMessage(content="[SQLDB] skipped")],
            "activity_log": [{
              "agent": "sql_db", "icon": "🗄️",
              "title": "SQL DB — skipped",
              "detail": "Router did not activate.", 
              "ts": _stamp()}],
            "current_agent": "sql_db",
        }

    # Only use tool calling for ChatOpenAI models
    if not isinstance(model, ChatOpenAI):
        # Fallback: direct search for non-OpenAI models
        raw = sql_search(state["query"], k=8)
        rows = [l for l in raw.split("\n") if l.strip()]
        # ─ GAP 1: Include keywords for context
        keywords_str = ", ".join(state.get("keywords", [])) or "(general)"
        system = SystemMessage(content=(
            "You are a SQL Database Agent. Extract the most relevant structured information "
            "from the SQL query results, prioritizing content related to scoping keywords."
        ))
        resp = model.invoke([system, HumanMessage(content=f"Query: {state['query']}\nFocus keywords: {keywords_str}\n\nSQL results:\n{raw}")])
        
        return {
            "sql_findings": resp.content,
            "messages": [AIMessage(content=f"[SQLDB] {resp.content[:120]}…")],
            "activity_log": [{
                "agent": "sql_db",
                "icon": "🗄️",
                "title": "SQL / DB agent (direct search)",
                "detail": f"{len(rows)} result(s) found",
                "ts": _stamp(),
            }],
            "current_agent": "sql_db",
        }

    # LLM-driven search with tool calling (for ChatOpenAI)
    system_prompt = SystemMessage(content=(
        "You are a SQL Database Agent. You have access to a database of structured topics, "
        "relationships, and facts. Based on the user's query and scoping keywords, decide:\n"
        "1. Whether querying the SQL database would be helpful\n"
        "2. What search query to use (prioritize scoping keywords if present)\n"
        "3. How many results to retrieve (1-20)\n\n"
        "Use the search_sqldb tool if you think the SQL database has relevant structured information. "
        "Otherwise, respond explaining why a SQL search is not needed."
    ))
    
    # ─ GAP 1 FIX: Inject scoping keywords to refine query
    keywords_str = ", ".join(state.get("keywords", [])) or "(none)"
    sub_q_str = "\n  ".join(state.get("sub_questions", [])[:3]) or "(none)"
    
    query_context = f"""User Query: {state['query']}

Scoping Keywords (prioritize in search): {keywords_str}

Research angles to address:
  {sub_q_str}

Decide whether and how to search the SQL database for structured facts."""
    
    messages = [
        system_prompt,
        HumanMessage(content=query_context)
    ]
    
    try:
        response = model.invoke(
            messages,
            tools=[SEARCH_SQLDB_TOOL],
            tool_choice="auto"
        )
        
        # Check if LLM called the tool
        if hasattr(response, 'tool_calls') and response.tool_calls:
            results_found = []
            tool_reasoning = []
            
            for tool_call in response.tool_calls:
                if tool_call.function.name == "search_sqldb":
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_result = handle_sqldb_search_tool(tool_args)
                        result_data = json.loads(tool_result)
                        
                        if result_data.get("results"):
                            results_found.extend(result_data["results"])
                            tool_reasoning.append(
                                f"Queried for '{result_data.get('query')}' "
                                f"→ Found {result_data.get('returned')} results"
                            )
                    except Exception as e:
                        tool_reasoning.append(f"Tool error: {e}")
            
            # Synthesize findings from tool results
            if results_found:
                formatted_results = "\n".join([r['content'] for r in results_found])
                synthesis_system = SystemMessage(content=(
                    "Synthesise the SQL database results into structured research notes. "
                    "Preserve the topic, relationship, and fact distinctions."
                ))
                synthesis_resp = model.invoke([
                    synthesis_system,
                    HumanMessage(content=f"Query: {state['query']}\n\nDatabase results:\n{formatted_results}")
                ])
                sql_findings = synthesis_resp.content
            else:
                sql_findings = "(No relevant records found in SQL database)"
            
            return {
                "sql_findings": sql_findings,
                "messages": [AIMessage(content=f"[SQLDB] Searched {len(results_found)} result(s)")],
                "activity_log": [{
                    "agent": "sql_db",
                    "icon": "🗄️",
                    "title": "SQL / DB agent (tool-driven)",
                    "detail": f"LLM searched SQL: {'; '.join(tool_reasoning) or 'no results'}",
                    "rows": results_found[:12],
                    "ts": _stamp(),
                }],
                "current_agent": "sql_db",
            }
        else:
            # LLM decided NOT to use the tool
            llm_decision = response.content if hasattr(response, 'content') else str(response)
            return {
                "sql_findings": f"(SQL search not needed: {llm_decision[:200]})",
                "messages": [AIMessage(content=f"[SQLDB] Skipped search (LLM decision)")],
                "activity_log": [{
                    "agent": "sql_db",
                    "icon": "🗄️",
                    "title": "SQL / DB agent (LLM decision)",
                    "detail": f"LLM decided no SQL search needed: {llm_decision[:100]}",
                    "ts": _stamp(),
                }],
                "current_agent": "sql_db",
            }
    
    except Exception as e:
        return {
            "sql_findings": f"(SQL error: {str(e)[:100]})",
            "messages": [AIMessage(content=f"[SQLDB] Error: {str(e)[:50]}")],
            "activity_log": [{
                "agent": "sql_db",
                "icon": "🗄️",
                "title": "SQL / DB agent (error)",
                "detail": f"Tool execution failed: {str(e)[:100]}",
                "ts": _stamp(),
            }],
            "current_agent": "sql_db",
        }

# ── 4. Web / arXiv Agent ──────────────────────────────────────────────────────
# FIX 3: Web agent now queries OpenAlex + Crossref + Semantic Scholar + arXiv + Conference Papers
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
    errors       = []   # API failures tracked separately and may be surfaced later in fallback/reporting content
    indexed      = 0
    sources_used = []
    conference_papers = []

    # Try conference paper search if query mentions conferences
    if HAS_CONFERENCE_SEARCH:
        try:
            # Check if query mentions any conference keywords
            query_lower = query.lower()
            conference_keywords = {"neurips", "icml", "iclr", "acl", "emnlp", "naacl", "eacl", "conference", "paper", "arxiv"}
            mentions_conference = any(kw in query_lower for kw in conference_keywords)
            
            if mentions_conference and isinstance(model, ChatOpenAI):
                # Try to use the conference paper search tool
                system_prompt = SystemMessage(content=(
                    "You are a research assistant. If the user is looking for papers from ML/NLP conferences "
                    "(NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, EACL), you should use the search_conference_papers tool "
                    "to find the most relevant papers. Determine which conferences, years, and keywords are most relevant "
                    "from the user's query, then call the tool accordingly."
                ))
                
                messages = [
                    system_prompt,
                    HumanMessage(content=f"Search for papers related to: {query}")
                ]
                
                # Call the model with the tool
                response = model.invoke(
                    messages,
                    tools=[SEARCH_PAPERS_TOOL],
                    tool_choice="auto"
                )
                
                # Process tool calls if any
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    for tool_call in response.tool_calls:
                        if tool_call.function.name == "search_conference_papers":
                            try:
                                tool_args = json.loads(tool_call.function.arguments)
                                tool_result = handle_conference_paper_tool_call(tool_args)
                                result_data = json.loads(tool_result)
                                
                                if result_data.get("papers"):
                                    conference_papers = result_data["papers"]
                                    sources_used.append(f"Conference Papers ({len(conference_papers)})")
                                    
                                    # Format and add to results
                                    for paper in conference_papers[:5]:
                                        authors = paper.get("authors", "")
                                        title = paper.get("title", "")
                                        conference = paper.get("conference", "")
                                        year = paper.get("year", "")
                                        results_text.append(
                                            f"[{conference} {year}] {title}\n"
                                            f"Authors: {authors}\nURL: {paper.get('url', '')}"
                                        )
                                        
                                        # Index into vector DB
                                        vdb.add_text(
                                            f"Title: {title}\nConference: {conference}\nYear: {year}\n"
                                            f"Authors: {authors}\nAbstract: {paper.get('abstract', '')}",
                                            {"source": f"{conference}_{year}", "title": title, "url": paper.get("url", "")}
                                        )
                                        indexed += 1
                            except Exception as e:
                                results_text.append(f"[Conference Paper Search error] {e}")
        except Exception as e:
            pass  # Gracefully skip if something goes wrong

        # New Start Fetch all four sources concurrently ────────────────────────────────

        def _fetch_openalex():
            papers = []
            try:
                r = requests.get("https://api.openalex.org/works",
                                 params={"search": query, "per-page": 5,
                                         "mailto": "research@example.com"},
                                 timeout=10)
                if r.ok:
                    for item in r.json().get("results", []):
                        title = item.get("title", "No title")
                        year = item.get("publication_year", "")
                        authors = [a.get("author", {}).get("display_name", "")
                                   for a in item.get("authorships", [])[:3]]
                        papers.append({
                            "text": f"[OpenAlex] {title} ({year}) — {', '.join(authors)}",
                            "doc": (f"Title: {title}\nAuthors: {', '.join(authors)}\nYear: {year}",
                                    {"source": "openalex", "title": title}),
                            "source": "OpenAlex",
                        })
            except Exception as e:
                return [], f"OpenAlex: {e}"
            return papers, None

        def _fetch_crossref():
            papers = []
            try:
                r = requests.get("https://api.crossref.org/works",
                                 params={"query": query, "rows": 5,
                                         "mailto": "research@example.com"},
                                 timeout=10)
                if r.ok:
                    for item in r.json().get("message", {}).get("items", []):
                        title = (item.get("title") or ["No title"])[0]
                        doi = item.get("DOI", "")
                        authors = [f"{a.get('given', '')} {a.get('family', '')}".strip()
                                   for a in item.get("author", [])[:3]]
                        papers.append({
                            "text": f"[Crossref] {title} — {', '.join(authors)} | doi:{doi}",
                            "doc": (f"Title: {title}\nAuthors: {', '.join(authors)}\nDOI: {doi}",
                                    {"source": "crossref", "title": title}),
                            "source": "Crossref",
                        })
            except Exception as e:
                return [], f"Crossref: {e}"
            return papers, None

        def _fetch_semantic_scholar():
            papers = []
            try:
                ss_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
                headers = {"x-api-key": ss_key} if ss_key else {}
                r = requests.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={"query": query, "limit": 5,
                            "fields": "title,year,citationCount,authors"},
                    headers=headers, timeout=10,
                )
                if r.ok:
                    for paper in r.json().get("data", []):
                        title = paper.get("title", "No title")
                        year = paper.get("year", "")
                        cites = paper.get("citationCount", 0)
                        authors = [a.get("name", "") for a in (paper.get("authors") or [])[:3]]
                        papers.append({
                            "text": (f"[Semantic Scholar] {title} ({year}) — "
                                     f"{', '.join(authors)} — {cites} citations"),
                            "doc": (f"Title: {title}\nAuthors: {', '.join(authors)}\n"
                                    f"Year: {year}\nCitations: {cites}",
                                    {"source": "semantic_scholar", "title": title}),
                            "source": "Semantic Scholar",
                        })
            except Exception as e:
                return [], f"Semantic Scholar: {e}"
            return papers, None

        def _fetch_arxiv():
            papers = []
            try:
                encoded = urllib.parse.quote(query)
                url = (f"http://export.arxiv.org/api/query"
                       f"?search_query=all:{encoded}&start=0&max_results=5&sortBy=relevance")
                with urllib.request.urlopen(url, timeout=15) as resp:
                    xml = resp.read().decode("utf-8")
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                root = ET.fromstring(xml)
                for entry in root.findall("atom:entry", ns):
                    title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
                    summary = (entry.findtext("atom:summary", "", ns) or "").strip()[:400]
                    eid = (entry.findtext("atom:id", "", ns) or "").strip()
                    authors = [a.findtext("atom:name", "", ns)
                               for a in entry.findall("atom:author", ns)]
                    papers.append({
                        "text": f"[arXiv] {title} — {', '.join(authors[:3])}\n{eid}",
                        "doc": (f"Title: {title}\nAuthors: {', '.join(authors)}\n"
                                f"Abstract: {summary}",
                                {"source": "arXiv", "title": title,
                                 "url": eid, "indexed_at": _stamp()}),
                        "source": "arXiv",
                    })
            except Exception as e:
                return [], f"arXiv: {e}"
            return papers, None

        # Run all four of the fetches in parallel collecting results in the main thread
        # so that vdb.add_text (FAISS) is never called from multiple threads
        fetchers = [_fetch_openalex, _fetch_crossref,
                    _fetch_semantic_scholar, _fetch_arxiv]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fn): fn for fn in fetchers}
            for future in as_completed(futures):
                papers, error = future.result()
                if error:
                    errors.append(error)
                elif papers:
                    for p in papers:
                        results_text.append(p["text"])
                        doc_text, doc_meta = p["doc"]
                        vdb.add_text(doc_text, doc_meta)
                        indexed += 1
                    sources_used.append(papers[0]["source"])    ## new end

    # Build content for the LLM — errors are noted but never treated as findings.
    if results_text:
        combined = "\n\n---\n".join(results_text)
        if errors:
            combined += f"\n\n(Note: the following sources failed and returned no results: {'; '.join(errors)})"
    elif errors:
        combined = f"(All sources failed to return results. Errors: {'; '.join(errors)})"
    else:
        combined = "(no results)"
    system = SystemMessage(content=(
        "You are a Web Research Agent. Summarise the scholarly search results below into "
        "structured research notes relevant to the query. Cite the source for each finding "
        "(Conference Papers, OpenAlex, Crossref, Semantic Scholar, or arXiv)."
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


# ── 5. Reading / Extraction Agent ────────────────────────────────────────────

_EMPTY_PLACEHOLDER = {"", "(none)", "(not activated)", "(no SQL results)"}

def _has_content(s: str) -> bool:
    return s.strip() not in _EMPTY_PLACEHOLDER

def reading_extraction_agent(state: AgentState, model: BaseChatModel) -> dict:
    vf = state.get("vector_findings", "")
    sf = state.get("sql_findings", "")
    wf = state.get("web_findings", "")

    # Skip if all retrievers returned empty or placeholder content
    if not any(_has_content(f) for f in (vf, sf, wf)):
        return {
            "extraction_findings": "(none)",
            "messages": [AIMessage(content="[ReadingExtraction] skipped — no retrieval content")],
            "activity_log": [{
                "agent":  "reading_extraction",
                "icon":   "📖",
                "title":  "Reading / Extraction — skipped",
                "detail": "No content from any retrieval channel.",
                "ts":     _stamp(),
            }],
            "current_agent": "reading_extraction",
        }

    # Build combined input, filtering out placeholder values
    sections = []
    if _has_content(vf):
        sections.append(f"=== Vector DB findings ===\n{vf}")
    if _has_content(sf):
        sections.append(f"=== SQL / DB findings ===\n{sf}")
    if _has_content(wf):
        sections.append(f"=== Web / API findings ===\n{wf}")
    combined = "\n\n".join(sections)

    system = SystemMessage(content=(
        "You are a Reading and Extraction Agent for academic research.\n"
        "Given synthesised retrieval findings from multiple sources, extract a structured record "
        "for every distinct paper, study, or source you can identify.\n\n"
        "For each paper output a block in this exact format:\n"
        "---\n"
        "**Title / Topic:** <title or identifier>\n"
        "**Provenance:** <abstract-only | full-text | structured-db>\n"
        "**Research Problem:** <one sentence>\n"
        "**Methodology:** <one sentence>\n"
        "**Key Findings:**\n- <bullet points>\n"
        "**Limitations:** <one sentence>\n"
        "**Future Work:** <one sentence>\n"
        "---\n\n"
        "Provenance rules:\n"
        "- 'structured-db' if the source is from the SQL/DB findings section\n"
        "- 'full-text' if large paragraphs of text were available (VectorDB)\n"
        "- 'abstract-only' if only title/abstract/year was available (Web API results)\n\n"
        "If no distinct papers can be identified, output exactly: NO_PAPERS_EXTRACTED\n\n"
        "Be precise. Do not hallucinate citations."
    ))
    resp = model.invoke([system, HumanMessage(
        content=f"Query: {state['query']}\n\nRetrieval findings:\n{combined}"
    )])

    extraction = resp.content.strip()
    # Count records by structured markers: prefer the required "**Title / Topic:**" field,
    # and fall back to counting exact '---' delimiter lines if that is missing.
    title_marker = "**Title / Topic:**"
    paper_count = extraction.count(title_marker)
    if paper_count == 0:
        # Each record is expected to have a start and end '---' delimiter, so count
        # delimiter lines and infer the number of papers from delimiter *pairs*.
        delimiter_lines = sum(1 for line in extraction.splitlines() if line.strip() == "---")
        paper_count = max(1, delimiter_lines // 2) if delimiter_lines > 0 else 0

    return {
        "extraction_findings": extraction,
        "messages": [AIMessage(content=f"[ReadingExtraction] {paper_count} paper(s) structured")],
        "activity_log": [{
            "agent":  "reading_extraction",
            "icon":   "📖",
            "title":  "Reading / Extraction agent",
            "detail": (
                f"{paper_count} paper(s) extracted — fields: research problem, methodology, "
                f"findings, limitations, future work. Provenance tagged per source."
            ),
            "ts":     _stamp(),
        }],
        "current_agent": "reading_extraction",
    }


def orchestrator_agent(state: AgentState, model: BaseChatModel) -> dict:
    block = "\n\n".join([
        f"=== Vector DB ===\n{state.get('vector_findings','')}",
        f"=== SQL / DB ===\n{state.get('sql_findings','')}",
        f"=== Web / APIs ===\n{state.get('web_findings','')}",
        f"=== Structured Extraction ===\n{state.get('extraction_findings','')}",
    ])
    system = SystemMessage(content=(
        "You are an Orchestrator Agent. Merge findings from four specialised agents "
        "(VectorDB, SQL/DB, Web, and Extraction):\n"
        "1. Deduplicate overlapping information\n"
        "2. Resolve contradictions, preferring higher-confidence structured sources\n"
        "3. Label each claim: [VectorDB] / [SQL] / [Web] / [Extraction]\n"
        "4. Produce one coherent merged context for downstream agents."
    ))
    resp = model.invoke([system, HumanMessage(content=f"Query: {state['query']}\n\n{block}")])
    active = [
        src for src, key in [
            ("Vector DB",  "vector_findings"),
            ("SQL DB",     "sql_findings"),
            ("Web",        "web_findings"),
            ("Extraction", "extraction_findings"),
        ]
        if state.get(key, "") not in ("", "(none)", "NO_PAPERS_EXTRACTED", "(not activated)")
        and "not activated" not in state.get(key, "")
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
    # The Orchestrator labels each claim inline: [VectorDB], [SQL], [Web], [Extraction].
    # Pass that mapping to the LLM so nodes can be attributed to the correct source.
    system = SystemMessage(content=(
        "You are a Knowledge Mapping Agent. Extract a knowledge graph from the merged context.\n"
        "The context labels each claim with its origin: [VectorDB], [SQL], [Web], or [Extraction]. "
        "Use those labels to set each node's \"source\" field:\n"
        "  [VectorDB]   → \"vector_db\"\n"
        "  [SQL]        → \"sql_db\"\n"
        "  [Web]        → \"web\"\n"
        "  [Extraction] → \"merged\"\n"
        "  no label     → \"merged\"\n\n"
        "Return ONLY valid JSON (no markdown, no extra text):\n"
        '{"nodes": [{"id":"str","label":"str","type":"concept","source":"vector_db"}],'
        '"edges": [{"source":"str","target":"str","relation":"str","weight":0.5}]}\n'
        "\"type\" must be exactly one of: concept, entity, fact, process.\n"
        "\"source\" must be exactly one of: vector_db, sql_db, web, merged.\n"
        "Include 12–20 nodes."
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
    nodes = state['knowledge_map'].get('nodes', [])
    edges = state['knowledge_map'].get('edges', [])
    # Normalise to lowercase strings so "Concept" and "concept" don't count twice.
    types = {
        t.lower() for n in nodes
        if isinstance(n, dict)
        for t in [n.get('type', '')]
        if isinstance(t, str) and t
    }

    # All three criteria are structural facts — compute deterministically in Python
    # rather than asking an LLM, which adds latency/tokens and can misparse output.
    failures = []
    if len(nodes) < 8:
        failures.append(f"only {len(nodes)} nodes (need ≥ 8)")
    if len(edges) < 4:
        failures.append(f"only {len(edges)} edges (need ≥ 4)")
    if len(types) < 2:
        failures.append(f"only {len(types)} distinct node type(s) (need ≥ 2)")

    needs = bool(failures)
    feedback = "Needs enrichment: " + "; ".join(failures) if failures else "Graph meets structural requirements."

    next_loop_count = state.get("loop_count", 0)
    if needs:
        next_loop_count += 1

    return {
        "critique":    feedback,
        "_needs_more": needs,
        "loop_count":  next_loop_count,
        "messages":    [AIMessage(content=f"[Critic] needs_more={needs}")],
        "activity_log": [{
            "agent": "critic", "icon": "🧐",
            "title": f"Critic — {'needs enrichment' if needs else 'approved'}",
            "detail": feedback, "ts": _stamp(),
        }],
        "current_agent": "critic",
    }

# ── 8. Summarizer ─────────────────────────────────────────────────────────────
def summarizer_agent(state: AgentState, model: BaseChatModel) -> dict:
    system = SystemMessage(content=(
        "You are given a research query and multi-source findings.\n"
        "Write a response that:\n"
        "- Answers ONLY what the query asks -- do not add background the user didn't request\n"
        "- Is <=400 words unless the query explicitly requires exhaustive coverage\n"
        "- Uses prose paragraphs, not bullet lists\n"
        "- Identifies at least one tension or disagreement across sources\n"
        "- Supports each distinct empirical claim with a verbatim evidence quote from Findings\n"
        "- Uses citation format exactly: (evidence: \"<exact quote from Findings>\")\n"
        "- Keeps each evidence quote 8-30 words and copied exactly from Findings\n"
        "- Never cites channels/agents/labels as evidence\n"
        "\n"
        "Forbidden citation styles include: (source: Web), (source: Extraction), (Vector DB), (SQL DB)."
    ))
    # Prefer synthesis_report over raw merged_context when available
    synthesis = state.get("synthesis_report", "")
    context_src = synthesis if synthesis and synthesis != "(no content to synthesise)" else state.get("merged_context", "")
    key_concepts = [n["label"] for n in state["knowledge_map"].get("nodes", [])]
    resp = model.invoke([system, HumanMessage(content=(
        f"Query: {state['query']}\n\n"
        f"Findings: {context_src}\n\n"
        f"Key concepts: {key_concepts}"
    ))])
    
    # ── BLOCK 2: Validate citations against retrieved sources ─────────────────────
    citation_grounding, grounding_score = validate_citations(
        resp.content,
        state.get("merged_context", ""),
        state.get("extraction_findings", ""),
    )
    
    grounded_count = sum(1 for v in citation_grounding.values() if v.get("grounded"))
    total_citations = len(citation_grounding)
    
    return {
        "summary":             resp.content,
        "citation_grounding":  citation_grounding,
        "grounding_score":     grounding_score,
        "messages":            [AIMessage(content=f"[Summarizer] {resp.content[:120]}…")],
        "activity_log":        [{
            "agent": "summarizer", "icon": "✍️", "title": "Summarizer — final answer",
            "detail": f"{len(resp.content)} chars · {grounded_count}/{total_citations} citations grounded ({int(grounding_score*100)}%)",
            "ts": _stamp(),
        }],
        "current_agent": "summarizer",
    }


# ── 9. Experiment Design ──────────────────────────────────────────────────────
def experiment_design_agent(state: AgentState, model: BaseChatModel) -> dict:
    system = SystemMessage(content=(
        "You are a Planning Agent in a multi-agent research assistant system. "
        "Your role is to translate a completed literature analysis into an actionable, "
        "citation-grounded structured research plan.\n\n"
        "You have access to: (1) the user's research question, (2) a synthesized literature "
        "summary, (3) per-paper extraction records containing each paper's research problem, "
        "methods, findings, limitations, and stated future work, and (4) a merged evidence "
        "context from all retrieval channels.\n\n"
        "Produce a structured research plan in markdown with EXACTLY these sections, in order:\n\n"
        "## Research Landscape Overview\n"
        "One concise paragraph: what is well-established, which methods dominate, and where "
        "the field currently stands relative to the research question.\n\n"
        "## Identified Research Gaps\n"
        "List 3–5 specific, concrete gaps derived from reported limitations and future-work "
        "statements in the literature. Each gap must cite the paper(s) that reveal it. "
        "Format each as:\n"
        "**Gap N: <short title>** — <description of what is missing or unresolved>  \n"
        "*Grounded in: <paper title / author shorthand>*\n\n"
        "## Proposed Hypotheses\n"
        "One falsifiable, testable hypothesis per gap. Each must be explicitly grounded in the "
        "evidence and directly address its corresponding gap. Format each as:\n"
        "**H-N** *(addresses Gap N)*: <hypothesis statement>\n\n"
        "## Recommended Methodologies\n"
        "For each hypothesis, specify: study design, experimental protocol, key procedures, "
        "and evaluation approach. Where the literature already validates a method, reference it "
        "by name and source. Note which methodologies are novel vs. established.\n\n"
        "## Datasets & Domains\n"
        "For each hypothesis, identify: concrete public datasets or benchmarks (with names), "
        "data collection approaches if no public dataset exists, domain scope and inclusion "
        "criteria, and approximate scale needed. Reference datasets already used in the "
        "literature where applicable.\n\n"
        "## Anticipated Challenges & Risks\n"
        "For each major risk (technical, logistical, or validity-related), briefly describe "
        "the risk and a concrete mitigation strategy. Include risks around reproducibility, "
        "data access, computational cost, and evaluation validity.\n\n"
        "## Short-term Next Steps (0–3 months)\n"
        "A numbered list of immediate, concrete actions a researcher could begin today. "
        "Be specific: name tools, datasets, baselines, or collaborators where relevant.\n\n"
        "## Medium-term Next Steps (3–12 months)\n"
        "Milestones that build on short-term work toward full experimental execution and "
        "publication. Include checkpoints for evaluating progress.\n\n"
        "IMPORTANT: Every claim must be grounded in the provided evidence. "
        "Do not fabricate paper titles, authors, or dataset names. "
        "If evidence is sparse for a section, say so explicitly rather than hallucinating."
    ))

    extraction = state.get("extraction_findings", "") or ""
    context    = state.get("merged_context", "") or ""
    km_nodes   = [n.get("label", "") for n in state.get("knowledge_map", {}).get("nodes", [])]

    resp = model.invoke([system, HumanMessage(content=(
        f"Research question: {state['query']}\n\n"
        f"Synthesized literature summary:\n{state['summary']}\n\n"
        f"Per-paper extraction records (problems · methods · findings · limitations · future work):\n"
        f"{extraction[:2500]}\n\n"
        f"Merged evidence context:\n{context[:2000]}\n\n"
        f"Key concepts from knowledge graph: {', '.join(km_nodes[:30])}"
    ))])

    plan = resp.content
    n_gaps  = plan.count("**Gap ")
    n_hyp   = plan.count("**H-")
    n_steps = plan.count("## Short-term") + plan.count("## Medium-term")

    return {
        "experiment_plan": plan,
        "messages":        [AIMessage(content=f"[ExperimentDesign] {plan[:120]}…")],
        "activity_log":    [{
            "agent":  "experiment_design",
            "icon":   "🧪",
            "title":  "Planning Agent — structured research plan generated",
            "detail": (f"{n_gaps} gaps identified · {n_hyp} hypotheses · "
                       f"{n_steps} step horizons · {len(plan)} characters"),
            "ts":     _stamp(),
        }],
        "current_agent": "experiment_design",
    }


# ╭─ MODULE DELEGATION ADAPTERS ───────────────────────────────────────────────
def _scoping_agent_delegate(state: AgentState, model: BaseChatModel) -> dict:
    return _mod_scoping_agent(state, model, stamp_fn=_stamp)


def _router_agent_delegate(state: AgentState, model: BaseChatModel) -> dict:
    return _mod_router_agent(state, model, stamp_fn=_stamp)


def _vector_db_agent_delegate(state: AgentState, model: BaseChatModel, vdb: VectorDBModule) -> dict:
    return _mod_vector_db_agent(
        state,
        model,
        vdb,
        search_tool=SEARCH_VECTORDB_TOOL,
        handle_tool_fn=handle_vectordb_search_tool,
        stamp_fn=_stamp,
    )


def _sql_db_agent_delegate(state: AgentState, model: BaseChatModel) -> dict:
    return _mod_sql_db_agent(
        state,
        model,
        sql_search_fn=sql_search,
        search_tool=SEARCH_SQLDB_TOOL,
        handle_tool_fn=handle_sqldb_search_tool,
        stamp_fn=_stamp,
    )


def _web_agent_delegate(state: AgentState, model: BaseChatModel, vdb: VectorDBModule) -> dict:
    return _mod_web_agent(state, model, vdb, stamp_fn=_stamp)


def _reading_extraction_agent_delegate(state: AgentState, model: BaseChatModel) -> dict:
    return _mod_reading_extraction_agent(state, model, stamp_fn=_stamp)


def _orchestrator_agent_delegate(state: AgentState, model: BaseChatModel) -> dict:
    return _mod_orchestrator_agent(state, model, stamp_fn=_stamp)


def _conflict_agent_delegate(state: AgentState, model: BaseChatModel) -> dict:
    return _mod_conflict_agent(state, model, stamp_fn=_stamp)


def _knowledge_mapper_agent_delegate(state: AgentState, model: BaseChatModel) -> dict:
    return _mod_knowledge_mapper_agent(state, model, stamp_fn=_stamp)


def _critic_agent_delegate(state: AgentState, model: BaseChatModel) -> dict:
    return _mod_critic_agent(state, model, stamp_fn=_stamp)


def _summarizer_agent_delegate(state: AgentState, model: BaseChatModel) -> dict:
    return _mod_summarizer_agent(state, model, stamp_fn=_stamp)


def _experiment_design_agent_delegate(state: AgentState, model: BaseChatModel) -> dict:
    return _mod_experiment_design_agent(state, model, stamp_fn=_stamp)


# Public callable names used across graph/tests now point to module implementations.
scoping_agent = _scoping_agent_delegate
router_agent = _router_agent_delegate
vector_db_agent = _vector_db_agent_delegate
sql_db_agent = _sql_db_agent_delegate
web_agent = _web_agent_delegate
reading_extraction_agent = _reading_extraction_agent_delegate
orchestrator_agent = _orchestrator_agent_delegate
conflict_agent = _conflict_agent_delegate
knowledge_mapper_agent = _knowledge_mapper_agent_delegate
critic_agent = _critic_agent_delegate
summarizer_agent = _summarizer_agent_delegate
experiment_design_agent = _experiment_design_agent_delegate


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_to_all_retrievers(state: AgentState) -> list[str]:
    """
    Router → Parallel retrieval tier (fan-out to all active retrievers).
    
    Returns list of active retriever nodes.
    If no retrievers are active, routes directly to extraction.
    Previously cascaded (vector_db → sql_db → web) sequentially.
    Now all active retrievers run in parallel, then fan-in to extraction.
    """
    active = state.get("active_agents", [])
    
    # Collect all active retriever nodes
    targets = []
    if "vector_db" in active:
        targets.append("vector_db")
    if "sql_db" in active:
        targets.append("sql_db")
    if "web" in active:
        targets.append("web")
    
    # If no retrievers are active, skip directly to extraction
    if not targets:
        return ["reading_extraction"]
    
    return targets


def _route_critic(state: AgentState) -> str:
    """
    Critic loop routing (FIXED: Block 2 Gap).
    
    OLD (broken): Looped to orchestrator, which re-merged same findings.
    NEW (correct): Loops to router with critique feedback, triggering fresh retrieval.
    
    This ensures the second pass actually retrieves new or differently-prioritized content.
    The critique is passed to router to adjust active_agents strategy.
    """
    if state.get("_needs_more") and state.get("loop_count", 0) < 2:
        return "router"
    return "summarizer"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(cfg: ProviderConfig, vdb: VectorDBModule):
    lm_sc = _llm(cfg, 0.2)   # scoping
    lm_r  = _llm(cfg, 0.0)   # router (deterministic)
    lm_s  = _llm(cfg, 0.3)   # retrieval agents
    lm_e  = _llm(cfg, 0.1)   # extraction
    lm_o  = _llm(cfg, 0.2)   # orchestrator
    lm_cf = _llm(cfg, 0.0)   # conflict detector (deterministic)
    lm_m  = _llm(cfg, 0.1)   # knowledge mapper
    lm_c  = _llm(cfg, 0.0)   # critic (fully deterministic)
    lm_z  = _llm(cfg, 0.5)   # summarizer
    lm_x  = _llm(cfg, 0.4)   # experiment design

    g = StateGraph(AgentState)

    # ── Register all nodes ────────────────────────────────────────────────────
    g.add_node("scoping",            lambda s: scoping_agent(s, lm_sc))
    g.add_node("router",             lambda s: router_agent(s, lm_r))
    g.add_node("vector_db",          lambda s: vector_db_agent(s, lm_s, vdb))
    g.add_node("sql_db",             lambda s: sql_db_agent(s, lm_s))
    g.add_node("web",                lambda s: web_agent(s, lm_s, vdb))
    g.add_node("reading_extraction", lambda s: reading_extraction_agent(s, lm_e))
    g.add_node("orchestrator",       lambda s: orchestrator_agent(s, lm_o))
    g.add_node("conflict_detector",  lambda s: conflict_agent(s, lm_cf))   # FIX (c): wired in
    g.add_node("knowledge_mapper",   lambda s: knowledge_mapper_agent(s, lm_m))
    g.add_node("critic",             lambda s: critic_agent(s, lm_c))
    g.add_node("summarizer",         lambda s: summarizer_agent(s, lm_z))
    g.add_node("experiment_design",  lambda s: experiment_design_agent(s, lm_x))

    # ── Entry point ───────────────────────────────────────────────────────────
    g.set_entry_point("scoping")
    g.add_edge("scoping", "router")

    # ── FIX (c): Parallel fan-out from router ─────────────────────────────────
    # LangGraph fan-out: add one unconditional edge per retriever target.
    # All three fire simultaneously; LangGraph waits for all before running
    # reading_extraction (fan-in is automatic when a node has multiple incoming edges).
    #
    # Agents that are not activated skip themselves internally (they check
    # state["active_agents"] and return immediately if not listed) — so it is
    # safe to always fan out to all three regardless of router decision.
    g.add_edge("router", "vector_db")
    g.add_edge("router", "sql_db")
    g.add_edge("router", "web")

    # ── Fan-in: all retrievers → extraction ───────────────────────────────────
    g.add_edge("vector_db",          "reading_extraction")
    g.add_edge("sql_db",             "reading_extraction")
    g.add_edge("web",                "reading_extraction")

    # ── Main pipeline ─────────────────────────────────────────────────────────
    g.add_edge("reading_extraction", "orchestrator")
    g.add_edge("orchestrator",       "conflict_detector")   # FIX (b): conflict slot
    g.add_edge("conflict_detector",  "knowledge_mapper")
    g.add_edge("knowledge_mapper",   "critic")

    # ── Critic loop: routes to router (fresh retrieval) or summarizer ─────────
    g.add_conditional_edges(
        "critic",
        _route_critic,
        {"router": "router", "summarizer": "summarizer"},
    )

    # ── Final pipeline ────────────────────────────────────────────────────────
    # FIX (a): removed g.add_edge("synthesis", "summarizer") — node never existed
    g.add_edge("summarizer",         "experiment_design")
    g.add_edge("experiment_design",  END)

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
        tmp_path = Path(f.name)
    try:
        net.save_graph(str(tmp_path))
        html = tmp_path.read_text()
    finally:
        tmp_path.unlink(missing_ok=True)
    return html


def _html_to_data_uri(content: str) -> str:
    """Embed inline HTML using a data URI for st.iframe."""
    return "data:text/html;charset=utf-8," + urllib.parse.quote(content)



for _d in [VECTOR_DIR, DOCS_DIR, MAPS_DIR, CACHE_DIR, SESSIONS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

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
.agent-card.scoping{border-color:#FF6B6B}
.agent-card.router{border-color:#9B59B6}
.agent-card.vector_db{border-color:#4A90D9}
.agent-card.sql_db{border-color:#E67E22}
.agent-card.web{border-color:#2ECC71}
.agent-card.reading_extraction{border-color:#27AE60}
.agent-card.orchestrator{border-color:#E74C3C}
.agent-card.knowledge_mapper{border-color:#1ABC9C}
.agent-card.critic{border-color:#F39C12}
.agent-card.summarizer{border-color:#95A5A6}
.agent-card.experiment_design{border-color:#00BCD4}
.agent-title{font-weight:600;font-size:0.88rem;margin-bottom:3px}
.agent-detail{font-size:0.78rem;color:#8b949e}
.src-badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.68rem;font-family:'JetBrains Mono',monospace;letter-spacing:.04em;margin:0 3px}
.bv{background:#1a3a5c;color:#4A90D9}
.bs{background:#4a2010;color:#E67E22}
.bw{background:#0f3d20;color:#2ECC71}
.be{background:#0d2e1a;color:#27AE60}
.bm{background:#2d1a4a;color:#9B59B6}
.bx{background:#002d33;color:#00BCD4}
</style>
""", unsafe_allow_html=True)

# ── Init ──────────────────────────────────────────────────────────────────────
init_sql_db()

if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now(timezone.utc).strftime("sess_%Y%m%d_%H%M%S")
if "turns" not in st.session_state:
    st.session_state.turns = []
if "vdb" not in st.session_state:
    st.session_state.vdb = None
if "web_results" not in st.session_state:
    st.session_state.web_results = []

if "web_results_query" not in st.session_state:
    st.session_state.web_results_query = ""

if "rag_answer" not in st.session_state:
    st.session_state.rag_answer = ""

if "rag_context_last" not in st.session_state:
    st.session_state.rag_context_last = ""

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
        key="provider_api_key",
    )

    default_model = _DEFAULT_MODELS.get(provider, "")

    base_url = None
    if provider == PROVIDER_LOCAL:
        base_url = st.text_input("Base URL", value=_DEFAULT_BASE_URL,
                                 placeholder=_DEFAULT_BASE_URL)
        _local_models = _fetch_local_models(base_url or _DEFAULT_BASE_URL)
        if _local_models:
            llm_model = st.selectbox("Model", options=_local_models)
        else:
            st.warning("Could not reach local server — enter model name manually.")
            llm_model = st.text_input("Model", value=default_model, placeholder="model name")
    else:
        llm_model = st.text_input("Model", value=default_model, placeholder=default_model or "model name")

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
tab_web, tab_research, tab_sql, tab_maps, tab_sessions, tab_cache = st.tabs([
    "🌐 Web Search",
    "🚀 Research",
    "🗄️ SQL DB",
    "🗺️ Saved Maps",
    "💬 Sessions",
    "🕐 Cache",
])

with tab_web:
    st.subheader("🌐 Web Search")

    web_q = st.text_input(
        "Web search query",
        value=st.session_state.get("query", ""),
        placeholder="e.g. latest RAG evaluation methods"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        do_search = st.button("Search Web", width="stretch")

    with c2:
        add_to_rag = st.button(
            "Add Results to RAG",
            width="stretch",
            disabled=not st.session_state.web_results
        )

    with c3:
        ask_rag = st.button(
            "Ask with RAG",
            width="stretch",
            disabled=not web_q
        )

    if do_search:
        with st.spinner("Searching web..."):
            st.session_state.web_results = web_search(web_q, limit=5)
            st.session_state.web_results_query = web_q

    if st.session_state.web_results:
        st.markdown("#### Search Results")
        total_search_tokens = 0
        for idx, r in enumerate(st.session_state.web_results, start=1):
            tok = estimate_tokens(r.get("content", ""))
            total_search_tokens += tok
            with st.expander(f"{idx}. {r.get('title', 'Untitled')}"):
                st.write(r.get("snippet", ""))
                if r.get("url"):
                    st.markdown(f"[Open result]({r['url']})")
                st.caption(f"Estimated tokens: {tok}")

        st.info(f"Estimated total tokens in search results: {total_search_tokens}")

    if add_to_rag and st.session_state.web_results:
        with st.spinner("Indexing search results into RAG..."):
            n = add_web_results_to_rag(
                vdb,
                st.session_state.web_results_query,
                st.session_state.web_results
            )
        st.success(f"Added web results to RAG in {n} chunk(s).")

    if ask_rag:
        with st.spinner("Processing with collaborative agent pipeline..."):
            # Use full agent pipeline instead of direct RAG
            cfg = ProviderConfig(provider=provider, api_key=api_key, model=llm_model, base_url=base_url)
            app = build_graph(cfg, vdb)
            
            # Initialize state with indexed web results
            full_state = {
                "messages": [],
                "query": web_q,
                "active_agents": ["vector_db", "sql_db"],  # Skip web since already indexed
                "router_reasoning": "",
                "vector_findings": "",
                "sql_findings": "",
                "web_findings": "",
                "extraction_findings": "",
                "tagged_findings": [],
                "activity_log": [],
                "merged_context": "",
                "knowledge_map": {},
                "critique": "",
                "loop_count": 0,
                "summary": "",
                "experiment_plan": "",
                "current_agent": "",
                "sub_questions": [],
                "keywords": [],
                "scoping_reasoning": "",
                "synthesis_report": "",
                "conflicts": [],
                "credibility_map": {},
                "citation_grounding": {},
                "grounding_score": 0.0,
                "_needs_more": False,
            }
            
            # Run through agent pipeline
            final_state = None
            for event in app.stream(full_state.copy()):
                final_state = event
            
            if final_state and isinstance(final_state, dict):
                # Get the summary from the last state
                st.session_state.rag_answer = final_state.get("summary", "No answer generated")
                st.session_state.rag_context_last = final_state.get("merged_context", "")
            else:
                st.session_state.rag_answer = "(Agent pipeline did not complete)"


    if st.session_state.rag_context_last:
        with st.expander("Retrieved RAG Context"):
            st.text(st.session_state.rag_context_last[:5000])

    if st.session_state.rag_answer:
        st.markdown("#### LLM Answer with RAG")
        st.write(st.session_state.rag_answer)


with tab_research:
    st.header("Collaborative Research Query")

    col_q, col_opts = st.columns([3, 1])
    with col_q:
        query = st.text_area("Query", height=90,
                             placeholder="e.g. How does RAG relate to transformer architecture?")
    with col_opts:
        use_cache     = st.checkbox("Use 20-day cache", value=True)
        auto_save_map = st.checkbox("Auto-save map",    value=True)

    _ready = bool(query) and (bool(api_key) or provider == PROVIDER_LOCAL)
    run = st.button("🚀 Run Pipeline", type="primary", disabled=not _ready)

    if run:
        cached = cache_load(query) if use_cache else None
        if cached:
            st.info("⚡ Loaded from cache")
            full_state = cached
        else:
            app      = build_graph(cfg, vdb)
            progress = st.progress(0, "Starting…")
            pct_map  = {
                "scoping": 5, "router": 12, "vector_db": 24, "sql_db": 36, "web": 47,
                "reading_extraction": 57, "orchestrator": 66,
                "knowledge_mapper": 74, "critic": 82, "synthesis": 89,
                "summarizer": 94, "experiment_design": 98,
            }

            # Accumulate state across ALL agents properly
            # Each agent returns a partial update - we merge them so nothing is lost
            full_state = {
                "messages":[], "query": query,
                # ── BLOCK 1: Query Scoping Fields ──
                "sub_questions":[], "keywords":[], "scoping_reasoning":"",
                # ──────────────────────────────────
                "active_agents":[], "router_reasoning":"",
                "vector_findings":"", "sql_findings":"", "web_findings":"",
                "extraction_findings":"", "tagged_findings":[],
                "activity_log":[], "merged_context":"",
                "knowledge_map":{}, "critique":"", "loop_count":0,
                # ── BLOCK 2: Citation Grounding Fields ──
                "citation_grounding":{}, "grounding_score":0.0,
                # ── BLOCK 3: Conflict Detection Fields ──
                "synthesis_report":"", "conflicts":[], "credibility_map":{},
                # ─────────────────────────────────────────
                "summary":"", "experiment_plan":"", "current_agent":"", "_needs_more":False,
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
            
            # ── BLOCK 1 FIX: Persist full_state to session_state so scoping panel survives reruns ──
            st.session_state.last_result = full_state

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

        r_act, r_ans, r_exp, r_map, r_ctx, r_find, r_log = st.tabs([
            "🤝 Agent Activity", "💡 Final Answer", "🧪 Research Plan",
            "🗺️ Knowledge Map", "🔀 Merged Context", "🔍 Per-Agent Findings",
            "💬 Message Log",
        ])

        with r_act:
            # ── BLOCK 1: Display Query Scoping Panel ──────────────────────────────────
            st.subheader("🔍 Query Understanding")
            sub_q = full_state.get("sub_questions", [])
            keywords = full_state.get("keywords", [])
            if sub_q or keywords:
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown("**🎯 Sub-Questions Identified**")
                    for i, q in enumerate(sub_q[:5], 1):
                        st.caption(f"{i}. {q}")
                with sc2:
                    st.markdown("**📌 Key Themes**")
                    kw_badges = " ".join([f"`{k}`" for k in keywords[:8]])
                    st.markdown(kw_badges)
                st.divider()
            
            st.subheader("What each agent did")
            badge = {
                "vector_db":          '<span class="src-badge bv">Vector DB</span>',
                "sql_db":             '<span class="src-badge bs">SQL DB</span>',
                "web":                '<span class="src-badge bw">Web</span>',
                "reading_extraction": '<span class="src-badge be">Extraction</span>',
                "orchestrator":       '<span class="src-badge bm">Orchestrator</span>',
                "experiment_design":  '<span class="src-badge bx">Planning Agent</span>',
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
            # ── BLOCK 4: Export Button ──────────────────────────────────────────────────
            if full_state.get("summary"):
                export_col1, export_col2, export_col3 = st.columns([2, 1, 1])
                
                with export_col1:
                    # Generate export
                    zip_bytes, zip_filename = create_export_zip(query, full_state)
                    
                    st.download_button(
                        label="📥 Export Review (.zip)",
                        data=zip_bytes,
                        file_name=zip_filename,
                        mime="application/zip",
                        help="Download Markdown review + BibTeX citations"
                    )
                
                with export_col2:
                    md_bytes = generate_markdown_report(
                        query=query,
                        sub_questions=full_state.get("sub_questions", []),
                        keywords=full_state.get("keywords", []),
                        summary=full_state.get("summary", ""),
                        experiment_plan=full_state.get("experiment_plan", ""),
                        extraction_findings=full_state.get("extraction_findings", ""),
                        knowledge_map=full_state.get("knowledge_map", {}),
                    ).encode('utf-8')
                    
                    st.download_button(
                        label="📄 Markdown Only",
                        data=md_bytes,
                        file_name=f"review_{_hash(query)}.md",
                        mime="text/markdown",
                        help="Download the review as .md file"
                    )
                
                with export_col3:
                    bibtex_content, _ = extract_bibtex_entries(
                        extraction_findings=full_state.get("extraction_findings", ""),
                        web_findings=full_state.get("web_findings", ""),
                    )
                    bib_bytes = bibtex_content.encode('utf-8')
                    
                    st.download_button(
                        label="📚 BibTeX Only",
                        data=bib_bytes,
                        file_name=f"citations_{_hash(query)}.bib",
                        mime="text/plain",
                        help="Download citations as .bib file"
                    )
                
                st.divider()
            
            # ── BLOCK 2: Display Citation Grounding Badge ────────────────────────────
            grounding = full_state.get("citation_grounding", {})
            grounding_score = full_state.get("grounding_score", 0.0)
            
            if grounding:
                grounded_count = sum(1 for v in grounding.values() if v.get("grounded"))
                total = len(grounding)
                
                # Color bar based on grounding percentage
                color = "🟢" if grounding_score >= 0.8 else "🟡" if grounding_score >= 0.6 else "🔴"
                
                mc1, mc2, mc3 = st.columns([2, 1, 1])
                with mc1:
                    st.markdown(f"### {color} Citation Grounding: {grounded_count}/{total} ({int(grounding_score*100)}%)")
                with mc2:
                    if grounding_score >= 0.8:
                        st.success("Well grounded")
                    elif grounding_score >= 0.6:
                        st.warning("Partially grounded")
                    else:
                        st.error("Low grounding")
                
                # Show grounding details
                with st.expander("📋 Citation Details"):
                    for citation, info in list(grounding.items())[:15]:
                        status = "✅" if info.get("grounded") else "⚠️"
                        source = info.get("source", "none")
                        st.markdown(f"{status} **{citation[:80]}...**")
                        st.caption(f"Source: `{source}`")
                        if info.get("evidence"):
                            st.caption(f"Evidence: _{info['evidence'][:100]}..._")
                
                st.divider()
            
            st.markdown(full_state.get("summary", ""))

        with r_exp:
            st.subheader("Structured Research Plan")
            plan = full_state.get("experiment_plan", "")
            if plan:
                n_gaps = plan.count("**Gap ")
                n_hyp  = plan.count("**H-")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Research Gaps", n_gaps)
                mc2.metric("Hypotheses",    n_hyp)
                mc3.metric("Plan Length",   f"{len(plan):,} chars")
                st.divider()
                st.markdown(plan)
            else:
                st.info("No research plan generated yet.")

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
                st.iframe(_html_to_data_uri(render_knowledge_map(km)), height=520, scrolling=True)
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
                ("🗂️ Vector DB",            "vector_findings",     "bv"),
                ("🗄️ SQL / DB",             "sql_findings",        "bs"),
                ("🌐 Web",                  "web_findings",        "bw"),
                ("📖 Reading / Extraction", "extraction_findings", "be"),
                    ("🧵 Synthesis",           "synthesis_report",     "bm"),
            ]:
                with st.expander(f"{label} findings"):
                    st.markdown(full_state.get(key, ""))

        with r_log:
            av = {
                "[Scoping]":"🔭","[Router]":"🔀","[VectorDB]":"🗂️","[SQLDB]":"🗄️","[Web]":"🌐",
                "[ReadingExtraction]":"📖","[Synthesis]":"🧵",
                "[Orchestrator]":"🤝","[KnowledgeMapper]":"🗺️",
                "[Critic]":"🧐","[Summarizer]":"✍️","[ExperimentDesign]":"🧪",
            }
            for msg in full_state.get("messages", []):
                icon = next((v for k, v in av.items() if k in msg.content), "🤖")
                st.chat_message("assistant", avatar=icon).write(msg.content)


# ════════════ TAB 2 — SQL DB ══════════════════════════════════════════════════
with tab_sql:
    st.header("SQL / DB Browser")
    topics = sql_list_topics()
    st.metric("Topics in DB", len(topics))
    st.dataframe(topics, width="stretch")
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
        filt = st.text_input("Filter", key="maps_filter")
        shown = [m for m in all_maps if not filt or filt.lower() in m["query"].lower()]
        for m in shown:
            with st.expander(
                f"🗺️ {m['query'][:65]}…  ·  {m['nodes']}n / {m['edges']}e  ·  "
                f"{m['saved_at'][:16].replace('T',' ')}"
            ):
                if st.button("Visualise", key=f"vis_{m['file']}"):
                    raw = map_load(m["file"])
                    if raw:
                        st.iframe(_html_to_data_uri(render_knowledge_map(raw["map"])), height=470, scrolling=True)


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
            st.session_state.session_id = datetime.now(timezone.utc).strftime("sess_%Y%m%d_%H%M%S")
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
        age   = datetime.now(timezone.utc) - datetime.fromisoformat(e["ts"]).replace(tzinfo=timezone.utc)
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
                        st.iframe(_html_to_data_uri(render_knowledge_map(km, 400)), height=420, scrolling=True)
    st.divider()
    if st.button("🧹 Clear ALL cache"):
        for p in CACHE_DIR.glob("*.json"):
            p.unlink()
        st.success("Cache cleared.")
