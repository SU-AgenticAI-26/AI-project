# AI Research Project

An integrated AI-powered research assistant that combines multiple scholarly APIs, vector databases, and multi-agent RAG systems for comprehensive academic research.

## Features

- 🔍 **Multi-API Scholarly Search**: OpenAlex, Crossref, Arxiv, Semantic Scholar
- 🧠 **Multi-Agent RAG**: Collaborative agents for research, analysis, and knowledge mapping
- 🗂️ **Vector Database**: FAISS-powered document indexing and retrieval
- 🗄️ **SQL Knowledge Base**: Structured storage of research topics and relationships
- 🌐 **Web Integration**: Real-time API queries for current research data
- 🗺️ **Knowledge Mapping**: Interactive graph visualization of research concepts
- 🔐 **Secure Secrets Management**: Environment variables and encrypted key storage

## Quick Start

### 1. Setup

```bash
# Clone and enter the project directory
cd AI-project

# Run the setup script
python setup.py
```

The setup script will:
- Check Python version and dependencies
- Guide you through API key configuration
- Create necessary directories
- Initialize the SQL database

### 2. Configure API Keys

The project supports multiple API keys for different services:

- **OpenAI**: Required for LLM functionality
- **NASA**: For astronomy picture of the day
- **Semantic Scholar**: For academic paper search
- **Anthropic, Google, HuggingFace**: Optional for extended functionality

#### Option A: Environment Variables (Recommended)
```bash
export OPENAI_API_KEY="sk-your-key-here"
export NASA_API_KEY="your-nasa-key"
export SEMANTIC_SCHOLAR_API_KEY="your-ss-key"
```

#### Option B: Secrets File
```bash
cp secrets.json.example secrets.json
# Edit secrets.json with your API keys
```

#### Option C: Interactive Setup
```bash
python secrets_manager.py
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

#### Integrated Research App (Recommended)
```bash
streamlit run integrated_research_app.py
```

#### Individual Components
```bash
# Collaborative RAG with all agents
streamlit run collaborative_Rag.py

# Simple multi-agent knowledge explorer
streamlit run streamlit_app.py

# Basic API testing
python basic-search-example.py "your research query"
```

## Architecture

```
User Query
    │
[Router Agent] ← decides which agents to activate
    │
┌───┼────────────────┐
▼   ▼                ▼
[VectorDB Agent] [SQL/DB Agent] [Web/API Agent]
│   │                │
└───┴────────────────┘
        │
[Orchestrator Agent] ← merges findings
        │
[Knowledge Mapper] ← builds concept graphs
        │
[Summarizer] ← final research answer
```

## API Integrations

- **OpenAlex**: Open academic works database
- **Crossref**: DOI resolution and metadata
- **Arxiv**: Preprint server for physics, math, CS
- **Semantic Scholar**: AI-powered academic search
- **NASA**: Astronomy picture of the day

## Project Structure

```
AI-project/
├── research_apis.py          # Unified API client
├── secrets_manager.py        # Secure API key management
├── collaborative_Rag.py      # Multi-agent RAG system
├── integrated_research_app.py # Main Streamlit app
├── setup.py                  # Project setup script
├── requirements.txt          # Python dependencies
├── secrets.json.example      # API key template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Security

- API keys are never stored in code
- Secrets file (`secrets.json`) is gitignored
- Environment variables take precedence over files
- Keys are masked in UI displays

## Development

### Adding New APIs

1. Add API key to `secrets_manager.py`
2. Implement API client in `research_apis.py`
3. Update the web agent in `collaborative_Rag.py`

### Extending Agents

The multi-agent system is modular. Add new agents by:
1. Creating agent functions following the pattern
2. Adding to the StateGraph in `build_graph()`
3. Updating the router logic

## Dependencies

Key packages:
- `streamlit`: Web interface
- `langchain-openai`: LLM integration
- `faiss-cpu`: Vector search
- `networkx`, `pyvis`: Graph visualization
- `arxiv`: Arxiv API client
- `requests`: HTTP client

## License

This project is for educational and research purposes.

---

Arxiv access library: https://github.com/lukasschwab/arxiv.py

### Basic search example

``````python 

basic-search-example.py "graph neural networks: a review of methods and applications"

OPENALEX

    Graph neural networks: A review of methods and applications
    → https://doi.org/10.1016/j.aiopen.2021.01.001

    Graph Neural Networks: A Review of Methods and Applications
    → http://arxiv.org/abs/1812.08434

    Hyperbolic Graph Neural Networks: A Review of Methods and Applications
    → http://arxiv.org/abs/2202.13852

    A Review on Graph Neural Network Methods in Financial Applications
    → https://doi.org/10.6339/22-jds1047

    Embedding Entities and Relations for Learning and Inference in Knowledge Bases
    → http://arxiv.org/abs/1412.6575

CROSSREF

    Graph Neural Networks: Graph Classification
    → https://link.springer.com/10.1007/978-981-16-6054-2_9

    Graph Neural Networks: Graph Generation
    → https://link.springer.com/10.1007/978-981-16-6054-2_11

    Graph Embedding: Methods, Taxonomies, and Applications
    → https://www.taylorfrancis.com/books/9781003329701/chapters/10.1201/9781003329701-3

    Review of Bidirected Graph Neural Networks and Multidirected Graph Neural Networks
    → https://www.techrxiv.org/users/868224/articles/1299362-review-of-bidirected-graph-neural-networks-and-multidirected-graph-neural-networks?commit=36148e7136bc3ae5a9bcb99bb9e97cf7d190e32d

    Graph Neural Networks: Graph Structure Learning
    → https://link.springer.com/10.1007/978-981-16-6054-2_14

ARXIV

    Observation of the rare Bs0→μ+μ−Bs0→μ+μ− decay from the combined analysis of CMS and LHCb data
    → https://arxiv.org/pdf/1411.4413v2

    Expected Performance of the ATLAS Experiment - Detector, Trigger and Physics
    → https://arxiv.org/pdf/0901.0512v4

    Deep Search for Joint Sources of Gravitational Waves and High-Energy Neutrinos with IceCube During the Third Observing Run of LIGO and Virgo
    → https://arxiv.org/pdf/2601.07595v3

    GWTC-4.0: An Introduction to Version 4.0 of the Gravitational-Wave Transient Catalog
    → https://arxiv.org/pdf/2508.18080v2

    GWTC-4.0: Methods for Identifying and Characterizing Gravitational-wave Transients
    → https://arxiv.org/pdf/2508.18081v1
``````




Now let me build this comprehensive file:Here's everything new in `rag_knowledge_app.py` compared to your original:

**RAG module** — a self-contained `RAGModule` class wraps FAISS + OpenAI embeddings. Upload `.txt`/`.md` files from the sidebar; they get chunked, embedded, and persisted to disk. Every research run auto-retrieves the top-k most relevant chunks and injects them into the researcher's prompt.

**Persistent Knowledge Maps** — every map is saved to `rag_data/knowledge_maps/` as a timestamped JSON file. The "Saved Maps" tab lets you search, visualize, and delete them individually.

**20-Day Query Cache** — results are keyed by a SHA-256 hash of the query and expire after 20 days. The "Cache Browser" tab shows age, node/summary stats, and lets you reload any past result without re-running the pipeline. A bulk-delete slider lets you prune by age.

**Conversation Sessions** — every query+summary+metadata is appended to the current session and saved to `rag_data/sessions/`. The "Conversations" tab lets you browse all past sessions, page through their turns, and start new sessions cleanly.

**Critic agent + loop** — a fourth agent reviews the knowledge map after the mapper. If it has fewer than 8 nodes or is missing key relationships, it loops back to the researcher (max 2 loops) with specific feedback before handing off to the summarizer.

```bash
pip install streamlit langgraph langchain-openai langchain-core \
            langchain-community langchain-text-splitters \
            faiss-cpu networkx pyvis sentence-transformers tiktoken

streamlit run rag_knowledge_app.py
```


**The collaborative architecture** — 8 agents wired together through LangGraph:

The `Router` agent reads the query first and decides which search agents to actually fire. It produces a JSON decision like `{"agents": ["vector_db", "sql_db"], "reasoning": "..."}`. Skipped agents log themselves as inactive so you always see what was considered.

The three search agents run in sequence (they can be parallelised with LangGraph's `Send` API if you need speed): `VectorDB` does semantic FAISS search over your indexed documents, `SQL/DB` does keyword matching across three tables (`topics`, `relationships`, `facts`), and `Web` handles live or knowledge-based lookups.

The `Orchestrator` then receives all three sets of findings and explicitly merges them — deduplicating, resolving conflicts, and labelling every fact by which source contributed it (Vector DB / SQL DB / Web).

**The activity panel** — the first results tab, "Agent Activity", is the key UI innovation. Every agent writes a structured log entry: what it searched, how many rows/chunks it found, with expandable drawers showing the actual retrieved chunks and matched SQL rows. You can see exactly what `VectorDB` pulled from which file, and exactly which SQL rows the `SQL/DB` agent matched.

**Graph enrichment loop** — the Critic reviews node count and source diversity. If the graph is sparse, it loops back to the Orchestrator (with specific gap feedback), not just to research again — so the enrichment builds on the already-merged context.

```bash
pip install streamlit langgraph langchain-openai langchain-core \
            langchain-community faiss-cpu langchain-text-splitters \
            networkx pyvis tiktoken sqlalchemy

streamlit run collaborative_rag.py
```

The SQLite DB bootstraps with sample ML topics on first run so the SQL agent has something to search immediately.