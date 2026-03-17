"""
Multi-Agent Streamlit App using LangGraph + OpenAI + Knowledge Maps

Install dependencies:
    pip install streamlit langgraph langchain-openai langchain-core networkx pyvis


Add a Critic agent that reviews the summary and asks the researcher to go deeper
Add conditional edges in LangGraph to loop back if the knowledge map is too sparse
Swap the LLM per agent (e.g. GPT-4o for research, GPT-4o-mini for summarizing)
Persist the graph to a database between sessions

Run:
    streamlit run app.py
"""

import json
import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile, os
from typing import TypedDict, Annotated, List
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from secrets_manager import get_api_key
from collaborative_Rag import VectorDBModule, index_arxiv_documents, index_nasa_documents, map_save, build_graph as collab_build_graph
from research_apis import ResearchAPIs

# ─────────────────────────────────────────────
# 1.  Shared State
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    query: str
    research_notes: str
    summary: str
    knowledge_map: dict          # {"nodes": [...], "edges": [...]}
    current_agent: str


# ─────────────────────────────────────────────
# 2.  Agents
# ─────────────────────────────────────────────

def make_llm(api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.3):
    return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)


def researcher_agent(state: AgentState, llm, enable_rag: bool = False, vector_db: VectorDBModule = None) -> AgentState:
    """Researches the query and returns structured notes."""
    print("[LOG] Researcher agent starting research...")
    
    # RAG retrieval if enabled
    retrieved_docs = ""
    if enable_rag and vector_db:
        print("[LOG] Performing RAG retrieval...")
        docs = vector_db.search(state['query'], k=3)
        if docs:
            retrieved_docs = "\n\n".join([f"[Retrieved Document]: {doc.page_content}" for doc in docs])
            print(f"[LOG] Retrieved {len(docs)} relevant documents")
        else:
            print("[LOG] No relevant documents found in vector DB")
    
    system = SystemMessage(content=(
        "You are a Research Agent. Given a user query, produce detailed research notes. "
        "Cover key concepts, facts, entities, and relationships. Be thorough."
        + (f"\n\nUse the following retrieved documents as additional context:\n{retrieved_docs}" if retrieved_docs else "")
    ))
    human = HumanMessage(content=f"Research this topic thoroughly:\n\n{state['query']}")
    response = llm.invoke([system, human])
    print(f"[LOG] Researcher agent completed: {len(response.content)} characters")
    return {
        "messages": [AIMessage(content=f"[Researcher] {response.content}")],
        "research_notes": response.content,
        "current_agent": "researcher",
    }


def knowledge_mapper_agent(state: AgentState, llm) -> AgentState:
    """Converts research notes into a knowledge map (nodes + edges as JSON)."""
    print("[LOG] Knowledge Mapper agent starting mapping...")
    system = SystemMessage(content=(
        "You are a Knowledge Mapping Agent. Given research notes, extract a knowledge graph. "
        "Return ONLY valid JSON with this exact schema:\n"
        '{"nodes": [{"id": "...", "label": "...", "type": "concept|entity|fact"}], '
        '"edges": [{"source": "...", "target": "...", "relation": "..."}]}\n'
        "Keep it to the 10-15 most important nodes. No extra text outside JSON."
    ))
    human = HumanMessage(content=f"Research notes:\n\n{state['research_notes']}")
    response = llm.invoke([system, human])

    # Robustly parse JSON
    raw = response.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        knowledge_map = json.loads(raw)
        print(f"[LOG] Knowledge Mapper completed: {len(knowledge_map.get('nodes', []))} nodes, {len(knowledge_map.get('edges', []))} edges")
    except json.JSONDecodeError:
        knowledge_map = {"nodes": [], "edges": [], "error": "Parse failed"}
        print("[LOG] Knowledge Mapper failed to parse JSON")

    return {
        "messages": [AIMessage(content=f"[KnowledgeMapper] Built map with "
                                       f"{len(knowledge_map.get('nodes', []))} nodes.")],
        "knowledge_map": knowledge_map,
        "current_agent": "knowledge_mapper",
    }


def summarizer_agent(state: AgentState, llm) -> AgentState:
    """Writes a concise answer using research notes + knowledge map."""
    print("[LOG] Summarizer agent starting summarization...")
    system = SystemMessage(content=(
        "You are a Summarizer Agent. Using the research notes and knowledge map provided, "
        "write a clear, structured, and concise answer to the original query."
    ))
    human = HumanMessage(content=(
        f"Original query: {state['query']}\n\n"
        f"Research notes:\n{state['research_notes']}\n\n"
        f"Knowledge map nodes: {[n['label'] for n in state['knowledge_map'].get('nodes', [])]}"
    ))
    response = llm.invoke([system, human])
    print(f"[LOG] Summarizer agent completed: {len(response.content)} characters")
    return {
        "messages": [AIMessage(content=f"[Summarizer] {response.content}")],
        "summary": response.content,
        "current_agent": "summarizer",
    }


# ─────────────────────────────────────────────
# 3.  Build LangGraph
# ─────────────────────────────────────────────

def build_graph(api_key: str, enable_rag: bool = False, vector_db: VectorDBModule = None):
    llm_instance = make_llm(api_key)

    graph = StateGraph(AgentState)

    graph.add_node("researcher",       lambda s: researcher_agent(s, llm_instance, enable_rag, vector_db))
    graph.add_node("knowledge_mapper", lambda s: knowledge_mapper_agent(s, llm_instance))
    graph.add_node("summarizer",       lambda s: summarizer_agent(s, llm_instance))

    graph.set_entry_point("researcher")
    graph.add_edge("researcher",       "knowledge_mapper")
    graph.add_edge("knowledge_mapper", "summarizer")
    graph.add_edge("summarizer",       END)

    return graph.compile()


# ─────────────────────────────────────────────
# 4.  Knowledge Map Visualisation (pyvis)
# ─────────────────────────────────────────────

TYPE_COLORS = {"concept": "#4A90D9", "entity": "#E67E22", "fact": "#2ECC71", "default": "#9B59B6"}

def render_knowledge_map(knowledge_map: dict) -> str:
    """Renders a pyvis graph and returns the HTML string."""
    net = Network(height="500px", width="100%", bgcolor="#1a1a2e", font_color="white",
                  directed=True)
    net.set_options("""
    {
      "edges": {"arrows": {"to": {"enabled": true}}},
      "physics": {"stabilization": {"iterations": 200}}
    }
    """)

    for node in knowledge_map.get("nodes", []):
        color = TYPE_COLORS.get(node.get("type", "default"), TYPE_COLORS["default"])
        net.add_node(node["id"], label=node["label"], color=color,
                     title=f"Type: {node.get('type','unknown')}", size=20)

    for edge in knowledge_map.get("edges", []):
        net.add_edge(edge["source"], edge["target"],
                     title=edge.get("relation", ""), label=edge.get("relation", ""))

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        return f.name


# ─────────────────────────────────────────────
# 5.  Streamlit UI
# ─────────────────────────────────────────────

st.set_page_config(page_title="Multi-Agent Knowledge Explorer", layout="wide",
                   page_icon="🧠")

st.title("🧠 Multi-Agent Knowledge Explorer")
st.caption("Powered by LangGraph · OpenAI · Knowledge Maps")

# Mode selection via tabs
tab_knowledge, tab_api, tab_collab = st.tabs(["🧠 Knowledge Explorer", "🔬 API Explorer", "🤝 Collaborative RAG"])

# Initialize variables
vector_db = None

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Get API key from secrets manager
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
    
    st.divider()
    st.info("Knowledge map nodes are color-coded:\n"
            "🔵 Concept  🟠 Entity  🟢 Fact  🟣 Other")

# ── Main Content ──

with tab_knowledge:
    st.header("🧠 Knowledge Explorer")
    st.markdown("Simple multi-agent pipeline with optional RAG")
    
    # RAG toggle for Knowledge Explorer
    enable_rag = st.checkbox("🔍 Enable RAG (Retrieval-Augmented Generation)", 
                           help="Retrieve relevant documents from knowledge base before research")
    
    # Initialize vector_db if RAG enabled
    vector_db = None
    if enable_rag:
        vector_db = VectorDBModule(api_key)
        doc_count = vector_db.count()
        st.info(f"📚 Vector DB loaded: {doc_count} documents indexed")
        if doc_count == 0:
            st.warning("⚠️ Vector DB is empty. Index some documents below.")
        
        # Document Indexing
        st.subheader("📚 Index Documents")
        
        # ArXiv Indexing
        with st.expander("🔍 Index ArXiv Papers"):
            col1, col2 = st.columns([3, 1])
            with col1:
                arxiv_query = st.text_input("ArXiv search query", 
                                           placeholder="e.g. transformer attention OR neural networks")
            with col2:
                arxiv_limit = st.number_input("Max docs", min_value=1, max_value=50, value=5, key="arxiv_limit")
            
            if st.button("🔍 Index ArXiv", help="Search and index ArXiv papers"):
                if arxiv_query:
                    with st.spinner(f"Indexing ArXiv papers for '{arxiv_query}'..."):
                        try:
                            indexed = index_arxiv_documents(arxiv_query, vector_db, arxiv_limit)
                            if indexed > 0:
                                st.success(f"✅ Indexed {indexed} ArXiv papers!")
                                st.rerun()  # Refresh to update doc count
                            else:
                                st.warning("No papers indexed. Try a different query.")
                        except Exception as e:
                            st.error(f"❌ Failed to index: {str(e)}")
                else:
                    st.error("Enter an ArXiv query.")
        
        # NASA Indexing
        with st.expander("🚀 Index NASA Data"):
            if st.button("🌌 Index NASA APOD", help="Fetch and index NASA's Astronomy Picture of the Day"):
                with st.spinner("Fetching NASA data..."):
                    try:
                        indexed = index_nasa_documents(vector_db)
                        if indexed > 0:
                            st.success(f"✅ Indexed {indexed} NASA document!")
                            st.rerun()
                        else:
                            st.warning("No NASA data indexed. Check NASA API key.")
                    except Exception as e:
                        st.error(f"❌ Failed to index NASA data: {str(e)}")
        
        # File Upload
        with st.expander("📄 Upload Files"):
            uploaded = st.file_uploader("Upload .txt or .md files", type=["txt", "md"], 
                                       accept_multiple_files=True, key="file_upload")
            if uploaded:
                total_chunks = 0
                for f in uploaded:
                    try:
                        chunks = vector_db.add_file(f)
                        total_chunks += chunks
                        st.success(f"✅ Added {f.name} → {chunks} chunks")
                    except Exception as e:
                        st.error(f"❌ Failed to add {f.name}: {str(e)}")
                if total_chunks > 0:
                    st.info(f"Total chunks added: {total_chunks}")
                    st.rerun()
    
    st.markdown("**Agent Pipeline**")
    st.markdown("1. 🔍 **Researcher** — deep-dives the topic" + (" (with RAG)" if enable_rag else ""))
    st.markdown("2. 🗺️ **Knowledge Mapper** — extracts nodes & edges")
    st.markdown("3. ✍️ **Summarizer** — crafts the final answer")
    
    st.divider()
    
    query = st.text_area("Enter your research query", height=100,
                         placeholder="e.g. How does transformer attention work?", key="knowledge_query")

    run_btn = st.button("🚀 Run Multi-Agent Pipeline", type="primary",
                        disabled=not query)

    if run_btn:
        # Build the agent graph
        vector_db_param = vector_db if enable_rag else None
        app = build_graph(api_key, enable_rag, vector_db_param)

        # Stream execution with progress
        progress = st.progress(0, text="Starting pipeline…")
        status_placeholder = st.empty()

        agent_labels = {
            "researcher":       ("🔍 Researcher agent working…",       33),
            "knowledge_mapper": ("🗺️ Knowledge Mapper building graph…", 66),
            "summarizer":       ("✍️ Summarizer writing answer…",       90),
        }

        final_state = None
        for event in app.stream({
            "messages": [],
            "query": query,
            "research_notes": "",
            "summary": "",
            "knowledge_map": {},
            "current_agent": "",
        }):
            for node_name, state_update in event.items():
                label, pct = agent_labels.get(node_name, ("Processing…", 50))
                progress.progress(pct, text=label)
                status_placeholder.markdown(f"**Current agent:** `{node_name}`")
                final_state = state_update   # keep updating; last = summarizer output

        progress.progress(100, text="✅ Pipeline complete!")
        status_placeholder.empty()

        # Reconstruct full state from stream (LangGraph streams partial updates)
        # Re-run without streaming to get full final state easily
        full_state = app.invoke({
            "messages": [],
            "query": query,
            "research_notes": "",
            "summary": "",
            "knowledge_map": {},
            "current_agent": "",
        })

        # Save knowledge map
        km = full_state["knowledge_map"]
        if km.get("nodes"):
            try:
                map_filename = map_save(query, km)
                st.success(f"💾 Knowledge map saved as {map_filename}")
            except Exception as e:
                st.warning(f"Failed to save knowledge map: {str(e)}")

        # ── Tabs for results ──
        tab1, tab2, tab3, tab4 = st.tabs(
            ["💡 Summary", "🗺️ Knowledge Map", "📝 Research Notes", "💬 Agent Messages"])

        with tab1:
            st.subheader("Final Answer")
            st.markdown(full_state["summary"])

        with tab2:
            st.subheader("Knowledge Map")
            km = full_state["knowledge_map"]
            if km.get("nodes"):
                html_path = render_knowledge_map(km)
                with open(html_path, "r") as f:
                    html_content = f.read()
                os.unlink(html_path)
                st.components.v1.html(html_content, height=520, scrolling=False)

                # Raw data expander
                with st.expander("📊 Raw Graph Data"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Nodes**")
                        st.dataframe(km["nodes"])
                    with col2:
                        st.markdown("**Edges**")
                        st.dataframe(km["edges"])
            else:
                st.warning("Knowledge map could not be generated.")
                st.json(km)

        with tab3:
            st.subheader("Research Notes")
            st.markdown(full_state["research_notes"])

        with tab4:
            st.subheader("Agent Message Log")
            for msg in full_state["messages"]:
                if "[Researcher]" in msg.content:
                    st.chat_message("assistant", avatar="🔍").write(msg.content)
                elif "[KnowledgeMapper]" in msg.content:
                    st.chat_message("assistant", avatar="🗺️").write(msg.content)
                elif "[Summarizer]" in msg.content:
                    st.chat_message("assistant", avatar="✍️").write(msg.content)
                else:
                    st.chat_message("assistant").write(msg.content)

with tab_api:
    st.header("🔬 Direct API Testing")
    st.markdown("Test research APIs directly")
    apis = ResearchAPIs()

    query = st.text_input("Query for APIs", placeholder="e.g. quantum computing", key="api_query")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Search Scholarly (OpenAlex)", help="Search academic works"):
            if query:
                results = apis.search_scholarly(query)
                st.subheader("Scholarly Results")
                for r in results[:5]:  # limit display
                    st.write(f"- **{r.get('title', 'No title')}** ({r.get('year', 'N/A')})")
                    if r.get('url'):
                        st.markdown(f"[Read more]({r['url']})")
            else:
                st.error("Enter a query.")
    
    with col2:
        if st.button("📄 Search ArXiv", help="Search arXiv papers"):
            if query:
                results = apis.search_arxiv(query)
                st.subheader("ArXiv Results")
                for r in results[:5]:
                    st.write(f"- {r.get('title', 'No title')}")
                    if r.get('pdf_url'):
                        st.markdown(f"[PDF]({r['pdf_url']})")
            else:
                st.error("Enter a query.")
    
    with col3:
        if st.button("🌌 NASA APOD", help="Get Astronomy Picture of the Day"):
            apod = apis.get_nasa_apod()
            if apod and 'error' not in apod:
                st.image(apod['url'], caption=apod.get('title', 'NASA APOD'))
                st.write(apod.get('explanation', ''))
            else:
                st.error(apod.get('error', 'Failed to get APOD'))

with tab_collab:
    st.header("🤝 Collaborative RAG")
    st.markdown("Full collaborative research with multiple data sources")
    
    st.markdown("**Agent Pipeline**")
    st.markdown("1. 🧭 **Router** — selects active agents")
    st.markdown("2. 🔍 **Vector DB + Web + SQL** — parallel research")
    st.markdown("3. 🎯 **Orchestrator** — merges findings")
    st.markdown("4. 🗺️ **Knowledge Mapper** — builds graph")
    st.markdown("5. ✍️ **Summarizer** — final answer")
    
    st.divider()
    
    # Initialize VectorDB
    vector_db = VectorDBModule(api_key)
    doc_count = vector_db.count()
    st.info(f"📚 Vector DB loaded: {doc_count} documents indexed")

    # Document Indexing
    st.subheader("📚 Index Documents")
    
    # ArXiv Indexing
    with st.expander("🔍 Index ArXiv Papers"):
        col1, col2 = st.columns([3, 1])
        with col1:
            arxiv_query = st.text_input("ArXiv search query", 
                                       placeholder="e.g. transformer attention OR neural networks", key="collab_arxiv_query")
        with col2:
            arxiv_limit = st.number_input("Max docs", min_value=1, max_value=50, value=5, key="collab_arxiv_limit")
        
        if st.button("🔍 Index ArXiv", help="Search and index ArXiv papers", key="collab_index_arxiv"):
            if arxiv_query:
                with st.spinner(f"Indexing ArXiv papers for '{arxiv_query}'..."):
                    try:
                        indexed = index_arxiv_documents(arxiv_query, vector_db, arxiv_limit)
                        if indexed > 0:
                            st.success(f"✅ Indexed {indexed} ArXiv papers!")
                            st.rerun()
                        else:
                            st.warning("No papers indexed. Try a different query.")
                    except Exception as e:
                        st.error(f"❌ Failed to index: {str(e)}")
            else:
                st.error("Enter an ArXiv query.")
    
    # NASA Indexing
    with st.expander("🚀 Index NASA Data"):
        if st.button("🌌 Index NASA APOD", help="Fetch and index NASA's Astronomy Picture of the Day", key="collab_index_nasa"):
            with st.spinner("Fetching NASA data..."):
                try:
                    indexed = index_nasa_documents(vector_db)
                    if indexed > 0:
                        st.success(f"✅ Indexed {indexed} NASA document!")
                        st.rerun()
                    else:
                        st.warning("No NASA data indexed. Check NASA API key.")
                except Exception as e:
                    st.error(f"❌ Failed to index NASA data: {str(e)}")
    
    # File Upload
    with st.expander("📄 Upload Files"):
        uploaded = st.file_uploader("Upload .txt or .md files", type=["txt", "md"], 
                                   accept_multiple_files=True, key="collab_file_upload")
        if uploaded:
            total_chunks = 0
            for f in uploaded:
                try:
                    chunks = vector_db.add_file(f)
                    total_chunks += chunks
                    st.success(f"✅ Added {f.name} → {chunks} chunks")
                except Exception as e:
                    st.error(f"❌ Failed to add {f.name}: {str(e)}")
            if total_chunks > 0:
                st.info(f"Total chunks added: {total_chunks}")
                st.rerun()

    st.divider()

    query = st.text_area("Enter your research query", height=100,
                        placeholder="e.g. How does transformer attention work?", key="collab_query")

    if st.button("🚀 Run Collaborative Research Pipeline", type="primary"):
        with st.spinner("Running collaborative research pipeline..."):
            app = collab_build_graph(api_key, vector_db)
            # Run the collaborative graph
            result = app.invoke({
                "messages": [],
                "query": query,
                "active_agents": ["vector_db", "web"],  # default agents
                "router_reasoning": "",
                "vector_findings": "",
                "sql_findings": "",
                "web_findings": "",
                "activity_log": [],
                "merged_context": "",
                "knowledge_map": {},
                "critique": "",
                "loop_count": 0,
                "summary": "",
                "current_agent": "",
            })
            
            # Display results
            tab1, tab2, tab3, tab4 = st.tabs(["💡 Summary", "🗺️ Knowledge Map", "📝 Merged Context", "💬 Activity Log"])
            
            with tab1:
                st.subheader("Final Summary")
                st.markdown(result["summary"])
            
            with tab2:
                st.subheader("Knowledge Map")
                km = result["knowledge_map"]
                if km.get("nodes"):
                    html_path = render_knowledge_map(km)
                    with open(html_path, "r") as f:
                        html_content = f.read()
                    os.unlink(html_path)
                    st.components.v1.html(html_content, height=520, scrolling=False)
                    
                    # Save map
                    try:
                        map_filename = map_save(query, km)
                        st.success(f"💾 Knowledge map saved as {map_filename}")
                    except Exception as e:
                        st.warning(f"Failed to save knowledge map: {str(e)}")
                else:
                    st.warning("Knowledge map could not be generated.")
            
            with tab3:
                st.subheader("Merged Research Context")
                st.markdown(result["merged_context"])
            
            with tab4:
                st.subheader("Agent Activity Log")
                for activity in result["activity_log"]:
                    st.write(f"- {activity}")