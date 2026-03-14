import streamlit as st
from collaborative_Rag import build_graph, VectorDBModule, index_arxiv_documents, index_nasa_documents, init_sql_db
from research_apis import ResearchAPIs
from secrets_manager import get_api_key, has_api_key, setup_secrets_interactive

st.set_page_config(page_title="Integrated Research AI", layout="wide")

st.title("🔬 Integrated Research AI Platform")

# Initialize SQL DB
init_sql_db()

# Sidebar for configuration
with st.sidebar:
    mode = st.selectbox("Mode", ["Multi-Agent RAG", "API Explorer"])
    api_key = get_api_key('openai')
    
    if not api_key:
        st.warning("⚠️ OpenAI API key not found. Please configure it below.")
        if st.button("🔧 Configure API Keys"):
            st.info("Run `python secrets_manager.py` in terminal to set up API keys")
            with st.expander("Manual Setup Instructions"):
                st.markdown("""
                **Option 1: Environment Variables (Recommended)**
                ```bash
                export OPENAI_API_KEY="your-key-here"
                export NASA_API_KEY="your-nasa-key"
                export SEMANTIC_SCHOLAR_API_KEY="your-ss-key"
                ```

                **Option 2: Secrets File**
                Create `secrets.json`:
                ```json
                {
                    "openai": "your-openai-key",
                    "nasa": "your-nasa-key",
                    "semantic_scholar": "your-semantic-scholar-key"
                }
                ```

                **Option 3: Interactive Setup**
                ```bash
                python secrets_manager.py
                ```
                """)
    else:
        st.success("✅ OpenAI API key configured")
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        st.caption(f"Key: {masked_key}")

    # Initialize VectorDB if we have API key
    if api_key and "vdb" not in st.session_state:
        st.session_state.vdb = VectorDBModule(api_key)

if mode == "API Explorer":
    st.header("Direct API Testing")
    apis = ResearchAPIs()

    query = st.text_input("Query")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Search Scholarly"):
            results = apis.search_scholarly(query)
            st.json(results)
    
    with col2:
        if st.button("Search Arxiv"):
            results = apis.search_arxiv(query)
            for r in results:
                st.write(f"- {r.title}")
    
    with col3:
        if st.button("NASA APOD"):
            apod = apis.get_nasa_apod()
            if apod and 'error' not in apod:
                st.image(apod['url'])
                st.write(apod['title'])
            else:
                st.error(apod.get('error', 'Failed to get APOD'))

elif mode == "Multi-Agent RAG":
    if not api_key:
        st.warning("Please configure your OpenAI API key in the sidebar.")
        st.stop()

    vdb = st.session_state.get("vdb")
    if not vdb:
        st.error("Failed to initialize VectorDB. Check your API key.")
        st.stop()

    # ArXiv Document Indexing
    st.subheader("📚 Index ArXiv Documents")
    col1, col2 = st.columns([3, 1])
    with col1:
        arxiv_query = st.text_input("ArXiv search query", 
                                   placeholder="e.g. transformer attention OR neural networks")
    with col2:
        index_limit = st.number_input("Max documents", min_value=1, max_value=50, value=10)
    
    if st.button("🔍 Search & Index ArXiv", help="Search arXiv and add documents to vector database"):
        if arxiv_query:
            with st.spinner(f"Searching arXiv for '{arxiv_query}'..."):
                try:
                    indexed_count = index_arxiv_documents(arxiv_query, vdb, index_limit)
                    if indexed_count > 0:
                        st.success(f"✅ Indexed {indexed_count} ArXiv documents!")
                        st.metric("Total vector chunks", vdb.count())
                    else:
                        st.warning("No documents were indexed. Check the query or try again.")
                except Exception as e:
                    st.error(f"❌ Failed to index documents: {str(e)}")
                    st.info("Check the terminal for detailed error logs.")
        else:
            st.error("Please enter an ArXiv search query.")

    # NASA Document Indexing
    st.subheader("🚀 Index NASA Data")
    if st.button("🌌 Index NASA APOD", help="Fetch and index NASA's Astronomy Picture of the Day"):
        with st.spinner("Fetching NASA data..."):
            try:
                indexed_count = index_nasa_documents(vdb)
                if indexed_count > 0:
                    st.success(f"✅ Indexed {indexed_count} NASA document!")
                    st.metric("Total vector chunks", vdb.count())
                else:
                    st.warning("No NASA data was indexed. Check your NASA API key.")
            except Exception as e:
                st.error(f"❌ Failed to index NASA data: {str(e)}")
                st.info("Make sure your NASA API key is configured.")

    st.divider()

    query = st.text_area("Enter your research query", height=100,
                        placeholder="e.g. How does transformer attention work?")

    # Document upload section
    st.subheader("📄 Upload Documents (optional)")
    uploaded = st.file_uploader("Upload .txt / .md files", type=["txt","md"],
                               accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            n = vdb.add_file(f)
            st.success(f"Added {f.name} → {n} chunks")
        st.metric("Total vector chunks", vdb.count())

    if st.button("🚀 Run Research Pipeline", type="primary"):
        with st.spinner("Running collaborative research pipeline..."):
            app = build_graph(api_key, vdb)
            
            # Run the pipeline
            result = app.invoke({
                "messages": [],
                "query": query,
                "active_agents": [],
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
            st.subheader("💡 Final Answer")
            st.markdown(result.get("summary", ""))

            # Activity log
            with st.expander("🤝 Agent Activity Log"):
                for entry in result.get("activity_log", []):
                    st.write(f"{entry['icon']} **{entry['title']}**: {entry['detail']}")

            # Knowledge map
            km = result.get("knowledge_map", {})
            if km.get("nodes"):
                st.subheader("🗺️ Knowledge Map")
                # Simple visualization (could be enhanced)
                st.write(f"Generated map with {len(km['nodes'])} nodes and {len(km['edges'])} edges")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Nodes**")
                    for node in km["nodes"][:10]:  # Show first 10
                        st.write(f"- {node['label']} ({node.get('type', 'concept')})")
                with col2:
                    st.markdown("**Edges**")
                    for edge in km["edges"][:10]:  # Show first 10
                        st.write(f"- {edge['source']} → {edge['target']} ({edge.get('relation', '')})")