#start fo\\

"""
Multi-Agent Streamlit App using LangGraph + OpenAI + Knowledge Maps

Install dependencies:
    pip install streamlit langgraph langchain-openai langchain-core networkx pyvis

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


def researcher_agent(state: AgentState, llm) -> AgentState:
    """Researches the query and returns structured notes."""
    system = SystemMessage(content=(
        "You are a Research Agent. Given a user query, produce detailed research notes. "
        "Cover key concepts, facts, entities, and relationships. Be thorough."
    ))
    human = HumanMessage(content=f"Research this topic thoroughly:\n\n{state['query']}")
    response = llm.invoke([system, human])
    return {
        "messages": [AIMessage(content=f"[Researcher] {response.content}")],
        "research_notes": response.content,
        "current_agent": "researcher",
    }


def knowledge_mapper_agent(state: AgentState, llm) -> AgentState:
    """Converts research notes into a knowledge map (nodes + edges as JSON)."""
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
    except json.JSONDecodeError:
        knowledge_map = {"nodes": [], "edges": [], "error": "Parse failed"}

    return {
        "messages": [AIMessage(content=f"[KnowledgeMapper] Built map with "
                                       f"{len(knowledge_map.get('nodes', []))} nodes.")],
        "knowledge_map": knowledge_map,
        "current_agent": "knowledge_mapper",
    }


def summarizer_agent(state: AgentState, llm) -> AgentState:
    """Writes a concise answer using research notes + knowledge map."""
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
    return {
        "messages": [AIMessage(content=f"[Summarizer] {response.content}")],
        "summary": response.content,
        "current_agent": "summarizer",
    }


# ─────────────────────────────────────────────
# 3.  Build LangGraph
# ─────────────────────────────────────────────

def build_graph(api_key: str):
    llm = make_llm(api_key)

    graph = StateGraph(AgentState)

    graph.add_node("researcher",       lambda s: researcher_agent(s, llm))
    graph.add_node("knowledge_mapper", lambda s: knowledge_mapper_agent(s, llm))
    graph.add_node("summarizer",       lambda s: summarizer_agent(s, llm))

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

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password",
                            placeholder="sk-...")
    st.divider()
    st.markdown("**Agent Pipeline**")
    st.markdown("1. 🔍 **Researcher** — deep-dives the topic")
    st.markdown("2. 🗺️ **Knowledge Mapper** — extracts nodes & edges")
    st.markdown("3. ✍️ **Summarizer** — crafts the final answer")
    st.divider()
    st.info("Knowledge map nodes are color-coded:\n"
            "🔵 Concept  🟠 Entity  🟢 Fact  🟣 Other")

# ── Main ──
query = st.text_area("Enter your research query", height=100,
                     placeholder="e.g. How does transformer attention work?")

run_btn = st.button("🚀 Run Multi-Agent Pipeline", type="primary",
                    disabled=not (api_key and query))

if run_btn:
    if not api_key.startswith("sk-"):
        st.error("Please enter a valid OpenAI API key.")
        st.stop()

    # Build the agent graph
    app = build_graph(api_key)

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