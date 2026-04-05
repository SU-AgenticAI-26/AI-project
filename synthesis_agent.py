"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Synthesis Agent — Standalone Test                        ║
║                                                                              ║
║  What it does (per proposal Section 5.5):                                    ║
║    Takes the merged evidence context + per-paper extraction records and      ║
║    produces a dedicated thematic literature review grouped by:               ║
║      • Topics     — what themes emerge across papers                         ║
║      • Methods    — what approaches are used and how they compare            ║
║      • Disagreements — where papers contradict or conflict                   ║
║    Multiple viewpoints are preserved rather than prematurely resolved.       ║
║                                                                              ║
║  Pipeline position:                                                          ║
║    ... → Orchestrator → Knowledge Mapper → Critic →                         ║
║          [Synthesis] → Summarizer → Experiment Design → END                 ║
║                                                                              ║
║  Run:   python -m streamlit run synthesis_agent.py                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import operator
import os
from typing import Annotated, List, TypedDict

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


# ══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# Mirrors the fields used in the full streamlit_app.py pipeline, plus
# the new field written by the Synthesis Agent.
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages:             Annotated[List, operator.add]
    query:                str
    # Inputs the Synthesis Agent reads
    merged_context:       str
    extraction_findings:  str
    knowledge_map:        dict
    # ── NEW field written by Synthesis Agent ──────────────────────────────────
    synthesis_report:     str   # thematic synthesis grouped by topic/method/disagreement
    # Downstream agents read synthesis_report
    summary:              str
    activity_log:         Annotated[List, operator.add]
    current_agent:        str


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS AGENT
# ══════════════════════════════════════════════════════════════════════════════

def synthesis_agent(state: AgentState, model: ChatOpenAI) -> dict:
    """
    Layer 3 — sits between Critic and Summarizer.

    Reads:  state['merged_context'], state['extraction_findings'],
            state['knowledge_map']
    Writes: state['synthesis_report']

    Produces a structured thematic literature review with three sections:
      1. Thematic Groups    — papers clustered by research topic
      2. Methodological Landscape — methods compared across papers
      3. Disagreements & Open Questions — conflicts preserved, not resolved

    The Summarizer then uses synthesis_report (instead of raw merged_context)
    to write its final grounded answer.
    """
    merged     = state.get("merged_context", "")
    extraction = state.get("extraction_findings", "")
    km_nodes   = [n.get("label", "") for n in
                  state.get("knowledge_map", {}).get("nodes", [])]

    # Graceful skip if there is nothing to synthesise
    if not merged.strip() and not extraction.strip():
        return {
            "synthesis_report": "(no content to synthesise)",
            "messages":         [AIMessage(content="[Synthesis] skipped — no content")],
            "activity_log":     [{
                "agent":  "synthesis",
                "icon":   "🧵",
                "title":  "Synthesis Agent — skipped",
                "detail": "No merged context or extraction findings available.",
                "ts":     _stamp(),
            }],
            "current_agent": "synthesis",
        }

    system = SystemMessage(content=(
        "You are a Synthesis Agent in a multi-agent academic research assistant.\n"
        "Your job is to produce a DEDICATED THEMATIC LITERATURE REVIEW — NOT a summary.\n\n"
        "You receive:\n"
        "  • A merged evidence context from multiple retrieval channels\n"
        "  • Per-paper structured extraction records (problem / method / findings / limitations)\n"
        "  • Key concept labels from the knowledge graph\n\n"
        "Produce a synthesis report in markdown with EXACTLY these three sections:\n\n"
        "## 1. Thematic Groups\n"
        "Cluster all papers and findings into 2–5 coherent research themes. "
        "For each theme:\n"
        "  - Give it a descriptive name\n"
        "  - List which papers/sources belong to it\n"
        "  - Summarise the shared focus and key contributions\n"
        "  - Note trends across the group\n\n"
        "## 2. Methodological Landscape\n"
        "Compare the methods used across all papers. For each distinct methodology:\n"
        "  - Name it and describe it briefly\n"
        "  - List which papers use it\n"
        "  - Compare strengths and weaknesses across papers\n"
        "  - Note whether it is dominant, emerging, or rarely used\n\n"
        "## 3. Disagreements & Open Questions\n"
        "Identify where papers CONTRADICT, DISAGREE, or leave questions unresolved. "
        "For each disagreement:\n"
        "  - Describe what the conflict is about\n"
        "  - Name which papers hold each position\n"
        "  - DO NOT resolve the disagreement — preserve both viewpoints\n"
        "  - Flag it as an open research question\n\n"
        "RULES:\n"
        "- Ground every claim in the provided evidence — do not hallucinate\n"
        "- Cite sources as [VectorDB], [SQL], [Web], or [Extraction] per claim\n"
        "- If evidence is sparse for a section, say so rather than fabricating\n"
        "- This is NOT a summary — it is a structured thematic analysis\n"
    ))

    human_content = (
        f"Research query: {state['query']}\n\n"
        f"Merged evidence context:\n{merged[:3000]}\n\n"
    )
    if extraction.strip() and extraction.strip() != "(none)":
        human_content += f"Per-paper extraction records:\n{extraction[:2000]}\n\n"
    if km_nodes:
        human_content += f"Key concepts from knowledge graph: {', '.join(km_nodes[:25])}\n"

    resp = model.invoke([system, HumanMessage(content=human_content)])
    report = resp.content.strip()

    # Count sections for the activity log
    n_themes        = report.count("###") or report.count("**Theme")
    n_disagreements = report.lower().count("disagreement") + report.lower().count("conflict")

    return {
        "synthesis_report": report,
        "messages":         [AIMessage(content=f"[Synthesis] {report[:120]}…")],
        "activity_log":     [{
            "agent":  "synthesis",
            "icon":   "🧵",
            "title":  "Synthesis Agent — thematic literature review generated",
            "detail": (
                f"3 sections produced: Thematic Groups · Methodological Landscape · "
                f"Disagreements & Open Questions | {len(report):,} characters"
            ),
            "ts": _stamp(),
        }],
        "current_agent": "synthesis",
    }


# ══════════════════════════════════════════════════════════════════════════════
# MOCK SUMMARIZER — shows how Summarizer uses synthesis_report
# ══════════════════════════════════════════════════════════════════════════════

def mock_summarizer_agent(state: AgentState, model: ChatOpenAI) -> dict:
    """
    Simulates the Summarizer using synthesis_report instead of raw merged_context.
    In the full pipeline this is replaced by summarizer_agent() in streamlit_app.py.
    """
    system = SystemMessage(content=(
        "You are a Summarizer Agent. Write a clear, well-structured final answer "
        "grounded in the thematic synthesis report. "
        "Cite which theme or source each key claim comes from."
    ))
    synthesis = state.get("synthesis_report", "")
    resp = model.invoke([system, HumanMessage(content=(
        f"Query: {state['query']}\n\n"
        f"Thematic synthesis:\n{synthesis[:2000]}"
    ))])
    return {
        "summary":      resp.content,
        "messages":     [AIMessage(content=f"[Summarizer] {resp.content[:120]}…")],
        "activity_log": [{
            "agent":  "summarizer",
            "icon":   "✍️",
            "title":  "Summarizer — final answer (using synthesis)",
            "detail": f"{len(resp.content)} characters",
            "ts":     _stamp(),
        }],
        "current_agent": "summarizer",
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _stamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


# ══════════════════════════════════════════════════════════════════════════════
# MINI GRAPH — Synthesis → Summarizer → END
# ══════════════════════════════════════════════════════════════════════════════

def build_test_graph(api_key: str):
    model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.3)

    g = StateGraph(AgentState)
    g.add_node("synthesis",  lambda s: synthesis_agent(s, model))
    g.add_node("summarizer", lambda s: mock_summarizer_agent(s, model))

    g.set_entry_point("synthesis")
    g.add_edge("synthesis",  "summarizer")
    g.add_edge("summarizer", END)

    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA for testing without running the full pipeline
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_MERGED_CONTEXT = """
## Merged Context: RLHF vs Supervised Fine-Tuning

**Reinforcement Learning from Human Feedback (RLHF)**: A machine learning approach where
models learn from human evaluators, combining RL techniques with human feedback. [VectorDB]

**Traditional Supervised Fine-Tuning (SFT)**: A method where a pre-trained model is further
trained on labeled datasets, minimizing prediction errors. [VectorDB]

**Key Differences:**
1. Learning Paradigm: RLHF uses iterative reward maximization; SFT minimizes fixed loss. [SQL]
2. Data Requirements: RLHF needs human preference data; SFT needs labeled pairs. [VectorDB]
3. Flexibility: RLHF adapts to changing preferences; SFT is static. [VectorDB]

**Applications:** RLHF used in InstructGPT, ChatGPT, Claude. [SQL]
SFT used for classification, NER tasks. [SQL]

**Relationship:** Models are often SFT-trained first, then RLHF-aligned. [VectorDB]
"""

SAMPLE_EXTRACTION = """---
**Title / Topic:** InstructGPT (Ouyang et al. 2022)
**Provenance:** abstract-only
**Research Problem:** Align GPT-3 with human intent using RLHF.
**Methodology:** SFT on demonstrations, reward model training, PPO optimization.
**Key Findings:**
- RLHF-trained model preferred over 175B GPT-3 by human raters
- Significant reduction in harmful outputs
**Limitations:** Human labeler subjectivity; expensive to scale.
**Future Work:** Reduce human annotation cost; extend to other modalities.
---
---
**Title / Topic:** Stanford Alpaca (Taori et al. 2023)
**Provenance:** abstract-only
**Research Problem:** Can SFT alone produce instruction-following models cheaply?
**Methodology:** Fine-tune LLaMA on 52K GPT-4 generated instruction pairs.
**Key Findings:**
- Competitive with InstructGPT on many tasks using only SFT
- Cost: ~$600 total for data + training
**Limitations:** No human preference alignment; weaker on complex reasoning.
**Future Work:** Combine with RLHF for better alignment.
---
---
**Title / Topic:** Direct Preference Optimization (Rafailov et al. 2023)
**Provenance:** abstract-only
**Research Problem:** RLHF is unstable and complex — can we simplify it?
**Methodology:** Reformulates RLHF as a classification problem without explicit reward model.
**Key Findings:**
- Matches or exceeds RLHF performance on alignment benchmarks
- Simpler to implement, more stable training
**Limitations:** Still requires preference data; not tested at very large scale.
**Future Work:** Scale to larger models; combine with Constitutional AI.
---
"""

SAMPLE_KM = {
    "nodes": [
        {"id": "rlhf", "label": "RLHF", "type": "concept", "source": "sql_db"},
        {"id": "sft", "label": "Supervised Fine-Tuning", "type": "concept", "source": "vector_db"},
        {"id": "reward_model", "label": "Reward Model", "type": "concept", "source": "web"},
        {"id": "ppo", "label": "PPO Optimization", "type": "process", "source": "web"},
        {"id": "dpo", "label": "Direct Preference Optimization", "type": "concept", "source": "web"},
        {"id": "human_feedback", "label": "Human Feedback", "type": "concept", "source": "sql_db"},
        {"id": "alignment", "label": "Model Alignment", "type": "concept", "source": "merged"},
        {"id": "instructgpt", "label": "InstructGPT", "type": "entity", "source": "sql_db"},
    ],
    "edges": [
        {"source": "rlhf", "target": "reward_model", "relation": "uses", "weight": 0.9},
        {"source": "rlhf", "target": "ppo", "relation": "optimizes_with", "weight": 0.8},
        {"source": "dpo", "target": "rlhf", "relation": "simplifies", "weight": 0.9},
        {"source": "sft", "target": "rlhf", "relation": "precedes", "weight": 0.7},
    ]
}


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT TEST UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Synthesis Agent Test", page_icon="🧵", layout="wide")
st.title("🧵 Synthesis Agent — Standalone Test")
st.caption(
    "Tests the Synthesis Agent in isolation before integrating into the full pipeline. "
    "Pipeline position: Orchestrator → Knowledge Mapper → Critic → **[Synthesis]** → Summarizer"
)

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
    **Input:**
    - Merged evidence context
    - Per-paper extraction records
    - Knowledge map nodes

    **Output — 3 sections:**
    1. 🗂️ **Thematic Groups** — papers clustered by topic
    2. 🔬 **Methodological Landscape** — methods compared
    3. ⚡ **Disagreements** — conflicts preserved

    **Pipeline position:**
    `Critic → Synthesis → Summarizer`
    """)
    st.divider()
    use_sample = st.checkbox("Use sample data (no pipeline needed)", value=True)

st.divider()

# ── Query Input ───────────────────────────────────────────────────────────────
query = st.text_area(
    "Research query",
    height=80,
    value="What are the key differences between RLHF and traditional supervised fine-tuning?",
    placeholder="Enter your research query…"
)

# ── Context Inputs ────────────────────────────────────────────────────────────
st.markdown("### Input Context")
st.caption("Paste your merged context and extraction findings, or use sample data.")

col1, col2 = st.columns(2)
with col1:
    merged_ctx = st.text_area(
        "Merged Context (from Orchestrator)",
        height=200,
        value=SAMPLE_MERGED_CONTEXT if use_sample else "",
        placeholder="Paste merged_context here…"
    )
with col2:
    extraction = st.text_area(
        "Extraction Findings (from Reading Agent)",
        height=200,
        value=SAMPLE_EXTRACTION if use_sample else "",
        placeholder="Paste extraction_findings here…"
    )

run = st.button("🧵 Run Synthesis Agent", type="primary",
                disabled=not (query and (merged_ctx or extraction)))

if run:
    with st.spinner("Running Synthesis → Summarizer…"):
        app = build_test_graph(api_key)

        full_state = {
            "messages":            [],
            "query":               query,
            "merged_context":      merged_ctx,
            "extraction_findings": extraction,
            "knowledge_map":       SAMPLE_KM if use_sample else {},
            "synthesis_report":    "",
            "summary":             "",
            "activity_log":        [],
            "current_agent":       "",
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
    tab_synth, tab_summary, tab_compare, tab_integrate = st.tabs([
        "🧵 Synthesis Report", "✍️ Final Answer", "🔍 Before vs After", "🔗 Integration Guide"
    ])

    with tab_synth:
        report = full_state.get("synthesis_report", "")
        if report and report != "(no content to synthesise)":

            # Section badges
            sections = ["Thematic Groups", "Methodological Landscape", "Disagreements"]
            found = [s for s in sections if s in report]
            cols = st.columns(len(found))
            for i, s in enumerate(found):
                cols[i].success(f"✅ {s}")

            st.divider()
            st.markdown(report)
            st.divider()
            st.metric("Report length", f"{len(report):,} chars")
        else:
            st.warning("No synthesis report generated — check your input context.")

    with tab_summary:
        st.markdown("### Final Answer (Summarizer using Synthesis output)")
        st.markdown(full_state.get("summary", ""))
        st.divider()
        st.caption(
            "Notice how the final answer now references thematic groups and "
            "methodological comparisons rather than just raw findings."
        )

    with tab_compare:
        st.markdown("### Before vs After adding the Synthesis Agent")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**❌ Without Synthesis Agent**")
            st.markdown("""
            The Summarizer receives raw merged context:
            ```
            [VectorDB] RLHF uses reward models...
            [SQL] InstructGPT uses PPO...
            [Web] DPO simplifies RLHF...
            ```
            Result: A list of facts, not a thematic analysis.
            Missing: theme groupings, method comparisons, disagreements.
            """)
        with c2:
            st.markdown("**✅ With Synthesis Agent**")
            st.markdown("""
            The Summarizer receives a structured thematic review:
            ```
            ## Thematic Groups
            Theme 1: Alignment via Human Feedback
            Theme 2: Efficient Fine-Tuning

            ## Methodological Landscape
            RLHF vs SFT vs DPO compared...

            ## Disagreements
            Papers disagree on whether RLHF
            is necessary or if SFT suffices...
            ```
            Result: A coherent thematic synthesis with conflict identification.
            """)

    with tab_integrate:
        st.subheader("How to integrate into streamlit_app.py")
        st.markdown("""
        Your teammates' `streamlit_app.py` already has the full pipeline. Adding the
        Synthesis Agent requires **5 changes**:

        **1. Add new field to `AgentState`:**
        ```python
        synthesis_report: str   # written by synthesis_agent, read by summarizer
        ```

        **2. Copy the `synthesis_agent()` function** from this file into `streamlit_app.py`

        **3. Update `build_graph()` to wire the new node:**
        ```python
        g.add_node("synthesis", lambda s: synthesis_agent(s, lm_y))
        # Change: critic → summarizer becomes critic → synthesis → summarizer
        g.add_conditional_edges(
            "critic", _route_critic,
            {"orchestrator": "orchestrator", "summarizer": "synthesis"},  # ← changed
        )
        g.add_edge("synthesis", "summarizer")   # ← new edge
        ```

        **4. Update `summarizer_agent()` to read `synthesis_report`:**
        ```python
        # In summarizer_agent, change the human message to use synthesis_report:
        synthesis = state.get("synthesis_report", state.get("merged_context", ""))
        resp = model.invoke([system, HumanMessage(content=(
            f"Query: {state['query']}\\n\\n"
            f"Thematic synthesis:\\n{synthesis}\\n\\n"
            f"Key concepts: {[n['label'] for n in state['knowledge_map'].get('nodes',[])]}"
        ))])
        ```

        **5. Update `pct_map` and `full_state` init:**
        ```python
        pct_map = {
            ...
            "critic": 85,
            "synthesis": 91,      # ← add this
            "summarizer": 95,
            "experiment_design": 98,
        }

        full_state = {
            ...existing fields...,
            "synthesis_report": "",   # ← add this
        }
        ```

        **6. Add to Per-Agent Findings tab:**
        ```python
        ("🧵 Synthesis", "synthesis_report", "bs"),
        ```

        **7. Add to Message Log `av` dict:**
        ```python
        "[Synthesis]": "🧵",
        ```
        """)

        st.divider()
        st.subheader("Where it fits in the new pipeline")
        st.code("""
Router → VectorDB → SQL → Web → Reading/Extraction
      → Orchestrator → Knowledge Mapper → Critic
      → [Synthesis]   ← YOUR AGENT GOES HERE
      → Summarizer → Experiment Design → END
        """, language="text")
