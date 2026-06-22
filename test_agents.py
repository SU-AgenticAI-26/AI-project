"""
Correctness and functionality tests for all agents, routing logic,
SQL search, cache helpers, and utility functions in streamlit_app.py.

Run with:
    python -m pytest test_agents.py -v

No OpenAI API key or network access required — all LLM and external calls
are mocked.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Import shims — identical to the ones in test_reading_extraction.py
# ─────────────────────────────────────────────────────────────────────────────

def _patch_imports() -> None:
    if "streamlit_app" in sys.modules:
        return  # already patched by another test module in this session

    mocks: dict = {}

    # streamlit
    st_mock = MagicMock()
    st_mock.button.return_value = False
    st_mock.form_submit_button.return_value = False
    st_mock.text_input.return_value = ""
    st_mock.text_area.return_value = ""
    st_mock.number_input.return_value = 5
    st_mock.checkbox.return_value = False
    st_mock.file_uploader.return_value = None
    st_mock.selectbox.return_value = "OpenAI"
    st_mock.tabs.side_effect = lambda labels: [MagicMock() for _ in labels]
    st_mock.columns.side_effect = (
        lambda n: [MagicMock() for _ in (range(n) if isinstance(n, int) else n)]
    )
    st_mock.session_state.__contains__ = lambda self, key: False
    mocks["streamlit"] = st_mock

    # pyvis
    pyvis_mod = types.ModuleType("pyvis")
    pyvis_net = types.ModuleType("pyvis.network")
    pyvis_net.Network = MagicMock()
    pyvis_mod.network = pyvis_net
    mocks["pyvis"] = pyvis_mod
    mocks["pyvis.network"] = pyvis_net

    # langchain_core
    class _Msg:
        def __init__(self, content: str = "", **_kw):
            self.content = content

    class _Doc:
        def __init__(self, page_content: str = "", metadata: dict | None = None, **_kw):
            self.page_content = page_content
            self.metadata = metadata or {}

    lc_core      = types.ModuleType("langchain_core")
    lc_core_docs = types.ModuleType("langchain_core.documents")
    lc_core_msgs = types.ModuleType("langchain_core.messages")
    lc_core_llms = types.ModuleType("langchain_core.language_models")
    lc_core_docs.Document      = _Doc
    lc_core_msgs.AIMessage     = _Msg
    lc_core_msgs.HumanMessage  = _Msg
    lc_core_msgs.SystemMessage = _Msg
    lc_core_llms.BaseChatModel = MagicMock
    lc_core.documents          = lc_core_docs
    lc_core.messages           = lc_core_msgs
    lc_core.language_models    = lc_core_llms
    mocks["langchain_core"]                 = lc_core
    mocks["langchain_core.documents"]       = lc_core_docs
    mocks["langchain_core.messages"]        = lc_core_msgs
    mocks["langchain_core.language_models"] = lc_core_llms

    for mod in [
        "langchain_openai",
        "langchain_community",
        "langchain_community.vectorstores",
        "langchain_community.vectorstores.FAISS",
        "langchain_community.embeddings",
        "langchain_text_splitters",
        "faiss",
    ]:
        mocks[mod] = MagicMock()
    mocks["langchain_community.vectorstores"].FAISS.load_local.side_effect = Exception("mock")

    # langgraph
    langgraph_mod   = types.ModuleType("langgraph")
    langgraph_graph = types.ModuleType("langgraph.graph")
    langgraph_graph.END = "END"

    class _FakeStateGraph:
        def __init__(self, *a, **kw): pass
        def add_node(self, *a, **kw): pass
        def add_edge(self, *a, **kw): pass
        def add_conditional_edges(self, *a, **kw): pass
        def set_entry_point(self, *a, **kw): pass
        def compile(self): return MagicMock()

    langgraph_graph.StateGraph = _FakeStateGraph
    langgraph_mod.graph = langgraph_graph
    mocks["langgraph"] = langgraph_mod
    mocks["langgraph.graph"] = langgraph_graph

    for key, val in mocks.items():
        sys.modules[key] = val


_patch_imports()
import streamlit_app as app  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_state(**overrides) -> app.AgentState:
    base: app.AgentState = {
        "messages":            [],
        "query":               "What is retrieval-augmented generation?",
        "active_agents":       ["vector_db", "sql_db", "web"],
        "router_reasoning":    "",
        "vector_findings":     "",
        "sql_findings":        "",
        "web_findings":        "",
        "extraction_findings": "",
        "activity_log":        [],
        "merged_context":      "",
        "knowledge_map":       {},
        "critique":            "",
        "loop_count":          0,
        "summary":             "",
        "experiment_plan":     "",
        "current_agent":       "",
        "_needs_more":         False,
    }
    base.update(overrides)
    return base


def _mock_llm(response_text: str) -> MagicMock:
    model = MagicMock()
    msg   = MagicMock()
    msg.content = response_text
    model.invoke.return_value = msg
    return model


def _mock_vdb(docs=None) -> MagicMock:
    """Return a mock VectorDBModule whose search() returns `docs`."""
    vdb = MagicMock()
    vdb.search.return_value = docs or []
    return vdb


# ═════════════════════════════════════════════════════════════════════════════
# Router Agent
# ═════════════════════════════════════════════════════════════════════════════

class TestRouterAgent(unittest.TestCase):

    def test_sql_trigger_patterns_force_sql_agent(self):
        """Enumeration/categorical queries should force sql_db even if LLM omits it."""
        state = _make_state(
            query="What are the main approaches and challenges in federated learning for healthcare applications?"
        )
        resp = json.dumps({"agents": ["vector_db", "web"], "reasoning": "recent research focus"})
        result = app.router_agent(state, _mock_llm(resp))
        self.assertIn("sql_db", result["active_agents"])

    def test_non_trigger_query_does_not_force_sql_agent(self):
        """Queries without SQL trigger patterns should not add sql_db automatically."""
        state = _make_state(query="Explain transformer attention with a simple example")
        resp = json.dumps({"agents": ["vector_db"], "reasoning": "semantic context is enough"})
        result = app.router_agent(state, _mock_llm(resp))
        self.assertEqual(result["active_agents"], ["vector_db"])

    def test_parses_all_three_agents(self):
        """Router correctly extracts all three agent names from JSON."""
        resp = json.dumps({"agents": ["vector_db", "sql_db", "web"], "reasoning": "all needed"})
        result = app.router_agent(_make_state(), _mock_llm(resp))
        self.assertEqual(result["active_agents"], ["vector_db", "sql_db", "web"])

    def test_parses_partial_agent_list(self):
        """Router can return a subset of agents."""
        resp = json.dumps({"agents": ["vector_db"], "reasoning": "only semantic search needed"})
        result = app.router_agent(_make_state(), _mock_llm(resp))
        self.assertEqual(result["active_agents"], ["vector_db"])

    def test_reasoning_extracted(self):
        """Router stores the reasoning string in the returned dict."""
        resp = json.dumps({"agents": ["sql_db"], "reasoning": "structured facts only"})
        result = app.router_agent(_make_state(), _mock_llm(resp))
        self.assertEqual(result["router_reasoning"], "structured facts only")

    def test_fallback_on_invalid_json(self):
        """Router defaults to [vector_db, sql_db] when LLM returns garbage."""
        result = app.router_agent(_make_state(), _mock_llm("not json at all"))
        self.assertEqual(result["active_agents"], ["vector_db", "sql_db"])

    def test_strips_markdown_fences(self):
        """Router handles LLM output wrapped in ```json ... ``` fences."""
        inner = json.dumps({"agents": ["web"], "reasoning": "live search needed"})
        fenced = f"```json\n{inner}\n```"
        result = app.router_agent(_make_state(), _mock_llm(fenced))
        self.assertEqual(result["active_agents"], ["web"])

    def test_current_agent_is_router(self):
        result = app.router_agent(_make_state(), _mock_llm('{"agents":[],"reasoning":""}'))
        self.assertEqual(result["current_agent"], "router")

    def test_activity_log_has_required_fields(self):
        resp = json.dumps({"agents": ["vector_db"], "reasoning": "r"})
        result = app.router_agent(_make_state(), _mock_llm(resp))
        entry = result["activity_log"][0]
        self.assertEqual(entry["agent"], "router")
        self.assertEqual(entry["icon"], "🔀")
        self.assertIn("title", entry)
        self.assertIn("detail", entry)
        self.assertIn("ts", entry)

    def test_messages_list_non_empty(self):
        resp = json.dumps({"agents": ["vector_db"], "reasoning": "r"})
        result = app.router_agent(_make_state(), _mock_llm(resp))
        self.assertGreater(len(result["messages"]), 0)


# ═════════════════════════════════════════════════════════════════════════════
# Vector DB Agent
# ═════════════════════════════════════════════════════════════════════════════

class TestVectorDbAgent(unittest.TestCase):

    def test_skips_when_not_activated(self):
        state  = _make_state(active_agents=["sql_db", "web"])
        result = app.vector_db_agent(state, _mock_llm("x"), _mock_vdb())
        self.assertEqual(result["vector_findings"], "(not activated)")
        self.assertEqual(result["current_agent"], "vector_db")

    def test_skip_does_not_call_llm(self):
        model  = _mock_llm("should not be called")
        state  = _make_state(active_agents=[])
        app.vector_db_agent(state, model, _mock_vdb())
        model.invoke.assert_not_called()

    def test_returns_llm_response_as_findings(self):
        state  = _make_state(active_agents=["vector_db"])
        result = app.vector_db_agent(state, _mock_llm("VECTOR_RESPONSE"), _mock_vdb())
        self.assertEqual(result["vector_findings"], "VECTOR_RESPONSE")

    def test_searches_vdb_with_query(self):
        vdb   = _mock_vdb()
        state = _make_state(active_agents=["vector_db"], query="my query")
        app.vector_db_agent(state, _mock_llm("r"), vdb)
        vdb.search.assert_called_once()
        call_args = vdb.search.call_args
        self.assertIn("my query", call_args[0])

    def test_activity_log_shows_chunk_count_when_docs_returned(self):
        doc = MagicMock()
        doc.page_content = "some text"
        doc.metadata = {"source": "test.pdf"}
        state  = _make_state(active_agents=["vector_db"])
        result = app.vector_db_agent(state, _mock_llm("r"), _mock_vdb([doc]))
        detail = result["activity_log"][0]["detail"]
        self.assertIn("1", detail)

    def test_activity_log_when_no_docs(self):
        state  = _make_state(active_agents=["vector_db"])
        result = app.vector_db_agent(state, _mock_llm("r"), _mock_vdb([]))
        detail = result["activity_log"][0]["detail"]
        self.assertIn("0", detail)

    def test_llm_receives_query_in_prompt(self):
        model = _mock_llm("r")
        state = _make_state(active_agents=["vector_db"], query="RAG architecture")
        app.vector_db_agent(state, model, _mock_vdb())
        human_content = model.invoke.call_args[0][0][-1].content
        self.assertIn("RAG architecture", human_content)


# ═════════════════════════════════════════════════════════════════════════════
# SQL DB Agent
# ═════════════════════════════════════════════════════════════════════════════

class TestSqlDbAgent(unittest.TestCase):

    def test_skips_when_not_activated(self):
        state  = _make_state(active_agents=["vector_db", "web"])
        result = app.sql_db_agent(state, _mock_llm("x"))
        self.assertEqual(result["sql_findings"], "(not activated)")
        self.assertEqual(result["current_agent"], "sql_db")

    def test_skip_does_not_call_llm(self):
        model  = _mock_llm("should not be called")
        state  = _make_state(active_agents=[])
        app.sql_db_agent(state, model)
        model.invoke.assert_not_called()

    def test_returns_llm_response_as_findings(self):
        state  = _make_state(active_agents=["sql_db"])
        result = app.sql_db_agent(state, _mock_llm("SQL_RESPONSE"))
        self.assertEqual(result["sql_findings"], "SQL_RESPONSE")

    def test_activity_log_structure(self):
        state  = _make_state(active_agents=["sql_db"])
        result = app.sql_db_agent(state, _mock_llm("r"))
        entry  = result["activity_log"][0]
        self.assertEqual(entry["agent"], "sql_db")
        self.assertEqual(entry["icon"], "🗄️")
        self.assertIn("rows", entry)

    def test_llm_receives_query_in_prompt(self):
        model = _mock_llm("r")
        state = _make_state(active_agents=["sql_db"], query="transformer attention")
        app.sql_db_agent(state, model)
        human_content = model.invoke.call_args[0][0][-1].content
        self.assertIn("transformer attention", human_content)


# ═════════════════════════════════════════════════════════════════════════════
# Web Agent
# ═════════════════════════════════════════════════════════════════════════════

class TestWebAgent(unittest.TestCase):

    def test_skips_when_not_activated(self):
        state  = _make_state(active_agents=["vector_db", "sql_db"])
        result = app.web_agent(state, _mock_llm("x"), _mock_vdb())
        self.assertEqual(result["web_findings"], "(not activated)")
        self.assertEqual(result["current_agent"], "web")

    def test_skip_does_not_call_llm(self):
        model = _mock_llm("should not be called")
        state = _make_state(active_agents=[])
        app.web_agent(state, model, _mock_vdb())
        model.invoke.assert_not_called()

    def test_returns_llm_response_as_findings(self):
        """With all HTTP calls patched to fail gracefully, LLM response is returned."""
        state  = _make_state(active_agents=["web"])
        result = app.web_agent(state, _mock_llm("WEB_RESPONSE"), _mock_vdb())
        self.assertEqual(result["web_findings"], "WEB_RESPONSE")

    def test_activity_log_structure(self):
        state  = _make_state(active_agents=["web"])
        result = app.web_agent(state, _mock_llm("r"), _mock_vdb())
        entry  = result["activity_log"][0]
        self.assertEqual(entry["agent"], "web")
        self.assertEqual(entry["icon"], "🌐")
        self.assertIn("ts", entry)

    def test_current_agent_set(self):
        state  = _make_state(active_agents=["web"])
        result = app.web_agent(state, _mock_llm("r"), _mock_vdb())
        self.assertEqual(result["current_agent"], "web")


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator Agent
# ═════════════════════════════════════════════════════════════════════════════

class TestOrchestratorAgent(unittest.TestCase):

    def _run(self, **state_overrides) -> tuple[dict, MagicMock]:
        model  = _mock_llm("MERGED_CONTEXT_OUTPUT")
        state  = _make_state(**state_overrides)
        result = app.orchestrator_agent(state, model)
        return result, model

    def test_merged_context_set_from_llm(self):
        result, _ = self._run(vector_findings="v", sql_findings="s", web_findings="w")
        self.assertEqual(result["merged_context"], "MERGED_CONTEXT_OUTPUT")

    def test_all_four_sections_sent_to_llm(self):
        model = _mock_llm("out")
        state = _make_state(
            vector_findings="VF_CONTENT",
            sql_findings="SF_CONTENT",
            web_findings="WF_CONTENT",
            extraction_findings="EF_CONTENT",
        )
        app.orchestrator_agent(state, model)
        human_content = model.invoke.call_args[0][0][-1].content
        for sentinel in ("VF_CONTENT", "SF_CONTENT", "WF_CONTENT", "EF_CONTENT"):
            self.assertIn(sentinel, human_content)

    def test_active_sources_in_activity_log(self):
        result, _ = self._run(
            vector_findings="real content",
            sql_findings="(not activated)",
            web_findings="real web content",
            extraction_findings="",
        )
        detail = result["activity_log"][0]["detail"]
        self.assertIn("Vector DB", detail)
        self.assertNotIn("SQL DB", detail)
        self.assertIn("Web", detail)

    def test_empty_findings_not_counted_as_active(self):
        result, _ = self._run(
            vector_findings="",
            sql_findings="",
            web_findings="",
            extraction_findings="",
        )
        detail = result["activity_log"][0]["detail"]
        self.assertIn("none", detail.lower())

    def test_current_agent_set(self):
        result, _ = self._run()
        self.assertEqual(result["current_agent"], "orchestrator")

    def test_query_sent_to_llm(self):
        model = _mock_llm("out")
        state = _make_state(query="UNIQUE_QUERY_SENTINEL", vector_findings="v")
        app.orchestrator_agent(state, model)
        human_content = model.invoke.call_args[0][0][-1].content
        self.assertIn("UNIQUE_QUERY_SENTINEL", human_content)


# ═════════════════════════════════════════════════════════════════════════════
# Knowledge Mapper Agent
# ═════════════════════════════════════════════════════════════════════════════

_KM_VALID_JSON = json.dumps({
    "nodes": [
        {"id": "RAG", "label": "RAG", "type": "concept", "source": "web"},
        {"id": "Transformer", "label": "Transformer", "type": "concept", "source": "vector_db"},
    ],
    "edges": [
        {"source": "RAG", "target": "Transformer", "relation": "uses", "weight": 0.9}
    ],
})


class TestKnowledgeMapperAgent(unittest.TestCase):

    def test_parses_valid_json(self):
        result = app.knowledge_mapper_agent(
            _make_state(merged_context="ctx"), _mock_llm(_KM_VALID_JSON)
        )
        self.assertEqual(len(result["knowledge_map"]["nodes"]), 2)
        self.assertEqual(len(result["knowledge_map"]["edges"]), 1)

    def test_fallback_on_invalid_json(self):
        result = app.knowledge_mapper_agent(
            _make_state(merged_context="ctx"), _mock_llm("not json")
        )
        km = result["knowledge_map"]
        self.assertEqual(km["nodes"], [])
        self.assertEqual(km["edges"], [])
        self.assertEqual(km.get("error"), "parse_failed")

    def test_strips_markdown_code_fences(self):
        fenced = f"```json\n{_KM_VALID_JSON}\n```"
        result = app.knowledge_mapper_agent(
            _make_state(merged_context="ctx"), _mock_llm(fenced)
        )
        self.assertEqual(len(result["knowledge_map"]["nodes"]), 2)

    def test_activity_log_node_edge_counts(self):
        result = app.knowledge_mapper_agent(
            _make_state(merged_context="ctx"), _mock_llm(_KM_VALID_JSON)
        )
        detail = result["activity_log"][0]["detail"]
        self.assertIn("2", detail)   # 2 nodes
        self.assertIn("1", detail)   # 1 edge

    def test_current_agent_set(self):
        result = app.knowledge_mapper_agent(
            _make_state(merged_context="ctx"), _mock_llm(_KM_VALID_JSON)
        )
        self.assertEqual(result["current_agent"], "knowledge_mapper")

    def test_merged_context_sent_to_llm(self):
        model = _mock_llm(_KM_VALID_JSON)
        state = _make_state(merged_context="UNIQUE_CTX_SENTINEL")
        app.knowledge_mapper_agent(state, model)
        human_content = model.invoke.call_args[0][0][-1].content
        self.assertIn("UNIQUE_CTX_SENTINEL", human_content)


# ═════════════════════════════════════════════════════════════════════════════
# Critic Agent
# ═════════════════════════════════════════════════════════════════════════════

_NEEDS_MORE_JSON  = json.dumps({"needs_more": True,  "feedback": "too few nodes"})
_APPROVED_JSON    = json.dumps({"needs_more": False, "feedback": ""})
_KM_WITH_NODES = {
    "nodes": [{"id": str(i), "label": f"node{i}", "type": "concept", "source": "web"}
              for i in range(10)],
    "edges": [{"source": "0", "target": "1", "relation": "r", "weight": 0.5}],
}


class TestCriticAgent(unittest.TestCase):

    def test_needs_more_true_when_json_says_so(self):
        state  = _make_state(knowledge_map=_KM_WITH_NODES)
        result = app.critic_agent(state, _mock_llm(_NEEDS_MORE_JSON))
        self.assertTrue(result["_needs_more"])

    def test_needs_more_false_when_json_says_so(self):
        state  = _make_state(knowledge_map=_KM_WITH_NODES)
        result = app.critic_agent(state, _mock_llm(_APPROVED_JSON))
        self.assertFalse(result["_needs_more"])

    def test_feedback_stored_in_critique(self):
        state  = _make_state(knowledge_map=_KM_WITH_NODES)
        result = app.critic_agent(state, _mock_llm(_NEEDS_MORE_JSON))
        self.assertEqual(result["critique"], "too few nodes")

    def test_loop_count_incremented(self):
        state  = _make_state(knowledge_map=_KM_WITH_NODES, loop_count=1)
        result = app.critic_agent(state, _mock_llm(_APPROVED_JSON))
        self.assertEqual(result["loop_count"], 2)

    def test_loop_count_starts_from_zero(self):
        state  = _make_state(knowledge_map=_KM_WITH_NODES, loop_count=0)
        result = app.critic_agent(state, _mock_llm(_APPROVED_JSON))
        self.assertEqual(result["loop_count"], 1)

    def test_fallback_on_invalid_json(self):
        """Critic defaults to needs_more=False on malformed LLM output."""
        state  = _make_state(knowledge_map=_KM_WITH_NODES)
        result = app.critic_agent(state, _mock_llm("definitely not json"))
        self.assertFalse(result["_needs_more"])
        self.assertEqual(result["critique"], "")

    def test_strips_markdown_fences(self):
        fenced = f"```json\n{_NEEDS_MORE_JSON}\n```"
        state  = _make_state(knowledge_map=_KM_WITH_NODES)
        result = app.critic_agent(state, _mock_llm(fenced))
        self.assertTrue(result["_needs_more"])

    def test_activity_log_title_reflects_needs_more(self):
        state   = _make_state(knowledge_map=_KM_WITH_NODES)
        result  = app.critic_agent(state, _mock_llm(_NEEDS_MORE_JSON))
        title   = result["activity_log"][0]["title"].lower()
        self.assertIn("enrich", title)

    def test_activity_log_title_reflects_approved(self):
        state   = _make_state(knowledge_map=_KM_WITH_NODES)
        result  = app.critic_agent(state, _mock_llm(_APPROVED_JSON))
        title   = result["activity_log"][0]["title"].lower()
        self.assertIn("approv", title)

    def test_current_agent_set(self):
        state  = _make_state(knowledge_map=_KM_WITH_NODES)
        result = app.critic_agent(state, _mock_llm(_APPROVED_JSON))
        self.assertEqual(result["current_agent"], "critic")


# ═════════════════════════════════════════════════════════════════════════════
# Summarizer Agent
# ═════════════════════════════════════════════════════════════════════════════

class TestSummarizerAgent(unittest.TestCase):

    def test_summary_set_from_llm(self):
        state  = _make_state(merged_context="ctx", knowledge_map=_KM_WITH_NODES)
        result = app.summarizer_agent(state, _mock_llm("THE_SUMMARY"))
        self.assertEqual(result["summary"], "THE_SUMMARY")

    def test_current_agent_set(self):
        state  = _make_state(merged_context="ctx", knowledge_map=_KM_WITH_NODES)
        result = app.summarizer_agent(state, _mock_llm("s"))
        self.assertEqual(result["current_agent"], "summarizer")

    def test_query_in_prompt(self):
        model = _mock_llm("s")
        state = _make_state(
            merged_context="ctx",
            knowledge_map=_KM_WITH_NODES,
            query="UNIQUE_SUMMARIZER_QUERY",
        )
        app.summarizer_agent(state, model)
        human_content = model.invoke.call_args[0][0][-1].content
        self.assertIn("UNIQUE_SUMMARIZER_QUERY", human_content)

    def test_merged_context_in_prompt(self):
        model = _mock_llm("s")
        state = _make_state(merged_context="UNIQUE_MERGED_CTX", knowledge_map={})
        app.summarizer_agent(state, model)
        human_content = model.invoke.call_args[0][0][-1].content
        self.assertIn("UNIQUE_MERGED_CTX", human_content)

    def test_knowledge_map_nodes_in_prompt(self):
        model = _mock_llm("s")
        km    = {"nodes": [{"id": "n1", "label": "UNIQUE_NODE_LABEL", "type": "c", "source": "web"}], "edges": []}
        state = _make_state(merged_context="ctx", knowledge_map=km)
        app.summarizer_agent(state, model)
        human_content = model.invoke.call_args[0][0][-1].content
        self.assertIn("UNIQUE_NODE_LABEL", human_content)

    def test_activity_log_reports_char_count(self):
        state  = _make_state(merged_context="ctx", knowledge_map={})
        result = app.summarizer_agent(state, _mock_llm("hello world"))
        detail = result["activity_log"][0]["detail"]
        self.assertIn("11", detail)   # len("hello world") == 11


# ═════════════════════════════════════════════════════════════════════════════
# Experiment Design Agent
# ═════════════════════════════════════════════════════════════════════════════

_PLAN_TEXT = (
    "## Research Landscape Overview\nSome overview.\n"
    "## Identified Research Gaps\n**Gap 1: Missing X** — desc\n*Grounded in: Paper A*\n"
    "**Gap 2: Missing Y** — desc\n*Grounded in: Paper B*\n"
    "## Proposed Hypotheses\n**H-1** *(addresses Gap 1)*: hypothesis A\n"
    "**H-2** *(addresses Gap 2)*: hypothesis B\n"
    "## Recommended Methodologies\nMethods.\n"
    "## Datasets & Domains\nDatasets.\n"
    "## Anticipated Challenges & Risks\nRisks.\n"
    "## Short-term Next Steps (0–3 months)\n1. Step A\n"
    "## Medium-term Next Steps (3–12 months)\n1. Milestone A\n"
)


class TestExperimentDesignAgent(unittest.TestCase):

    def test_experiment_plan_set_from_llm(self):
        state  = _make_state(summary="sum", merged_context="ctx")
        result = app.experiment_design_agent(state, _mock_llm(_PLAN_TEXT))
        self.assertEqual(result["experiment_plan"], _PLAN_TEXT)

    def test_current_agent_set(self):
        state  = _make_state(summary="sum", merged_context="ctx")
        result = app.experiment_design_agent(state, _mock_llm(_PLAN_TEXT))
        self.assertEqual(result["current_agent"], "experiment_design")

    def test_gap_count_in_activity_log(self):
        state  = _make_state(summary="sum", merged_context="ctx")
        result = app.experiment_design_agent(state, _mock_llm(_PLAN_TEXT))
        detail = result["activity_log"][0]["detail"]
        self.assertIn("2", detail)   # 2 gaps

    def test_hypothesis_count_in_activity_log(self):
        state  = _make_state(summary="sum", merged_context="ctx")
        result = app.experiment_design_agent(state, _mock_llm(_PLAN_TEXT))
        detail = result["activity_log"][0]["detail"]
        self.assertIn("2", detail)   # 2 hypotheses

    def test_extraction_findings_sent_to_llm(self):
        model = _mock_llm(_PLAN_TEXT)
        state = _make_state(
            summary="sum",
            merged_context="ctx",
            extraction_findings="UNIQUE_EXTRACTION_SENTINEL",
        )
        app.experiment_design_agent(state, model)
        human_content = model.invoke.call_args[0][0][-1].content
        self.assertIn("UNIQUE_EXTRACTION_SENTINEL", human_content)

    def test_summary_sent_to_llm(self):
        model = _mock_llm(_PLAN_TEXT)
        state = _make_state(summary="UNIQUE_SUMMARY_SENTINEL", merged_context="ctx")
        app.experiment_design_agent(state, model)
        human_content = model.invoke.call_args[0][0][-1].content
        self.assertIn("UNIQUE_SUMMARY_SENTINEL", human_content)

    def test_activity_log_icon(self):
        state  = _make_state(summary="s", merged_context="c")
        result = app.experiment_design_agent(state, _mock_llm(_PLAN_TEXT))
        self.assertEqual(result["activity_log"][0]["icon"], "🧪")


# ═════════════════════════════════════════════════════════════════════════════
# _route_critic
# ═════════════════════════════════════════════════════════════════════════════

class TestRouteCritic(unittest.TestCase):

    def test_routes_to_orchestrator_when_needs_more_and_loop_zero(self):
        state = _make_state(_needs_more=True, loop_count=0)
        self.assertEqual(app._route_critic(state), "orchestrator")

    def test_routes_to_orchestrator_when_needs_more_and_loop_one(self):
        state = _make_state(_needs_more=True, loop_count=1)
        self.assertEqual(app._route_critic(state), "orchestrator")

    def test_routes_to_summarizer_when_loop_count_reaches_limit(self):
        """At loop_count >= 2 the critic must stop looping regardless of needs_more."""
        state = _make_state(_needs_more=True, loop_count=2)
        self.assertEqual(app._route_critic(state), "summarizer")

    def test_routes_to_summarizer_when_loop_count_exceeds_limit(self):
        state = _make_state(_needs_more=True, loop_count=5)
        self.assertEqual(app._route_critic(state), "summarizer")

    def test_routes_to_summarizer_when_not_needs_more(self):
        state = _make_state(_needs_more=False, loop_count=0)
        self.assertEqual(app._route_critic(state), "summarizer")

    def test_routes_to_summarizer_when_needs_more_is_none(self):
        """Missing _needs_more key should default to summarizer (safe path)."""
        state = _make_state()
        state.pop("_needs_more", None)
        self.assertEqual(app._route_critic(state), "summarizer")


# ═════════════════════════════════════════════════════════════════════════════
# _has_content utility
# ═════════════════════════════════════════════════════════════════════════════

class TestHasContent(unittest.TestCase):

    def test_empty_string_is_not_content(self):
        self.assertFalse(app._has_content(""))

    def test_whitespace_only_is_not_content(self):
        self.assertFalse(app._has_content("   "))

    def test_not_activated_is_not_content(self):
        self.assertFalse(app._has_content("(not activated)"))

    def test_none_sentinel_is_not_content(self):
        self.assertFalse(app._has_content("(none)"))

    def test_no_sql_results_is_not_content(self):
        self.assertFalse(app._has_content("(no SQL results)"))

    def test_real_text_is_content(self):
        self.assertTrue(app._has_content("This is a real finding about RAG."))

    def test_single_word_is_content(self):
        self.assertTrue(app._has_content("transformer"))

    def test_padded_placeholder_is_not_content(self):
        """Leading/trailing spaces around a placeholder should still be non-content."""
        self.assertFalse(app._has_content("  (not activated)  "))


# ═════════════════════════════════════════════════════════════════════════════
# estimate_tokens utility
# ═════════════════════════════════════════════════════════════════════════════

class TestEstimateTokens(unittest.TestCase):

    def test_empty_string_returns_zero(self):
        self.assertEqual(app.estimate_tokens(""), 0)

    def test_none_returns_zero(self):
        self.assertEqual(app.estimate_tokens(None), 0)

    def test_single_word(self):
        result = app.estimate_tokens("hello")
        self.assertGreater(result, 0)

    def test_longer_text_has_more_tokens_than_shorter(self):
        short = app.estimate_tokens("hello world")
        long  = app.estimate_tokens("hello world " * 100)
        self.assertGreater(long, short)

    def test_token_estimate_is_positive_int(self):
        result = app.estimate_tokens("the quick brown fox jumps over the lazy dog")
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_word_count_scales_linearly(self):
        """Doubling the word count should roughly double the estimate."""
        one_copy = "word " * 10
        two_copy = "word " * 20
        self.assertAlmostEqual(
            app.estimate_tokens(two_copy),
            app.estimate_tokens(one_copy) * 2,
            delta=2,
        )


# ═════════════════════════════════════════════════════════════════════════════
# sql_search — correctness against the seeded in-memory DB
# ═════════════════════════════════════════════════════════════════════════════

class TestSqlSearch(unittest.TestCase):
    """
    init_sql_db() runs at import time and seeds the DB with known topics,
    relationships, and facts. We can test sql_search() against those.
    """

    def test_returns_string(self):
        result = app.sql_search("transformer")
        self.assertIsInstance(result, str)

    def test_known_topic_found(self):
        """'transformer' appears in the seeded topics."""
        result = app.sql_search("transformer")
        self.assertIn("[TOPIC]", result)
        self.assertIn("Transformer", result)

    def test_rlhf_topic_found(self):
        result = app.sql_search("rlhf")
        self.assertIn("RLHF", result)

    def test_rag_topic_found(self):
        # sql_search filters words with len <= 3, so use the full word from the
        # seeded summary ("Retrieval-Augmented Generation") rather than the acronym.
        result = app.sql_search("retrieval")
        self.assertIn("RAG", result)

    def test_relationship_found(self):
        """'rlhf' (4 chars, passes length filter) matches relationship endpoints."""
        result = app.sql_search("rlhf")
        self.assertIn("[REL]", result)

    def test_fact_found(self):
        """'transformer' should surface the Vaswani et al. fact."""
        result = app.sql_search("transformer")
        self.assertIn("[FACT]", result)

    def test_unknown_query_returns_no_results(self):
        result = app.sql_search("zxqvwkjhgfdsapoiuyt_notaword")
        self.assertEqual(result, "(no SQL results)")

    def test_result_count_capped_at_k(self):
        """k parameter limits rows returned."""
        result_3 = app.sql_search("transformer", k=3)
        result_1 = app.sql_search("transformer", k=1)
        lines_3  = [l for l in result_3.split("\n") if l.strip()]
        lines_1  = [l for l in result_1.split("\n") if l.strip()]
        self.assertGreaterEqual(len(lines_3), len(lines_1))
        self.assertLessEqual(len(lines_1), 1)

    def test_no_duplicate_rows(self):
        """Results should not contain duplicate lines."""
        result = app.sql_search("attention")
        lines  = [l for l in result.split("\n") if l.strip()]
        self.assertEqual(len(lines), len(set(lines)))


# ═════════════════════════════════════════════════════════════════════════════
# Cache helpers — save / load / expiry
# ═════════════════════════════════════════════════════════════════════════════

class TestCache(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig   = app.CACHE_DIR
        app.CACHE_DIR = Path(self._tmpdir.name)

    def tearDown(self):
        app.CACHE_DIR = self._orig
        self._tmpdir.cleanup()

    def test_save_and_load_round_trip(self):
        payload = {"summary": "test summary", "knowledge_map": {"nodes": [], "edges": []}}
        app.cache_save("my query", payload)
        loaded = app.cache_load("my query")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["summary"], "test summary")

    def test_load_returns_none_for_missing_query(self):
        result = app.cache_load("query that was never saved")
        self.assertIsNone(result)

    def test_cache_is_keyed_by_query(self):
        app.cache_save("query A", {"summary": "A"})
        app.cache_save("query B", {"summary": "B"})
        self.assertEqual(app.cache_load("query A")["summary"], "A")
        self.assertEqual(app.cache_load("query B")["summary"], "B")

    def test_messages_stripped_from_cache(self):
        """LangChain message objects must not be persisted to the cache."""
        payload = {"summary": "s", "messages": [MagicMock()]}
        app.cache_save("q", payload)
        loaded = app.cache_load("q")
        self.assertNotIn("messages", loaded)

    def test_expired_cache_returns_none(self):
        """Entries older than CACHE_TTL_DAYS should be evicted on load."""
        from datetime import datetime, timedelta, timezone
        payload = {"summary": "old"}
        app.cache_save("old query", payload)

        # Manually backdate the timestamp in the written JSON
        h    = app._hash("old query")
        path = app.CACHE_DIR / f"{h}.json"
        data = json.loads(path.read_text())
        old_ts = (datetime.now(timezone.utc) - timedelta(days=app.CACHE_TTL_DAYS + 1)).isoformat()
        data["ts"] = old_ts
        path.write_text(json.dumps(data))

        result = app.cache_load("old query")
        self.assertIsNone(result)

    def test_cache_list_returns_saved_entries(self):
        app.cache_save("list query", {"summary": "x"})
        entries = app.cache_list()
        queries = [e["query"] for e in entries]
        self.assertIn("list query", queries)

    def test_query_normalised_before_hashing(self):
        """Same query with different casing/whitespace should hit the same cache slot."""
        app.cache_save("  My Query  ", {"summary": "upper"})
        result = app.cache_load("my query")
        self.assertIsNotNone(result)


# ═════════════════════════════════════════════════════════════════════════════
# VectorDBModule — count, add_text, search, sources, reindex_saved_docs
# ═════════════════════════════════════════════════════════════════════════════

class TestVectorDBModule(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._vdb_dir = Path(self._tmpdir.name) / "vectorstore"
        self._docs_dir = Path(self._tmpdir.name) / "docs"
        self._docs_dir.mkdir(parents=True)
        # Patch DOCS_DIR so VectorDBModule.sources() and reindex use our temp dir
        self._orig_docs = app.DOCS_DIR
        app.DOCS_DIR = self._docs_dir

    def tearDown(self):
        app.DOCS_DIR = self._orig_docs
        self._tmpdir.cleanup()

    def _make_vdb(self):
        """Return a VectorDBModule with a mock embeddings object."""
        embeddings = MagicMock()
        return app.VectorDBModule(embeddings, vector_dir=self._vdb_dir)

    def test_count_is_zero_when_no_store(self):
        vdb = self._make_vdb()
        # FAISS.load_local raises (from mock), so _store is None
        self.assertEqual(vdb.count(), 0)

    def test_search_returns_empty_when_no_store(self):
        vdb = self._make_vdb()
        self.assertEqual(vdb.search("anything"), [])

    def test_add_text_creates_store(self):
        vdb = self._make_vdb()
        # Patch FAISS.from_documents to return a mock store with index.ntotal=1
        mock_store = MagicMock()
        mock_store.index.ntotal = 1
        import langchain_community.vectorstores as lc_vs
        with patch.object(lc_vs.FAISS, "from_documents", return_value=mock_store):
            vdb.add_text("hello world", {"source": "test"})
        self.assertIsNotNone(vdb._store)

    def test_sources_returns_empty_when_docs_dir_empty(self):
        vdb = self._make_vdb()
        # DOCS_DIR is set to our temp dir (empty)
        self.assertEqual(vdb.sources(), [])

    def test_sources_lists_files_in_docs_dir(self):
        (self._docs_dir / "paper.txt").write_text("hello")
        (self._docs_dir / "notes.md").write_text("world")
        vdb = self._make_vdb()
        sources = vdb.sources()
        self.assertIn("paper.txt", sources)
        self.assertIn("notes.md", sources)

    def test_reindex_skips_if_count_positive(self):
        vdb = self._make_vdb()
        mock_store = MagicMock()
        mock_store.index.ntotal = 5
        vdb._store = mock_store
        (self._docs_dir / "doc.txt").write_text("content")
        result = vdb.reindex_saved_docs()
        self.assertEqual(result, 0)   # skipped, no re-indexing

    def test_reindex_skips_if_docs_dir_missing(self):
        app.DOCS_DIR = Path(self._tmpdir.name) / "nonexistent_docs"
        vdb = self._make_vdb()
        result = vdb.reindex_saved_docs()
        self.assertEqual(result, 0)

    def test_reindex_indexes_txt_and_md_files(self):
        (self._docs_dir / "paper.txt").write_text("some content about RAG")
        (self._docs_dir / "notes.md").write_text("# Notes\nSome markdown")

        mock_store = MagicMock()
        mock_store.index.ntotal = 0
        import langchain_community.vectorstores as lc_vs
        with patch.object(lc_vs.FAISS, "from_documents", return_value=mock_store):
            vdb = self._make_vdb()
            result = vdb.reindex_saved_docs()
        self.assertEqual(result, 2)

    def test_reindex_ignores_non_txt_md_files(self):
        (self._docs_dir / "data.csv").write_text("a,b,c")
        (self._docs_dir / "image.png").write_bytes(b"\x89PNG")
        vdb = self._make_vdb()
        result = vdb.reindex_saved_docs()
        self.assertEqual(result, 0)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
