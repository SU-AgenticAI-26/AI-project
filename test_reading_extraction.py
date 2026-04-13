"""
Tests for the Reading/Extraction Agent added to streamlit_app.py.

Run with:
    python -m pytest test_reading_extraction.py -v

No OpenAI API key required — all LLM calls are mocked.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


# ─────────────────────────────────────────────────────────────────────────────
# Module-level mocking helpers
# All Streamlit UI/DB initialisation is gated inside main(), so it never runs
# during import.  We only need to stub out the heavy third-party packages so
# that `import streamlit_app` succeeds in environments where they are absent.
# ─────────────────────────────────────────────────────────────────────────────

def _patch_imports():
    """Stub out third-party packages that may be absent in the test environment."""
    # conftest.py installs all shims (including the correctly-typed ChatOpenAI
    # sentinel class) before any test module is collected and sets this sentinel.
    # Return immediately to avoid overwriting those stubs with a flat MagicMock,
    # which would make isinstance(model, ChatOpenAI) raise TypeError.
    if "_shims_installed" in sys.modules or "streamlit_app" in sys.modules:
        return
    mocks = {}

    # streamlit — configure return values so module-level UI code doesn't
    # execute conditional branches during import.
    st_mock = MagicMock()
    st_mock.button.return_value = False
    st_mock.form_submit_button.return_value = False
    st_mock.text_input.return_value = ""
    st_mock.text_area.return_value = ""
    st_mock.number_input.return_value = 5
    st_mock.checkbox.return_value = False
    st_mock.file_uploader.return_value = None
    st_mock.selectbox.return_value = "OpenAI"
    # st.tabs / st.columns must return the right number of context managers
    st_mock.tabs.side_effect = lambda labels: [MagicMock() for _ in labels]
    st_mock.columns.side_effect = lambda n: [MagicMock() for _ in (range(n) if isinstance(n, int) else n)]
    st_mock.session_state.__contains__ = MagicMock(return_value=False)
    mocks["streamlit"] = st_mock

    # pyvis
    pyvis_mod  = types.ModuleType("pyvis")
    pyvis_net  = types.ModuleType("pyvis.network")
    pyvis_net.Network = MagicMock()
    pyvis_mod.network = pyvis_net
    mocks["pyvis"] = pyvis_mod
    mocks["pyvis.network"] = pyvis_net

    # langchain_core — provide lightweight message/document stubs that preserve
    # the `content` kwarg so prompt-inspection tests can read it back.
    class _Msg:
        def __init__(self, content: str = "", **_kw):
            self.content = content

    class _Doc:
        def __init__(self, page_content: str = "", metadata: dict | None = None, **_kw):
            self.page_content = page_content
            self.metadata = metadata or {}

    lc_core       = types.ModuleType("langchain_core")
    lc_core_docs  = types.ModuleType("langchain_core.documents")
    lc_core_msgs  = types.ModuleType("langchain_core.messages")
    lc_core_llms  = types.ModuleType("langchain_core.language_models")
    lc_core_docs.Document       = _Doc
    lc_core_msgs.AIMessage      = _Msg
    lc_core_msgs.HumanMessage   = _Msg
    lc_core_msgs.SystemMessage  = _Msg
    lc_core_llms.BaseChatModel  = MagicMock
    lc_core.documents           = lc_core_docs
    lc_core.messages            = lc_core_msgs
    lc_core.language_models     = lc_core_llms
    mocks["langchain_core"]                    = lc_core
    mocks["langchain_core.documents"]          = lc_core_docs
    mocks["langchain_core.messages"]           = lc_core_msgs
    mocks["langchain_core.language_models"]    = lc_core_llms

    # langchain / openai stubs
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
    # Make FAISS.load_local raise so VectorDBModule._load sets _store=None
    # and VectorDBModule.count() reliably returns 0 (an int) during import.
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

    return mocks


# Apply patches before any import of streamlit_app
_patch_imports()

# Now we can safely import the module
import streamlit_app as app  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_state(**overrides) -> app.AgentState:
    """Return a minimal valid AgentState with optional field overrides."""
    base: app.AgentState = {
        "messages":            [],
        "query":               "test query about neural networks",
        "active_agents":       ["vector_db", "sql_db", "web"],
        "router_reasoning":    "test",
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
    """Return a mock ChatOpenAI whose .invoke() returns response_text."""
    model = MagicMock()
    ai_msg = MagicMock()
    ai_msg.content = response_text
    model.invoke.return_value = ai_msg
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Tests — skip logic
# ─────────────────────────────────────────────────────────────────────────────

class TestReadingExtractionSkipLogic(unittest.TestCase):

    def test_skip_when_all_findings_empty(self):
        """Agent should skip without calling the LLM when all findings are empty."""
        model = _mock_llm("should not be called")
        state = _make_state(
            vector_findings="",
            sql_findings="",
            web_findings="",
        )
        result = app.reading_extraction_agent(state, model)

        model.invoke.assert_not_called()
        self.assertEqual(result["extraction_findings"], "(none)")
        self.assertEqual(result["current_agent"], "reading_extraction")

    def test_skip_when_all_not_activated(self):
        """Agent should skip when every retriever returned '(not activated)'."""
        model = _mock_llm("should not be called")
        state = _make_state(
            vector_findings="(not activated)",
            sql_findings="(not activated)",
            web_findings="(not activated)",
        )
        result = app.reading_extraction_agent(state, model)

        model.invoke.assert_not_called()
        self.assertEqual(result["extraction_findings"], "(none)")

    def test_skip_when_sql_only_no_results(self):
        """Agent should skip when sql returned '(no SQL results)' and others empty."""
        model = _mock_llm("should not be called")
        state = _make_state(
            vector_findings="",
            sql_findings="(no SQL results)",
            web_findings="",
        )
        result = app.reading_extraction_agent(state, model)

        model.invoke.assert_not_called()

    def test_skip_activity_log_has_correct_structure(self):
        """Skip path should still write a valid activity log entry."""
        model = _mock_llm("x")
        state = _make_state()
        result = app.reading_extraction_agent(state, model)

        self.assertEqual(len(result["activity_log"]), 1)
        entry = result["activity_log"][0]
        self.assertEqual(entry["agent"], "reading_extraction")
        self.assertEqual(entry["icon"], "📖")
        self.assertIn("title", entry)
        self.assertIn("ts", entry)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — LLM invocation and output
# ─────────────────────────────────────────────────────────────────────────────

_SINGLE_RECORD_RESPONSE = """---
**Title / Topic:** Attention Is All You Need
**Provenance:** abstract-only
**Research Problem:** Replace recurrent networks with pure attention mechanisms.
**Methodology:** Encoder-decoder with multi-head self-attention and positional encoding.
**Key Findings:**
- Achieves state-of-the-art on WMT translation tasks
- Trains significantly faster than RNN baselines
**Limitations:** Quadratic complexity in sequence length.
**Future Work:** Investigate linear attention variants.
---"""

_MULTI_RECORD_RESPONSE = """---
**Title / Topic:** Paper One
**Provenance:** abstract-only
**Research Problem:** Problem A.
**Methodology:** Method A.
**Key Findings:** - Finding A
**Limitations:** Not stated
**Future Work:** Not stated
---
---
**Title / Topic:** Paper Two
**Provenance:** full-text
**Research Problem:** Problem B.
**Methodology:** Method B.
**Key Findings:** - Finding B
**Limitations:** Limited dataset.
**Future Work:** Extend to multilingual settings.
---
---
**Title / Topic:** Paper Three
**Provenance:** structured-db
**Research Problem:** Problem C.
**Methodology:** Method C.
**Key Findings:** - Finding C
**Limitations:** Not stated
**Future Work:** Not stated
---"""


class TestReadingExtractionOutput(unittest.TestCase):

    def test_calls_llm_when_vector_findings_present(self):
        """LLM should be invoked when vector_findings has real content."""
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        state = _make_state(vector_findings="Some retrieved vector content about RAG.")
        app.reading_extraction_agent(state, model)
        model.invoke.assert_called_once()

    def test_calls_llm_when_web_findings_present(self):
        """LLM should be invoked when web_findings has real content."""
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        state = _make_state(web_findings="arXiv paper about transformers.")
        app.reading_extraction_agent(state, model)
        model.invoke.assert_called_once()

    def test_calls_llm_when_sql_findings_present(self):
        """LLM should be invoked when sql_findings has real content."""
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        state = _make_state(sql_findings="[TOPIC] Transformer Architecture (ML): Self-attention model.")
        app.reading_extraction_agent(state, model)
        model.invoke.assert_called_once()

    def test_extraction_findings_field_populated(self):
        """Return dict must set extraction_findings to the LLM response."""
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        state = _make_state(vector_findings="RAG content")
        result = app.reading_extraction_agent(state, model)
        self.assertEqual(result["extraction_findings"], _SINGLE_RECORD_RESPONSE)

    def test_single_record_count(self):
        """Activity log detail should reflect one extracted record."""
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        state = _make_state(vector_findings="RAG content")
        result = app.reading_extraction_agent(state, model)
        detail = result["activity_log"][0]["detail"]
        self.assertIn("1", detail)

    def test_multi_record_count(self):
        """Activity log detail should reflect three extracted records."""
        model = _mock_llm(_MULTI_RECORD_RESPONSE)
        state = _make_state(web_findings="Three arXiv papers about deep learning.")
        result = app.reading_extraction_agent(state, model)
        detail = result["activity_log"][0]["detail"]
        self.assertIn("3", detail)

    def test_messages_field_populated(self):
        """Return dict must include a non-empty messages list."""
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        state = _make_state(vector_findings="content")
        result = app.reading_extraction_agent(state, model)
        self.assertIsInstance(result["messages"], list)
        self.assertGreater(len(result["messages"]), 0)

    def test_current_agent_set_correctly(self):
        """current_agent must be 'reading_extraction' in all paths."""
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        for state in [
            _make_state(),                                        # skip path
            _make_state(vector_findings="some content"),          # normal path
        ]:
            result = app.reading_extraction_agent(state, model)
            self.assertEqual(result["current_agent"], "reading_extraction")


# ─────────────────────────────────────────────────────────────────────────────
# Tests — LLM prompt construction
# ─────────────────────────────────────────────────────────────────────────────

class TestReadingExtractionPrompt(unittest.TestCase):

    def _get_human_message_content(self, model: MagicMock) -> str:
        """Extract the HumanMessage content from the first invoke() call."""
        call_args = model.invoke.call_args
        messages = call_args[0][0]          # positional arg: list of messages
        # HumanMessage is the last message in the list
        return messages[-1].content

    def test_query_included_in_prompt(self):
        """The user's query must appear in the human message."""
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        state = _make_state(
            query="effects of climate change on bird migration",
            vector_findings="Some content",
        )
        app.reading_extraction_agent(state, model)
        human_content = self._get_human_message_content(model)
        self.assertIn("effects of climate change on bird migration", human_content)

    def test_vector_findings_in_prompt(self):
        """Vector findings must appear in the combined content sent to LLM."""
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        state = _make_state(vector_findings="UNIQUE_VECTOR_CONTENT_XYZ")
        app.reading_extraction_agent(state, model)
        human_content = self._get_human_message_content(model)
        self.assertIn("UNIQUE_VECTOR_CONTENT_XYZ", human_content)

    def test_web_findings_in_prompt(self):
        """Web findings must appear in the combined content sent to LLM."""
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        state = _make_state(web_findings="UNIQUE_WEB_CONTENT_ABC")
        app.reading_extraction_agent(state, model)
        human_content = self._get_human_message_content(model)
        self.assertIn("UNIQUE_WEB_CONTENT_ABC", human_content)

    def test_sql_findings_in_prompt(self):
        """SQL findings must appear in the combined content sent to LLM."""
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        state = _make_state(sql_findings="[TOPIC] Knowledge Graph (Data): Entities and edges.")
        app.reading_extraction_agent(state, model)
        human_content = self._get_human_message_content(model)
        self.assertIn("Knowledge Graph", human_content)

    def test_not_activated_findings_excluded_from_prompt(self):
        """'(not activated)' findings must NOT be forwarded to the LLM."""
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        state = _make_state(
            vector_findings="(not activated)",
            sql_findings="Real SQL content about attention mechanisms.",
            web_findings="(not activated)",
        )
        app.reading_extraction_agent(state, model)
        human_content = self._get_human_message_content(model)
        self.assertNotIn("(not activated)", human_content)
        self.assertIn("attention mechanisms", human_content)

    def test_system_message_mentions_provenance(self):
        """System prompt must mention provenance to guide the LLM."""
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        state = _make_state(vector_findings="content")
        app.reading_extraction_agent(state, model)
        messages = model.invoke.call_args[0][0]
        system_content = messages[0].content
        self.assertIn("Provenance", system_content)

    def test_system_message_mentions_all_extraction_fields(self):
        """System prompt must list all five required extraction fields."""
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        state = _make_state(vector_findings="content")
        app.reading_extraction_agent(state, model)
        messages = model.invoke.call_args[0][0]
        system_content = messages[0].content
        for field in ["Research Problem", "Methodology", "Key Findings",
                      "Limitations", "Future Work"]:
            self.assertIn(field, system_content)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — AgentState schema
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentStateSchema(unittest.TestCase):

    def test_extraction_findings_field_exists(self):
        """AgentState TypedDict must declare extraction_findings."""
        annotations = app.AgentState.__annotations__
        self.assertIn("extraction_findings", annotations)

    def test_extraction_findings_type_is_str(self):
        """extraction_findings must be typed as str.

        Note: with ``from __future__ import annotations`` the annotation is a
        ForwardRef rather than the bare ``str`` type, so we compare the string
        representation instead of using ``assertIs``.
        """
        annotations = app.AgentState.__annotations__
        ann = annotations["extraction_findings"]
        # Accepts both the resolved type (str) and a ForwardRef whose arg is 'str'
        self.assertIn("str", str(ann))

    def test_all_original_fields_still_present(self):
        """Adding the new field must not remove any pre-existing fields."""
        required = [
            "messages", "query", "active_agents", "router_reasoning",
            "vector_findings", "sql_findings", "web_findings",
            "activity_log", "merged_context", "knowledge_map",
            "critique", "loop_count", "summary", "current_agent",
        ]
        annotations = app.AgentState.__annotations__
        for field in required:
            self.assertIn(field, annotations, f"Missing field: {field}")


# ─────────────────────────────────────────────────────────────────────────────
# Tests — activity log entry structure
# ─────────────────────────────────────────────────────────────────────────────

class TestActivityLogEntry(unittest.TestCase):

    def _run(self, **state_overrides):
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        state = _make_state(**state_overrides)
        return app.reading_extraction_agent(state, model)

    def test_entry_has_agent_field(self):
        result = self._run(vector_findings="content")
        self.assertEqual(result["activity_log"][0]["agent"], "reading_extraction")

    def test_entry_has_icon_field(self):
        result = self._run(vector_findings="content")
        self.assertEqual(result["activity_log"][0]["icon"], "📖")

    def test_entry_has_title_field(self):
        result = self._run(vector_findings="content")
        self.assertIn("title", result["activity_log"][0])

    def test_entry_has_detail_field(self):
        result = self._run(vector_findings="content")
        self.assertIn("detail", result["activity_log"][0])

    def test_entry_has_ts_field(self):
        result = self._run(vector_findings="content")
        self.assertIn("ts", result["activity_log"][0])

    def test_normal_detail_mentions_extraction_fields(self):
        """Normal-path detail should mention the key structured fields."""
        result = self._run(vector_findings="content")
        detail = result["activity_log"][0]["detail"]
        for term in ["methodology", "findings", "provenance"]:
            self.assertIn(term, detail.lower())


# ─────────────────────────────────────────────────────────────────────────────
# Tests — interaction with other agents (orchestrator integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorIntegration(unittest.TestCase):
    """Verify orchestrator_agent receives extraction_findings in its prompt."""

    def _run_orchestrator(self, extraction_findings: str) -> str:
        """Return the human message content sent to the orchestrator LLM."""
        model = _mock_llm("Merged context output.")
        state = _make_state(
            vector_findings="Vector content",
            sql_findings="SQL content",
            web_findings="Web content",
            extraction_findings=extraction_findings,
        )
        app.orchestrator_agent(state, model)
        call_args = model.invoke.call_args
        messages = call_args[0][0]
        return messages[-1].content

    def test_extraction_findings_forwarded_to_orchestrator(self):
        """Orchestrator must include extraction_findings in its context block."""
        human_content = self._run_orchestrator("EXTRACTION_SENTINEL_VALUE")
        self.assertIn("EXTRACTION_SENTINEL_VALUE", human_content)

    def test_extraction_findings_label_in_orchestrator_prompt(self):
        """Orchestrator context block must be labelled 'Structured Extraction'."""
        human_content = self._run_orchestrator("some extraction")
        self.assertIn("Structured Extraction", human_content)

    def test_orchestrator_system_prompt_mentions_extraction(self):
        """Orchestrator system message must reference the Extraction agent."""
        model = _mock_llm("output")
        state = _make_state(extraction_findings="some content")
        app.orchestrator_agent(state, model)
        messages = model.invoke.call_args[0][0]
        system_content = messages[0].content
        self.assertIn("Extraction", system_content)

    def test_empty_extraction_does_not_break_orchestrator(self):
        """Orchestrator should still work when extraction_findings is empty."""
        human_content = self._run_orchestrator("")
        # Just verify it ran and produced something
        self.assertIsInstance(human_content, str)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — edge cases and robustness
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):

    def test_only_whitespace_findings_treated_as_empty(self):
        """Whitespace-only content should not trigger LLM invocation."""
        # The current implementation strips the findings string, so
        # whitespace-only content is treated as empty and should skip the LLM.
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        state = _make_state(vector_findings="   ")
        result = app.reading_extraction_agent(state, model)
        # Ensure the agent runs and does not call the LLM for whitespace-only input
        self.assertIn("extraction_findings", result)
        model.invoke.assert_not_called()

    def test_very_long_findings_handled(self):
        """Agent should not error on large input content."""
        model = _mock_llm(_MULTI_RECORD_RESPONSE)
        long_content = "Abstract content. " * 500
        state = _make_state(vector_findings=long_content)
        result = app.reading_extraction_agent(state, model)
        self.assertIsNotNone(result["extraction_findings"])

    def test_zero_record_response(self):
        """LLM returning no titled records produces n_records == 0 gracefully."""
        model = _mock_llm("I could not identify any distinct papers.")
        state = _make_state(vector_findings="some vague content")
        result = app.reading_extraction_agent(state, model)
        detail = result["activity_log"][0]["detail"]
        self.assertIn("0", detail)

    def test_result_never_contains_none(self):
        """All return dict values must be non-None."""
        model = _mock_llm(_SINGLE_RECORD_RESPONSE)
        for state in [
            _make_state(),
            _make_state(vector_findings="content"),
            _make_state(sql_findings="sql content"),
            _make_state(web_findings="web content"),
        ]:
            result = app.reading_extraction_agent(state, model)
            for key, val in result.items():
                self.assertIsNotNone(val, f"None found for key: {key}")

    def test_provenance_flags_in_sample_response(self):
        """Verify the three provenance categories appear in the multi-record fixture."""
        provenances = ["abstract-only", "full-text", "structured-db"]
        for prov in provenances:
            self.assertIn(prov, _MULTI_RECORD_RESPONSE)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
