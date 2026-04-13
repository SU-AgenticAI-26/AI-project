"""
conftest.py — shared test infrastructure for the AI-project test suite.

pytest guarantees this file is loaded before any test module is collected or
imported, so sys.modules patches installed here are visible to every test file
regardless of collection order.

Both test_agents.py and test_reading_extraction.py define their own
_patch_imports() helpers for historical reasons.  Those functions check for
the sentinel key "_shims_installed" in sys.modules and return immediately if
found, so they are no-ops when conftest has already run.  This eliminates the
order-dependent behaviour where one test file's flat MagicMock for
langchain_openai could overwrite the correctly-typed stub installed by
another, causing isinstance(model, ChatOpenAI) to raise TypeError in
vector_db_agent and sql_db_agent.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

# Guard: only patch once per process (pytest can import conftest multiple times
# in edge cases such as --import-mode=importlib).
if "_shims_installed" not in sys.modules:

    _mocks: dict = {}

    # ── streamlit ──────────────────────────────────────────────────────────────
    _st = MagicMock()
    _st.button.return_value               = False
    _st.form_submit_button.return_value   = False
    _st.text_input.return_value           = ""
    _st.text_area.return_value            = ""
    _st.number_input.return_value         = 5
    _st.checkbox.return_value             = False
    _st.file_uploader.return_value        = None
    _st.selectbox.return_value            = "OpenAI"
    _st.tabs.side_effect    = lambda labels: [MagicMock() for _ in labels]
    _st.columns.side_effect = lambda n: [MagicMock() for _ in (range(n) if isinstance(n, int) else n)]
    _st.session_state.__contains__ = lambda self, key: False
    _mocks["streamlit"] = _st

    # ── pyvis ──────────────────────────────────────────────────────────────────
    _pyvis     = types.ModuleType("pyvis")
    _pyvis_net = types.ModuleType("pyvis.network")
    _pyvis_net.Network = MagicMock()
    _pyvis.network     = _pyvis_net
    _mocks["pyvis"]         = _pyvis
    _mocks["pyvis.network"] = _pyvis_net

    # ── langchain_core ─────────────────────────────────────────────────────────
    class _Msg:
        def __init__(self, content: str = "", **_kw):
            self.content = content

    class _Doc:
        def __init__(self, page_content: str = "", metadata: dict | None = None, **_kw):
            self.page_content = page_content
            self.metadata     = metadata or {}

    _lc_core      = types.ModuleType("langchain_core")
    _lc_core_docs = types.ModuleType("langchain_core.documents")
    _lc_core_msgs = types.ModuleType("langchain_core.messages")
    _lc_core_llms = types.ModuleType("langchain_core.language_models")
    _lc_core_docs.Document      = _Doc
    _lc_core_msgs.AIMessage     = _Msg
    _lc_core_msgs.HumanMessage  = _Msg
    _lc_core_msgs.SystemMessage = _Msg
    _lc_core_llms.BaseChatModel = MagicMock
    _lc_core.documents          = _lc_core_docs
    _lc_core.messages           = _lc_core_msgs
    _lc_core.language_models    = _lc_core_llms
    _mocks["langchain_core"]                 = _lc_core
    _mocks["langchain_core.documents"]       = _lc_core_docs
    _mocks["langchain_core.messages"]        = _lc_core_msgs
    _mocks["langchain_core.language_models"] = _lc_core_llms

    # ── langchain_openai ───────────────────────────────────────────────────────
    # ChatOpenAI MUST be a real class, not a MagicMock instance.
    # isinstance(model, ChatOpenAI) raises TypeError when ChatOpenAI is a
    # MagicMock object rather than a type.  Using a sentinel class means
    # isinstance returns False for MagicMock models, routing agents to the
    # non-tool-calling fallback path that the tests actually exercise.
    _lo = types.ModuleType("langchain_openai")

    class _ChatOpenAI:
        """Sentinel class — never instantiated in tests."""

    _lo.ChatOpenAI       = _ChatOpenAI
    _lo.OpenAIEmbeddings = MagicMock()
    _mocks["langchain_openai"] = _lo

    # ── other langchain / faiss stubs ──────────────────────────────────────────
    for _mod in [
        "langchain_community",
        "langchain_community.vectorstores",
        "langchain_community.vectorstores.FAISS",
        "langchain_community.embeddings",
        "langchain_text_splitters",
        "faiss",
    ]:
        _mocks[_mod] = MagicMock()
    # Make FAISS.load_local raise so VectorDBModule sets _store=None on startup.
    _mocks["langchain_community.vectorstores"].FAISS.load_local.side_effect = Exception("mock")

    # ── langgraph ──────────────────────────────────────────────────────────────
    _lg       = types.ModuleType("langgraph")
    _lg_graph = types.ModuleType("langgraph.graph")
    _lg_graph.END = "END"

    class _FakeStateGraph:
        def __init__(self, *a, **kw):          pass
        def add_node(self, *a, **kw):          pass
        def add_edge(self, *a, **kw):          pass
        def add_conditional_edges(self, *a, **kw): pass
        def set_entry_point(self, *a, **kw):   pass
        def compile(self):                     return MagicMock()

    _lg_graph.StateGraph = _FakeStateGraph
    _lg.graph            = _lg_graph
    _mocks["langgraph"]       = _lg
    _mocks["langgraph.graph"] = _lg_graph

    # ── apply all mocks ────────────────────────────────────────────────────────
    for _key, _val in _mocks.items():
        sys.modules[_key] = _val

    # Sentinel: signals to _patch_imports() in both test files that shims are
    # already installed and they should skip their own patching logic.
    sys.modules["_shims_installed"] = object()  # type: ignore[assignment]
