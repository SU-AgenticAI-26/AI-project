"""Project-wide interpreter startup customizations.

This keeps third-party import expectations stable across the eval scripts.
"""

from __future__ import annotations

import importlib
import sys
import types


def _install_langchain_vertexai_compat() -> None:
    """Provide ragas' legacy Vertex AI import path if newer langchain-community removed it."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError:
        pass

    try:
        from langchain_google_vertexai import ChatVertexAI  # type: ignore
    except Exception:
        return

    compat_module = types.ModuleType(module_name)
    compat_module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = compat_module


_install_langchain_vertexai_compat()
