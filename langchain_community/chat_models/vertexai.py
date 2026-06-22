"""Compatibility bridge for ragas expecting the old Vertex AI import path."""

from __future__ import annotations

try:
    from langchain_google_vertexai import ChatVertexAI
except ImportError as exc:  # pragma: no cover - import-time compatibility guard
    raise ImportError(
        "langchain-google-vertexai is required for ragas' legacy vertexai import path"
    ) from exc

__all__ = ["ChatVertexAI"]