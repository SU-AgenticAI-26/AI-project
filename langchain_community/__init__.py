"""Project-local compatibility overlay for legacy ragas imports.

This keeps the installed ``langchain_community`` package available while
providing the removed ``chat_models.vertexai`` module that older ragas
releases still import.
"""

from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
