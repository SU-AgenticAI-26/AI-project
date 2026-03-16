"""
llm/llamacpp_provider.py — llama.cpp server provider.

llama.cpp's server exposes an OpenAI-compatible API.
Start the server:
  ./llama-server -m model.gguf --host 0.0.0.0 --port 8080 [--n-gpu-layers 99]

Tool calling works for models compiled with the right GGUF (llama3.1, mistral, qwen2.5, etc.)
and llama.cpp built with LLAMA_CURL. Falls back to ReAct for models that don't support it.

This is a thin wrapper around OpenAIProvider with a llama.cpp-specific label and defaults.
"""

from __future__ import annotations
from llm.openai_provider import OpenAIProvider


class LlamaCppProvider(OpenAIProvider):

    def __init__(
        self,
        model:    str = "local-model",   # appears in /v1/models — use the exact model name
        host:     str = "http://localhost:8080",
        api_key:  str = "not-needed",    # llama.cpp ignores the key
    ):
        super().__init__(
            model    = model,
            api_key  = api_key,
            base_url = f"{host.rstrip('/')}/v1",
            label    = f"llama.cpp/{model}@{host}",
        )

    @property
    def name(self) -> str:
        return f"llama.cpp ({self._label})"
