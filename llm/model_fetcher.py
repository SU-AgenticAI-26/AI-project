"""
llm/model_fetcher.py — Fetch and filter available models for each provider.

Each fetch_*_models() function returns a list of model IDs suitable for
this application: text generation + preferably tool/function calling support.
Returns [] on any error so callers can fall back to manual entry.
"""
from __future__ import annotations
import os

# ── Filter helpers ─────────────────────────────────────────────────────────────

# Anthropic: blocklist prefixes that are NOT chat/tool models
_ANTHROPIC_EXCLUDE = ("claude-instant", "claude-1", "claude-2.0", "claude-2.1")
# OpenAI: keep only models whose ID starts with one of these
_OPENAI_PREFIXES   = ("gpt-4", "gpt-3.5", "o1", "o3", "o4")
_OPENAI_EXCLUDE    = ("instruct", "vision", "audio", "realtime", "search",
                      "tts", "whisper", "dall", "embed", "babbage", "davinci",
                      "curie", "ada")
# Gemini: only generative models that support tool use
_GEMINI_REQUIRED_METHOD = "generateContent"
_GEMINI_EXCLUDE    = ("embedding", "aqa", "gemini-1.0")


def fetch_anthropic_models(api_key: str = "") -> list[str]:
    try:
        import anthropic
        key    = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        client = anthropic.Anthropic(api_key=key)
        page   = client.models.list(limit=100)
        ids    = [m.id for m in page.data]
        ids    = [i for i in ids if not any(i.startswith(e) for e in _ANTHROPIC_EXCLUDE)]
        return sorted(ids, reverse=True)
    except Exception:
        return []


def fetch_openai_models(api_key: str = "", base_url: str = "") -> list[str]:
    try:
        from openai import OpenAI
        key    = api_key or os.getenv("OPENAI_API_KEY", "")
        kwargs = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        models = client.models.list()
        ids    = [m.id for m in models.data]
        # Keep chat-capable models only
        ids = [
            i for i in ids
            if any(i.startswith(p) for p in _OPENAI_PREFIXES)
            and not any(ex in i for ex in _OPENAI_EXCLUDE)
        ]
        return sorted(ids, reverse=True)
    except Exception:
        return []


def fetch_gemini_models(api_key: str = "") -> list[str]:
    key = api_key or os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    if not key:
        return []
    # Try new SDK first
    try:
        import google.genai as genai
        client = genai.Client(api_key=key)
        ids = []
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            # name is like "models/gemini-1.5-pro-latest"
            model_id = name.split("/")[-1] if "/" in name else name
            methods  = getattr(m, "supported_generation_methods", []) or []
            if _GEMINI_REQUIRED_METHOD not in methods:
                continue
            if any(ex in model_id for ex in _GEMINI_EXCLUDE):
                continue
            ids.append(model_id)
        return sorted(set(ids), reverse=True)
    except Exception:
        pass
    # Fall back to legacy SDK
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        ids = []
        for m in genai.list_models():
            name     = getattr(m, "name", "") or ""
            model_id = name.split("/")[-1] if "/" in name else name
            methods  = getattr(m, "supported_generation_methods", []) or []
            if _GEMINI_REQUIRED_METHOD not in methods:
                continue
            if any(ex in model_id for ex in _GEMINI_EXCLUDE):
                continue
            ids.append(model_id)
        return sorted(set(ids), reverse=True)
    except Exception:
        return []


def fetch_ollama_models(host: str = "http://localhost:11434") -> list[str]:
    try:
        import requests
        resp = requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return sorted(m["name"] for m in models if m.get("name"))
    except Exception:
        return []
