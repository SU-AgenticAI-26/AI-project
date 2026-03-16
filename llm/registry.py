"""
llm/registry.py — Provider registry and AgentConfig for 5-agent architecture.

Agent names (matching the new proposal):
  orchestrator
  scoping_agent
  search_reading_agent     (merged Search + Reading)
  synthesis_planning_agent (merged Synthesis + Planning)
  validation_agent

Config dict schema
──────────────────
{
  "default": {
    "provider": "anthropic" | "openai" | "gemini" | "ollama" | "llamacpp",
    "model":    "model-id-string",
    "api_key":  "...",      # optional — falls back to env var
    "base_url": "...",      # for openai-compat / llamacpp / ollama
    "host":     "...",      # alias for base_url (ollama / llamacpp)
  },
  "agents": {
    "scoping_agent":            { ...override... },
    "search_reading_agent":     { ...override... },
    "synthesis_planning_agent": { ...override... },
    "validation_agent":         { ...override... },
    "orchestrator":             { ...override... },
  }
}

Any agent not listed uses the "default" config.
"""

from __future__ import annotations
from llm.base import LLMProvider

AGENT_NAMES = [
    "orchestrator",
    "scoping_agent",
    "search_reading_agent",
    "synthesis_planning_agent",
    "validation_agent",
]

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai":    "gpt-4o",
    "gemini":    "gemini-1.5-pro",
    "ollama":    "llama3.1:8b",
    "llamacpp":  "local-model",
}


def _build_provider(cfg: dict) -> LLMProvider:
    provider_id = cfg.get("provider", "anthropic").lower()
    model       = cfg.get("model") or DEFAULT_MODELS.get(provider_id, "")
    api_key     = cfg.get("api_key")
    base_url    = cfg.get("base_url", "")
    host        = cfg.get("host", "")

    if provider_id == "anthropic":
        from llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model, api_key=api_key)

    elif provider_id == "openai":
        from llm.openai_provider import OpenAIProvider
        return OpenAIProvider(model=model, api_key=api_key,
                              base_url=base_url or None)

    elif provider_id == "gemini":
        from llm.gemini_provider import GeminiProvider
        return GeminiProvider(model=model, api_key=api_key)

    elif provider_id == "ollama":
        from llm.ollama_provider import OllamaProvider
        return OllamaProvider(model=model,
                              host=host or base_url or "http://localhost:11434")

    elif provider_id == "llamacpp":
        from llm.llamacpp_provider import LlamaCppProvider
        return LlamaCppProvider(model=model,
                                host=host or base_url or "http://localhost:8080")

    else:
        raise ValueError(
            f"Unknown provider: '{provider_id}'. "
            f"Valid options: anthropic, openai, gemini, ollama, llamacpp"
        )


class AgentConfig:
    """Maps each agent name to an LLMProvider instance."""

    def __init__(self, providers: dict[str, LLMProvider]):
        self._providers = providers

    def __getitem__(self, agent_name: str) -> LLMProvider:
        return self._providers.get(agent_name, self._providers["_default"])

    def summary(self) -> dict[str, str]:
        return {agent: self[agent].name for agent in AGENT_NAMES}


def build_config(config_dict: dict) -> AgentConfig:
    """
    Build an AgentConfig from a config dict.

    Example:
        build_config({
            "default": {"provider": "llamacpp", "host": "http://localhost:8080",
                        "model": "llama3.1"},
            "agents":  {"validation_agent": {"provider": "anthropic"}},
        })
    """
    default_cfg      = config_dict.get("default", {"provider": "anthropic"})
    default_provider = _build_provider(default_cfg)

    providers: dict[str, LLMProvider] = {"_default": default_provider}

    for agent_name, override in config_dict.get("agents", {}).items():
        providers[agent_name] = _build_provider(override)

    return AgentConfig(providers)


def build_config_from_env() -> AgentConfig:
    """
    Fallback: build a config from environment variables alone.

    Env vars checked (in priority order):
      DEFAULT_PROVIDER   — explicit: anthropic/openai/gemini/ollama/llamacpp
      ANTHROPIC_API_KEY  — implies anthropic
      OPENAI_API_KEY     — implies openai
      GEMINI_API_KEY     — implies gemini
      LLAMACPP_HOST      — implies llamacpp
      OLLAMA_HOST        — implies ollama
    """
    import os

    explicit = os.getenv("DEFAULT_PROVIDER", "").lower()

    if explicit:
        provider = explicit
    elif os.getenv("ANTHROPIC_API_KEY"):
        provider = "anthropic"
    elif os.getenv("OPENAI_API_KEY"):
        provider = "openai"
    elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        provider = "gemini"
    elif os.getenv("LLAMACPP_HOST"):
        provider = "llamacpp"
    elif os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_MODEL"):
        provider = "ollama"
    else:
        provider = "anthropic"  # will error at call time if no key

    model_map = {
        "anthropic": os.getenv("CLAUDE_MODEL",    "claude-sonnet-4-20250514"),
        "openai":    os.getenv("OPENAI_MODEL",    "gpt-4o"),
        "gemini":    os.getenv("GEMINI_MODEL",    "gemini-1.5-pro"),
        "ollama":    os.getenv("OLLAMA_MODEL",    "llama3.1:8b"),
        "llamacpp":  os.getenv("LLAMACPP_MODEL",  "local-model"),
    }

    cfg: dict = {"default": {"provider": provider, "model": model_map[provider]}}

    if provider == "ollama":
        cfg["default"]["host"] = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    elif provider == "llamacpp":
        cfg["default"]["host"] = os.getenv("LLAMACPP_HOST", "http://localhost:8080")

    return build_config(cfg)
