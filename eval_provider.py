"""
eval_provider.py — Provider abstraction layer for evaluation scripts.

Supports the same four providers as the main application:
  openai  — OpenAI API          (env: OPENAI_API_KEY)
  gemini  — Google Gemini       (env: GEMINI_API_KEY)
  claude  — Anthropic Claude    (env: ANTHROPIC_API_KEY)
  local   — llama.cpp / Ollama / LMStudio (OpenAI-compatible API, no key needed)

Two independent provider slots exist:
  pipeline  — runs the research agents
  judge     — runs RAGAS and DeepEval scoring (defaults to pipeline config)

This lets you, e.g., run the pipeline on a local model but judge with OpenAI,
or run everything locally end-to-end.

Usage in eval scripts:
    from eval_provider import EvalConfig, add_provider_args, cfg_from_args

    parser = argparse.ArgumentParser()
    add_provider_args(parser)
    args = parser.parse_args()
    cfg = cfg_from_args(args)

    # LangChain LLM for pipeline use
    lm = cfg.pipeline_llm(temperature=0.3)

    # RAGAS-compatible LLM
    ragas_llm = cfg.ragas_llm()

    # DeepEval model (string for OpenAI, DeepEvalBaseLLM wrapper otherwise)
    de_model = cfg.deepeval_model()
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Pydantic plugin discovery can fail on some Windows Python installs when a
# broken distribution metadata entry is present. Disable plugin loading to keep
# LangChain imports stable for eval runs.
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "1")

# Map short provider names → streamlit_app provider constants
_APP_PROVIDER = {
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "claude": "Anthropic Claude",
    "local":  "Local (llama.cpp / Ollama / LMStudio)",
}

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "claude": "claude-haiku-4-5-20251001",
    "local":  "",  # caller must supply via --model
}

_DEFAULT_BASE_URL = "http://localhost:8080/v1"  # llama.cpp default

# Cost per 1M tokens (input / output) — used by perf and baseline evals.
# Local models have zero API cost. Update OpenAI/Gemini/Claude entries if pricing changes.
PROVIDER_PRICING = {
    "openai": {"input": 0.15,  "output": 0.60},   # gpt-4o-mini, Apr 2026
    "gemini": {"input": 0.075, "output": 0.30},   # gemini-2.0-flash
    "claude": {"input": 0.80,  "output": 4.00},   # claude-haiku-4-5
    "local":  {"input": 0.0,   "output": 0.0},
}


def _load_env_from_project_root() -> None:
    """Load .env from project root and override stale process env values."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    parsed: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        parsed[key] = value

    for key, value in parsed.items():
        os.environ[key] = value

    if "OPENAI_API_KEY" not in parsed:
        for alias in ("key", "openai_key", "OPENAI_KEY"):
            if alias in parsed and parsed[alias]:
                os.environ["OPENAI_API_KEY"] = parsed[alias]
                break


_load_env_from_project_root()


def _resolve_api_key(provider: str) -> str:
    """Read the conventional environment variable for a provider."""
    env_var = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "local":  "",
    }.get(provider, "")
    return os.environ.get(env_var, "") if env_var else ""


def _query_local_models(base_url: str) -> list[str]:
    """
    Query a local OpenAI-compatible server for its available model IDs.
    Returns an empty list if the server is unreachable or returns no models.
    """
    import urllib.request
    import json as _json
    root = base_url.rstrip("/")
    if not root.endswith("/v1"):
        root = root + "/v1"
    try:
        with urllib.request.urlopen(f"{root}/models", timeout=3) as resp:
            data = _json.loads(resp.read())
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []


def _resolve_local_model(base_url: str, requested: str) -> str:
    """
    Resolve the model name to use for a local provider.

    - If `requested` is non-empty and the server lists it, use it as-is.
    - If `requested` is non-empty but NOT listed, print a warning with available
      models and still try (the server may accept it anyway).
    - If `requested` is empty, auto-detect: use the first model from /v1/models.
      If the server is unreachable, fall back to "local-model" (llama.cpp ignores
      the model field when only one model is loaded).
    """
    available = _query_local_models(base_url)

    if requested:
        if available and requested not in available:
            print(
                f"  WARNING: model '{requested}' not found on local server.\n"
                f"  Available models: {available}\n"
                f"  Use --model with one of the names above, or omit --model to auto-detect."
            )
        return requested

    # Auto-detect
    if available:
        chosen = available[0]
        print(f"  [local] Auto-detected model: '{chosen}'  (from {base_url}/v1/models)")
        return chosen

    # Server unreachable or returned no models — use empty string so
    # langchain_openai skips model validation; llama.cpp will use whatever is loaded.
    print(
        f"  WARNING: Could not reach {base_url}/v1/models to auto-detect model name.\n"
        f"  Proceeding with model='local-model'. Pass --model <name> to be explicit."
    )
    return "local-model"


@dataclass
class EvalConfig:
    """Holds provider configuration for both the pipeline and the judge."""

    # Pipeline (research agents)
    provider: str = "openai"
    api_key:  str = ""
    model:    str = ""
    base_url: str = _DEFAULT_BASE_URL

    # Judge (RAGAS / DeepEval) — empty fields inherit from pipeline
    judge_provider: str = ""
    judge_api_key:  str = ""
    judge_model:    str = ""
    judge_base_url: str = ""

    # ── Derived accessors ────────────────────────────────────────────────────

    def _jp(self) -> str:
        return self.judge_provider or self.provider

    def _jk(self) -> str:
        return self.judge_api_key or self.api_key

    def _jm(self) -> str:
        return self.judge_model or self.model or _DEFAULT_MODELS[self._jp()]

    def _jb(self) -> str:
        return self.judge_base_url or self.base_url

    def pricing(self) -> dict:
        """Return input/output pricing dict for the pipeline provider."""
        return PROVIDER_PRICING.get(self.provider, PROVIDER_PRICING["openai"])

    def judge_pricing(self) -> dict:
        return PROVIDER_PRICING.get(self._jp(), PROVIDER_PRICING["openai"])

    # ── LLM factories ────────────────────────────────────────────────────────

    def _app_cfg(self, provider: str, api_key: str, model: str, base_url: str):
        """Return a streamlit_app.ProviderConfig."""
        from test_queries import _add_project_to_path
        _add_project_to_path()
        from streamlit_app import ProviderConfig
        return ProviderConfig(
            provider=_APP_PROVIDER[provider],
            api_key=api_key,
            model=model or _DEFAULT_MODELS[provider],
            base_url=base_url if provider == "local" else None,
        )

    def pipeline_app_cfg(self):
        """streamlit_app.ProviderConfig for pipeline agents."""
        return self._app_cfg(self.provider, self.api_key,
                             self.model, self.base_url)

    def pipeline_llm(self, temperature: float = 0.3):
        """LangChain BaseChatModel for pipeline agents."""
        from test_queries import _add_project_to_path
        _add_project_to_path()
        from streamlit_app import _llm
        return _llm(self.pipeline_app_cfg(), temperature)

    def judge_langchain_llm(self, temperature: float = 0.0):
        """LangChain BaseChatModel for the judge (RAGAS / DeepEval)."""
        from test_queries import _add_project_to_path
        _add_project_to_path()
        from streamlit_app import _llm
        cfg = self._app_cfg(self._jp(), self._jk(), self._jm(), self._jb())
        return _llm(cfg, temperature)

    def ragas_llm(self):
        """
        Return a RAGAS-compatible LLM.

        Uses ragas.llms.base.LangchainLLMWrapper for all providers, which
        delegates to the standard LangChain completion API.

        Note: ragas.llms.llm_factory (InstructorLLM) is intentionally avoided
        because it relies on instructor structured-output parsing and silently
        returns NaN statements when parsing fails, resulting in faithfulness=NaN.
        ragas.llms.LangchainLLMWrapper (the top-level import) is a
        DeprecationHelper in ragas 0.4+ and must be imported from the base module.
        """
        from ragas.llms.base import LangchainLLMWrapper
        return LangchainLLMWrapper(self.judge_langchain_llm(temperature=0.0))

    def ragas_embeddings(self):
        """
        Return a RAGAS-compatible embeddings object.

        All providers go through LangchainEmbeddingsWrapper so that metrics
        like ResponseRelevancy (which internally call embed_query / embed_documents)
        work correctly.  The native RagasOpenAIEmbeddings exposes embed_text /
        embed_texts instead and breaks ResponseRelevancy in ragas 0.4.x.
        For OpenAI: wraps langchain_openai.OpenAIEmbeddings.
        For all other providers (including local): wraps HuggingFaceEmbeddings.

        Both ragas.llms.LangchainLLMWrapper and ragas.embeddings.LangchainEmbeddingsWrapper
        are DeprecationHelpers (not the real classes) in ragas 0.4+; always import
        from the respective .base submodule.

        The result is cached on the instance so model weights / clients are only
        initialised once per evaluation run, not once per query.
        """
        if hasattr(self, "_ragas_emb_cache"):
            return self._ragas_emb_cache

        # Always go through LangchainEmbeddingsWrapper so that metrics like
        # ResponseRelevancy (which call embed_query / embed_documents) work
        # regardless of provider.  RagasOpenAIEmbeddings exposes a different
        # interface (embed_text / embed_texts) that breaks ResponseRelevancy.
        from ragas.embeddings.base import LangchainEmbeddingsWrapper

        p = self._jp()
        if p == "openai":
            from langchain_openai import OpenAIEmbeddings
            emb = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(api_key=self._jk())
            )
        else:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            emb = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            )
        object.__setattr__(self, "_ragas_emb_cache", emb)
        return emb

    def deepeval_model(self):
        """
        Return a DeepEval-compatible model.

        For OpenAI: returns the model name string (DeepEval handles it natively).
        For other providers: returns a DeepEvalBaseLLM wrapper that delegates
        to the LangChain model.
        """
        p = self._jp()
        m = self._jm()
        if p == "openai":
            return m
        return _make_langchain_deepeval_model(
            self.judge_langchain_llm(temperature=0.0),
            name=m or f"{p}-judge",
        )


# ── DeepEval wrapper for non-OpenAI providers ────────────────────────────────

def _make_langchain_deepeval_model(lc_model, name: str = "local-model"):
    """
    Return a DeepEvalBaseLLM instance backed by any LangChain BaseChatModel.

    Uses a proper subclass definition (not dynamic type()) so that isinstance
    checks inside DeepEval pass reliably across versions.
    """
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
    except ImportError:
        raise ImportError("pip install deepeval")

    class _LCDEModel(DeepEvalBaseLLM):
        # Capture lc_model and name via closure over factory arguments.
        def __init__(self):
            pass  # DeepEvalBaseLLM.__init__ requires model_name; skip it here

        def load_model(self):
            return lc_model

        def generate(self, prompt: str) -> str:
            return _lc_generate_sync(lc_model, prompt)

        async def a_generate(self, prompt: str) -> str:
            return await _lc_generate_async(lc_model, prompt)

        def get_model_name(self) -> str:
            return name

    return _LCDEModel()


def _lc_generate_sync(lc_model, prompt: str) -> str:
    from langchain_core.messages import HumanMessage
    return lc_model.invoke([HumanMessage(content=prompt)]).content


async def _lc_generate_async(lc_model, prompt: str) -> str:
    from langchain_core.messages import HumanMessage
    resp = await lc_model.ainvoke([HumanMessage(content=prompt)])
    return resp.content


# ── Token counting and cost estimation ───────────────────────────────────────

try:
    import tiktoken as _tiktoken
    _ENC = _tiktoken.encoding_for_model("gpt-4o-mini")

    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken (gpt-4o-mini encoding)."""
        return len(_ENC.encode(str(text)))

except ImportError:
    def count_tokens(text: str) -> int:  # type: ignore[misc]
        """Fallback token count: ~4 chars per token."""
        return len(str(text)) // 4


def estimate_cost(
    input_tokens: int,
    pricing: dict,
    output_tokens: int | None = None,
) -> float:
    """
    Estimate USD cost from token counts and a provider pricing dict.

    If output_tokens is omitted, applies a 25 % output-fraction heuristic
    (i.e. output ≈ 25 % of input).
    """
    if output_tokens is None:
        output_tokens = int(input_tokens * 0.25)
    return round(
        input_tokens  * pricing["input"]  / 1_000_000 +
        output_tokens * pricing["output"] / 1_000_000,
        6,
    )


# ── CLI helpers ───────────────────────────────────────────────────────────────

def add_provider_args(parser) -> None:
    """
    Add standard provider arguments to an argparse.ArgumentParser.

    All arguments are optional; sensible defaults are derived from environment
    variables so existing workflows (OPENAI_API_KEY set, no flags) continue to
    work without changes.
    """
    g = parser.add_argument_group("provider")
    g.add_argument(
        "--provider",
        default=os.environ.get("EVAL_PROVIDER", "openai"),
        choices=["openai", "gemini", "claude", "local"],
        help="LLM provider for the research pipeline (default: openai)",
    )
    g.add_argument(
        "--api-key",
        default=None,
        dest="api_key",
        help="API key. Defaults to OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY",
    )
    g.add_argument(
        "--model",
        default=None,
        help="Model name for the pipeline. Provider default used if omitted.",
    )
    g.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        dest="base_url",
        help=f"Base URL for local OpenAI-compatible server (default: {_DEFAULT_BASE_URL})",
    )
    g.add_argument(
        "--judge-provider",
        default=None,
        dest="judge_provider",
        choices=["openai", "gemini", "claude", "local"],
        help="LLM provider for RAGAS/DeepEval judging. Defaults to --provider.",
    )
    g.add_argument(
        "--judge-model",
        default=None,
        dest="judge_model",
        help="Model for RAGAS/DeepEval judging. Defaults to --model.",
    )
    g.add_argument(
        "--judge-api-key",
        default=None,
        dest="judge_api_key",
        help="API key for the judge provider. Defaults to --api-key.",
    )
    g.add_argument(
        "--judge-base-url",
        default=None,
        dest="judge_base_url",
        help="Base URL for a local judge provider. Defaults to --base-url.",
    )


def cfg_from_args(args) -> EvalConfig:
    """Build an EvalConfig from parsed argparse Namespace."""
    provider = args.provider
    api_key  = args.api_key or _resolve_api_key(provider)
    base_url = args.base_url

    # For local providers, resolve the model name against the server's /v1/models
    # to catch mismatches early and auto-detect when --model is omitted.
    if provider == "local":
        model = _resolve_local_model(base_url, args.model or "")
    else:
        model = args.model or _DEFAULT_MODELS[provider]

    jp = getattr(args, "judge_provider", None) or provider
    jk = getattr(args, "judge_api_key",  None) or _resolve_api_key(jp) or api_key
    jb = getattr(args, "judge_base_url", None) or base_url

    raw_jm = getattr(args, "judge_model", None)
    if jp == "local":
        jm = _resolve_local_model(jb, raw_jm or "")
    else:
        jm = raw_jm or _DEFAULT_MODELS[jp]

    return EvalConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        judge_provider=jp,
        judge_api_key=jk,
        judge_model=jm,
        judge_base_url=jb,
    )
