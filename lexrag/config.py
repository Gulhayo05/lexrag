"""Runtime configuration (environment variables, optional .env file)."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEXRAG_", env_file=ROOT / ".env", extra="ignore")

    # data
    source_dir: str = "/Users/gulkhayo/Downloads/data"
    txt_dir: str = str(ROOT / "data" / "txt")
    db_path: str = str(ROOT / "data" / "lex.sqlite")
    results_db: str = str(ROOT / "data" / "batches.sqlite")

    # API
    api_keys: str = "test-key-change-me"           # comma separated
    rate_limit_per_sec: float = 2.0                  # batches / second / key (spec: >= 1)
    sync_timeout_s: float = 110.0                    # > this -> 202 + async
    question_timeout_s: float = 60.0
    max_concurrency: int = 10                        # questions processed in parallel

    # LLM provider: auto | anthropic | deepseek | gemini
    # auto = first provider whose key is set, in the order anthropic, deepseek, gemini
    llm_provider: str = "auto"
    # DeepSeek (key from DEEPSEEK_API_KEY)
    deepseek_model: str = "deepseek-flash"
    deepseek_rewrite_model: str = "deepseek-flash"
    deepseek_answer_effort: str = "none"             # none | low | high | max  (none = no thinking, fastest)
    deepseek_rewrite_effort: str = "none"
    deepseek_call_timeout_s: float = 60.0
    # Gemini (key from GEMINI_API_KEY)
    gemini_model: str = "gemini-3.5-flash"
    gemini_rewrite_model: str = "gemini-3.5-flash-lite"
    # tried in order when the primary model times out / is overloaded / out of quota
    gemini_fallback_models: str = "gemini-3.6-flash,gemini-3-flash-preview,gemini-3.1-flash-lite"
    gemini_call_timeout_s: float = 25.0
    gemini_answer_thinking: str = "medium"           # low | medium | high
    gemini_rewrite_thinking: str = "low"
    gemini_rpm: float = 10.0                         # client-side pacing (free tier ~10; paid Tier 1: raise, e.g. 300)
    gemini_max_concurrency: int = 3                  # parallel questions with Gemini (paid: 10)
    # Anthropic (key from ANTHROPIC_API_KEY)
    llm_model: str = "claude-opus-5-5"
    rewrite_model: str = "claude-opus-5-5"
    answer_effort: str = "medium"
    rewrite_effort: str = "low"
    llm_fallbacks: bool = True                       # server-side refusal fallback ("default")
    offline: bool = False                            # force extractive mode (no LLM)

    # retrieval
    candidates: int = 60                             # lexical candidates before rerank
    rerank_pool: int = 20                            # candidates shown to the LLM reranker (recall@20 = recall@32 on the 100-question set)
    max_chunks_per_doc: int = 12                     # diversity cap in the candidate list
    pipeline_version: str = "1.1.0"
    prompt_version: str = "p7"


def _load_anthropic_env() -> None:
    """Expose ANTHROPIC_* / GEMINI_* entries of .env to os.environ (the SDKs read them there)."""
    import os
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        k, v = k.strip(), v.split("#", 1)[0].strip().strip('"').strip("'")
        if sep and k.startswith(("ANTHROPIC_", "GEMINI_", "GOOGLE_API_KEY", "DEEPSEEK_")) and v and k not in os.environ:
            os.environ[k] = v


_load_anthropic_env()
settings = Settings()
_env = __import__("os").environ
if settings.llm_provider == "gemini" or (settings.llm_provider == "auto" and not _env.get("ANTHROPIC_API_KEY")
                                         and not _env.get("DEEPSEEK_API_KEY") and (_env.get("GEMINI_API_KEY") or _env.get("GOOGLE_API_KEY"))):
    # free tier: fewer parallel calls, each question makes 2 requests.  Every HTTP call has its own
    # timeout (gemini_call_timeout_s), so the per-question limit must also cover time spent waiting
    # for a rate-limit slot; batches slower than sync_timeout_s switch to async (202 + GET).
    settings.max_concurrency = min(settings.max_concurrency, settings.gemini_max_concurrency)
    settings.question_timeout_s = max(settings.question_timeout_s, 300.0)
