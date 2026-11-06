"""Claude calls: query rewriting and grounded answer generation (with LLM rerank).

Both calls use structured outputs (``messages.parse`` + Pydantic) so the response
is always valid JSON.  Sampling parameters are not sent: current Claude models
reject ``temperature``/``seed``; reproducibility is guaranteed instead by the
generation cache in :mod:`lexrag.store` (same question + same contexts + same
seed -> the stored answer is returned).
"""
from __future__ import annotations

import logging
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from . import deepseek, gemini
from .config import settings

log = logging.getLogger("lexrag.llm")


class LLMError(Exception):
    def __init__(self, code: str, message: str, retryable: bool):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #
class Rewrite(BaseModel):
    language: Literal["uz-Latn", "uz-Cyrl", "ru"]
    question_uz: str = Field(description="The question translated / normalised into Uzbek (Latin script), legal register")
    search_queries: list[str] = Field(description="2-3 short alternative Uzbek Latin search phrasings using lex.uz legal terminology")
    doc_hints: list[str] = Field(description="Names of the most likely acts, e.g. 'Mehnat kodeksi', 'Fuqarolik kodeksi', 'Soliq kodeksi'. Empty if unsure")
    article_hints: list[str] = Field(description="Article numbers explicitly mentioned or confidently known, e.g. '216-modda'. Usually empty")
    in_scope: bool = Field(description="False if the question is not about law / legal regulation of Uzbekistan at all")


class Relevance(BaseModel):
    ctx: int
    score: float = Field(description="0..1 how useful this context is for answering")


class CitationOut(BaseModel):
    ref: int = Field(description="Marker number used in the answer text, e.g. 1 for [1]")
    ctx: int = Field(description="Context number Cn that supports the claim")
    quote: str = Field(description="Verbatim excerpt copied exactly from that context (one sentence or clause)")


class AnswerOut(BaseModel):
    relevance: list[Relevance]
    no_answer: bool
    answer: str
    citations: list[CitationOut]
    confidence: float = Field(description="0..1 probability that the answer is correct and complete")


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
REWRITE_SYSTEM = """You are the query-understanding step of a retrieval system over the official legislation database of Uzbekistan (lex.uz). The index contains Uzbek-language (Latin script) texts of codes, laws, decrees and resolutions.

Given a user question (Uzbek Latin, Uzbek Cyrillic or Russian, possibly colloquial), produce search inputs:
- question_uz: faithful translation into formal Uzbek Latin (use the official letters oʻ, gʻ and ʼ).
- search_queries: 2-3 short Uzbek Latin phrasings with the exact legal terms the statute would use (e.g. "yillik asosiy taʼtil muddati", "daʼvo muddati umumiy muddat"). Expand colloquial words into legal terms.
- doc_hints: the act(s) most likely to regulate this (e.g. "Mehnat kodeksi", "Fuqarolik kodeksi", "Jinoyat kodeksi", "Soliq kodeksi", "Oila kodeksi", "Konstitutsiya", or a specific law "Iste'molchilarning huquqlarini himoya qilish toʻgʻrisidagi qonun"). Leave empty when unsure.
- article_hints: only article numbers mentioned in the question or that you know with certainty for the CURRENT edition; otherwise empty.
- in_scope: false when the question is not about the law of Uzbekistan: everyday advice, opinions, predictions, how-to questions, or a fictional/impossible premise (a non-existent ministry or institution, property on another planet, etc.). Questions that a real statute regulates are in scope even if phrased casually.

Always target the law IN FORCE today. If the question contains a premise about an old or repealed act (e.g. "1995-yilgi kodeks boʻyicha 15 ish kuni edi"), do not repeat that premise or its numbers in search_queries; phrase the queries for the current act instead. The current Mehnat kodeksi (2022), Soliq kodeksi (2019) and Konstitutsiya (2023) use newer terminology (e.g. "har yilgi asosiy mehnat taʼtili", "davomiyligi"); include such official wording as one of the variants when relevant."""

ANSWER_SYSTEM = """You answer legal questions strictly from provided excerpts of Uzbekistan legislation (lex.uz). You never use outside knowledge for the substance of the answer.

Input: a question, the answer language, and numbered contexts [C1]..[Cn]. Each context shows the act title, its type, its status (amalda = in force, kuchini_yoqotgan = no longer in force), its edition date and the article.

Do the following:
1. relevance: list ONLY the contexts that are relevant to the question (score >= 0.3), each with a score 0..1 (1 = contains the rule asked about). Omit unrelated contexts; they count as 0.
2. Decide no_answer. Set no_answer=true when the contexts do not contain the rule that answers THIS question, when the question is outside law (everyday advice, opinions, predictions, how-to questions), when its premise is fictional or impossible (a non-existent ministry, institution, place or situation), or when only repealed (kuchini_yoqotgan) acts contain it. A loosely related regulation is not an answer. Do not guess. Then answer with one short sentence in the answer language saying the provided legislation does not answer the question, and return no citations.
3. Otherwise write a concise answer (1-4 sentences) in the answer language. Every factual claim ends with a marker like [1]. Copy numbers, terms, amounts, percentages and BHM (bazaviy hisoblash miqdori) multiples exactly as written in the context; never convert units. Prefer in-force acts and the most specific article. Mention the act and article in the text (e.g. "Mehnat kodeksining 216-moddasiga koʻra ...").
4. citations: one entry per marker. ref = marker number, ctx = context number, quote = an exact, verbatim excerpt (a sentence or clause, max ~300 characters) copied character-for-character from that context that supports the claim.
5. confidence: your probability (0..1) that the answer is correct and complete.

Source priority: the constitution, codes and laws prevail over government resolutions and ministry regulations. A regulation that quotes or paraphrases an article of a code may be outdated (it may cite a repealed edition). If a code in the contexts states the rule directly, answer from the code. Never answer from text that restates a repealed act. If the question contains a premise from an old or repealed act, answer per the law in force and briefly say the rule has changed.

Some contexts may be marked "DIQQAT: … as_of_date …": that article was amended after the date the user asked about, and only the current wording is available. If you rely on such a context, say in the answer that the article has been amended since that date and the quoted text is the current edition.

Answer language codes: uz-Latn = Uzbek Latin (use oʻ, gʻ, ʼ), uz-Cyrl = Uzbek Cyrillic, ru = Russian. Quotes always stay in the original language of the context."""


def _client() -> anthropic.AsyncAnthropic:
    global _CLIENT
    try:
        return _CLIENT
    except NameError:
        _CLIENT = anthropic.AsyncAnthropic(timeout=settings.question_timeout_s, max_retries=2)
        return _CLIENT


_FALLBACK_OK = True


async def _parse(model: str, effort: str, system: str, user: str, schema, max_tokens: int):
    global _FALLBACK_OK
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_format=schema,
        output_config={"effort": effort},
    )
    if settings.llm_fallbacks and _FALLBACK_OK:
        kwargs["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
        kwargs["extra_body"] = {"fallbacks": "default"}
    try:
        resp = await _client().messages.parse(**kwargs)
    except anthropic.BadRequestError as e:
        msg = str(e)
        if "fallback" in msg.lower() and "extra_body" in kwargs:
            _FALLBACK_OK = False  # endpoint/proxy without fallback support
            kwargs.pop("extra_headers"), kwargs.pop("extra_body")
            resp = await _client().messages.parse(**kwargs)
        elif "too long" in msg.lower() or "context" in msg.lower() and "token" in msg.lower():
            raise LLMError("CONTEXT_TOO_LONG", msg[:300], False) from e
        else:
            raise LLMError("LLM_ERROR", msg[:300], False) from e
    if resp.stop_reason == "refusal":
        raise LLMError("LLM_ERROR", "model declined the request", False)
    if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
        raise LLMError("LLM_ERROR", f"incomplete structured output (stop_reason={resp.stop_reason})", True)
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
             "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0}
    return resp.parsed_output, usage


async def call(fn, *args):
    try:
        return await fn(*args)
    except LLMError:
        raise
    except anthropic.APITimeoutError as e:
        raise LLMError("LLM_TIMEOUT", f"generation exceeded {settings.question_timeout_s:.0f}s", True) from e
    except anthropic.RateLimitError as e:
        raise LLMError("LLM_ERROR", "LLM rate limited", True) from e
    except anthropic.AuthenticationError as e:
        raise LLMError("LLM_ERROR", "LLM authentication failed (check ANTHROPIC_API_KEY)", False) from e
    except anthropic.APIStatusError as e:
        raise LLMError("LLM_ERROR", f"LLM API error {e.status_code}", e.status_code >= 500) from e
    except anthropic.APIConnectionError as e:
        raise LLMError("LLM_ERROR", "LLM connection error", True) from e


_DISABLED: set[str] = set()   # providers that failed hard this process (no balance, bad key)


def provider() -> str | None:
    """Active LLM provider: 'anthropic', 'deepseek', 'gemini' or None (extractive fallback).

    LEXRAG_LLM_PROVIDER is the preferred one; if it has no key or failed hard (e.g. DeepSeek
    balance exhausted), the next provider with a key takes over (order: anthropic, deepseek, gemini)."""
    import os
    if settings.offline:
        return None
    keys = {
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
        "deepseek": bool(deepseek.api_key()),
        "gemini": bool(gemini.api_key()),
    }
    pref = settings.llm_provider.lower()
    order = ([pref] if pref in keys else []) + [p for p in ("anthropic", "deepseek", "gemini") if p != pref]
    for p in order:
        if keys[p] and p not in _DISABLED:
            return p
    return None


_HARD_FAIL = ("insufficient balance", "authentication failed", "check deepseek_api_key", "check gemini_api_key",
              "check anthropic_api_key")


def _with_fallback(fn):
    """Retry once on the next provider when the current one cannot work at all (no balance / bad key)."""
    import functools

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        for _ in range(3):
            current = provider()
            try:
                return await fn(*args, **kwargs)
            except LLMError as e:
                if current and any(h in e.message.lower() for h in _HARD_FAIL):
                    _DISABLED.add(current)
                    log.warning("LLM provider %s disabled (%s); falling back to %s", current, e.message, provider())
                    if provider():
                        continue
                raise
        return await fn(*args, **kwargs)
    return wrapper


def model_id(kind: str = "answer") -> str | None:
    p = provider()
    if p == "anthropic":
        return settings.llm_model if kind == "answer" else settings.rewrite_model
    if p == "gemini":
        return settings.gemini_model if kind == "answer" else settings.gemini_rewrite_model
    if p == "deepseek":
        return settings.deepseek_model if kind == "answer" else settings.deepseek_rewrite_model
    return None


async def _deepseek(model, effort, system, user, schema, max_tokens):
    try:
        return await deepseek.generate(model, effort, system, user, schema, max_tokens)
    except deepseek.DeepSeekError as e:
        raise LLMError(e.code, e.message, e.retryable) from e


LAST_MODEL: dict[str, str] = {}


async def _gemini(model, thinking, system, user, schema, max_tokens, seed=None, kind="answer"):
    """Primary model, then the fallback chain on retryable failures (timeout, 503, exhausted quota)."""
    chain = [model] + [m.strip() for m in settings.gemini_fallback_models.split(",") if m.strip() and m.strip() != model]
    errors = []
    for m in chain:
        try:
            parsed, usage = await gemini.generate(m, thinking, system, user, schema, max_tokens, seed)
            LAST_MODEL[kind] = m
            return parsed, {**usage, "model": m}
        except gemini.GeminiError as e:
            errors.append(e)
            if not e.retryable and "not found" not in e.message:
                break
    e = errors[-1]
    raise LLMError(e.code, "; ".join(x.message for x in errors)[:500], e.retryable)


def _rewrite_user(question: str, history: list[dict] | None) -> str:
    if not history:
        return f"Question:\n{question}"
    turns = "\n".join(f"User: {h.get('question', '')}\nAssistant: {(h.get('answer') or '')[:600]}" for h in history[-5:])
    return (f"Conversation so far (most recent last):\n{turns}\n\nNew question:\n{question}\n\n"
            "The new question may be a follow-up (\"va bu qaysi moddada?\", \"a agar ...?\"). Make question_uz a fully "
            "self-contained question that includes everything needed from the conversation; base search_queries and "
            "doc_hints on it.")


@_with_fallback
async def rewrite(question: str, history: list[dict] | None = None) -> tuple[Rewrite, dict]:
    user = _rewrite_user(question, history)
    if provider() == "deepseek":
        return await _deepseek(settings.deepseek_rewrite_model, settings.deepseek_rewrite_effort, REWRITE_SYSTEM,
                               user, Rewrite, 2000)
    if provider() == "gemini":
        return await _gemini(settings.gemini_rewrite_model, settings.gemini_rewrite_thinking, REWRITE_SYSTEM,
                             user, Rewrite, 4096, 42, "rewrite")
    return await call(_parse, settings.rewrite_model, settings.rewrite_effort, REWRITE_SYSTEM, user, Rewrite, 2000)


@_with_fallback
async def answer(question: str, language: str, contexts_block: str, seed: int | None = None) -> tuple[AnswerOut, dict]:
    user = f"Answer language: {language}\n\nQuestion:\n{question}\n\nContexts:\n{contexts_block}"
    if provider() == "deepseek":
        return await _deepseek(settings.deepseek_model, settings.deepseek_answer_effort, ANSWER_SYSTEM, user, AnswerOut, 16000)
    if provider() == "gemini":
        return await _gemini(settings.gemini_model, settings.gemini_answer_thinking, ANSWER_SYSTEM, user, AnswerOut,
                             16384, seed if seed is not None else 42, "answer")
    return await call(_parse, settings.llm_model, settings.answer_effort, ANSWER_SYSTEM, user, AnswerOut, 8000)


def available() -> bool:
    return provider() is not None


async def structured(system: str, user: str, schema, model: str | None = None, effort: str | None = None,
                     max_tokens: int = 8000):
    """Provider-neutral structured call (used by the test-set builder and the eval judge)."""
    p = provider()
    if p == "deepseek":
        return await _deepseek(model or settings.deepseek_model, effort or "none", system, user, schema, max_tokens)
    if p == "gemini":
        return await _gemini(model or settings.gemini_model, effort or "low", system, user, schema, max_tokens, 42, "aux")
    if p == "anthropic":
        return await call(_parse, model or settings.llm_model, effort or "low", system, user, schema, max_tokens)
    raise LLMError("LLM_ERROR", "no LLM provider configured", False)
