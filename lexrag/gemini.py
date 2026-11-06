"""Google Gemini provider (REST via httpx; no extra package needed).

Same contract as the Claude path in :mod:`lexrag.llm`: one call returns a
validated Pydantic object + token usage, errors are mapped to ``LLMError``.

Robustness for the free tier:
* client-side request pacing (``LEXRAG_GEMINI_RPM``) + retry on 429 honouring
  ``retryDelay`` when the API returns one
* if the endpoint rejects ``responseJsonSchema`` or ``thinkingConfig`` the call
  is retried without it (schema then goes into the prompt) and the feature is
  switched off for the rest of the process
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

import httpx
from pydantic import BaseModel, ValidationError

from .config import settings

BASE = "https://generativelanguage.googleapis.com/v1beta"
_transport: httpx.AsyncBaseTransport | None = None  # tests inject a MockTransport
_client: httpx.AsyncClient | None = None
_client_loop = None
_features = {"json_schema": True, "thinking": True}
_pace_lock: asyncio.Lock | None = None
_last_call = 0.0


class GeminiError(Exception):
    def __init__(self, code: str, message: str, retryable: bool):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


def api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _http() -> httpx.AsyncClient:
    """One client (and pacing lock) per event loop."""
    global _client, _client_loop, _pace_lock
    loop = asyncio.get_running_loop()
    if _client is None or _client_loop is not loop:
        _client = httpx.AsyncClient(base_url=BASE, timeout=settings.gemini_call_timeout_s, transport=_transport)
        _client_loop, _pace_lock = loop, asyncio.Lock()
    return _client


def _inline_refs(schema: dict) -> dict:
    """Pydantic emits $defs/$ref; inline them (simpler for the API, no recursion here)."""
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(dict(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


async def _pace() -> None:
    """Spread requests to stay under the free-tier requests-per-minute limit."""
    global _last_call
    gap = 60.0 / max(1.0, settings.gemini_rpm)
    _http()
    async with _pace_lock:
        wait = _last_call + gap - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call = time.monotonic()


def _retry_delay(resp: httpx.Response) -> float | None:
    try:
        for d in resp.json().get("error", {}).get("details", []):
            if "retryDelay" in d:
                return float(str(d["retryDelay"]).rstrip("s"))
    except (ValueError, AttributeError):
        pass
    ra = resp.headers.get("retry-after")
    return float(ra) if ra and ra.replace(".", "").isdigit() else None


def _extract_json(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    if not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            text = text[i:j + 1]
    return text


async def generate(model: str, thinking: str | None, system: str, user: str, schema: type[BaseModel],
                   max_tokens: int, seed: int | None = None) -> tuple[BaseModel, dict]:
    key = api_key()
    if not key:
        raise GeminiError("LLM_ERROR", "GEMINI_API_KEY is not set", False)
    json_schema = _inline_refs(schema.model_json_schema())
    attempts, max_attempts = 0, 4
    while True:
        gen: dict = {"responseMimeType": "application/json", "maxOutputTokens": max_tokens, "temperature": 0}
        if seed is not None:
            gen["seed"] = seed
        sys_text = system
        if _features["json_schema"]:
            gen["responseJsonSchema"] = json_schema
        else:
            sys_text += "\n\nReturn ONLY a JSON object that validates against this JSON Schema:\n" + json.dumps(json_schema, ensure_ascii=False)
        if thinking and _features["thinking"]:
            gen["thinkingConfig"] = {"thinkingLevel": thinking.upper()}
        body = {
            "systemInstruction": {"parts": [{"text": sys_text}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }
        await _pace()
        try:
            resp = await _http().post(f"/models/{model}:generateContent", json=body, headers={"x-goog-api-key": key})
        except httpx.TimeoutException as e:
            raise GeminiError("LLM_TIMEOUT", f"{model}: no response within {settings.gemini_call_timeout_s:.0f}s", True) from e
        except httpx.TransportError as e:
            raise GeminiError("LLM_ERROR", f"Gemini connection error: {e!r}"[:300], True) from e

        if resp.status_code == 400:
            msg = resp.text[:500]
            if _features["json_schema"] and re.search(r"responseJsonSchema|response_json_schema|schema", msg, re.I):
                _features["json_schema"] = False
                continue
            if _features["thinking"] and re.search(r"thinking", msg, re.I):
                _features["thinking"] = False
                continue
            code = "CONTEXT_TOO_LONG" if re.search(r"token|too long|exceeds", msg, re.I) else "LLM_ERROR"
            raise GeminiError(code, f"Gemini 400: {msg}", False)
        if resp.status_code in (401, 403):
            raise GeminiError("LLM_ERROR", "Gemini authentication failed (check GEMINI_API_KEY)", False)
        if resp.status_code == 404:
            raise GeminiError("LLM_ERROR", f"Gemini model {model!r} not found; set LEXRAG_GEMINI_MODEL "
                                           "(python3 -m lexrag.gemini lists available models)", False)
        if resp.status_code in (429, 500, 502, 503, 504):
            attempts += 1
            if attempts >= (2 if resp.status_code == 503 else max_attempts):
                what = "rate limited (free-tier quota?)" if resp.status_code == 429 else f"server error {resp.status_code}"
                raise GeminiError("LLM_ERROR", f"Gemini {model} {what}", True)
            delay = _retry_delay(resp) or (5 * 3 ** (attempts - 1))
            await asyncio.sleep(min(delay, 60))
            continue
        if resp.status_code != 200:
            raise GeminiError("LLM_ERROR", f"Gemini HTTP {resp.status_code}: {resp.text[:300]}", resp.status_code >= 500)

        data = resp.json()
        if data.get("promptFeedback", {}).get("blockReason"):
            raise GeminiError("LLM_ERROR", f"prompt blocked: {data['promptFeedback']['blockReason']}", False)
        cands = data.get("candidates") or []
        if not cands:
            raise GeminiError("LLM_ERROR", "Gemini returned no candidates", True)
        cand = cands[0]
        finish = cand.get("finishReason", "STOP")
        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
        if finish == "MAX_TOKENS":
            raise GeminiError("LLM_ERROR", "incomplete output (MAX_TOKENS)", True)
        if finish not in ("STOP", "FINISH_REASON_UNSPECIFIED") and not text:
            raise GeminiError("LLM_ERROR", f"generation stopped: {finish}", False)
        try:
            parsed = schema.model_validate_json(_extract_json(text))
        except ValidationError as e:
            attempts += 1
            if attempts < 2:
                continue  # one retry on malformed JSON
            raise GeminiError("LLM_ERROR", f"invalid structured output: {str(e)[:200]}", True) from e
        um = data.get("usageMetadata", {})
        usage = {"input_tokens": um.get("promptTokenCount", 0),
                 "output_tokens": um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0),
                 "cache_read_input_tokens": um.get("cachedContentTokenCount", 0)}
        return parsed, usage


async def list_models() -> list[str]:
    resp = await _http().get("/models", headers={"x-goog-api-key": api_key() or ""}, params={"pageSize": 200})
    resp.raise_for_status()
    return [m["name"].removeprefix("models/") for m in resp.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])]


if __name__ == "__main__":  # python3 -m lexrag.gemini  -> models available to your key
    for name in asyncio.run(list_models()):
        print(name)
