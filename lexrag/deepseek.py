"""DeepSeek provider (OpenAI-style /chat/completions over httpx; no extra package).

DeepSeek's JSON mode guarantees *some* JSON object but does not enforce a
schema, so the schema + an example go into the system prompt and every reply is
validated with Pydantic; blank / malformed replies are retried.
"""
from __future__ import annotations

import asyncio
import json
import os
import re

import httpx
from pydantic import BaseModel, ValidationError

from .config import settings

BASE = "https://api.deepseek.com"
_transport: httpx.AsyncBaseTransport | None = None  # tests inject a MockTransport
_client: httpx.AsyncClient | None = None
_client_loop = None


class DeepSeekError(Exception):
    def __init__(self, code: str, message: str, retryable: bool):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


def api_key() -> str | None:
    return os.environ.get("DEEPSEEK_API_KEY")


def _http() -> httpx.AsyncClient:
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is None or _client_loop is not loop:
        _client = httpx.AsyncClient(base_url=BASE, timeout=settings.deepseek_call_timeout_s, transport=_transport)
        _client_loop = loop
    return _client


def _inline(schema: dict) -> dict:
    defs = schema.pop("$defs", {})

    def walk(n):
        if isinstance(n, dict):
            if "$ref" in n:
                return walk(dict(defs[n["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in n.items() if k != "title"}
        if isinstance(n, list):
            return [walk(x) for x in n]
        return n

    return walk(schema)


def _example(s: dict):
    """Tiny example value for a JSON schema (DeepSeek asks for an example of the format)."""
    if "enum" in s:
        return s["enum"][0]
    t = s.get("type")
    if "anyOf" in s:
        return _example(s["anyOf"][0])
    if t == "object":
        return {k: _example(v) for k, v in s.get("properties", {}).items()}
    if t == "array":
        return [_example(s.get("items", {}))]
    return {"string": "...", "integer": 1, "number": 0.5, "boolean": False}.get(t, None)


def _json_text(text: str) -> str:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    if text and not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            text = text[i:j + 1]
    return text


async def generate(model: str, effort: str | None, system: str, user: str, schema: type[BaseModel],
                   max_tokens: int) -> tuple[BaseModel, dict]:
    key = api_key()
    if not key:
        raise DeepSeekError("LLM_ERROR", "DEEPSEEK_API_KEY is not set", False)
    js = _inline(schema.model_json_schema())
    sys_text = (
        f"{system}\n\nOutput format: reply with a single json object only (no prose, no markdown) that validates "
        f"against this JSON Schema:\n{json.dumps(js, ensure_ascii=False)}\n"
        f"Example of the shape (values are placeholders):\n{json.dumps(_example(js), ensure_ascii=False)}"
    )
    body = {
        "model": model,
        "messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if effort:
        body["reasoning_effort"] = effort  # "none" disables thinking
    backoff = [2, 6, 18]
    attempt, bad_json = 0, 0
    while True:
        try:
            resp = await _http().post("/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"})
        except httpx.TimeoutException as e:
            raise DeepSeekError("LLM_TIMEOUT", f"{model}: no response within {settings.deepseek_call_timeout_s:.0f}s", True) from e
        except httpx.TransportError as e:
            raise DeepSeekError("LLM_ERROR", f"DeepSeek connection error: {e!r}"[:300], True) from e

        sc = resp.status_code
        if sc in (429, 500, 502, 503, 504):
            if attempt < len(backoff):
                await asyncio.sleep(backoff[attempt])
                attempt += 1
                continue
            raise DeepSeekError("LLM_ERROR", f"DeepSeek {model} HTTP {sc} (rate limit / overloaded)", True)
        if sc == 401:
            raise DeepSeekError("LLM_ERROR", "DeepSeek authentication failed (check DEEPSEEK_API_KEY)", False)
        if sc == 402:
            raise DeepSeekError("LLM_ERROR", "DeepSeek account has insufficient balance (top up)", False)
        if sc in (400, 422):
            msg = resp.text[:400]
            if "reasoning_effort" in body and re.search(r"reasoning_effort|thinking", msg, re.I):
                body.pop("reasoning_effort")  # model without effort control
                continue
            code = "CONTEXT_TOO_LONG" if re.search(r"context|too long|maximum.*tokens", msg, re.I) else "LLM_ERROR"
            raise DeepSeekError(code, f"DeepSeek {sc}: {msg}", False)
        if sc != 200:
            raise DeepSeekError("LLM_ERROR", f"DeepSeek HTTP {sc}: {resp.text[:300]}", sc >= 500)

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        finish = choice.get("finish_reason")
        content = ((choice.get("message") or {}).get("content")) or ""
        if finish == "length":
            raise DeepSeekError("LLM_ERROR", "incomplete output (max_tokens reached)", True)
        if finish == "content_filter":
            raise DeepSeekError("LLM_ERROR", "DeepSeek content filter stopped the answer", False)
        if finish == "insufficient_system_resource":
            if attempt < len(backoff):
                await asyncio.sleep(backoff[attempt]); attempt += 1
                continue
            raise DeepSeekError("LLM_ERROR", "DeepSeek insufficient system resource", True)
        try:
            parsed = schema.model_validate_json(_json_text(content))
        except ValidationError as e:
            bad_json += 1  # blank content is a documented JSON-mode quirk -> retry
            if bad_json < 3:
                continue
            raise DeepSeekError("LLM_ERROR", f"invalid structured output: {str(e)[:200]}", True) from e
        u = data.get("usage") or {}
        usage = {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0),
                 "cache_read_input_tokens": u.get("prompt_cache_hit_tokens", 0), "model": data.get("model", model)}
        return parsed, usage


async def list_models() -> list[str]:
    resp = await _http().get("/models", headers={"Authorization": f"Bearer {api_key() or ''}"})
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


if __name__ == "__main__":  # python3 -m lexrag.deepseek -> models available to your key
    for name in asyncio.run(list_models()):
        print(name)
