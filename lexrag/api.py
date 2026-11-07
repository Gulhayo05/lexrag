"""REST API (spec: "RAG API — test spetsifikatsiyasi (lex.uz)").

    uvicorn lexrag.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from datetime import date, datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from . import llm, store
from .config import settings
from .pipeline import answer_item, error_result, system_info
from .retrieval import db

log = logging.getLogger("lexrag.api")
TZ = ZoneInfo("Asia/Tashkent")
LANGS = {"uz-Latn", "uz-Cyrl", "ru"}


class UTF8JSON(JSONResponse):
    media_type = "application/json; charset=utf-8"

    def render(self, content: Any) -> bytes:
        return json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def err(status: int, code: str, message: str, headers: dict | None = None, **extra) -> UTF8JSON:
    return UTF8JSON({"error": {"code": code, "message": message, **extra}}, status_code=status, headers=headers)


# --------------------------------------------------------------------------- #
# request models
# --------------------------------------------------------------------------- #
class Options(BaseModel):
    top_k: int = Field(5, ge=1, le=20)
    return_contexts: bool = True
    return_debug: bool = False
    temperature: float | None = Field(0, ge=0, le=2)
    seed: int | None = None
    as_of_date: date | None = None
    answer_language: Literal["auto", "uz-Latn", "uz-Cyrl", "ru"] = "auto"
    # opt-in: drop contexts whose rerank_score is below this (top_k becomes an upper bound).
    # Default None = always return top_k contexts, as in the spec.
    min_rerank_score: float | None = Field(None, ge=0, le=1)


class Item(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    question: str
    language: str | None = None

    @field_validator("question")
    @classmethod
    def _q(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must not be empty")
        if len(v) > 2000:
            raise ValueError("question must be 1-2000 characters")
        return v


class BatchRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=128)
    batch_id: str = Field(min_length=1, max_length=128)
    batch_index: int = Field(ge=1)
    total_batches: int = Field(ge=1)
    options: Options = Options()
    items: list[Item] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def _ids(self):
        ids = [i.id for i in self.items]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate item ids: {dup}")
        if self.batch_index > self.total_batches:
            raise ValueError("batch_index must be <= total_batches")
        return self


class Turn(BaseModel):
    question: str = Field(max_length=2000)
    answer: str | None = Field(default=None, max_length=8000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    id: str = "q"
    language: str | None = None
    options: Options = Options()
    history: list[Turn] = Field(default_factory=list, max_length=5,
                                description="chat memory: up to 5 previous turns, oldest first (for follow-up questions)")


# --------------------------------------------------------------------------- #
app = FastAPI(title="lex.uz RAG API", version=settings.pipeline_version, default_response_class=UTF8JSON)


@app.exception_handler(RequestValidationError)
async def _validation(_req: Request, exc: RequestValidationError):
    details = [{"loc": [str(x) for x in e.get("loc", [])], "msg": e.get("msg")} for e in exc.errors()]
    return err(400, "VALIDATION_ERROR", "request validation failed", details=details)


def auth(authorization: str | None = Header(default=None)) -> str:
    keys = {k.strip() for k in settings.api_keys.split(",") if k.strip()}
    if not authorization or not authorization.lower().startswith("bearer "):
        raise _HTTP(401, "UNAUTHORIZED", "missing Authorization: Bearer <API_KEY>")
    key = authorization[7:].strip()
    if key not in keys:
        raise _HTTP(403, "FORBIDDEN", "invalid API key")
    return key


class _HTTP(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        self.status, self.code, self.message, self.headers = status, code, message, headers


@app.exception_handler(_HTTP)
async def _http(_req: Request, exc: _HTTP):
    return err(exc.status, exc.code, exc.message, exc.headers)


# token bucket rate limiter (per API key)
_buckets: dict[str, tuple[float, float]] = {}


def rate_limit(key: str = Depends(auth)) -> str:
    rate, burst = settings.rate_limit_per_sec, max(2.0, settings.rate_limit_per_sec * 2)
    now = time.monotonic()
    tokens, last = _buckets.get(key, (burst, now))
    tokens = min(burst, tokens + (now - last) * rate)
    if tokens < 1:
        retry = max(1, int((1 - tokens) / rate + 0.999))
        _buckets[key] = (tokens, now)
        raise _HTTP(429, "RATE_LIMITED", "too many requests", {"Retry-After": str(retry)})
    _buckets[key] = (tokens - 1, now)
    return key


# --------------------------------------------------------------------------- #
_sem: asyncio.Semaphore | None = None
_running: dict[str, asyncio.Task] = {}
_STARTED = time.time()  # rows still "processing" from before this process started were interrupted


def _interrupted(batch_id: str, row: dict) -> bool:
    """A batch left in "processing" by a previous server process (restart / crash mid-batch).

    Only rows last updated before this process started count, so a batch that another
    worker of the same deployment is still processing is never taken over."""
    return (row["status"] == "processing" and batch_id not in _running
            and (row.get("updated_at") or 0) < _STARTED)


def _interrupted_response(batch_id: str, row: dict) -> dict:
    return {"run_id": row.get("run_id"), "batch_id": batch_id, "status": "failed",
            "error": {"code": "INTERRUPTED", "message": "batch processing was interrupted by a server restart; "
                      "resend the same request with the same batch_id", "retryable": True}}


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(settings.max_concurrency)
    return _sem


async def _one(item: dict, options: dict) -> dict:
    lang = item.get("language")
    if lang and lang not in LANGS:
        return error_result(item, "INVALID_LANGUAGE", f"unsupported language {lang!r}; use uz-Latn, uz-Cyrl or ru", False)
    async with _semaphore():
        t0 = time.perf_counter()
        ms = lambda: int((time.perf_counter() - t0) * 1000)  # noqa: E731
        try:
            return await asyncio.wait_for(answer_item(item, options), timeout=settings.question_timeout_s)
        except asyncio.TimeoutError:
            return error_result(item, "LLM_TIMEOUT", f"question exceeded {settings.question_timeout_s:.0f}s", True,
                                status="timeout", total_ms=ms())
        except llm.LLMError as e:
            return error_result(item, e.code, e.message, e.retryable, total_ms=ms())
        except Exception as e:  # noqa: BLE001 - one question must never fail the batch
            log.exception("question %s failed", item.get("id"))
            return error_result(item, "INTERNAL_ERROR", str(e)[:300], True, total_ms=ms())


async def _process(req: dict, received: datetime, previous: dict | None = None) -> dict:
    opts = req["options"]
    keep = {}
    if previous:  # retry of a batch: keep good answers, recompute retryable failures only
        keep = {r["id"]: r for r in previous["results"] if not (r.get("error") or {}).get("retryable")}
    todo = [i for i in req["items"] if i["id"] not in keep]
    results = await asyncio.gather(*(_one(i, opts) for i in todo))
    by_id = {**keep, **{r["id"]: r for r in results}}
    ordered = [by_id[i["id"]] for i in req["items"]]
    done = datetime.now(TZ)
    summary = {"total": len(ordered), "ok": 0, "no_answer": 0, "error": 0, "timeout": 0}
    for r in ordered:
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    return {
        "run_id": req["run_id"], "batch_id": req["batch_id"], "batch_index": req["batch_index"],
        "total_batches": req["total_batches"], "status": "completed", "system": system_info(),
        "timing": {"received_at": received.isoformat(timespec="seconds"), "completed_at": done.isoformat(timespec="seconds"),
                   "total_ms": int((done - received).total_seconds() * 1000)},
        "summary": summary, "results": ordered,
    }


async def _run_and_store(batch_id: str, req: dict, received: datetime, previous: dict | None) -> dict:
    try:
        resp = await _process(req, received, previous)
        store.finish_batch(batch_id, "completed", resp)
        return resp
    except Exception as e:  # noqa: BLE001
        log.exception("batch %s failed", batch_id)
        resp = {"run_id": req["run_id"], "batch_id": batch_id, "status": "failed", "error": {"code": "INTERNAL_ERROR", "message": str(e)[:300], "retryable": True}}
        store.finish_batch(batch_id, "failed", resp)
        return resp
    finally:
        _running.pop(batch_id, None)


@app.post("/v1/rag/batch")
async def post_batch(body: BatchRequest, request: Request, _key: str = Depends(rate_limit),
                     prefer: str | None = Header(default=None)):
    received = datetime.now(TZ)
    req = body.model_dump(mode="json")
    content = {k: req[k] for k in ("run_id", "batch_index", "total_batches", "options", "items")}
    h = store.canonical_hash(content)
    created, existing = store.claim_batch(body.batch_id, body.run_id, h)
    previous = None
    if not created:
        if existing["request_hash"] != h:
            return err(409, "BATCH_ID_CONFLICT", "this batch_id was already used with different content; use a new batch_id")
        interrupted = _interrupted(body.batch_id, existing)
        if existing["status"] == "processing" and not interrupted:
            return UTF8JSON({"batch_id": body.batch_id, "status": "processing"}, status_code=202)
        prev = existing["response"] or {}
        has_retryable = any((r.get("error") or {}).get("retryable") for r in prev.get("results", []))
        if existing["status"] == "completed" and not has_retryable:
            return UTF8JSON(prev, headers={"Idempotent-Replay": "true"})
        # an interrupted retry may still hold the earlier completed results: keep their good answers
        previous = prev if (existing["status"] == "completed" or interrupted) and prev.get("results") else None
        store.finish_batch(body.batch_id, "processing", prev or None)

    task = asyncio.create_task(_run_and_store(body.batch_id, req, received, previous))
    _running[body.batch_id] = task
    if prefer and "respond-async" in prefer.lower():
        return UTF8JSON({"batch_id": body.batch_id, "status": "processing"}, status_code=202)
    try:
        resp = await asyncio.wait_for(asyncio.shield(task), timeout=settings.sync_timeout_s)
    except asyncio.TimeoutError:
        return UTF8JSON({"batch_id": body.batch_id, "status": "processing"}, status_code=202,
                        headers={"Location": f"/v1/rag/batch/{body.batch_id}", "Retry-After": "10"})
    if resp.get("status") == "failed":
        return err(500, "INTERNAL_ERROR", resp["error"]["message"], retryable=True)
    return UTF8JSON(resp)


@app.get("/v1/rag/batch/{batch_id}")
async def get_batch(batch_id: str, _key: str = Depends(auth)):
    row = store.get_batch(batch_id)
    if row is None:
        return err(404, "NOT_FOUND", f"unknown batch_id {batch_id!r}")
    if _interrupted(batch_id, row):
        # not persisted: a resend of the same batch_id recomputes and keeps any earlier good answers
        return UTF8JSON(_interrupted_response(batch_id, row))
    if row["status"] == "processing":
        return UTF8JSON({"batch_id": batch_id, "status": "processing"}, status_code=202, headers={"Retry-After": "10"})
    return UTF8JSON(row["response"])


@app.post("/v1/rag/query")
async def post_query(body: QueryRequest, _key: str = Depends(rate_limit)):
    received = datetime.now(TZ)
    r = await _one({"id": body.id, "question": body.question, "language": body.language,
                    "history": [t.model_dump() for t in body.history]}, body.options.model_dump(mode="json"))
    done = datetime.now(TZ)
    return UTF8JSON({"system": system_info(), "timing": {"received_at": received.isoformat(timespec="seconds"),
                     "completed_at": done.isoformat(timespec="seconds"), "total_ms": int((done - received).total_seconds() * 1000)},
                     "result": r})


DEMO_HTML = Path(__file__).with_name("static") / "chat.html"
RUNS_DIR = Path(__file__).resolve().parent.parent / "eval" / "runs"


@app.get("/v1/admin/overview", include_in_schema=False)
async def admin_overview(_key: str = Depends(auth)):
    """Staff panel: system state + evaluation runs (spec section 6 metrics vs targets)."""
    runs = []
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            mf = d / "metrics.json"
            if not mf.exists():
                continue
            rep = json.loads(mf.read_text(encoding="utf-8"))
            cats: dict[str, dict] = {}
            for q in rep.get("per_question", []):
                c = cats.setdefault(q.get("category") or "?", {"n": 0, "errors": 0, "correctness": [], "misses": 0})
                c["n"] += 1
                c["errors"] += q.get("status") in ("error", "timeout")
                c["misses"] += bool(q.get("attribution"))
                if q.get("judge"):
                    c["correctness"].append(q["judge"]["correctness"])
            for c in cats.values():
                c["correctness"] = round(sum(c["correctness"]) / len(c["correctness"]), 2) if c["correctness"] else None
            meta = {}
            if (d / "meta.json").exists():
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            runs.append({"run_id": d.name, "modified": datetime.fromtimestamp(mf.stat().st_mtime, TZ).isoformat(timespec="minutes"),
                         "metrics": rep.get("metrics"), "targets": rep.get("targets"), "categories": cats,
                         "system": {k: meta.get(k) for k in ("pipeline_version", "llm_model", "index_version", "lex_snapshot_date")}})
    return UTF8JSON({"system": system_info(), "llm_provider": llm.provider(), "runs": runs[:20]})


@app.get("/", include_in_schema=False)
async def demo_page():
    """Browser demo: ask a question, see the answer, citations and retrieved contexts."""
    return HTMLResponse(DEMO_HTML.read_text(encoding="utf-8"))


@app.get("/v1/meta")
async def meta():
    return UTF8JSON({**system_info(), "languages": sorted(LANGS), "max_batch_items": 10,
                     "sync_timeout_s": settings.sync_timeout_s, "rate_limit_per_sec": settings.rate_limit_per_sec,
                     "result_retention_days": 30, "context_filter_option": "options.min_rerank_score (opt-in, default off)", "error_codes": ["RETRIEVAL_FAILED", "INDEX_UNAVAILABLE", "LLM_TIMEOUT",
                     "LLM_ERROR", "CONTEXT_TOO_LONG", "INVALID_LANGUAGE", "INTERNAL_ERROR", "INTERRUPTED"]})


@app.get("/v1/health")
async def health():
    checks = {}
    try:
        db().execute("SELECT 1 FROM docs LIMIT 1").fetchone()
        checks["index"] = "ok"
    except Exception as e:  # noqa: BLE001
        checks["index"] = f"unavailable: {e}"
    checks["llm"] = f"{llm.provider()}: {llm.model_id()}" if llm.available() else "not configured (extractive fallback)"
    try:
        store.con().execute("SELECT 1").fetchone()
        checks["results_store"] = "ok"
    except Exception as e:  # noqa: BLE001
        checks["results_store"] = f"error: {e}"
    ok = checks["index"] == "ok" and checks["results_store"] == "ok"
    return UTF8JSON({"status": "ok" if ok else "degraded", "checks": checks, "time": datetime.now(TZ).isoformat(timespec="seconds")},
                    status_code=200 if ok else 503)
