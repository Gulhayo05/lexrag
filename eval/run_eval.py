"""Tester-side client + metrics for the lex.uz RAG API (spec sections 1, 3, 5, 6).

Run a test set against the API (sequential batches, spec retry policy), save
every batch response to ``eval/runs/<run_id>/bNN.json`` and compute metrics:

    python eval/run_eval.py run  --testset eval/testset_sample.csv --base-url http://localhost:8000 --api-key KEY
    python eval/run_eval.py score --testset eval/testset_sample.csv --run-id eval-2026-11-06 [--judge]
    python eval/run_eval.py compare eval-2026-11-03 eval-2026-11-06

Expected answers / sources are never sent to the API.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import httpx

RUNS = Path(__file__).resolve().parent / "runs"
TARGETS = {  # spec section 6 (first-run proposals)
    "hit5_doc": (">=", 0.90), "hit5_article": (">=", 0.80), "mrr": (">=", 0.70), "context_precision": (">=", 0.60),
    "answer_correctness": (">=", 1.6), "faithfulness": (">=", 0.95), "citation_accuracy": (">=", 0.90),
    "fabricated_citations": ("==", 0), "outdated_citations": ("==", 0), "no_answer_accuracy": (">=", 0.9),
    "latency_p50_s": ("<=", 5.0), "latency_p95_s": ("<=", 12.0), "error_rate": ("<=", 0.02),
}


def _project_test_key() -> str:
    """First key of LEXRAG_API_KEYS from the project's .env (so the harness works out of the box)."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("LEXRAG_API_KEYS="):
                return line.split("=", 1)[1].split("#")[0].strip().split(",")[0]
    return "test-key-change-me"


def load_testset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["answerable"] = str(r.get("answerable", "true")).strip().lower() in ("true", "1", "yes")
        r["expected_doc_ids"] = [x.strip() for x in (r.get("expected_doc_id") or "").split(";") if x.strip()]
        r["expected_articles"] = [x.strip() for x in (r.get("expected_article") or "").split(";") if x.strip()]
    return rows


def make_batches(rows: list[dict], size: int = 10) -> list[list[dict]]:
    if all(r.get("batch") for r in rows):
        groups: dict[str, list] = defaultdict(list)
        for r in rows:
            groups[r["batch"]].append(r)
        return [groups[k] for k in sorted(groups)]
    # interleave categories so each batch is a small stratified sample
    by_cat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat[r.get("category", "")].append(r)
    mixed, cats = [], sorted(by_cat)
    while any(by_cat.values()):
        for c in cats:
            if by_cat[c]:
                mixed.append(by_cat[c].pop(0))
    n = math.ceil(len(mixed) / size)
    return [mixed[i::n] for i in range(n)]


# --------------------------------------------------------------------------- #
def post_with_policy(client: httpx.Client, url: str, body: dict, headers: dict) -> dict:
    backoff = [5, 15, 45]
    attempt = 0
    while True:
        try:
            r = client.post(url, json=body, headers=headers, timeout=180)
        except httpx.TransportError as e:
            if attempt >= 3:
                raise
            print(f"   transport error {e!r}; retry in {backoff[attempt]}s", file=sys.stderr)
            time.sleep(backoff[attempt]); attempt += 1
            continue
        if r.status_code == 200:
            data = r.json()
            retryable = [x["id"] for x in data["results"] if (x.get("error") or {}).get("retryable")]
            if retryable and attempt < 3:
                print(f"   retryable item errors {retryable}; resending same batch_id in {backoff[attempt]}s", file=sys.stderr)
                time.sleep(backoff[attempt]); attempt += 1
                continue
            return data
        if r.status_code == 202:
            bid = body["batch_id"]
            while True:
                time.sleep(int(r.headers.get("Retry-After", "5")))
                r = client.get(url.rsplit("/batch", 1)[0] + f"/batch/{bid}", headers=headers, timeout=60)
                if r.status_code == 200:
                    return r.json()
                if r.status_code != 202:
                    raise RuntimeError(f"poll failed {r.status_code}: {r.text[:300]}")
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "2"))
            time.sleep(wait)
            continue
        if r.status_code in (500, 502, 503, 504) and attempt < 3:
            print(f"   HTTP {r.status_code}; retry in {backoff[attempt]}s", file=sys.stderr)
            time.sleep(backoff[attempt]); attempt += 1
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")


def cmd_run(a) -> str:
    rows = load_testset(a.testset)
    batches = make_batches(rows)
    run_id = a.run_id or f"eval-{date.today().isoformat()}"
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {a.api_key}", "Content-Type": "application/json; charset=utf-8"}
    url = a.base_url.rstrip("/") + "/v1/rag/batch"
    with httpx.Client() as client:
        meta = client.get(a.base_url.rstrip("/") + "/v1/meta", timeout=30).json()
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        for i, b in enumerate(batches, 1):
            body = {
                "run_id": run_id, "batch_id": f"{run_id}-b{i:02d}", "batch_index": i, "total_batches": len(batches),
                "options": {"top_k": 5, "return_contexts": True, "return_debug": True, "temperature": 0, "seed": a.seed,
                            "as_of_date": a.as_of_date, "answer_language": "auto"},
                "items": [{"id": r["id"], "question": r["question"]} for r in b],
            }
            t = time.time()
            data = post_with_policy(client, url, body, headers)
            (out / f"b{i:02d}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"b{i:02d}: {data['summary']}  {time.time() - t:.1f}s", file=sys.stderr)
    print(f"saved to {out}", file=sys.stderr)
    return run_id


# --------------------------------------------------------------------------- #
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def _norm_art(a: str | None) -> str:
    """'546⁵-modda', '546-5-modda', '546^5-modda' -> '546.5-modda'."""
    a = (a or "").lower().replace(" ", "").replace("–", "-").split("(")[0]
    a = re.sub(r"^(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: m.group(1) + "." + m.group(2).translate(_SUP), a)
    return re.sub(r"^(\d+)[\^-](\d+)-", r"\1.\2-", a)


JUDGE_SYSTEM = """You grade answers of a legal QA system over Uzbekistan's legislation. Be strict and literal.
correctness (compare with the expected answer; language/script of the answer may differ, that is fine):
  2 = states the same rule with the same numbers/conditions (extra correct detail is fine)
  1 = partially correct or incomplete (misses a key condition or one of several parts)
  0 = wrong, contradicts the expected answer, or answers from a repealed act
faithfulness: split the system answer into atomic factual claims. total_claims = their number; faithful_claims = how many
are directly supported by the retrieved contexts (not by your own knowledge).
citation_supports: true if every cited quote actually supports the claim it is attached to.
relevant_contexts: the rank numbers (1-5) of the TOP-5 retrieved contexts that are relevant to the question
(contain the rule asked about or information directly needed to answer it)."""


def _judge(rows_by_id: dict, results: dict) -> dict[str, dict]:
    """LLM judge (spec section 6): answer correctness 0/1/2, faithfulness, citation support.

    Uses the active provider of lexrag (DeepSeek -> deepseek-v4-pro, a different model than the
    answering one; Gemini; Claude).  Override with JUDGE_MODEL.  Answerable questions the system
    refused count as correctness 0 without a judge call."""
    import asyncio

    from pydantic import BaseModel

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lexrag import llm as _llm  # loads .env keys

    class Verdict(BaseModel):
        correctness: int
        faithful_claims: int
        total_claims: int
        citation_supports: bool
        relevant_contexts: list[int]
        reason: str

    from lexrag.config import settings as _settings
    if os.environ.get("JUDGE_PROVIDER"):            # e.g. JUDGE_PROVIDER=gemini: grade with another vendor
        _settings.llm_provider = os.environ["JUDGE_PROVIDER"]
    default = {"deepseek": "deepseek-v4-pro", "gemini": "gemini-3.5-flash", "anthropic": "claude-opus-5-5"}
    model = os.environ.get("JUDGE_MODEL") or default.get(_llm.provider() or "", None)
    out: dict[str, dict] = {}
    todo = []
    for qid, res in results.items():
        row = rows_by_id.get(qid)
        if not row or not row["answerable"]:
            continue
        if res.get("status") != "ok":  # refused / failed on an answerable question
            out[qid] = {"correctness": 0, "faithful_claims": 0, "total_claims": 0, "citation_supports": False,
                        "relevant_contexts": None, "reason": f"system status {res.get('status')}"}
            continue
        ctx = "\n\n".join(f"[rank {c['rank']}] {c['doc_title'][:120]} | {c.get('article')}\n{c.get('text') or ''}"
                            for c in res["contexts"][:5])
        cits = "\n".join(f"[{c['ref']}] {c['doc_title']} {c['article']}: {c['quote']}" for c in res["citations"])
        prompt = (f"QUESTION: {row['question']}\nEXPECTED ANSWER: {row['expected_answer']}\n\nSYSTEM ANSWER: {res['answer']}\n\n"
                  f"CITATIONS:\n{cits}\n\nRETRIEVED CONTEXTS:\n{ctx}")
        todo.append((qid, prompt))

    async def run():
        sem = asyncio.Semaphore(8)

        async def one(qid, prompt):
            async with sem:
                try:
                    v, _ = await _llm.structured(JUDGE_SYSTEM, prompt, Verdict, model=model, effort="low", max_tokens=8000)
                    return qid, v.model_dump()
                except _llm.LLMError as e:
                    print(f"   judge failed for {qid}: {e.message}", file=sys.stderr)
                    return qid, None
        return await asyncio.gather(*(one(q, p) for q, p in todo))

    for qid, v in asyncio.run(run()):
        if v:
            v["correctness"] = max(0, min(2, int(v["correctness"])))
            out[qid] = v
    return out


def cmd_score(a) -> dict:
    rows = load_testset(a.testset)
    rows_by_id = {r["id"]: r for r in rows}
    run_dir = RUNS / a.run_id
    results = {}
    for f in sorted(run_dir.glob("b*.json")):
        for r in json.loads(f.read_text(encoding="utf-8"))["results"]:
            results[r["id"]] = r
    missing = [r["id"] for r in rows if r["id"] not in results]

    hit_doc, hit_art, rr, prec = [], [], [], []
    na_correct, na_total, errors, lat = 0, 0, 0, []
    fabricated, outdated, cit_ok, cit_total = 0, 0, 0, 0
    per_q = []
    for row in rows:
        res = results.get(row["id"])
        if res is None:
            errors += 1
            continue
        if res["status"] in ("error", "timeout"):
            errors += 1
        lat.append((res.get("timing") or {}).get("total_ms", 0) / 1000)
        ctx = res.get("contexts", [])[:5]
        ctx_ids = {c["chunk_id"] for c in res.get("contexts", [])}
        for c in res.get("citations", []):
            if c["chunk_id"] not in ctx_ids or not c.get("url", "").startswith("https://lex.uz/docs/"):
                fabricated += 1
            if c.get("doc_status") != "amalda":
                outdated += 1
        attribution = None
        if not row["answerable"]:
            na_total += 1
            ok = bool(res.get("no_answer"))
            na_correct += ok
            attribution = None if ok else "generatsiya (javobsiz savolga javob berdi)"
        else:
            docs = [c["doc_id"] for c in ctx]
            arts = [(c["doc_id"], _norm_art(c.get("article"))) for c in ctx]
            exp_docs = set(row["expected_doc_ids"])
            exp_arts = {_norm_art(x) for x in row["expected_articles"]}
            hd = any(d in exp_docs for d in docs)
            rank = next((i for i, (d, ar) in enumerate(arts, 1) if d in exp_docs and ar in exp_arts), None)
            hit_doc.append(hd)
            hit_art.append(rank is not None)
            rr.append(1 / rank if rank else 0.0)
            prec.append(sum(d in exp_docs for d in docs) / max(1, len(docs)))
            cited = {(c["doc_id"], _norm_art(c.get("article"))) for c in res.get("citations", [])}
            if res.get("citations"):
                cit_total += 1
                cit_ok += any(d in exp_docs and ar in exp_arts for d, ar in cited)
            if rank is None:
                attribution = "retrieval (toʻgʻri modda topilmadi)"
            elif res.get("no_answer"):
                attribution = "generatsiya (modda topilgan, javob berilmadi)"
            elif not any(d in exp_docs and ar in exp_arts for d, ar in cited):
                attribution = "manba (havola notoʻgʻri)"
        per_q.append({"id": row["id"], "category": row.get("category"), "status": res["status"], "attribution": attribution})

    n = len(rows)
    lat_sorted = sorted(lat) or [0]
    m = {
        "run_id": a.run_id, "questions": n, "missing": missing,
        "hit5_doc": _mean(hit_doc), "hit5_article": _mean(hit_art), "mrr": _mean(rr), "context_precision": _mean(prec),
        "citation_accuracy": cit_ok / cit_total if cit_total else None,
        "fabricated_citations": fabricated, "outdated_citations": outdated,
        "no_answer_accuracy": na_correct / na_total if na_total else None,
        "latency_p50_s": round(statistics.median(lat_sorted), 2),
        "latency_p95_s": round(lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))], 2),
        "error_rate": errors / n if n else 0,
    }
    if a.judge:
        v = _judge(rows_by_id, results)
        if v:
            m["answer_correctness"] = sum(x["correctness"] for x in v.values()) / len(v)
            answered = [x for x in v.values() if x["total_claims"]]
            tot = sum(x["total_claims"] for x in answered)
            m["faithfulness"] = round(sum(x["faithful_claims"] for x in answered) / tot, 4) if tot else None
            m["citation_accuracy_judge"] = round(sum(x["citation_supports"] for x in answered) / len(answered), 4) if answered else None
            m["answer_correctness"] = round(m["answer_correctness"], 3)
            precs = [len({r for r in x["relevant_contexts"] if 1 <= r <= 5}) / 5 for x in v.values() if x.get("relevant_contexts") is not None]
            if precs:  # spec definition: share of the top-5 contexts that are relevant to the question
                m["context_precision_proxy_same_doc"] = m["context_precision"]
                m["context_precision"] = round(sum(precs) / len(precs), 4)
            for q in per_q:
                if q["id"] in v:
                    q["judge"] = v[q["id"]]
                    if q["attribution"] is None and v[q["id"]]["correctness"] == 0:
                        q["attribution"] = "generatsiya (javob notoʻgʻri)"
    by_cause = defaultdict(int)
    for q in per_q:
        if q["attribution"]:
            by_cause[q["attribution"].split(" ")[0]] += 1
    m["error_attribution"] = dict(by_cause)
    report = {"metrics": m, "targets": _check(m), "per_question": per_q}
    (run_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    _print(m)
    return report


def _mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def _check(m: dict) -> dict:
    out = {}
    for k, (op, t) in TARGETS.items():
        v = m.get(k)
        if v is None:
            continue
        out[k] = {"value": v, "target": f"{op} {t}", "pass": (v >= t) if op == ">=" else (v <= t) if op == "<=" else (v == t)}
    return out


def _print(m: dict) -> None:
    for k, v in m.items():
        if k in TARGETS and v is not None:
            op, t = TARGETS[k]
            ok = (v >= t) if op == ">=" else (v <= t) if op == "<=" else (v == t)
            print(f"  {'PASS' if ok else 'FAIL'}  {k:22} {v!s:>8}   target {op} {t}")
        elif k not in TARGETS:
            print(f"        {k:22} {v}")


def cmd_compare(a) -> None:
    ms = []
    for rid in a.run_ids:
        p = RUNS / rid / "metrics.json"
        meta = json.loads((RUNS / rid / "meta.json").read_text()) if (RUNS / rid / "meta.json").exists() else {}
        ms.append((rid, json.loads(p.read_text())["metrics"], meta))
    keys = [k for k in TARGETS if any(m.get(k) is not None for _, m, _ in ms)]
    print(f"{'metric':24}" + "".join(f"{rid:>24}" for rid, _, _ in ms))
    for k in ("pipeline_version", "llm_model", "index_version"):
        print(f"{k:24}" + "".join(f"{str(meta.get(k))[:22]:>24}" for _, _, meta in ms))
    for k in keys:
        print(f"{k:24}" + "".join(f"{str(m.get(k)):>24}" for _, m, _ in ms))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--testset", required=True)
    r.add_argument("--base-url", default="http://localhost:8000")
    r.add_argument("--api-key", default=os.environ.get("LEXRAG_TEST_KEY") or _project_test_key())
    r.add_argument("--run-id")
    r.add_argument("--as-of-date", default=None)
    r.add_argument("--seed", type=int, default=42, help="a new seed bypasses the answer cache (fresh generation, real latency)")
    r.add_argument("--score", action="store_true", help="score right after the run")
    r.add_argument("--judge", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--testset", required=True)
    s.add_argument("--run-id", required=True)
    s.add_argument("--judge", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("run_ids", nargs="+")
    a = p.parse_args()
    if a.cmd == "run":
        a.run_id = cmd_run(a)
        if a.score:
            cmd_score(a)
    elif a.cmd == "score":
        cmd_score(a)
    else:
        cmd_compare(a)


if __name__ == "__main__":
    main()
