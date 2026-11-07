"""Acceptance tests = the developer checklist of the spec (section 7).

    python -m unittest discover -s tests -v

Runs against the real index.  The LLM is replaced by a deterministic fake so the
tests need no API key and check the contract, not model quality.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_tmp = tempfile.mkdtemp()
os.environ["LEXRAG_RESULTS_DB"] = os.path.join(_tmp, "batches.sqlite")
os.environ["LEXRAG_API_KEYS"] = "test-key"
os.environ["LEXRAG_RATE_LIMIT_PER_SEC"] = "1000"
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-for-tests")

from fastapi.testclient import TestClient  # noqa: E402

from lexrag import api, llm, pipeline  # noqa: E402
from lexrag.config import settings  # noqa: E402

H = {"Authorization": "Bearer test-key"}
CALLS = {"answer": 0, "fail_ids": set()}


async def fake_rewrite(question: str, history=None):
    hint = []
    if "Гражданск" in question or "da'vo" in question.lower():
        hint = ["Fuqarolik kodeksi"]
    if "ta'til" in question.lower() or "отпуск" in question.lower():
        hint = ["Mehnat kodeksi"]
    q_uz = {"Какой срок исковой давности установлен Гражданским кодексом?": "Umumiy daʼvo muddati qancha"}.get(question, question)
    return llm.Rewrite(language="uz-Latn", question_uz=q_uz, search_queries=[], doc_hints=hint, article_hints=[],
                       in_scope="mars" not in question.lower()), {"input_tokens": 10, "output_tokens": 5}


async def fake_answer(question: str, language: str, block: str, seed=None):
    CALLS["answer"] += 1
    if "FAILME" in question:
        raise llm.LLMError("LLM_TIMEOUT", "Generation exceeded 60s", True)
    if "mars" in question.lower():
        return llm.AnswerOut(relevance=[], no_answer=True, answer="Javob topilmadi.", citations=[], confidence=0.1), {}
    c1 = block.split("[C2]")[0].split("\n")[1:]
    quote = next(l for l in c1 if l.strip() and not l.startswith("("))[:120]  # first line of C1 text
    return llm.AnswerOut(
        relevance=[llm.Relevance(ctx=1, score=0.95), llm.Relevance(ctx=2, score=0.4)], no_answer=False,
        answer="Javob matni [1].", citations=[llm.CitationOut(ref=1, ctx=1, quote=quote),
                                              llm.CitationOut(ref=2, ctx=99, quote="tizim koʻrmagan manba")],
        confidence=0.8,
    ), {"input_tokens": 100, "output_tokens": 20}


_REAL = (llm.rewrite, llm.answer, llm.available)


def batch(n=2, batch_id="b-1", questions=None, **opts):
    qs = questions or ["Yillik asosiy mehnat ta'tilining eng kam muddati qancha?",
                       "Какой срок исковой давности установлен Гражданским кодексом?",
                       "Йиллик асосий меҳнат таътили қанча?", "Nikoh yoshi necha yosh?"] * 3
    return {"run_id": "test-run", "batch_id": batch_id, "batch_index": 1, "total_batches": 1,
            "options": {"top_k": 5, "return_debug": True, "seed": 42, **opts},
            "items": [{"id": f"q{i + 1:03d}", "question": qs[i]} for i in range(n)]}


class Checklist(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        llm.rewrite, llm.answer, llm.available = _REAL
        cls.c.__exit__(None, None, None)

    @classmethod
    def setUpClass(cls):
        llm.rewrite, llm.answer = fake_rewrite, fake_answer
        llm._DISABLED.clear()
        llm.available = lambda: True
        cls.c = TestClient(api.app).__enter__()  # keep one event loop (background tasks)

    def test_01_accepts_1_to_10_and_rejects_11(self):
        r = self.c.post("/v1/rag/batch", json=batch(10, "b-ten"), headers=H)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.json()["results"]), 10)
        r = self.c.post("/v1/rag/batch", json=batch(11, "b-eleven"), headers=H)
        self.assertEqual(r.status_code, 400)

    def test_02_validation_400(self):
        b = batch(2, "b-dup"); b["items"][1]["id"] = "q001"
        self.assertEqual(self.c.post("/v1/rag/batch", json=b, headers=H).status_code, 400)
        b = batch(1, "b-empty"); b["items"][0]["question"] = "   "
        self.assertEqual(self.c.post("/v1/rag/batch", json=b, headers=H).status_code, 400)
        b = batch(1, "b-long"); b["items"][0]["question"] = "a" * 2001
        self.assertEqual(self.c.post("/v1/rag/batch", json=b, headers=H).status_code, 400)

    def test_03_auth(self):
        self.assertEqual(self.c.post("/v1/rag/batch", json=batch(1, "b-a1")).status_code, 401)
        self.assertEqual(self.c.post("/v1/rag/batch", json=batch(1, "b-a2"), headers={"Authorization": "Bearer nope"}).status_code, 403)

    def test_04_results_ids_and_required_fields(self):
        r = self.c.post("/v1/rag/batch", json=batch(4, "b-fields"), headers=H).json()
        ids = [x["id"] for x in r["results"]]
        self.assertEqual(ids, ["q001", "q002", "q003", "q004"])
        for k in ("pipeline_version", "llm_model", "embedding_model", "reranker", "index_version", "lex_snapshot_date"):
            self.assertIn(k, r["system"])
        self.assertIn("total_ms", r["timing"])
        for res in r["results"]:
            self.assertIn(res["status"], ("ok", "no_answer", "error", "timeout"))
            self.assertTrue(res["contexts"], "contexts must be returned")
            ctx_ids = {c["chunk_id"] for c in res["contexts"]}
            for c in res["contexts"]:
                for k in ("rank", "chunk_id", "doc_id", "article", "text", "retrieval_score", "rerank_score", "used_in_answer"):
                    self.assertIn(k, c)
            for cit in res["citations"]:
                for k in ("ref", "doc_id", "url", "article", "part", "edition_date", "doc_status", "chunk_id", "quote"):
                    self.assertIn(k, cit)
                self.assertIn(cit["chunk_id"], ctx_ids, "citation must point to a returned context")
                ctx = next(c for c in res["contexts"] if c["chunk_id"] == cit["chunk_id"])
                self.assertIn(cit["quote"], " ".join(ctx["text"].split()), "quote must be verbatim")
                self.assertTrue(cit["url"].startswith("https://lex.uz/docs/"))
                self.assertIn(f"[{cit['ref']}]", res["answer"])
            # the fake model cited ctx 99 (unseen) -> must be dropped
            self.assertLessEqual(len(res["citations"]), 1)

    def test_05_no_answer(self):
        r = self.c.post("/v1/rag/batch", json=batch(1, "b-na", ["Marsda yer sotib olish tartibi qanday?"]), headers=H).json()
        res = r["results"][0]
        self.assertTrue(res["no_answer"])
        self.assertEqual(res["status"], "no_answer")
        self.assertEqual(res["citations"], [])

    def test_06_question_error_isolated_and_retry(self):
        b = batch(3, "b-err", ["Nikoh yoshi necha yosh?", "FAILME savol", "Umumiy da'vo muddati qancha?"])
        r = self.c.post("/v1/rag/batch", json=b, headers=H)
        self.assertEqual(r.status_code, 200)
        res = {x["id"]: x for x in r.json()["results"]}
        self.assertEqual(res["q002"]["status"], "error")
        self.assertEqual(res["q002"]["error"]["code"], "LLM_TIMEOUT")
        self.assertTrue(res["q002"]["error"]["retryable"])
        self.assertEqual(res["q001"]["status"], "ok")
        self.assertEqual(r.json()["summary"]["error"], 1)
        # resend with the same batch_id: good answers are kept, failed one recomputed
        before = CALLS["answer"]
        r2 = self.c.post("/v1/rag/batch", json=b, headers=H).json()
        self.assertEqual(CALLS["answer"] - before, 1, "only the retryable item is recomputed")
        self.assertEqual(r2["results"][0]["answer"], r.json()["results"][0]["answer"])

    def test_07_idempotent_and_conflict(self):
        b = batch(2, "b-idem")
        r1 = self.c.post("/v1/rag/batch", json=b, headers=H).json()
        before = CALLS["answer"]
        r2 = self.c.post("/v1/rag/batch", json=b, headers=H)
        self.assertEqual(CALLS["answer"], before, "stored result must be returned, not recomputed")
        self.assertEqual(r2.json(), r1)
        self.assertEqual(r2.headers.get("Idempotent-Replay"), "true")
        b2 = batch(3, "b-idem")
        self.assertEqual(self.c.post("/v1/rag/batch", json=b2, headers=H).status_code, 409)
        g = self.c.get("/v1/rag/batch/b-idem", headers=H)
        self.assertEqual(g.status_code, 200)
        self.assertEqual(g.json()["batch_id"], "b-idem")
        self.assertEqual(self.c.get("/v1/rag/batch/unknown", headers=H).status_code, 404)

    def test_08_deterministic_sources(self):
        a = self.c.post("/v1/rag/batch", json=batch(3, "b-det1"), headers=H).json()
        b = self.c.post("/v1/rag/batch", json=batch(3, "b-det2"), headers=H).json()
        for x, y in zip(a["results"], b["results"]):
            self.assertEqual([c["chunk_id"] for c in x["contexts"]], [c["chunk_id"] for c in y["contexts"]])
            self.assertEqual(x["answer"], y["answer"])

    def test_09_languages_and_apostrophes(self):
        q = "Oʻzbekiston Respublikasida yillik asosiy ta'til g'oyasi qanday?"
        r = self.c.post("/v1/rag/batch", json=batch(1, "b-lang", [q]), headers=H)
        self.assertIn("charset=utf-8", r.headers["content-type"])
        self.assertIn("ʻ", r.text)  # not escaped / not mangled
        r = self.c.post("/v1/rag/batch", json=batch(3, "b-lang2"), headers=H).json()
        langs = [x["answer_language"] for x in r["results"]]
        self.assertEqual(langs, ["uz-Latn", "ru", "uz-Cyrl"])
        b = batch(1, "b-badlang"); b["items"][0]["language"] = "en"
        res = self.c.post("/v1/rag/batch", json=b, headers=H).json()["results"][0]
        self.assertEqual(res["error"]["code"], "INVALID_LANGUAGE")

    def test_10_async_mode(self):
        r = self.c.post("/v1/rag/batch", json=batch(2, "b-async"), headers={**H, "Prefer": "respond-async"})
        self.assertEqual(r.status_code, 202)
        self.assertEqual(r.json()["status"], "processing")
        for _ in range(100):
            g = self.c.get("/v1/rag/batch/b-async", headers=H)
            if g.status_code == 200:
                break
            asyncio.run(asyncio.sleep(0.1))
        self.assertEqual(g.status_code, 200)
        self.assertEqual(g.json()["status"], "completed")

    def test_10b_interrupted_batch_recovers(self):
        """A batch left "processing" by a previous server process must not hang forever."""
        from lexrag import store
        b = batch(2, "b-interrupted")
        req = api.BatchRequest(**b).model_dump(mode="json")
        h = store.canonical_hash({k: req[k] for k in ("run_id", "batch_index", "total_batches", "options", "items")})
        self.assertTrue(store.claim_batch("b-interrupted", "test-run", h)[0])
        store.con().execute("UPDATE batches SET updated_at=? WHERE batch_id=?", (api._STARTED - 60, "b-interrupted"))
        g = self.c.get("/v1/rag/batch/b-interrupted", headers=H)
        self.assertEqual(g.status_code, 200)
        self.assertEqual(g.json()["status"], "failed")
        self.assertEqual(g.json()["error"]["code"], "INTERRUPTED")
        self.assertTrue(g.json()["error"]["retryable"])
        r = self.c.post("/v1/rag/batch", json=b, headers=H)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "completed")
        self.assertEqual(self.c.get("/v1/rag/batch/b-interrupted", headers=H).json()["status"], "completed")
        # a batch being processed right now (updated after start-up) is still reported as processing
        self.assertTrue(store.claim_batch("b-live", "test-run", h)[0])
        self.assertEqual(self.c.get("/v1/rag/batch/b-live", headers=H).status_code, 202)

    def test_10c_min_rerank_score_opt_in(self):
        """Default: top_k contexts.  With min_rerank_score: only reranked-relevant ones, cited kept."""
        q = ["Yillik asosiy mehnat ta'tilining eng kam muddati qancha?"]
        full = self.c.post("/v1/rag/batch", json=batch(1, "b-rr-off", q), headers=H).json()["results"][0]
        self.assertEqual(len(full["contexts"]), 5)
        b = batch(1, "b-rr-on", q, min_rerank_score=0.5)
        res = self.c.post("/v1/rag/batch", json=b, headers=H).json()["results"][0]
        self.assertGreaterEqual(len(res["contexts"]), 1)
        self.assertLess(len(res["contexts"]), 5)
        cited = {c["chunk_id"] for c in res["citations"]}
        ctx = {c["chunk_id"] for c in res["contexts"]}
        self.assertTrue(cited <= ctx)
        for c in res["contexts"]:
            self.assertTrue(c["rerank_score"] >= 0.5 or c["chunk_id"] in cited)
        self.assertEqual([c["rank"] for c in res["contexts"]], list(range(1, len(res["contexts"]) + 1)))
        bad = batch(1, "b-rr-bad", q, min_rerank_score=2)
        self.assertEqual(self.c.post("/v1/rag/batch", json=bad, headers=H).status_code, 400)

    def test_11_meta_health(self):
        self.assertEqual(self.c.get("/v1/health").status_code, 200)
        m = self.c.get("/v1/meta").json()
        self.assertTrue(m["index_version"] and m["lex_snapshot_date"])

    def test_12_outdated_code_is_marked(self):
        from lexrag.retrieval import db
        st = db().execute("SELECT status FROM docs WHERE doc_id='lex:-142859'").fetchone()[0]
        self.assertEqual(st, "kuchini_yoqotgan", "1995 Labour Code must be marked as no longer in force")
        st = db().execute("SELECT status FROM docs WHERE doc_id='lex:-6257288'").fetchone()[0]
        self.assertEqual(st, "amalda")

    def test_12b_as_of_date(self):
        from lexrag.retrieval import retrieve
        cands, _ = retrieve(["mehnat taʼtili davomiyligi", "yillik asosiy taʼtil"], ["Mehnat kodeksi"], [], "2015-01-01", 40)
        self.assertTrue(cands)
        for c in cands:
            self.assertFalse(c.adopted and c.adopted > "2015-01-01", f"{c.doc_id} adopted after as_of_date")
            self.assertEqual(c.amended_after_as_of, bool(c.edition_date and c.edition_date > "2015-01-01"))
        self.assertNotIn("lex:-6257288", {c.doc_id for c in cands}, "2022 Labour Code did not exist in 2015")
        # the 1995 code was in force in 2015 -> not penalised as repealed at that date
        old = [c for c in cands if c.doc_id == "lex:-142859"]
        self.assertTrue(old and all(c.signals["in_force"] for c in old))

    def test_13_rate_limit(self):
        old = settings.rate_limit_per_sec
        settings.rate_limit_per_sec = 1.0
        api._buckets.clear()
        try:
            codes = [self.c.post("/v1/rag/batch", json=batch(1, f"b-rl{i}"), headers=H).status_code for i in range(4)]
            self.assertIn(429, codes)
        finally:
            settings.rate_limit_per_sec = old
            api._buckets.clear()


if __name__ == "__main__":
    unittest.main(verbosity=2)
