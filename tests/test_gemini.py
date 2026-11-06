"""Gemini provider tests against a mocked HTTP API (no key, no network)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LEXRAG_RESULTS_DB", os.path.join(tempfile.mkdtemp(), "b.sqlite"))

import httpx  # noqa: E402

from lexrag import gemini, llm  # noqa: E402
from lexrag.config import settings  # noqa: E402

settings.gemini_rpm = 100000  # no pacing in tests
REQS: list[dict] = []
SCRIPT: list = []


def handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content or b"{}")
    REQS.append({"url": str(request.url), "headers": dict(request.headers), "body": body})
    step = SCRIPT.pop(0)
    return step(body) if callable(step) else step


def ok_json(obj, thought=True, finish="STOP"):
    parts = ([{"text": "thinking...", "thought": True}] if thought else []) + [{"text": "```json\n" + json.dumps(obj, ensure_ascii=False) + "\n```"}]
    return httpx.Response(200, json={"candidates": [{"content": {"parts": parts}, "finishReason": finish}],
                                     "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 5}})


REWRITE = {"language": "ru", "question_uz": "Umumiy daʼvo muddati qancha?", "search_queries": ["umumiy daʼvo muddati"],
           "doc_hints": ["Fuqarolik kodeksi"], "article_hints": [], "in_scope": True}


class GeminiProvider(unittest.TestCase):
    def setUp(self):
        gemini._transport = httpx.MockTransport(handler)
        gemini._client = None
        gemini._features.update(json_schema=True, thinking=True)
        REQS.clear(); SCRIPT.clear()
        os.environ["GEMINI_API_KEY"] = "test-gemini-key"
        self._a = os.environ.pop("ANTHROPIC_API_KEY", None)
        self._d = os.environ.pop("DEEPSEEK_API_KEY", None)
        llm._DISABLED.clear()
        settings.llm_provider = "auto"
        settings.gemini_fallback_models = ""

    def tearDown(self):
        llm._DISABLED.clear()
        if self._d:
            os.environ["DEEPSEEK_API_KEY"] = self._d
        if self._a:
            os.environ["ANTHROPIC_API_KEY"] = self._a
        os.environ.pop("GEMINI_API_KEY", None)

    def test_provider_selection(self):
        self.assertEqual(llm.provider(), "gemini")
        self.assertEqual(llm.model_id(), settings.gemini_model)
        os.environ["ANTHROPIC_API_KEY"] = "x"
        self.assertEqual(llm.provider(), "anthropic")  # auto prefers Anthropic when both exist
        settings.llm_provider = "gemini"
        self.assertEqual(llm.provider(), "gemini")
        os.environ.pop("ANTHROPIC_API_KEY")

    def test_request_shape_and_parse(self):
        SCRIPT.append(ok_json(REWRITE))
        out, usage = asyncio.run(llm.rewrite("Какой срок исковой давности?"))
        self.assertEqual(out.doc_hints, ["Fuqarolik kodeksi"])
        self.assertEqual(usage["output_tokens"], 25)
        r = REQS[0]
        self.assertIn(f"/models/{settings.gemini_rewrite_model}:generateContent", r["url"])
        self.assertEqual(r["headers"]["x-goog-api-key"], "test-gemini-key")
        gc = r["body"]["generationConfig"]
        self.assertEqual(gc["responseMimeType"], "application/json")
        self.assertNotIn("$ref", json.dumps(gc["responseJsonSchema"]))
        self.assertEqual(gc["thinkingConfig"], {"thinkingLevel": "LOW"})
        self.assertEqual(gc["seed"], 42)
        self.assertIn("systemInstruction", r["body"])

    def test_answer_schema_nested_inlined(self):
        ans = {"relevance": [{"ctx": 1, "score": 0.9}], "no_answer": False, "answer": "Uch yil [1].",
               "citations": [{"ref": 1, "ctx": 1, "quote": "Umumiy daʼvo muddati — uch yil."}], "confidence": 0.9}
        SCRIPT.append(ok_json(ans, thought=False))
        out, _ = asyncio.run(llm.answer("q", "uz-Latn", "[C1] ...", seed=7))
        self.assertEqual(out.citations[0].quote, "Umumiy daʼvo muddati — uch yil.")
        self.assertEqual(REQS[0]["body"]["generationConfig"]["seed"], 7)
        self.assertEqual(REQS[0]["body"]["generationConfig"]["thinkingConfig"], {"thinkingLevel": "MEDIUM"})

    def test_429_retry_with_retry_delay(self):
        import lexrag.gemini as g
        orig = g.asyncio.sleep
        slept = []

        async def fake_sleep(s):
            slept.append(s)
        g.asyncio.sleep = fake_sleep
        try:
            SCRIPT.extend([httpx.Response(429, json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                           "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}]}}),
                           ok_json(REWRITE)])
            out, _ = asyncio.run(llm.rewrite("q"))
            self.assertIn(7.0, slept)
            self.assertEqual(len(REQS), 2)
            # quota never recovers -> retryable LLM_ERROR (tester resends the batch later)
            SCRIPT.extend([httpx.Response(429, json={})] * 4)
            with self.assertRaises(llm.LLMError) as cm:
                asyncio.run(llm.rewrite("q2"))
            self.assertTrue(cm.exception.retryable)
        finally:
            g.asyncio.sleep = orig

    def test_feature_fallbacks(self):
        SCRIPT.extend([httpx.Response(400, json={"error": {"message": "Unknown name \"responseJsonSchema\""}}),
                       httpx.Response(400, json={"error": {"message": "thinking_level is not supported for this model"}}),
                       ok_json(REWRITE)])
        out, _ = asyncio.run(llm.rewrite("q"))
        self.assertEqual(out.language, "ru")
        last = REQS[-1]["body"]
        self.assertNotIn("responseJsonSchema", last["generationConfig"])
        self.assertNotIn("thinkingConfig", last["generationConfig"])
        self.assertIn("JSON Schema", last["systemInstruction"]["parts"][0]["text"])

    def test_fallback_chain(self):
        settings.gemini_fallback_models = "backup-model"
        SCRIPT.extend([httpx.Response(503, json={}), httpx.Response(503, json={}), ok_json(REWRITE)])
        import lexrag.gemini as g
        orig = g.asyncio.sleep

        async def no_sleep(s):
            pass
        g.asyncio.sleep = no_sleep
        try:
            out, _ = asyncio.run(llm.rewrite("q"))
        finally:
            g.asyncio.sleep = orig
        self.assertEqual(out.language, "ru")
        self.assertIn("/models/backup-model:generateContent", REQS[-1]["url"])
        self.assertEqual(llm.LAST_MODEL["rewrite"], "backup-model")

    def test_errors(self):
        SCRIPT.append(httpx.Response(404, json={}))
        with self.assertRaises(llm.LLMError) as cm:
            asyncio.run(llm.rewrite("q"))
        self.assertIn("not found", cm.exception.message)
        SCRIPT.append(ok_json(REWRITE, finish="MAX_TOKENS"))
        with self.assertRaises(llm.LLMError) as cm:
            asyncio.run(llm.rewrite("q"))
        self.assertTrue(cm.exception.retryable)
        SCRIPT.append(httpx.Response(403, json={}))
        with self.assertRaises(llm.LLMError) as cm:
            asyncio.run(llm.rewrite("q"))
        self.assertFalse(cm.exception.retryable)


if __name__ == "__main__":
    unittest.main(verbosity=2)
