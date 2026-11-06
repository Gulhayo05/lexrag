"""DeepSeek provider tests against a mocked HTTP API (no key, no network)."""
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

from lexrag import deepseek, llm  # noqa: E402
from lexrag.config import settings  # noqa: E402

REQS: list[dict] = []
SCRIPT: list = []


def handler(request: httpx.Request) -> httpx.Response:
    REQS.append({"url": str(request.url), "headers": dict(request.headers), "body": json.loads(request.content or b"{}")})
    return SCRIPT.pop(0)


def reply(obj_or_text, finish="stop"):
    content = obj_or_text if isinstance(obj_or_text, str) else json.dumps(obj_or_text, ensure_ascii=False)
    return httpx.Response(200, json={"model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": finish,
                          "message": {"role": "assistant", "content": content, "reasoning_content": "..."}}],
                          "usage": {"prompt_tokens": 120, "completion_tokens": 30, "prompt_cache_hit_tokens": 100}})


REWRITE = {"language": "uz-Latn", "question_uz": "Nikoh yoshi necha yosh?", "search_queries": ["nikoh yoshi"],
           "doc_hints": ["Oila kodeksi"], "article_hints": [], "in_scope": True}


class DeepSeekProvider(unittest.TestCase):
    def setUp(self):
        deepseek._transport = httpx.MockTransport(handler)
        deepseek._client = None
        REQS.clear(); SCRIPT.clear()
        self._saved = {k: os.environ.pop(k, None) for k in ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")}
        llm._DISABLED.clear()
        os.environ["DEEPSEEK_API_KEY"] = "test-ds-key"
        settings.llm_provider = "auto"
        self._sleep = deepseek.asyncio.sleep

        async def no_sleep(_s):
            pass
        deepseek.asyncio.sleep = no_sleep

    def tearDown(self):
        llm._DISABLED.clear()
        deepseek.asyncio.sleep = self._sleep
        os.environ.pop("DEEPSEEK_API_KEY", None)
        for k, v in self._saved.items():
            if v:
                os.environ[k] = v

    def test_selection_and_request_shape(self):
        self.assertEqual(llm.provider(), "deepseek")
        SCRIPT.append(reply(REWRITE))
        out, usage = asyncio.run(llm.rewrite("Nikoh yoshi?"))
        self.assertEqual(out.doc_hints, ["Oila kodeksi"])
        self.assertEqual(usage["cache_read_input_tokens"], 100)
        r = REQS[0]
        self.assertTrue(r["url"].endswith("/chat/completions"))
        self.assertEqual(r["headers"]["authorization"], "Bearer test-ds-key")
        b = r["body"]
        self.assertEqual(b["response_format"], {"type": "json_object"})
        self.assertEqual(b["reasoning_effort"], settings.deepseek_rewrite_effort)
        self.assertIn("json", b["messages"][0]["content"].lower())  # required by JSON mode
        self.assertEqual(b["model"], settings.deepseek_rewrite_model)

    def test_blank_content_retry_and_fenced_json(self):
        SCRIPT.extend([reply("   "), reply("```json\n" + json.dumps(REWRITE) + "\n```")])
        out, _ = asyncio.run(llm.rewrite("q"))
        self.assertEqual(out.language, "uz-Latn")
        self.assertEqual(len(REQS), 2)

    def test_answer_schema(self):
        ans = {"relevance": [{"ctx": 1, "score": 1.0}], "no_answer": False, "answer": "18 yosh [1].",
               "citations": [{"ref": 1, "ctx": 1, "quote": "Nikoh yoshi ... oʻn sakkiz yosh"}], "confidence": 0.9}
        SCRIPT.append(reply(ans))
        out, _ = asyncio.run(llm.answer("q", "uz-Latn", "[C1] ..."))
        self.assertEqual(out.citations[0].ref, 1)
        self.assertEqual(REQS[0]["body"]["reasoning_effort"], settings.deepseek_answer_effort)

    def test_errors(self):
        SCRIPT.append(httpx.Response(402, json={"error": {"message": "Insufficient Balance"}}))
        with self.assertRaises(llm.LLMError) as cm:
            asyncio.run(llm.rewrite("q"))
        self.assertIn("balance", cm.exception.message)
        self.assertFalse(cm.exception.retryable)
        llm._DISABLED.clear()  # the 402 disabled DeepSeek for this process; re-enable for the next checks
        SCRIPT.extend([httpx.Response(429, json={}), reply(REWRITE)])
        out, _ = asyncio.run(llm.rewrite("q"))
        self.assertEqual(out.language, "uz-Latn")
        SCRIPT.extend([httpx.Response(503, json={})] * 4)
        with self.assertRaises(llm.LLMError) as cm:
            asyncio.run(llm.rewrite("q"))
        self.assertTrue(cm.exception.retryable)
        SCRIPT.append(reply(REWRITE, finish="length"))
        with self.assertRaises(llm.LLMError) as cm:
            asyncio.run(llm.rewrite("q"))
        self.assertTrue(cm.exception.retryable)

    def test_provider_fallback_on_no_balance(self):
        os.environ["GEMINI_API_KEY"] = "test-gemini-key"
        import lexrag.gemini as g
        calls = []

        async def fake_generate(model, thinking, system, user, schema, max_tokens, seed=None):
            calls.append(model)
            return schema(**REWRITE), {"input_tokens": 1, "output_tokens": 1}
        orig = g.generate
        g.generate = fake_generate
        try:
            SCRIPT.append(httpx.Response(402, json={"error": {"message": "Insufficient Balance"}}))
            out, u = asyncio.run(llm.rewrite("q"))
            self.assertEqual(out.language, "uz-Latn")
            self.assertIn("deepseek", llm._DISABLED)
            self.assertEqual(llm.provider(), "gemini")
            self.assertTrue(calls)
        finally:
            g.generate = orig
            os.environ.pop("GEMINI_API_KEY", None)

    def test_effort_param_rejected_fallback(self):
        SCRIPT.extend([httpx.Response(400, json={"error": {"message": "unknown parameter reasoning_effort"}}), reply(REWRITE)])
        asyncio.run(llm.rewrite("q"))
        self.assertNotIn("reasoning_effort", REQS[-1]["body"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
