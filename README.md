# lexrag — RAG API over lex.uz

Implementation of **"RAG API — test spetsifikatsiyasi (lex.uz)"** (3 Nov 2026), built on the
lex.uz export in `~/Downloads/data` (`acts/`, `docs/`, `kodekslar/` — 41,658 `.doc` files).

```
question ─► rewrite (Claude: language, uz-Latn translation, legal phrasings, act hints)
         ─► retrieve (SQLite FTS5 BM25 × N queries + act-title filter + article lookup
                      + authority prior + in-force filter, RRF fusion)       ~0.1–0.5 s
         ─► rerank + answer (Claude, one structured call over 24 candidates)
         ─► validate (verbatim quotes, citation ⊂ contexts, ref renumbering, no-answer guard)
```

## Quick start

```bash
cd lexrag
pip install -r requirements.txt            # all already present on this Mac
./convert.sh                               # .doc -> text (macOS textutil), ~5 min, only reads ~/Downloads/data
python3 -m lexrag.ingest                   # builds data/lex.sqlite (~1.7 GB), ~90 s
cp .env.example .env                       # put GEMINI_API_KEY (free) or ANTHROPIC_API_KEY + your test API key here
python3 -m uvicorn lexrag.api:app --host 0.0.0.0 --port 8000
```

### LLM providers

Pick one with `LEXRAG_LLM_PROVIDER=anthropic|deepseek|gemini`. With `auto`, the first provider that has a key wins, in the order anthropic → deepseek → gemini.

| provider | key | default models | notes |
|---|---|---|---|
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-flash` (V4.1 Flash), answer effort `low`, rewrite `none` | cheap, high concurrency; JSON mode, schema validated client-side |
| Claude | `ANTHROPIC_API_KEY` | `claude-opus-5-5` | structured outputs enforced by the API |
| Gemini | `GEMINI_API_KEY` | `gemini-3.5-flash` / `3.5-flash-lite` + fallback chain | free tier works, but is rate-limited |

`python3 -m lexrag.deepseek` and `python3 -m lexrag.gemini` list the models your key can use.

**DeepSeek details:**
* `response_format: json_object`, with the schema and an example in the system prompt.
* Every reply is validated with Pydantic. Blank or malformed JSON (a documented JSON-mode quirk) is retried.
* `reasoning_effort` controls thinking: `none` is fastest, `low` is the default for answers.
* 429 / 503 are retried with backoff. 402 (no balance) is reported as a non-retryable error.
* DeepSeek has no `seed`, so reproducibility comes from the generation cache.

Gemini is called over REST with `httpx`, so no extra package is needed. Its behaviour:
* Requests use JSON-schema structured output, `seed` (the spec's reproducibility option is honoured) and `thinkingLevel`.
* Requests are paced to `LEXRAG_GEMINI_RPM` (default 10/min). With Gemini, at most 3 questions run at a time.
* A 429 is retried with the server's `retryDelay`. If the quota stays exhausted, the item fails with `retryable: true` and the eval client resends the batch.
* If the endpoint rejects `responseJsonSchema` or `thinkingConfig`, the request is retried without it.

`python3 -m lexrag.gemini` lists the models your key can use; pick one with `LEXRAG_GEMINI_MODEL`. Each question makes 2 calls (rewrite + answer), so a 100-question run is about 200 requests. On the free tier, check your daily quota in AI Studio. If you hit it, lower `LEXRAG_GEMINI_RPM` or split the run. `system.llm_provider` / `llm_model` in every response record which model answered, so runs from different providers can be compared side by side.

Without any LLM key the service still runs in an **extractive fallback** mode (best matching
sentence of the best in-force article). `system.llm_model` then says `extractive-fallback`, so runs
are never confused.

Run the tests (the spec's section-7 checklist; fake LLM, real index):

```bash
python3 -m unittest discover -s tests -v
```

**Serve it to testers** (checklist: "separate test API key and base URL"):

```bash
./run_server.sh        # prints the base URL on your network, e.g. http://10.30.4.69:8000
```

The test key is `LEXRAG_API_KEYS` in `.env`. It was generated randomly, so send it to testers privately. To reach the API from outside your network, deploy it on a server.

**Build the spec's 100-question test set** (section 6, exact category mix 35/15/15/10/10/10/5):

```bash
python3 eval/build_testset.py --out eval/testset_100.csv
```

How the builder works:
* Each answerable question is generated from a real in-force article, so its expected doc and article are correct by construction.
* A second, stronger model (`deepseek-v4-pro`) checks every item against the article: is it answerable, is the answer correct and complete, is it unambiguous?
* The numbers in each expected answer must occur in the source text.
* If an almost identical article exists in another act (e.g. the civil, economic and administrative procedure codes), the question must name its act.
* Amending laws and repealed acts are never used as sources.
* Provenance (source chunks, verifier notes) goes to `eval/testset_100.provenance.json`.

The spec intends a human tester to own the test set, so skim it before relying on the numbers.

Run an evaluation exactly as the spec's test scenario describes:

```bash
python3 eval/run_eval.py run --testset eval/testset_sample.csv --run-id eval-2026-11-06 --score
python3 eval/run_eval.py score --testset eval/testset_sample.csv --run-id eval-2026-11-06 --judge   # + LLM judge
python3 eval/run_eval.py compare eval-2026-11-03 eval-2026-11-06
JUDGE_PROVIDER=gemini python3 eval/run_eval.py score --testset eval/testset_100.csv --run-id <run> --judge   # grade with another vendor
python3 eval/run_eval.py run ... --seed 9001      # a new seed bypasses the answer cache (fresh answers, real latency)
```

The judge measures:
* answer correctness (0/1/2);
* faithfulness (supported claims ÷ all claims);
* citation support;
* context precision, as the spec defines it: the share of the top-5 contexts that are relevant to the question. The older same-document proxy is still reported as `context_precision_proxy_same_doc`.

Answerable questions the system refused score correctness 0.

Each batch response is saved to `eval/runs/<run_id>/bNN.json`, metrics to `metrics.json`.

## Endpoints

| Endpoint | Notes |
|---|---|
| `POST /v1/rag/batch` | 1–10 items; 11 → `400`; sync ≤ 110 s, otherwise `202` + `GET`. `Prefer: respond-async` forces async. |
| `GET /v1/rag/batch/{batch_id}` | `202 {"status":"processing"}` / `200` full result / `404`. Results kept in `data/batches.sqlite` (no expiry → ≥ 30 days). |
| `POST /v1/rag/query` | single question (debug) |
| `GET /v1/meta`, `GET /v1/health` | versions, index size, snapshot date; health checks index + store + LLM config |

Auth `Authorization: Bearer <key>` (401 missing / 403 wrong). Rate limit: token bucket per key
(default 2 batch/s) → `429` + `Retry-After`. All JSON is `application/json; charset=utf-8` with
`ensure_ascii=False`, so `oʻ`, `gʻ`, `ʼ` are never escaped or mangled. Every response carries the
`system` object (pipeline, prompt, models, index version, lex snapshot date).

## How the open questions of the spec are answered

* **Snapshot date:** `lex_snapshot_date` = the newest edition date found in the export (2026-11-03). Refresh = re-run `convert.sh` + `ingest`; `index_version` changes with it.
* **Chunking:** by structure, not fixed length. `N-modda` for codes and laws, `N-band` for decrees and resolutions. Each article is split into parts (*qism*) and packed into chunks of ≤ 1,400 characters. A chunk never crosses an article, so `citations[].article` is exact. Chunk ids are stable: `lex:<id>#art-<N>#p<part>`.
* **Repealed acts** stay in the index, marked `doc_status = kuchini_yoqotgan`. They are heavily down-weighted in ranking, and the model is told never to answer from them alone. Four sources mark a document as repealed (≈ 8,000 acts in total):
  * lex.uz's own banner at the top of the act ("Mazkur Kodeks … oʻz kuchini yoʻqotgan"), which covers 7,199 acts;
  * renamed successors (Xoʻjalik protsessual → Iqtisodiy protsessual kodeksi);
  * repeal statements in other acts ("… oʻz kuchini yoʻqotgan deb topilsin" with a link to the act, or an annex listing repealed acts);
  * a newer code, constitution or law with the same name. This is how the 1995 Labor Code, the 1997 Tax Code and the 1992 Constitution are marked as superseded.

## Problems found and fixed (in the data and in the spec)

**Data**
1. **1,477 acts are stored twice**, as `N.doc` and `-N.doc`, with byte-identical text. Duplicates would fill the top-5 with copies and double-count Hit@5. They are de-duplicated, keeping the id that lex.uz itself links to.
2. **`kodekslar/` files are named by date, with no lex.uz id.** Each one is matched to its copy in `acts/` by text similarity. If there is no copy, the id comes from a table checked against how many other documents link to it (ingest prints the mapping).
3. **`kodekslar/21.12.1995.doc` is the old Labor Code.** It lost force when the 2022 code (`lex:-6257288`) was adopted. The spec's own example question q001 targets the new code ("kalendar kun"). Without status detection the system would cite the repealed code.
4. **Superscript articles** (`546⁵-modda`) are flattened to `5465-modda` by the `.doc` conversion. The parser detects and restores them.
5. Word field codes (`HYPERLINK …`, `INCLUDEPICTURE …`), "Oldingi tahrirga qarang" lines and editorial notes are removed from the indexed text. The notes are used for `edition_date`.

**Spec**
1. **`temperature = 0` / `seed` can't be sent to current Claude models** (sampling parameters are rejected). Reproducibility is guaranteed differently:
   * retrieval is fully deterministic, with ties broken by `chunk_id`;
   * generation is cached by (model, prompt version, question, the exact candidate chunk ids, seed).

   So the same question on the same index returns the same sources and the same answer. A new seed or a new index produces a fresh answer.
2. **Idempotency vs. retry conflict:** the spec says to resend the same `batch_id` after a `retryable: true` error, but also that the same `batch_id` returns the stored result. This server does both: good answers are kept, and only the retryable items are recomputed.
3. FastAPI's default `422` is mapped to the spec's `400`. `summary` also counts `timeout`.
4. **"Context precision" is not defined precisely.** The harness uses the share of top-5 contexts from an expected document. With `--judge`, the LLM judge also scores correctness (0/1/2), faithfulness (supported claims ÷ all claims) and citation support.
5. **Error attribution is automated.** Each miss is labelled retrieval (expected article not in top-5), generatsiya (found but answered wrongly / refused), or manba (correct answer, wrong citation).
6. **`as_of_date` limitation:** the export only has current editions. Acts adopted after the date are excluded, and repealed acts count as in force if they were repealed after it. Historical wording of amended articles can't be reconstructed from this data.

## Guarantees enforced in code (not just asked of the model)

* Every `citations[].quote` is a **verbatim substring** of the cited context. Quotes the model paraphrased are snapped to the closest sentence of that context, or dropped.
* Every `citations[].chunk_id` is in `contexts`. Citations to contexts the model never saw are dropped.
* Answer markers `[n]` are renumbered to match `citations[].ref`.
* An answer with no verifiable citation becomes `no_answer: true` instead of an ungrounded answer.
* An error in one question never fails the batch (`status: error|timeout` + `error{code,message,retryable}`).

## Current results

`eval/testset_sample.csv` is a 10-question starter set with verified expected doc and article. It covers factual, numbers, multi-article, amended-norm, Russian / Uzbek Cyrillic, colloquial and unanswerable questions.

Run `deepseek-v4` used DeepSeek `deepseek-flash` with an empty cache, so every rewrite and answer was generated fresh:

| metric | value | target |
|---|---|---|
| Hit@5 doc / article | 1.0 / 1.0 | ≥ 0.90 / 0.80 |
| MRR | 0.94 | ≥ 0.70 |
| context precision | 0.73 | ≥ 0.60 |
| citation accuracy | 1.0 | ≥ 0.90 |
| fabricated / outdated citations | 0 / 0 | 0 / 0 |
| no-answer accuracy | 1.0 | ≥ 0.9 |
| latency p50 / p95 | 6.4 s / 9.9 s | ≤ 5 / 12 s |
| error rate | 0 | ≤ 2% |

Per question: rewrite ~2.3 s, retrieval ~1 s (inside a concurrent batch of 10), generation ~3 s.

Next: expand the test set to the spec's 100 questions (35 / 15 / 15 / 10 / 10 / 10 / 5). On ten questions a single miss moves a metric by 10–12 points.

### Latency
* **Speculative answering.** While the LLM rewrites the question, the system retrieves with the raw question and already generates an answer. If the full pipeline's top-5 evidence is all in that pool, the speculative answer is used. The accept/reject decision is a pure function of the question, the cached rewrite and the index, so repeated questions always get the same pool and the same cached answer.
* **Out-of-scope questions** (as judged by the rewriter) are refused without a generation call.
* **Sparse relevance output:** the model lists only relevant contexts, which halves output tokens.
* **Parallel FTS queries,** frequent-term pruning, and a search cache warmed during the rewrite.

### Retrieval improvements made after live evaluation
* **Act discovery.** Acts whose distinctive title words the question covers ("narkologik", "oilaviy tadbirkorlik") are searched inside, in addition to the rewriter's guesses. Fuzzy title matching resolves hints whose word forms differ from the official title.
* **Amending laws** ("…ga oʻzgartish va qoʻshimchalar kiritish toʻgʻrisida", 8,071 acts) get a low authority prior, because the amended code already contains the current text.
* The model reranks 32 candidates. In offline checks, the expected article is among them for 96.7% of the questions.
* **In-act search.** When the question names an act ("Jinoyat kodeksi"), every chunk of that act is scored directly. Its best 6 chunks always reach the reranker. The hint resolves to the exact act: "Fuqarolik kodeksi" → both parts of the Civil Code, not the Civil Procedure Code.
* **Uzbek derivational stems.** `yillik` / `har yilgi` → `yil`, so the 2022 Labor Code's wording matches older phrasing.
* **Very frequent terms** (in > 10% of chunks) are left out of the FTS expression. FTS queries run in parallel threads, and searches for the raw question start while the LLM is still rewriting it.
* **Prompts handle premises from repealed acts.** The rewriter targets current law. The answerer prefers codes over regulations that quote outdated editions, and corrects the premise.

## Layout

```
lexrag/textnorm.py   apostrophes, Cyrillic→Latin, language detection, folding, Uzbek stemmer
lexrag/parse.py      .txt → document metadata + structure-aware chunks, repeal statements
lexrag/ingest.py     parallel parse, id resolution, de-dup, status graph, SQLite/FTS5 build
lexrag/retrieval.py  hybrid lexical retrieval, priors, in-force filter
lexrag/llm.py        rewrite + rerank/answer prompts; provider switch; Claude path (structured outputs, refusal fallback)
lexrag/deepseek.py   DeepSeek path (OpenAI-style REST, JSON mode + validation, retries)
lexrag/gemini.py     Gemini path (REST, JSON schema, pacing, 429 retry, feature fallback)
lexrag/pipeline.py   orchestration, quote validation, citations/contexts assembly
lexrag/store.py      batch store (idempotency, async) + generation cache
lexrag/api.py        FastAPI service
eval/run_eval.py     tester client (sequential batches, retry policy, polling) + metrics + LLM judge
eval/build_testset.py builds + verifies the 100-question test set (spec category mix)
run_server.sh        serves the API on the local network for testers
tests/test_api.py    section-7 checklist as automated tests
tests/test_gemini.py Gemini provider against a mocked HTTP API
tests/test_deepseek.py DeepSeek provider against a mocked HTTP API
```

Configuration: `lexrag/config.py`, overridden by `LEXRAG_*` environment variables or `.env`.
The default model is `claude-opus-5-5`, with `medium` effort for answers and `low` for rewriting.
Server-side refusal fallback (`fallbacks: "default"`) is on; it is turned off automatically if the endpoint rejects it.
