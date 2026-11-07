"""Question -> answer pipeline:  rewrite -> retrieve -> (LLM rerank + generate) -> validate."""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from . import llm, store
from .config import settings
from .retrieval import Candidate, index_meta, retrieve
from .textnorm import cyr_to_lat, detect_language, fold, query_terms, unify_apostrophes

NO_ANSWER_TEXT = {
    "uz-Latn": "Taqdim etilgan qonun hujjatlarida bu savolga aniq javob topilmadi.",
    "uz-Cyrl": "Тақдим этилган қонун ҳужжатларида бу саволга аниқ жавоб топилмади.",
    "ru": "В предоставленных нормативных актах ответ на этот вопрос не найден.",
}

# Russian / Cyrillic names of the main acts -> Uzbek titles used on lex.uz
RU_ACTS = {
    r"трудов\w* кодекс": "Mehnat kodeksi", r"гражданск\w* процессуальн\w* кодекс": "Fuqarolik protsessual kodeksi",
    r"гражданск\w* кодекс": "Fuqarolik kodeksi", r"уголовно[- ]процессуальн\w* кодекс": "Jinoyat-protsessual kodeksi",
    r"уголовно[- ]исполнительн\w* кодекс": "Jinoyat-ijroiya kodeksi", r"уголовн\w* кодекс": "Jinoyat kodeksi",
    r"налогов\w* кодекс": "Soliq kodeksi", r"семейн\w* кодекс": "Oila kodeksi", r"земельн\w* кодекс": "Yer kodeksi",
    r"жилищн\w* кодекс": "Uy-joy kodeksi", r"таможенн\w* кодекс": "Bojxona kodeksi", r"бюджетн\w* кодекс": "Budjet kodeksi",
    r"воздушн\w* кодекс": "Havo kodeksi", r"градостроительн\w* кодекс": "Shaharsozlik kodeksi",
    r"избирательн\w* кодекс": "Saylov kodeksi", r"экономическ\w* процессуальн\w* кодекс": "Iqtisodiy protsessual kodeksi",
    r"административн\w* ответственност\w*": "Maʼmuriy javobgarlik toʻgʻrisidagi kodeks", r"конституц\w*": "Konstitutsiya",
}
UZ_ACT_RE = re.compile(r"((?:\b[A-Za-zʻʼ'][A-Za-zʻʼ'\-]+\s){1,3}(?:kodeksi\w*|kodeks\w*)|konstitutsiya\w*)", re.IGNORECASE)


def _heuristic_rewrite(question: str, lang: str) -> dict:
    q_uz = cyr_to_lat(question) if lang == "uz-Cyrl" else question
    hints = []
    for rx, name in RU_ACTS.items():
        if re.search(rx, question, re.IGNORECASE):
            hints.append(name)
    for m in UZ_ACT_RE.findall(unify_apostrophes(q_uz)):
        words = [w for w in m.split() if fold(w) not in ("va", "bilan", "uchun", "qaysi", "qanday", "yilgi", "amaldagi", "yangi", "eski")]
        words[-1] = re.sub(r"(kodeks)\w*", r"\1i", words[-1], flags=re.IGNORECASE)
        if len(words) >= 2 or "konstitutsiya" in fold(words[-1]):
            hints.append(" ".join(words[-2:]))
    return {"language": lang, "question_uz": q_uz, "search_queries": [], "doc_hints": hints[:3], "article_hints": [], "in_scope": True}


# --------------------------------------------------------------------------- #
# quote validation
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unify_apostrophes(s)).strip()


_SENT_RE = re.compile(r"(?<=[.;:!?])\s+|\n+")


def verbatim_quote(quote: str, text: str) -> tuple[str, float]:
    """Return an excerpt that occurs verbatim in ``text`` (+ similarity to the model's quote)."""
    flat = _norm(text)
    q = _norm(quote).strip(" .…\"“”«»")
    if q and q in flat:
        return q, 1.0
    if q and q.lower() in flat.lower():
        i = flat.lower().index(q.lower())
        return flat[i:i + len(q)], 1.0
    qt = set(query_terms(q))
    best, best_s = "", 0.0
    for sent in _SENT_RE.split(text):
        sent = _norm(sent)
        if len(sent) < 15:
            continue
        st = set(query_terms(sent))
        s = len(qt & st) / max(1, len(qt))
        if s > best_s:
            best, best_s = sent, s
    return best[:400], best_s


# --------------------------------------------------------------------------- #
def _ctx_block(pool: list[Candidate]) -> str:
    out = []
    for i, c in enumerate(pool, 1):
        art = c.article or "—"
        out.append(
            f"[C{i}] {c.doc_title[:200]} | turi: {c.doc_type} | holati: {c.status} | tahrir: {c.edition_date or '?'} | {art}"
            + (f" | {c.part}" if c.part else "")
            + (" | DIQQAT: bu modda soʻralgan sanadan (as_of_date) keyin oʻzgartirilgan; matn hozirgi tahrirda" if c.amended_after_as_of else "")
            + (f"\n({c.path[:160]})" if c.path else "") + f"\n{c.text}\n"
        )
    return "\n".join(out)


def _extractive(question: str, lang: str, pool: list[Candidate]) -> dict:
    """LLM-free fallback: answer with the best matching sentence of the best context."""
    if not pool:
        return {"no_answer": True, "answer": NO_ANSWER_TEXT[lang], "citations": [], "confidence": 0.0, "relevance": {}, "_extractive": True}
    top = pool[0]
    cov = top.signals.get("coverage", 0)
    if cov < 0.67 or not top.signals.get("in_force", True):
        return {"no_answer": True, "answer": NO_ANSWER_TEXT[lang], "citations": [], "confidence": round(0.3 * cov, 3), "relevance": {}, "_extractive": True}
    quote, _ = verbatim_quote(question, top.text)
    quote = quote or _norm(top.text)[:300]
    ref = f"{top.doc_title}, {top.article}" if top.article else top.doc_title
    return {
        "no_answer": False,
        "answer": f"{quote} [1] ({ref})",
        "citations": [{"ref": 1, "ctx": 1, "quote": quote}],
        "confidence": round(0.6 * cov, 3),
        "relevance": {},
        "_extractive": True,
    }


async def answer_item(item: dict, options: dict) -> dict:
    t0 = time.perf_counter()
    qid, question = item["id"], unify_apostrophes(item["question"].strip())
    history = [h for h in (item.get("history") or []) if h.get("question")][-5:]  # chat memory: last 5 turns
    lang = item.get("language") or detect_language(question)
    ans_lang = options.get("answer_language") or "auto"
    ans_lang = lang if ans_lang == "auto" else ans_lang
    top_k = options.get("top_k", 5)
    use_llm = llm.available()
    usage = {"input_tokens": 0, "output_tokens": 0}
    models_used: dict[str, str] = {}
    timing = {"rewrite_ms": 0, "retrieval_ms": 0, "rerank_ms": 0, "generation_ms": 0, "total_ms": 0}

    n_pool = max(settings.rerank_pool, top_k)
    as_of = options.get("as_of_date")

    async def generate(pool_: list[Candidate]) -> tuple[dict, dict]:
        """Rerank + answer over a pool; cached by (model, prompt, question, language, pool ids, seed)."""
        key = store.canonical_hash(["ans", llm.model_id("answer"), settings.prompt_version, question, ans_lang,
                                    [c.chunk_id for c in pool_], options.get("seed")])
        g = store.cache_get(key)
        if g is not None:
            g["relevance"] = {int(k): v for k, v in g["relevance"].items()}
            return g, {}
        q_for_answer = question
        if history and rw.get("question_uz") and fold(rw["question_uz"]) != fold(question):
            q_for_answer = f"{question}\n(Suhbat kontekstida toʻliq savol: {rw['question_uz']})"
        out, u = await llm.answer(q_for_answer, ans_lang, _ctx_block(pool_), options.get("seed"))
        g = out.model_dump()
        g["relevance"] = {r["ctx"]: max(0.0, min(1.0, r["score"])) for r in g["relevance"]}
        g["answer_model"] = u.get("model") or llm.model_id("answer")
        store.cache_put(key, g)
        return g, u

    def add_usage(u: dict) -> None:
        nonlocal usage
        usage = {k: usage[k] + (u or {}).get(k, 0) for k in usage}

    # 1. query rewriting (+ speculative answer from the raw question) ---------
    ts = time.perf_counter()
    heur = _heuristic_rewrite(question, lang)
    rw = heur
    spec_task: asyncio.Task | None = None
    spec_cands_f: asyncio.Future | None = None

    def spec_retrieve():
        return retrieve([question, heur["question_uz"]], heur["doc_hints"], [], as_of, settings.candidates)[0]

    if use_llm:
        key = store.canonical_hash(["rw", llm.model_id("rewrite"), settings.prompt_version, question, history or None])
        cached = store.cache_get(key)
        if cached is None:
            if not history:  # follow-ups ("va bu qaysi moddada?") are meaningless without the rewrite
                # speculation: while the LLM rewrites, retrieve with the raw question and already generate
                spec_cands_f = asyncio.get_running_loop().create_future()

                async def speculate():
                    try:
                        c = await asyncio.to_thread(spec_retrieve)
                    except Exception as e:  # noqa: BLE001
                        spec_cands_f.set_exception(e)
                        raise
                    spec_cands_f.set_result(c)
                    return await generate(c[:n_pool]) if c else None
                spec_task = asyncio.create_task(speculate())
                spec_task.add_done_callback(lambda t: t.cancelled() or t.exception())
            try:
                out, u = await llm.rewrite(question, history)
            except BaseException:
                if spec_task:
                    spec_task.cancel()
                raise
            cached = out.model_dump()
            store.cache_put(key, cached)
            add_usage(u)
            models_used["rewrite"] = u.get("model") or llm.model_id("rewrite")
        rw = {**cached, "doc_hints": list(dict.fromkeys(cached["doc_hints"] + heur["doc_hints"]))}
        if not (rw.get("question_uz") or "").strip():  # already formal Uzbek Latin: the rewriter does not copy it
            rw["question_uz"] = heur["question_uz"]
    timing["rewrite_ms"] = int((time.perf_counter() - ts) * 1000)

    # 2. retrieval -------------------------------------------------------------
    ts = time.perf_counter()
    queries = list(dict.fromkeys([question, rw["question_uz"], *rw.get("search_queries", [])]))
    try:
        cands, rinfo = await asyncio.to_thread(
            retrieve, queries, rw.get("doc_hints"), rw.get("article_hints"), as_of, settings.candidates
        )
    except Exception as e:  # noqa: BLE001
        if spec_task:
            spec_task.cancel()
        if not index_meta():  # the index database itself cannot be opened / read
            raise llm.LLMError("INDEX_UNAVAILABLE", f"search index unavailable: {e}"[:300], True) from e
        raise llm.LLMError("RETRIEVAL_FAILED", f"retrieval failed: {e}"[:300], True) from e
    timing["retrieval_ms"] = int((time.perf_counter() - ts) * 1000)
    pool = cands[:n_pool]

    # 3. rerank + generation ------------------------------------------------
    ts = time.perf_counter()
    speculative = None
    if use_llm and rw.get("in_scope") is False:
        if spec_task:
            spec_task.cancel()
        # the rewriter judged the question to be outside law: refuse without inventing an answer
        gen = {"no_answer": True, "answer": NO_ANSWER_TEXT.get(ans_lang, NO_ANSWER_TEXT["uz-Latn"]), "citations": [],
               "confidence": 0.9, "relevance": {}, "out_of_scope": True}
    elif use_llm and pool:
        # Accept the raw-question pool when it already contains the full pipeline's top-5 evidence.
        # Pure function of (question, rewrite, index) -> identical decision on cached repeats.
        try:
            if history:
                spec_cands = []
            else:
                spec_cands = await spec_cands_f if spec_cands_f else await asyncio.to_thread(spec_retrieve)
        except Exception:  # noqa: BLE001
            spec_cands = []
        spec_pool = spec_cands[:n_pool]
        accept = bool(spec_pool) and {c.chunk_id for c in pool[:5]} <= {c.chunk_id for c in spec_pool}
        gen = None
        if accept:  # a generation error here surfaces as the item's (retryable) error, no second call
            res = await spec_task if spec_task else await generate(spec_pool)
            if res:
                gen, u = res
                add_usage(u)
                pool, speculative = spec_pool, True
        elif spec_pool:
            # Second chance, decided by content not timing: generate on the full pool in parallel and
            # use the speculative answer if every chunk it cites is in the full pipeline's top-5
            # (both pipelines agree on the evidence).  Generations are cached, so a repeated
            # question reaches the same decision and the same answer.
            full_task = asyncio.create_task(generate(pool))
            full_task.add_done_callback(lambda t: t.cancelled() or t.exception())
            try:
                res = await spec_task if spec_task else await generate(spec_pool)
            except Exception:  # noqa: BLE001 - fall back to the full answer
                res = None
            full_top = {c.chunk_id for c in pool[:5]}
            if res and not res[0]["no_answer"] and res[0]["citations"]:
                cited = [c["ctx"] for c in res[0]["citations"]]
                if all(1 <= i <= len(spec_pool) and spec_pool[i - 1].chunk_id in full_top for i in cited):
                    full_task.cancel()
                    gen, u = res
                    add_usage(u)
                    pool, speculative = spec_pool, True
            if gen is None:
                gen, u = await full_task
                add_usage(u)
                speculative = False
        else:
            if spec_task:
                spec_task.cancel()
            speculative = False
        if gen is None:
            gen, u = await generate(pool)
            add_usage(u)
        models_used["answer"] = gen.get("answer_model") or llm.model_id("answer")
    else:
        gen = _extractive(question, ans_lang, pool)
    timing["generation_ms"] = int((time.perf_counter() - ts) * 1000)

    # 4. contexts (reranked) --------------------------------------------------
    rel = gen.get("relevance") or {}
    for i, c in enumerate(pool, 1):
        # the model lists only relevant contexts; unlisted ones scored 0 by the reranker
        c.rerank_score = (round(rel[i], 4) if i in rel else 0.0) if (use_llm and pool and not gen.get("_extractive")) else None
    order = list(range(1, len(pool) + 1))
    if rel:
        # contexts the reranker did not list fill the remaining slots; prefer ones from the act of the
        # best-rated context (neighbouring parts / articles) over unrelated acts
        valid = [i for i in rel if 1 <= i <= len(pool)]
        best_doc = pool[max(valid, key=lambda i: (rel[i], -i)) - 1].doc_id if valid else None
        order.sort(key=lambda i: (-(0.75 * rel.get(i, 0.0) + 0.25 * pool[i - 1].retrieval_score
                                    + (0.1 if i not in rel and pool[i - 1].doc_id == best_doc else 0.0)), pool[i - 1].chunk_id))
    chosen = order[:top_k]
    cited_ctx = [c["ctx"] for c in gen["citations"] if 1 <= c["ctx"] <= len(pool)] if not gen["no_answer"] else []
    for ci in dict.fromkeys(cited_ctx):  # every cited chunk must appear in contexts
        if ci not in chosen:
            for j in range(len(chosen) - 1, -1, -1):
                if chosen[j] not in cited_ctx:
                    chosen[j] = ci
                    break
            else:
                chosen.append(ci)
    chosen.sort(key=order.index)

    # 5. citations: verbatim quotes, ref renumbering ---------------------------
    answer_text = gen["answer"] or ""
    citations, ref_map, dropped = [], {}, 0
    if not gen["no_answer"]:
        for c in sorted(gen["citations"], key=lambda c: c["ref"]):
            if not (1 <= c["ctx"] <= len(pool)) or c["ref"] in ref_map:
                dropped += 1
                continue
            cand = pool[c["ctx"] - 1]
            quote, sim = verbatim_quote(c["quote"], cand.text)
            if sim < 0.2 or not quote:
                dropped += 1
                continue
            new_ref = len(citations) + 1
            ref_map[c["ref"]] = new_ref
            citations.append({
                "ref": new_ref, "doc_id": cand.doc_id, "doc_title": cand.doc_title, "doc_type": cand.doc_type,
                "url": cand.url, "article": cand.article, "part": cand.part, "edition_date": cand.edition_date,
                "doc_status": "amalda" if cand.status == "amalda" else "kuchini_yoqotgan",
                "chunk_id": cand.chunk_id, "quote": quote, "_ctx": c["ctx"], "_quote_exact": sim == 1.0,
                "amended_after_as_of": cand.amended_after_as_of,
            })

        def _sub(m: re.Match) -> str:
            nums = [ref_map.get(int(n)) for n in re.findall(r"\d+", m.group(0))]
            nums = [n for n in nums if n]
            return "".join(f"[{n}]" for n in dict.fromkeys(nums))
        answer_text = re.sub(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]", _sub, answer_text)
        answer_text = re.sub(r"\s+([.,;])", r"\1", answer_text).strip()
        if not citations:  # no verifiable support -> do not return an ungrounded answer
            gen["no_answer"] = True
    no_answer = bool(gen["no_answer"])
    if no_answer:
        answer_text = answer_text if (answer_text and not re.search(r"\[\d+\]", answer_text)) else NO_ANSWER_TEXT.get(ans_lang, NO_ANSWER_TEXT["uz-Latn"])
        citations = []
    cited_chunks = {c["chunk_id"] for c in citations}
    for c in citations:
        c.pop("_ctx"), c.pop("_quote_exact")

    # opt-in: return only contexts the reranker found relevant (top_k stays the upper bound).
    # Cited chunks are always kept; at least the best context is returned.
    min_rr = options.get("min_rerank_score")
    if min_rr is not None and chosen and any(pool[i - 1].rerank_score is not None for i in chosen):
        kept = [i for i in chosen if (pool[i - 1].rerank_score or 0.0) >= min_rr or pool[i - 1].chunk_id in cited_chunks]
        chosen = kept or chosen[:1]

    contexts = []
    for rank, i in enumerate(chosen, 1):
        c = pool[i - 1]
        contexts.append({
            "rank": rank, "chunk_id": c.chunk_id, "doc_id": c.doc_id, "doc_title": c.doc_title, "url": c.url,
            "article": c.article, "part": c.part, "doc_status": "amalda" if c.status == "amalda" else "kuchini_yoqotgan",
            "edition_date": c.edition_date, "text": c.text if options.get("return_contexts", True) else None,
            "retrieval_score": c.retrieval_score, "rerank_score": c.rerank_score, "used_in_answer": c.chunk_id in cited_chunks,
            "amended_after_as_of": c.amended_after_as_of,
        })
    timing["total_ms"] = int((time.perf_counter() - t0) * 1000)
    res: dict[str, Any] = {
        "id": qid,
        "status": "no_answer" if no_answer else "ok",
        "answer": answer_text,
        "answer_language": ans_lang,
        "question_language": lang,
        "no_answer": no_answer,
        "confidence": round(float(gen.get("confidence") or 0.0), 3),
        "citations": citations,
        "contexts": contexts,
        "timing": timing,
        "usage": {**usage, "models": models_used or None},
        "error": None,
    }
    if options.get("return_debug"):
        res["debug"] = {
            "rewritten_query": rw.get("question_uz"),
            "search_queries": queries,
            "doc_hints": rw.get("doc_hints"),
            "article_hints": rw.get("article_hints"),
            "in_scope": rw.get("in_scope"),
            "retrieved_total": rinfo.get("retrieved_total"),
            "rerank_pool": len(pool),
            "rerank": "llm-joint" if use_llm else None,
            "speculative_answer_used": speculative,
            "citations_dropped": dropped,
            "generation_cached": None,
        }
    return res


def error_result(item: dict, code: str, message: str, retryable: bool, status: str = "error", total_ms: int = 0) -> dict:
    return {
        "id": item.get("id"), "status": status, "answer": None, "no_answer": False, "confidence": None,
        "citations": [], "contexts": [],
        "timing": {"rewrite_ms": 0, "retrieval_ms": 0, "rerank_ms": 0, "generation_ms": 0, "total_ms": total_ms},
        "usage": None,
        "error": {"code": code, "message": message, "retryable": retryable},
    }


def system_info() -> dict:
    meta = index_meta()
    use_llm = llm.available()
    return {
        "pipeline_version": settings.pipeline_version,
        "prompt_version": settings.prompt_version,
        "llm_provider": llm.provider(),
        "llm_model": llm.model_id("answer") if use_llm else "extractive-fallback (no LLM key)",
        "rewrite_model": llm.model_id("rewrite"),
        "embedding_model": None,
        "retriever": "bm25-fts5 multi-query + title/article hints + authority prior",
        "reranker": f"{llm.model_id('answer')} (llm-joint rerank)" if use_llm else None,
        "index_version": meta.get("index_version"),
        "lex_snapshot_date": meta.get("lex_snapshot_date"),
        "index_docs": int(meta.get("n_docs", 0) or 0),
        "index_chunks": int(meta.get("n_chunks", 0) or 0),
        "chunking": meta.get("chunking"),
    }
