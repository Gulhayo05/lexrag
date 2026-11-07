"""Hybrid lexical retrieval over the lex.uz index.

Signals combined per chunk (all deterministic -> same question, same index, same
sources):
* BM25 (FTS5, fields title/heading/body) for every query variant -> RRF fusion
* in-act search of the documents the question names ("Mehnat kodeksi") or whose title it covers
* direct article lookup when the question names an article ("134-modda")
* term coverage of the chunk, authority prior of the document type, citation
  in-degree, and a strong penalty for documents that lost force
"""
from __future__ import annotations

import functools
import math
import re
import sqlite3
import threading
from dataclasses import dataclass, field

from .config import settings
from .textnorm import fold, query_terms

_local = threading.local()
_POOL = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="fts")


def db() -> sqlite3.Connection:
    con = getattr(_local, "con", None)
    if con is None:
        con = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute(f"PRAGMA mmap_size={settings.sqlite_mmap_mb * 1_000_000}")
        con.execute(f"PRAGMA cache_size=-{settings.sqlite_cache_mb * 1000}")
        _local.con = con
    return con


def index_meta() -> dict[str, str]:
    try:
        return {r["key"]: r["value"] for r in db().execute("SELECT key, value FROM meta")}
    except sqlite3.Error:
        return {}


@dataclass
class Candidate:
    rowid: int
    chunk_id: str
    doc_id: str
    article: str | None
    part: str | None
    heading: str
    path: str
    text: str
    doc_title: str
    doc_type: str
    url: str
    status: str
    edition_date: str | None
    adopted: str | None
    repeal_date: str | None
    score: float = 0.0
    retrieval_score: float = 0.0
    rerank_score: float | None = None
    amended_after_as_of: bool = False   # article text was changed after the requested as_of_date
    signals: dict = field(default_factory=dict)


@functools.lru_cache(maxsize=50000)
def _df(token: str) -> int:
    """Number of chunks matching an FTS token expression (cached per process)."""
    try:
        return db().execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (token,)).fetchone()[0]
    except sqlite3.Error:
        return 0


@functools.lru_cache(maxsize=1)
def _n_chunks() -> int:
    return int(index_meta().get("n_chunks", 0) or 0) or 1


def _fts_expr(terms: list[str], max_df_ratio: float = 0.10) -> str:
    toks = []
    for t in terms:
        t = re.sub(r"[^a-z0-9]", "", t)
        if t:
            toks.append(f'"{t}"*' if len(t) >= 3 and not t.isdigit() else f'"{t}"')
    toks = list(dict.fromkeys(toks))
    # very frequent terms (e.g. "yil" in 28% of chunks) add almost no ranking signal but make the
    # OR-query score a huge part of the index; drop them when enough rarer terms remain
    limit = max_df_ratio * _n_chunks()
    rare = [t for t in toks if _df(t) <= limit]
    if len(rare) >= 2:
        toks = rare
    return " OR ".join(toks)


@functools.lru_cache(maxsize=8192)
def _search(expr: str, limit: int, title_filter: str | None = None) -> tuple[tuple[int, float], ...]:
    """Cached: prefetching the raw question during query rewriting makes the later full search cheaper."""
    return tuple(_search_uncached(expr, limit, title_filter))


def _search_uncached(expr: str, limit: int, title_filter: str | None = None) -> list[tuple[int, float]]:
    if not expr:
        return []
    q = f"title : ({title_filter}) AND ({expr})" if title_filter else expr
    try:
        rows = db().execute(
            "SELECT rowid, bm25(chunks_fts, 4.0, 3.0, 1.0) AS s FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?",
            (q, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(r[0], r[1]) for r in rows]


def _title_filter(hint: str) -> str | None:
    terms = [t for t in query_terms(hint) if t not in ("kodeks", "qonun", "qaror", "farmon")] or query_terms(hint)
    is_code = "kodeks" in fold(hint)
    is_law = "qonun" in fold(hint)
    terms = [re.sub(r"[^a-z0-9]", "", t) for t in terms][:4]
    terms = [t for t in terms if t]
    if not terms:
        return None
    expr = " AND ".join(f'"{t}"*' if len(t) >= 4 else f'"{t}"' for t in terms)
    if is_code:
        expr += ' AND "kodeks"*'
    elif is_law:
        expr += ' AND "qonun"*'
    return expr


def _hint_doc_ids(hint: str, limit: int = 5) -> list[str]:
    tf = _title_filter(hint)
    if not tf:
        return []
    try:
        rows = db().execute(
            "SELECT d.doc_id FROM docs_fts f JOIN docs d ON d.doc_id = f.doc_id WHERE docs_fts MATCH ? "
            "ORDER BY (d.status='amalda') DESC, d.prior DESC, length(d.title) ASC, d.inbound DESC LIMIT ?",
            (f"title : ({tf})", limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r[0] for r in rows]


_TITLE_BOILERPLATE = {"ozbekiston", "respublikasining", "respublikasi", "qonuni", "birinchi", "ikkinchi", "qism",
                      "yangi", "tahriri", "kodeksi", "kodeks", "konstitutsiyasi", "konstitutsiya", "togrisidagi",
                      "togrisida", "qonun", "umumiy"}


@functools.lru_cache(maxsize=2048)
def primary_docs(hint: str) -> tuple[str, ...]:
    """The act a hint names exactly ('Fuqarolik kodeksi' -> both parts of the Civil Code, not the
    Civil Procedure Code): in-force codes/laws whose title has the fewest words beyond the hint."""
    hw = {w[:6] for w in re.findall(r"[a-z]+", fold(hint))}
    best: list[tuple[int, str]] = []
    for doc_id in _hint_doc_ids(hint, 8):
        r = db().execute("SELECT title, doc_type, status, n_chunks FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        if not r or r["status"] != "amalda" or r["doc_type"] not in ("kodeks", "konstitutsiya", "qonun") or r["n_chunks"] > 3000:
            continue
        words = {w for w in re.findall(r"[a-z]+", fold(r["title"])) if w not in _TITLE_BOILERPLATE}
        extra = sum(1 for w in words if w[:6] not in hw)
        best.append((extra, doc_id))
    if not best:
        # strict title match failed (word forms differ: "olib kirish" vs "olib kirilishi"): fuzzy prefixes
        return discover_acts([query_terms(hint)], k=1, min_cov=0.5)
    m = min(e for e, _ in best)
    return tuple(d for e, d in sorted(best) if e == m)[:2]


@functools.lru_cache(maxsize=64)
def _doc_chunks(doc_id: str) -> tuple[tuple[int, str, str], ...]:
    """(rowid, folded heading, folded heading+text) of every chunk of a document (cached)."""
    rows = db().execute("SELECT id, heading, text FROM chunks WHERE doc_id=? ORDER BY ord", (doc_id,)).fetchall()
    return tuple((r[0], fold(r[1] or ""), fold((r[1] or "") + " " + r[2])) for r in rows)


def _in_doc_search(doc_ids: tuple[str, ...], term_sets: list[list[str]], k: int = 40) -> list[int]:
    """Score every chunk of the named act(s) by query-term coverage (headings count double)."""
    scored = []
    for d in doc_ids:
        for rid, head, body in _doc_chunks(d):
            covs = [(sum(1 for t in ts if t in body) + sum(1 for t in ts if t in head)) / (2 * len(ts)) for ts in term_sets]
            if covs:
                scored.append((0.5 * sum(covs) / len(covs) + 0.5 * max(covs), rid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [rid for sc, rid in scored[:k] if sc > 0]


def discover_acts(term_sets: list[list[str]], exclude: tuple[str, ...] = (), k: int = 2,
                  min_cov: float = 0.3) -> tuple[str, ...]:
    """Acts whose *distinctive* title words the question covers ("narkologik", "oilaviy tadbirkorlik"),
    independent of the rewriter's guesses.  In-force, non-amending codes/laws only."""
    terms = list(dict.fromkeys(t for ts in term_sets for t in ts if len(t) >= 4))
    if not terms:
        return ()
    expr = " OR ".join(f'"{t}"*' for t in terms[:24])
    try:
        rows = db().execute(
            "SELECT d.doc_id, d.title_fold, d.n_chunks FROM docs_fts f JOIN docs d ON d.doc_id = f.doc_id "
            "WHERE docs_fts MATCH ? AND d.status='amalda' AND d.prior >= 0.8 AND d.n_chunks <= 3000 "
            "ORDER BY bm25(docs_fts) LIMIT 15", (f"title : ({expr})",)).fetchall()
    except sqlite3.OperationalError:
        return ()
    scored = []
    for r in rows:
        if r["doc_id"] in exclude:
            continue
        words = [w for w in re.findall(r"[a-z]+", r["title_fold"]) if w not in _TITLE_BOILERPLATE and len(w) >= 4]
        words = list(dict.fromkeys(words))
        if not words:
            continue
        hit = sum(1 for w in words if any(w.startswith(t[:5]) or t.startswith(w[:5]) for t in terms))
        cov = hit / len(words)
        if hit >= 2 and cov >= min_cov:
            scored.append((hit + cov, r["doc_id"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return tuple(d for _, d in scored[:k])


_SELECT = (
    "SELECT c.id, c.chunk_id, c.doc_id, c.article, c.part, c.heading, c.path, c.text, d.title, d.doc_type, d.url, "
    "d.status, COALESCE(c.edition_date, d.edition_date) AS edition_date, d.adopted, d.repeal_date, d.prior, d.inbound "
    "FROM chunks c JOIN docs d ON d.doc_id = c.doc_id "
)


def _load(rowids: list[int]) -> dict[int, sqlite3.Row]:
    out: dict[int, sqlite3.Row] = {}
    for i in range(0, len(rowids), 500):
        part = rowids[i:i + 500]
        q = _SELECT + f"WHERE c.id IN ({','.join('?' * len(part))})"
        for r in db().execute(q, part):
            out[r[0]] = r
    return out


ARTICLE_IN_Q = re.compile(r"(\d+)\s*[-–]?\s*(?:modda|moddasi|moddasining|статья|статьи|статье|ст\.)", re.IGNORECASE)
ARTICLE_IN_Q2 = re.compile(r"(?:статья|статьи|статье|ст\.)\s*(\d+)", re.IGNORECASE)


def retrieve(
    queries: list[str],
    doc_hints: list[str] | None = None,
    article_hints: list[str] | None = None,
    as_of_date: str | None = None,
    limit: int | None = None,
) -> tuple[list[Candidate], dict]:
    limit = limit or settings.candidates
    doc_hints = [h for h in (doc_hints or []) if h and h.strip()][:2]
    queries = list(dict.fromkeys(q for q in queries if q and q.strip()))[:8]
    lists: list[list[int]] = []
    all_terms: list[str] = []
    term_sets: list[list[str]] = []
    jobs: list[tuple[str, int, str | None]] = []  # (fts expression, limit, title filter)
    for q in queries:
        terms = query_terms(q)
        all_terms += [t for t in terms if t not in all_terms]
        if not terms:
            continue
        term_sets.append(terms)
        # One global BM25 query per variant.  Named acts are searched by _in_doc_search below;
        # extra title-filtered BM25 queries per hint cost 2x the FTS work and, measured on the
        # 100-question set, lowered recall (they push hinted-but-wrong acts up the list).
        jobs.append((_fts_expr(terms), 150, None))
    jobs = list(dict.fromkeys(jobs))
    # independent FTS queries run in parallel (sqlite releases the GIL; one connection per thread)
    results = list(_POOL.map(lambda j: _search(*j), jobs)) if len(jobs) > 1 else [_search(*j) for j in jobs]
    lists += [[r for r, _ in res] for res in results if res]

    # article lookup ("134-modda" in question or hinted by the rewriter)
    arts: list[str] = []
    for q in queries:
        arts += [m for m in ARTICLE_IN_Q.findall(q)] + [m for m in ARTICLE_IN_Q2.findall(q)]
    arts += [re.sub(r"\D.*", "", a) for a in (article_hints or []) if re.match(r"\d", a or "")]
    art_rows: list[int] = []
    if arts:
        hint_docs = [d for h in doc_hints for d in _hint_doc_ids(h, 2)]
        for a in dict.fromkeys(arts):
            if hint_docs:
                q = f"SELECT id FROM chunks WHERE article = ? AND doc_id IN ({','.join('?' * len(hint_docs))}) ORDER BY ord LIMIT 4"
                art_rows += [r[0] for r in db().execute(q, [f"{a}-modda", *hint_docs])]
    if art_rows:
        lists.append(art_rows)
    # search inside the act(s) the question names (fast: a code has only ~200-1500 chunks)
    prim = tuple(dict.fromkeys(d for h in doc_hints for d in primary_docs(h)))
    reserved: list[int] = []  # best in-act chunks per named act: always reach the reranker
    if term_sets:
        groups = [primary_docs(h) for h in doc_hints]
        found = discover_acts(term_sets, exclude=prim)          # acts the question's words point to
        prim = tuple(dict.fromkeys(prim + found))
        groups += [(d,) for d in found]
        for docs_h in groups:
            if not docs_h:
                continue
            in_doc = _in_doc_search(docs_h, term_sets)
            if in_doc:
                lists.append(in_doc)
                small = sum(len(_doc_chunks(d)) for d in docs_h) <= 40
                reserved += in_doc[: 8 if small else 4]

    rrf: dict[int, float] = {}
    for lst in lists:
        for rank, rid in enumerate(lst):
            rrf[rid] = rrf.get(rid, 0.0) + 1.0 / (40 + rank)
    for rid in art_rows:
        rrf[rid] = rrf.get(rid, 0.0) + 0.05
    if not rrf:
        return [], {"retrieved_total": 0, "terms": all_terms}

    pool = sorted(rrf, key=lambda r: -rrf[r])[: max(limit * 3, 150)]
    reserved = list(dict.fromkeys(reserved))[:12]  # cap: reserved slots must not crowd out strong global hits
    pool += [r for r in reserved if r not in pool]
    rows = _load(pool)
    hint_folds = [fold(h) for h in doc_hints]
    cands: list[Candidate] = []
    max_rrf = max(rrf.values())
    for rid in pool:
        r = rows.get(rid)
        if r is None:
            continue
        adopted, repeal_date = r["adopted"], r["repeal_date"]
        if as_of_date and adopted and adopted > as_of_date:
            continue  # did not exist yet at as_of_date
        body = fold(r["heading"] + " " + r["text"] + " " + r["title"])
        # coverage per query variant: mean of all variants and the best one.  The mean keeps
        # short generic variants from saturating; the max keeps colloquial words of the raw
        # question from penalising an article that matches the legal phrasing perfectly.
        covs = [sum(1 for t in ts if t in body) / len(ts) for ts in term_sets] or [0.0]
        cov = 0.5 * (sum(covs) / len(covs)) + 0.5 * max(covs)
        in_force = r["status"] == "amalda" or (as_of_date is not None and repeal_date is not None and repeal_date > as_of_date)
        prior = r["prior"] + 0.15 * min(1.0, math.log1p(r["inbound"]) / 8)
        hint_match = r["doc_id"] in prim or any(_hint_match(hf, fold(r["title"])) for hf in hint_folds)
        s = 0.45 * (rrf[rid] / max_rrf) + 0.30 * cov + 0.15 * prior + (0.15 if hint_match else 0.0)
        if not in_force:
            s *= 0.35
        c = Candidate(
            rowid=rid, chunk_id=r["chunk_id"], doc_id=r["doc_id"], article=r["article"], part=r["part"],
            heading=r["heading"], path=r["path"], text=r["text"], doc_title=r["title"], doc_type=r["doc_type"],
            url=r["url"], status=r["status"], edition_date=r["edition_date"], adopted=adopted,
            amended_after_as_of=bool(as_of_date and r["edition_date"] and r["edition_date"] > as_of_date),
            repeal_date=repeal_date, score=s,
            signals={"rrf": round(rrf[rid] / max_rrf, 4), "coverage": round(cov, 3), "prior": round(prior, 3),
                     "hint": hint_match, "in_force": in_force},
        )
        cands.append(c)
    cands.sort(key=lambda c: (-c.score, c.chunk_id))
    # diversity: at most 5 chunks per document in the candidate list
    per_doc: dict[str, int] = {}
    out: list[Candidate] = []
    res_set = set(reserved)
    must = [c for c in cands if c.rowid in res_set]
    for c in must:  # reserved in-act chunks first (their own order is kept by score below)
        per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
    for c in cands:
        if c.rowid in res_set:
            continue
        if per_doc.get(c.doc_id, 0) >= settings.max_chunks_per_doc:
            continue
        per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
        out.append(c)
        if len(out) >= limit:
            break
    merged = sorted(out[: max(0, limit - len(must))] + must, key=lambda c: (-c.score, c.chunk_id))
    cutoff = max(1, min(settings.rerank_pool, len(merged)) - 6)
    for c in [m for m in merged if m.rowid in res_set]:
        i = merged.index(c)
        if i >= cutoff:  # keep reserved chunks inside the reranker's window
            merged.insert(cutoff - 1, merged.pop(i))
    out = merged
    top = max((c.score for c in out), default=1.0)
    for c in out:
        c.retrieval_score = round(c.score / top, 4) if top else 0.0
    return out, {"retrieved_total": len(rrf), "terms": all_terms, "article_hits": len(art_rows)}


def _hint_match(hint_fold: str, title_fold: str) -> bool:
    words = [w for w in re.findall(r"[a-z]+", hint_fold) if len(w) > 3 and w not in ("ozbekiston", "respublikasining", "respublikasi")]
    return bool(words) and all(w[:6] in title_fold for w in words)
