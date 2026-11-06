"""Build the lex.uz search index (SQLite + FTS5) from the converted text files.

    python -m lexrag.ingest            # uses LEXRAG_TXT_DIR / LEXRAG_DB

Steps
1. parse every document in parallel (acts, docs, kodekslar)
2. resolve the lex.uz id of the 20 code files (their file names are dates)
3. de-duplicate (a code present both in ``kodekslar`` and ``acts`` is kept once,
   the more recent edition wins)
4. derive ``doc_status`` (amalda / kuchini_yoqotgan) from repeal statements
   found in other documents ("... oʻz kuchini yoʻqotgan deb topilsin")
5. write docs, chunks and a contentless FTS5 index over folded text
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import settings
from .parse import Document, parse_document
from .textnorm import fold, tokens

# Official lex.uz ids of the codes shipped as date-named files.  Used only when
# the code cannot be matched to a copy inside acts/ docs/; every id is checked
# against the citation graph during ingestion (see ``resolve_codes``).
KNOWN_CODE_IDS = {
    "HAVO KODEKSI": "-55594",
    "BOJXONA KODEKSI": "-2876354",
    "FUQAROLIK KODEKSI|BIRINCHI": "-111189",
    "FUQAROLIK KODEKSI|IKKINCHI": "-180552",
    "MEHNAT KODEKSI|1995": "-142859",
    "MEHNAT KODEKSI|2022": "-6257288",
    "FUQAROLIK PROTSESSUAL KODEKSI": "-3517337",
    "SHAHARSOZLIK KODEKSI": None,
    "JINOYAT-PROTSESSUAL KODEKSI": "-111460",
    "JINOYAT KODEKSI": "-111453",
    "IQTISODIY PROTSESSUAL KODEKSI": "-3523891",
    "UY-JOY KODEKSI": "-106136",
    "MAʼMURIY SUD ISHLARINI YURITISH": "-3527353",
    "JINOYAT-IJROIYA KODEKSI": "-163629",
    "SAYLOV KODEKSI": "-4386848",
    "BUDJET KODEKSI": "-2304140",
    "OILA KODEKSI": "-104720",
    "YER KODEKSI": "-152653",
    "SOLIQ KODEKSI|2019": "-4674893",
    "KONSTITUTSIYASI|2023": "-6445145",
}

AMENDING_RE = re.compile(r"ozgartish|ozgartirish|qoshimcha(lar)? kiritish|kiritish togrisida|tuzatish(lar)? kiritish|kuchini yoqotgan deb")

TYPE_PRIOR = {  # authority prior used at ranking time
    "konstitutsiya": 1.0, "kodeks": 1.0, "qonun": 0.8, "farmon": 0.55, "prezident_qarori": 0.5,
    "vm_qarori": 0.45, "qaror": 0.3, "nizom": 0.3, "farmoyish": 0.2, "buyruq": 0.25, "boshqa": 0.2,
}


def _parse_file(args: tuple[str, str]) -> Document | None:
    path, source = args
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if source == "kodekslar":
        m = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", p.stem)
        hint = f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None
        doc = parse_document(raw, None, source, adopted_hint=hint)
        doc.lex_num = "unknown"  # resolved later
        return doc
    return parse_document(raw, p.stem, source)


def _code_key(doc: Document) -> str:
    t = doc.title.upper()
    for key in KNOWN_CODE_IDS:
        name, _, qual = key.partition("|")
        if name not in t:
            continue
        if qual == "BIRINCHI" and "IKKINCHI" in t:
            continue
        if qual == "IKKINCHI" and "IKKINCHI" not in t:
            continue
        if qual.isdigit() and not (doc.adopted or "").startswith(qual):
            continue
        if name == "JINOYAT KODEKSI" and ("PROTSESSUAL" in t or "IJROIYA" in t):
            continue
        if name == "FUQAROLIK KODEKSI" and "PROTSESSUAL" in t:
            continue
        return key
    return ""


def _sig(doc: Document, n: int = 40) -> set[str]:
    words = tokens(" ".join(c.text for c in doc.chunks[:n]))
    return {" ".join(words[i:i + 3]) for i in range(0, max(len(words) - 2, 0), 2)}


def resolve_codes(codes: list[Document], by_id: dict[str, Document], inbound: Counter) -> None:
    """Give each date-named code its lex.uz id (match against corpus, else known table)."""
    candidates = [d for d in by_id.values() if d.doc_type in ("kodeks", "konstitutsiya")]
    for doc in codes:
        sig = _sig(doc)
        best, best_sim = None, 0.0
        for cand in candidates:
            if fold(cand.title)[:40] != fold(doc.title)[:40] and _code_key(cand) != _code_key(doc):
                continue
            cs = _sig(cand)
            sim = len(sig & cs) / max(1, min(len(sig), len(cs)))
            if sim > best_sim:
                best, best_sim = cand, sim
        if best is not None and best_sim >= 0.35:
            lex, how = best.lex_num, f"matched {best.doc_id} sim={best_sim:.2f}"
        else:
            lex = KNOWN_CODE_IDS.get(_code_key(doc))
            how = f"known-table key={_code_key(doc)!r} inbound_links={inbound.get(lex, 0) if lex else 0}"
            if not lex:
                lex = "kodeks-" + (doc.adopted or "x").replace("-", "")
                how = "UNRESOLVED (synthetic id)"
        doc.lex_num = lex
        doc.doc_id = f"lex:{lex}"
        doc.url = f"https://lex.uz/docs/{lex}"
        for c in doc.chunks:
            c.doc_id = doc.doc_id
            c.chunk_id = doc.doc_id + c.chunk_id[c.chunk_id.index("#"):]
        print(f"  code {doc.title[:55]!r:60} -> {doc.doc_id:16} ({how})", file=sys.stderr)


SCHEMA = """
PRAGMA journal_mode=OFF;
PRAGMA synchronous=OFF;
CREATE TABLE docs(
  doc_id TEXT PRIMARY KEY, lex_num TEXT, url TEXT, title TEXT, title_fold TEXT, doc_type TEXT,
  adopted TEXT, edition_date TEXT, number TEXT, source TEXT, status TEXT, repealed_by TEXT,
  repeal_date TEXT, inbound INTEGER, n_chunks INTEGER, n_chars INTEGER, prior REAL);
CREATE TABLE chunks(
  id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, doc_id TEXT, article TEXT, part TEXT,
  heading TEXT, path TEXT, text TEXT, ord INTEGER, edition_date TEXT);
CREATE INDEX chunks_doc ON chunks(doc_id, ord);
CREATE INDEX chunks_art ON chunks(doc_id, article);
CREATE VIRTUAL TABLE chunks_fts USING fts5(title, heading, body, content='', tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE docs_fts USING fts5(doc_id UNINDEXED, title, tokenize='unicode61 remove_diacritics 2');
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
"""


def build(txt_dir: Path, db_path: Path, workers: int = os.cpu_count() or 4) -> None:
    t0 = time.time()
    jobs = [(str(p), src) for src in ("acts", "docs", "kodekslar") for p in sorted((txt_dir / src).glob("*.txt"))]
    print(f"parsing {len(jobs)} files with {workers} workers ...", file=sys.stderr)
    docs: list[Document] = []
    with ProcessPoolExecutor(workers) as ex:
        for i, d in enumerate(ex.map(_parse_file, jobs, chunksize=64)):
            if d is not None and d.chunks:
                docs.append(d)
            if i % 5000 == 0:
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", file=sys.stderr)

    by_id: dict[str, Document] = {}
    codes: list[Document] = []
    for d in docs:
        if d.source == "kodekslar":
            codes.append(d)
        elif d.doc_id not in by_id or d.n_chars > by_id[d.doc_id].n_chars:
            by_id[d.doc_id] = d

    inbound: Counter = Counter()
    for d in by_id.values():
        for l in set(d.links):
            inbound[l] += 1
        for l in d.self_refs:          # "Oldingi tahrirga qarang" links name the canonical id
            inbound[l] += 50

    # The export contains many acts twice, as N and -N with identical text.
    # Keep the id that lex.uz itself links to (canonical), drop the twin.
    n_twins = 0
    for d in list(by_id.values()):
        if not d.lex_num.lstrip("-").isdigit():
            continue
        twin_id = "lex:" + (d.lex_num[1:] if d.lex_num.startswith("-") else "-" + d.lex_num)
        t = by_id.get(twin_id)
        if t is None or d.doc_id not in by_id or t.n_chars != d.n_chars:
            continue
        keep, drop = sorted((d, t), key=lambda x: (-inbound.get(x.lex_num, 0), not x.lex_num.startswith("-")))
        del by_id[drop.doc_id]
        n_twins += 1
    print(f"  removed {n_twins} duplicate twins (N / -N)", file=sys.stderr)
    resolve_codes(codes, by_id, inbound)
    for c in codes:  # newer edition wins
        old = by_id.get(c.doc_id)
        if old is None or (c.edition_date or "") >= (old.edition_date or ""):
            if old is not None:
                c.repeals = sorted(set(c.repeals) | set(old.repeals))
            by_id[c.doc_id] = c

    # ---- status from repeal graph -------------------------------------------
    repealed_by: dict[str, tuple[str, str | None]] = {}
    for d in by_id.values():
        for target in d.repeals:
            tdoc = by_id.get(f"lex:{target}")
            if tdoc is not None and d.adopted and tdoc.adopted and d.adopted < tdoc.adopted:
                continue  # a document cannot repeal a later one (link was re-pointed by lex.uz)
            prev = repealed_by.get(target)
            if prev is None or (d.adopted or "9999") < (prev[1] or "9999"):
                repealed_by[target] = (d.doc_id, d.adopted)

    # ---- lex.uz banner on the act itself ------------------------------------
    n_banner = 0
    for d in by_id.values():
        if d.header_repealed and d.lex_num not in repealed_by:
            repealed_by[d.lex_num] = ("", None if d.header_repealed == "?" else d.header_repealed)
            n_banner += 1
    print(f"  {n_banner} acts marked repealed by their own lex.uz banner", file=sys.stderr)

    # ---- renamed successors (supersession by name cannot see these) ----------
    RENAMED = {"xojalik protsessual kodeksi": "iqtisodiy protsessual kodeksi",
               "mehnat qonunlari kodeksi": "mehnat kodeksi",
               "mamuriy huquqbuzarliklar togrisidagi kodeksi": "mamuriy javobgarlik togrisidagi kodeksi"}
    for old_name, new_name in RENAMED.items():
        news = [d for d in by_id.values() if d.doc_type == "kodeks" and new_name in fold(d.title)]
        for d in by_id.values():
            if d.doc_type == "kodeks" and old_name in fold(d.title) and news and d.lex_num not in repealed_by:
                nd = max(news, key=lambda x: x.adopted or "")
                repealed_by[d.lex_num] = (nd.doc_id, nd.adopted)
                print(f"  renamed successor: {d.doc_id} -> {nd.doc_id}", file=sys.stderr)

    # ---- supersession: a newer code / constitution / law with the same name ----
    def core_title(d: Document) -> str:
        t = fold(d.title)
        t = re.sub(r"\(.*?\)", " ", t)
        t = re.sub(r"\b(ozbekiston|respublikasining|respublikasi|qonuni|yangi|tahriri|tahrirda)\b", " ", t)
        if d.doc_type in ("kodeks", "konstitutsiya"):
            m = re.search(r"([a-z\- ]*?(kodeksi|konstitutsiyasi)( (birinchi|ikkinchi) qism)?)", t)
            t = m.group(1) if m else t
        return re.sub(r"\s+", " ", t).strip()

    groups: dict[str, list[Document]] = defaultdict(list)
    for d in by_id.values():
        ft = fold(d.title)
        if d.doc_type == "qonun" and re.search(r"ozgartish|ozgartirish|qoshimcha|kiritish|tasdiqlash|ratifikatsiya|munosabati|qonuniga|kodeksiga|kodekslariga|qonunlariga|hujjatlariga|tuzatish|amalga|bekor|kuchini", ft):
            continue
        if d.doc_type in ("kodeks", "konstitutsiya", "qonun") and d.adopted and "qoraqalpog" not in ft:
            groups[(d.doc_type, core_title(d))].append(d)
    n_sup = 0
    for (_t, name), ds in groups.items():
        if len(ds) < 2 or len(name) < 6:
            continue
        ds.sort(key=lambda d: d.adopted)
        newest = ds[-1]
        for old in ds[:-1]:
            days = (time.mktime(time.strptime(newest.adopted, "%Y-%m-%d")) - time.mktime(time.strptime(old.adopted, "%Y-%m-%d"))) / 86400
            if days > 180 and old.lex_num not in repealed_by:
                repealed_by[old.lex_num] = (newest.doc_id, newest.adopted)
                n_sup += 1
                print(f"  superseded: {old.doc_id} ({old.adopted}) by {newest.doc_id} ({newest.adopted}) [{name[:50]}]", file=sys.stderr)
    print(f"  {n_sup} documents superseded by a newer act with the same name", file=sys.stderr)

    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    n_chunks = 0
    for d in sorted(by_id.values(), key=lambda x: x.doc_id):
        rb = repealed_by.get(d.lex_num)
        status = "kuchini_yoqotgan" if rb else "amalda"
        prior = TYPE_PRIOR.get(d.doc_type, 0.2)
        if d.doc_type in ("qonun", "farmon", "prezident_qarori", "vm_qarori", "qaror") and AMENDING_RE.search(fold(d.title)):
            prior = min(prior, 0.3)  # amending act: the amended code already contains the current text
        con.execute(
            "INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (d.doc_id, d.lex_num, d.url, d.title, fold(d.title), d.doc_type, d.adopted, d.edition_date, d.number,
             d.source, status, (rb[0] or None) if rb else None, rb[1] if rb else None, inbound.get(d.lex_num, 0),
             len(d.chunks), d.n_chars, prior),
        )
        con.execute("INSERT INTO docs_fts(doc_id, title) VALUES (?,?)", (d.doc_id, fold(d.title)))
        title_f = fold(d.title)
        for c in d.chunks:
            n_chunks += 1
            cur = con.execute(
                "INSERT INTO chunks(chunk_id, doc_id, article, part, heading, path, text, ord, edition_date) VALUES (?,?,?,?,?,?,?,?,?)",
                (c.chunk_id, c.doc_id, c.article, c.part, c.heading, c.path, c.text, c.ord,
                 c.edition_date if (c.edition_date and (not d.adopted or c.edition_date >= d.adopted)) else d.adopted),
            )
            con.execute(
                "INSERT INTO chunks_fts(rowid, title, heading, body) VALUES (?,?,?,?)",
                (cur.lastrowid, title_f, fold(c.path + " " + c.heading), fold(c.text)),
            )
    snap = max((d.edition_date or "") for d in by_id.values() if (d.edition_date or "") <= time.strftime("%Y-%m-%d"))
    meta = {
        "index_version": f"lex-idx-{time.strftime('%Y-%m-%d')}",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "lex_snapshot_date": snap,
        "n_docs": str(len(by_id)), "n_chunks": str(n_chunks),
        "n_repealed": str(sum(1 for d in by_id.values() if d.lex_num in repealed_by)),
        "chunking": "structure-aware: N-modda / N-band, parts packed to <=1400 chars",
    }
    con.executemany("INSERT INTO meta VALUES (?,?)", meta.items())
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
    con.commit()
    con.execute("ANALYZE")
    con.close()
    print(json.dumps(meta, indent=1), file=sys.stderr)
    print(f"done in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    build(Path(settings.txt_dir), Path(settings.db_path))
