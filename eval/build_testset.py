"""Build the spec's 100-question test set from the index (section 6 "100 savol tarkibi").

    python eval/build_testset.py --out eval/testset_100.csv

Every answerable question is generated FROM a real in-force article, so the
expected doc/article is correct by construction; an independent, stronger model
then verifies each item against the article text and only passing items are kept.

Category mix (spec):  faktik 35 · raqamlar 15 · bir_nechta 15 · ozgargan 10 ·
rus_kirill 10 · javobsiz 10 · noaniq 5.  Each batch b01..b10 gets a mix.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lexrag import llm  # noqa: E402
from lexrag.retrieval import _df, _search, db  # noqa: E402
from lexrag.textnorm import fold, query_terms  # noqa: E402

MIX = {"faktik": 35, "raqamlar": 15, "bir_nechta": 15, "ozgargan": 10, "rus_kirill": 10, "javobsiz": 10, "noaniq": 5}
GEN_MODEL = "deepseek-flash"
VERIFY_MODEL = "deepseek-v4-pro"   # a different, stronger model checks every item

NUM_WORDS = r"(bir|ikki|uch|toʻrt|besh|olti|yetti|sakkiz|toʻqqiz|oʻn|yigirma|oʻttiz|qirq|ellik|oltmish|yetmish|sakson|toʻqson|yuz|ming|million|\d+)"
NUM_CONTEXT = re.compile(NUM_WORDS + r"\s*(\S+\s+)?(foiz|baravar|bazaviy hisoblash|kalendar kun|ish kun|kun|oy|yil|soat|yosh|eng kam)", re.IGNORECASE)


# --------------------------------------------------------------------------- schemas
class GenItem(BaseModel):
    usable: bool = Field(description="false if the article cannot support a clear, self-contained factual question")
    question: str = Field(description="the question in Uzbek Latin script")
    expected_answer: str = Field(description="short correct answer in Uzbek Latin, with the act and article")


class Translated(BaseModel):
    question: str


class NoAnswer(BaseModel):
    questions: list[str]


class Verdict(BaseModel):
    answerable_from_text: bool
    answer_correct_and_complete: bool
    unambiguous: bool
    reason: str


GEN_SYSTEM = """You write evaluation questions for a legal question-answering system over Uzbekistan's legislation (lex.uz).
You get the text of one or two articles of an act that is IN FORCE. Write ONE question a real citizen or lawyer would ask whose
correct answer is fully contained in the given text.
Rules:
- Uzbek Latin script with the official letters oʻ, gʻ, ʼ.
- Paraphrase in everyday words; do not copy long phrases from the article (the system must not win by keyword overlap).
- Never mention the article number. Mention the act name (e.g. "Mehnat kodeksiga koʻra") only when the question would be ambiguous without it.
- The answer must be checkable: a specific rule, condition, term, amount, age, percentage or list.
- expected_answer: 1-2 sentences in Uzbek Latin, copying numbers exactly as the text gives them, ending with the act and article, e.g. "(Mehnat kodeksi, 217-modda)".
- Set usable=false if the text is only a definition list, a pure reference to other acts, or too vague for a clear question."""

VERIFY_SYSTEM = """You audit items of a legal QA test set. You get the source text (one or more articles), a question and its expected answer.
Judge strictly using ONLY the source text:
- answerable_from_text: the question can be answered from this text alone.
- answer_correct_and_complete: the expected answer is correct and complete per the text (numbers exactly right).
- unambiguous: the question has one clear answer and would not reasonably be answered from a different article or act instead.
Explain briefly in reason."""


def _act_name(title: str) -> str:
    t = re.sub(r"^Oʻzbekiston Respublikasi(ning)?\s+", "", title, flags=re.IGNORECASE).strip()
    return t[0].upper() + t[1:] if t else title


def _pool(rng: random.Random, n_docs_codes=25) -> dict[str, list]:
    con = db()
    rows = con.execute(
        "SELECT c.chunk_id, c.doc_id, c.article, c.heading, c.text, d.title, d.doc_type, d.inbound FROM chunks c JOIN docs d USING(doc_id) "
        "WHERE d.status='amalda' AND (d.doc_type IN ('kodeks','konstitutsiya') OR (d.doc_type='qonun' AND d.inbound>=20)) "
        "AND c.article LIKE '%-modda' AND c.chunk_id LIKE '%#p1' AND length(c.text) BETWEEN 250 AND 1400 "
        "AND c.text NOT LIKE '%chiqarib tashlangan%' AND d.prior > 0.3 ORDER BY c.chunk_id"
    ).fetchall()
    by_doc: dict[str, list] = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(dict(r))
    return by_doc


def _sample_articles(by_doc: dict, rng: random.Random, k: int, pred=None, used=set()) -> list[dict]:
    codes = sorted(d for d, v in by_doc.items() if v[0]["doc_type"] in ("kodeks", "konstitutsiya"))
    laws = sorted(d for d, v in by_doc.items() if v[0]["doc_type"] == "qonun")
    out, tries = [], 0
    while len(out) < k and tries < k * 200:
        tries += 1
        docs = codes if rng.random() < 0.6 else laws
        d = rng.choice(docs)
        cands = [a for a in by_doc[d] if a["chunk_id"] not in used and (pred is None or pred(a))]
        if not cands:
            continue
        a = rng.choice(cands)
        used.add(a["chunk_id"])
        out.append(a)
    return out


_BOILER = {"ozbekiston", "respublikasining", "respublikasi", "qonuni", "kodeksi", "togrisida", "togrisidagi", "birinchi",
           "ikkinchi", "qism", "yangi", "tahriri", "umumiy"}


def act_keywords(title: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]{4,}", fold(title)) if w not in _BOILER][:3]


def has_parallel(a: dict) -> bool:
    """Does an almost identical article exist in ANOTHER in-force act? (e.g. the civil / economic /
    administrative procedure codes share many rules) -> the question must name its act."""
    terms = sorted(set(query_terms(a["text"])), key=lambda t: _df(f'"{t}"*') if len(t) >= 3 else 10**9)[:8]
    if len(terms) < 3:
        return False
    expr = " OR ".join(f'"{t}"*' for t in terms)
    src = set(re.findall(r"[a-z]+", fold(a["text"])))
    hits = _search(expr, 30)
    if not hits:
        return False
    ids = [h[0] for h in hits]
    rows = db().execute(f"SELECT c.text FROM chunks c JOIN docs d USING(doc_id) WHERE c.id IN ({','.join('?' * len(ids))}) "
                        "AND c.doc_id != ? AND d.status='amalda'", (*ids, a["doc_id"])).fetchall()
    for r in rows:
        other = set(re.findall(r"[a-z]+", fold(r[0])))
        if len(src & other) / max(1, len(src | other)) >= 0.45:
            return True
    return False


def names_act(question: str, title: str) -> bool:
    q = fold(question)
    kws = act_keywords(title)
    return bool(kws) and all(k[:5] in q for k in kws[:2])


def _numbers(s: str) -> set[str]:
    return {m.lower() for m in re.findall(NUM_WORDS, s, flags=re.IGNORECASE)}


def _numbers_supported(answer: str, text: str) -> bool:
    """Every number in the expected answer occurs in the source (digits or Uzbek number words)."""
    src = fold(text)
    for n in _numbers(re.sub(r"\(.*?modda\)", "", answer)):
        if n.isdigit():
            if n not in src and not re.search(rf"\b{n}\b", src):
                return False
        elif fold(n) not in src:
            return False
    return True


# --------------------------------------------------------------------------- generation
SEM = asyncio.Semaphore(8)


async def gen(articles: list[dict], instruction: str) -> GenItem | None:
    src = "\n\n".join(f"[{_act_name(a['title'])}, {a['article']}]\n{a['text']}" for a in articles)
    if articles[0].get("_parallel"):
        instruction += (f" IMPORTANT: almost the same rule exists in other acts, so the question MUST explicitly name the act "
                        f"'{_act_name(articles[0]['title'])}'.")
    async with SEM:
        try:
            out, _ = await llm.structured(GEN_SYSTEM, f"{instruction}\n\nSOURCE:\n{src}", GenItem, model=GEN_MODEL, max_tokens=1500)
        except llm.LLMError:
            return None
    return out if out.usable else None


async def verify(articles: list[dict], question: str, answer: str) -> Verdict | None:
    src = "\n\n".join(f"[{_act_name(a['title'])}, {a['article']}]\n{a['text']}" for a in articles)
    async with SEM:
        try:
            v, _ = await llm.structured(VERIFY_SYSTEM, f"SOURCE:\n{src}\n\nQUESTION: {question}\nEXPECTED ANSWER: {answer}", Verdict,
                                        model=VERIFY_MODEL, effort="low", max_tokens=4000)
        except llm.LLMError:
            return None
    return v


async def make_items(category: str, groups: list[list[dict]], instruction: str, need: int) -> list[dict]:
    items = []
    for g in groups:
        g[0]["_parallel"] = has_parallel(g[0])
    gens = await asyncio.gather(*(gen(g, instruction) for g in groups))
    pending = [(g, x) for g, x in zip(groups, gens) if x and "modda" not in x.question.lower()
               and _numbers_supported(x.expected_answer, " ".join(a["text"] for a in g))
               and (not g[0]["_parallel"] or names_act(x.question, g[0]["title"]))]
    verdicts = await asyncio.gather(*(verify(g, x.question, x.expected_answer) for g, x in pending))
    for (g, x), v in zip(pending, verdicts):
        if v and v.answerable_from_text and v.answer_correct_and_complete and v.unambiguous:
            items.append({"category": category, "question": x.question, "expected_answer": x.expected_answer,
                          "expected_doc_id": g[0]["doc_id"], "expected_article": ";".join(dict.fromkeys(a["article"] for a in g)),
                          "answerable": "true", "_source_chunks": [a["chunk_id"] for a in g], "_verify": v.reason,
                          "_parallel_in_other_acts": g[0]["_parallel"]})
        if len(items) >= need:
            break
    print(f"  {category:11} generated {len([x for x in gens if x])}/{len(groups)}, verified {len(items)} (need {need})", file=sys.stderr)
    return items


def _changed_pairs(rng: random.Random) -> list[list[dict]]:
    """(old repealed article, new in-force article) pairs of the same rule with different wording/numbers."""
    con = db()
    pairs = []
    sups = con.execute("SELECT o.doc_id AS old, o.repealed_by AS new FROM docs o JOIN docs n ON n.doc_id=o.repealed_by "
                       "WHERE o.status='kuchini_yoqotgan' AND o.doc_type IN ('kodeks','konstitutsiya','qonun') "
                       "AND n.doc_type IN ('kodeks','konstitutsiya','qonun') AND n.status='amalda'").fetchall()
    for s in sups:
        old = con.execute("SELECT chunk_id, doc_id, article, heading, text FROM chunks WHERE doc_id=? AND chunk_id LIKE '%#p1' AND article LIKE '%-modda'", (s["old"],)).fetchall()
        new = con.execute("SELECT c.chunk_id, c.doc_id, c.article, c.heading, c.text, d.title, d.doc_type FROM chunks c JOIN docs d USING(doc_id) "
                          "WHERE c.doc_id=? AND d.prior > 0.3 AND c.chunk_id LIKE '%#p1' AND c.article LIKE '%-modda' AND length(c.text) BETWEEN 200 AND 1400", (s["new"],)).fetchall()
        hw = lambda h: {w for w in re.findall(r"[a-z]{4,}", fold(re.sub(r"^\S+\s*", "", h or "")))}
        for n in new:
            nh = hw(n["heading"])
            if len(nh) < 2:
                continue
            best = max(old, key=lambda o: len(nh & hw(o["heading"])) / max(1, len(nh | hw(o["heading"]))), default=None)
            if best is None:
                continue
            sim = len(nh & hw(best["heading"])) / max(1, len(nh | hw(best["heading"])))
            if sim >= 0.6 and _numbers(n["text"]) != _numbers(best["text"]) and NUM_CONTEXT.search(n["text"]):
                pairs.append((dict(n), dict(best)))
    rng.shuffle(pairs)
    return pairs


async def build(seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_doc = _pool(rng)
    used: set[str] = set()
    over = 1.8  # oversample, verification drops some
    plan = {}
    plan["faktik"] = [[a] for a in _sample_articles(by_doc, rng, int(MIX["faktik"] * over), used=used)]
    plan["raqamlar"] = [[a] for a in _sample_articles(by_doc, rng, int(MIX["raqamlar"] * over), pred=lambda a: bool(NUM_CONTEXT.search(a["text"])), used=used)]
    multi = []
    for a in _sample_articles(by_doc, rng, int(MIX["bir_nechta"] * over * 1.5), used=used):
        sib = [b for b in by_doc[a["doc_id"]] if b["chunk_id"] != a["chunk_id"] and b["chunk_id"] not in used
               and len(set(fold(b["heading"]).split()) & set(fold(a["heading"]).split()) - {"modda"}) >= 2]
        if sib:
            b = rng.choice(sib)
            used.add(b["chunk_id"])
            multi.append([a, b])
    plan["bir_nechta"] = multi
    base_tr = [[a] for a in _sample_articles(by_doc, rng, int(MIX["rus_kirill"] * over), used=used)]
    base_col = [[a] for a in _sample_articles(by_doc, rng, int(MIX["noaniq"] * over), used=used)]
    changed = _changed_pairs(rng)[: int(MIX["ozgargan"] * over * 1.5)]

    I_FACT = "Write a factual question about one specific rule in this article."
    I_NUM = "Write a question whose answer is a specific number from the text (term, amount, percentage, age, BHM multiple, number of days)."
    I_MULTI = "Two articles are given. Write ONE question whose complete answer needs information from BOTH articles; the expected answer must cite both articles."
    I_COLLOQ = "Write the question in casual spoken Uzbek, as a worried person would type it in a chat (short, informal, maybe without the legal terms), but still answerable from the text."
    res = await asyncio.gather(
        make_items("faktik", plan["faktik"], I_FACT, MIX["faktik"]),
        make_items("raqamlar", plan["raqamlar"], I_NUM, MIX["raqamlar"]),
        make_items("bir_nechta", plan["bir_nechta"], I_MULTI, MIX["bir_nechta"]),
        make_items("_tr", base_tr, I_FACT, MIX["rus_kirill"]),
        make_items("noaniq", base_col, I_COLLOQ, MIX["noaniq"]),
    )
    items = [x for r in res for x in r]

    # rus_kirill: 5 Russian + 5 Uzbek Cyrillic translations of verified factual items
    tr = [x for x in items if x["category"] == "_tr"]
    items = [x for x in items if x["category"] != "_tr"]
    for i, x in enumerate(tr[: MIX["rus_kirill"]]):
        target = "Russian" if i % 2 == 0 else "Uzbek in the Cyrillic script (ўзбек кирилл ёзуви)"
        out, _ = await llm.structured("Translate the legal question faithfully. Return only the translation.", f"Target: {target}\n\n{x['question']}", Translated, model=GEN_MODEL)
        items.append({**x, "category": "rus_kirill", "question": out.question, "_orig_question": x["question"]})

    # ozgargan: rule changed by a newer act; expected = the in-force article
    ch_items = []
    for k, (new, old) in enumerate(changed):
        new["title"] = new.get("title") or ""
        instr = ("The FIRST article is the CURRENT law. The SECOND is the same rule from a repealed act. Write a question about the current rule; "
                 + ("mention the old value as the user's (outdated) belief, e.g. 'avval ... edi, hozir-chi?'. " if k % 2 == 0 else "do not mention the old act. ")
                 + "expected_answer: the CURRENT rule only, citing the current act and article.")
        new["_parallel"] = has_parallel(new)
        g = await gen([new, {**old, "title": "KUCHINI YOʻQOTGAN hujjat"}], instr)
        if not g or "modda" in g.question.lower() or not _numbers_supported(g.expected_answer, new["text"]) \
                or (new["_parallel"] and not names_act(g.question, new["title"])):
            continue
        v = await verify([new], g.question, g.expected_answer)
        if v and v.answerable_from_text and v.answer_correct_and_complete:
            ch_items.append({"category": "ozgargan", "question": g.question, "expected_answer": g.expected_answer,
                             "expected_doc_id": new["doc_id"], "expected_article": new["article"], "answerable": "true",
                             "_source_chunks": [new["chunk_id"], old["chunk_id"]], "_verify": v.reason})
        if len(ch_items) >= MIX["ozgargan"]:
            break
    print(f"  ozgargan    verified {len(ch_items)} (need {MIX['ozgargan']})", file=sys.stderr)
    items += ch_items

    # javobsiz: clearly outside Uzbek law or about non-existent institutions
    out, _ = await llm.structured(
        "You write unanswerable test questions for a QA system over Uzbekistan's legislation.",
        "Write 14 questions in Uzbek Latin that Uzbek legislation does NOT answer: 5 about everyday non-legal topics, 5 about fictional "
        "or impossible situations phrased like legal questions (e.g. property on Mars, a non-existent ministry), 4 asking for opinions or "
        "predictions. They must not be answerable by any real law.", NoAnswer, model=GEN_MODEL)
    for q in out.questions[: MIX["javobsiz"]]:
        items.append({"category": "javobsiz", "question": q, "expected_answer": "", "expected_doc_id": "", "expected_article": "",
                      "answerable": "false", "_source_chunks": [], "_verify": "unanswerable by design"})
    return items


def assign(items: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_cat: dict[str, list] = {}
    for x in items:
        by_cat.setdefault(x["category"], []).append(x)
    for v in by_cat.values():
        rng.shuffle(v)
    order, slot = [], 0
    batches: list[list[dict]] = [[] for _ in range(10)]
    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        for x in by_cat[cat]:
            # next batch with room, round-robin
            for _ in range(10):
                b = slot % 10
                slot += 1
                if len(batches[b]) < 10:
                    batches[b].append(x)
                    break
    n = 0
    for bi, b in enumerate(batches, 1):
        for x in b:
            n += 1
            order.append({**x, "id": f"q{n:03d}", "batch": f"b{bi:02d}"})
    return order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "eval" / "testset_100.csv"))
    ap.add_argument("--seed", type=int, default=20261106)
    a = ap.parse_args()
    items = asyncio.run(build(a.seed))
    counts = {c: sum(1 for x in items if x["category"] == c) for c in MIX}
    print("counts:", counts, file=sys.stderr)
    short = {c: MIX[c] - counts[c] for c in MIX if counts[c] < MIX[c]}
    if short:
        print(f"WARNING: categories short of the spec mix: {short} (re-run with another --seed or raise oversampling)", file=sys.stderr)
    rows = assign(items, a.seed)
    cols = ["id", "batch", "category", "question", "expected_answer", "expected_doc_id", "expected_article", "answerable"]
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    meta = Path(a.out).with_suffix(".provenance.json")
    meta.write_text(json.dumps([{k: r[k] for k in ("id", "category", "_source_chunks", "_verify") if k in r} | {"orig_question": r.get("_orig_question")} for r in rows],
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {a.out} (+ provenance {meta.name})", file=sys.stderr)


if __name__ == "__main__":
    main()
