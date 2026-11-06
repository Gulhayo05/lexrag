"""Parse a lex.uz text export into document metadata + article-level chunks.

Chunking strategy (answer to the spec's open question): **structure first**.
Text is split on ``N-modda`` headings (laws/codes) or ``N.`` points (decrees,
resolutions, regulations).  Each article/point is then split into *parts*
(paragraphs = qism/xatboshi) and parts are packed into chunks of at most
``MAX_CHUNK_CHARS`` characters, so a chunk never crosses an article boundary and
``citations[].article`` is always exact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .textnorm import unify_apostrophes

MAX_CHUNK_CHARS = 1400
MONTHS = {
    "yanvar": 1, "fevral": 2, "mart": 3, "aprel": 4, "may": 5, "iyun": 6, "iyul": 7,
    "avgust": 8, "sentabr": 9, "sentyabr": 9, "oktabr": 10, "oktyabr": 10, "noyabr": 11, "dekabr": 12,
}
_MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))
UZ_DATE_RE = re.compile(rf"(\d{{4}})-yil\s+(\d{{1,2}})-({_MONTH_ALT})", re.IGNORECASE)
DOT_DATE_RE = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})-y\b")

FIELD_JUNK_RE = re.compile(r'\s*INCLUDEPICTURE\s+"[^"]*"(?:\s*\\[a-z*]+(?:\s*"[^"]*")?)*\s*|LexUZ sharhi')
HYPERLINK_RE = re.compile(r'\s*HYPERLINK\s+"([^"]*)"(?:\s*\\l\s*"([^"]*)")?(?:\s*\\[a-z]\s*"[^"]*")*\s*')
DOC_LINK_RE = re.compile(r"lex\.uz/(?:[a-z]{2}/)?(?:docs/|[a-z]{2}(?=-))(-?\d+)")
PREV_EDITION_RE = re.compile(r"^\s*(Oldingi|Avvalgi) tahrirga qarang\.?\s*$", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^\s*(\d+(?:[-–]\d+)?)\s*[-–]\s*modda\.?\s*(.*)$", re.IGNORECASE)
POINT_RE = re.compile(r"^\s*(\d{1,3})\.\s+(\S.*)$")
HEADING_RE = re.compile(
    r"^\s*(?:[IVXLC\d]+[-–]?\s*(?:bob|boʻlim|BOB|BOʻLIM|kichik boʻlim|KICHIK BOʻLIM)\b|(?:BIRINCHI|IKKINCHI|UCHINCHI|TOʻRTINCHI|BESHINCHI|OLTINCHI|YETTINCHI|SAKKIZINCHI|TOʻQQIZINCHI|OʻNINCHI)\s+BOʻLIM|\d+\s*-\s*§|§\s*\d+|UMUMIY QISM|MAXSUS QISM)",
)
# editorial notes: "(135-moddaning ... Qonuni tahririda — ...)"
NOTE_RE = re.compile(
    r"^\s*\(.*(tahririda|bilan kiritilgan|bilan chiqarilgan|bilan qoʻshilgan|bilan oʻzgartirilgan|chiqarib tashlangan|kuchini yoʻqotgan|bilan toʻldirilgan|Qonuniga asosan|qaroriga asosan|farmoniga asosan|qarori bilan|Qonuni bilan|Farmoni bilan).*\)\s*$",
    re.IGNORECASE,
)
# banner lex.uz puts on top of acts that lost force (whole act, not "… deb topilsin" directives)
HEADER_REPEAL_RE = re.compile(
    r"(Mazkur|Ushbu)\s+(Kodeks|Qonun|hujjat|qaror|Qaror|Farmon|farmon|Nizom|nizom|Farmoyish|farmoyish|buyruq|Buyruq|Yoʻriqnoma|yoʻriqnoma|Tartib|tartib|Reglament)\b(.{0,500}?)oʻz\s+kuchini\s+yoʻqotgan(?!\s+deb)",
    re.DOTALL,
)
REPEAL_PHRASE_RE = re.compile(r"(oʻz\s+)?kuchini\s+yoʻqotgan\s+deb\s+(topilsin|hisoblansin|topilgan|hisoblanadi)|bekor\s+qilinsin", re.IGNORECASE)
PARTIAL_RE = re.compile(r"\d+[-–](band|modda|qism|xatboshi|ilova|kichik band|bob|boʻlim)", re.IGNORECASE)

DOC_TYPE_RULES = [
    ("konstitutsiya", re.compile(r"KONSTITUTSIYASI")),
    ("kodeks", re.compile(r"\bKODEKSI\b")),
    ("qonun", re.compile(r"\bQONUNI\b")),
    ("farmon", re.compile(r"\bFARMONI\b")),
    ("prezident_qarori", re.compile(r"PREZIDENTINING\s+QARORI")),
    ("vm_qarori", re.compile(r"VAZIRLAR\s+MAHKAMASINING\s+QARORI")),
    ("qaror", re.compile(r"\bQARORI\b|\bQAROR\b")),
    ("farmoyish", re.compile(r"FARMOYISHI")),
    ("buyruq", re.compile(r"BUYRUGʻI")),
    ("nizom", re.compile(r"\bNIZOM\b")),
]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    article: str | None
    part: str | None
    heading: str          # article/point heading line
    path: str             # chapter / section path
    text: str
    ord: int
    edition_date: str | None = None  # last amendment of this article/point seen in editorial notes


@dataclass
class Document:
    doc_id: str                 # "lex:-100122"
    lex_num: str                # "-100122"
    url: str
    title: str
    doc_type: str
    adopted: str | None         # ISO date
    edition_date: str | None    # ISO date of last amendment seen in text
    number: str | None
    source: str                 # acts|docs|kodekslar
    links: list[str] = field(default_factory=list)        # lex nums referenced
    repeals: list[str] = field(default_factory=list)      # lex nums this doc declares void
    self_refs: dict[str, int] = field(default_factory=dict)
    header_repealed: str | None = None   # lex.uz banner "Mazkur Kodeks … oʻz kuchini yoʻqotgan": date (or "?")
    chunks: list[Chunk] = field(default_factory=list)
    n_chars: int = 0


def _iso(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def find_dates(text: str) -> list[str]:
    out = []
    for y, d, mon in UZ_DATE_RE.findall(text):
        iso = _iso(int(y), MONTHS[mon.lower()], int(d))
        if iso:
            out.append(iso)
    for d, m, y in DOT_DATE_RE.findall(text):
        iso = _iso(int(y), int(m), int(d))
        if iso:
            out.append(iso)
    return out


def _strip_links(line: str, sink: list[tuple[str, str | None]]) -> str:
    def repl(m: re.Match) -> str:
        sink.append((m.group(1), m.group(2)))
        return " "
    line = HYPERLINK_RE.sub(repl, FIELD_JUNK_RE.sub(" ", line))
    return re.sub(r"[ \t ]+", " ", line).strip()


def _link_tails(rawline: str) -> list[tuple[str, str | None, str]]:
    """(url, anchor, text after the link up to the next link) for each hyperlink in a raw line."""
    ms = list(HYPERLINK_RE.finditer(rawline))
    out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(rawline)
        out.append((m.group(1), m.group(2), rawline[m.end():end]))
    return out


def _doc_type(header: list[str]) -> str:
    if not header:
        return "boshqa"
    first = header[0].upper()
    if re.search(r"KONSTITUTSIYASI\b", first) and not re.search(r"QONUNI|QARORI|FARMONI", first):
        return "konstitutsiya"
    if re.search(r"KODEKSI\b", first) and not re.search(r"QONUNI|QARORI|FARMONI", first):
        return "kodeks"
    zone = ""
    for h in header:
        zone += " " + h.upper()
        if re.search(r"\b(QONUNI|FARMONI|QARORI|QAROR|FARMOYISHI|BUYRUGʻI|NIZOM)\b", h.upper()):
            break
    for name, rx in DOC_TYPE_RULES:
        if name in ("kodeks", "konstitutsiya"):
            continue
        if rx.search(zone):
            return name
    return "boshqa"


def _sentence_case(s: str) -> str:
    s = s.strip(" .")
    if not s:
        return s
    if s.upper() != s:  # already mixed case
        return s
    low = s.lower()
    low = low[0].upper() + low[1:]
    for w in ("oʻzbekiston", "respublikasi", "respublikasining", "prezidenti", "prezidentining",
              "vazirlar mahkamasi", "oliy majlis", "markaziy bank", "toshkent", "qoraqalpogʻiston"):
        low = re.sub(rf"\b{w}", lambda m: m.group(0)[0].upper() + m.group(0)[1:], low)
    return low


def parse_document(raw: str, lex_num: str | None, source: str, adopted_hint: str | None = None) -> Document:
    raw = unify_apostrophes(raw.replace("\r\n", "\n").replace("\r", "\n").replace("\x0b", "\n").replace("\x0c", "\n"))
    links: list[tuple[str, str | None]] = []
    lines: list[tuple[str, bool]] = []  # (text, is_note)
    repeal_targets: list[str] = []
    self_refs: dict[str, int] = {}
    in_repeal_annex = False
    for rawline in raw.split("\n"):
        line_links: list[tuple[str, str | None]] = []
        line = _strip_links(rawline, line_links)
        links.extend(line_links)
        if PREV_EDITION_RE.match(line):
            # "Oldingi tahrirga qarang" always links to the document itself
            for url, _a in line_links:
                m = DOC_LINK_RE.search(url)
                if m:
                    self_refs[m.group(1)] = self_refs.get(m.group(1), 0) + 1
            continue
        if not line or line.startswith("PDF-faylidagi rasmiy manba"):
            continue
        is_note = bool(NOTE_RE.match(line)) or line.startswith("(")
        # ---- repeal detection: whole-document repeals only ------------------
        up = line.upper()
        if not is_note and "KUCHINI YOʻQOTGAN" in up and "ROʻYXAT" in up and len(line) < 400 and not line_links:
            in_repeal_annex = True          # heading of an annex listing repealed acts
            lines.append((line, is_note))
            continue
        if in_repeal_annex and not line_links and (up == line and len(line) > 15 or up.startswith("ILOVA")):
            in_repeal_annex = False         # next annex / heading
        annex_item = in_repeal_annex and bool(re.match(r"^\d+[.)]", line))
        if line_links and not is_note and (REPEAL_PHRASE_RE.search(line) or annex_item):
            for url, anchor, tail in _link_tails(rawline):
                m = DOC_LINK_RE.search(url)
                if not m or anchor:
                    continue
                if re.search(r"muvofiq|asosan|ijrosi|maqsadida|nazarda tutilgan|bilan tasdiqlangan", tail[:120], re.IGNORECASE):
                    continue
                if PARTIAL_RE.search(tail[:160]) and not annex_item:
                    continue
                repeal_targets.append(m.group(1))
        lines.append((line, is_note))

    # ---- self id (kodekslar have no id in the file name) --------------------
    if lex_num is None and self_refs:
        lex_num = max(self_refs.items(), key=lambda kv: kv[1])[0]
    lex_num = lex_num or "unknown"

    # ---- header / title ------------------------------------------------------
    header: list[str] = []
    for text, _ in lines[:12]:
        if text.startswith("[") or re.match(r"(Mazkur|Ushbu)\s", text) or text.upper() in ("MUQADDIMA", "PREAMBULA") or ARTICLE_RE.match(text) or POINT_RE.match(text) or HEADING_RE.match(text):
            break
        if text.upper() != text and len(header) >= 2:
            break
        header.append(text)
        if len(" ".join(header)) > 600:
            break
    header_txt = " ".join(header)
    doc_type = _doc_type(header)
    title = _sentence_case(header_txt)[:400] or f"lex.uz hujjati {lex_num}"

    # ---- dates ---------------------------------------------------------------
    tail = "\n".join(t for t, _ in lines[-25:])
    adopted = adopted_hint
    number = None
    if not adopted:
        tm = re.search(r"Toshkent\s+sh(?:ahri|\.)?[,\s]*\n?(.*)", tail, re.IGNORECASE | re.DOTALL)
        seg = tm.group(1)[:200] if tm else tail[-600:]
        ds = find_dates(seg)
        if ds:
            adopted = ds[0] if tm else min(ds)
        nm = re.search(r"\n?\s*((?:OʻRQ|ZRU|PF|PQ|PҚ|O‘RQ|QHT)?[-–]?\s*\d+[\w/–-]*)-son", seg)
        if nm:
            number = nm.group(1).strip()
    if not adopted:
        ds = find_dates(header_txt + " " + " ".join(t for t, _ in lines[len(header):len(header) + 3]))
        adopted = ds[0] if ds else None
    note_dates = [d for t, is_note in lines if is_note for d in find_dates(t)]
    edition = max(note_dates) if note_dates else adopted
    if adopted and edition and edition < adopted:
        edition = adopted

    # ---- banner "Mazkur Kodeks … oʻz kuchini yoʻqotgan" in the first lines ----------
    header_repealed = None
    head_txt = " ".join(t for t, _ in lines[:8])[:2500]
    hm = HEADER_REPEAL_RE.search(head_txt)
    if hm and not PARTIAL_RE.search(hm.group(3)) and not re.search(r"\b(bandi|moddasi|qismi|xatboshisi|ilovasi|boʻlimi|bobi)\b", hm.group(3)):
        ds = find_dates(hm.group(3))
        header_repealed = max(ds) if ds else "?"

    doc = Document(
        doc_id=f"lex:{lex_num}", lex_num=lex_num, url=f"https://lex.uz/docs/{lex_num}", title=title,
        doc_type=doc_type, adopted=adopted, edition_date=edition, number=number, source=source,
        links=sorted({m.group(1) for u, _ in links if (m := DOC_LINK_RE.search(u))} - {lex_num}),
        repeals=sorted(set(repeal_targets) - {lex_num}), self_refs=self_refs,
        n_chars=sum(len(t) for t, _ in lines),
        header_repealed=header_repealed,
    )
    doc.chunks = _chunk(doc, lines[len(header):])
    return doc


_SUP_BACK = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def _ascii_art(num: str) -> str:
    """'546⁵' -> '546.5' for chunk ids."""
    m = re.match(r"^(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)$", num)
    return f"{m.group(1)}.{m.group(2).translate(_SUP_BACK)}" if m else num


def _chunk(doc: Document, lines: list[tuple[str, bool]]) -> list[Chunk]:
    has_articles = sum(1 for t, _ in lines if ARTICLE_RE.match(t)) >= 2
    units: list[tuple[str | None, str, str, list[str], str | None]] = []  # (label, heading, path, parts, edition)
    path_parts: dict[int, str] = {}
    cur_label: str | None = None
    cur_heading = ""
    cur_parts: list[str] = []
    section = 1
    last_point = 0
    cur_path = ""
    cur_dates: list[str] = []
    last_art = 0
    sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

    def flush():
        if cur_parts or cur_heading:
            units.append((cur_label, cur_heading, cur_path, list(cur_parts), max(cur_dates) if cur_dates else None))
        cur_dates.clear()

    for text, is_note in lines:
        if is_note:
            cur_dates.extend(find_dates(text))  # "(…-modda … 2021-yil 21-apreldagi … tahririda)"
            continue
        am = ARTICLE_RE.match(text) if has_articles else None
        pm = None if has_articles else POINT_RE.match(text)
        if am and len(text) < 400:
            flush()
            num = am.group(1).replace("–", "-")
            if num.isdigit():
                n, base = int(num), str(last_art)
                # "546⁵-modda" is flattened to "5465" by the .doc converter: restore the superscript
                if last_art and n > last_art + 50 and num.startswith(base) and len(num) > len(base):
                    num = base + num[len(base):].translate(sup)
                else:
                    last_art = n
            cur_label = f"{num}-modda"
            cur_heading, cur_parts, cur_path = text, [], " > ".join(path_parts[k] for k in sorted(path_parts))
            continue
        if pm and len(pm.group(2)) > 3:
            n = int(pm.group(1))
            if n <= last_point and n == 1:
                section += 1  # numbering restarted -> new annex / regulation
            last_point = n
            flush()
            cur_label = f"{n}-band" if section == 1 else f"{n}-band ({section}-boʻlim)"
            cur_heading, cur_parts, cur_path = "", [text], " > ".join(path_parts[k] for k in sorted(path_parts))
            continue
        if HEADING_RE.match(text) and len(text) < 300:
            level = 0 if re.search(r"boʻlim|BOʻLIM|QISM", text) else (2 if "§" in text else 1)
            path_parts = {k: v for k, v in path_parts.items() if k < level}
            path_parts[level] = text
            continue
        cur_parts.append(text)
    flush()

    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for label, heading, path, parts, art_edition in units:
        if not parts and not heading:
            continue
        base = "preamble" if label is None else ("art-" + _ascii_art(label.split("-modda")[0]) if label.endswith("modda") else "band-" + label.split("-band")[0] + (f"s{label.split('(')[1].split('-')[0]}" if "(" in label else ""))
        # pack parts into chunks
        groups: list[tuple[int, int, list[str]]] = []
        start, buf, size = 1, [], len(heading)
        for i, p in enumerate(parts or [""], 1):
            if buf and size + len(p) > MAX_CHUNK_CHARS:
                groups.append((start, i - 1, buf))
                start, buf, size = i, [], len(heading)
            buf.append(p)
            size += len(p) + 1
            # very long single paragraph: hard split
            while size > MAX_CHUNK_CHARS * 2 and len(buf) == 1:
                head, rest = buf[0][:MAX_CHUNK_CHARS], buf[0][MAX_CHUNK_CHARS:]
                groups.append((start, i, [head]))
                buf, size = [rest], len(rest)
        if buf:
            groups.append((start, len(parts or [""]), buf))
        for a, b, buf in groups:
            cid = f"{doc.doc_id}#{base}#p{a}"
            if cid in seen:
                seen[cid] += 1
                cid = f"{cid}-{seen[cid]}"
            else:
                seen[cid] = 1
            body = "\n".join(x for x in buf if x)
            text = (heading + "\n" + body).strip() if heading else body
            part = None
            if label and label.endswith("modda"):
                part = f"{a}-qism" if a == b else f"{a}–{b}-qism"
            chunks.append(Chunk(cid, doc.doc_id, label, part, heading, path, text, len(chunks), art_edition))
    return chunks
