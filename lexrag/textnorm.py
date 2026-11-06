"""Text normalisation for Uzbek (Latin / Cyrillic) and Russian legal text.

Two forms are used throughout the system:

* **display form** – the original text, only with apostrophe variants unified to
  the official Uzbek letters (``oʻ``, ``gʻ``, ``ʼ``) so that ``o'`` / ``g'`` never
  get mangled in answers or quotes.
* **fold form** – used only for search: lower case, Cyrillic transliterated to
  Latin, all apostrophes removed (``taʼtil`` -> ``tatil``, ``oʻn`` -> ``on``).
"""
from __future__ import annotations

import re
import unicodedata

# Every apostrophe-like character seen in Uzbek texts.
_APOS = "'`ʻʼ‘’ʹʽ´′"
_APOS_RE = re.compile(f"[{re.escape(_APOS)}]")

# o' / g' (any apostrophe variant) -> official modifier letter turned comma (U+02BB)
_OG_RE = re.compile(r"([oOgG])[" + re.escape(_APOS) + r"]")
# remaining apostrophes (tutuq belgisi) -> U+02BC
_TUTUQ_RE = re.compile(r"(?<=\w)(?<![oOgG])[" + re.escape(_APOS.replace("\u02bb", "")) + r"](?=\w)")


def unify_apostrophes(text: str) -> str:
    """Return text with ``o'``/``g'`` -> ``oʻ``/``gʻ`` and other in-word apostrophes -> ``ʼ``."""
    text = unicodedata.normalize("NFC", text)
    text = _OG_RE.sub(lambda m: m.group(1) + "ʻ", text)
    return _TUTUQ_RE.sub("ʼ", text)


# --------------------------------------------------------------------------- #
# Cyrillic -> Latin (Uzbek official 1995 alphabet)
# --------------------------------------------------------------------------- #
_CYR2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ж": "j", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x",
    "ц": "s", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "ʼ", "ы": "i", "ь": "",
    "э": "e", "ю": "yu", "я": "ya", "ё": "yo", "ў": "oʻ", "қ": "q", "ғ": "gʻ",
    "ҳ": "h",
}
_CYR_VOWELS = set("аеёиоуэюяўы")


def cyr_to_lat(text: str) -> str:
    out: list[str] = []
    prev = ""
    for ch in text:
        low = ch.lower()
        if low == "е":
            # "е" is "ye" at word start and after vowels / ъ / ь
            lat = "ye" if (not prev or not prev.isalpha() or prev.lower() in _CYR_VOWELS or prev.lower() in "ъь") else "e"
        elif low in _CYR2LAT:
            lat = _CYR2LAT[low]
        else:
            out.append(ch)
            prev = ch
            continue
        if ch != low and lat:
            lat = lat[0].upper() + lat[1:]
        out.append(lat)
        prev = ch
    return "".join(out)


_CYR_RE = re.compile(r"[Ѐ-ӿ]")
_UZ_CYR_LETTERS = re.compile(r"[ўқғҳЎҚҒҲ]")
_RU_MARKERS = re.compile(
    r"(?:ы|щ|\b(?:и|в|не|что|как|какой|какая|какие|каков|ли|по|для|при|это|или|сколько|кодекс|закон|срок|размер)\b)",
    re.IGNORECASE,
)
_UZ_CYR_WORDS = re.compile(r"\b(?:ва|учун|билан|қандай|канча|қанча|нима|ким|бўйича|муддати|хисобланади)\b", re.IGNORECASE)


def detect_language(text: str) -> str:
    """Return ``uz-Latn``, ``uz-Cyrl`` or ``ru``."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "uz-Latn"
    cyr = sum(1 for c in letters if _CYR_RE.match(c))
    if cyr / len(letters) < 0.3:
        return "uz-Latn"
    if _UZ_CYR_LETTERS.search(text) or _UZ_CYR_WORDS.search(text):
        return "uz-Cyrl"
    if _RU_MARKERS.search(text):
        return "ru"
    return "ru"


# --------------------------------------------------------------------------- #
# Folding + tokenisation for lexical search
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def fold(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    if _CYR_RE.search(text):
        text = cyr_to_lat(text)
    text = _APOS_RE.sub("", text.replace("ʼ", "").replace("ʻ", ""))
    text = text.lower()
    # strip diacritics that may remain (e.g. imported names)
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(fold(text))


# Light Uzbek suffix stripper (agglutinative morphology).  Longest suffix first.
_SUFFIXES = sorted(
    {
        "larining", "laridagi", "laridan", "larida", "lariga", "larini", "larning", "lardagi",
        "lardan", "larda", "larga", "larni", "lari", "lar",
        "ining", "idagi", "ining", "sining", "sidagi", "sidan", "sida", "siga", "sini", "si",
        "ning", "dagi", "dan", "da", "ga", "ka", "qa", "ni", "ning", "ini", "iga", "ida", "idan",
        "ligi", "lik", "lig", "chi", "lar", "gan", "kan", "qan", "moqda", "adi", "ydi", "mi",
        "dir", "di", "gi", "i", "nig",
    },
    key=len,
    reverse=True,
)
_STOP = {
    # uz
    "va", "bilan", "uchun", "boyicha", "qanday", "qancha", "nima", "necha", "kim", "qaysi",
    "bu", "shu", "u", "ham", "yoki", "agar", "esa", "deb", "edi", "emas", "bolsa", "bolgan",
    "boladi", "mumkinmi", "mumkin", "kerak", "kerakmi", "qilinadi", "etiladi", "nechta",
    "ozbekiston", "respublikasi", "respublikasining", "togrisida", "togrisidagi", "haqida",
    "belgilangan", "qilib", "etib", "orqali", "hisoblanadi", "hamda", "ularning", "uning", "men",
    "biz", "siz", "mi", "chi", "ekan", "qachon", "nega", "nimaga", "kimga", "qanaqa",
    # translit ru
    "i", "v", "ne", "chto", "kak", "kakoy", "kakaya", "kakie", "li", "po", "dlya", "pri", "eto",
    "ili", "skolko", "na", "s", "o", "ob", "k", "ot", "do", "za", "iz", "u", "ego", "ee",
}


# derivational suffixes that may leave a 3-letter root (yillik / har yilgi -> yil, oylik -> oy+)
_SHORT_ROOT_SUFFIXES = ("lik", "ligi", "lig", "gi", "dagi")


def stem(tok: str) -> str:
    if tok.isdigit() or len(tok) <= 4:
        return tok
    for _ in range(3):
        for suf in _SUFFIXES:
            min_root = 3 if suf in _SHORT_ROOT_SUFFIXES else 4
            if tok.endswith(suf) and len(tok) - len(suf) >= min_root:
                tok = tok[: -len(suf)]
                break
        else:
            break
    return tok


def query_terms(text: str) -> list[str]:
    """Distinct stemmed content terms of a query (order preserved)."""
    seen: dict[str, None] = {}
    for t in tokens(text):
        if t in _STOP or (len(t) < 2 and not t.isdigit()):
            continue
        seen.setdefault(stem(t), None)
    return list(seen)
