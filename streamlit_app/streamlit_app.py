"""Streamlit front end for the lex.uz RAG system.

Two modes, chosen by Streamlit secrets (Community Cloud: App settings -> Secrets):

* API mode: set LEXRAG_API_URL (+ LEXRAG_API_KEY). The app calls a running lexrag API.
* Built-in mode (no LEXRAG_API_URL): the app downloads the prebuilt index from a public
  Hugging Face dataset (LEXRAG_INDEX_REPO, default gulkhayo5775/lexrag-index) and runs the
  same pipeline in-process. Needs an LLM key, e.g.:

      DEEPSEEK_API_KEY = "..."
      LEXRAG_LLM_PROVIDER = "deepseek"

Run locally:
    streamlit run streamlit_app/streamlit_app.py                       # built-in, uses data/lex.sqlite
    LEXRAG_API_URL=http://127.0.0.1:8000 LEXRAG_API_KEY=... streamlit run streamlit_app/streamlit_app.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import httpx
import streamlit as st

st.set_page_config(page_title="Lex RAG — lex.uz", page_icon="⚖️", layout="wide")
ROOT = Path(__file__).resolve().parent.parent


def _secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # noqa: BLE001 - no secrets file locally
        pass
    return os.environ.get(name, default)


API_URL = _secret("LEXRAG_API_URL").rstrip("/")
API_KEY = _secret("LEXRAG_API_KEY")
BUILT_IN = not API_URL


# --------------------------------------------------------------------------- #
# built-in mode: index from a Hugging Face dataset, pipeline in this process
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Qonunchilik indeksi yuklanmoqda (birinchi ishga tushishda 1–3 daqiqa)…")
def _engine():
    try:  # root-level Streamlit secrets -> environment, before lexrag reads its settings
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)):
                os.environ.setdefault(k, str(v))
    except Exception:  # noqa: BLE001
        pass
    local = ROOT / "data" / "lex.sqlite"
    if not os.environ.get("LEXRAG_DB_PATH"):
        if local.exists():
            os.environ["LEXRAG_DB_PATH"] = str(local)
        else:  # small host: modest SQLite memory, results/cache DB in /tmp
            from huggingface_hub import hf_hub_download
            os.environ["LEXRAG_DB_PATH"] = hf_hub_download(
                repo_id=os.environ.get("LEXRAG_INDEX_REPO", "gulkhayo5775/lexrag-index"),
                filename="lex.sqlite", repo_type="dataset")
            os.environ.setdefault("LEXRAG_SQLITE_MMAP_MB", "0")
            os.environ.setdefault("LEXRAG_SQLITE_CACHE_MB", "16")
            os.environ.setdefault("LEXRAG_RESULTS_DB", "/tmp/lexrag-results.sqlite")
    sys.path.insert(0, str(ROOT))
    from lexrag import llm, pipeline
    from lexrag.config import settings
    return pipeline, llm, settings


def _ask_built_in(question: str, options: dict, history: list[dict]) -> dict:
    pipeline, llm, settings = _engine()
    item = {"id": "q", "question": question, "language": None, "history": history[-5:]}
    t0 = time.perf_counter()

    async def run():
        return await asyncio.wait_for(pipeline.answer_item(item, options), timeout=settings.question_timeout_s)
    try:
        r = asyncio.run(run())
    except asyncio.TimeoutError:
        r = pipeline.error_result(item, "LLM_TIMEOUT", "javob berish vaqti tugadi", True, status="timeout")
    except llm.LLMError as e:
        r = pipeline.error_result(item, e.code, e.message, e.retryable)
    return {"system": pipeline.system_info(), "timing": {"total_ms": int((time.perf_counter() - t0) * 1000)}, "result": r}


LANG_LABELS = {"auto": "Savol tilida (auto)", "uz-Latn": "Oʻzbek (lotin)", "uz-Cyrl": "Ўзбек (кирилл)", "ru": "Русский"}
EXAMPLES = [
    "Yillik asosiy mehnat taʼtilining eng kam muddati qancha?",
    "Nikoh yoshi necha yosh?",
    "Какой срок исковой давности установлен Гражданским кодексом?",
]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_meta(url: str) -> dict | None:
    try:
        r = httpx.get(f"{url}/v1/meta", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        return None


def ask(question: str, options: dict, history: list[dict]) -> dict:
    if BUILT_IN:
        return _ask_built_in(question, options, history)
    body = {"question": question, "options": options, "history": history[-5:]}
    r = httpx.post(f"{API_URL}/v1/rag/query", json=body, timeout=180,
                   headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json; charset=utf-8"})
    if r.status_code == 429:
        raise RuntimeError(f"Soʻrovlar juda koʻp. {r.headers.get('Retry-After', '2')} soniyadan keyin qayta urinib koʻring.")
    if r.status_code in (401, 403):
        raise RuntimeError("API kaliti notoʻgʻri yoki berilmagan (LEXRAG_API_KEY).")
    if r.status_code >= 400:
        raise RuntimeError(f"API xatosi {r.status_code}: {r.text[:300]}")
    return r.json()


def render_result(res: dict, timing: dict | None = None) -> None:
    r = res["result"] if "result" in res else res
    if r.get("status") in ("error", "timeout"):
        err = r.get("error") or {}
        st.error(f"{err.get('code', 'ERROR')}: {err.get('message', '')}")
        return
    if r.get("no_answer"):
        st.warning(r.get("answer") or "Javob topilmadi.")
    else:
        st.markdown(r.get("answer") or "")
    meta = []
    if r.get("confidence") is not None:
        meta.append(f"ishonch: {r['confidence']:.2f}")
    if timing and timing.get("total_ms") is not None:
        meta.append(f"{timing['total_ms'] / 1000:.1f} s")
    if meta:
        st.caption(" · ".join(meta))
    cites = r.get("citations") or []
    if cites:
        st.markdown("**Manbalar**")
        for c in cites:
            status = "" if c.get("doc_status") == "amalda" else " · ⚠️ kuchini yoʻqotgan"
            head = f"[{c['ref']}] [{c['doc_title']}]({c['url']}) — {c.get('article') or ''}" + (f", {c['part']}" if c.get("part") else "")
            st.markdown(head)
            st.caption(f"tahrir: {c.get('edition_date') or '?'}{status}")
            st.markdown(f"> {c.get('quote', '')}")
    ctxs = r.get("contexts") or []
    if ctxs:
        with st.expander(f"Topilgan kontekstlar ({len(ctxs)})"):
            for c in ctxs:
                used = " ✅" if c.get("used_in_answer") else ""
                rr = c.get("rerank_score")
                st.markdown(f"**{c['rank']}. [{c['doc_title']}]({c['url']}) — {c.get('article') or ''}**{used}")
                st.caption(f"retrieval {c.get('retrieval_score')} · rerank {rr if rr is not None else '—'} · {c['chunk_id']}")
                if c.get("text"):
                    st.text(c["text"])


# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚖️ Lex RAG")
    st.caption("lex.uz qonunchiligi boʻyicha savol-javob. Har bir javob modda va aynan iqtibos bilan.")
    answer_language = st.selectbox("Javob tili", list(LANG_LABELS), format_func=LANG_LABELS.get)
    top_k = st.slider("Kontekstlar soni (top_k)", 1, 10, 5)
    only_relevant = st.toggle("Faqat tegishli kontekstlar", value=True,
                              help="options.min_rerank_score = 0.5: reranker tegishli deb topganlarini qaytaradi")
    as_of = st.date_input("Sana boʻyicha (as_of_date)", value=None)
    if st.button("Suhbatni tozalash"):
        st.session_state.messages = []
    if BUILT_IN:
        try:
            meta = _engine()[0].system_info()
        except Exception as e:  # noqa: BLE001
            meta = None
            st.error(f"Indeksni yuklab boʻlmadi: {e}")
    else:
        meta = fetch_meta(API_URL)
    st.divider()
    if meta:
        st.caption(f"Model: {meta.get('llm_model')}  \nIndeks: {meta.get('index_version')}  \n"
                   f"lex.uz holati: {meta.get('lex_snapshot_date')}  \nPipeline: {meta.get('pipeline_version')}")
    elif not BUILT_IN:
        st.error(f"API ga ulanib boʻlmadi: {API_URL}")
    if not BUILT_IN and not API_KEY:
        st.warning("LEXRAG_API_KEY sozlanmagan (Secrets).")

st.title("lex.uz huquqiy yordamchi")
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.caption("Misollar:")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            st.session_state.pending = ex

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_result(m["content"], m.get("timing"))

question = st.chat_input("Savolingizni yozing (oʻzbek lotin / kirill yoki rus tilida)") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    options = {"top_k": top_k, "return_contexts": True, "answer_language": answer_language, "temperature": 0, "seed": 42}
    if only_relevant:
        options["min_rerank_score"] = 0.5
    if as_of:
        options["as_of_date"] = as_of.isoformat()
    history = []
    msgs = st.session_state.messages[:-1]
    for u, a in zip(msgs[::2], msgs[1::2]):
        if u["role"] == "user" and a["role"] == "assistant":
            history.append({"question": u["content"], "answer": (a["content"].get("result") or {}).get("answer")})
    with st.chat_message("assistant"):
        with st.spinner("Qonun hujjatlaridan qidirilmoqda…"):
            try:
                res = ask(question, options, history)
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
                st.session_state.messages.pop()
            else:
                render_result(res, res.get("timing"))
                st.session_state.messages.append({"role": "assistant", "content": res, "timing": res.get("timing")})
