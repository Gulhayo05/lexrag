---
title: Lex RAG
emoji: ⚖️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Legal Q&A over lex.uz with cited articles (RAG API)
---

# Lex RAG — lex.uz huquqiy savol-javob

RAG API and chat assistant over Uzbekistan's legislation (lex.uz). Each answer cites the act and article, with a verbatim quote.

* Chat: `/`
* API docs: `/docs`
* Spec endpoints: `/v1/rag/batch`, `/v1/rag/query`, `/v1/meta`, `/v1/health`

Access needs the API key that the owner shares (`Authorization: Bearer <key>`).
