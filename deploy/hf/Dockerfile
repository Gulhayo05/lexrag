# Lex RAG — Hugging Face Space (Docker SDK). Built by HF from the Space repository.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# code, chat UI, prebuilt lex.uz index (data/lex.sqlite) and evaluation summaries
COPY --chown=user . .

USER user
# secrets (DEEPSEEK_API_KEY, GEMINI_API_KEY, LEXRAG_API_KEYS) come from the Space settings, never from the image
ENV PYTHONUNBUFFERED=1 \
    LEXRAG_RESULTS_DB=/tmp/batches.sqlite \
    LEXRAG_SOURCE_DIR=/app/data

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "lexrag.api:app", "--host", "0.0.0.0", "--port", "7860"]
