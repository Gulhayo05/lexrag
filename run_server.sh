#!/bin/bash
# Start the lex.uz RAG API for testers on this machine's network.
#   ./run_server.sh            -> http://<this-mac-ip>:8000
# The test API key is LEXRAG_API_KEYS in .env (send it to testers privately).
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)
echo "Base URL for testers:  http://$IP:$PORT     (local: http://127.0.0.1:$PORT)"
echo "Health:                http://$IP:$PORT/v1/health"
echo "Test key:              LEXRAG_API_KEYS in $(pwd)/.env"
exec python3 -m uvicorn lexrag.api:app --host 0.0.0.0 --port "$PORT" --workers 1 --log-level warning
