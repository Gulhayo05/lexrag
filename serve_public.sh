#!/bin/bash
# Serve the RAG API publicly from this Mac through a free Cloudflare quick tunnel (no account).
#   ./serve_public.sh          -> prints "PUBLIC API URL: https://<random>.trycloudflare.com"
# Keep this window open and the Mac on: closing it stops the API and the tunnel.
# The URL changes every time the script starts; it is also written to .public_url.
# Needs: cloudflared (brew install cloudflared). API keys: LEXRAG_API_KEYS in .env.
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
health() { curl -s -o /dev/null -m 3 -w "%{http_code}" "http://127.0.0.1:$PORT/v1/health" | grep -q 200; }

if ! health; then
  echo "starting the API on port $PORT …"
  python3 -m uvicorn lexrag.api:app --host 127.0.0.1 --port "$PORT" --workers 1 --log-level warning &
  SERVER=$!
  for _ in $(seq 1 60); do health && break; sleep 1; done
  health || { echo "API did not start"; exit 1; }
fi
cleanup() { [ -n "$SERVER" ] && kill "$SERVER" 2>/dev/null; rm -f .public_url; }
trap cleanup EXIT

caffeinate -dimsu -w $$ &   # keep the Mac awake while this script runs
echo "API is up on http://127.0.0.1:$PORT — opening the public tunnel (keep this window open) …"
# retry: account-less quick tunnels occasionally fail to start or drop; a new URL is printed each time
while true; do
  cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" 2>&1 | while IFS= read -r line; do
    url=$(grep -oE 'https://[a-z0-9]+(-[a-z0-9]+)+\.trycloudflare\.com' <<<"$line" | head -1)
    if [ -n "$url" ]; then
      echo "$url" > .public_url
      echo
      echo "PUBLIC API URL: $url"
      echo "API docs:       $url/docs"
      echo "Health:         $url/v1/health"
      echo
    elif grep -qiE "ERR|failed" <<<"$line"; then
      echo "cloudflared: $line"
    fi
  done
  echo "tunnel stopped; retrying in 5 s (Ctrl-C to quit) …"
  sleep 5
done
