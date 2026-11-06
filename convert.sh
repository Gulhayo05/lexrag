#!/bin/bash
# Convert lex.uz .doc exports to UTF-8 text with macOS textutil (parallel).
SRC="${1:-/Users/gulkhayo/Downloads/data}"
DST="$(cd "$(dirname "$0")" && pwd)/data/txt"
for sub in kodekslar docs acts; do
  mkdir -p "$DST/$sub"
  find "$SRC/$sub" -name '*.doc' -print0 | xargs -0 -P 10 -n 20 sh -c '
    dst="$0"
    for f in "$@"; do
      b=$(basename "$f" .doc); out="$dst/$b.txt"
      [ -s "$out" ] || textutil -convert txt -encoding UTF-8 -output "$out" "$f" 2>/dev/null || echo "FAIL $f" >&2
    done' "$DST/$sub"
done
echo DONE
