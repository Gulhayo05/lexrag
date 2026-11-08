"""Upload the prebuilt index (data/lex.sqlite) to the public Hugging Face dataset that the
Streamlit app downloads it from.  Resumable: re-run it after a dropped connection.

    python3 deploy/upload_index.py

Reads HF_TOKEN from .env (write permission on the dataset owner's account).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "gulkhayo5775/lexrag-index"


def main() -> None:
    token = next((line.split("=", 1)[1].split("#")[0].strip() for line in (ROOT / ".env").read_text().splitlines()
                  if line.startswith("HF_TOKEN=")), None)
    if not token:
        sys.exit("HF_TOKEN missing in .env")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)
    for attempt in range(1, 6):
        try:
            t = time.time()
            print(f"uploading data/lex.sqlite to {REPO} (attempt {attempt}) …", flush=True)
            api.upload_file(path_or_fileobj=str(ROOT / "data/lex.sqlite"), path_in_repo="lex.sqlite", repo_id=REPO,
                            repo_type="dataset", commit_message="lex.uz search index (lex-idx-2026-11-06)")
            print(f"DONE in {int(time.time() - t)} s: https://huggingface.co/datasets/{REPO}", flush=True)
            return
        except Exception as e:  # noqa: BLE001 - network drops on slow connections
            print(f"failed: {e!s:.200}; retrying in 20 s", flush=True)
            time.sleep(20)
    sys.exit("upload failed 5 times; check the connection and run again")


if __name__ == "__main__":
    main()
