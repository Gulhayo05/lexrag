"""Deploy Lex RAG to a Hugging Face Space (Docker SDK).

    python3 deploy/deploy_hf.py --dry-run        # show what would be uploaded, upload nothing
    python3 deploy/deploy_hf.py                  # create/update the PRIVATE Space and upload
    python3 deploy/deploy_hf.py --public         # make the Space public instead

Reads from the project's .env (never uploaded):
    HF_TOKEN            Hugging Face access token with write permission (required)
    DEEPSEEK_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, LEXRAG_API_KEYS, LEXRAG_LLM_PROVIDER
The LLM keys and the API key are stored as Space *secrets* (encrypted, not visible in the
repository); the uploaded files are only: Dockerfile, README (Space card), requirements.txt,
lexrag/*.py, the chat page, data/lex.sqlite (the prebuilt index) and evaluation summaries.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRET_KEYS = ("DEEPSEEK_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "LEXRAG_API_KEYS")
VARIABLE_KEYS = ("LEXRAG_LLM_PROVIDER", "LEXRAG_DEEPSEEK_MODEL", "LEXRAG_DEEPSEEK_ANSWER_EFFORT",
                 "LEXRAG_GEMINI_MODEL", "LEXRAG_GEMINI_REWRITE_MODEL")


def read_env() -> dict[str, str]:
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            v = v.split("#", 1)[0].strip().strip('"').strip("'")
            if sep and k.strip() and v:
                env[k.strip()] = v
    return env


def files_to_upload() -> list[tuple[Path, str]]:
    out = [(ROOT / "deploy/hf/Dockerfile", "Dockerfile"), (ROOT / "deploy/hf/README.md", "README.md"),
           (ROOT / "requirements.txt", "requirements.txt"), (ROOT / "data/lex.sqlite", "data/lex.sqlite"),
           (ROOT / "lexrag/static/chat.html", "lexrag/static/chat.html")]
    out += [(p, f"lexrag/{p.name}") for p in sorted((ROOT / "lexrag").glob("*.py"))]
    for d in sorted((ROOT / "eval/runs").glob("*/")):
        for name in ("metrics.json", "meta.json"):
            if (d / name).exists() and (d / "metrics.json").exists():
                out.append((d / name, f"eval/runs/{d.name}/{name}"))
    for src, _ in out:
        if not src.exists():
            sys.exit(f"missing file: {src}")
        if src.name == ".env":
            sys.exit("refusing to upload .env")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="lexrag", help="Space name (repo id will be <your-user>/<name>)")
    ap.add_argument("--public", action="store_true", help="make the Space public (default: private)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    env = read_env()
    files = files_to_upload()
    total = sum(p.stat().st_size for p, _ in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB:")
    for p, dest in files:
        if p.stat().st_size > 5_000_000 or dest in ("Dockerfile", "README.md"):
            print(f"  {dest:40} {p.stat().st_size / 1e6:9.1f} MB")
    print(f"  … {sum(1 for _, d in files if d.startswith('lexrag/'))} code files, "
          f"{sum(1 for _, d in files if d.startswith('eval/'))} evaluation summaries")
    secrets = {k: env[k] for k in SECRET_KEYS if env.get(k)}
    variables = {k: env[k] for k in VARIABLE_KEYS if env.get(k)}
    print("secrets to set (values hidden):", ", ".join(secrets) or "none")
    print("variables to set:", variables or "none")
    if "LEXRAG_API_KEYS" not in secrets:
        sys.exit("LEXRAG_API_KEYS missing in .env: the deployed API would only accept the default test key")
    if not (secrets.keys() & {"DEEPSEEK_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"}):
        print("WARNING: no LLM key -> the Space will answer in extractive fallback mode")
    if a.dry_run:
        print("dry run: nothing uploaded")
        return

    token = env.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN missing in .env (create a token with 'write' permission at https://huggingface.co/settings/tokens)")
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{a.space}"
    print(f"\ncreating / updating Space {repo_id} ({'public' if a.public else 'private'}) …")
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=not a.public, exist_ok=True)
    for k, v in secrets.items():
        api.add_space_secret(repo_id, k, v)
    for k, v in variables.items():
        api.add_space_variable(repo_id, k, v)
    print("secrets and variables set; uploading files (the 1.6 GB index takes a few minutes) …")
    ops = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(src)) for src, dest in files]
    api.create_commit(repo_id, repo_type="space", operations=ops, commit_message="Deploy Lex RAG")

    url = f"https://{user.lower().replace('_', '-')}-{a.space.lower()}.hf.space"
    print(f"uploaded. Space page: https://huggingface.co/spaces/{repo_id}\nApp URL: {url}\nwaiting for the build …")
    last = None
    for _ in range(120):  # up to ~20 minutes
        stage = api.get_space_runtime(repo_id).stage
        if stage != last:
            print("  stage:", stage)
            last = stage
        if stage in ("RUNNING", "BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"):
            break
        time.sleep(10)
    print(f"\nfinal stage: {last}\nApp URL: {url}")
    if last != "RUNNING":
        print(f"see the build logs at https://huggingface.co/spaces/{repo_id}?logs=build")


if __name__ == "__main__":
    main()
