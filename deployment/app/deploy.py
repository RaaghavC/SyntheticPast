#!/usr/bin/env python3
"""
Redeploy SyntheticPast to its Hugging Face Space.

WHAT THIS DOES
  - Creates (or reuses) the Docker Space  RaaghavC/synthetic-past
  - Sets the 3 API keys as encrypted Space secrets
  - Uploads every file next to this script (the app bundle)
  The Space then rebuilds automatically and goes live at:
      https://raaghavc-synthetic-past.hf.space

PREREQUISITES
  pip install huggingface_hub python-dotenv

HOW TO RUN  (from inside this  deployment/app/  folder)
  1) Get a Hugging Face WRITE token:  https://huggingface.co/settings/tokens
  2) export HF_TOKEN=hf_xxxxxxxx
  3) Provide the 3 API keys ONE of these ways:
        a) env vars:
             export OPENAI_API_KEY=...
             export STABLE_DIFFUSION_API_KEY=...
             export IMGUR_API_KEY=...
        b) point at a .env file that contains them:
             python deploy.py /path/to/.env        (e.g. the project root .env)
  4) python deploy.py
"""
import os
import sys
from huggingface_hub import HfApi, whoami

# Owner/space. Change the owner ("RaaghavC") if deploying under a different HF account.
REPO = "RaaghavC/synthetic-past"
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def load_keys():
    keys = {}
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        from dotenv import dotenv_values
        keys.update({k: v for k, v in dotenv_values(sys.argv[1]).items()})
    for k in ("OPENAI_API_KEY", "STABLE_DIFFUSION_API_KEY", "IMGUR_API_KEY"):
        if os.environ.get(k):
            keys[k] = os.environ[k]
    return keys


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: set HF_TOKEN to a Hugging Face WRITE token first "
                 "(https://huggingface.co/settings/tokens).")
    api = HfApi(token=token)
    user = whoami(token=token)["name"]
    repo = REPO if "/" in REPO else f"{user}/{REPO}"
    print("Deploying to:", repo)

    api.create_repo(repo_id=repo, repo_type="space", space_sdk="docker", exist_ok=True)

    keys = load_keys()
    for k in ("OPENAI_API_KEY", "STABLE_DIFFUSION_API_KEY", "IMGUR_API_KEY"):
        v = (keys.get(k) or "").strip()
        if v:
            api.add_space_secret(repo_id=repo, key=k, value=v)
            print(f"  secret set: {k} ({len(v)} chars)")
        else:
            print(f"  WARNING: no value for {k}; set it manually in Space > Settings > Secrets")

    api.upload_folder(
        folder_path=APP_DIR,
        repo_id=repo,
        repo_type="space",
        ignore_patterns=["deploy.py", ".env", "__pycache__*", "*.pyc", "pdf_vectorstore*", "static*", "*.DS_Store"],
        commit_message="Redeploy SyntheticPast",
    )
    print("DONE ->", f"https://{user.lower().replace('_', '-')}-{repo.split('/')[-1]}.hf.space")


if __name__ == "__main__":
    main()
