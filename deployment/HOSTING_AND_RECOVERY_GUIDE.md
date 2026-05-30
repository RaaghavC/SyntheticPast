# SyntheticPast — Hosting & Recovery Guide

Everything you need to keep the site running, fix it when it breaks, and rebuild it from scratch.

---

## 1. What is live right now

| Thing | Value |
|---|---|
| **Public app (give this out)** | **https://raaghavc-synthetic-past.hf.space** |
| Space dashboard (manage it) | https://huggingface.co/spaces/RaaghavC/synthetic-past |
| Host | Hugging Face **Spaces**, **Docker** SDK, free CPU tier (16 GB RAM) |
| HF account | **RaaghavC** |
| App file that runs | `newvr_v2.py` (port 7860) |
| Image model | `juggernaut-xl` (photorealistic Stable Diffusion, via ModelsLab) |
| Secrets on the Space | `OPENAI_API_KEY`, `STABLE_DIFFUSION_API_KEY`, `IMGUR_API_KEY` |

Visitors need **no setup** — they open the link and use it. Every generation/chat spends **your** API credits.

---

## 2. The moving parts (what can break)

1. **Hugging Face Space** — hosts and runs the app 24/7. Free tier **sleeps after ~48h with no visitors** and **auto-wakes on the next visit** (~30s).
2. **3 API accounts** the app calls at runtime:
   - **OpenAI** (platform.openai.com) — powers the chat **and** the image judge.
   - **ModelsLab** (modelslab.com) — generates the images. *(A lapsed ModelsLab plan is what broke generation before.)*
   - **Imgur** — hosts generated images for the VR viewer (non-critical; app falls back to showing the image directly if Imgur fails).
3. **`vectorstore.zip`** — the chat's knowledge base; unzipped on boot.
4. **Pinned dependencies** — locked versions so library updates can't silently break it.

---

## 3. Keeping it healthy (do these)

- **Keep OpenAI + ModelsLab funded**, and **set spending limits** on both (high enough for normal use, capped so a flood of visitors can't drain you).
- **Don't delete** the Space, its **Secrets**, or `vectorstore.zip`.
- **Don't let the HF account get suspended.**
- If you want it to **never sleep** (instant for every visitor): Space → **Settings** → upgrade hardware (paid, optional). The free tier's auto-wake is fine for occasional judges.

---

## 4. TROUBLESHOOTING & RECOVERY

> First stop for almost any problem: open the **Logs** on the Space page (https://huggingface.co/spaces/RaaghavC/synthetic-past → **Logs** tab). It shows build + runtime errors.

### A. The link won't load / says the Space is sleeping or paused
1. Just **reload** — free Spaces auto-wake (~30s).
2. If still stuck: Space → **Settings** → **Restart this Space**.
3. If it won't start cleanly: Space → **Settings** → **Factory rebuild** (rebuilds the container from scratch).

### B. Build failed / "Runtime error" / app shows an error page
1. Open **Logs** and read the error.
2. Space → **Settings** → **Factory rebuild**.
3. If still broken, **redeploy from this bundle** (§5) — it contains a known-good copy.

### C. "Generate New Panorama" spins forever, fails, or says "Failed to generate panorama"
This is almost always **ModelsLab**, in this order:
1. **ModelsLab account/credits** — log into modelslab.com and confirm the plan is active and has credits. *(This exact thing broke it before.)* If you renew/replace the key, update the Space secret (step E).
2. **OpenAI billing** — the judge needs OpenAI; confirm credits/billing at platform.openai.com.
3. Confirm the secrets exist: Space → **Settings** → **Variables and secrets** → `STABLE_DIFFUSION_API_KEY` and `OPENAI_API_KEY` present.

### D. Chat ("Gold Rush Guide") errors
- OpenAI problem: confirm OpenAI billing/credits and that the `OPENAI_API_KEY` secret is set. After fixing, **Restart this Space**.

### E. A key changed and you need to update a secret (no redeploy needed)
1. Space → **Settings** → **Variables and secrets**.
2. Edit the secret (`OPENAI_API_KEY` / `STABLE_DIFFUSION_API_KEY` / `IMGUR_API_KEY`) → save.
3. **Restart this Space** so it picks up the new value.

### F. The app shows the panorama but the 360° viewer is blank / images missing
- Usually Imgur — non-critical; the app falls back to showing the image directly. Check the `IMGUR_API_KEY` secret if you want the VR viewer back.

### G. The Space was deleted, or you're starting over
→ Do a **full from-scratch deploy** (§5). This bundle has everything needed.

---

## 5. Full from-scratch deploy / redeploy (the important one)

Everything needed is in **`deployment/app/`**: `newvr_v2.py`, `default_panorama.jpg`, `vectorstore.zip`, `Dockerfile`, `requirements.txt`, `README.md`, and `deploy.py`.

### Option 1 — automated (recommended), using `deploy.py`
On any computer with Python:
```bash
pip install huggingface_hub python-dotenv
cd deployment/app

# 1) Get a Hugging Face WRITE token: https://huggingface.co/settings/tokens  (type: Write)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx

# 2) Provide the 3 API keys — easiest is to point at your project .env:
python deploy.py /Users/rc/Documents/SyntheticPast2025/.env
#    (or set OPENAI_API_KEY / STABLE_DIFFUSION_API_KEY / IMGUR_API_KEY as env vars and run: python deploy.py)
```
This creates/updates the Space `RaaghavC/synthetic-past`, sets the 3 secrets, uploads the app, and the Space rebuilds automatically (~2-5 min). Then open https://raaghavc-synthetic-past.hf.space.

> Deploying under a **different** HF account? Edit the `REPO = "RaaghavC/synthetic-past"` line in `deploy.py` to `"<your-username>/synthetic-past"` first. Your URL becomes `https://<your-username>-synthetic-past.hf.space`.

### Option 2 — manual via the website (no code)
1. https://huggingface.co/new-space → Owner = you, Space name = `synthetic-past`, **SDK = Docker**, **Blank**, Create.
2. In the Space → **Files** → upload all files from `deployment/app/` **except** `deploy.py` (i.e. `newvr_v2.py`, `default_panorama.jpg`, `vectorstore.zip`, `Dockerfile`, `requirements.txt`, `README.md`).
3. Space → **Settings** → **Variables and secrets** → add 3 **secrets**: `OPENAI_API_KEY`, `STABLE_DIFFUSION_API_KEY`, `IMGUR_API_KEY` (values from your `.env`).
4. The Space builds and goes live automatically.

---

## 6. Running it locally (for testing/edits)
```bash
cd deployment/app
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put your 3 keys in a .env file in THIS folder (OPENAI_API_KEY=..., STABLE_DIFFUSION_API_KEY=..., IMGUR_API_KEY=...)
streamlit run newvr_v2.py
# open http://localhost:8501
```
Notes:
- **`chromadb==0.5.0` is required** (already pinned). chromadb 1.x **cannot** open `vectorstore.zip`.
- `vectorstore.zip` auto-extracts to `pdf_vectorstore/` on first run; `default_panorama.jpg` must sit next to `newvr_v2.py`.
- First run downloads the ~420 MB embedding model (cached afterward).

---

## 7. Making edits then redeploying
1. Edit `deployment/app/newvr_v2.py` (the live app's source).
2. Test locally (§6).
3. Redeploy (§5, Option 1).

---

## 8. Quick reference
- **App URL:** https://raaghavc-synthetic-past.hf.space
- **Manage:** https://huggingface.co/spaces/RaaghavC/synthetic-past  (Logs, Settings → Restart / Factory rebuild / Secrets / Hardware)
- **HF tokens:** https://huggingface.co/settings/tokens
- **OpenAI:** https://platform.openai.com  •  **ModelsLab:** https://modelslab.com  •  **Imgur:** https://imgur.com/account/settings/apps
- **Deploy bundle:** `deployment/app/`  •  **Redeploy:** `app/deploy.py`
- **Golden rule:** keep ModelsLab + OpenAI funded with spend caps; don't delete the Space; it auto-wakes for each visitor.
