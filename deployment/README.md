# SyntheticPast — deployment

A self-contained, deployable build of the SyntheticPast web app, with deployment and operations docs.

## Live app
**https://raaghavc-synthetic-past.hf.space** — public, no setup required for visitors.

## Contents
| Path | What it is |
|---|---|
| `app/` | The complete, self-contained, deployable app (matches what is live on Hugging Face). |
| `app/newvr_v2.py` | The app: Juggernaut-XL photorealistic panorama generation, a calibrated GPT-4o-vision historical-accuracy judge, a vision-enabled retrieval-augmented chatbot, and a default panorama. |
| `app/default_panorama.jpg` | Default panorama shown in the 360° viewer on load. |
| `app/vectorstore.zip` | The chatbot's knowledge base (auto-extracted on first run). |
| `app/Dockerfile`, `app/requirements.txt`, `app/README.md` | Deployment config (Docker Space, pinned dependencies, Space metadata). |
| `app/deploy.py` | One-command (re)deploy script — see the hosting guide. |
| `TECHNICAL_NOTES.md` | Architecture and implementation notes (model, prompt, judge, seamless 360° wrap, pipeline, deployment). |
| `HOSTING_AND_RECOVERY_GUIDE.md` | How it is hosted, how to keep it running, how to recover it, and how to redeploy from scratch. |
| `exploration_panoramas/` | Candidate panoramas from model and prompt comparisons. |
| `screenshots/` | Reference screenshots. |

## Keeping it running
Keep the OpenAI and ModelsLab accounts funded (with spend caps), don't delete the Hugging Face Space, and the link auto-wakes on each visit. Details in `HOSTING_AND_RECOVERY_GUIDE.md`.
