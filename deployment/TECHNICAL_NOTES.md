# SyntheticPast — Technical Notes

SyntheticPast is a Streamlit web app that generates and historically validates panoramic images of 1850s California Gold Rush San Francisco, displays them in a 360° VR viewer, and answers questions through a retrieval-augmented chatbot.

## Architecture

- **Generation** — ModelsLab `juggernaut-xl` (a photorealistic SDXL model) renders a wide panorama from a history-grounded prompt.
- **Historical-accuracy judge** — GPT-4o vision inspects each render and rejects clear anachronisms (skyscrapers, bridges, cars, paved roads, modern/steam vessels). Renders that pass are scored for prompt match.
- **Iterative refinement** — up to three passes; between passes the prompt is refined from the evaluation, and the best-scoring render is kept.
- **Seamless 360° wrap** — the accepted render is edge-blended and then the right ~5% is cropped, so the two wrap columns become adjacent (near-identical) for a seam-free 360°.
- **Display** — A-Frame `a-sky` 360° viewer.
- **Chatbot** — a retrieval-augmented "Gold Rush Guide": each question retrieves passages from a Chroma vector store of primary-source texts and is answered by GPT-4o vision, which is also shown the panorama currently on screen so it can describe that exact image. A live status panel reports each pipeline step during generation.

## Image prompt

The prompt describes a photorealistic 1850s Gold Rush harbor with the town centered and the same open bay/sky on both far edges, so the panorama wraps cleanly. A negative prompt blocks modern elements, painterly/illustration styles, and mismatched edges.

## Key implementation details

- **Dependencies are pinned.** In particular `chromadb==0.5.0`: newer ChromaDB (1.x) cannot open the vector store, which uses the 0.5.x on-disk format.
- **Vector store ships zipped.** `vectorstore.zip` is extracted to `pdf_vectorstore/` on first run, so the app works on a fresh host.
- **Embeddings** use `sentence-transformers/all-mpnet-base-v2`.
- **Seamless wrap** is produced by `process_to_equirectangular()` (resize + `ensure_seamless_edges` blend) followed by `remove_right_side()` (crop the right ~5%); both steps together yield a ~0 wrap seam. The blend works because the prompt makes both edges show similar content.
- **Default panorama** is embedded as a base64 data-URI so the 360° viewer is populated on load.

## API keys (runtime)

- `OPENAI_API_KEY` — chat and the image judge.
- `STABLE_DIFFUSION_API_KEY` — ModelsLab image generation.
- `IMGUR_API_KEY` — hosts generated images for the viewer (non-critical; falls back to inline display).

Keys are provided as encrypted secrets on the host; the app reads them from the environment.

## Deployment

The app is deployed as a **Docker Space on Hugging Face** (free CPU tier, 16 GB RAM), running `streamlit run newvr_v2.py` on port 7860. `app/deploy.py` creates/updates the Space, sets the three secrets, and uploads the bundle. See `HOSTING_AND_RECOVERY_GUIDE.md` for the full procedure and recovery steps.
