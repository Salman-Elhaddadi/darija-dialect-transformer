# Streamlit Community Cloud deployment (MARBERT, frozen variant)

This is the free host that actually works for the transformer. The other two deployment
targets in this repo each fail for a specific, documented reason:

- `deploy/render/` — Render's free instance is 512 MB RAM. MARBERT is 654 MB in fp32
  before torch's own footprint, so it cannot fit at any batch size. Render serves the
  logistic-regression baseline instead, which is the honest version of "we shipped both".
- `deploy/hf_space/` — Hugging Face now refuses cpu-basic compute to new free accounts
  (`403: You've reached your cpu-basic quota limit`), and ZeroGPU has moved behind a PRO
  subscription. The Space is committed and correct; it just cannot be scheduled.

Streamlit Community Cloud is free, needs no card, and gives enough RAM to hold the encoder.

## Deploying

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → **Deploy a public app from GitHub**.
3. Fill in:
   - Repository: `Salman-Elhaddadi/darija-dialect-transformer`
   - Branch: `main`
   - Main file path: `deploy/streamlit/app.py`
4. Deploy. First boot takes several minutes: it installs torch and downloads the ~654 MB
   MARBERT encoder from the Hub. Later boots reuse the cached encoder.

Nothing needs to be uploaded. The encoder comes from the Hub at boot and the only weights
this repo ships are the 5 KB head in `models/marbert_frozen_head.pt` — which is the entire
argument for the frozen variant over the fine-tuned one.

## If the app runs out of memory

Free-tier RAM has changed more than once. If the app is killed during `load_model`, the
fix is to shrink the encoder rather than the head:

```python
encoder = AutoModel.from_pretrained(ENCODER, dtype=torch.float16)
```

fp16 halves the encoder to ~327 MB. CPU fp16 matmuls are slower and this changes the
numerics slightly, so it is a deliberate fallback and not the default — the deployed
model should match the evaluated model unless there is a reason it cannot.
