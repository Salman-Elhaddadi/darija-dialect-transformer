# Darija Dialect Transformer

Binary MSA-vs-Darija dialect ID, MARBERT fine-tuned on the MAC corpus, benchmarked against from-scratch classical baselines from [darija-dialect-classifier](https://github.com/Salman-Elhaddadi/darija-dialect-classifier).

Same task, same split (fixed seed 42, stratified, sha256-pinned), same eval protocol as the baseline repo — only the model changes. That's the point: it isolates what a pretrained transformer actually buys over hand-engineered features.

## Results

Test split is 1,852 rows, 70.7% MSA / 29.3% Darija. Everything is macro-F1, because on a 7:3 split a model that always predicts MSA already scores 70.7% accuracy (and 0.414 macro-F1).

| Model | macro-F1 | Errors (/1852) | Deployment artifact |
|---|---|---|---|
| Always-MSA (degenerate reference) | 0.4141 | 543 | — |
| LR (char n-gram) | 0.8625 | 213 | ~1 MB |
| MLP (char n-gram) | 0.8497 | 236 | ~1 MB |
| MARBERT frozen encoder + linear head | 0.8983 | 155 | 5 KB head (encoder pulled from HF Hub) |
| MARBERT full fine-tune | **0.9094** | 138 | 654 MB |

Those MARBERT figures are the single seed whose predictions are persisted for error analysis, and it happens to be the best of the five that were run. **The number to quote is the 5-seed mean:** 0.8964 ± 0.0017 frozen, 0.9063 ± 0.0025 fine-tuned. The ranking is not seed noise — the frozen mean beats LR by 0.034, about twenty times its own standard deviation — but the gap between the two MARBERT variants (~0.010) is only a few standard deviations, so treat it as real but small.

Full breakdown, including error rate by text length and the pooling selection: **[RESULTS.md](RESULTS.md)**.

## Is it actually a different model, or the same boundary done better?

LR and the from-scratch MLP share 84.5% of their errors — same decision boundary, different execution. MARBERT breaks that pattern:

| | Overlap with LR's errors |
|---|---|
| MLP | 84.5% |
| MARBERT frozen | 45.1% |
| MARBERT fine-tuned | 43.2% |

MARBERT is wrong about genuinely different examples than the classical models, not just fewer of the same ones. Full error analysis, quartile breakdown, and the 68 rows every model gets wrong: `reports/error_analysis.json`.

## Deployment

The frozen variant is what gets deployed. It scores 0.010 macro-F1 below the full fine-tune while shipping a 5 KB head instead of a 654 MB checkpoint, and the encoder it depends on is pulled unmodified from the Hub at boot — so nothing large is ever stored in git or uploaded to a host.

| Target | Serves | Status |
|---|---|---|
| [Streamlit Community Cloud](deploy/streamlit/) | MARBERT, frozen | Works on free tier — deploys straight from this repo |
| [Render](deploy/render/) | LR baseline | Works on free tier |
| [HF Space](deploy/hf_space/) | MARBERT, frozen | Committed and correct, but unschedulable — see below |

Two of these are constrained by hosting economics rather than by the models:

- **Render's free instance is 512 MB RAM.** MARBERT is 654 MB in fp32 before torch's own footprint, so it does not fit at any batch size. Render serves the logistic-regression baseline instead. That split is the deployment half of the accuracy-versus-artifact-size tradeoff above, not an accident of tooling.
- **Hugging Face free tier now returns `403: You've reached your cpu-basic quota limit`** on new accounts, and ZeroGPU has moved behind a PRO subscription. The Space at [Salman-elhaddadi/darija-dialect-id](https://huggingface.co/spaces/Salman-elhaddadi/darija-dialect-id) is fully configured and will run if that quota ever frees up; it is kept here as a working config, not as a live demo.

Streamlit Community Cloud is the free host with enough RAM to actually hold the encoder, which is why the live transformer demo lives there. Deployment steps: [`deploy/streamlit/README.md`](deploy/streamlit/README.md).

## Reproducing

```
pip install -r requirements.txt
python src/marbert.py
```

`src/data.py` regenerates `data/processed/{train,val,test}.csv` deterministically from the raw MAC corpus (fixed seed, stratified split) and verifies it against `splits.sha256` before training — a mismatch there means you're not comparing against the same test set the numbers above were computed on. `src/fetch_encoder.py` downloads the MARBERT checkpoint with chunked/resumable transfer (useful on a slow connection).

One caveat worth knowing before you try to match published numbers: fixing a seed pins the *sampling*, not the row order the sampler sees. A different pandas/scikit-learn pair can reorder rows ahead of the stratified split and produce a different `splits.sha256`. Every model in the tables above was therefore retrained and rescored in a single environment against one freshly generated split, so the comparison is internally valid even where the absolute figures differ from earlier runs of the baseline repo.

## Repo layout

- `src/marbert.py` — training entry point, both variants, 5-seed eval
- `src/data.py` — deterministic split + verification
- `src/analyze.py` — cross-model error analysis (reads persisted predictions only, no retraining)
- `src/fetch_encoder.py` — resumable MARBERT weight download
- `src/smoke_test.py` — checks the shipped head loads and discriminates through the *deployment* path (Hub encoder, torch + transformers only); catches breakage that accuracy metrics cannot, including inverted label polarity
- `src/score_deployed.py` — scores the shipped head on the test split; `--encoder local|hub` also checks that the Hub encoder the demos use matches the snapshot training used
- `src/serve.py` — FastAPI serving for any of the three models, selected by `MODEL` env var
- `models/marbert_frozen_head.pt` — the trained linear head, 5 KB, with its pooling mode recorded alongside the weights
- `deploy/` — Streamlit (MARBERT), Render (LR baseline), HF Space (MARBERT) configs
- `reports/` — per-example predictions and metrics for every model
- `RESULTS.md` — full results write-up

## Known limitations

- Latency/throughput benchmarking not yet done — the tables above are accuracy and artifact size only. MARBERT is obviously slower per request than a TF-IDF pipeline, but that hasn't been quantified, so no claim is made about it.
- Trained and evaluated on one dataset (MAC corpus); generalization to other Darija sources (YouTube comments, forums) is untested.
- The corpus is Arabic-script only, so Latin-script Darija (Arabizi) is out of domain. The serving code flags it rather than scoring it silently.
- The two classical baselines were not re-run across seeds in this repo, so they appear as a single number each. LR is deterministic; the MLP is not.
- The shipped head is the seed with the best *validation* score, while the single-seed column above is the best *test* seed — so the deployed model sits somewhere in 0.8944–0.8983 rather than exactly at 0.8983. Selecting on validation is the right call; it just means the deployed model's exact score has to be measured with `src/score_deployed.py` instead of read off the table.
