# Darija Dialect Transformer

Binary MSA-vs-Darija dialect ID, MARBERT fine-tuned on the MAC corpus, benchmarked against from-scratch classical baselines from [darija-dialect-classifier](https://github.com/Salman-Elhaddadi/darija-dialect-classifier).

Same task, same split (fixed seed 42, stratified, sha256-pinned), same eval protocol as the baseline repo — only the model changes. That's the point: it isolates what a pretrained transformer actually buys over hand-engineered features.

## Results

| Model | macro-F1 | Errors (/1852) | Deployment artifact |
|---|---|---|---|
| LR (char n-gram) | 0.8625 | 213 | ~1 MB |
| MLP (char n-gram) | 0.8497 | 236 | ~1 MB |
| MARBERT frozen encoder + linear head | 0.8983 | 155 | ~3 KB head (encoder pulled from HF Hub) |
| MARBERT full fine-tune | **0.9094** | 138 | 654 MB |

Both MARBERT variants clearly beat both classical baselines. Full fine-tuning only buys ~1 point of macro-F1 over the frozen encoder, at the cost of a 654 MB checkpoint instead of a ~3 KB linear head.

## Is it actually a different model, or the same boundary done better?

The baseline repo found LR and the from-scratch MLP share 84.5% of their errors — same decision boundary, different execution. MARBERT breaks that pattern:

| | Overlap with LR's errors |
|---|---|
| MLP | 84.5% |
| MARBERT frozen | 45.1% |
| MARBERT fine-tuned | 43.2% |

MARBERT is wrong about genuinely different examples than the classical models, not just fewer of the same ones. Full error analysis, quartile breakdown, and the 68 rows every model gets wrong: `reports/error_analysis.json`.

## Reproducing

```
pip install -r requirements.txt
python src/marbert.py
```

`src/data.py` regenerates `data/processed/{train,val,test}.csv` deterministically from the raw MAC corpus (fixed seed, stratified split) and verifies it against `splits.sha256` before training — a mismatch there means you're not comparing against the same test set the numbers above were computed on. `src/fetch_encoder.py` downloads the MARBERT checkpoint with chunked/resumable transfer (useful on a slow connection).

## Repo layout

- `src/marbert.py` — training entry point, both variants, 5-seed eval
- `src/data.py` — deterministic split + verification
- `src/analyze.py` — cross-model error analysis (reads persisted predictions only, no retraining)
- `src/fetch_encoder.py` — resumable MARBERT weight download
- `deploy/` — Render (LR baseline) and HF Space (MARBERT) deployment configs
- `reports/` — per-example predictions and metrics for every model

## Known limitations

- Latency/throughput benchmarking not yet done — the table above is accuracy and artifact size only.
- Trained and evaluated on one dataset (MAC corpus); generalization to other Darija sources (YouTube comments, forums) is untested.
