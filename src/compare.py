"""The deployment table: macro-F1 vs. model size vs. CPU inference latency.

Accuracy alone does not decide what ships. This measures the two costs that do —
what you have to store and how long a single request takes — under the condition
these models would actually be served in for a project like this: one text at a
time, on CPU, no batching, no GPU.

Latency is per-request wall time end to end (raw text in, label out), including
vectorization/tokenization, because that is what a caller waits for. Batch size
is 1 deliberately: batching is what you would do to make a throughput number look
good, and it is exactly what a low-traffic HTTP endpoint never gets to do.

Size is what must be present at inference, not what was trained. Variant A's own
artifact is a ~3KB linear head, but it cannot answer a request without the 654MB
encoder underneath it, so it is charged for both.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer

import data
from baselines import DialectNet
from marbert import DialectClassifier, ENCODER_DIR, MAX_LENGTH

MODELS = Path("models")
REPORTS = Path("reports")
N_LATENCY_SAMPLES = 200
WARMUP = 20


def measure(predict_one, texts) -> tuple[float, float]:
    for text in texts[:WARMUP]:
        predict_one(text)
    timings = []
    for text in texts[:N_LATENCY_SAMPLES]:
        t0 = time.perf_counter()
        predict_one(text)
        timings.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(timings, 50)), float(np.percentile(timings, 95))


def dir_size(*paths) -> int:
    total = 0
    for path in paths:
        path = Path(path)
        if path.is_dir():
            total += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        elif path.exists():
            total += path.stat().st_size
    return total


def main() -> None:
    data.verify()
    test = data.load("test")
    texts = test["text"].tolist()

    baseline_metrics = json.loads((REPORTS / "metrics_baselines.json").read_text())
    marbert_metrics = json.loads((REPORTS / "metrics_marbert.json").read_text())
    rows = []

    # --- LR ---
    lr = joblib.load(MODELS / "baseline/baseline_lr.joblib")
    p50, p95 = measure(lambda t: lr.predict([t]), texts)
    rows.append({
        "model": "LR (char n-gram TF-IDF)",
        "macro_f1": f"{baseline_metrics['lr']['test_macro_f1']:.3f}",
        "size_mb": dir_size(MODELS / "baseline/baseline_lr.joblib") / 1e6,
        "p50_ms": p50, "p95_ms": p95,
    })

    # --- MLP ---
    vectorizer = joblib.load(MODELS / "baseline/vectorizer.joblib")
    mlp = DialectNet(len(vectorizer.vocabulary_))
    mlp.load_state_dict(torch.load(MODELS / "baseline/nn_seed0.pt"))
    mlp.eval()

    def mlp_predict(text):
        x = torch.tensor(vectorizer.transform([text]).toarray(), dtype=torch.float32)
        with torch.no_grad():
            return int(torch.sigmoid(mlp(x)).item() >= 0.5)

    p50, p95 = measure(mlp_predict, texts)
    mlp_stats = baseline_metrics["mlp"]
    rows.append({
        "model": "MLP (char n-gram TF-IDF)",
        "macro_f1": f"{mlp_stats['test_macro_f1_mean']:.3f} ± {mlp_stats['test_macro_f1_std']:.3f}",
        "size_mb": dir_size(MODELS / "baseline/vectorizer.joblib", MODELS / "baseline/nn_seed0.pt") / 1e6,
        "p50_ms": p50, "p95_ms": p95,
    })

    # --- MARBERT variants ---
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_DIR)
    encoder_size = dir_size(ENCODER_DIR)

    if "marbert_frozen" in marbert_metrics:
        saved = torch.load(MODELS / "marbert_frozen_head.pt")
        model = DialectClassifier(AutoModel.from_pretrained(ENCODER_DIR), pooling=saved["pooling"], frozen=True)
        model.head.load_state_dict(saved["head"])
        model.eval()

        def frozen_predict(text):
            batch = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            with torch.no_grad():
                logit = model(batch["input_ids"], batch["attention_mask"])
            return int(torch.sigmoid(logit).item() >= 0.5)

        p50, p95 = measure(frozen_predict, texts)
        stats = marbert_metrics["marbert_frozen"]
        rows.append({
            "model": f"MARBERT frozen + linear head ({saved['pooling']} pool)",
            "macro_f1": f"{stats['test_macro_f1_mean']:.3f} ± {stats['test_macro_f1_std']:.3f}",
            "size_mb": (encoder_size + dir_size(MODELS / "marbert_frozen_head.pt")) / 1e6,
            "p50_ms": p50, "p95_ms": p95,
        })
        del model

    if "marbert_finetuned" in marbert_metrics:
        model = DialectClassifier(AutoModel.from_pretrained(ENCODER_DIR), pooling="mean", frozen=False)
        model.load_state_dict(torch.load(MODELS / "marbert_finetuned.pt"))
        model.eval()

        def finetuned_predict(text):
            batch = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            with torch.no_grad():
                logit = model(batch["input_ids"], batch["attention_mask"])
            return int(torch.sigmoid(logit).item() >= 0.5)

        p50, p95 = measure(finetuned_predict, texts)
        stats = marbert_metrics["marbert_finetuned"]
        rows.append({
            "model": "MARBERT fully fine-tuned",
            "macro_f1": f"{stats['test_macro_f1_mean']:.3f} ± {stats['test_macro_f1_std']:.3f}",
            "size_mb": dir_size(MODELS / "marbert_finetuned.pt", ENCODER_DIR / "vocab.txt",
                                ENCODER_DIR / "config.json") / 1e6,
            "p50_ms": p50, "p95_ms": p95,
        })

    table = pd.DataFrame(rows)
    baseline_p50 = table["p50_ms"].iloc[0]
    table["vs LR latency"] = (table["p50_ms"] / baseline_p50).map(lambda v: f"{v:.0f}x")
    table["size_mb"] = table["size_mb"].map(lambda v: f"{v:.1f}")
    table["p50_ms"] = table["p50_ms"].map(lambda v: f"{v:.1f}")
    table["p95_ms"] = table["p95_ms"].map(lambda v: f"{v:.1f}")

    print(f"\nCPU: {torch.get_num_threads()} torch threads, batch size 1, "
          f"{N_LATENCY_SAMPLES} requests after {WARMUP} warmup\n")
    print(table.to_string(index=False))

    (REPORTS / "comparison.json").write_text(json.dumps(rows, indent=2))
    (REPORTS / "comparison.md").write_text(table.to_markdown(index=False) + "\n")
    print(f"\nwrote {REPORTS}/comparison.json and comparison.md")


if __name__ == "__main__":
    main()
