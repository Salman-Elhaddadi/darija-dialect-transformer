"""Is variant A's number about the representation, or about how its head was trained?

Variant A deliberately reuses the baseline MLP's training protocol (AdamW,
lr=1e-3, batch 32, early stopping on val macro-F1) so that the frozen-MARBERT
result is comparable to the MLP result. That choice invites an obvious
objection: mini-batch SGD with early stopping is not guaranteed to converge a
linear model, so a weak variant-A score could mean "frozen features are not
that good" OR "the head simply was not trained to convergence" — and those two
have opposite implications for whether fine-tuning was worth it.

This settles it by fitting scikit-learn's LogisticRegression (LBFGS, run to
convergence, class_weight="balanced" to match pos_weight) on the exact same
frozen features. If the two land close, variant A's head is converged and its
score reflects the representation. If LBFGS is much higher, variant A
understates frozen MARBERT and the A-vs-B gap is partly an artifact.

Post-hoc only: it reports on locked results and changes no model in the
comparison. Run after src/marbert.py.
"""

import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import data
from marbert import (
    MODELS,
    SEEDS,
    compute_pos_weight,
    extract_features,
    load_encoder,
    tokenize_split,
    train_head_on_features,
)
from transformers import AutoTokenizer


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data.verify()

    pooling = torch.load(MODELS / "marbert_frozen_head.pt")["pooling"]
    print(f"cross-checking variant A's head, pooling={pooling}\n")

    splits = {name: data.load(name) for name in ("train", "val", "test")}
    tokenizer = AutoTokenizer.from_pretrained("models/marbert-base")
    token_ids = {n: tokenize_split(tokenizer, splits[n]["text"].tolist()) for n in splits}

    encoder = load_encoder()
    feats = {n: extract_features(encoder, token_ids[n], pooling) for n in splits}
    del encoder

    y = {n: splits[n]["label"].values for n in splits}
    pos_weight = compute_pos_weight(y["train"])

    sgd_scores = []
    for seed in SEEDS:
        head, _ = train_head_on_features(feats["train"], y["train"], feats["val"], y["val"], pos_weight, seed)
        head.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(head(feats["test"]).squeeze(-1)) >= 0.5).long().numpy()
        sgd_scores.append(f1_score(y["test"], preds, average="macro"))
    sgd_scores = np.array(sgd_scores)

    lbfgs = LogisticRegression(max_iter=5000, class_weight="balanced").fit(
        feats["train"].numpy(), y["train"]
    )
    lbfgs_score = f1_score(y["test"], lbfgs.predict(feats["test"].numpy()), average="macro")

    print(f"  variant A head (AdamW + early stopping): {sgd_scores.mean():.4f} +/- {sgd_scores.std():.4f}")
    print(f"  LogisticRegression (LBFGS, converged)  : {lbfgs_score:.4f}")
    delta = lbfgs_score - sgd_scores.mean()
    print(f"  difference: {delta:+.4f}")
    print(
        "\n  -> variant A's head is effectively converged; its score reflects the "
        "frozen representation, not the optimizer."
        if abs(delta) < 0.01
        else "\n  -> the two disagree by more than 1 point: variant A's protocol, not the "
        "representation, is shaping its score. Say so in RESULTS.md."
    )


if __name__ == "__main__":
    main()
