"""Re-score the locked LR and MLP baselines on this repo's test split.

Nothing is retrained here. The artifacts are the exact ones the baseline repo
(darija-sentiment-classifier) trained and reported from — copied into
`models/baseline/` — so the numbers this prints must reproduce that repo's
published figures (LR 0.862, MLP 0.843 +/- 0.005) exactly. They are asserted
below rather than merely printed: if the copied splits or artifacts ever drift,
the comparison this whole repo rests on is invalid, and that should fail loudly
here rather than quietly shift a number in RESULTS.md.

Writes per-example test predictions so the transformer error analysis can ask
where MARBERT diverges from these two, not just whether it scores higher.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

import data

MODELS = Path("models/baseline")
REPORTS = Path("reports")

# Mirrors darija-sentiment-classifier/src/nn.py. Duplicated rather than imported
# so this repo stands alone; the state dicts below will not load if it drifts.
HIDDEN_WIDTH = 32
DROPOUT_P = 0.2
SEEDS = [0, 1, 2, 3, 4]

PUBLISHED = {"lr": 0.862, "mlp_mean": 0.843, "mlp_std": 0.005}


class DialectNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_WIDTH),
            nn.ReLU(),
            nn.Dropout(DROPOUT_P),
            nn.Linear(HIDDEN_WIDTH, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def main() -> None:
    data.verify()
    REPORTS.mkdir(exist_ok=True)
    test = data.load("test")
    y_test = test["label"].values

    lr_model = joblib.load(MODELS / "baseline_lr.joblib")
    lr_preds = lr_model.predict(test["text"])
    lr_f1 = f1_score(y_test, lr_preds, average="macro")
    print(f"LR   test macro-F1 = {lr_f1:.4f}  (published {PUBLISHED['lr']})")

    vectorizer = joblib.load(MODELS / "vectorizer.joblib")
    X_test = torch.tensor(vectorizer.transform(test["text"]).toarray(), dtype=torch.float32)

    mlp_f1s, mlp_preds_per_seed = [], []
    for seed in SEEDS:
        model = DialectNet(X_test.shape[1])
        model.load_state_dict(torch.load(MODELS / f"nn_seed{seed}.pt"))
        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(model(X_test)) >= 0.5).long().numpy()
        mlp_f1s.append(f1_score(y_test, preds, average="macro"))
        mlp_preds_per_seed.append(preds)
        print(f"  MLP seed {seed}: test macro-F1 = {mlp_f1s[-1]:.4f}")

    mlp_f1s = np.array(mlp_f1s)
    print(
        f"MLP  test macro-F1 = {mlp_f1s.mean():.4f} +/- {mlp_f1s.std():.4f}  "
        f"(published {PUBLISHED['mlp_mean']} +/- {PUBLISHED['mlp_std']})"
    )

    assert abs(lr_f1 - PUBLISHED["lr"]) < 5e-4, f"LR drifted: {lr_f1:.4f}"
    assert abs(mlp_f1s.mean() - PUBLISHED["mlp_mean"]) < 5e-4, f"MLP drifted: {mlp_f1s.mean():.4f}"

    # Best-test-seed, matching the convention the baseline repo's diagnose.py used
    # for its per-example error-overlap table -- so "did the 84.5% overlap shrink?"
    # is asked against the same MLP predictions that produced the 84.5%.
    best = int(mlp_f1s.argmax())

    pd.DataFrame(
        {"text": test["text"], "label": y_test, "lr": lr_preds, "mlp": mlp_preds_per_seed[best]}
    ).to_csv(REPORTS / "preds_baselines.csv", index=False)

    (REPORTS / "metrics_baselines.json").write_text(
        json.dumps(
            {
                "lr": {"test_macro_f1": float(lr_f1)},
                "mlp": {
                    "test_macro_f1_mean": float(mlp_f1s.mean()),
                    "test_macro_f1_std": float(mlp_f1s.std()),
                    "per_seed": [float(f) for f in mlp_f1s],
                    "preds_from_seed": SEEDS[best],
                },
            },
            indent=2,
        )
    )
    print(f"\nwrote {REPORTS}/preds_baselines.csv and metrics_baselines.json")


if __name__ == "__main__":
    main()
