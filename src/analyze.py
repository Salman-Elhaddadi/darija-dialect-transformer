"""Error analysis across LR, MLP, and both MARBERT variants on the test split.

Reads persisted per-example predictions only (`reports/preds_*.csv`) — no
retraining, no threshold search, no hyperparameter decisions. Everything here
characterizes results that are already locked.

The questions this answers are the ones the baseline repo's RESULTS.md left
open, asked against the same test rows and the same quartile boundaries:

  A. Where does the transformer beat LR, and where does it still lose?
  B. Does the 84.5% LR/MLP shared-error overlap shrink — i.e. is MARBERT finding
     a genuinely different decision boundary, or the same one executed better?
  C. Does the short-text penalty (both classical models roughly halve their error
     rate from shortest to longest quartile) survive subword pretraining?
  D. Which examples does every model get wrong? A floor of irreducibly hard or
     mislabeled rows bounds what any further modelling can buy.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPORTS = Path("reports")
MODEL_ORDER = ["lr", "mlp", "marbert_frozen", "marbert_finetuned"]
PRETTY = {
    "lr": "LR (char n-gram)",
    "mlp": "MLP (char n-gram)",
    "marbert_frozen": "MARBERT frozen",
    "marbert_finetuned": "MARBERT fine-tuned",
}


def load_predictions() -> pd.DataFrame:
    baselines = pd.read_csv(REPORTS / "preds_baselines.csv")
    marbert_path = REPORTS / "preds_marbert.csv"
    if not marbert_path.exists():
        raise FileNotFoundError(f"{marbert_path} missing — run `python src/marbert.py` first.")
    marbert = pd.read_csv(marbert_path)

    assert (baselines["label"].values == marbert["label"].values).all(), (
        "baseline and MARBERT prediction files disagree on the test labels — "
        "they are not aligned to the same rows"
    )
    frame = baselines.copy()
    for column in marbert.columns:
        if column.startswith("marbert"):
            frame[column] = marbert[column]
    return frame


def main() -> None:
    # Windows consoles default to cp1252; section D echoes Arabic corpus rows.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    frame = load_predictions()
    y = frame["label"].values
    models = [m for m in MODEL_ORDER if m in frame.columns]
    wrong = {m: frame[m].values != y for m in models}

    print("=== headline: test macro-F1 (best-seed predictions) ===")
    for m in models:
        print(f"  {PRETTY[m]:22s} macro-F1={f1_score(y, frame[m], average='macro'):.4f}  errors={wrong[m].sum()}")

    # ------------------------------------------------------------------
    # A: pairwise win/loss against LR
    # ------------------------------------------------------------------
    print("\n=== A: where each model beats / loses to LR ===")
    lr_wrong = wrong["lr"]
    for m in models:
        if m == "lr":
            continue
        fixed = int((lr_wrong & ~wrong[m]).sum())
        broke = int((~lr_wrong & wrong[m]).sum())
        print(
            f"  {PRETTY[m]:22s} fixes {fixed:4d} of LR's {int(lr_wrong.sum())} errors, "
            f"breaks {broke:4d} that LR got right  (net {fixed - broke:+d})"
        )

    # ------------------------------------------------------------------
    # B: error overlap with LR -- same boundary or a different one?
    # ------------------------------------------------------------------
    print("\n=== B: error overlap with LR ===")
    print("  (fraction of LR's errors that this model ALSO gets wrong)")
    for m in models:
        if m == "lr":
            continue
        overlap = (lr_wrong & wrong[m]).sum() / lr_wrong.sum()
        print(f"  {PRETTY[m]:22s} {overlap:.3f}   ({int((lr_wrong & wrong[m]).sum())}/{int(lr_wrong.sum())})")

    print("\n  pairwise error-overlap matrix (Jaccard: shared errors / union of errors)")
    matrix = pd.DataFrame(index=[PRETTY[m] for m in models], columns=[PRETTY[m] for m in models], dtype=float)
    for a in models:
        for b in models:
            union = (wrong[a] | wrong[b]).sum()
            matrix.loc[PRETTY[a], PRETTY[b]] = (wrong[a] & wrong[b]).sum() / union if union else 1.0
    print(matrix.round(3).to_string())

    # ------------------------------------------------------------------
    # C: error rate by text-length quartile
    # ------------------------------------------------------------------
    lengths = frame["text"].str.len().values
    quartile = pd.qcut(lengths, 4, labels=["Q1 shortest", "Q2", "Q3", "Q4 longest"])
    print("\n=== C: error rate by text-length quartile ===")
    by_quartile = pd.DataFrame(
        {PRETTY[m]: pd.Series(wrong[m]).groupby(quartile, observed=True).mean() for m in models}
    )
    print((by_quartile * 100).round(1).to_string())
    print("\n  ratio Q1/Q4 (how much harder the shortest quartile is for each model):")
    for m in models:
        rates = pd.Series(wrong[m]).groupby(quartile, observed=True).mean()
        print(f"    {PRETTY[m]:22s} {rates.iloc[0] / rates.iloc[-1]:.2f}x")

    # ------------------------------------------------------------------
    # D: the shared floor
    # ------------------------------------------------------------------
    all_wrong = np.logical_and.reduce([wrong[m] for m in models])
    none_wrong = np.logical_and.reduce([~wrong[m] for m in models])
    print("\n=== D: agreement floor ===")
    print(f"  every model correct: {int(none_wrong.sum())} / {len(y)}")
    print(f"  every model wrong:   {int(all_wrong.sum())} / {len(y)}  "
          f"({all_wrong.sum() / len(y) * 100:.1f}% -- a floor no model in this comparison clears)")
    print(f"  true-label split among always-wrong rows: "
          f"{dict(pd.Series(y[all_wrong]).map({0: 'standard', 1: 'dialectal'}).value_counts())}")

    print("\n  10 shortest rows every model gets wrong:")
    hard = frame[all_wrong].assign(n_chars=lengths[all_wrong]).nsmallest(10, "n_chars")
    for _, row in hard.iterrows():
        truth = "dialectal" if row["label"] == 1 else "standard "
        print(f"    [{truth}] {row['text'][:70]}")

    summary = {
        "macro_f1": {m: float(f1_score(y, frame[m], average="macro")) for m in models},
        "errors": {m: int(wrong[m].sum()) for m in models},
        "overlap_with_lr": {
            m: float((lr_wrong & wrong[m]).sum() / lr_wrong.sum()) for m in models if m != "lr"
        },
        "error_rate_by_quartile": {
            m: [float(v) for v in pd.Series(wrong[m]).groupby(quartile, observed=True).mean()]
            for m in models
        },
        "all_models_wrong": int(all_wrong.sum()),
    }
    (REPORTS / "error_analysis.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {REPORTS}/error_analysis.json")


if __name__ == "__main__":
    main()
