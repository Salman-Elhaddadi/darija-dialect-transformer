"""Load the MSA-vs-Darija splits, pinned to the exact rows the baseline repo used.

The splits are NOT re-derived here. They are copied verbatim from
darija-sentiment-classifier (`data/processed/{train,val,test}.csv`) so that
every macro-F1 in this repo is computed over the identical rows as the
TF-IDF+LR and from-scratch-MLP numbers it is being compared against.
Re-running that repo's `data.py` would also reproduce them, but only as long
as scikit-learn's `train_test_split` RNG stays stable across versions — a
dependency the comparison should not rest on.

The CSVs themselves are gitignored (the MAC corpus states no license, so this
repo follows the baseline repo in not vendoring it). `splits.sha256` at the
repo root is checked in instead: it pins the exact bytes, so a mismatch is a
loud failure rather than a silently different test set.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")
MANIFEST = Path("splits.sha256")
SPLITS = ("train", "val", "test")
LABELS = ["standard", "dialectal"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / f"{name}.csv")


def verify() -> None:
    """Fail loudly if the split files differ from the ones the manifest pins.

    Silent drift here would invalidate every cross-repo comparison in
    RESULTS.md, so this runs before training rather than being trusted.
    """
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"{MANIFEST} missing — run `python src/data.py --write-manifest` once "
            "after copying the splits in from darija-sentiment-classifier."
        )

    expected = dict(
        line.split()[::-1] for line in MANIFEST.read_text().strip().splitlines()
    )
    for name in SPLITS:
        path = PROCESSED / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — copy the splits in from "
                "darija-sentiment-classifier/data/processed/."
            )
        actual = sha256(path)
        if actual != expected[str(path).replace("\\", "/")]:
            raise RuntimeError(
                f"{path} does not match {MANIFEST}. These splits are not the ones "
                "the baseline numbers were computed on — the comparison would be invalid."
            )


def write_manifest() -> None:
    lines = [
        f"{sha256(PROCESSED / f'{name}.csv')}  {PROCESSED.as_posix()}/{name}.csv"
        for name in SPLITS
    ]
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"wrote {MANIFEST}:")
    print("\n".join(lines))


def main() -> None:
    if "--write-manifest" in sys.argv:
        write_manifest()
        return

    verify()
    print(f"splits verified against {MANIFEST}\n")
    for name in SPLITS:
        df = load(name)
        chars = df["text"].str.len()
        print(
            f"{name:5s} n={len(df):5d}  dialectal={df['label'].mean():.3f}  "
            f"chars: median={chars.median():.0f} p95={chars.quantile(0.95):.0f} max={chars.max()}"
        )


if __name__ == "__main__":
    main()
