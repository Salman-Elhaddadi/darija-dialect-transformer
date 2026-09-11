"""What does the head we actually ship score on the test set?

RESULTS.md quotes variant A as 0.8964 +/- 0.0017 over five seeds, and 0.8983 for the
single seed whose predictions are persisted. Neither of those is necessarily the score
of the checkpoint in models/marbert_frozen_head.pt, because run_variant_a picks them
by different criteria:

    line 419:  best_val, best_head = val_f1, head          # saved by best VALIDATION
    line 430:  "preds": preds_per_seed[test_f1s.argmax()]  # reported by best TEST

Selecting the shipped head on validation is the correct choice — you do not get to
pick a checkpoint using the test set. But it means the deployed model is some seed in
the 0.8944-0.8983 range and, until this script is run, nobody knows which. "What does
your deployed model score?" is a fair interview question and it currently has no
answer. This produces one.

It also settles a second thing, via --encoder:

    --encoder local   scores through models/marbert-base, the snapshot training used
    --encoder hub     scores through UBC-NLP/MARBERT pulled from the Hub, which is
                      what the Streamlit app and the Space actually serve

If the two agree, the Hub encoder is equivalent to the training snapshot and the demos
are serving the model that was evaluated. If they disagree, the demos are serving
something that was never measured, and the gap is the size of that error. Run both.

    python src/fetch_encoder.py          # only needed for --encoder local
    python src/score_deployed.py --encoder local
    python src/score_deployed.py --encoder hub

Needs the data splits and passes them through data.verify() first, so a score can
never be reported against a test set that is not the pinned one.
"""

import argparse
import sys

import torch
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModel, AutoTokenizer

import data
from marbert import ENCODER_DIR, LABELS, MODELS, extract_features, load_encoder, tokenize_split

HUB_ENCODER = "UBC-NLP/MARBERT"


def build_encoder(source: str):
    if source == "hub":
        return AutoTokenizer.from_pretrained(HUB_ENCODER), AutoModel.from_pretrained(HUB_ENCODER)
    return AutoTokenizer.from_pretrained(ENCODER_DIR), load_encoder()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["local", "hub"], default="local",
                        help="local = the snapshot training used; hub = what the demos serve")
    args = parser.parse_args()

    data.verify()

    head_path = MODELS / "marbert_frozen_head.pt"
    if not head_path.exists():
        sys.exit(f"{head_path} is missing — run `python src/marbert.py --variant a` first.")
    saved = torch.load(head_path, map_location="cpu", weights_only=False)
    pooling = saved["pooling"]

    test = data.load("test")
    print(f"scoring the shipped head (pooling={pooling}) on {len(test)} test rows "
          f"via the {args.encoder} encoder\n")

    tokenizer, encoder = build_encoder(args.encoder)
    token_ids = tokenize_split(tokenizer, test["text"].tolist())
    features = extract_features(encoder, token_ids, pooling)
    del encoder

    head = torch.nn.Linear(features.shape[1], 1)
    head.load_state_dict(saved["head"])  # strict on purpose
    head.eval()
    with torch.no_grad():
        predictions = (torch.sigmoid(head(features).squeeze(-1)) >= 0.5).long().numpy()

    y = test["label"].values
    macro_f1 = f1_score(y, predictions, average="macro")

    print(classification_report(y, predictions, target_names=LABELS, digits=4))
    print(f"  shipped head, {args.encoder} encoder: macro-F1 = {macro_f1:.4f}")
    print(f"  variant A 5-seed mean reported in RESULTS.md: 0.8964 +/- 0.0017")
    print(f"  difference from the mean: {macro_f1 - 0.8964:+.4f}")
    print("\n  This is the number to quote for the deployed model. Run the other\n"
          "  --encoder setting too: if local and hub agree, the demos are serving\n"
          "  the model that was evaluated.")


if __name__ == "__main__":
    main()
