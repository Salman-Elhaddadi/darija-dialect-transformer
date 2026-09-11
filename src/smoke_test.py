"""Does the SHIPPED model actually work when loaded the way a deployment loads it?

RESULTS.md already establishes that this model is good: 0.8964 +/- 0.0017 macro-F1
over 1,852 held-out rows. This script is not trying to re-establish that, and eight
hand-picked sentences could not. It tests something the metrics cannot touch.

Every number in RESULTS.md came from src/marbert.py holding a model in memory, with
the encoder read from the local models/marbert-base snapshot. The demos do something
different: they read models/marbert_frozen_head.pt off disk, rebuild the architecture
from scratch, and pull the encoder from the Hub. That serialise -> reload -> re-serve
round trip is a separate code path, and it can be completely broken while every
published number stays valid. Concretely, this catches:

  - a head checkpoint that does not load into the architecture the demos declare
  - a pooling mode disagreeing between training and serving
  - the Hub encoder differing from the local snapshot in a way that breaks scoring
  - INVERTED LABEL POLARITY, which macro-F1 physically cannot detect: flip every
    prediction and the score is unchanged, but the demo confidently calls Darija
    "MSA". This is the failure mode that most deserves a test.

Deliberately self-contained: it imports only torch and transformers, never marbert.py,
because marbert.py needs pandas, scikit-learn and the data splits, none of which exist
in a deployment environment. Importing it would make this pass in a place the demo
would fail. The cost is that DialectClassifier is restated here, a third copy
alongside deploy/streamlit/app.py and deploy/hf_space/app.py. That duplication is
forced by the HF Space being its own repo that cannot import from src/, and it is the
one thing this test does NOT protect: if an app's copy drifts from this one, this
still passes. Keep the three in sync by hand.

The cases are real corpus rows that all four models classify correctly. That is on
purpose: a smoke test should use unambiguous inputs, because it is asking "is this
wired up at all", not "how accurate is it". Ambiguous rows belong in the error
analysis, not here.

    python src/smoke_test.py          # exits non-zero if anything fails

Runs on CPU. Needs network on first run only, to pull the encoder from the Hub.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ENCODER = "UBC-NLP/MARBERT"
HEAD_PATH = Path(__file__).resolve().parents[1] / "models" / "marbert_frozen_head.pt"
MAX_LENGTH = 96
THRESHOLD = 0.5

# 1 = dialectal (Darija), 0 = standard (MSA) — matches data.LABELS = ["standard", "dialectal"]
CASES = [
    (1, "آلله سهل عليك خويا الف مبروك"),
    (1, "يا ربي تربح الحسنية سواسة الله يعمرها دار"),
    (1, "بالصح لكن ماشي للسباحة"),
    (1, "تشرفنا فاي وقت اخي مرحبا بيك عندنا"),
    (0, "كيف لا اقف احتراما امام هده الكلمات"),
    (0, "بلد تقتل ناسها وتلاحق جمهورها العاشق للكرة"),
    (0, "صباح الخيرات و البركات كل البلدان لا تضاهي جمال بلدي"),
    (0, "اللهم اكتب لي ما تراه خير لي و إرضني به"),
]


class DialectClassifier(torch.nn.Module):
    """Mirrors src/marbert.py so the saved state dict loads unchanged."""

    def __init__(self, encoder, pooling="mean"):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.head = torch.nn.Linear(encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled).squeeze(-1)


def load_served_model():
    if not HEAD_PATH.exists():
        sys.exit(f"FAIL: {HEAD_PATH} is missing — there is nothing to serve.")

    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    encoder = AutoModel.from_pretrained(ENCODER)

    saved = torch.load(HEAD_PATH, map_location="cpu", weights_only=False)
    model = DialectClassifier(encoder, pooling=saved["pooling"])
    # strict=True: a shape mismatch must crash here rather than silently leaving a
    # randomly initialised head in place, which would still "run" and score ~0.5.
    model.head.load_state_dict(saved["head"])
    model.eval()
    return tokenizer, model, saved["pooling"]


@torch.no_grad()
def probability_dialectal(tokenizer, model, text: str) -> float:
    batch = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    return torch.sigmoid(model(batch["input_ids"], batch["attention_mask"])).item()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    tokenizer, model, pooling = load_served_model()
    print(f"loaded {HEAD_PATH.name} (pooling={pooling}) onto {ENCODER} from the Hub\n")

    failures, probabilities = [], []
    print(f"{'gold':>8}  {'P(darija)':>9}  {'predicted':>9}  text")
    for gold, text in CASES:
        p = probability_dialectal(tokenizer, model, text)
        predicted = int(p >= THRESHOLD)
        probabilities.append(p)
        ok = predicted == gold
        if not ok:
            failures.append((gold, predicted, p, text))
        print(
            f"{'darija' if gold else 'msa':>8}  {p:>9.4f}  "
            f"{'darija' if predicted else 'msa':>9}  {'ok ' if ok else 'MISS'} {text[:44]}"
        )

    spread = max(probabilities) - min(probabilities)
    print(f"\nprobability spread across cases: {spread:.4f}")

    # A head that failed to load produces near-identical output for every input. That
    # looks like "uncertainty" but is actually a dead model, so it is failed explicitly
    # rather than left for a human to notice.
    if spread < 0.10:
        sys.exit(
            "FAIL: every input scored almost the same. That is the signature of an\n"
            "      unloaded or randomly initialised head, not genuine uncertainty."
        )

    if failures:
        print(f"\nFAIL: {len(failures)}/{len(CASES)} unambiguous cases on the wrong side "
              f"of {THRESHOLD}.")
        if all(predicted != gold for gold, predicted, _, _ in failures) and len(failures) == len(CASES):
            print("      Every single case is inverted — suspect flipped label polarity,\n"
                  "      not a weak model.")
        sys.exit(1)

    print(f"\nPASS: {len(CASES)}/{len(CASES)} correct. The serving path loads and "
          f"discriminates.\n      This says nothing about accuracy — see RESULTS.md for that.")


if __name__ == "__main__":
    main()
