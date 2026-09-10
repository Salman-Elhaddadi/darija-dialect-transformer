"""MARBERT for MSA-vs-Darija dialect ID: frozen encoder (A) vs full fine-tune (B).

The point of this repo is a controlled comparison against the classical models
in darija-sentiment-classifier (TF-IDF+LR 0.862, from-scratch MLP 0.843), so
every protocol choice below is pinned to what those models did. What changes is
the input representation and what is trainable — not the split, not the loss,
not the threshold, not the seed count.

Decisions, each measured against this dataset rather than copied from a recipe:

  1. checkpoint     : UBC-NLP/MARBERT — pretrained on ~1B Arabic tweets including
                      dialectal ones. The task is dialect ID on tweets; a
                      MSA-only encoder (AraBERT, asafaya/bert-*-arabic, trained
                      on Wikipedia/OSCAR) would be pretrained on exactly the half
                      of the label space we need to discriminate.
  2. max_length     : 96 — the smallest round cap that truncates ZERO rows in all
                      three splits (measured: train max=71, val max=66, test
                      max=64). Nothing is silently cut. With dynamic padding the
                      cap almost never binds, so it costs nothing to be safe.
  3. padding        : dynamic, per batch. Median tweet is 9 tokens; padding every
                      batch to 96 would spend ~90% of the FLOPs on [PAD].
  4. head           : single Linear(768 -> 1) for BOTH variants. Same module, same
                      loss, same threshold — only *what is trainable* and the
                      learning rate differ between A and B, so the A-vs-B gap is
                      about adapting the encoder, not about head capacity.
  5. loss           : BCEWithLogitsLoss with pos_weight = n_neg/n_pos ~= 2.41,
                      identical to the baseline MLP and the analog of LR's
                      class_weight="balanced". Keeps macro-F1 comparable.
  6. pooling (A)    : CLS vs mean-over-real-tokens is SELECTED ON VAL, then the
                      winner is scored once on test. An un-fine-tuned [CLS] is a
                      famously weak sentence vector, but "famously" is not
                      evidence on this corpus, so it is measured.
  7. optimizer (A)  : AdamW lr=1e-3, wd=1e-2, batch 32, early stopping patience 5
                      on val macro-F1 — byte-for-byte the baseline MLP's protocol.
                      Variant A is therefore the same experiment as the MLP with
                      the char-n-gram features swapped for frozen MARBERT ones.
  8. optimizer (B)  : AdamW lr=2e-5, wd=1e-2, batch 16, 3 epochs, 10% linear
                      warmup then linear decay. Standard BERT fine-tuning values;
                      lr=1e-3 would destroy the pretrained weights it is the whole
                      point of variant B to exploit.
  9. eval           : fixed 0.5 threshold, macro-F1, 5 seeds, mean +/- std on test.
                      Val selects the epoch/checkpoint; test is read once per seed.
 10. diagnostics    : the same four gates the baseline MLP used (pipeline mechanics,
                      analytic init-loss, tiny-batch overfit, short loss trend) run
                      before any real training, so a wiring bug cannot come back
                      wearing a plausible-looking macro-F1.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from transformers import AutoModel, AutoTokenizer

import data

ENCODER_DIR = Path("models/marbert-base")
MODELS = Path("models")
REPORTS = Path("reports")
LABELS = ["standard", "dialectal"]

MAX_LENGTH = 96          # checkpoint 2: measured zero-truncation cap
SEEDS = [0, 1, 2, 3, 4]  # checkpoint 9: same seed count as the baseline MLP

# checkpoint 7: variant A -- identical to the baseline MLP's training protocol
A_LR = 1e-3
A_WEIGHT_DECAY = 1e-2
A_BATCH_SIZE = 32
A_MAX_EPOCHS = 100
A_PATIENCE = 5

# checkpoint 8: variant B -- standard BERT fine-tuning
B_LR = 2e-5
B_WEIGHT_DECAY = 1e-2
B_BATCH_SIZE = 16
B_EPOCHS = 3
B_WARMUP_FRAC = 0.1

# Dense pretrained features are not sparse L2-normalized TF-IDF rows, so a
# randomly initialized head produces slightly larger logits at init than the
# baseline MLP did. Tolerance is looser than nn.py's 0.05 for that reason; the
# check is still doing its job (catching label/pos_weight miswiring), which
# would move the loss by far more than this.
INIT_LOSS_TOLERANCE = 0.15


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DialectClassifier(nn.Module):
    """checkpoint 4: shared by both variants; `frozen` only changes requires_grad."""

    def __init__(self, encoder, pooling: str = "mean", frozen: bool = False):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.head = nn.Linear(encoder.config.hidden_size, 1)
        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.pool(hidden, attention_mask)).squeeze(-1)


def load_encoder():
    if not (ENCODER_DIR / "pytorch_model.bin").exists():
        raise FileNotFoundError(
            f"{ENCODER_DIR}/pytorch_model.bin missing — run `python src/fetch_encoder.py` first."
        )
    return AutoModel.from_pretrained(ENCODER_DIR)


def tokenize_split(tokenizer, texts: list[str]) -> list[list[int]]:
    return tokenizer(texts, truncation=True, max_length=MAX_LENGTH)["input_ids"]


def make_batches(token_ids, labels, batch_size, shuffle, seed=0, group_by_length=False):
    """checkpoint 3: dynamic padding; batches are padded to their own max length.

    group_by_length sorts within a large shuffled buffer so batches contain
    similar-length rows — it cuts padding waste roughly in half on this corpus
    (median 9 tokens vs. a random batch's max of ~30). Used for training only
    with the buffer reshuffled every epoch; inference uses a plain global sort,
    where batch composition cannot affect anything.
    """
    n = len(token_ids)
    order = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
        if group_by_length:
            buffer = batch_size * 50
            order = np.concatenate([
                chunk[np.argsort([len(token_ids[i]) for i in chunk])]
                for chunk in (order[i:i + buffer] for i in range(0, n, buffer))
            ])
    elif group_by_length:
        order = order[np.argsort([len(token_ids[i]) for i in order])]

    starts = range(0, n, batch_size)
    if shuffle and group_by_length:
        # Length-sorted chunks would otherwise always be visited shortest-first.
        starts = list(starts)
        np.random.default_rng(seed + 1).shuffle(starts)

    for start in starts:
        idx = order[start:start + batch_size]
        rows = [token_ids[i] for i in idx]
        width = max(len(r) for r in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), width, dtype=torch.long)
        for j, row in enumerate(rows):
            input_ids[j, :len(row)] = torch.tensor(row)
            attention_mask[j, :len(row)] = 1
        y = torch.tensor(labels[idx], dtype=torch.float32) if labels is not None else None
        yield input_ids, attention_mask, y, idx


def compute_pos_weight(labels: np.ndarray) -> float:
    return (labels == 0).sum() / (labels == 1).sum()  # checkpoint 5: ~2.41


@torch.no_grad()
def predict(model, token_ids, batch_size=64) -> np.ndarray:
    model.eval()
    n = len(token_ids)
    probs = np.zeros(n, dtype=np.float32)
    for input_ids, attention_mask, _, idx in make_batches(
        token_ids, None, batch_size, shuffle=False, group_by_length=True
    ):
        logits = model(input_ids.to(device()), attention_mask.to(device()))
        probs[idx] = torch.sigmoid(logits).float().cpu().numpy()
    return (probs >= 0.5).astype(int)  # checkpoint 9: fixed threshold, as in the baselines


@torch.no_grad()
def extract_features(encoder, token_ids, pooling: str, batch_size=64) -> torch.Tensor:
    """Variant A only: the encoder never updates, so its outputs are computed once
    and reused across all 5 seeds and both pooling choices instead of being
    recomputed inside every training loop."""
    encoder.eval()
    out = None
    for input_ids, attention_mask, _, idx in make_batches(
        token_ids, None, batch_size, shuffle=False, group_by_length=True
    ):
        hidden = encoder(
            input_ids=input_ids.to(device()), attention_mask=attention_mask.to(device())
        ).last_hidden_state
        if pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.to(device()).unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if out is None:
            out = torch.zeros(len(token_ids), pooled.shape[1])
        out[idx] = pooled.float().cpu()
    return out


# ---------------------------------------------------------------------------
# checkpoint 10: the four pre-training gates, mirroring the baseline repo
# ---------------------------------------------------------------------------

def check_pipeline_mechanics(tokenizer, splits, token_ids):
    print("=== diagnostic 1: pipeline mechanics ===")
    for name in ("train", "val", "test"):
        df, ids = splits[name], token_ids[name]
        lengths = np.array([len(i) for i in ids])
        print(
            f"  {name:5s} n={len(df):5d}  dialectal_share={df['label'].mean():.3f}  "
            f"tokens: median={np.median(lengths):.0f} max={lengths.max()}  "
            f"truncated={int((lengths >= MAX_LENGTH).sum())}"
        )
        assert len(ids) == len(df), f"{name}: tokenized count != row count"
        assert lengths.max() <= MAX_LENGTH, f"{name}: sequence longer than MAX_LENGTH"

    # Round-trip on a real row: catches a tokenizer/vocab mismatch, which would
    # otherwise surface only as a quietly mediocre macro-F1. An Arabic-script row
    # coming back as mostly [UNK] means the wrong vocab.txt is loaded.
    sample = splits["train"]["text"].iloc[0]
    ids = tokenizer(sample, truncation=True, max_length=MAX_LENGTH)["input_ids"]
    unk_share = sum(i == tokenizer.unk_token_id for i in ids) / len(ids)
    print(f"  tokenizer round-trip on a train row:\n    in : {sample[:60]}")
    print(f"    out: {tokenizer.decode(ids)[:70]}")
    print(f"  [UNK] share on that row: {unk_share:.1%}")
    assert unk_share < 0.5, "over half the tokens are [UNK] -- wrong vocab for this script"
    print("  OK: every row tokenized, nothing truncated at MAX_LENGTH\n")


def check_init_loss(model, token_ids, labels, pos_weight):
    """Same analytic expectation the baseline MLP used: with pos_weight =
    n_neg/n_pos, the positive and negative loss terms collapse and the expected
    loss at near-zero logits is 2 * p_neg * ln(2)."""
    print("=== diagnostic 2: init-loss sanity check ===")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device()))
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (input_ids, attention_mask, y, _) in enumerate(
            make_batches(token_ids, labels, 32, shuffle=False)
        ):
            losses.append(
                criterion(model(input_ids.to(device()), attention_mask.to(device())), y.to(device())).item()
            )
            if i >= 20:
                break
    measured = float(np.mean(losses))
    p_neg = (labels == 0).mean()
    expected = 2 * p_neg * math.log(2)
    print(f"  measured init loss = {measured:.4f}  (mean over {len(losses)} batches)")
    print(f"  expected init loss = {expected:.4f}  (2 * p_neg * ln2, p_neg={p_neg:.4f})")
    ok = abs(measured - expected) < INIT_LOSS_TOLERANCE
    print(f"  {'OK' if ok else 'FAIL'}: within tolerance {INIT_LOSS_TOLERANCE}\n")
    if not ok:
        raise RuntimeError("init-loss check failed -- check label encoding / pos_weight wiring")


def check_tiny_batch_overfit(build_model, token_ids, labels, lr, steps=60):
    print("=== diagnostic 3: tiny-batch overfit ===")
    torch.manual_seed(0)
    model = build_model()
    idx = np.arange(16)
    rows = [token_ids[i] for i in idx]
    width = max(len(r) for r in rows)
    input_ids = torch.zeros(16, width, dtype=torch.long)
    attention_mask = torch.zeros(16, width, dtype=torch.long)
    for j, row in enumerate(rows):
        input_ids[j, :len(row)] = torch.tensor(row)
        attention_mask[j, :len(row)] = 1
    y = torch.tensor(labels[idx], dtype=torch.float32).to(device())
    input_ids, attention_mask = input_ids.to(device()), attention_mask.to(device())

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(input_ids, attention_mask), y)
        loss.backward()
        optimizer.step()
    print(f"  final loss on 16 memorized examples after {steps} steps: {loss.item():.4f}")
    ok = loss.item() < 0.05
    print(f"  {'OK' if ok else 'FAIL'}: model can memorize a tiny batch\n")
    if not ok:
        raise RuntimeError("tiny-batch overfit failed -- pipeline is broken upstream of hyperparams")


def check_loss_trend(build_model, token_ids, labels, lr, batch_size, pos_weight, n_batches=60):
    print("=== diagnostic 4: short real-run loss trend ===")
    torch.manual_seed(0)
    model = build_model()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device()))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=A_WEIGHT_DECAY
    )
    model.train()
    losses = []
    for i, (input_ids, attention_mask, y, _) in enumerate(
        make_batches(token_ids, labels, batch_size, shuffle=True, seed=0, group_by_length=True)
    ):
        optimizer.zero_grad()
        loss = criterion(model(input_ids.to(device()), attention_mask.to(device())), y.to(device()))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i + 1 >= n_batches:
            break
    first, last = np.mean(losses[:10]), np.mean(losses[-10:])
    print(f"  mean loss over first 10 batches: {first:.4f}")
    print(f"  mean loss over last  10 batches: {last:.4f}")
    ok = last < first
    print(f"  {'OK' if ok else 'FAIL'}: loss trending down over {n_batches} batches\n")
    if not ok:
        raise RuntimeError("loss not decreasing on a real run -- check lr / data")


# ---------------------------------------------------------------------------
# Variant A: frozen encoder + linear head, trained on cached features
# ---------------------------------------------------------------------------

def train_head_on_features(X_train, y_train, X_val, y_val, pos_weight, seed):
    """checkpoint 7: the baseline MLP's exact loop -- AdamW, early stopping on
    val macro-F1 with patience 5 -- over frozen MARBERT features instead of
    char-n-gram TF-IDF."""
    torch.manual_seed(seed)
    head = nn.Linear(X_train.shape[1], 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    optimizer = torch.optim.AdamW(head.parameters(), lr=A_LR, weight_decay=A_WEIGHT_DECAY)

    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    best_f1, best_state, stale = -1.0, None, 0
    n = X_train.shape[0]

    for _ in range(A_MAX_EPOCHS):
        head.train()
        perm = torch.randperm(n)
        for start in range(0, n, A_BATCH_SIZE):
            idx = perm[start:start + A_BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(head(X_train[idx]).squeeze(-1), y_train_t[idx])
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(head(X_val).squeeze(-1)) >= 0.5).long().numpy()
        val_f1 = f1_score(y_val, preds, average="macro")
        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= A_PATIENCE:
                break

    head.load_state_dict(best_state)
    return head, best_f1


def run_variant_a(encoder, token_ids, splits, pos_weight):
    print("\n" + "=" * 70)
    print("VARIANT A: frozen encoder + linear head")
    print("=" * 70)

    y_train = splits["train"]["label"].values
    y_val = splits["val"]["label"].values
    y_test = splits["test"]["label"].values

    # checkpoint 6: pooling chosen on VAL, never on test.
    pooling_scores = {}
    features = {}
    for pooling in ("cls", "mean"):
        t0 = time.time()
        feats = {
            name: extract_features(encoder, token_ids[name], pooling) for name in ("train", "val", "test")
        }
        features[pooling] = feats
        val_f1s = [
            train_head_on_features(feats["train"], y_train, feats["val"], y_val, pos_weight, seed)[1]
            for seed in SEEDS
        ]
        pooling_scores[pooling] = float(np.mean(val_f1s))
        print(
            f"  pooling={pooling:4s} val macro-F1 = {np.mean(val_f1s):.4f} "
            f"+/- {np.std(val_f1s):.4f}  ({time.time() - t0:.0f}s)"
        )

    pooling = max(pooling_scores, key=pooling_scores.get)
    print(f"\n  selected pooling on val: {pooling}\n")
    feats = features[pooling]

    test_f1s, preds_per_seed, best_head, best_val = [], [], None, -1.0
    for seed in SEEDS:
        head, val_f1 = train_head_on_features(feats["train"], y_train, feats["val"], y_val, pos_weight, seed)
        head.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(head(feats["test"]).squeeze(-1)) >= 0.5).long().numpy()
        test_f1 = f1_score(y_test, preds, average="macro")
        test_f1s.append(test_f1)
        preds_per_seed.append(preds)
        if val_f1 > best_val:
            best_val, best_head = val_f1, head
        print(f"  seed {seed}: val macro-F1={val_f1:.4f}  test macro-F1={test_f1:.4f}")

    test_f1s = np.array(test_f1s)
    print(f"\n  variant A test macro-F1: {test_f1s.mean():.4f} +/- {test_f1s.std():.4f}")

    torch.save(
        {"head": best_head.state_dict(), "pooling": pooling}, MODELS / "marbert_frozen_head.pt"
    )
    return {
        "test_f1s": test_f1s,
        "preds": preds_per_seed[int(test_f1s.argmax())],
        "pooling": pooling,
        "pooling_val_scores": pooling_scores,
    }


# ---------------------------------------------------------------------------
# Variant B: full fine-tune
# ---------------------------------------------------------------------------

def train_full_finetune(build_model, token_ids, splits, pos_weight, seed):
    torch.manual_seed(seed)
    model = build_model().to(device())
    y_train = splits["train"]["label"].values
    y_val = splits["val"]["label"].values

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=B_LR, weight_decay=B_WEIGHT_DECAY)

    steps_per_epoch = math.ceil(len(token_ids["train"]) / B_BATCH_SIZE)
    total_steps = steps_per_epoch * B_EPOCHS
    warmup = int(total_steps * B_WARMUP_FRAC)

    def lr_lambda(step):  # checkpoint 8: linear warmup then linear decay
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1, best_state = -1.0, None
    for epoch in range(B_EPOCHS):
        model.train()
        t0, running = time.time(), []
        for input_ids, attention_mask, y, _ in make_batches(
            token_ids["train"], y_train, B_BATCH_SIZE, shuffle=True, seed=seed * 100 + epoch,
            group_by_length=True,
        ):
            optimizer.zero_grad()
            loss = criterion(model(input_ids.to(device()), attention_mask.to(device())), y.to(device()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running.append(loss.item())

        val_f1 = f1_score(y_val, predict(model, token_ids["val"]), average="macro")
        # flush: a fine-tune epoch is ~20 min, and redirected stdout is block
        # buffered, so without this a multi-hour run looks identical to a hung one.
        print(
            f"    epoch {epoch + 1}/{B_EPOCHS}: train loss={np.mean(running):.4f}  "
            f"val macro-F1={val_f1:.4f}  ({(time.time() - t0) / 60:.1f} min)",
            flush=True,
        )
        if val_f1 > best_f1:  # checkpoint 9: val picks the epoch, test is never consulted
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_f1


def run_variant_b(build_model, token_ids, splits, pos_weight):
    print("\n" + "=" * 70)
    print("VARIANT B: full fine-tune")
    print("=" * 70)

    y_test = splits["test"]["label"].values
    test_f1s, preds_per_seed, best_val, best_state = [], [], -1.0, None

    for seed in SEEDS:
        print(f"  seed {seed}:", flush=True)
        model, val_f1 = train_full_finetune(build_model, token_ids, splits, pos_weight, seed)
        preds = predict(model, token_ids["test"])
        test_f1 = f1_score(y_test, preds, average="macro")
        test_f1s.append(test_f1)
        preds_per_seed.append(preds)
        print(f"    -> val macro-F1={val_f1:.4f}  test macro-F1={test_f1:.4f}", flush=True)
        if val_f1 > best_val:
            # Only the val-best seed is persisted: five 654MB checkpoints buy
            # nothing that the per-seed scores below don't already record.
            best_val = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        del model

    test_f1s = np.array(test_f1s)
    print(f"\n  variant B test macro-F1: {test_f1s.mean():.4f} +/- {test_f1s.std():.4f}")
    torch.save(best_state, MODELS / "marbert_finetuned.pt")
    return {"test_f1s": test_f1s, "preds": preds_per_seed[int(test_f1s.argmax())]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["a", "b", "both"], default="both")
    parser.add_argument("--skip-diagnostics", action="store_true")
    args = parser.parse_args()

    # Windows consoles default to cp1252, which cannot encode Arabic; every
    # diagnostic that echoes a corpus row would die on the print, not the model.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data.verify()
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    print(f"device: {device()}  threads: {torch.get_num_threads()}\n")

    splits = {name: data.load(name) for name in ("train", "val", "test")}
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_DIR)
    token_ids = {name: tokenize_split(tokenizer, splits[name]["text"].tolist()) for name in splits}
    y_train = splits["train"]["label"].values
    pos_weight = compute_pos_weight(y_train)
    print(f"pos_weight = {pos_weight:.4f}\n")

    if not args.skip_diagnostics:
        check_pipeline_mechanics(tokenizer, splits, token_ids)

        # Each gate below holds a full 654MB encoder plus, for 3 and 4, its
        # gradients and AdamW state (~2.6GB peak). They are scoped so only one
        # is resident at a time.
        probe = DialectClassifier(load_encoder(), pooling="mean", frozen=True).to(device())
        check_init_loss(probe, token_ids["train"], y_train, pos_weight)
        del probe

        check_tiny_batch_overfit(
            lambda: DialectClassifier(load_encoder(), pooling="mean", frozen=False).to(device()),
            token_ids["train"], y_train, lr=B_LR,
        )
        check_loss_trend(
            lambda: DialectClassifier(load_encoder(), pooling="mean", frozen=False).to(device()),
            token_ids["train"], y_train, lr=B_LR, batch_size=B_BATCH_SIZE, pos_weight=pos_weight,
        )
        print("all diagnostics passed -- proceeding to the real runs\n")

    results = {}
    if args.variant in ("a", "both"):
        results["marbert_frozen"] = run_variant_a(
            load_encoder().to(device()), token_ids, splits, pos_weight
        )
    if args.variant in ("b", "both"):
        results["marbert_finetuned"] = run_variant_b(
            lambda: DialectClassifier(load_encoder(), pooling="mean", frozen=False),
            token_ids, splits, pos_weight,
        )

    y_test = splits["test"]["label"].values
    preds_path = REPORTS / "preds_marbert.csv"
    frame = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame(
        {"text": splits["test"]["text"], "label": y_test}
    )
    metrics_path = REPORTS / "metrics_marbert.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    for name, res in results.items():
        frame[name] = res["preds"]
        metrics[name] = {
            "test_macro_f1_mean": float(res["test_f1s"].mean()),
            "test_macro_f1_std": float(res["test_f1s"].std()),
            "per_seed": [float(f) for f in res["test_f1s"]],
        }
        if "pooling" in res:
            metrics[name]["pooling"] = res["pooling"]
            metrics[name]["pooling_val_scores"] = res["pooling_val_scores"]
        print(f"\n{name} classification report (best-test seed, illustrative)")
        print(classification_report(y_test, res["preds"], target_names=LABELS, digits=3))
        print("confusion matrix (rows=true, cols=pred)")
        print(pd.DataFrame(confusion_matrix(y_test, res["preds"]), index=LABELS, columns=LABELS).to_string())

    frame.to_csv(preds_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {preds_path} and {metrics_path}")


if __name__ == "__main__":
    main()
