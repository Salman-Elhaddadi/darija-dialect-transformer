"""Streamlit Community Cloud demo for MSA-vs-Darija dialect ID.

Why this exists alongside deploy/hf_space/: Hugging Face's free tier now refuses to
allocate cpu-basic compute to new accounts ("You've reached your cpu-basic quota
limit"), and ZeroGPU — which the Space README recommends — has moved behind a PRO
subscription. Streamlit Community Cloud is the remaining free host with enough RAM
to hold MARBERT, and it deploys straight from this repo, so the demo stays version-
controlled next to the training code instead of living in a separate Space repo.

It serves the FROZEN variant, which is the whole point of that variant existing: the
encoder is pulled unmodified from the Hub at boot and the only thing this repo has to
ship is the 5 KB linear head in models/marbert_frozen_head.pt. Nothing large is stored
in git, and no large upload is needed to deploy.

Memory: MARBERT is ~654 MB in fp32, plus torch and streamlit runtime, so this needs a
host with more than ~1.5 GB RAM. That is why it is not on Render's 512 MB free tier —
see deploy/render/, which serves the logistic-regression baseline instead.

Run locally:  streamlit run deploy/streamlit/app.py
"""

from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer

ENCODER = "UBC-NLP/MARBERT"
MAX_LENGTH = 96

# Resolve from this file, not the process cwd, so it works both from the repo root
# (Streamlit Cloud) and from inside deploy/streamlit/ (local runs).
HEAD_PATH = Path(__file__).resolve().parents[2] / "models" / "marbert_frozen_head.pt"


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


@st.cache_resource(show_spinner="Loading MARBERT (first boot downloads ~654 MB)...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    encoder = AutoModel.from_pretrained(ENCODER)

    saved = torch.load(HEAD_PATH, map_location="cpu", weights_only=False)
    model = DialectClassifier(encoder, pooling=saved["pooling"])
    # strict load on purpose: a shape mismatch between the checkpoint and this class
    # should crash at boot, not silently serve a randomly initialised head.
    model.head.load_state_dict(saved["head"])
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    model.eval()
    return tokenizer, model, saved["pooling"]


def has_arabic(text: str) -> bool:
    """The MAC corpus is Arabic-script only. Latin-script Darija is not a hard case for
    this model, it is a no-signal case, so it gets flagged rather than scored silently."""
    return any("؀" <= character <= "ۿ" for character in text)


@torch.no_grad()
def classify(tokenizer, model, text: str) -> float:
    batch = tokenizer(
        text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    logit = model(batch["input_ids"], batch["attention_mask"])
    return torch.sigmoid(logit).item()


st.set_page_config(page_title="Darija vs. MSA dialect ID", page_icon="\U0001f30d")

st.title("Darija vs. MSA dialect identification")
st.caption(
    "MARBERT with a frozen encoder and a trained linear head, on the MAC corpus. "
    "Test macro-F1 0.8964 ± 0.0017 over 5 seeds, against 0.8625 for a TF-IDF + "
    "logistic-regression baseline. "
    "[Full write-up and error analysis]"
    "(https://github.com/Salman-Elhaddadi/darija-dialect-transformer/blob/main/RESULTS.md)"
)

EXAMPLES = {
    "Darija — everyday": "واش نتا لاباس عليك اليوم بزاف ديال الخدمة",
    "MSA — news register": "أعلنت الوزارة عن برنامج جديد لمكافحة التصحر في المنطقة",
    "Darija — short": "شنو كاين؟ فين غادي؟",
}

choice = st.selectbox("Load an example", ["(type your own)"] + list(EXAMPLES))
default = EXAMPLES.get(choice, "")

text = st.text_area("Arabic text", value=default, height=120)

if st.button("Classify", type="primary") and text.strip():
    tokenizer, model, pooling = load_model()
    probability = classify(tokenizer, model, text.strip())

    left, right = st.columns(2)
    left.metric("Dialectal (Darija)", f"{probability:.1%}")
    right.metric("Standard (MSA)", f"{1 - probability:.1%}")
    st.progress(probability)

    if not has_arabic(text):
        st.warning(
            "No Arabic script detected. This model only ever saw Arabic-script text, "
            "so this score is not meaningful for Latin-script (Arabizi) input."
        )
    elif 0.4 < probability < 0.6:
        st.info(
            "Low confidence. Short inputs carry fewer dialect markers — error rate on "
            "the shortest quartile of the test set is 11.7% versus 7.4% on the longest."
        )

    st.caption(f"Frozen MARBERT encoder, {pooling} pooling, decision threshold 0.5.")
