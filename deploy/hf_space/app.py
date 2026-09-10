"""Hugging Face Space demo for MSA-vs-Darija dialect ID.

Why Gradio here and FastAPI on Render: Hugging Face now requires a paid plan to
create Docker Spaces, while free personal accounts can still run Gradio Spaces on
ZeroGPU. So the transformer — which cannot fit on Render's 512MB free instance at
all — is served here, and the FastAPI/Docker service in deploy/render/ serves the
LR baseline. That split is the deployment half of the tradeoff written up in
RESULTS.md, not an accident of tooling.

Gradio still exposes a callable HTTP API for this function (see the "Use via API"
link at the bottom of the Space), so this is a real endpoint, not only a UI.

  MODEL_REPO  HF repo id (or local path) holding the fine-tuned weights.
              Unset -> frozen-encoder variant: pulls UBC-NLP/MARBERT from the Hub
              and applies the ~3KB linear head committed alongside this file.
"""

import os

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

ENCODER = "UBC-NLP/MARBERT"
MODEL_REPO = os.environ.get("MODEL_REPO", "").strip()
MAX_LENGTH = 96

try:  # present only on Spaces; absent when running this file locally
    import spaces

    gpu = spaces.GPU
except ImportError:
    def gpu(fn):
        return fn


class DialectClassifier(torch.nn.Module):
    """Mirrors src/marbert.py so the saved state dict loads unchanged."""

    def __init__(self, encoder, pooling="mean"):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.head = torch.nn.Linear(encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled).squeeze(-1)


tokenizer = AutoTokenizer.from_pretrained(ENCODER)

if MODEL_REPO:
    weights = hf_hub_download(MODEL_REPO, "marbert_finetuned.pt")
    model = DialectClassifier(AutoModel.from_pretrained(ENCODER))
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    VARIANT = "MARBERT, fully fine-tuned"
else:
    saved = torch.load("marbert_frozen_head.pt", map_location="cpu")
    model = DialectClassifier(AutoModel.from_pretrained(ENCODER), pooling=saved["pooling"])
    model.head.load_state_dict(saved["head"])
    for p in model.encoder.parameters():
        p.requires_grad = False
    VARIANT = f"MARBERT, frozen encoder + linear head ({saved['pooling']} pooling)"

model.eval()

# The MAC corpus is Arabic-script only. Latin-script Darija is not a hard case for
# this model, it is a no-signal case, so it is flagged rather than scored silently.
def has_arabic(text: str) -> bool:
    return any("؀" <= c <= "ۿ" for c in text)


@gpu
def classify(text: str):
    text = (text or "").strip()
    if not text:
        return {}, "Enter some text."
    batch = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        probability = torch.sigmoid(model(batch["input_ids"], batch["attention_mask"])).item()
    note = "" if has_arabic(text) else (
        "⚠️ No Arabic script detected. This model only ever saw Arabic-script text, "
        "so this score is not meaningful for Latin-script (Arabizi) input."
    )
    return {"dialectal (Darija)": probability, "standard (MSA)": 1 - probability}, note


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, label="Arabic text", placeholder="واش نتا لاباس عليك؟"),
    outputs=[gr.Label(label="Prediction"), gr.Markdown()],
    title="Darija vs. MSA dialect identification",
    description=(
        f"Serving **{VARIANT}**, fine-tuned on the MAC corpus for binary Moroccan-Darija "
        "vs. Modern Standard Arabic classification. Full accuracy/latency/size comparison "
        "against TF-IDF+LR and a from-scratch MLP: "
        "[github.com/Salman-Elhaddadi/darija-dialect-transformer]"
        "(https://github.com/Salman-Elhaddadi/darija-dialect-transformer)"
    ),
    examples=[
        ["واش نتا لاباس عليك اليوم بزاف ديال الخدمة"],
        ["أعلنت الوزارة عن برنامج جديد لمكافحة التصحر في المنطقة"],
        ["بغيت نمشي للدار دابا حيت عييت بزاف"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
