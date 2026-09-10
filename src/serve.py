"""FastAPI endpoint for MSA-vs-Darija dialect ID.

Which model is served is an environment variable, not a hardcoded winner: the
comparison in RESULTS.md exists precisely because the most accurate model is not
automatically the right one to deploy.

  MODEL    lr | marbert_frozen | marbert_finetuned   (default: lr)
  ENCODER  local dir or HF repo id for the MARBERT weights
           (default: models/marbert-base, falls back to UBC-NLP/MARBERT)

torch and transformers are imported lazily, inside the MARBERT branch only. That
is a deployment constraint, not tidiness: this app runs on a 512MB Render free
instance when MODEL=lr, and importing torch there costs ~200MB of resident memory
for a model that is a scikit-learn pipeline and never touches a tensor. The
MARBERT variants cannot fit on that instance at all (654MB of fp32 weights) and
are served from a Hugging Face Space instead — see deploy/.
"""

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODELS = Path("models")
MODEL_NAME = os.environ.get("MODEL", "lr")
ENCODER = os.environ.get("ENCODER", "models/marbert-base")
MAX_LENGTH = 96
LABELS = {0: "standard", 1: "dialectal"}

# The MAC corpus is Arabic-script only, so every model here has only ever seen
# Arabic script. Latin-script Darija ("wach nta labas") is not a hard case for
# them, it is a no-signal case, and a probability returned for it would be noise
# wearing a confident-looking number. Flagged rather than silently scored.
ARABIC = re.compile(r"[؀-ۿݐ-ݿ]")

state: dict = {}


class Request(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def build_lr_scorer():
    model = joblib.load(MODELS / "baseline/baseline_lr.joblib")
    return lambda text: float(model.predict_proba([text])[0, 1])


def build_marbert_scorer():
    import torch
    from transformers import AutoModel, AutoTokenizer

    from marbert import DialectClassifier

    source = ENCODER if Path(ENCODER).exists() else "UBC-NLP/MARBERT"
    tokenizer = AutoTokenizer.from_pretrained(source)

    if MODEL_NAME == "marbert_frozen":
        saved = torch.load(MODELS / "marbert_frozen_head.pt", map_location="cpu")
        model = DialectClassifier(AutoModel.from_pretrained(source), pooling=saved["pooling"], frozen=True)
        model.head.load_state_dict(saved["head"])
    else:
        model = DialectClassifier(AutoModel.from_pretrained(source), pooling="mean", frozen=False)
        model.load_state_dict(torch.load(MODELS / "marbert_finetuned.pt", map_location="cpu"))
    model.eval()

    def score(text: str) -> float:
        batch = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            logit = model(batch["input_ids"], batch["attention_mask"])
        return float(torch.sigmoid(logit).item())

    return score


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["score"] = build_lr_scorer() if MODEL_NAME == "lr" else build_marbert_scorer()
    yield
    state.clear()


app = FastAPI(title="Darija dialect ID", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict")
def predict(request: Request):
    probability = state["score"](request.text)
    response = {
        "label": LABELS[int(probability >= 0.5)],
        "probability_dialectal": round(probability, 4),
        "model": MODEL_NAME,
    }
    if not ARABIC.search(request.text):
        response["warning"] = (
            "No Arabic script detected. This model was trained on Arabic-script text only; "
            "the score above is not meaningful for Latin-script (Arabizi) input."
        )
    return response


@app.get("/", response_class=HTMLResponse)
def index():
    return f"""<!doctype html>
<title>Darija dialect ID</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem; }}
 textarea {{ width: 100%; font-size: 1rem; padding: .5rem; }}
 button {{ font-size: 1rem; padding: .4rem 1rem; margin-top: .5rem; cursor: pointer; }}
 #out {{ margin-top: 1rem; padding: 1rem; background: #f4f4f5; border-radius: .4rem; white-space: pre-wrap; }}
</style>
<h1>Darija vs. MSA</h1>
<p>Serving <code>{MODEL_NAME}</code>. Paste Arabic-script text.</p>
<textarea id="t" rows="4">واش نتا لاباس عليك اليوم</textarea>
<button onclick="go()">Classify</button>
<div id="out"></div>
<script>
async function go() {{
  const r = await fetch('/predict', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{text: document.getElementById('t').value}})
  }});
  document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
}}
</script>
"""
