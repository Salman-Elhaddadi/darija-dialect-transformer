---
title: Darija vs MSA Dialect ID
emoji: 🇲🇦
colorFrom: green
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
models:
  - UBC-NLP/MARBERT
---

# Darija vs. MSA dialect identification

Binary Moroccan-Darija vs. Modern Standard Arabic classification, fine-tuned from
[UBC-NLP/MARBERT](https://huggingface.co/UBC-NLP/MARBERT) on the MAC corpus.

Full write-up, including the accuracy/latency/size comparison against a
TF-IDF+logistic-regression baseline and a from-scratch PyTorch MLP:
[github.com/Salman-Elhaddadi/darija-dialect-transformer](https://github.com/Salman-Elhaddadi/darija-dialect-transformer)

## Deploying this Space

Set the hardware to **ZeroGPU** in Space settings (free personal accounts can run
up to 2 Gradio Spaces on ZeroGPU; Docker Spaces now require a paid plan).

By default the Space serves the **frozen-encoder** variant: it pulls MARBERT from
the Hub at startup and applies the ~3KB linear head committed next to `app.py`, so
no large weight upload is needed. To serve the **fully fine-tuned** variant
instead, upload `marbert_finetuned.pt` to a model repo and set the `MODEL_REPO`
variable in Space settings to that repo id.

```
huggingface-cli upload <user>/darija-marbert models/marbert_finetuned.pt
```

That upload is ~654MB. On a slow uplink it is the single most expensive step in
this project — see the repo's README for why that cost, not accuracy alone,
decided what gets deployed where.
