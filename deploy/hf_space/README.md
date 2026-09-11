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

## Status: configured but not schedulable on a free account

This Space is complete and correct — `app.py`, `requirements.txt` and the 5 KB head are
committed and the metadata is valid — but Hugging Face will not currently start it:

```
403  You've reached your cpu-basic quota limit, please upgrade your account,
     or pause your previous Spaces to restart this one
```

That is returned on a brand-new account whose only Space is this one and is already
paused, so it is a free-tier allocation limit rather than something the repo can fix.
**ZeroGPU is no longer a way around it** — it now requires a PRO subscription, despite
older docs describing it as free for personal accounts.

The live transformer demo is therefore hosted on Streamlit Community Cloud instead; see
`deploy/streamlit/`. This Space is kept as a working Gradio config that will run if the
quota frees up or the account is upgraded.

## Deploying this Space

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
