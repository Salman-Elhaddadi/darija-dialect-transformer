# LR baseline serving image, sized for Render's 512MB free instance.
#
# Build context is the repo root:
#   docker build -f deploy/render/Dockerfile -t darija-lr .
#   docker run -p 8000:8000 darija-lr
#
# The MARBERT variants are NOT servable from this image. 654MB of fp32 weights
# does not fit in 512MB of RAM, and Render's free instance gets 0.1 CPU, which
# would put a BERT forward pass in the tens of seconds. That is the deployment
# half of the accuracy/cost tradeoff in RESULTS.md, not an oversight — the
# transformer is served from a Hugging Face Space instead.

FROM python:3.13-slim

WORKDIR /app

# Matches the interpreter the joblib artifact was pickled under, so the pickle
# protocol and scikit-learn version both line up at load time.
COPY deploy/render/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/serve.py src/
COPY models/baseline/baseline_lr.joblib models/baseline/

ENV MODEL=lr \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Render injects $PORT; the default keeps `docker run` working locally.
CMD ["sh", "-c", "uvicorn serve:app --host 0.0.0.0 --port ${PORT:-8000}"]
