# FusionTrace: FastAPI app with CPU-only PyTorch detectors.
#
#   docker build -t fusiontrace .
#   docker run --rm -p 8000:8000 -v fusiontrace-models:/app/models -v fusiontrace-data:/app/data fusiontrace
#
# One port: $PORT, default 8000 (Hugging Face Spaces uses 7860).
#
# Model weights are not baked into the image. On start, the entrypoint runs
# scripts/download_models.py, which fetches the enabled detectors' weights from
# the Hugging Face Hub into /app/models and skips bundles already there. Chosen
# over a build-time download because:
#   * HF_TOKEN never ends up in an image layer or build cache;
#   * CI and contributors can build the image without credentials or ~1 GB of downloads;
#   * switching detectors (FUSIONTRACE_<KIND>_MODEL) needs a restart, not a rebuild;
#   * with /app/models on a volume, restarts are instant and work offline.
# Cost: the first start on an empty volume (and every cold start on ephemeral
# hosts such as free HF Spaces) waits for the download.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg (ships ffprobe) for audio extraction and video frame sampling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as UID 1000; use the same UID everywhere.
RUN useradd --create-home --uid 1000 app

WORKDIR /app

# Dependencies first, for layer caching. requirements.txt is generated from
# uv.lock: every package is pinned with hashes, and its --extra-index-url line
# makes torch/torchvision/torchaudio resolve to the CPU-only wheels.
COPY requirements.txt ./
RUN pip install --require-hashes -r requirements.txt

# Application code stays root-owned (read-only for the app user); only the data
# and model directories are writable.
COPY backend/ ./
COPY deploy/docker/entrypoint.sh /usr/local/bin/fusiontrace-entrypoint
RUN sed -i 's/\r$//' /usr/local/bin/fusiontrace-entrypoint \
    && chmod 0755 /usr/local/bin/fusiontrace-entrypoint \
    && mkdir -p /app/data /app/models \
    && chown app:app /app/data /app/models

ENV PORT=8000 \
    FUSIONTRACE_DATA_DIR=/app/data \
    FUSIONTRACE_MODEL_DIR=/app/models \
    FUSIONTRACE_DOWNLOAD_MODELS=1 \
    HF_HOME=/app/models/.hf-home \
    TORCH_HOME=/app/models/.torch-home \
    HF_HUB_DISABLE_TELEMETRY=1

USER app

EXPOSE 8000

# Generous start period: on an empty models volume the server only starts
# listening after the weights have been downloaded.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10m --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=8)" || exit 1

ENTRYPOINT ["fusiontrace-entrypoint"]
CMD ["serve"]
