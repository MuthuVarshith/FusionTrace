# HuggingFace Spaces — Docker SDK
# Free tier: 16 GB RAM, 2 vCPUs — perfect for ML models
#
# Required Space Secrets (set in HF Space → Settings → Variables and secrets):
#   GEMINI_API_KEY   — your Google Gemini API key
#   HF_REPO_ID       — e.g. your-username/FusionTrace-models  (where model weights live)
#   HF_TOKEN         — HuggingFace read token  (if the model repo is private)

FROM python:3.11-slim

# System dependencies — ffmpeg for audio/video processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user (HuggingFace Spaces runs as UID 1000)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend
COPY backend/ .

# Pre-create runtime directories with correct ownership
RUN mkdir -p data/uploads data/artifacts models/audio_model models/image_model && \
    chown -R appuser:appuser /app

USER appuser

# Download model weights at BUILD time so startup is instant.
# HF_REPO_ID and HF_TOKEN are passed as build args (Space secrets).
ARG HF_REPO_ID=""
ARG HF_TOKEN=""
ENV HF_REPO_ID=${HF_REPO_ID} \
    HF_TOKEN=${HF_TOKEN}

RUN python scripts/download_models.py

# HuggingFace Spaces always exposes port 7860
EXPOSE 7860

# Runtime env vars (also set in Space secrets panel)
ENV FUSIONTRACE_CORS_ORIGINS="*" \
    FUSIONTRACE_MAX_UPLOAD_MB=200

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
