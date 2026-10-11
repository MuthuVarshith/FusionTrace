<div align="center">

# 🔍 FusionTrace

### Multimodal Media Authenticity Assessment

*Evidence-based risk signals for images, audio and video, with the reasoning shown.*

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking%20%26%20registry-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0.0-6366F1?style=for-the-badge)]()

[![Demo](https://img.shields.io/badge/▶%20Demo-GIF-FF6B6B?style=flat-square)](resources/images/FusionTrace_Demo.gif)
[![Sample report](https://img.shields.io/badge/📄%20Sample%20Report-PDF-4A90E2?style=flat-square)](resources/docs/1.pdf)

</div>

---

## Overview

FusionTrace scores an uploaded image, audio clip or video for synthetic-media risk. It runs
several independent signals, fuses them into a weighted risk score, and shows every signal, the
model behind it, and its status. All inference runs on your own machine; uploaded media is never
sent anywhere.

The output is a **risk signal for review**, not proof that media is real or fake.

## Features

- **Image analysis** with a Vision Transformer detector and an attention heatmap showing which
  regions drove the result.
- **Audio analysis** over the whole clip, using sliding 4-second windows and a top-window
  aggregate, so a short spliced segment is not diluted away.
- **Video analysis** combining per-frame image scores, the soundtrack's audio score, and a
  frame-to-frame consistency signal.
- **Evidence report and PDF export** with per-signal model names, weights, statuses and reasons.
- **Human review trail:** reviewers can label a scan (confirmed real, confirmed fake, needs review)
  with notes, kept as an audit history.
- **Optional AI summary** (Google Gemini), off by default, sent detector outputs only.
- **Optional API keys, rate limits and a bounded job queue** for shared deployments.
- **Swappable models** through one detector interface and a config file.
- **LoRA fine-tuning pipeline** for the image model, with MLflow tracking and a `candidate`
  registry alias (see [LoRA fine-tuning](#lora-fine-tuning)).

## Models and accuracy

Each signal is evaluated on a small labelled set. These are quick checks, not benchmarks: small
samples, one run, CPU, decision threshold 0.5.

| Input | Model | Test set | Accuracy | AUC |
|---|---|---|---|---|
| **Image** | Community Forensics ViT-S/16-384 (MIT) | 90 real + 90 fake AI-generated faces | **98.9%** | 1.000 |
| **Audio** | Wav2Vec2-base, FusionTrace model | In-the-Wild: 50 real + 50 fake clips | **87.0%** | 0.916 |
| **Audio** | Wav2Vec2-base, FusionTrace model | Local test set: 9 real + 6 fake WAVs | 80.0% | 0.907 |
| **Video (temporal)** | Frame-consistency heuristic | 10 real + 10 fake clips | — (not scored) | 0.50 |

**Scope of the image result:** the 98.9% is measured on AI-generated faces from a generator that
is part of the model's training data. The model has not been validated on face swaps or other
manipulation types, and that is not covered by these numbers.

**Audio result:** the 87.0% is accuracy at threshold 0.5. Real recall is 100% and fake recall is
74%. The In-the-Wild sample is public and small.

**Video:** the temporal signal did not separate real from fake on its check (AUC 0.50), so it is
computed and shown in the report with fusion weight 0. It does not change the score.

Model choice, every candidate considered, licences, and the full quick-check tables:
[docs/model-choice.md](docs/model-choice.md).

## How a scan works

```
Upload ─► validate (content checks, size and duration limits)
        ─► queue (bounded worker pool)
        ─► route by media type
             image : image model                       (weight 0.9)
             audio : audio model, sliding 4 s windows  (weight 0.9)
             video : image model on sampled frames     (weight 0.8)
                   + audio model on the soundtrack     (weight 0.9)
                   + temporal consistency heuristic    (weight 0, reported only)
             all   : file inspection (SHA-256, MIME, size; weight 0.1, risk 0)
        ─► fuse: weighted mean of signals that ran
        ─► report, heatmap (image), optional AI summary, PDF
```

**Fusion.** `risk = Σ(risk × weight) / Σ(weight)` over signals that ran. The rating is `high`
at 70 or above, `medium` at 40 or above, and `low` otherwise. If no signal runs, the scan fails
instead of reporting "low".

## Architecture

| Layer | Implementation |
|---|---|
| API | FastAPI, background job queue with a worker cap, SQLite in WAL mode |
| Detectors | One interface (`backend/app/detectors/base.py`), selected per media type by `config.yaml` |
| Image model | timm ViT-S/16-384, sigmoid output, Grad-CAM-style heatmap |
| Audio model | Wav2Vec2-base sequence classifier, 16 kHz, sliding windows, top-3 mean |
| Video | FFmpeg frame sampling, per-frame image scores, soundtrack, temporal heuristic |
| Validation | Pillow content checks for images, `ffprobe` stream and duration checks for audio and video |
| Frontend | Vanilla JavaScript, no framework |
| Packaging | One pinned manifest (`pyproject.toml`) and `uv.lock`; Docker image |

## MLOps

- **Experiment tracking:** MLflow logs parameters, per-epoch metrics, thresholds, and the data
  version (a content hash of the manifest) for each training run.
- **Model registry:** versions carry the aliases `candidate` and `production`. Promotion is a
  separate, deliberate step; nothing is promoted automatically.
- **Export and optimisation:** ONNX export with a numerical parity check against the PyTorch model,
  INT8 quantisation (dynamic and static), and a benchmark for latency, memory and model size.
- **Evaluation tooling:** AUC, precision, recall, F1 and confusion matrices per detector, on
  labelled folders of images, audio or video.
- **Reproducible builds:** exact pins in `pyproject.toml`, a `uv.lock` lock file, and a hash-pinned
  `requirements.txt` generated from it.
- **Continuous integration:** GitHub Actions checks the lock file, runs ruff and pytest, and builds
  the Docker image.
- **Data governance:** every dataset's licence and source is recorded, and training refuses any
  split where one group appears in two splits.

Commands: [docs/mlops.md](docs/mlops.md).

## LoRA fine-tuning

LoRA (Low-Rank Adaptation) trains a small set of added weights while the pretrained model stays
frozen. FusionTrace applies it to the image model's attention layers, with rank 8, and also trains
the output head. That is 221,569 trainable parameters out of 22 million, about 1%. The adapter is
about 1 MB, and training takes roughly 15 minutes per epoch on a T4 GPU.

- **Data:** FairFace real faces (CC BY 4.0) and face-swap fakes from `bitmind/face-swap`
  (Apache-2.0). Splits are grouped by perceptual hash, and every image is normalised to 256×256
  JPEG.
- **Threshold:** chosen on the validation split only, never on test data.
- **Status:** the adapter is trained and registered with the `candidate` alias. It is **off by
  default** and is not promoted to `production`. Its held-out evaluation is still in progress, and
  its results are not reported here.
- **Enable it:** set `detectors.image.options.lora.enabled: true` in `backend/config.yaml`.

Full steps for Google Colab and local CPU: [docs/how-to-finetune.md](docs/how-to-finetune.md).

## Getting started

### Local

```bash
git clone https://github.com/MuthuVarshith/FusionTrace.git
cd FusionTrace
uv sync                                    # or: pip install -r requirements.txt
cp backend/.env.example backend/.env       # optional settings

cd backend
python scripts/download_models.py          # fetches the default model weights
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000>. `GET /health` lists each detector, its model, its weight, and whether
it is available and loaded.

> The audio model's weights are FusionTrace's own and are not on a public Hub repo. Set
> `HF_REPO_ID` to your mirror and `HF_TOKEN` if it is private. Without them, audio scans report the
> audio model as unavailable.

### Docker

```bash
cp backend/.env.example backend/.env
docker compose up --build                  # http://localhost:8000
```

The container downloads missing weights into a volume on start, then serves on `$PORT`. Weights are
not baked into the image.

### Hugging Face Spaces

Step-by-step guide: [docs/deploy-hf-spaces.md](docs/deploy-hf-spaces.md).

## Configuration

Detectors are set in [`backend/config.yaml`](backend/config.yaml). Each setting can also come from
the environment:

| Variable | Effect |
|---|---|
| `FUSIONTRACE_<IMAGE\|AUDIO\|VIDEO>_MODEL` | Detector to use (registry key) |
| `FUSIONTRACE_<KIND>_ENABLED` | `false` switches a signal off |
| `FUSIONTRACE_<KIND>_WEIGHT` | Fusion weight |
| `FUSIONTRACE_MODEL_DIR` | Where model weights live |
| `FUSIONTRACE_DEVICE` | `auto`, `cpu` or `cuda` |
| `FUSIONTRACE_API_KEYS` | Comma-separated keys; when set, `/api/*` requires `X-API-Key` and scans are visible only to their key |
| `FUSIONTRACE_UPLOAD_RATE_LIMIT`, `FUSIONTRACE_FEEDBACK_RATE_LIMIT` | Requests per window; 429 when exceeded |
| `FUSIONTRACE_MAX_CONCURRENT_SCANS` | Worker pool size (default 1) |
| `FUSIONTRACE_MAX_UPLOAD_MB`, `_MAX_AUDIO_SECONDS`, `_MAX_VIDEO_SECONDS`, `_MAX_IMAGE_PIXELS` | Upload limits |
| `FUSIONTRACE_RETENTION_HOURS` | How long uploads and heatmaps are kept |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Enable the optional AI summary |

[`backend/.env.example`](backend/.env.example) lists every variable, with defaults.

## REST API

Interactive docs at `/docs` when the server is running.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Detector status and whether summaries are available |
| `POST` | `/api/scans` | Upload a file (multipart: `file`, optional `generate_ai_summary`); returns `202` and the scan id |
| `GET` | `/api/scans/{id}` | Status (`queued`, `processing`, `completed`, `failed`) and the report |
| `GET` | `/api/scans` | Recent scans |
| `POST` | `/api/scans/{id}/feedback` | Reviewer label and notes |
| `GET` | `/api/scans/{id}/heatmap` | Image attention heatmap, once ready |

## Evaluation

```bash
cd backend
python scripts/evaluate_models.py          # AUC, precision, recall, F1, confusion matrix
python scripts/check_image_models.py       # quick image check
python scripts/check_audio_models.py       # quick audio check
```

## Privacy and security

- Inference is local. Media files are never uploaded to a third party.
- Uploads are checked by content, not by file extension.
- Optional API keys scope each scan to the key that created it. Keys are never stored, only hashed.
- Rate limits on uploads and feedback; a bounded queue.
- Working files are deleted after each scan; uploads and heatmaps are kept for the retention period.
- The AI summary is off by default and sends detector outputs only, never the media.

## Development

```bash
uv sync --extra dev
FUSIONTRACE_WARMUP=0 pytest                # backend/tests
ruff check .
```

## Project structure

```
FusionTrace/
├── backend/
│   ├── app/
│   │   ├── main.py            # routes, auth, rate limits
│   │   ├── service.py         # scan lifecycle, fusion, persistence
│   │   ├── pipeline.py        # media type → signals
│   │   ├── detectors/         # shared interface, registry, image/audio/video models
│   │   └── ai_summary.py      # optional Gemini summary
│   ├── finetune/              # LoRA dataset preparation, model, data splits
│   ├── mlops/                 # MLflow tracking and registry helpers
│   ├── scripts/               # downloads, evaluation, export, quantisation, fine-tuning
│   ├── templates/             # web UI
│   └── tests/
├── deploy/                    # Docker entrypoint, Hugging Face Spaces kit
├── docs/                      # model choice, fine-tuning, MLOps, deployment, upgrade notes
├── pyproject.toml, uv.lock    # pinned dependencies
└── Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Documentation

- [Model choice and candidates](docs/model-choice.md)
- [MLOps commands](docs/mlops.md)
- [LoRA fine-tuning](docs/how-to-finetune.md)
- [Deploying to Hugging Face Spaces](docs/deploy-hf-spaces.md)
- [Upgrade notes](docs/upgrade-notes.md)

## License

MIT. Third-party model weights and datasets keep their own licences; see
[docs/model-choice.md](docs/model-choice.md).
