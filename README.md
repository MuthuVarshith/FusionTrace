## FusionTrace: Multimodal Media-Authenticity Assessment

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A self-hostable, enterprise-ready web application that evaluates authenticity risk in **images, audio, and sampled video frames**. It combines specialised ML models with file inspection, GradCAM heatmaps, AI-powered forensic summarization, and human audit trail logging.

---

## Key Features

- **Multimodal Media Analysis**: Inspect image (JPG, PNG, WebP), audio (WAV, MP3, M4A), and video (MP4, MOV, WebM) files.
- **High-Performance AI Models**:
  - **EfficientNetV2**: Fine-tuned visual deepfake and manipulation detection.
  - **Wav2Vec2**: Fine-tuned acoustic analysis for detecting voice synthesis and cloning artifacts.
  - **In-Memory Caching**: Pre-warmed audio & image models for fast response times.
- **GradCAM Visual Heatmap**: Generates a spatial attention heatmap overlay highlighting the exact pixel regions that triggered an image manipulation flag.
- **AI Forensic Summarization**: Integrated **Gemini 3.6 Flash** generates 2–3 sentence plain-language forensic summaries using hedged, evidence-based guardrails.
- **Standalone A4 Forensic Report PDF**: 1-page official PDF export with scan metadata, risk scores, detector breakdown, GradCAM maps, and disclaimer notes.
- **Human Review Audit Trail**: Allows reviewers to log feedback labels (`Confirmed Real`, `Confirmed Fake`, `Needs Review`) with persistent history.

---

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── ai_summary.py        # Gemini 3.6 Flash forensic summarization
│   │   ├── audio_detection.py   # Wav2Vec2 audio deepfake classification
│   │   ├── config.py            # Environment configuration & FFmpeg auto-setup
│   │   ├── image_detection.py   # EfficientNetV2 + GradCAM heatmap generation
│   │   ├── main.py              # FastAPI application routes & REST endpoints
│   │   └── service.py           # Scan orchestrator, evidence fusion & SQLite persistence
│   │
│   ├── data/
│   │   ├── artifacts/           # Stored heatmaps and extracted audio tracks
│   │   ├── uploads/             # Temporary uploaded source media
│   │   └── fusiontrace.db       # SQLite database with scan & audit history
│   │
│   ├── templates/
│   │   ├── index.html           # Main web application dashboard
│   │   └── static/
│   │       ├── css/styles.css   # Modern dark/light UI styling
│   │       └── js/script.js     # Frontend scanner, polling & A4 PDF generator
│   │
│   └── tests/                   # Unit test suite
│
├── models/
│   ├── audio_model/             # Wav2Vec2 pretrained model weights
│   └── image_model/             # EfficientNetV2 model weights (.pth)
│
└── README.md
```

---

## Deployment Guide

### Recommended Deployment Options

1. **Render.com / Railway.app (Web Service - RECOMMENDED)**
   - Connect your GitHub repo.
   - Environment variables required:
     - `GEMINI_API_KEY`: Your Google Gemini API key.
     - `FUSIONTRACE_CORS_ORIGINS`: Set to your deployed domain (e.g. `https://fusiontrace.onrender.com`).
   - Build Command: `pip install -r pyproject.toml` or `uv sync`
   - Start Command: `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`

2. **Docker / Cloud VPS (DigitalOcean / AWS EC2 / Hugging Face Spaces)**
   - Runs as a standalone container with FFmpeg pre-installed.
   - Mount a persistent volume at `backend/data/` to keep SQLite database & scan history.

---

## Local Setup & Run

### 1. Clone the Repository
```bash
git clone https://github.com/MuthuVarshith/FusionTrace.git
cd FusionTrace/backend
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

### 3. Environment Variables
Create a `.env` file in `backend/`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run Development Server
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Verification & Testing

Run the full unit test suite from `backend/`:
```bash
python -m unittest discover -s tests
```

---

## Contact & Maintainer

- **Maintainer**: Muthu Varshith
- **GitHub**: [MuthuVarshith/FusionTrace](https://github.com/MuthuVarshith/FusionTrace)
- **Email**: `muthuvarshith290@gmail.com`
