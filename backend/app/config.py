"""Centralised, environment-aware application settings.

Every environment variable the API layer reads is parsed here (detector
selection variables are read by ``app.detectors.registry``). Logging is
configured once, in this module, because it is the first app module imported.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(
    level=os.getenv("FUSIONTRACE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("fusiontrace")

with (BASE_DIR / "config.yaml").open("r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file) or {}


def _path(setting: str, default: str) -> Path:
    value = os.getenv(setting, default)
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (BASE_DIR / candidate).resolve()


def _int(setting: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(setting, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", setting, raw, default)
        return default


def _float(setting: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(setting, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", setting, raw, default)
        return default


# ── Paths ───────────────────────────────────────────────────────────────────

AUDIO_MODEL_PATH = _path("FUSIONTRACE_AUDIO_MODEL_PATH", config.get("model_path", {}).get("audio", "models/audio_model"))
IMAGE_MODEL_PATH = _path("FUSIONTRACE_IMAGE_MODEL_PATH", config.get("model_path", {}).get("image", "models/image_model/EfficientnetV2_model.pth"))
DATA_DIR = _path("FUSIONTRACE_DATA_DIR", str(config.get("data_dir") or "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
ARTIFACT_DIR = DATA_DIR / "artifacts"
DATABASE_PATH = DATA_DIR / "fusiontrace.db"

# ── Upload limits and validation ────────────────────────────────────────────

MAX_UPLOAD_MB = _int("FUSIONTRACE_MAX_UPLOAD_MB", 200, minimum=1)
MAX_IMAGE_MB = min(_int("FUSIONTRACE_MAX_IMAGE_MB", 30, minimum=1), MAX_UPLOAD_MB)
MAX_IMAGE_PIXELS = _int("FUSIONTRACE_MAX_IMAGE_PIXELS", 64_000_000, minimum=1)
MAX_AUDIO_SECONDS = _float("FUSIONTRACE_MAX_AUDIO_SECONDS", 600.0, minimum=1.0)
MAX_VIDEO_SECONDS = _float("FUSIONTRACE_MAX_VIDEO_SECONDS", 600.0, minimum=1.0)

# ── Retention, queue and processing budget ─────────────────────────────────

RETENTION_HOURS = _int("FUSIONTRACE_RETENTION_HOURS", 24, minimum=1)
MAX_CONCURRENT_SCANS = _int("FUSIONTRACE_MAX_CONCURRENT_SCANS", 1, minimum=1)
MAX_QUEUED_SCANS = _int("FUSIONTRACE_MAX_QUEUED_SCANS", 50, minimum=1)
SCAN_TIMEOUT_SECONDS = _float("FUSIONTRACE_SCAN_TIMEOUT_SECONDS", 900.0)  # 0 disables the budget
SQLITE_BUSY_TIMEOUT_MS = _int("FUSIONTRACE_SQLITE_BUSY_TIMEOUT_MS", 5000, minimum=0)

# ── Rate limits (per API key, or per client IP when auth is off; 0 disables) ─

RATE_LIMIT_WINDOW_SECONDS = _float("FUSIONTRACE_RATE_LIMIT_WINDOW_SECONDS", 60.0, minimum=1.0)
UPLOAD_RATE_LIMIT = _int("FUSIONTRACE_UPLOAD_RATE_LIMIT", 10)
FEEDBACK_RATE_LIMIT = _int("FUSIONTRACE_FEEDBACK_RATE_LIMIT", 30)

# ── HTTP ────────────────────────────────────────────────────────────────────

CORS_ORIGINS = [origin.strip() for origin in os.getenv("FUSIONTRACE_CORS_ORIGINS", "http://localhost:8000").split(",") if origin.strip()]

# ── Optional AI summary (Gemini). Only detector scores are sent, never media ─

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or None
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.6-flash"

# ── External tools ──────────────────────────────────────────────────────────
# FFmpeg/ffprobe must come from the system PATH. No bundled binaries are copied
# or prepended to PATH; video analysis and audio/video validation report
# themselves unavailable when the tools are missing.

if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    logger.warning("FFmpeg/ffprobe not found on PATH: audio/video uploads will be rejected or degraded.")

for directory in (DATA_DIR, UPLOAD_DIR, ARTIFACT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
