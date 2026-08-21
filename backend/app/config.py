"""Centralised, environment-aware application settings."""

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fusiontrace")

# Configure FFmpeg automatically using imageio-ffmpeg if available
try:
    import imageio_ffmpeg
    import shutil
    
    ffmpeg_exe_path = imageio_ffmpeg.get_ffmpeg_exe()
    if ffmpeg_exe_path and os.path.exists(ffmpeg_exe_path):
        ffmpeg_dir = os.path.dirname(ffmpeg_exe_path)
        target_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")
        
        # Ensure a file named exactly "ffmpeg.exe" exists for libraries that expect that name
        if not os.path.exists(target_ffmpeg) and ffmpeg_exe_path != target_ffmpeg:
            try:
                shutil.copy(ffmpeg_exe_path, target_ffmpeg)
            except Exception:
                pass  # Fallback gracefully
                
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            logger.info(f"Dynamically added FFmpeg to PATH: {ffmpeg_dir}")
except ImportError:
    logger.debug("imageio-ffmpeg not found; relying on system-installed FFmpeg.")

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
with (BASE_DIR / "config.yaml").open("r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file) or {}


def _path(setting: str, default: str) -> Path:
    value = os.getenv(setting, default)
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (BASE_DIR / candidate).resolve()


AUDIO_MODEL_PATH = _path("FUSIONTRACE_AUDIO_MODEL_PATH", config.get("model_path", {}).get("audio", "models/audio_model"))
IMAGE_MODEL_PATH = _path("FUSIONTRACE_IMAGE_MODEL_PATH", config.get("model_path", {}).get("image", "models/image_model/EfficientnetV2_model.pth"))
DATA_DIR = _path("FUSIONTRACE_DATA_DIR", "data")
UPLOAD_DIR = DATA_DIR / "uploads"
ARTIFACT_DIR = DATA_DIR / "artifacts"
DATABASE_PATH = DATA_DIR / "fusiontrace.db"
MAX_UPLOAD_MB = int(os.getenv("FUSIONTRACE_MAX_UPLOAD_MB", "200"))
RETENTION_HOURS = int(os.getenv("FUSIONTRACE_RETENTION_HOURS", "24"))
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or None

for directory in (DATA_DIR, UPLOAD_DIR, ARTIFACT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
