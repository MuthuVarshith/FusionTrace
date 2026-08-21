"""
download_models.py — Run this during Render build to pull model weights from Hugging Face.

Usage:
    python scripts/download_models.py

Required environment variables (set in Render dashboard):
    HF_TOKEN   — your Hugging Face access token (for private repos)
    HF_REPO_ID — e.g. "your-username/FusionTrace-models"
"""
import os
import sys
from pathlib import Path

def download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub …")
        os.system(f"{sys.executable} -m pip install -q huggingface_hub")
        from huggingface_hub import hf_hub_download

    repo_id = os.environ.get("HF_REPO_ID")
    token   = os.environ.get("HF_TOKEN")

    if not repo_id:
        print("⚠  HF_REPO_ID not set — skipping model download (models must be present locally).")
        return

    base = Path(__file__).resolve().parent.parent  # backend/

    files = [
        ("audio_model/model.safetensors",          base / "models" / "audio_model" / "model.safetensors"),
        ("image_model/EfficientnetV2_model.pth",   base / "models" / "image_model" / "EfficientnetV2_model.pth"),
    ]

    for repo_path, local_path in files:
        if local_path.exists() and local_path.stat().st_size > 1_000_000:
            print(f"✓  {local_path.name} already present ({local_path.stat().st_size // 1_048_576} MB) — skipping.")
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"⬇  Downloading {repo_path} from {repo_id} …")
        hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            token=token,
            local_dir=str(base / "models"),
            local_dir_use_symlinks=False,
        )
        print(f"✓  Saved to {local_path}")

    print("✅  All models ready.")

if __name__ == "__main__":
    download()
