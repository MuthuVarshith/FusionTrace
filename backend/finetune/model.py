"""Base commfor ViT, LoRA adapters on attention layers, and batched scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from app.detectors.image.commfor import CommunityForensicsDetector

# Attention projections only. A bare "proj" would also hit the patch-embedding conv.
LORA_TARGETS = r".*\.attn\.(qkv|proj)"
HEAD = "head"


def _weights_path() -> Path:
    """The default detector's weights file, under the registry's model directory."""
    from app.detectors import registry

    return registry.model_dir() / CommunityForensicsDetector.weights()[0].local_subdir / "model.safetensors"


def build_base(weights: Path | None = None) -> torch.nn.Module:
    import timm

    module = timm.create_model("vit_small_patch16_384", pretrained=False, num_classes=1)
    state = {key.removeprefix("vit."): value for key, value in load_file(str(weights or _weights_path())).items()}
    module.load_state_dict(state, strict=True)
    return module


def build_lora(rank: int, alpha: int, dropout: float, weights: Path | None = None) -> torch.nn.Module:
    """Base model with LoRA on attention and the head trainable; everything else frozen."""
    from peft import LoraConfig, get_peft_model

    base = build_base(weights)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGETS,
        modules_to_save=[HEAD],
        bias="none",
    )
    return get_peft_model(base, config)


def load_adapter(adapter_dir: Path, weights: Path | None = None) -> torch.nn.Module:
    from peft import PeftModel

    return PeftModel.from_pretrained(build_base(weights), str(adapter_dir)).eval()


def trainable_summary(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


@torch.no_grad()
def score(model: torch.nn.Module, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """P(fake) for each batch item, and the labels, in loader order."""
    model.eval()
    risks, labels = [], []
    for pixels, targets in loader:
        logits = model(pixels.to(device)).reshape(-1)
        risks.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(risks), np.concatenate(labels).astype(int)
