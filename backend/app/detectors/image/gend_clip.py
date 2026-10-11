"""GenD CLIP ViT-L/14 (Yermakov, Cech & Matas, WACV 2026; MIT).

UnivFD-style: CLIP ViT-L/14 image encoder (only LayerNorms tuned), CLS
feature L2-normalised, linear probe to 2 classes (index 1 = fake). Trained on
FaceForensics++ face crops (RetinaFace, scale 1.3). FusionTrace runs it on the
whole image: no face detector is bundled, so off-centre or small faces score
worse than in the paper.

Weights ``yermandy/GenD_CLIP_L_14`` are rebuilt with stock ``transformers``
CLIP classes (the repo's ``modeling_gend.py`` is not executed).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from ..base import DetectorInfo, MediaKind, ModelUnavailableError, WeightsSpec
from ._common import (
    CLIP_MEAN,
    CLIP_STD,
    TorchImageDetector,
    softmax_prob,
    token_gradcam,
)

if TYPE_CHECKING:
    import numpy as np
    import torch
    from PIL import Image

REPO_ID = "yermandy/GenD_CLIP_L_14"
REVISION = "891ce014a0308386c4d7d25b3dcf436a22db5504"
SUBDIR = "image-gend-clip-l14"
FAKE_INDEX = 1


def _build_gend() -> torch.nn.Module:
    import torch
    import torch.nn.functional as F
    from transformers import CLIPVisionConfig, CLIPVisionModel

    class GenD(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            config = CLIPVisionConfig(
                hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16,
                image_size=224, patch_size=14, projection_dim=768, hidden_act="quick_gelu", layer_norm_eps=1e-5,
            )
            self.vision = CLIPVisionModel(config).vision_model
            self.head = torch.nn.Linear(1024, 2)

        def logits_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
            return self.head(F.normalize(pooled, p=2, dim=1))

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            return self.logits_from_pooled(self.vision(pixel_values=pixels).pooler_output)

    return GenD()


class GenDClipDetector(TorchImageDetector):
    info: ClassVar[DetectorInfo] = DetectorInfo(
        name="gend-clip-l14",
        kind=MediaKind.IMAGE,
        display_name="GenD CLIP ViT-L/14",
        source=REPO_ID,
        license="MIT",
        version=REVISION[:7],
        description="CLIP ViT-L/14 features + linear probe, trained on FaceForensics++ face crops (WACV 2026).",
    )

    @classmethod
    def weights(cls) -> list[WeightsSpec]:
        return [WeightsSpec(repo_id=REPO_ID, local_subdir=SUBDIR, revision=REVISION, allow_patterns=("model.safetensors", "config.json"))]

    def _weights_file(self) -> Path:
        return self.model_dir / SUBDIR / "model.safetensors"

    def is_available(self) -> bool:
        return self._weights_file().is_file()

    def _build_module(self) -> torch.nn.Module:
        from safetensors.torch import load_file

        if not self.is_available():
            raise ModelUnavailableError(f"{self.info.name}: missing {self._weights_file()}")
        module = _build_gend()
        raw = load_file(str(self._weights_file()))
        state = {}
        for key, value in raw.items():
            if key.startswith("feature_extractor.vision_model."):
                state["vision." + key.removeprefix("feature_extractor.vision_model.")] = value
            elif key.startswith("model.linear."):
                state["head." + key.removeprefix("model.linear.")] = value
        missing, unexpected = module.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.endswith("position_ids")]
        if missing or unexpected:
            raise ModelUnavailableError(f"{self.info.name}: weight mismatch, missing={missing[:5]} unexpected={unexpected[:5]}")
        return module

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
        return transform(image)

    def _risk_from_logits(self, logits: np.ndarray) -> float:
        return softmax_prob(logits, FAKE_INDEX)

    def _staged_forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        module = self._module
        assert module is not None
        vision = module.vision
        with torch.no_grad():
            h = vision.pre_layrnorm(vision.embeddings(pixels))
            for layer in vision.encoder.layers[:-1]:
                h = layer(h, None, None)[0]
        acts = h.detach().requires_grad_(True)
        out = vision.encoder.layers[-1](acts, None, None)[0]
        logits = module.logits_from_pooled(vision.post_layernorm(out[:, 0, :]))[0]
        return acts, logits[FAKE_INDEX] - logits[1 - FAKE_INDEX]

    def _cam(self, acts: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
        return token_gradcam(acts, grads, (16, 16), prefix_tokens=1)
