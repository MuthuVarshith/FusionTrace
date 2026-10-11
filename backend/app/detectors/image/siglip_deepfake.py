"""prithivMLmods/deepfake-detector-model-v1 (SigLIP base/16 @224, Apache-2.0).

Fine-tuned SigLIP vision encoder + linear classifier on mean-pooled tokens,
trained on ``prithivMLmods/OpenDeepfake-Preview``. Labelled "experimental" by
its author. Class index for "Fake" is read from ``config.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from ..base import DetectorInfo, MediaKind, ModelUnavailableError, WeightsSpec
from ._common import TorchImageDetector, softmax_prob, token_gradcam

if TYPE_CHECKING:
    import numpy as np
    import torch
    from PIL import Image

REPO_ID = "prithivMLmods/deepfake-detector-model-v1"
REVISION = "c5cb24c6a159dd2b57ca15c6a1065bd0ce8fa380"
SUBDIR = "image-siglip-deepfake-v1"


class SiglipDeepfakeDetector(TorchImageDetector):
    info: ClassVar[DetectorInfo] = DetectorInfo(
        name="siglip-deepfake-v1",
        kind=MediaKind.IMAGE,
        display_name="SigLIP deepfake detector v1 (prithivMLmods)",
        source=REPO_ID,
        license="Apache-2.0",
        version=REVISION[:7],
        description="SigLIP base/16 fine-tuned on OpenDeepfake-Preview (author marks it experimental).",
    )

    @classmethod
    def weights(cls) -> list[WeightsSpec]:
        return [WeightsSpec(repo_id=REPO_ID, local_subdir=SUBDIR, revision=REVISION,
                            allow_patterns=("model.safetensors", "config.json", "preprocessor_config.json"))]

    def _dir(self) -> Path:
        return self.model_dir / SUBDIR

    def is_available(self) -> bool:
        return (self._dir() / "model.safetensors").is_file() and (self._dir() / "config.json").is_file()

    def _build_module(self) -> torch.nn.Module:
        import torch
        from transformers import SiglipForImageClassification

        if not self.is_available():
            raise ModelUnavailableError(f"{self.info.name}: missing weights in {self._dir()}")
        labels = json.loads((self._dir() / "config.json").read_text(encoding="utf-8"))["id2label"]
        self._fake_index = next(int(i) for i, name in labels.items() if name.lower() == "fake")
        inner = SiglipForImageClassification.from_pretrained(str(self._dir()), local_files_only=True)

        class LogitsOnly(torch.nn.Module):
            def __init__(self, model: torch.nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, pixels: torch.Tensor) -> torch.Tensor:
                return self.model(pixel_values=pixels).logits

        return LogitsOnly(inner)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return transform(image)

    def _risk_from_logits(self, logits: np.ndarray) -> float:
        return softmax_prob(logits, self._fake_index)

    def _staged_forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        assert self._module is not None
        model = self._module.model
        vision = model.vision_model
        with torch.no_grad():
            h = vision.embeddings(pixels)
            for layer in vision.encoder.layers[:-1]:
                h = layer(h, None)[0]
        acts = h.detach().requires_grad_(True)
        out = vision.post_layernorm(vision.encoder.layers[-1](acts, None)[0])
        logits = model.classifier(out.mean(dim=1))[0]
        return acts, logits[self._fake_index] - logits[1 - self._fake_index]

    def _cam(self, acts: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
        return token_gradcam(acts, grads, (14, 14), prefix_tokens=0)
