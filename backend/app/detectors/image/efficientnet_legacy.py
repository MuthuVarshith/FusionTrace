"""The v1 FusionTrace image model (EfficientNetV2-L + custom head), kept as a non-default option.

It scores well on its own training distribution (140k-real-and-fake-faces +
real-vs-ai-generated-faces, Kaggle) but does not generalise to other data; see
``docs/model-choice-image.md``.

Options:

* ``threshold`` (float, default 0.5).
* ``normalization``: ``"imagenet"`` (default; mean/std 0.485,0.456,0.406 /
  0.229,0.224,0.225, as in the training notebook) or ``"half"`` (mean = std =
  0.5, what v1 inference used — a train/inference mismatch kept selectable for
  comparison only). Read from the instance options (config), not per call.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ..base import DetectorInfo, MediaKind, ModelUnavailableError, WeightsSpec
from ._common import IMAGENET_MEAN, IMAGENET_STD, TorchImageDetector, conv_gradcam, sigmoid

if TYPE_CHECKING:
    import numpy as np
    import torch
    from PIL import Image

WEIGHTS_FILE = "image_model/EfficientnetV2_model.pth"
_NORMALIZATION = {"imagenet": (IMAGENET_MEAN, IMAGENET_STD), "half": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))}
_MIN_WEIGHTS_BYTES = 1_000_000  # anything smaller is a Git LFS pointer, not weights


class EfficientNetV2LegacyDetector(TorchImageDetector):
    info: ClassVar[DetectorInfo] = DetectorInfo(
        name="efficientnetv2-legacy",
        kind=MediaKind.IMAGE,
        display_name="EfficientNetV2-L (FusionTrace legacy)",
        source="local",
        license="MIT",
        version="v1",
        description=(
            "FusionTrace v1 image model: tf_efficientnetv2_l with a custom head, fine-tuned on Kaggle "
            "140k-real-and-fake-faces and real-vs-ai-generated-faces. Does not generalise beyond that data."
        ),
    )

    @classmethod
    def weights(cls) -> list[WeightsSpec]:
        # No public repo: only downloadable from the private mirror named by HF_REPO_ID
        # (same layout as v1's scripts/download_models.py). Lands at <model_dir>/image_model/...
        return [WeightsSpec(repo_id="", local_subdir=".", allow_patterns=(WEIGHTS_FILE,), repo_id_env="HF_REPO_ID")]

    def weights_path(self) -> Path:
        """``FUSIONTRACE_IMAGE_MODEL_PATH`` / config ``model_path.image`` first, then ``<model_dir>/image_model/...``."""
        from ...config import IMAGE_MODEL_PATH

        for candidate in (Path(IMAGE_MODEL_PATH), self.model_dir / WEIGHTS_FILE):
            if candidate.is_file() and candidate.stat().st_size >= _MIN_WEIGHTS_BYTES:
                return candidate
        return Path(IMAGE_MODEL_PATH)

    def is_available(self) -> bool:
        path = self.weights_path()
        return path.is_file() and path.stat().st_size >= _MIN_WEIGHTS_BYTES

    def _build_module(self) -> torch.nn.Module:
        import torch

        from ._legacy_net import EfficientNetV2

        path = self.weights_path()
        if not self.is_available():
            raise ModelUnavailableError(f"{self.info.name}: weights missing or an LFS pointer at {path}")
        module = EfficientNetV2()
        state = torch.load(path, map_location="cpu", weights_only=True)
        module.load_state_dict(state)
        return module

    def _normalization(self) -> str:
        # Instance option only (set from config at construction): preprocessing must be
        # identical for analyze, explain and export.
        choice = str(self.options.get("normalization", "imagenet")).lower()
        if choice not in _NORMALIZATION:
            raise ValueError(f"normalization must be one of {sorted(_NORMALIZATION)}, got {choice!r}")
        return choice

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        from torchvision import transforms

        mean, std = _NORMALIZATION[self._normalization()]
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)])
        return transform(image)

    def _details(self, pixels: torch.Tensor) -> dict[str, Any]:
        return {**super()._details(pixels), "normalization": self._normalization()}

    def _risk_from_logits(self, logits: np.ndarray) -> float:
        return sigmoid(float(logits[0]))

    def _staged_forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        assert self._module is not None
        with torch.no_grad():
            acts = self._module.last_block_activation(pixels)
        acts = acts.detach().requires_grad_(True)
        return acts, self._module.logit_from_last_block(acts).sum()

    def _cam(self, acts: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
        return conv_gradcam(acts, grads)
