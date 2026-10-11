"""Community Forensics ViT-S/16 @384 (Park & Owens, CVPR 2025; MIT).

Trained on 2.7M images from 4,803 generators. Official weights
``OwensLab/commfor-model-384`` (timm ``vit_small_patch16_384``, one logit,
sigmoid = P(fake)). Preprocessing follows the official test transform:
resize shorter side to 440, centre-crop 384, ImageNet normalisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ..base import AnalysisContext, DetectorInfo, MediaKind, ModelUnavailableError, WeightsSpec
from ._common import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TorchImageDetector,
    sigmoid,
    token_gradcam,
)
from .lora import LoraSettings

if TYPE_CHECKING:
    import numpy as np
    import torch
    from PIL import Image

REPO_ID = "OwensLab/commfor-model-384"
REVISION = "6076002bf0d9dd37537f965ee2f06f826c333b61"
SUBDIR = "image-commfor-vit-s384"


class CommunityForensicsDetector(TorchImageDetector):
    _lora: Any = None  # set by _build_module when a LoRA adapter is enabled
    info: ClassVar[DetectorInfo] = DetectorInfo(
        name="commfor-vit-s384",
        kind=MediaKind.IMAGE,
        display_name="Community Forensics ViT-S/16-384",
        source=REPO_ID,
        license="MIT",
        version=REVISION[:7],
        description="ViT-Small trained on 2.7M real and AI-generated images from 4,803 generators (Park & Owens, CVPR 2025).",
    )

    @classmethod
    def weights(cls) -> list[WeightsSpec]:
        return [WeightsSpec(repo_id=REPO_ID, local_subdir=SUBDIR, revision=REVISION, allow_patterns=("model.safetensors", "config.json"))]

    def _weights_file(self) -> Path:
        return self.model_dir / SUBDIR / "model.safetensors"

    def is_available(self) -> bool:
        return self._weights_file().is_file()

    def _build_module(self) -> torch.nn.Module:
        import timm
        from safetensors.torch import load_file

        if not self.is_available():
            raise ModelUnavailableError(f"{self.info.name}: missing {self._weights_file()}")
        module = timm.create_model("vit_small_patch16_384", pretrained=False, num_classes=1)
        state = {key.removeprefix("vit."): value for key, value in load_file(str(self._weights_file())).items()}
        module.load_state_dict(state, strict=True)
        self._lora = None
        settings = LoraSettings.from_options(self.options)
        if settings.enabled:
            from .lora import merged_module, resolve

            self._lora = resolve(settings, cache_dir=self.model_dir / "lora")
            module = merged_module(module, self._lora.adapter_dir)
        return module

    def threshold(self, ctx: AnalysisContext | None) -> float:
        # An adapter brings the threshold chosen on its validation split, unless one is set explicitly.
        explicit = "threshold" in self.options or (ctx is not None and "threshold" in ctx.options)
        if self._lora is not None and not explicit:
            return self._lora.threshold
        return super().threshold(ctx)

    def _details(self, pixels: torch.Tensor) -> dict[str, Any]:
        details = super()._details(pixels)
        if self._lora is not None:
            details["lora"] = self._lora.provenance
        return details

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        from torchvision import transforms

        transform = transforms.Compose(
            [transforms.Resize(440), transforms.CenterCrop(384), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        )
        return transform(image)

    def _risk_from_logits(self, logits: np.ndarray) -> float:
        return sigmoid(float(logits[0]))

    def _staged_forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        vit = self._module
        assert vit is not None
        with torch.no_grad():
            x = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(pixels))))
            for block in vit.blocks[:-1]:
                x = block(x)
        acts = x.detach().requires_grad_(True)  # residual stream entering the last block
        out = vit.forward_head(vit.norm(vit.blocks[-1](acts)))
        return acts, out.sum()

    def _cam(self, acts: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
        return token_gradcam(acts, grads, (24, 24), prefix_tokens=1)
