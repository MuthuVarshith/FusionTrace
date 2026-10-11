"""Network definition of the v1 FusionTrace image model (moved from ``app/image_detection.py``).

Kept in its own module so importing the detector registry does not import
torch/timm. The layer names must stay exactly as trained: the state dict in
``EfficientnetV2_model.pth`` uses the ``base_model.*`` / ``classifier.*`` keys.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm import create_model


class EfficientNetV2(nn.Module):
    """``tf_efficientnetv2_l`` trunk + custom MLP head, one logit (P(fake) after sigmoid).

    Same architecture as ``backend/notebook/image_detection/image_deepfake_detection (2).ipynb``
    (the training-time ``freeze_layers`` helper is dropped; the detector freezes
    every parameter after loading).
    """

    def __init__(self, num_classes: int = 1, dropout_rate: float = 0.3, pretrained: bool = False) -> None:
        super().__init__()
        self.base_model = create_model("tf_efficientnetv2_l", pretrained=pretrained, num_classes=0)
        num_features = self.base_model.num_features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model.forward_features(x)
        out = self.classifier(features)
        if out.dim() == 2 and out.size(1) == 1:
            out = out.squeeze(1)
        return out

    def last_block_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Trunk output up to and including ``base_model.blocks[-1]`` (the v1 GradCAM layer)."""
        trunk = self.base_model
        x = trunk.bn1(trunk.conv_stem(x))
        return trunk.blocks(x)

    def logit_from_last_block(self, acts: torch.Tensor) -> torch.Tensor:
        trunk = self.base_model
        features = trunk.bn2(trunk.conv_head(acts))
        return self.classifier(features).reshape(-1)
