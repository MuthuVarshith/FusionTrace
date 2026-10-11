"""Shared plumbing for the image detectors: decoding, results, explanations.

Every concrete image detector subclasses :class:`TorchImageDetector` and
supplies five small hooks (build module, preprocess, logits -> risk, staged
forward for explanations, CAM from activations/gradients). The base class
turns those into the ``Detector`` contract, including the export hooks used by
the ONNX tooling.

Thread-safety model
-------------------
* ``_load`` runs once under ``Detector.load``'s lock. The module is put in
  eval mode and every parameter is frozen (``requires_grad=False``); nothing
  ever switches it back, so the module is read-only after load.
* ``analyze`` runs the forward pass under ``torch.inference_mode()`` (a
  thread-local mode) and keeps all intermediate state in local variables.
* ``explain`` never registers hooks and never calls ``.backward()``. It runs a
  staged forward pass inside the call: the trunk up to the explanation layer
  under ``no_grad``, then a *detached copy* of that activation with
  ``requires_grad=True`` through the rest of the network, and takes
  ``torch.autograd.grad(score, activation)``. Gradients therefore exist only
  for that call-local tensor; parameter ``.grad`` is never written and no
  module state changes, so concurrent ``analyze``/``explain`` calls cannot see
  each other's activations or gradients. The per-instance ``_explain_lock``
  only serialises explanations to bound peak memory and CPU; correctness does
  not depend on it.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ..base import AnalysisContext, Detector, DetectorError, DetectorResult, InferenceError, UnsupportedInputError

if TYPE_CHECKING:
    import numpy as np
    import torch
    from PIL import Image

MIN_SIDE_PIXELS = 32  # smaller images carry no usable evidence
MAX_OVERLAY_SIDE = 1024  # explanation JPEGs are downscaled to this longest side
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_rgb(path: Path) -> Image.Image:
    """Decode ``path`` as an upright RGB image or raise UnsupportedInputError."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(path) as handle:
            image = ImageOps.exif_transpose(handle)
            rgb = image.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise UnsupportedInputError(f"cannot decode image {Path(path).name}: {exc}", public_message="The image could not be decoded.") from exc
    except (OSError, ValueError, SyntaxError) as exc:  # truncated / corrupt files
        raise UnsupportedInputError(f"cannot decode image {Path(path).name}: {exc}", public_message="The image could not be decoded.") from exc
    if min(rgb.size) < MIN_SIDE_PIXELS:
        raise UnsupportedInputError(
            f"image {Path(path).name} is {rgb.size[0]}x{rgb.size[1]}", public_message="The image is too small to analyse."
        )
    return rgb


def binary_result(risk: float, threshold: float, details: Mapping[str, Any] | None = None) -> DetectorResult:
    risk = min(1.0, max(0.0, float(risk)))
    label = "Fake" if risk >= threshold else "Real"
    confidence = risk if label == "Fake" else 1.0 - risk
    return DetectorResult(risk=risk, label=label, confidence=confidence, details={"threshold": threshold, **(details or {})})


def to_numpy(value: Any) -> np.ndarray:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def softmax_prob(logits: np.ndarray, index: int) -> float:
    import numpy as np

    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return float(exp[index] / exp.sum())


def normalise_cam(cam: np.ndarray) -> np.ndarray | None:
    """Min-max scale to [0, 1]; None for a flat map (nothing to show)."""
    import numpy as np

    cam = np.nan_to_num(np.asarray(cam, dtype=np.float64))
    low, high = float(cam.min()), float(cam.max())
    if high - low < 1e-8:
        return None
    return (cam - low) / (high - low)


def conv_gradcam(acts: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
    """GradCAM for a CNN feature map. ``acts``/``grads``: [1, C, H, W] -> [H, W]."""
    import torch

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1))[0]
    return cam.detach().cpu().numpy()


def token_gradcam(acts: torch.Tensor, grads: torch.Tensor, grid: tuple[int, int], prefix_tokens: int) -> np.ndarray:
    """GradCAM over ViT patch tokens. ``acts``/``grads``: [1, T, D] -> [gh, gw].

    Prefix tokens (CLS / registers) are dropped; the remaining tokens are the
    patch grid in row-major order, as in timm, CLIP and SigLIP.
    """
    import torch

    patch_acts = acts[0, prefix_tokens:, :]
    patch_grads = grads[0, prefix_tokens:, :]
    weights = patch_grads.mean(dim=0)
    cam = torch.relu((patch_acts * weights).sum(dim=-1))
    return cam.reshape(grid).detach().cpu().numpy()


def save_overlay(image: Image.Image, cam: np.ndarray, out_path: Path, *, alpha: float = 0.6, quality: int = 88) -> Path:
    """Blend a [0, 1] heat map (any grid size) over ``image`` with a jet colormap; write a JPEG.

    Pure NumPy + PIL (no OpenCV/matplotlib), same look as the v1 GradCAM overlay.
    """
    import numpy as np
    from PIL import Image

    base = image
    if max(base.size) > MAX_OVERLAY_SIDE:
        base = base.copy()
        base.thumbnail((MAX_OVERLAY_SIDE, MAX_OVERLAY_SIDE), Image.BILINEAR)
    width, height = base.size
    heat = Image.fromarray((np.clip(cam, 0.0, 1.0) * 255).astype(np.uint8)).resize((width, height), Image.BILINEAR)
    t = np.asarray(heat, dtype=np.float32) / 255.0
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    colour = np.stack([r, g, b], axis=-1) * 255.0
    blended = ((1.0 - alpha) * np.asarray(base, dtype=np.float32) + alpha * colour).clip(0, 255).astype(np.uint8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended).save(out_path, format="JPEG", quality=quality)
    return out_path


class TorchImageDetector(Detector):
    """A single-image torch classifier behind the Detector contract.

    Options (config ``detectors.image.options``, or per-call ``ctx.options``):

    * ``threshold`` (float, default ``default_threshold``): risk at or above
      which the label is "Fake".
    """

    supports_explanation: ClassVar[bool] = True
    exportable: ClassVar[bool] = True
    default_threshold: ClassVar[float] = 0.5

    def __init__(self, model_dir: Path, device: str = "cpu", options: Mapping[str, Any] | None = None) -> None:
        super().__init__(model_dir=model_dir, device=device, options=options)
        self._module: torch.nn.Module | None = None
        self._explain_lock = threading.Lock()

    # ── subclass hooks ──

    def _build_module(self) -> torch.nn.Module:
        """Construct the network and load weights (called once, under the load lock)."""
        raise NotImplementedError

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL RGB image -> float tensor [3, H, W] as the network expects."""
        raise NotImplementedError

    def _risk_from_logits(self, logits: np.ndarray) -> float:
        """Raw outputs for one image (1-D) -> P(synthetic)."""
        raise NotImplementedError

    def _staged_forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(activation, fake_score)``.

        ``activation`` must be a call-local tensor with ``requires_grad=True``
        (computed under ``no_grad`` and detached), and ``fake_score`` a scalar
        computed from it that increases with P(synthetic).
        """
        raise NotImplementedError

    def _cam(self, acts: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
        raise NotImplementedError

    # ── helpers ──

    def option(self, ctx: AnalysisContext | None, key: str, default: Any) -> Any:
        if ctx is not None and key in ctx.options:
            return ctx.options[key]
        return self.options.get(key, default)

    def threshold(self, ctx: AnalysisContext | None) -> float:
        value = float(self.option(ctx, "threshold", self.default_threshold))
        if not 0.0 < value < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {value}")
        return value

    def _pixels(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image).unsqueeze(0).to(self.device)

    def _details(self, pixels: torch.Tensor) -> dict[str, Any]:
        return {"input_size": list(pixels.shape[-2:])}

    # ── Detector contract ──

    def _load(self) -> None:
        module = self._build_module()
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        self._module = module.to(self.device)

    def _forward(self, pixels: torch.Tensor) -> torch.Tensor:
        import torch

        assert self._module is not None
        with torch.inference_mode():
            return self._module(pixels)

    def _analyze(self, path: Path, ctx: AnalysisContext) -> DetectorResult:
        image = load_rgb(path)
        pixels = self._pixels(image)
        outputs = self._forward(pixels)
        risk = self._risk_from_logits(to_numpy(outputs).reshape(-1))
        return binary_result(risk, self.threshold(ctx), self._details(pixels))

    def explain(self, path: Path, out_path: Path, ctx: AnalysisContext) -> Path | None:
        if not self.supports_explanation:
            return None
        self.load()
        ctx.check_deadline()
        try:
            import torch

            image = load_rgb(path)
            pixels = self._pixels(image)
            with self._explain_lock, torch.enable_grad():
                acts, score = self._staged_forward(pixels)
                (grads,) = torch.autograd.grad(score, acts)
            cam = normalise_cam(self._cam(acts.detach(), grads.detach()))
            if cam is None:
                return None
            return save_overlay(image, cam, out_path)
        except DetectorError:
            raise
        except Exception as exc:
            raise InferenceError(f"{self.info.name}: explanation failed on {Path(path).name}: {exc}") from exc

    # ── export hooks (unit = one image) ──

    def export_module(self) -> torch.nn.Module:
        self.load()
        assert self._module is not None
        return self._module

    def prepare_inputs(self, path: Path) -> tuple[torch.Tensor, ...]:
        return (self._preprocess(load_rgb(path)).unsqueeze(0),)

    def risk_from_outputs(self, outputs: Sequence[Any]) -> float:
        return self._risk_from_logits(to_numpy(outputs[0]).reshape(-1))
