"""Transformers audio-classification detectors (Wav2Vec2 family).

* ``wav2vec2-legacy`` — the FusionTrace v1 model (wav2vec2-base fine-tuned on
  4 s clips; class 0 = Real, 1 = Fake), with the preprocessing of its training
  notebook: per-window zero-mean/unit-variance, then zero padding to 4 s.
* Hub candidates evaluated in ``docs/model-choice-audio.md``; they use their own
  feature extractor, as their model cards do.

All weights load from local directories with ``use_safetensors=True`` (no
pickle, no remote code). Inference runs under ``torch.inference_mode`` on an
eval-mode model and touches no shared mutable state, so ``analyze`` is safe to
call concurrently on one instance.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..base import DetectorInfo, MediaKind, ModelUnavailableError, WeightsSpec
from .chunked import ChunkedAudioDetector

logger = logging.getLogger("fusiontrace")

_FAKE_NAMES = {"fake", "spoof", "synthetic", "deepfake", "ai", "generated"}
_REAL_NAMES = {"real", "bonafide", "bona-fide", "bona_fide", "genuine", "human"}
_MIN_WEIGHTS_BYTES = 1_000_000  # anything smaller is a Git LFS pointer, not weights


def _weights_present(directory: Path) -> bool:
    weights = directory / "model.safetensors"
    return (directory / "config.json").is_file() and weights.is_file() and weights.stat().st_size > _MIN_WEIGHTS_BYTES


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class HFAudioClassifierDetector(ChunkedAudioDetector):
    """A two-class ``*ForSequenceClassification`` model scored window by window."""

    exportable = True
    fake_index: ClassVar[int] = 1  # fallback when config.id2label has no recognisable names
    pad_short_clips: ClassVar[bool] = False  # zero-pad clips shorter than one window
    use_feature_extractor: ClassVar[bool] = True  # False: legacy per-window standardisation
    hub_weights: ClassVar[WeightsSpec | None] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model: Any = None
        self._feature_extractor: Any = None
        self._export_module: Any = None
        self._fake_index = self.fake_index

    # ── weights ──

    @classmethod
    def weights(cls) -> list[WeightsSpec]:
        return [cls.hub_weights] if cls.hub_weights is not None else []

    def weight_dirs(self) -> list[Path]:
        """Directories searched, in order, for ``config.json`` + ``model.safetensors``."""
        spec = self.hub_weights
        return [self.model_dir / spec.local_subdir] if spec is not None else []

    def weights_dir(self) -> Path | None:
        return next((directory for directory in self.weight_dirs() if _weights_present(directory)), None)

    def is_available(self) -> bool:
        return self.weights_dir() is not None

    def _load(self) -> None:
        directory = self.weights_dir()
        if directory is None:
            searched = ", ".join(str(d) for d in self.weight_dirs()) or "(none)"
            raise ModelUnavailableError(f"{self.info.name}: no weights found; searched {searched}")
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        model = AutoModelForAudioClassification.from_pretrained(str(directory), local_files_only=True, use_safetensors=True)
        if int(model.config.num_labels) != 2:
            raise ModelUnavailableError(f"{self.info.name}: expected a 2-class model, got {model.config.num_labels} labels")
        self._fake_index = self._resolve_fake_index(model.config)
        model.eval()
        model.to(torch.device(self.device))
        if self.use_feature_extractor:
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(str(directory), local_files_only=True)
        self._model = model
        logger.info("Loaded audio detector %s from %s (fake class index %d)", self.info.name, directory, self._fake_index)

    def _resolve_fake_index(self, config: Any) -> int:
        names = {int(k): str(v).strip().lower() for k, v in (getattr(config, "id2label", None) or {}).items()}
        fake = [index for index, name in names.items() if name in _FAKE_NAMES]
        real = [index for index, name in names.items() if name in _REAL_NAMES]
        if len(fake) == 1 and len(real) == 1:
            if fake[0] != self.fake_index:
                logger.warning("%s: config says fake class is %d, class default is %d; using config", self.info.name, fake[0], self.fake_index)
            return fake[0]
        return self.fake_index

    # ── preprocessing ──

    def _prepare(self, windows: list[np.ndarray], window_samples: int) -> dict[str, Any]:
        """Model inputs for a batch of raw windows (all the same length except a lone short clip)."""
        import torch

        if self.use_feature_extractor:
            batch = [w if not self.pad_short_clips or w.size >= window_samples else np.pad(w, (0, window_samples - w.size)) for w in windows]
            inputs = self._feature_extractor(batch, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
            return {key: value.to(self.device) for key, value in inputs.items()}
        rows = []
        for w in windows:
            w = w.astype(np.float32)
            # Training notebook: (x - mean) / (std + 1e-6) with torch's unbiased std, then zero padding.
            std = float(w.std(ddof=1)) if w.size > 1 else 0.0
            w = (w - w.mean()) / (std + 1e-6)
            if self.pad_short_clips and w.size < window_samples:
                w = np.pad(w, (0, window_samples - w.size))
            rows.append(w.astype(np.float32))
        return {"input_values": torch.from_numpy(np.stack(rows)).to(self.device)}

    # ── inference ──

    def _score_batch(self, windows: list[np.ndarray], window_samples: int) -> list[float]:
        import torch

        inputs = self._prepare(windows, window_samples)
        with torch.inference_mode():
            logits = self._model(**inputs).logits
        probs = _softmax(logits.float().cpu().numpy().astype(np.float64))
        return [float(p) for p in probs[:, self._fake_index]]

    # ── export hooks (unit = one window) ──

    def export_module(self) -> Any:
        self.load()
        if self._export_module is None:
            import torch

            class LogitsOnly(torch.nn.Module):
                """``input_values`` [batch, samples] -> logits [batch, 2]."""

                def __init__(self, inner: torch.nn.Module) -> None:
                    super().__init__()
                    self.inner = inner

                def forward(self, input_values: torch.Tensor) -> torch.Tensor:
                    return self.inner(input_values=input_values).logits

            self._export_module = LogitsOnly(self._model).eval()
        return self._export_module

    def prepare_inputs(self, path: Path) -> tuple[Any, ...]:
        self.load()
        window_samples = round(self.chunk_options().window_seconds * self.sample_rate)
        inputs = self._prepare([self.first_window(Path(path))], window_samples)
        return (inputs["input_values"].cpu(),)

    def risk_from_outputs(self, outputs: Sequence[Any]) -> float:
        first = outputs[0]
        if hasattr(first, "detach"):
            first = first.detach().cpu().numpy()
        logits = np.asarray(first, dtype=np.float64).reshape(-1, 2)[0]
        return float(_softmax(logits)[self._fake_index])


class Wav2Vec2LegacyDetector(HFAudioClassifierDetector):
    """FusionTrace v1 audio model, now scored over the whole clip."""

    info = DetectorInfo(
        name="wav2vec2-legacy",
        kind=MediaKind.AUDIO,
        display_name="Wav2Vec2-base (FusionTrace legacy)",
        source="local",
        license="MIT",
        version="v1",
        description="facebook/wav2vec2-base fine-tuned by FusionTrace on 4 s real/synthetic speech clips.",
    )
    fake_index = 1  # training notebook: label 0 = real, 1 = fake
    pad_short_clips = True
    use_feature_extractor = False

    @classmethod
    def weights(cls) -> list[WeightsSpec]:
        # Not public: fetched from the private mirror named by HF_REPO_ID into <model_dir>/audio_model.
        return [WeightsSpec(repo_id="", local_subdir="", allow_patterns=("audio_model/*",), repo_id_env="HF_REPO_ID")]

    def weight_dirs(self) -> list[Path]:
        from ...config import AUDIO_MODEL_PATH  # honours FUSIONTRACE_AUDIO_MODEL_PATH

        dirs = [Path(AUDIO_MODEL_PATH), self.model_dir / "audio_model"]
        return list(dict.fromkeys(d.resolve() for d in dirs))


class XlsrGaryStaffordDetector(HFAudioClassifierDetector):
    """garystafford/wav2vec2-deepfake-voice-detector (XLS-R 300M, 2025)."""

    info = DetectorInfo(
        name="wav2vec2-xlsr-garystafford",
        kind=MediaKind.AUDIO,
        display_name="Wav2Vec2 XLS-R 300M deepfake voice detector (garystafford)",
        source="garystafford/wav2vec2-deepfake-voice-detector",
        license="Apache-2.0",
        version="c66306024a7e",
        description="XLS-R 300M fine-tuned on ASVspoof2019 (Gustking) then on ElevenLabs, Polly, Kokoro, Hume AI, Speechify and Luvvoice speech.",
    )
    fake_index = 1
    hub_weights = WeightsSpec(
        repo_id="garystafford/wav2vec2-deepfake-voice-detector",
        local_subdir="audio-garystafford-xlsr",
        revision="c66306024a7ede0be291e9c4558b37634782dc4e",
        allow_patterns=("config.json", "preprocessor_config.json", "model.safetensors"),
    )


class Wav2Vec2MelodyMachineDetector(HFAudioClassifierDetector):
    """MelodyMachine/Deepfake-audio-detection-V2 (wav2vec2-base, 2024)."""

    info = DetectorInfo(
        name="wav2vec2-base-melodymachine",
        kind=MediaKind.AUDIO,
        display_name="Wav2Vec2-base Deepfake-audio-detection-V2 (MelodyMachine)",
        source="MelodyMachine/Deepfake-audio-detection-V2",
        license="Apache-2.0",
        version="de3cde5a29c4",
        description="wav2vec2-base fine-tuned for real/fake speech; training data not documented by the authors.",
    )
    fake_index = 0  # config.id2label: {0: "fake", 1: "real"}
    hub_weights = WeightsSpec(
        repo_id="MelodyMachine/Deepfake-audio-detection-V2",
        local_subdir="audio-melodymachine-v2",
        revision="de3cde5a29c449bb5268814e421b46bf6ebdcd72",
        allow_patterns=("config.json", "preprocessor_config.json", "model.safetensors"),
    )
