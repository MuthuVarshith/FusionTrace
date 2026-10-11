"""Pick a detector for the offline tools, always through ``app.detectors.registry``.

A detector is chosen by

* media kind only            -> the configured detector (``config.yaml`` / env), honouring
                                ``registry.override`` (tests inject stubs that way);
* a registry key             -> that implementation, searched in ``--kind`` or across all kinds;
* ``package.module:Class``   -> an explicit import, for detectors that are not registered
                                (e.g. ``app.detectors.testing:TinyTorchDetector`` for smoke tests).
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.detectors import registry
from app.detectors.base import Detector, DetectorError, MediaKind

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".flac", ".ogg"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
EXTENSIONS: dict[MediaKind, frozenset[str]] = {
    MediaKind.IMAGE: IMAGE_EXTENSIONS,
    MediaKind.AUDIO: AUDIO_EXTENSIONS,
    MediaKind.VIDEO: VIDEO_EXTENSIONS,
}
# Folder names for labelled data: <dir>/real/* and <dir>/fake/* (the bundled set uses "deepfakes").
REAL_DIRS = ("real",)
FAKE_DIRS = ("fake", "deepfakes")


class SelectionError(RuntimeError):
    """The requested detector cannot be used by this tool; the message says why."""


def parse_kind(value: str | MediaKind | None) -> MediaKind | None:
    if value is None or isinstance(value, MediaKind):
        return value
    try:
        return MediaKind(value.strip().lower())
    except ValueError as exc:
        raise SelectionError(f"unknown media kind {value!r}; use one of {[kind.value for kind in MediaKind]}") from exc


def parse_options(pairs: Sequence[str] | None) -> dict[str, Any]:
    """``["key=value", ...]`` -> dict; values are parsed as JSON when possible (numbers, bools, lists)."""
    options: dict[str, Any] = {}
    for pair in pairs or ():
        key, sep, raw = pair.partition("=")
        if not sep or not key.strip():
            raise SelectionError(f"option {pair!r} must look like key=value")
        try:
            options[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            options[key.strip()] = raw
    return options


def _import_class(spec: str) -> type[Detector]:
    module_name, _, attribute = spec.partition(":")
    try:
        candidate = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise SelectionError(f"cannot import detector class {spec!r}: {exc}") from exc
    if not (isinstance(candidate, type) and issubclass(candidate, Detector)):
        raise SelectionError(f"{spec!r} is not a Detector subclass")
    return candidate


def _instantiate(cls: type[Detector], options: Mapping[str, Any]) -> Detector:
    try:
        return cls(model_dir=registry.model_dir(), device=registry.device(), options=dict(options))
    except TypeError:
        # Test doubles such as StubDetector take no model_dir; fall back to their defaults.
        return cls()


def _find_registered(name: str, kind: MediaKind | None) -> tuple[MediaKind, type[Detector]]:
    kinds = [kind] if kind is not None else list(MediaKind)
    available: dict[str, list[str]] = {}
    for candidate_kind in kinds:
        detectors, _ = registry.catalogue(candidate_kind)
        if name in detectors:
            return candidate_kind, detectors[name]
        available[candidate_kind.value] = sorted(detectors)
    raise SelectionError(
        f"detector {name!r} is not registered{f' for {kind.value}' if kind else ''}; registered: {available}. "
        "Use a registry key or an import path such as app.detectors.testing:TinyTorchDetector."
    )


def resolve_detector(kind: str | MediaKind | None, name: str | None = None, options: Mapping[str, Any] | None = None) -> Detector:
    """Return an (unloaded) detector instance for the tools. Raises SelectionError with a readable reason."""
    media_kind = parse_kind(kind)
    extra = dict(options or {})
    if name and ":" in name:
        cls = _import_class(name)
        if media_kind is not None and cls.info.kind != media_kind:
            raise SelectionError(f"{name} is a {cls.info.kind.value} detector, not {media_kind.value}")
        base = registry.settings(cls.info.kind).options if _is_configured(cls.info.kind, cls.info.name) else {}
        return _instantiate(cls, {**base, **extra})
    if name:
        found_kind, cls = _find_registered(name, media_kind)
        base = registry.settings(found_kind).options if _is_configured(found_kind, name) else {}
        return _instantiate(cls, {**base, **extra})
    if media_kind is None:
        raise SelectionError("pass --kind (image, audio or video) and/or --detector")
    try:
        detector = registry.get(media_kind)
    except DetectorError as exc:
        raise SelectionError(
            f"no usable {media_kind.value} detector is configured ({exc}). "
            "Pass --detector <registry key> or an import path such as app.detectors.testing:TinyTorchDetector."
        ) from exc
    if detector is None:
        raise SelectionError(f"the {media_kind.value} detector is disabled by configuration; pass --detector to choose one explicitly")
    if extra:
        detector.options.update(extra)
    return detector


def _is_configured(kind: MediaKind, name: str) -> bool:
    try:
        return registry.settings(kind).model == name
    except Exception:  # noqa: BLE001  # an empty or broken package must not block explicit selection
        return False


def require_exportable(detector: Detector) -> None:
    """Fail clearly when a detector lacks the export hooks (``exportable = False``)."""
    if not getattr(detector, "exportable", False):
        info = detector.info
        raise SelectionError(
            f"detector {info.name!r} ({info.kind.value}) is not exportable: it does not implement the export hooks "
            "export_module() / prepare_inputs() / risk_from_outputs(). ONNX export, quantisation and the "
            "torch/ONNX benchmark are unavailable for it."
        )


def detector_spec(detector: Detector) -> str:
    """``module:Class`` import path that recreates this detector class."""
    cls = type(detector)
    return f"{cls.__module__}:{cls.__qualname__}"


def labelled_files(data_dir: Path, kind: MediaKind) -> list[tuple[Path, str]]:
    """(path, "Real"|"Fake") pairs under ``<data_dir>/real`` and ``<data_dir>/fake`` (or ``deepfakes``), recursive."""
    extensions = EXTENSIONS[kind]
    pairs: list[tuple[Path, str]] = []
    for folders, label in ((REAL_DIRS, "Real"), (FAKE_DIRS, "Fake")):
        for folder_name in folders:
            folder = data_dir / folder_name
            if not folder.is_dir():
                continue
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in extensions:
                    pairs.append((path, label))
    return pairs


def media_files(folder: Path, kind: MediaKind, limit: int | None = None) -> list[Path]:
    """Files of ``kind`` anywhere under ``folder``, sorted, optionally capped."""
    files = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS[kind])
    return files[:limit] if limit else files


def default_sample(kind: MediaKind) -> Path | None:
    """First bundled test file of ``kind`` (``backend/test/real`` then ``deepfakes``), if any."""
    from app.config import BASE_DIR

    for path, _ in labelled_files(BASE_DIR / "test", kind):
        return path
    return None


def resolve_sample(detector: Detector, sample: Path | None) -> Path:
    if sample is not None:
        if not sample.is_file():
            raise SelectionError(f"sample file {sample} does not exist")
        return sample
    found = default_sample(detector.info.kind)
    if found is None:
        raise SelectionError(f"no bundled {detector.info.kind.value} sample found; pass --sample <file>")
    return found
