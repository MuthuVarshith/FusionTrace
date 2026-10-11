"""Fast checks for the image detectors. No downloads; weight-backed tests skip when weights are absent."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.detectors import registry
from app.detectors.base import (
    AnalysisContext,
    Detector,
    MediaKind,
    ModelUnavailableError,
    UnsupportedInputError,
)
from app.detectors.image import DEFAULT, DETECTORS
from app.detectors.image._common import (
    binary_result,
    load_rgb,
    normalise_cam,
    save_overlay,
)

TEST_DIR = Path(__file__).resolve().parents[1] / "test"


def _sample(folder: str) -> Path:
    return next(p for p in sorted((TEST_DIR / folder).iterdir()) if p.suffix.lower() in {".jpg", ".png"})


def test_registry_contract() -> None:
    assert DEFAULT in DETECTORS
    assert "efficientnetv2-legacy" in DETECTORS
    for name, cls in DETECTORS.items():
        assert issubclass(cls, Detector)
        assert cls.info.name == name and cls.info.kind is MediaKind.IMAGE
        for spec in cls.weights():
            assert spec.allow_patterns
            assert spec.revision or spec.repo_id_env  # public repos pinned; legacy via private mirror
    assert DETECTORS[DEFAULT].supports_explanation and DETECTORS[DEFAULT].exportable
    assert registry.catalogue(MediaKind.IMAGE)[1] == DEFAULT


def test_missing_weights_is_typed(tmp_path: Path) -> None:
    for name, cls in DETECTORS.items():
        if name == "efficientnetv2-legacy":
            continue  # resolves its path from app.config as well
        detector = cls(model_dir=tmp_path)
        assert not detector.is_available()
        with pytest.raises(ModelUnavailableError):
            detector.load()


def test_undecodable_and_tiny_images(tmp_path: Path) -> None:
    from PIL import Image

    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(UnsupportedInputError):
        load_rgb(bad)
    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8)).save(tiny)
    with pytest.raises(UnsupportedInputError):
        load_rgb(tiny)


def test_result_and_overlay_helpers(tmp_path: Path) -> None:
    import numpy as np
    from PIL import Image

    assert binary_result(0.7, 0.5).label == "Fake"
    assert binary_result(0.2, 0.5).confidence == pytest.approx(0.8)
    assert normalise_cam(np.ones((4, 4))) is None
    cam = normalise_cam(np.arange(16, dtype=float).reshape(4, 4))
    out = save_overlay(Image.new("RGB", (64, 48), "white"), cam, tmp_path / "o.jpg")
    with Image.open(out) as written:
        assert written.size == (64, 48) and written.format == "JPEG"


@pytest.fixture(scope="module")
def default_detector() -> Detector:
    detector = DETECTORS[DEFAULT](model_dir=registry.model_dir())
    if not detector.is_available():
        pytest.skip(f"weights for {DEFAULT} not present under {registry.model_dir()}")
    detector.load()
    return detector


def test_default_scores_and_explains(default_detector: Detector, tmp_path: Path) -> None:
    ctx = AnalysisContext(work_dir=tmp_path)
    result = default_detector.analyze(_sample("deepfakes"), ctx)
    assert 0.0 <= result.risk <= 1.0 and result.label in ("Fake", "Real")
    out = default_detector.explain(_sample("deepfakes"), tmp_path / "cam.jpg", ctx)
    assert out is not None and out.stat().st_size > 0
    assert all(p.grad is None for p in default_detector.export_module().parameters())


def test_default_concurrent_calls_agree(default_detector: Detector, tmp_path: Path) -> None:
    ctx = AnalysisContext(work_dir=tmp_path)
    paths = [_sample("real"), _sample("deepfakes")]
    expected = [default_detector.analyze(p, ctx).risk for p in paths]
    got: dict[int, float] = {}
    errors: list[BaseException] = []

    def run(i: int) -> None:
        try:
            if i % 3 == 2:
                default_detector.explain(paths[1], tmp_path / f"cam{i}.jpg", ctx)
            else:
                got[i] = default_detector.analyze(paths[i % 2], ctx).risk
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for i, risk in got.items():
        assert risk == pytest.approx(expected[i % 2], abs=1e-5)


def test_default_export_hooks(default_detector: Detector) -> None:
    import torch

    (pixels,) = default_detector.prepare_inputs(_sample("real"))
    with torch.no_grad():
        outputs = default_detector.export_module()(pixels)
    risk = default_detector.risk_from_outputs([outputs])
    assert risk == pytest.approx(default_detector.analyze(_sample("real"), AnalysisContext(work_dir=Path("."))).risk, abs=1e-5)
