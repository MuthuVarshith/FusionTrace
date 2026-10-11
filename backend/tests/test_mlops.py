"""Quick tests for the MLOps tooling: metric math, evaluation with stubs, ONNX export parity."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "scripts"))

from app.detectors.base import (  # noqa: E402
    AnalysisContext,
    DetectorInfo,
    DetectorResult,
    MediaKind,
    UnsupportedInputError,
)
from app.detectors.testing import StubDetector, TinyTorchDetector  # noqa: E402
from mlops import metrics  # noqa: E402
from mlops.selection import SelectionError, require_exportable  # noqa: E402


def test_auc_known_example():
    assert metrics.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_auc_ties_count_half():
    # pairs: (0.5 vs 0.5)=0.5, (0.5 vs 0.2)=1, (0.9 vs 0.5)=1, (0.9 vs 0.2)=1 -> 3.5/4
    assert metrics.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.2, 0.9]) == pytest.approx(0.875)
    assert metrics.roc_auc([1, 1], [0.3, 0.4]) is None


def test_auc_matches_sklearn_when_available():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    labels = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
    scores = [0.2, 0.7, 0.7, 0.4, 0.9, 0.7, 0.1, 0.3, 0.6, 0.5]
    assert metrics.roc_auc(labels, scores) == pytest.approx(sklearn_metrics.roc_auc_score(labels, scores))


def test_confusion_metrics():
    report = metrics.binary_report(["Fake", "Fake", "Fake", "Real", "Real"], ["Fake", "Fake", "Real", "Fake", "Real"])
    assert report["confusion_matrix"]["matrix"] == [[1, 1], [1, 2]]
    assert report["fake_precision"] == pytest.approx(2 / 3)
    assert report["fake_recall"] == pytest.approx(2 / 3)
    assert report["fake_f1"] == pytest.approx(2 / 3)
    assert report["real_recall"] == pytest.approx(0.5)
    assert metrics.binary_report(["Real"], ["Real"])["fake_precision"] is None


class _NameRisk(StubDetector):
    """Risk taken from the file name (``x_0.8.jpg``); ``bad`` files raise."""

    def _analyze(self, path: Path, ctx: AnalysisContext) -> DetectorResult:
        if path.stem.startswith("bad"):
            raise UnsupportedInputError("no face")
        risk = float(path.stem.split("_")[1])
        return DetectorResult(risk=risk, label="Fake" if risk >= 0.5 else "Real", confidence=max(risk, 1 - risk))


def test_evaluate_detector_with_stub(tmp_path):
    evaluate_models = pytest.importorskip("evaluate_models")
    for folder, names in (("real", ["a_0.1", "b_0.6", "bad_x"]), ("deepfakes", ["c_0.9", "d_0.4"])):
        (tmp_path / folder).mkdir()
        for name in names:
            (tmp_path / folder / f"{name}.jpg").write_bytes(b"x")
    info = DetectorInfo(name="name-risk", kind=MediaKind.IMAGE, display_name="t", source="local", license="MIT")
    section = evaluate_models.evaluate_detector(_NameRisk(info=info), tmp_path, MediaKind.IMAGE, threshold=0.95)
    assert section["counts"] == {"files": 5, "scored": 4, "real": 2, "fake": 2, "errors": 1}
    assert section["errors"][0]["error_type"] == "UnsupportedInputError"
    assert section["roc_auc"] == pytest.approx(0.75)
    assert section["at_detector_threshold"]["confusion_matrix"]["matrix"] == [[1, 1], [1, 1]]
    assert section["at_threshold"]["fake_recall"] == 0.0


def test_not_exportable_is_reported_clearly():
    with pytest.raises(SelectionError, match="not exportable"):
        require_exportable(StubDetector())


@pytest.mark.skipif(importlib.util.find_spec("onnx") is None or importlib.util.find_spec("onnxruntime") is None, reason="onnx not installed")
def test_export_tiny_torch_parity(tmp_path):
    export_onnx = pytest.importorskip("export_onnx")
    sample = tmp_path / "sample.jpg"
    sample.write_bytes(b"x")
    meta = export_onnx.export(TinyTorchDetector(model_dir=tmp_path), sample, tmp_path / "out")
    assert (tmp_path / "out" / "model.onnx").is_file() and (tmp_path / "out" / "model.json").is_file()
    assert meta["parity"]["passed"] and meta["opset"] >= 17 and meta["dynamic_batch"]
    assert meta["parity"]["risk_abs_diff"] < 1e-5
