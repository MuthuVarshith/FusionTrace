"""HTTP API tests. Stub detectors via ``registry.override``; no model weights are loaded."""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import sqlite3
import struct
import subprocess
import threading
import time
import wave
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

os.environ.setdefault("FUSIONTRACE_WARMUP", "0")

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
import app.service as service_module
from app import ai_summary
from app import config as settings
from app.auth import owner_id
from app.detectors import registry
from app.detectors.base import AnalysisContext, DetectorResult, InferenceError, MediaKind
from app.detectors.testing import StubDetector
from app.jobs import ScanQueue
from app.ratelimit import RateLimiter
from app.service import GENERIC_SCAN_ERROR, INTERRUPTED_SCAN_ERROR, ScanService

SECRET = "C:/srv/secret-internal/weights.pth"
needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe not on PATH")


class ImageStub(StubDetector):
    """Writes into the work dir (to prove it is cleaned up) and produces a heatmap."""

    supports_explanation = True

    def _analyze(self, path: Path, ctx: AnalysisContext) -> DetectorResult:
        (ctx.work_dir / "scratch.bin").write_bytes(b"temporary")
        return super()._analyze(path, ctx)

    def explain(self, path: Path, out_path: Path, ctx: AnalysisContext) -> Path | None:
        Image.new("RGB", (8, 8), (255, 0, 0)).save(out_path, "JPEG")
        return out_path


# ── helpers ─────────────────────────────────────────────────────────────────


def png_bytes(size: tuple[int, int] = (32, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 30, 200)).save(buffer, "PNG")
    return buffer.getvalue()


def wav_bytes(seconds: float = 1.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = int(seconds * rate)
        handle.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(frames)))
    return buffer.getvalue()


def upload(client: TestClient, name: str, data: bytes, headers: dict[str, str] | None = None, **form: str):
    return client.post("/api/scans", files={"file": (name, data, "application/octet-stream")}, data=form, headers=headers or {})


def wait_for(client: TestClient, scan_id: str, until: Callable[[dict[str, Any]], bool] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
    until = until or (lambda scan: scan["status"] in ("completed", "failed"))
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = client.get(f"/api/scans/{scan_id}", headers=headers or {})
        assert response.status_code == 200, response.text
        scan = response.json()
        if until(scan):
            return scan
        time.sleep(0.05)
    raise AssertionError(f"scan {scan_id} did not reach the expected state")


def evidence_by_source(scan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["source"]: item for item in scan["report"]["evidence"]}


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("FUSIONTRACE_API_KEYS", raising=False)
    svc = ScanService(database_path=tmp_path / "test.db", upload_dir=tmp_path / "uploads", artifact_dir=tmp_path / "artifacts")
    monkeypatch.setattr(main, "service", svc)
    monkeypatch.setattr(main, "upload_limiter", RateLimiter(100, 60))
    monkeypatch.setattr(main, "feedback_limiter", RateLimiter(100, 60))
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def image_detector() -> Iterator[ImageStub]:
    detector = ImageStub.for_kind(MediaKind.IMAGE, risk=0.9)
    with registry.override(MediaKind.IMAGE, detector):
        yield detector


# ── scans, reports, heatmap ─────────────────────────────────────────────────


def test_image_scan_reports_models_heatmap_and_cleans_work_dir(client: TestClient, image_detector: ImageStub) -> None:
    response = upload(client, "photo.png", png_bytes())
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "queued"
    scan_id = response.json()["id"]

    scan = wait_for(client, scan_id, lambda s: s["status"] == "completed" and s["report"]["heatmap_status"] != "pending")
    report = scan["report"]
    image_item = evidence_by_source(scan)["image-model"]
    assert image_item["status"] == "ok"
    assert image_item["model"]["display_name"] == "Stub image detector"
    assert report["assessment"]["risk_rating"] == "high"
    assert report["ai_summary_status"] == "not_requested"
    assert report["heatmap_ready"] is True
    assert report["heatmap_model"]["display_name"] == "Stub image detector"

    heatmap = client.get(f"/api/scans/{scan_id}/heatmap")
    assert heatmap.status_code == 200
    assert heatmap.headers["content-type"] == "image/jpeg"

    assert not (main.service.artifact_dir / scan_id).exists(), "per-scan work dir must be deleted"
    assert main.service.heatmap_path(scan_id).exists()
    assert [s["id"] for s in client.get("/api/scans").json()] == [scan_id]


@needs_ffmpeg
def test_video_scan_names_every_signal_including_temporal(client: TestClient, tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=2", "-pix_fmt", "yuv420p", str(video)],
        check=True, timeout=60,
    )
    with (
        registry.override(MediaKind.IMAGE, StubDetector.for_kind(MediaKind.IMAGE, risk=0.2)),
        registry.override(MediaKind.AUDIO, StubDetector.for_kind(MediaKind.AUDIO, risk=0.2)),
        registry.override(MediaKind.VIDEO, StubDetector.for_kind(MediaKind.VIDEO, risk=0.3)),
    ):
        response = upload(client, "clip.mp4", video.read_bytes())
        assert response.status_code == 202, response.text
        scan = wait_for(client, response.json()["id"])
    assert scan["status"] == "completed", scan
    items = evidence_by_source(scan)
    assert items["video-temporal-model"]["model"]["display_name"] == "Stub video detector"
    assert items["video-frame-model"]["model"]["display_name"] == "Stub image detector"
    assert items["video-audio-model"]["status"] == "skipped"  # testsrc has no soundtrack
    assert items["video-audio-model"]["details"]["reason"]


# ── AI summary opt-in ───────────────────────────────────────────────────────


def test_ai_summary_is_opt_in_and_degrades_without_key(client: TestClient, image_detector: ImageStub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    assert client.get("/health").json()["ai_summary_available"] is False
    page = client.get("/").text
    assert 'id="generate-ai-summary" disabled' in page
    assert "never your media file" in page

    requested = wait_for(client, upload(client, "a.png", png_bytes(), generate_ai_summary="true").json()["id"])
    assert requested["report"]["ai_summary_requested"] is True
    assert requested["report"]["ai_summary_status"] == "unavailable"

    calls: list[Any] = []

    def fake_summary(evidence: list[dict[str, Any]], assessment: dict[str, Any]) -> str:
        calls.append(ai_summary._build_user_message(evidence, assessment))
        return "The detectors indicate an elevated signal."

    monkeypatch.setattr(service_module, "summaries_available", lambda: True)
    monkeypatch.setattr(ai_summary, "generate_summary", fake_summary)

    not_requested = wait_for(client, upload(client, "b.png", png_bytes()).json()["id"])
    assert not_requested["report"]["ai_summary_status"] == "not_requested"
    assert calls == []

    ready = wait_for(client, upload(client, "c.png", png_bytes(), generate_ai_summary="true").json()["id"], lambda s: s["status"] == "completed" and s["report"]["ai_summary_status"] != "pending")
    assert ready["report"]["ai_summary_status"] == "ready"
    assert ready["report"]["ai_summary"] == "The detectors indicate an elevated signal."
    # Only detector outputs are sent: no hash, filename or file metadata.
    sha256 = evidence_by_source(ready)["file-inspection"]["details"]["sha256"]
    assert len(calls) == 1 and sha256 not in calls[0] and "c.png" not in calls[0] and "exif" not in calls[0]
    assert "Stub image detector" in calls[0]


# ── errors ──────────────────────────────────────────────────────────────────


def test_detector_failure_returns_generic_message(client: TestClient) -> None:
    failing = StubDetector.for_kind(MediaKind.IMAGE, error=InferenceError(f"CUDA crash loading {SECRET}"))
    with registry.override(MediaKind.IMAGE, failing):
        response = upload(client, "a.png", png_bytes())
        scan = wait_for(client, response.json()["id"])
    assert scan["status"] == "failed"
    assert scan["error"] == "No detector could analyse this file."
    assert SECRET not in json.dumps(scan)


def test_unexpected_failure_is_not_leaked_or_persisted(client: TestClient, image_detector: ImageStub, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(f"ffmpeg: {SECRET}: Invalid data found")

    monkeypatch.setattr(service_module.pipeline, "analyze", explode)
    scan_id = upload(client, "a.png", png_bytes()).json()["id"]
    scan = wait_for(client, scan_id)
    assert scan["status"] == "failed"
    assert scan["error"] == GENERIC_SCAN_ERROR
    with sqlite3.connect(main.service.database_path) as db:
        stored = db.execute("SELECT error FROM scans WHERE id = ?", (scan_id,)).fetchone()[0]
    assert SECRET not in stored
    assert not (main.service.artifact_dir / scan_id).exists()


def test_invalid_uploads_are_rejected_with_safe_messages(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    bad_image = upload(client, "a.png", b"definitely not an image")
    assert bad_image.status_code == 400
    assert bad_image.json()["detail"] == "The file is not a valid JPG, PNG or WebP image."

    assert upload(client, "notes.pdf", b"%PDF-1.4").status_code == 400

    monkeypatch.setattr(settings, "MAX_IMAGE_PIXELS", 100)
    too_big = upload(client, "a.png", png_bytes((32, 32)))
    assert too_big.status_code == 400
    assert "megapixel" in too_big.json()["detail"]
    assert list(main.service.upload_dir.iterdir()) == [], "rejected uploads must be deleted"


@needs_ffmpeg
def test_audio_validation_uses_ffprobe_metadata(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    with registry.override(MediaKind.AUDIO, StubDetector.for_kind(MediaKind.AUDIO, risk=0.1)):
        accepted = upload(client, "voice.wav", wav_bytes(1.0))
        assert accepted.status_code == 202, accepted.text
        scan = wait_for(client, accepted.json()["id"])
        assert evidence_by_source(scan)["audio-model"]["model"]["display_name"] == "Stub audio detector"

        mismatched = upload(client, "voice.wav", png_bytes())
        assert mismatched.status_code == 400
        assert "uploads" not in mismatched.json()["detail"]

        monkeypatch.setattr(settings, "MAX_AUDIO_SECONDS", 0.5)
        too_long = upload(client, "voice.wav", wav_bytes(1.0))
        assert too_long.status_code == 400
        assert "limit" in too_long.json()["detail"]


# ── auth, scoping, rate limits ──────────────────────────────────────────────


def test_api_keys_scope_scans_to_their_owner(client: TestClient, image_detector: ImageStub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUSIONTRACE_API_KEYS", "key-a, key-b")
    a, b = {"X-API-Key": "key-a"}, {"X-API-Key": "key-b"}

    assert client.get("/api/scans").status_code == 401
    assert client.get("/api/scans", headers={"X-API-Key": "wrong"}).status_code == 401
    assert upload(client, "a.png", png_bytes()).status_code == 401
    health = client.get("/health")
    assert health.status_code == 200 and health.json()["auth_required"] is True

    response = upload(client, "a.png", png_bytes(), headers=a)
    assert response.status_code == 202
    scan_id = response.json()["id"]
    wait_for(client, scan_id, lambda s: s["status"] == "completed" and s["report"]["heatmap_status"] != "pending", headers=a)

    assert client.get("/api/scans", headers=b).json() == []
    assert client.get(f"/api/scans/{scan_id}", headers=b).status_code == 404
    assert client.get(f"/api/scans/{scan_id}/heatmap", headers=b).status_code == 404
    assert client.post(f"/api/scans/{scan_id}/feedback", json={"label": "needs_review"}, headers=b).status_code == 404

    assert [s["id"] for s in client.get("/api/scans", headers=a).json()] == [scan_id]
    assert client.get(f"/api/scans/{scan_id}/heatmap", headers=a).status_code == 200
    assert client.post(f"/api/scans/{scan_id}/feedback", json={"label": "needs_review"}, headers=a).status_code == 201

    with sqlite3.connect(main.service.database_path) as db:
        stored_owner = db.execute("SELECT owner FROM scans WHERE id = ?", (scan_id,)).fetchone()[0]
    assert stored_owner == owner_id("key-a") and "key-a" not in stored_owner

    monkeypatch.delenv("FUSIONTRACE_API_KEYS")
    assert client.get("/api/scans").status_code == 200  # auth off: local use needs nothing


def test_uploads_and_feedback_are_rate_limited(client: TestClient, image_detector: ImageStub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "upload_limiter", RateLimiter(1, 60))
    monkeypatch.setattr(main, "feedback_limiter", RateLimiter(1, 60))
    first = upload(client, "a.png", png_bytes())
    assert first.status_code == 202
    limited = upload(client, "b.png", png_bytes())
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1

    scan_id = first.json()["id"]
    wait_for(client, scan_id)
    assert client.post(f"/api/scans/{scan_id}/feedback", json={"label": "confirmed_fake"}).status_code == 201
    assert client.post(f"/api/scans/{scan_id}/feedback", json={"label": "confirmed_fake"}).status_code == 429


# ── queue, restart recovery, retention ──────────────────────────────────────


@pytest.mark.parametrize("workers", [1, 2])
def test_queue_never_exceeds_concurrency_cap(workers: int) -> None:
    lock = threading.Lock()
    state = {"active": 0, "peak": 0, "done": 0}

    def handler(scan_id: str) -> None:
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.03)
        with lock:
            state["active"] -= 1
            state["done"] += 1

    queue = ScanQueue(handler, workers=workers, max_pending=50)
    queue.start()
    for index in range(6):
        queue.submit(f"scan-{index}")
    queue.join()
    queue.stop()
    assert state["done"] == 6
    assert state["peak"] <= workers


def test_startup_marks_interrupted_scans_failed(tmp_path: Path) -> None:
    svc = ScanService(database_path=tmp_path / "test.db", upload_dir=tmp_path / "uploads", artifact_dir=tmp_path / "artifacts")
    svc.initialise()
    for scan_id, status in (("queued-1", "queued"), ("busy-1", "processing")):
        with sqlite3.connect(svc.database_path) as db:
            db.execute("INSERT INTO scans (id, created_at, filename, media_type, status) VALUES (?, ?, ?, ?, ?)", (scan_id, "2026-10-11T00:00:00Z", "a.png", "image", status))
        (svc.upload_dir / f"{scan_id}.png").write_bytes(b"x")
    (svc.artifact_dir / "busy-1").mkdir()

    svc.initialise()  # what the next process start does

    for scan_id in ("queued-1", "busy-1"):
        scan = svc.get_scan(scan_id)
        assert scan["status"] == "failed"
        assert scan["error"] == INTERRUPTED_SCAN_ERROR
    assert list(svc.upload_dir.iterdir()) == []
    assert not (svc.artifact_dir / "busy-1").exists()


def test_retention_sweep_skips_active_scans(tmp_path: Path) -> None:
    svc = ScanService(database_path=tmp_path / "test.db", upload_dir=tmp_path / "uploads", artifact_dir=tmp_path / "artifacts")
    svc.initialise()
    with sqlite3.connect(svc.database_path) as db:
        db.execute("INSERT INTO scans (id, created_at, filename, media_type, status) VALUES ('active', '2026-10-11T00:00:00Z', 'a.png', 'image', 'processing')")
    old = time.time() - (settings.RETENTION_HOURS + 1) * 3600
    active_file = svc.upload_dir / "active.png"
    stale_file = svc.upload_dir / "stale.png"
    stale_heatmap = svc.artifact_dir / "stale_heatmap.jpg"
    for path in (active_file, stale_file, stale_heatmap):
        path.write_bytes(b"x")
        os.utime(path, (old, old))

    svc.cleanup_expired_files()

    assert active_file.exists()
    assert not stale_file.exists()
    assert not stale_heatmap.exists()
