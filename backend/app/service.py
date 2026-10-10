"""Scan lifecycle, persistence and evidence fusion.

The service deliberately keeps ML verdicts separate from provenance/technical
signals. A verdict is an assessment, never proof that a file is fake.
"""

import json
import mimetypes
import os
import shutil
import sqlite3
import subprocess
import hashlib
import contextlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from . import pipeline
from .config import ARTIFACT_DIR, DATABASE_PATH, MAX_UPLOAD_MB, RETENTION_HOURS, UPLOAD_DIR
from .detectors import registry
from .detectors.base import AnalysisContext, DetectorError, MediaKind

ALLOWED_EXTENSIONS = {"image": {".jpg", ".jpeg", ".png", ".webp"}, "audio": {".wav", ".mp3", ".m4a"}, "video": {".mp4", ".mov", ".webm"}}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def media_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    for kind, suffixes in ALLOWED_EXTENSIONS.items():
        if suffix in suffixes:
            return kind
    raise ValueError("Unsupported file type. Use an image (JPG/PNG/WebP), audio (WAV/MP3/M4A), or video (MP4/MOV/WebM).")


def fuse_evidence(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a transparent weighted risk assessment from detector signals."""
    weighted = [(item["risk"], item["weight"]) for item in evidence if item.get("risk") is not None]
    score = round(sum(risk * weight for risk, weight in weighted) / sum(weight for _, weight in weighted) * 100, 1) if weighted else 0.0
    rating = "high" if score >= 70 else "medium" if score >= 40 else "low"
    contributing_sources = [item["label"] for item in evidence if item.get("risk") is not None and item["risk"] >= 0.4]
    basis = ", ".join(contributing_sources) if contributing_sources else "no elevated detector signals"
    action = {
        "high": "Manual review required before taking action on this media.",
        "medium": "Inconclusive automated result; seek additional evidence or manual review.",
        "low": "No elevated automated signal; this is not verification of authenticity.",
    }[rating]
    return {"risk_score": score, "risk_rating": rating, "recommended_action": action, "summary": f"The assessment is based on {basis}. It is not proof of authenticity or manipulation."}


class ScanService:
    def initialise(self) -> None:
        with contextlib.closing(sqlite3.connect(DATABASE_PATH)) as db:
            with db:
                db.execute("""CREATE TABLE IF NOT EXISTS scans (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, completed_at TEXT,
                    filename TEXT NOT NULL, media_type TEXT NOT NULL, status TEXT NOT NULL,
                    error TEXT, report TEXT NOT NULL DEFAULT '{}')""")
                columns = {column[1] for column in db.execute("PRAGMA table_info(scans)")}
                if "generate_ai_summary" not in columns:
                    db.execute("ALTER TABLE scans ADD COLUMN generate_ai_summary INTEGER NOT NULL DEFAULT 0")
                db.execute("""CREATE TABLE IF NOT EXISTS scan_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id TEXT NOT NULL,
                    label TEXT NOT NULL CHECK(label IN ('confirmed_real', 'confirmed_fake', 'needs_review')),
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(scan_id) REFERENCES scans(id)
                )""")
                db.execute("CREATE INDEX IF NOT EXISTS idx_scan_feedback_scan_id_created_at ON scan_feedback(scan_id, created_at DESC)")
        self.cleanup_expired_files()
        # Pre-warm the configured detectors so the first scan doesn't pay the
        # cold-load penalty. FUSIONTRACE_WARMUP=0 skips it (tests, low memory).
        if os.getenv("FUSIONTRACE_WARMUP", "1") != "0":
            import threading
            threading.Thread(target=registry.warm_up, daemon=True).start()

    def _connection(self):
        connection = sqlite3.connect(DATABASE_PATH)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    async def create_scan(self, upload: UploadFile, generate_ai_summary: bool = False) -> dict[str, Any]:
        self.cleanup_expired_files()
        filename = Path(upload.filename or "upload").name
        kind = media_type(filename)
        scan_id = str(uuid4())
        destination = UPLOAD_DIR / f"{scan_id}{Path(filename).suffix.lower()}"
        total = 0
        with destination.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_MB * 1024 * 1024:
                    output.close()
                    destination.unlink(missing_ok=True)
                    raise ValueError(f"File exceeds the {MAX_UPLOAD_MB} MB limit.")
                output.write(chunk)
        try:
            self._validate_upload(destination, kind)
        except ValueError:
            destination.unlink(missing_ok=True)
            raise
        with contextlib.closing(self._connection()) as db:
            with db:
                db.execute(
                    "INSERT INTO scans (id, created_at, filename, media_type, status, generate_ai_summary) VALUES (?, ?, ?, ?, ?, ?)",
                    (scan_id, utc_now(), filename, kind, "queued", int(generate_ai_summary)),
                )
        return {"id": scan_id, "filename": filename, "media_type": kind, "status": "queued", "generate_ai_summary": generate_ai_summary}

    @staticmethod
    def _validate_upload(path: Path, kind: str) -> None:
        """Validate file contents, rather than trusting a user-controlled filename."""
        try:
            if kind == "image":
                from PIL import Image
                with Image.open(path) as image:
                    image.verify()
                return
            if kind == "audio" and path.suffix.lower() == ".wav":
                import soundfile as sf
                info = sf.info(path)
                if info.frames <= 0 or info.samplerate <= 0:
                    raise ValueError("WAV contains no readable audio frames")
                return
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise ValueError("Server media validator is unavailable")
            completed = subprocess.run(
                [ffmpeg, "-v", "error", "-i", str(path), "-map", "0", "-f", "null", "-"],
                capture_output=True, timeout=30, check=False,
            )
            if completed.returncode != 0:
                raise ValueError("file contents cannot be decoded as the declared media type")
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
            raise ValueError(f"Invalid {kind} upload: {exc}") from exc

    def process_scan(self, scan_id: str) -> None:
        with contextlib.closing(self._connection()) as db:
            with db:
                row = db.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
                if row is None:
                    return
                db.execute("UPDATE scans SET status = ? WHERE id = ?", ("processing", scan_id))
        try:
            path = next(UPLOAD_DIR.glob(f"{scan_id}.*"))
            evidence = [self._file_evidence(path)]
            if row["media_type"] == "image":
                evidence.append(self._image_metadata_evidence(path))
            evidence.extend(pipeline.analyze(row["media_type"], path, ARTIFACT_DIR / scan_id))
            assessment = fuse_evidence(evidence)
            # ── Phase 1: save verdict immediately so the UI can render ──────
            report = {
                "scan_id": scan_id, "media_type": row["media_type"],
                "evidence": evidence, "assessment": assessment,
                "ai_summary": None,
                "ai_summary_requested": bool(row["generate_ai_summary"]),
                "ai_summary_status": "pending" if row["generate_ai_summary"] else "not_requested",
                "heatmap_ready": False,
                "retention_hours": RETENTION_HOURS,
            }
            with contextlib.closing(self._connection()) as db:
                with db:
                    db.execute(
                        "UPDATE scans SET status = ?, completed_at = ?, report = ? WHERE id = ?",
                        ("completed", utc_now(), json.dumps(report), scan_id),
                    )
            # ── Phase 2: enrichments (scan already 'completed' in DB) ────────
            image_path = path if row["media_type"] == "image" else None
            self._enrich_scan(scan_id, evidence, assessment, image_path, bool(row["generate_ai_summary"]))
        except Exception as exc:
            with contextlib.closing(self._connection()) as db:
                with db:
                    db.execute("UPDATE scans SET status = ?, completed_at = ?, error = ? WHERE id = ?", ("failed", utc_now(), str(exc), scan_id))

    def _file_evidence(self, path: Path) -> dict[str, Any]:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"source": "file-inspection", "label": "File inspection", "risk": 0.0, "weight": 0.1, "details": {"size_bytes": path.stat().st_size, "mime_type": mimetypes.guess_type(path.name)[0], "sha256": digest.hexdigest()}}

    def _image_metadata_evidence(self, path: Path) -> dict[str, Any]:
        from PIL import Image
        with Image.open(path) as image:
            metadata = {str(key): str(value)[:200] for key, value in image.getexif().items()}
            details = {"format": image.format, "dimensions": [image.width, image.height], "exif_present": bool(metadata), "exif": metadata}
        return {"source": "image-metadata", "label": "Image metadata inspection", "risk": None, "weight": 0, "details": details}

    def _patch_report(self, scan_id: str, **fields: Any) -> None:
        """Update specific fields inside the JSON report column without touching other fields."""
        with contextlib.closing(self._connection()) as db:
            row = db.execute("SELECT report FROM scans WHERE id = ?", (scan_id,)).fetchone()
            if row is None:
                return
            report = json.loads(row["report"])
            report.update(fields)
            with db:
                db.execute("UPDATE scans SET report = ? WHERE id = ?", (json.dumps(report), scan_id))

    def _enrich_scan(
        self,
        scan_id: str,
        evidence: list[dict[str, Any]],
        assessment: dict[str, Any],
        image_path: Path | None,
        generate_ai_summary: bool,
    ) -> None:
        """Phase-2 enrichments: GradCAM heatmap + AI summary.

        Runs after the scan is already marked 'completed' in the DB, so the
        UI can render the verdict immediately while these run in the background.
        Each enrichment is independently guarded — failure of one does not
        prevent the other from running.
        """
        from .config import logger

        # ── GradCAM heatmap (image scans only) ────────────────────────────────
        if image_path is not None:
            try:
                detector = registry.get(MediaKind.IMAGE)
                if detector is not None and detector.supports_explanation:
                    heatmap_path = ARTIFACT_DIR / f"{scan_id}_heatmap.jpg"
                    ctx = AnalysisContext(work_dir=ARTIFACT_DIR / scan_id, options=registry.settings(MediaKind.IMAGE).options)
                    if detector.explain(image_path, heatmap_path, ctx):
                        self._patch_report(scan_id, heatmap_ready=True, heatmap_model=detector.info.to_report())
            except (DetectorError, OSError) as exc:
                logger.warning("Explanation enrichment failed: %s", exc)

        # ── AI forensic summary ───────────────────────────────────────────────
        if generate_ai_summary:
            try:
                from .ai_summary import generate_summary
                summary = generate_summary(evidence, assessment)
                if summary:
                    self._patch_report(scan_id, ai_summary=summary, ai_summary_status="ready")
                else:
                    self._patch_report(scan_id, ai_summary_status="unavailable")
            except Exception as exc:
                logger.warning("AI summary enrichment failed: %s", exc)
                self._patch_report(scan_id, ai_summary_status="unavailable")

    def get_scan(self, scan_id: str) -> dict[str, Any] | None:
        with contextlib.closing(self._connection()) as db:
            row = db.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
            if row is None:
                return None
            feedback_rows = db.execute("SELECT label, notes, created_at FROM scan_feedback WHERE scan_id = ? ORDER BY created_at DESC", (scan_id,)).fetchall()
            
        report = json.loads(row["report"])
        feedback_history = [{"label": f["label"], "notes": f["notes"], "created_at": f["created_at"]} for f in feedback_rows]
        latest_feedback = feedback_history[0] if feedback_history else None
        
        return {
            "id": row["id"], "created_at": row["created_at"], "completed_at": row["completed_at"],
            "filename": row["filename"], "media_type": row["media_type"], "status": row["status"],
            "error": row["error"], "report": report or None,
            "feedback_history": feedback_history, "latest_feedback": latest_feedback
        }

    def get_recent_scans(self, limit: int = 50) -> list[dict[str, Any]]:
        with contextlib.closing(self._connection()) as db:
            rows = db.execute("""
                SELECT s.id, s.filename, s.media_type, s.created_at, s.status, s.report,
                       (SELECT json_object('label', f.label, 'notes', f.notes, 'created_at', f.created_at) 
                        FROM scan_feedback f WHERE f.scan_id = s.id ORDER BY f.created_at DESC LIMIT 1) as latest_feedback
                FROM scans s
                ORDER BY s.created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        
        results = []
        for row in rows:
            report_data = json.loads(row["report"]) if row["report"] else {}
            assessment = report_data.get("assessment", {})
            latest_feedback = json.loads(row["latest_feedback"]) if row["latest_feedback"] else None
            results.append({
                "id": row["id"],
                "filename": row["filename"],
                "media_type": row["media_type"],
                "created_at": row["created_at"],
                "status": row["status"],
                "risk_rating": assessment.get("risk_rating"),
                "risk_score": assessment.get("risk_score"),
                "latest_feedback": latest_feedback
            })
        return results

    def add_feedback(self, scan_id: str, label: str, notes: str | None = None) -> None:
        with contextlib.closing(self._connection()) as db:
            try:
                with db:
                    db.execute("INSERT INTO scan_feedback (scan_id, label, notes, created_at) VALUES (?, ?, ?, ?)", (scan_id, label, notes, utc_now()))
            except sqlite3.IntegrityError as e:
                if "FOREIGN KEY" in str(e).upper():
                    raise ValueError("Scan not found") from e
                if "CHECK constraint failed" in str(e):
                    raise ValueError("Invalid label") from e
                raise

    def health(self) -> dict[str, Any]:
        detectors: dict[str, Any] = {}
        for kind in MediaKind:
            current = registry.settings(kind)
            entry: dict[str, Any] = {"enabled": current.enabled, "model": current.model, "weight": current.weight}
            try:
                detector = registry.get(kind)
                entry["available"] = detector is not None and detector.is_available()
                entry["loaded"] = detector is not None and detector.loaded
            except DetectorError:
                entry["available"] = False
                entry["loaded"] = False
            detectors[kind.value] = entry
        return {"status": "ok", "detectors": detectors, "retention_hours": RETENTION_HOURS}

    def cleanup_expired_files(self) -> None:
        """Delete stored source files and artifacts older than the configured retention window."""
        cutoff = datetime.now(timezone.utc).timestamp() - RETENTION_HOURS * 3600
        for directory in (UPLOAD_DIR, ARTIFACT_DIR):
            for entry in directory.iterdir():
                if entry.stat().st_mtime >= cutoff:
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink(missing_ok=True)
