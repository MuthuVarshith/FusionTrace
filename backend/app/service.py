"""Scan lifecycle, persistence and evidence fusion.

The service deliberately keeps ML verdicts separate from provenance/technical
signals. A verdict is an assessment, never proof that a file is fake.

Lifecycle: ``create_scan`` validates and stores the upload and inserts a
``queued`` row; a worker from ``app.jobs`` calls ``process_scan``, which moves
it to ``processing`` and then ``completed`` or ``failed``. The per-scan work
directory (``ARTIFACT_DIR/<scan_id>``) is deleted when processing ends, success
or failure; only the explanation heatmap (``ARTIFACT_DIR/<scan_id>_heatmap.jpg``)
is kept for the report until the retention sweep removes it.

Restart policy: scans still ``queued`` or ``processing`` when the process
starts were interrupted by a restart or crash. They are marked ``failed`` with
a client-safe message and their upload is deleted; they are not re-queued, so
a file that crashed the process cannot crash it again in a loop. Clients
upload the file again.

Errors: clients only ever see ``DetectorError.public_message`` or a generic
string. Exception text, paths and tool output go to the server log.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import mimetypes
import os
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from . import config as settings
from . import pipeline
from .ai_summary import summaries_available
from .config import ARTIFACT_DIR, DATABASE_PATH, RETENTION_HOURS, UPLOAD_DIR
from .detectors import registry
from .detectors.base import AnalysisContext, Detector, DetectorError, MediaKind
from .validation import (
    ALLOWED_EXTENSIONS,
    FileTooLarge,
    max_bytes,
    media_type,
    validate_upload,
)

logger = logging.getLogger("fusiontrace")

__all__ = ["ALLOWED_EXTENSIONS", "ScanService", "fuse_evidence", "media_type", "public_error"]

GENERIC_SCAN_ERROR = "Analysis failed. Please try again or use a different file."
INTERRUPTED_SCAN_ERROR = "The server restarted before this scan finished. Please upload the file again."
_SCHEMA_VERSION = 2
_CHUNK_BYTES = 1024 * 1024


class UploadMissingError(Exception):
    public_message = "The uploaded file is no longer available. Please upload it again."


def public_error(exc: BaseException) -> str:
    """The only error text that may reach a client for a failed scan."""
    if isinstance(exc, (DetectorError, UploadMissingError)):
        return exc.public_message
    return GENERIC_SCAN_ERROR


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _scan_id_of(entry: Path) -> str:
    """``<id>.jpg``, ``<id>`` and ``<id>_heatmap.jpg`` all belong to scan ``<id>`` (UUIDs contain no '_' or '.')."""
    return entry.name.split(".", 1)[0].split("_", 1)[0]


class ScanService:
    def __init__(self, database_path: Path | None = None, upload_dir: Path | None = None, artifact_dir: Path | None = None) -> None:
        # Module globals are read here, not as default arguments, so tests can patch them.
        self.database_path = Path(database_path or DATABASE_PATH)
        self.upload_dir = Path(upload_dir or UPLOAD_DIR)
        self.artifact_dir = Path(artifact_dir or ARTIFACT_DIR)

    # ── database ────────────────────────────────────────────────────────────

    def _connection(self) -> sqlite3.Connection:
        busy_ms = settings.SQLITE_BUSY_TIMEOUT_MS
        connection = sqlite3.connect(self.database_path, timeout=busy_ms / 1000)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(busy_ms)}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialise(self) -> None:
        for directory in (self.database_path.parent, self.upload_dir, self.artifact_dir):
            directory.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(self._connection()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS scans (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, completed_at TEXT,
                filename TEXT NOT NULL, media_type TEXT NOT NULL, status TEXT NOT NULL,
                error TEXT, report TEXT NOT NULL DEFAULT '{}')""")
            columns = {column[1] for column in db.execute("PRAGMA table_info(scans)")}
            if "generate_ai_summary" not in columns:
                db.execute("ALTER TABLE scans ADD COLUMN generate_ai_summary INTEGER NOT NULL DEFAULT 0")
            if "owner" not in columns:
                # Hash of the caller's API key (see app.auth); NULL when auth was off.
                db.execute("ALTER TABLE scans ADD COLUMN owner TEXT")
            db.execute("CREATE INDEX IF NOT EXISTS idx_scans_owner_created_at ON scans(owner, created_at DESC)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_scans_status ON scans(status)")
            db.execute("""CREATE TABLE IF NOT EXISTS scan_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id TEXT NOT NULL,
                label TEXT NOT NULL CHECK(label IN ('confirmed_real', 'confirmed_fake', 'needs_review')),
                notes TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(scan_id) REFERENCES scans(id)
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_scan_feedback_scan_id_created_at ON scan_feedback(scan_id, created_at DESC)")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version < _SCHEMA_VERSION:
                # v1 stored raw exception text (paths, tool output) in scans.error. Scrub it.
                db.execute("UPDATE scans SET error = ? WHERE error IS NOT NULL", (GENERIC_SCAN_ERROR,))
                db.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        self.recover_interrupted_scans()
        self.cleanup_expired_files()
        # Pre-warm the configured detectors so the first scan doesn't pay the
        # cold-load penalty. FUSIONTRACE_WARMUP=0 skips it (tests, low memory).
        if os.getenv("FUSIONTRACE_WARMUP", "1") != "0":
            threading.Thread(target=registry.warm_up, name="detector-warmup", daemon=True).start()

    def recover_interrupted_scans(self) -> list[str]:
        """Mark scans left queued/processing by a previous process as failed (see module docstring)."""
        with contextlib.closing(self._connection()) as db, db:
            ids = [row["id"] for row in db.execute("SELECT id FROM scans WHERE status IN ('queued', 'processing')")]
            if ids:
                db.execute(
                    "UPDATE scans SET status = 'failed', completed_at = ?, error = ? WHERE status IN ('queued', 'processing')",
                    (utc_now(), INTERRUPTED_SCAN_ERROR),
                )
        for scan_id in ids:
            self._delete_upload(scan_id)
            shutil.rmtree(self.artifact_dir / scan_id, ignore_errors=True)
        if ids:
            logger.warning("Marked %d scan(s) interrupted by a restart as failed", len(ids))
        return ids

    # ── upload ──────────────────────────────────────────────────────────────

    async def create_scan(self, upload: UploadFile, generate_ai_summary: bool = False, owner: str | None = None) -> dict[str, Any]:
        """Store and validate an upload, then insert a queued scan row. Raises ``UploadRejected``."""
        self.cleanup_expired_files()
        filename = Path(upload.filename or "upload").name[:255] or "upload"
        kind = media_type(filename)
        limit = max_bytes(kind)
        scan_id = str(uuid4())
        destination = self.upload_dir / f"{scan_id}{Path(filename).suffix.lower()}"
        total = 0
        try:
            with destination.open("wb") as output:
                while chunk := await upload.read(_CHUNK_BYTES):
                    total += len(chunk)
                    if total > limit:
                        raise FileTooLarge(f"File exceeds the {limit // (1024 * 1024)} MB limit for {kind} uploads.")
                    output.write(chunk)
            # ffprobe/PIL are blocking; keep them off the event loop.
            await asyncio.to_thread(validate_upload, destination, kind)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        with contextlib.closing(self._connection()) as db, db:
            db.execute(
                "INSERT INTO scans (id, created_at, filename, media_type, status, generate_ai_summary, owner) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (scan_id, utc_now(), filename, kind, "queued", int(generate_ai_summary), owner),
            )
        return {"id": scan_id, "filename": filename, "media_type": kind, "status": "queued", "generate_ai_summary": generate_ai_summary}

    def _upload_path(self, scan_id: str) -> Path:
        path = next(self.upload_dir.glob(f"{scan_id}.*"), None)
        if path is None:
            raise UploadMissingError(f"upload for scan {scan_id} not found in {self.upload_dir}")
        return path

    def _delete_upload(self, scan_id: str) -> None:
        for path in self.upload_dir.glob(f"{scan_id}.*"):
            path.unlink(missing_ok=True)

    def heatmap_path(self, scan_id: str) -> Path:
        return self.artifact_dir / f"{scan_id}_heatmap.jpg"

    def fail_scan(self, scan_id: str, message: str = GENERIC_SCAN_ERROR) -> None:
        """Mark a scan failed with a client-safe ``message``."""
        with contextlib.closing(self._connection()) as db, db:
            db.execute("UPDATE scans SET status = ?, completed_at = ?, error = ? WHERE id = ?", ("failed", utc_now(), message, scan_id))

    # ── processing (runs on a worker thread) ────────────────────────────────

    def process_scan(self, scan_id: str) -> None:
        with contextlib.closing(self._connection()) as db, db:
            row = db.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
            if row is None:
                return
            claimed = db.execute("UPDATE scans SET status = 'processing' WHERE id = ? AND status = 'queued'", (scan_id,)).rowcount
        if not claimed:
            logger.info("Scan %s is no longer queued (status %s); skipping", scan_id, row["status"])
            return
        work_dir = self.artifact_dir / scan_id
        kind = row["media_type"]
        try:
            path = self._upload_path(scan_id)
            timeout = settings.SCAN_TIMEOUT_SECONDS
            deadline = time.monotonic() + timeout if timeout > 0 else None
            evidence = [self._file_evidence(path)]
            if kind == "image":
                evidence.append(self._image_metadata_evidence(path))
            evidence.extend(pipeline.analyze(kind, path, work_dir, deadline=deadline))
            assessment = fuse_evidence(evidence)
            explainer = self._image_explainer() if kind == "image" else None
            summary_requested = bool(row["generate_ai_summary"])
            summary_status = "not_requested"
            if summary_requested:
                summary_status = "pending" if summaries_available() else "unavailable"
            # ── Phase 1: save verdict immediately so the UI can render ──────
            report = {
                "scan_id": scan_id, "media_type": kind,
                "evidence": evidence, "assessment": assessment,
                "ai_summary": None,
                "ai_summary_requested": summary_requested,
                "ai_summary_status": summary_status,
                "heatmap_ready": False,
                "heatmap_status": "pending" if explainer is not None else "not_applicable",
                "heatmap_model": None,
                "retention_hours": RETENTION_HOURS,
            }
            with contextlib.closing(self._connection()) as db, db:
                db.execute(
                    "UPDATE scans SET status = ?, completed_at = ?, report = ?, error = NULL WHERE id = ?",
                    ("completed", utc_now(), json.dumps(report), scan_id),
                )
        except Exception as exc:
            if isinstance(exc, (DetectorError, UploadMissingError)):
                logger.warning("Scan %s failed: %s: %s", scan_id, type(exc).__name__, exc)
            else:
                logger.exception("Scan %s failed with an unexpected error", scan_id)
            self.fail_scan(scan_id, public_error(exc))
        else:
            # ── Phase 2: enrichments (scan already 'completed' in DB) ────────
            self._enrich_scan(scan_id, evidence, assessment, path if explainer is not None else None, summary_status == "pending")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _file_evidence(self, path: Path) -> dict[str, Any]:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
                digest.update(chunk)
        return {
            "source": "file-inspection", "label": "File inspection", "model": None, "status": "ok",
            "risk": 0.0, "weight": 0.1,
            "details": {"size_bytes": path.stat().st_size, "mime_type": mimetypes.guess_type(path.name)[0], "sha256": digest.hexdigest()},
        }

    def _image_metadata_evidence(self, path: Path) -> dict[str, Any]:
        from PIL import Image

        with Image.open(path) as image:
            metadata = {str(key): str(value)[:200] for key, value in image.getexif().items()}
            details = {"format": image.format, "dimensions": [image.width, image.height], "exif_present": bool(metadata), "exif": metadata}
        return {"source": "image-metadata", "label": "Image metadata inspection", "model": None, "status": "ok", "risk": None, "weight": 0, "details": details}

    @staticmethod
    def _image_explainer() -> Detector | None:
        """The configured image detector if it can produce a visual explanation."""
        try:
            detector = registry.get(MediaKind.IMAGE)
        except DetectorError:
            return None
        return detector if detector is not None and detector.supports_explanation else None

    def _patch_report(self, scan_id: str, **fields: Any) -> None:
        """Update specific fields inside the JSON report column without touching other fields."""
        with contextlib.closing(self._connection()) as db, db:
            row = db.execute("SELECT report FROM scans WHERE id = ?", (scan_id,)).fetchone()
            if row is None:
                return
            report = json.loads(row["report"] or "{}")
            if not report:
                return
            report.update(fields)
            db.execute("UPDATE scans SET report = ? WHERE id = ?", (json.dumps(report), scan_id))

    def _enrich_scan(
        self,
        scan_id: str,
        evidence: list[dict[str, Any]],
        assessment: dict[str, Any],
        image_path: Path | None,
        generate_ai_summary: bool,
    ) -> None:
        """Phase-2 enrichments: explanation heatmap + AI summary.

        Runs after the scan is already marked 'completed' in the DB, so the
        UI can render the verdict immediately. Each enrichment is guarded
        independently; a failure of one does not prevent the other.
        """
        # ── Explanation heatmap (image scans with an explaining detector) ───
        if image_path is not None:
            heatmap_status = "unavailable"
            try:
                detector = self._image_explainer()
                if detector is not None:
                    heatmap_path = self.heatmap_path(scan_id)
                    ctx = AnalysisContext(work_dir=self.artifact_dir / scan_id, options=registry.settings(MediaKind.IMAGE).options)
                    written = detector.explain(image_path, heatmap_path, ctx)
                    if written is not None and Path(written).exists():
                        if Path(written) != heatmap_path:
                            shutil.move(str(written), heatmap_path)
                        self._patch_report(scan_id, heatmap_ready=True, heatmap_status="ready", heatmap_model=detector.info.to_report())
                        heatmap_status = "ready"
            except Exception:
                logger.exception("Explanation enrichment failed for scan %s", scan_id)
            if heatmap_status != "ready":
                self._patch_report(scan_id, heatmap_status=heatmap_status)

        # ── AI forensic summary (opt-in per scan; detector scores only) ─────
        if generate_ai_summary:
            try:
                from .ai_summary import generate_summary

                summary = generate_summary(evidence, assessment)
                if summary:
                    self._patch_report(scan_id, ai_summary=summary, ai_summary_status="ready")
                else:
                    self._patch_report(scan_id, ai_summary_status="unavailable")
            except Exception:
                logger.exception("AI summary enrichment failed for scan %s", scan_id)
                self._patch_report(scan_id, ai_summary_status="unavailable")

    # ── reads ───────────────────────────────────────────────────────────────

    def get_scan(self, scan_id: str, owner: str | None = None) -> dict[str, Any] | None:
        """Return a scan, or None if it does not exist or belongs to someone else.

        ``owner`` is None when authentication is off; then every scan is visible.
        """
        with contextlib.closing(self._connection()) as db:
            if owner is None:
                row = db.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
            else:
                row = db.execute("SELECT * FROM scans WHERE id = ? AND owner = ?", (scan_id, owner)).fetchone()
            if row is None:
                return None
            feedback_rows = db.execute("SELECT label, notes, created_at FROM scan_feedback WHERE scan_id = ? ORDER BY created_at DESC, id DESC", (scan_id,)).fetchall()

        report = json.loads(row["report"] or "{}")
        feedback_history = [{"label": f["label"], "notes": f["notes"], "created_at": f["created_at"]} for f in feedback_rows]
        latest_feedback = feedback_history[0] if feedback_history else None

        return {
            "id": row["id"], "created_at": row["created_at"], "completed_at": row["completed_at"],
            "filename": row["filename"], "media_type": row["media_type"], "status": row["status"],
            "error": row["error"], "report": report or None,
            "feedback_history": feedback_history, "latest_feedback": latest_feedback,
        }

    def get_recent_scans(self, limit: int = 50, owner: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        where = "" if owner is None else "WHERE s.owner = ?"
        params: tuple[Any, ...] = (limit,) if owner is None else (owner, limit)
        with contextlib.closing(self._connection()) as db:
            rows = db.execute(f"""
                SELECT s.id, s.filename, s.media_type, s.created_at, s.status, s.report,
                       (SELECT json_object('label', f.label, 'notes', f.notes, 'created_at', f.created_at)
                        FROM scan_feedback f WHERE f.scan_id = s.id ORDER BY f.created_at DESC, f.id DESC LIMIT 1) as latest_feedback
                FROM scans s
                {where}
                ORDER BY s.created_at DESC
                LIMIT ?
            """, params).fetchall()

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
                "latest_feedback": latest_feedback,
            })
        return results

    def add_feedback(self, scan_id: str, label: str, notes: str | None = None, owner: str | None = None) -> None:
        with contextlib.closing(self._connection()) as db:
            if owner is not None and db.execute("SELECT 1 FROM scans WHERE id = ? AND owner = ?", (scan_id, owner)).fetchone() is None:
                raise ValueError("Scan not found")
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
                entry["display_name"] = detector.info.display_name if detector is not None else None
                entry["heuristic"] = bool(detector is not None and detector.info.heuristic)
                entry["available"] = detector is not None and detector.is_available()
                entry["loaded"] = detector is not None and detector.loaded
            except DetectorError:
                entry["available"] = False
                entry["loaded"] = False
            detectors[kind.value] = entry
        return {
            "status": "ok",
            "detectors": detectors,
            "retention_hours": RETENTION_HOURS,
            "ai_summary_available": summaries_available(),
            "ffmpeg_available": shutil.which("ffmpeg") is not None,
            "ffprobe_available": shutil.which("ffprobe") is not None,
            "max_concurrent_scans": settings.MAX_CONCURRENT_SCANS,
        }

    # ── retention ───────────────────────────────────────────────────────────

    def _active_scan_ids(self) -> set[str]:
        with contextlib.closing(self._connection()) as db:
            return {row["id"] for row in db.execute("SELECT id FROM scans WHERE status IN ('queued', 'processing')")}

    def cleanup_expired_files(self) -> None:
        """Delete stored uploads and artifacts older than the retention window.

        Files belonging to scans that are still queued or processing are never
        touched, however old they are.
        """
        cutoff = time.time() - RETENTION_HOURS * 3600
        active = self._active_scan_ids()
        expired_heatmaps: list[str] = []
        for directory in (self.upload_dir, self.artifact_dir):
            if not directory.exists():
                continue
            for entry in list(directory.iterdir()):
                scan_id = _scan_id_of(entry)
                if scan_id in active:
                    continue
                try:
                    if entry.stat().st_mtime >= cutoff:
                        continue
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        entry.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("Retention sweep could not remove %s: %s", entry.name, exc)
                    continue
                if entry.name.endswith("_heatmap.jpg"):
                    expired_heatmaps.append(scan_id)
        for scan_id in expired_heatmaps:
            self._patch_report(scan_id, heatmap_ready=False, heatmap_status="expired")
