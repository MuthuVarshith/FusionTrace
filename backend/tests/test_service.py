"""Fast tests for deterministic scan/report logic; ML weights are not loaded."""

import unittest
import tempfile
import sqlite3
import json
from pathlib import Path
from unittest.mock import patch

from app.service import ScanService, fuse_evidence, media_type


class ServiceTests(unittest.TestCase):
    def test_media_type_routes_supported_extensions(self):
        self.assertEqual(media_type("photo.webp"), "image")
        self.assertEqual(media_type("voice.m4a"), "audio")
        self.assertEqual(media_type("clip.webm"), "video")

    def test_media_type_rejects_unknown_extension(self):
        with self.assertRaises(ValueError):
            media_type("document.pdf")

    def test_fusion_is_weighted_and_explained(self):
        result = fuse_evidence([
            {"label": "Technical check", "risk": 0.0, "weight": 0.1},
            {"label": "Image model", "risk": 0.8, "weight": 0.9},
        ])
        self.assertEqual(result["risk_score"], 72.0)
        self.assertEqual(result["risk_rating"], "high")
        self.assertIn("Image model", result["summary"])
        self.assertIn("Manual review", result["recommended_action"])

class ServiceDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        
        self.patcher_db = patch("app.service.DATABASE_PATH", self.db_path)
        self.patcher_db.start()
        
        self.service = ScanService()
        self.service.initialise()
        
        import contextlib
        # Insert a dummy scan to attach feedback to
        with contextlib.closing(self.service._connection()) as db:
            with db:
                db.execute("INSERT INTO scans (id, created_at, filename, media_type, status, report) VALUES (?, ?, ?, ?, ?, ?)",
                           ("scan1", "2026-08-20T00:00:00Z", "test.jpg", "image", "completed", json.dumps({"assessment": {"risk_rating": "low", "risk_score": 10.0}})))
                db.execute("INSERT INTO scans (id, created_at, filename, media_type, status, report) VALUES (?, ?, ?, ?, ?, ?)",
                           ("scan2", "2026-08-20T01:00:00Z", "test2.jpg", "image", "completed", json.dumps({"assessment": {"risk_rating": "high", "risk_score": 90.0}})))

    def tearDown(self):
        self.patcher_db.stop()
        self.temp_dir.cleanup()

    def test_add_feedback_valid(self):
        self.service.add_feedback("scan1", "confirmed_real", "looks good")
        scan = self.service.get_scan("scan1")
        self.assertEqual(len(scan["feedback_history"]), 1)
        self.assertEqual(scan["latest_feedback"]["label"], "confirmed_real")
        self.assertEqual(scan["latest_feedback"]["notes"], "looks good")

    def test_add_feedback_multiple_history_preserved(self):
        self.service.add_feedback("scan1", "needs_review", "first look")
        self.service.add_feedback("scan1", "confirmed_fake", "actually fake")
        scan = self.service.get_scan("scan1")
        self.assertEqual(len(scan["feedback_history"]), 2)
        # Should be ordered descending by created_at (which is current UTC time in service)
        # We can't guarantee order if they have the exact same timestamp string, but sqlite AUTOINCREMENT or insert order will keep it if we sleep, or just check contents
        self.assertEqual(scan["latest_feedback"]["label"], "confirmed_fake")

    def test_add_feedback_invalid_label_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.service.add_feedback("scan1", "not_a_valid_label", "hmm")
        self.assertIn("Invalid label", str(ctx.exception))

    def test_add_feedback_missing_scan_returns_404_equivalent(self):
        with self.assertRaises(ValueError) as ctx:
            self.service.add_feedback("missing_scan", "confirmed_real", "hmm")
        self.assertIn("Scan not found", str(ctx.exception))

    def test_get_recent_scans_list_order_and_limit(self):
        scans = self.service.get_recent_scans(limit=1)
        self.assertEqual(len(scans), 1)
        # scan2 was created later in setUp (01:00:00 vs 00:00:00)
        self.assertEqual(scans[0]["id"], "scan2")
        self.assertIsNone(scans[0]["latest_feedback"])

    def test_get_recent_scans_includes_latest_feedback(self):
        self.service.add_feedback("scan2", "confirmed_fake", "obvious fake")
        scans = self.service.get_recent_scans()
        self.assertEqual(scans[0]["id"], "scan2")
        self.assertIsNotNone(scans[0]["latest_feedback"])
        self.assertEqual(scans[0]["latest_feedback"]["label"], "confirmed_fake")

if __name__ == "__main__":
    unittest.main()
