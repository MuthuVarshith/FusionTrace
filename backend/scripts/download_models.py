"""Download detector weights from the Hugging Face Hub into the model directory.

Every registered detector declares the weight bundles it needs through
``Detector.weights()`` (a list of ``WeightsSpec``). This script resolves which
detectors are wanted, then fetches each bundle into
``<model_dir>/<spec.local_subdir>`` with ``huggingface_hub.snapshot_download``.

Usage (from ``backend/``)::

    python scripts/download_models.py                 # detectors enabled by config.yaml + env
    python scripts/download_models.py --all           # every registered detector
    python scripts/download_models.py --kind image    # restrict to one or more kinds
    python scripts/download_models.py --dry-run       # show the plan, download nothing
    python scripts/download_models.py --refresh       # re-sync completed bundles with the Hub

Environment:
    FUSIONTRACE_MODEL_DIR   root for weights (default: ``model_dir`` in config.yaml, i.e. backend/models)
    FUSIONTRACE_<KIND>_ENABLED / _MODEL   select detectors, exactly as the app does
    HF_TOKEN                optional; needed only for private or gated repos
    <spec.repo_id_env>      optional per-bundle repo override (e.g. HF_REPO_ID for a private mirror)

Idempotent: a completed bundle leaves a ``.fusiontrace-download.json`` marker in
its directory, and later runs skip it without touching the network while the
requested repo, revision and patterns are unchanged. Pin ``revision`` to a
commit sha in the detector's ``WeightsSpec`` for reproducible builds.

Exit status: 0 when every wanted bundle is present, 1 when any failed or a
configured detector is not registered.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.detectors import registry  # noqa: E402  (needs BACKEND_DIR on sys.path)
from app.detectors.base import Detector, MediaKind, WeightsSpec  # noqa: E402

MARKER_NAME = ".fusiontrace-download.json"

logger = logging.getLogger("fusiontrace.download")


@dataclass(frozen=True)
class DownloadJob:
    """One weight bundle to fetch, after env overrides are applied."""

    repo_id: str
    repo_type: str
    revision: str | None
    allow_patterns: tuple[str, ...] | None
    local_subdir: str
    target: Path

    @property
    def marker_path(self) -> Path:
        return self.target / MARKER_NAME

    def marker(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": self.revision,
            "allow_patterns": list(self.allow_patterns) if self.allow_patterns is not None else None,
        }

    def is_complete(self) -> bool:
        """True when a previous run finished this exact bundle and its files are still there."""
        try:
            recorded = json.loads(self.marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if recorded != self.marker():
            return False
        return any(path.name != MARKER_NAME for path in self.target.iterdir())

    def describe(self) -> str:
        where = f"{self.repo_type}:{self.repo_id}" if self.repo_type != "model" else self.repo_id
        revision = f"@{self.revision}" if self.revision else ""
        patterns = f" [{', '.join(self.allow_patterns)}]" if self.allow_patterns else ""
        return f"{where}{revision}{patterns} -> {self.local_subdir}/"


# ── Selection ───────────────────────────────────────────────────────────────


def select_detectors(kinds: Sequence[MediaKind], include_all: bool) -> tuple[list[type[Detector]], list[str]]:
    """Detector classes whose weights are wanted, plus configuration errors."""
    errors: list[str] = []
    if include_all:
        return [cls for cls in registry.all_detector_classes() if cls.info.kind in kinds], errors

    selected: list[type[Detector]] = []
    for kind in kinds:
        try:
            current = registry.settings(kind)
            if not current.enabled:
                logger.info("%-5s  disabled by configuration; skipping", kind.value)
                continue
            if current.model is None:
                logger.info("%-5s  no detector registered yet; nothing to download", kind.value)
                continue
            detectors, _ = registry.catalogue(kind)
        except Exception as exc:  # a broken detector package must not hide the other kinds
            errors.append(f"{kind.value}: could not read detector configuration: {exc}")
            continue
        cls = detectors.get(current.model)
        if cls is None:
            errors.append(
                f"{kind.value}: configured model {current.model!r} is not registered; available: {sorted(detectors)}"
            )
            continue
        selected.append(cls)
    return selected, errors


def resolve_repo_id(spec: WeightsSpec) -> str:
    if spec.repo_id_env:
        override = os.getenv(spec.repo_id_env, "").strip()
        if override:
            return override
    return spec.repo_id


def plan_jobs(classes: Iterable[type[Detector]], root: Path) -> tuple[list[DownloadJob], list[str]]:
    """Turn detector classes into de-duplicated download jobs under ``root``."""
    jobs: dict[str, DownloadJob] = {}
    errors: list[str] = []
    root = root.resolve()
    for cls in classes:
        for spec in cls.weights():
            repo_id = resolve_repo_id(spec)
            if not repo_id:
                # Legacy bundles live only in a private mirror named by repo_id_env.
                logger.info("skipped   %s: no repo id (set %s to fetch it)", cls.info.name, spec.repo_id_env or "a repo id")
                continue
            target = (root / spec.local_subdir).resolve()
            # A mirror bundle may mirror the model dir's own layout (local_subdir "." or ""),
            # so the root itself is allowed; anything outside it is not.
            if target != root and root not in target.parents:
                errors.append(f"{cls.info.name}: local_subdir {spec.local_subdir!r} must be inside the model dir")
                continue
            job = DownloadJob(
                repo_id=repo_id,
                repo_type=spec.repo_type,
                revision=spec.revision,
                allow_patterns=tuple(spec.allow_patterns) if spec.allow_patterns is not None else None,
                local_subdir=spec.local_subdir,
                target=target,
            )
            key = str(target)
            existing = jobs.get(key)
            if existing is None:
                jobs[key] = job
            elif existing != job:
                errors.append(
                    f"{cls.info.name}: {spec.local_subdir!r} is already claimed by a different bundle "
                    f"({existing.describe()}); give each bundle its own local_subdir"
                )
    return list(jobs.values()), errors


# ── Download ────────────────────────────────────────────────────────────────


def fetch(job: DownloadJob, token: str | None, force: bool) -> None:
    from huggingface_hub import snapshot_download

    job.target.mkdir(parents=True, exist_ok=True)
    job.marker_path.unlink(missing_ok=True)  # an interrupted run must not look complete
    snapshot_download(
        repo_id=job.repo_id,
        repo_type=job.repo_type,
        revision=job.revision,
        allow_patterns=list(job.allow_patterns) if job.allow_patterns is not None else None,
        local_dir=str(job.target),
        token=token,
        force_download=force,
    )
    job.marker_path.write_text(json.dumps(job.marker(), indent=2) + "\n", encoding="utf-8")


def folder_size_mb(path: Path) -> float:
    total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and ".cache" not in item.parts)
    return total / 1_048_576


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FusionTrace detector weights from the Hugging Face Hub.")
    parser.add_argument("--all", action="store_true", help="every registered detector, not only the enabled ones")
    parser.add_argument(
        "--kind",
        action="append",
        choices=[kind.value for kind in MediaKind],
        help="restrict to this media kind (repeatable)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan without downloading")
    parser.add_argument("--refresh", action="store_true", help="re-sync completed bundles with the Hub")
    parser.add_argument("--force", action="store_true", help="re-download every file, even if present")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    kinds = [MediaKind(value) for value in args.kind] if args.kind else list(MediaKind)

    root = registry.model_dir()
    logger.info("Model dir: %s", root)
    classes, errors = select_detectors(kinds, include_all=args.all)
    jobs, plan_errors = plan_jobs(classes, root)
    errors.extend(plan_errors)
    if classes:
        logger.info("Detectors: %s", ", ".join(sorted(cls.info.name for cls in classes)))
    if not jobs:
        logger.info("No weight bundles to download.")

    token = os.getenv("HF_TOKEN") or None
    for job in jobs:
        if job.is_complete() and not (args.refresh or args.force):
            logger.info("present   %s", job.describe())
            continue
        if args.dry_run:
            logger.info("would get %s", job.describe())
            continue
        logger.info("fetching  %s", job.describe())
        try:
            fetch(job, token=token, force=args.force)
        except Exception as exc:  # report every failure, then exit non-zero
            errors.append(f"{job.describe()}: {type(exc).__name__}: {exc}")
            continue
        logger.info("done      %s (%.1f MB on disk)", job.local_subdir, folder_size_mb(job.target))

    for message in errors:
        logger.error("ERROR: %s", message)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
