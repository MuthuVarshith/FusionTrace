"""Upload validation by metadata inspection, never by a full decode.

* Images: Pillow reads the header, the pixel count is checked against
  ``FUSIONTRACE_MAX_IMAGE_PIXELS`` (decompression-bomb guard) and then
  ``Image.verify()`` checks the file structure.
* Audio and video: ``ffprobe`` reads container and stream metadata. The
  container must match the file extension, the expected stream must exist
  (audio for audio files, a real video stream for video files), audio files
  must not carry a video track, and the duration must be known and within
  ``FUSIONTRACE_MAX_AUDIO_SECONDS`` / ``FUSIONTRACE_MAX_VIDEO_SECONDS``.

Every :class:`UploadRejected` message is client-safe. Tool output, paths and
library errors go to the server log only.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config as settings

logger = logging.getLogger("fusiontrace")

ALLOWED_EXTENSIONS: dict[str, set[str]] = {
    "image": {".jpg", ".jpeg", ".png", ".webp"},
    "audio": {".wav", ".mp3", ".m4a"},
    "video": {".mp4", ".mov", ".webm"},
}

_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
_MOV_FAMILY = {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}
# ffprobe ``format_name`` values (comma-separated families) accepted per extension.
_CONTAINERS: dict[str, set[str]] = {
    ".wav": {"wav"},
    ".mp3": {"mp3"},
    ".m4a": _MOV_FAMILY,
    ".mp4": _MOV_FAMILY,
    ".mov": _MOV_FAMILY,
    ".webm": {"matroska", "webm"},
}
PROBE_TIMEOUT_SECONDS = 20


class UploadRejected(ValueError):
    """The upload failed validation. ``str(exc)`` is safe to show to a client."""

    status_code = 400


class FileTooLarge(UploadRejected):
    status_code = 413


class ValidatorUnavailable(UploadRejected):
    """A required validation tool (ffprobe) is missing on the server."""

    status_code = 503


@dataclass(frozen=True)
class MediaProbe:
    kind: str
    format_name: str | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool = False
    has_video: bool = False


def media_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    for kind, suffixes in ALLOWED_EXTENSIONS.items():
        if suffix in suffixes:
            return kind
    raise UploadRejected("Unsupported file type. Use an image (JPG/PNG/WebP), audio (WAV/MP3/M4A), or video (MP4/MOV/WebM).")


def max_bytes(kind: str) -> int:
    megabytes = settings.MAX_IMAGE_MB if kind == "image" else settings.MAX_UPLOAD_MB
    return megabytes * 1024 * 1024


def validate_upload(path: Path, kind: str) -> MediaProbe:
    """Check that ``path`` really is media of ``kind`` and within the configured limits."""
    size = path.stat().st_size
    if size == 0:
        raise UploadRejected("The uploaded file is empty.")
    if size > max_bytes(kind):
        raise FileTooLarge(f"File exceeds the {max_bytes(kind) // (1024 * 1024)} MB limit for {kind} uploads.")
    if kind == "image":
        return _validate_image(path)
    if kind in ("audio", "video"):
        return _validate_av(path, kind)
    raise UploadRejected("Unsupported media type.")


# ── images ──────────────────────────────────────────────────────────────────


def _validate_image(path: Path) -> MediaProbe:
    from PIL import Image

    invalid = "The file is not a valid JPG, PNG or WebP image."
    try:
        with Image.open(path) as image:
            image_format, (width, height) = image.format, image.size
            if image_format not in _IMAGE_FORMATS:
                raise UploadRejected(invalid)
            if width <= 0 or height <= 0:
                raise UploadRejected(invalid)
            if width * height > settings.MAX_IMAGE_PIXELS:
                raise UploadRejected(_too_many_pixels())
            image.verify()
    except UploadRejected:
        raise
    except Image.DecompressionBombError as exc:
        logger.info("Image %s rejected as a decompression bomb: %s", path.name, exc)
        raise UploadRejected(_too_many_pixels()) from exc
    except Exception as exc:  # Pillow raises a wide range of types for corrupt input
        logger.info("Image validation failed for %s: %s", path.name, exc)
        raise UploadRejected(invalid) from exc
    return MediaProbe(kind="image", format_name=image_format.lower(), width=width, height=height, has_video=False)


def _too_many_pixels() -> str:
    return f"Image dimensions exceed the {settings.MAX_IMAGE_PIXELS / 1_000_000:g}-megapixel limit."


# ── audio / video ───────────────────────────────────────────────────────────


def _ffprobe(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        logger.error("ffprobe is not installed; cannot validate %s", path.name)
        raise ValidatorUnavailable("Audio and video uploads cannot be validated on this server right now.")
    command = [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=PROBE_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as exc:
        logger.warning("ffprobe timed out on %s", path.name)
        raise UploadRejected("The file could not be inspected in time.") from exc
    except OSError as exc:
        logger.error("ffprobe could not run on %s: %s", path.name, exc)
        raise ValidatorUnavailable("Audio and video uploads cannot be validated on this server right now.") from exc
    if completed.returncode != 0:
        logger.info("ffprobe rejected %s: %s", path.name, completed.stderr.decode("utf-8", "replace")[:2000])
        raise UploadRejected("The file is not readable audio or video.")
    try:
        return json.loads(completed.stdout or b"{}")
    except ValueError as exc:
        logger.warning("ffprobe returned invalid JSON for %s", path.name)
        raise UploadRejected("The file is not readable audio or video.") from exc


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _validate_av(path: Path, kind: str) -> MediaProbe:
    info = _ffprobe(path)
    fmt = info.get("format") or {}
    streams = info.get("streams") or []
    suffix = path.suffix.lower()

    format_names = {name.strip() for name in str(fmt.get("format_name") or "").split(",") if name.strip()}
    if not format_names & _CONTAINERS.get(suffix, set()):
        logger.info("Container %s does not match extension %s for %s", sorted(format_names), suffix, path.name)
        raise UploadRejected(f"The file contents do not match the {suffix} file type.")

    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    video_streams = [
        s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
    ]
    if kind == "audio":
        if not audio_streams:
            raise UploadRejected("The file contains no audio track.")
        if video_streams:
            raise UploadRejected("This file contains video; upload it as MP4, MOV or WebM instead.")
        limit = settings.MAX_AUDIO_SECONDS
    else:
        if not video_streams:
            raise UploadRejected("The file contains no video track.")
        limit = settings.MAX_VIDEO_SECONDS

    durations = [_as_float(fmt.get("duration"))] + [_as_float(s.get("duration")) for s in audio_streams + video_streams]
    known = [d for d in durations if d is not None]
    if not known:
        raise UploadRejected(f"The {kind} duration could not be determined from the file.")
    duration = max(known)
    if duration > limit:
        raise UploadRejected(f"The {kind} is {duration:.0f} seconds long; the limit is {limit:g} seconds.")

    width = height = None
    if video_streams:
        width, height = video_streams[0].get("width"), video_streams[0].get("height")
        if kind == "video" and (not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0):
            raise UploadRejected("The video track has no valid frame size.")
    return MediaProbe(
        kind=kind,
        format_name=str(fmt.get("format_name") or "") or None,
        duration_seconds=duration,
        width=width,
        height=height,
        has_audio=bool(audio_streams),
        has_video=bool(video_streams),
    )
