"""FFmpeg helpers for temporal video detectors: probing, clip planning, clip extraction.

A temporal detector looks at *clips*: K consecutive frames (native frame rate)
starting at N evenly spaced positions across the video. Everything here shells
out to ``ffprobe``/``ffmpeg`` with bounded timeouts and never sends media
anywhere; frames land in the per-scan scratch directory.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..base import ModelUnavailableError, UnsupportedInputError

DEFAULT_FPS = 25.0
MAX_FRAME_SIDE = 1280  # extracted frames are downscaled to this longest side (bounds CPU cost)


@dataclass(frozen=True)
class VideoProbe:
    duration: float  # seconds
    fps: float  # average frame rate; DEFAULT_FPS when the container does not say
    width: int
    height: int
    frame_count: int | None  # from container metadata, may be missing


@dataclass(frozen=True)
class ClipPlan:
    index: int
    start_seconds: float
    frames: int


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise ModelUnavailableError("ffmpeg/ffprobe not found on PATH", public_message="FFmpeg is not installed in the runtime.")


def _parse_rate(raw: str | None) -> float | None:
    if not raw or raw in {"0/0", "N/A"}:
        return None
    try:
        if "/" in raw:
            num, den = raw.split("/", 1)
            value = float(num) / float(den) if float(den) else 0.0
        else:
            value = float(raw)
    except ValueError:
        return None
    return value if 1.0 <= value <= 240.0 else None


def _parse_float(raw: object) -> float | None:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def probe(path: Path, timeout: float = 30.0) -> VideoProbe:
    """Read duration, frame rate and size of the first video stream."""
    require_ffmpeg()
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
        "-of", "json", str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=max(1.0, timeout), check=False)
    except subprocess.TimeoutExpired as exc:
        raise UnsupportedInputError(f"ffprobe timed out on {path.name}") from exc
    if completed.returncode != 0:
        raise UnsupportedInputError(f"ffprobe failed on {path.name}: {completed.stderr[-300:]!r}")
    try:
        data = json.loads(completed.stdout or b"{}")
    except json.JSONDecodeError as exc:
        raise UnsupportedInputError(f"ffprobe returned invalid JSON for {path.name}") from exc
    streams = data.get("streams") or []
    if not streams:
        raise UnsupportedInputError(f"{path.name} has no video stream", public_message="This file has no video stream to analyse.")
    stream = streams[0]
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate")) or DEFAULT_FPS
    duration = _parse_float(stream.get("duration")) or _parse_float((data.get("format") or {}).get("duration"))
    frame_count = None
    try:
        frame_count = int(stream["nb_frames"]) if stream.get("nb_frames") not in (None, "N/A") else None
    except (TypeError, ValueError):
        frame_count = None
    if duration is None and frame_count:
        duration = frame_count / fps
    if duration is None:
        raise UnsupportedInputError(f"could not determine the duration of {path.name}")
    return VideoProbe(duration=duration, fps=fps, width=int(stream.get("width") or 0), height=int(stream.get("height") or 0), frame_count=frame_count)


def plan_clips(info: VideoProbe, max_clips: int, frames_per_clip: int) -> list[ClipPlan]:
    """N clips of K consecutive frames, evenly spaced over the video.

    Clip ``i`` starts at ``(i + 0.5) / N`` of the usable range, so clips avoid
    the very first and last frames (fades, encoder warm-up). A video shorter
    than one clip yields a single clip from t=0.
    """
    max_clips = max(1, int(max_clips))
    frames_per_clip = max(3, int(frames_per_clip))
    span = frames_per_clip / info.fps
    usable = info.duration - span
    if usable <= 0:
        return [ClipPlan(index=0, start_seconds=0.0, frames=frames_per_clip)]
    # Do not plan more clips than fit without overlapping.
    count = min(max_clips, max(1, int(info.duration // span)))
    return [ClipPlan(index=i, start_seconds=round(usable * (i + 0.5) / count, 3), frames=frames_per_clip) for i in range(count)]


def extract_clip(path: Path, clip: ClipPlan, out_dir: Path, timeout: float) -> list[Path]:
    """Decode ``clip.frames`` consecutive frames starting at ``clip.start_seconds`` to JPEGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = f"scale='if(gt(iw,ih),min(iw,{MAX_FRAME_SIDE}),-2)':'if(gt(iw,ih),-2,min(ih,{MAX_FRAME_SIDE}))'"
    base = ["ffmpeg", "-y", "-v", "error", "-ss", f"{clip.start_seconds:.3f}", "-i", str(path), "-map", "0:v:0", "-frames:v", str(clip.frames), "-vf", scale, "-q:v", "2"]
    pattern = str(out_dir / "f-%03d.jpg")
    # ``-fps_mode passthrough`` keeps every decoded frame (no duplication for VFR input);
    # older FFmpeg builds lack the flag, so fall back to the default muxer behaviour.
    for extra in (["-fps_mode", "passthrough"], []):
        try:
            completed = subprocess.run([*base, *extra, pattern], capture_output=True, timeout=max(1.0, timeout), check=False)
        except subprocess.TimeoutExpired as exc:
            raise UnsupportedInputError(f"frame extraction timed out at {clip.start_seconds:.2f}s") from exc
        if completed.returncode == 0:
            break
    else:
        raise UnsupportedInputError(f"frame extraction failed at {clip.start_seconds:.2f}s: {completed.stderr[-300:]!r}")
    return sorted(out_dir.glob("f-*.jpg"))
