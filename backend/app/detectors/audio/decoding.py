"""Audio decoding for the audio detectors.

WAV/FLAC/OGG are read with soundfile; anything else (MP3, M4A, video
soundtracks...) is converted by an ffmpeg subprocess to a 16 kHz mono PCM WAV
in the per-scan work directory. Samples are read lazily, one window at a time,
so a long upload never has to fit in memory as float32 in one piece. Resampling
uses torchaudio. Nothing is written outside the work directory.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

from ..base import UnsupportedInputError

logger = logging.getLogger("fusiontrace")

SOUNDFILE_SUFFIXES = frozenset({".wav", ".wave", ".flac", ".ogg"})
FFMPEG_TIMEOUT_SECONDS = 300.0
_RESAMPLE_MARGIN = 64  # native samples read on each side of a window so resampling has context


class AudioSource:
    """A mono view of an open audio file at ``target_rate``, read by sample span."""

    def __init__(self, handle: sf.SoundFile, target_rate: int, decoder: str) -> None:
        self._handle = handle
        self.native_rate = int(handle.samplerate)
        self.channels = int(handle.channels)
        self.target_rate = int(target_rate)
        self.decoder = decoder
        native_frames = max(0, int(handle.frames))
        self.num_samples = native_frames if self.native_rate == self.target_rate else int(native_frames * self.target_rate // self.native_rate)
        self._native_frames = native_frames
        self._resampler = None
        if self.native_rate != self.target_rate:
            import torchaudio

            self._resampler = torchaudio.transforms.Resample(self.native_rate, self.target_rate)

    @property
    def duration_seconds(self) -> float:
        return self.num_samples / self.target_rate

    def _read_native(self, start: int, stop: int) -> np.ndarray:
        start, stop = max(0, start), min(self._native_frames, stop)
        if stop <= start:
            return np.zeros(0, dtype=np.float32)
        self._handle.seek(start)
        data = self._handle.read(stop - start, dtype="float32", always_2d=True)
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        return np.ascontiguousarray(mono, dtype=np.float32)

    def read(self, start: int, end: int) -> np.ndarray:
        """Samples ``[start, end)`` at ``target_rate``, mono float32 in [-1, 1]."""
        length = max(0, min(end, self.num_samples) - start)
        if length == 0:
            return np.zeros(0, dtype=np.float32)
        if self._resampler is None:
            return self._read_native(start, start + length)

        import torch

        ratio = self.native_rate / self.target_rate
        native_start = max(0, math.floor(start * ratio) - _RESAMPLE_MARGIN)
        native_stop = math.ceil((start + length) * ratio) + _RESAMPLE_MARGIN
        native = self._read_native(native_start, native_stop)
        with torch.inference_mode():
            resampled = self._resampler(torch.from_numpy(native)).numpy()
        offset = round(start - native_start / ratio)
        out = resampled[offset : offset + length]
        if out.size < length:
            out = np.pad(out, (0, length - out.size))
        return np.ascontiguousarray(out, dtype=np.float32)


def _ffmpeg_to_wav(path: Path, work_dir: Path, target_rate: int, timeout: float) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise UnsupportedInputError(
            "ffmpeg is not installed; cannot decode non-WAV audio",
            public_message="This audio format cannot be decoded on this server.",
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"audio-{uuid.uuid4().hex}.wav"
    command = [
        ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(path),
        "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(target_rate), "-c:a", "pcm_s16le", str(out),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=max(1.0, timeout), check=False)
    except subprocess.TimeoutExpired as exc:
        out.unlink(missing_ok=True)
        raise UnsupportedInputError(f"ffmpeg timed out decoding {path.name}", public_message="The audio could not be decoded in time.") from exc
    except OSError as exc:
        raise UnsupportedInputError(f"ffmpeg failed to start: {exc}", public_message="This audio format cannot be decoded on this server.") from exc
    if completed.returncode != 0 or not out.exists() or out.stat().st_size <= 44:
        out.unlink(missing_ok=True)
        stderr = completed.stderr.decode("utf-8", "replace")[-400:]
        raise UnsupportedInputError(f"ffmpeg could not decode {path.name}: {stderr}", public_message="The file contains no decodable audio.")
    return out


@contextmanager
def open_audio(path: Path, work_dir: Path, target_rate: int = 16000, timeout: float | None = None) -> Iterator[AudioSource]:
    """Open ``path`` for windowed reading at ``target_rate`` Hz mono.

    Raises :class:`UnsupportedInputError` when the file cannot be decoded. A
    converted WAV (non-WAV input) lives in ``work_dir`` and is removed on exit.
    """
    path = Path(path)
    converted: Path | None = None
    handle: sf.SoundFile | None = None
    decoder = "soundfile"
    try:
        if path.suffix.lower() in SOUNDFILE_SUFFIXES:
            try:
                handle = sf.SoundFile(str(path))
            except (sf.LibsndfileError, RuntimeError, OSError) as exc:
                logger.info("soundfile could not open %s (%s); falling back to ffmpeg", path.name, exc)
        if handle is None:
            converted = _ffmpeg_to_wav(path, Path(work_dir), target_rate, FFMPEG_TIMEOUT_SECONDS if timeout is None else min(timeout, FFMPEG_TIMEOUT_SECONDS))
            decoder = "ffmpeg"
            try:
                handle = sf.SoundFile(str(converted))
            except (sf.LibsndfileError, RuntimeError, OSError) as exc:
                raise UnsupportedInputError(f"converted audio unreadable: {exc}", public_message="The file contains no decodable audio.") from exc
        if handle.samplerate <= 0:
            raise UnsupportedInputError("audio reports a non-positive sample rate", public_message="The file contains no decodable audio.")
        yield AudioSource(handle, target_rate, decoder)
    finally:
        if handle is not None:
            handle.close()
        if converted is not None:
            converted.unlink(missing_ok=True)
