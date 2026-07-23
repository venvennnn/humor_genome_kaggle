"""Video ingestion helpers: frames, audio, transcripts.

Uses ffmpeg/ffprobe (via subprocess) to turn an uploaded clip into things a
Gemma model and our reaction detector can consume:

- a mono 16 kHz WAV for laughter/applause detection,
- a handful of evenly spaced JPEG frames for the multimodal (Gemma 3 / 3n) path,
- parsed transcript cues (SRT/VTT with timestamps, or plain text).

All ffmpeg use is optional and degrades gracefully: if ffmpeg is missing we
raise a clear error the UI/CLI can surface, rather than crashing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple


class FFmpegUnavailable(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _require_ffmpeg() -> None:
    if not ffmpeg_available():
        raise FFmpegUnavailable(
            "ffmpeg/ffprobe not found on PATH. Install ffmpeg to analyze videos."
        )


def probe_duration(path: str) -> float:
    _require_ffmpeg()
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", path,
            ],
            capture_output=True, text=True, timeout=60,
        )
        return float(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return 0.0


def extract_audio_wav(path: str, out_path: str, rate: int = 16000) -> str:
    _require_ffmpeg()
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", path,
            "-ac", "1", "-ar", str(rate), "-vn", out_path,
        ],
        capture_output=True, timeout=300, check=True,
    )
    return out_path


def extract_frames(path: str, out_dir: str, n: int = 6) -> List[str]:
    """Extract ``n`` evenly spaced JPEG frames. Returns sorted file paths."""
    _require_ffmpeg()
    os.makedirs(out_dir, exist_ok=True)
    duration = probe_duration(path) or 0.0
    paths: List[str] = []
    if duration <= 0:
        # fallback: grab frames by fps sampling
        pattern = os.path.join(out_dir, "frame_%03d.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-vf", "fps=1", "-frames:v", str(n), pattern],
            capture_output=True, timeout=300, check=False,
        )
    else:
        for i in range(n):
            ts = duration * (i + 0.5) / n
            fp = os.path.join(out_dir, f"frame_{i:03d}.jpg")
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", path,
                 "-frames:v", "1", "-q:v", "3", fp],
                capture_output=True, timeout=120, check=False,
            )
            if os.path.exists(fp):
                paths.append(fp)
    if not paths:
        paths = sorted(
            os.path.join(out_dir, f) for f in os.listdir(out_dir)
            if f.endswith(".jpg")
        )
    return sorted(paths)


# ------------------------------------------------------------ transcript utils

_TS_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})"
)


@dataclass
class Cue:
    start_s: float
    end_s: float
    text: str


def _hms_to_s(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_transcript(raw: str) -> Tuple[str, List[Cue]]:
    """Parse an SRT/VTT/plain transcript.

    Returns (plain_text, cues). ``cues`` is empty for a plain transcript.
    """
    if not raw:
        return "", []
    cues: List[Cue] = []
    lines = raw.splitlines()
    i = 0
    pending: List[str] = []
    cur: Optional[Tuple[float, float]] = None

    def flush():
        nonlocal pending, cur
        if cur is not None and pending:
            cues.append(Cue(cur[0], cur[1], " ".join(pending).strip()))
        pending = []
        cur = None

    for line in lines:
        m = _TS_RE.search(line)
        if m:
            flush()
            g = m.groups()
            cur = (_hms_to_s(*g[:4]), _hms_to_s(*g[4:]))
        elif line.strip().isdigit() or line.strip().upper() == "WEBVTT":
            continue
        elif line.strip() == "":
            flush()
        else:
            pending.append(line.strip())
    flush()

    if cues:
        plain = " ".join(c.text for c in cues)
    else:
        plain = " ".join(l.strip() for l in lines if l.strip())
    return plain.strip(), cues
