"""Audience-reaction detection from an audio track.

This is a lightweight, dependency-light detector for *where the crowd reacts*
(laughter / applause) in a clip. It does NOT try to be a trained laughter
classifier — instead it finds sustained, broadband energy bursts that sit above
the local speech baseline, which in practice tracks laughter and applause well
enough to align jokes to reactions in a prototype.

Signal used:
- Short-time RMS energy (loudness).
- Short-time zero-crossing rate (broadband/noisy vs. tonal speech).
Laughter/applause tends to be both loud AND broadband/sustained, which lets us
separate it from a single loud spoken word.

Everything here works on a mono waveform as a numpy array, so it is unit-testable
with synthetic signals and needs no model.
"""

from __future__ import annotations

import wave
from typing import List, Tuple

import numpy as np

from .genome import ReactionMoment


def read_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    """Read a WAV file into a float32 mono waveform in [-1, 1] and its rate."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
    data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    max_val = float(np.iinfo(dtype).max) or 1.0
    return data / max_val, rate


def _frame_signal(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def detect_reactions(
    waveform: np.ndarray,
    rate: int,
    min_duration_s: float = 0.35,
    frame_s: float = 0.05,
) -> List[ReactionMoment]:
    """Detect laughter/applause bursts in a mono waveform.

    Returns reaction moments with normalized intensity in [0, 1].
    """
    if rate <= 0 or waveform.size == 0:
        return []

    frame = max(1, int(frame_s * rate))
    hop = max(1, frame // 2)
    frames = _frame_signal(waveform, frame, hop)

    # per-frame loudness and broadbandness
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-9)
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)

    # normalize
    rms_n = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)
    # a reaction frame: loud AND noisy relative to the track
    loud_thr = float(np.mean(rms_n) + 0.6 * np.std(rms_n))
    zcr_thr = float(np.median(zcr))
    active = (rms_n > loud_thr) & (zcr >= zcr_thr)

    times = np.arange(len(active)) * hop / rate
    min_frames = max(1, int(min_duration_s / (hop / rate)))

    reactions: List[ReactionMoment] = []
    i = 0
    peak_intensity = 0.0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            if (j - i) >= min_frames:
                seg = rms_n[i:j]
                intensity = float(np.mean(seg))
                peak_intensity = max(peak_intensity, intensity)
                reactions.append(
                    ReactionMoment(
                        start_s=round(float(times[i]), 2),
                        end_s=round(float(times[min(j, len(times) - 1)]), 2),
                        intensity=round(intensity, 3),
                    )
                )
            i = j
        else:
            i += 1

    # rescale intensities so the biggest reaction ~= 1.0
    if peak_intensity > 0:
        for r in reactions:
            r.intensity = round(min(1.0, r.intensity / peak_intensity), 3)
    return reactions


def laugh_coverage(reactions: List[ReactionMoment], duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    total = sum(r.duration for r in reactions)
    return round(min(1.0, total / duration_s), 3)


def biggest_laugh(reactions: List[ReactionMoment]) -> float:
    if not reactions:
        return 0.0
    top = max(reactions, key=lambda r: r.intensity)
    return round((top.start_s + top.end_s) / 2, 2)
