"""Split a full set/special transcript into ordered comedic 'bits'.

Handles timestamped transcripts (SRT/VTT) — grouping cues into bits and keeping
each bit's start time — and plain text (splitting on blank lines or sentence
groups). Timestamps, when available, let the callback graph and trajectory be
placed on a real timeline.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .video import parse_transcript, Cue


def _group_sentences(text: str, per_bit: int = 2) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    bits = []
    for i in range(0, len(sentences), per_bit):
        chunk = " ".join(sentences[i : i + per_bit])
        if chunk:
            bits.append(chunk)
    return bits


def _group_cues(cues: List[Cue], gap_s: float = 2.5, max_words: int = 45) -> List[Tuple[str, float]]:
    bits: List[Tuple[str, float]] = []
    cur: List[str] = []
    cur_start = 0.0
    prev_end = None
    word_count = 0
    for c in cues:
        starts_new = (
            prev_end is not None and (c.start_s - prev_end) > gap_s
        ) or word_count >= max_words
        if cur and starts_new:
            bits.append((" ".join(cur).strip(), cur_start))
            cur, word_count = [], 0
        if not cur:
            cur_start = c.start_s
        cur.append(c.text)
        word_count += len(c.text.split())
        prev_end = c.end_s
    if cur:
        bits.append((" ".join(cur).strip(), cur_start))
    return bits


def split_into_bits(
    transcript: str, max_bits: int = 40
) -> List[Tuple[str, float]]:
    """Return an ordered list of (bit_text, start_time_s).

    ``start_time_s`` is 0.0 when the transcript has no timestamps.
    """
    plain, cues = parse_transcript(transcript)
    if cues:
        bits = _group_cues(cues)
    else:
        # try paragraph splits first, else sentence groups
        paras = [p.strip() for p in re.split(r"\n\s*\n", transcript) if p.strip()]
        if len(paras) >= 2:
            bits = [(p, 0.0) for p in paras]
        else:
            bits = [(b, 0.0) for b in _group_sentences(plain)]

    bits = [b for b in bits if b[0].strip()]
    return bits[:max_bits]
