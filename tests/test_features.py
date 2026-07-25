"""Tests for the new features: clustering, set splitting, predicted-vs-actual,
set analysis (style clusters + callbacks)."""

import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humor_genome.engine import HumorGenomeEngine
from humor_genome.gemma_client import GemmaConfig
from humor_genome import clustering, setsplit
from humor_genome.genome import PredictedLaugh, ReactionMoment
from humor_genome.laughter import reaction_envelope
from humor_genome.video import ffmpeg_available


def _engine():
    return HumorGenomeEngine(config=GemmaConfig(backend="mock"))


# --------------------------------------------------------------- clustering
def test_kmeans_separates_two_blobs():
    a = np.random.default_rng(0).normal(0, 0.2, (10, 6)) + 1
    b = np.random.default_rng(1).normal(0, 0.2, (10, 6)) + 8
    x = np.vstack([a, b])
    labels, centroids = clustering.kmeans(x, 2, seed=0)
    assert len(labels) == 20
    assert len(set(labels.tolist())) == 2
    # points within a blob should mostly share a label
    assert len(set(labels[:10].tolist())) == 1 or len(set(labels[10:].tolist())) == 1


def test_pca_2d_shape():
    x = np.random.default_rng(0).normal(size=(12, 6))
    coords = clustering.pca_2d(x)
    assert coords.shape == (12, 2)


def test_pca_handles_single_row():
    assert clustering.pca_2d(np.zeros((1, 6))).shape == (1, 2)


def test_suggest_k_monotonic():
    assert clustering.suggest_k(2) <= clustering.suggest_k(30)


# --------------------------------------------------------------- set splitting
def test_split_plain_paragraphs():
    txt = "First bit about cats.\n\nSecond bit about dogs.\n\nThird bit about birds."
    bits = setsplit.split_into_bits(txt)
    assert len(bits) == 3
    assert all(t == 0.0 for _, t in bits)


def test_split_srt_keeps_times():
    srt = ("1\n00:00:02,000 --> 00:00:05,000\nSetup line one.\n\n"
           "2\n00:00:20,000 --> 00:00:24,000\nA much later bit.\n")
    bits = setsplit.split_into_bits(srt)
    assert len(bits) >= 2
    assert bits[0][1] == pytest.approx(2.0, abs=0.01)
    assert bits[-1][1] >= 20.0


def test_split_respects_max_bits():
    txt = "\n\n".join(f"Bit number {i}." for i in range(60))
    assert len(setsplit.split_into_bits(txt, max_bits=10)) == 10


# --------------------------------------------------------------- envelope
def test_reaction_envelope():
    rate = 16000
    wf = np.random.default_rng(0).normal(0, 0.1, rate * 4)
    t, v = reaction_envelope(wf, rate)
    assert len(t) == len(v) > 0
    assert all(0.0 <= x <= 1.0 for x in v)


# --------------------------------------------------- predicted vs actual (unit)
def test_compare_laughs_matches_and_gaps():
    predicted = [
        PredictedLaugh(time_s=3.0, quote="a", expected_intensity=0.6),
        PredictedLaugh(time_s=20.0, quote="b", expected_intensity=0.7),
    ]
    reactions = [
        ReactionMoment(start_s=3.5, end_s=4.2, intensity=0.9),   # matches pred 1
        ReactionMoment(start_s=40.0, end_s=41.0, intensity=0.8),  # surprise
    ]
    gaps, hit_rate = HumorGenomeEngine._compare_laughs(predicted, reactions)
    kinds = sorted(g.kind for g in gaps)
    assert "matched" in kinds
    assert "bombed" in kinds      # pred at 20s had no laugh
    assert "surprise" in kinds    # laugh at 40s not predicted
    assert hit_rate == pytest.approx(0.5)


# --------------------------------------------------------------- set analysis
def test_analyze_set_end_to_end():
    transcript = (
        "I bought a treadmill to get fit. It is now a coat rack.\n\n"
        "My dog watched me fail at assembly. He respects me less.\n\n"
        "I tried meditation. My brain hates silence.\n\n"
        "Back to that treadmill — I used it to dry a poncho.\n\n"
        "The meditation app now guilt-trips me about my streak.\n\n"
        "My dog meditates all day. We used to call it sleeping."
    )
    report = _engine().analyze_set(transcript)
    assert len(report.jokes) == 6
    assert all(j.axes for j in report.jokes)
    assert len(report.clusters) >= 1
    assert len(report.pca_coords) == 6
    # every joke assigned to a cluster
    assert all(j.cluster >= 0 for j in report.jokes)
    # treadmill callback should be found (bit 0 -> bit 3)
    assert any(cb.setup_index < cb.callback_index for cb in report.callbacks)
    d = report.to_dict()
    import json
    assert json.dumps(d)


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_analyze_video_populates_prediction():
    rate, dur = 16000, 12
    t = np.linspace(0, dur, dur * rate, endpoint=False)
    sig = 0.05 * np.sin(2 * np.pi * 180 * t)
    for a, b in [(4.0, 5.0), (9.8, 10.8)]:
        m = (t >= a) & (t < b)
        sig[m] += 0.6 * np.random.default_rng(0).standard_normal(int(m.sum()))
    sig = np.clip(sig, -1, 1)
    tmp = tempfile.mkdtemp()
    wav = os.path.join(tmp, "a.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((sig * 32767).astype(np.int16).tobytes())
    mp4 = os.path.join(tmp, "clip.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"testsrc=duration={dur}:size=320x240:rate=10", "-i", wav,
         "-shortest", "-pix_fmt", "yuv420p", mp4],
        capture_output=True, check=True,
    )
    srt = ("1\n00:00:02,000 --> 00:00:05,000\nI bought a treadmill.\n\n"
           "2\n00:00:09,000 --> 00:00:11,000\nMy dog respects me less.\n")
    report = _engine().analyze_video(mp4, transcript=srt)
    assert len(report.envelope_t) > 0
    assert len(report.predicted_laughs) >= 1
    assert len(report.laugh_gaps) >= 1
    assert 0.0 <= report.prediction_hit_rate <= 1.0
