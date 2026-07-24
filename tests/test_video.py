"""Tests for the video / audience-reaction analysis path (mock backend)."""

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
from humor_genome.laughter import (
    detect_reactions,
    read_wav_mono,
    laugh_coverage,
    biggest_laugh,
)
from humor_genome.video import parse_transcript, ffmpeg_available


def _synth_wav(path, dur=10, rate=16000, bursts=((3.5, 4.6), (8.0, 9.2)), seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, dur, dur * rate, endpoint=False)
    sig = 0.05 * np.sin(2 * np.pi * 180 * t)
    for a, b in bursts:
        m = (t >= a) & (t < b)
        sig[m] += 0.6 * rng.standard_normal(int(m.sum()))
    sig = np.clip(sig, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((sig * 32767).astype(np.int16).tobytes())


# ------------------------------------------------------------- reaction detect
def test_detect_reactions_finds_bursts():
    tmp = tempfile.mkdtemp()
    wav = os.path.join(tmp, "a.wav")
    _synth_wav(wav, bursts=((3.5, 4.6), (8.0, 9.2)))
    waveform, rate = read_wav_mono(wav)
    reactions = detect_reactions(waveform, rate)
    assert len(reactions) == 2
    # first burst roughly at 3.5-4.6s
    assert 3.0 <= reactions[0].start_s <= 4.0
    assert reactions[0].end_s <= 5.2
    assert all(0.0 <= r.intensity <= 1.0 for r in reactions)


def test_no_reactions_in_quiet_audio():
    tmp = tempfile.mkdtemp()
    wav = os.path.join(tmp, "quiet.wav")
    _synth_wav(wav, bursts=())
    waveform, rate = read_wav_mono(wav)
    reactions = detect_reactions(waveform, rate)
    assert reactions == []


def test_coverage_and_biggest():
    tmp = tempfile.mkdtemp()
    wav = os.path.join(tmp, "a.wav")
    _synth_wav(wav, bursts=((3.5, 4.6),))
    waveform, rate = read_wav_mono(wav)
    reactions = detect_reactions(waveform, rate)
    cov = laugh_coverage(reactions, 10.0)
    assert 0.0 < cov < 1.0
    assert 3.0 <= biggest_laugh(reactions) <= 5.0


# --------------------------------------------------------------- transcript
def test_parse_srt():
    srt = "1\n00:00:01,000 --> 00:00:04,000\nHello there.\n\n2\n00:00:05,000 --> 00:00:08,000\nGeneral Kenobi.\n"
    plain, cues = parse_transcript(srt)
    assert len(cues) == 2
    assert cues[0].start_s == 1.0 and cues[0].end_s == 4.0
    assert "Hello there" in plain and "General Kenobi" in plain


def test_parse_plain_transcript():
    plain, cues = parse_transcript("just a plain line\nand another")
    assert cues == []
    assert "plain line" in plain


def test_parse_empty_transcript():
    assert parse_transcript("") == ("", [])


# --------------------------------------------------------------- end to end
@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_analyze_video_end_to_end():
    tmp = tempfile.mkdtemp()
    wav = os.path.join(tmp, "a.wav")
    _synth_wav(wav, dur=10, bursts=((3.5, 4.6), (8.0, 9.2)))
    mp4 = os.path.join(tmp, "clip.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         "testsrc=duration=10:size=320x240:rate=10", "-i", wav,
         "-shortest", "-pix_fmt", "yuv420p", mp4],
        capture_output=True, check=True,
    )
    engine = HumorGenomeEngine(config=GemmaConfig(backend="mock"))
    report = engine.analyze_video(
        mp4, transcript="I assembled furniture. Now it's modern art."
    )
    assert report.duration_s == pytest.approx(10, abs=1)
    assert len(report.reactions) == 2
    assert len(report.beats) >= 1
    assert any(b.landed for b in report.beats)
    # mock backend is not multimodal, so no frames should be sent
    assert report.multimodal_used is False
    assert report.to_dict()["source"].endswith(".mp4")


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_analyze_video_falls_back_when_vision_fails():
    """If the vision (image) call fails, analysis should degrade to text-only."""
    import json as _json
    from humor_genome.gemma_client import GemmaClient
    from humor_genome.mock_brain import mock_response

    class VisionFailsClient(GemmaClient):
        def __init__(self):
            self.config = GemmaConfig(backend="mock")
            self.active_backend = "ollama"  # pretend we're on a real backend

        @property
        def supports_images(self):
            return True

        def generate(self, prompt, system=None, images=None):
            if images:
                raise RuntimeError("400: model does not support images")
            return mock_response(prompt)

    tmp = tempfile.mkdtemp()
    wav = os.path.join(tmp, "a.wav")
    _synth_wav(wav, dur=8, bursts=((3.0, 4.0),))
    mp4 = os.path.join(tmp, "clip.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         "testsrc=duration=8:size=320x240:rate=10", "-i", wav,
         "-shortest", "-pix_fmt", "yuv420p", mp4],
        capture_output=True, check=True,
    )
    engine = HumorGenomeEngine(client=VisionFailsClient())
    report = engine.analyze_video(mp4, transcript="A joke. The punchline.")
    # it should NOT raise; instead fall back to text-only
    assert report.multimodal_used is False
    assert report.frames_analyzed == 0
    assert any("vision" in n.lower() for n in report.notes)
    assert len(report.beats) >= 1
