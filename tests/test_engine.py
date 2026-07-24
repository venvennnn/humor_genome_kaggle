"""Tests for the humor genome engine using the deterministic mock backend."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humor_genome.engine import HumorGenomeEngine, _extract_json, DEFAULT_AUDIENCES
from humor_genome.gemma_client import GemmaClient, GemmaConfig
from humor_genome.genome import GENOME_AXES


def _engine():
    return HumorGenomeEngine(config=GemmaConfig(backend="mock"))


def test_backend_is_mock():
    assert _engine().backend == "mock"


def test_analyze_returns_all_axes():
    report = _engine().analyze("Why did the scarecrow win an award? He was outstanding in his field.")
    names = [d.name for d in report.dimensions]
    assert names == GENOME_AXES
    for d in report.dimensions:
        assert 0 <= d.score <= 10


def test_analyze_covers_requested_audiences():
    auds = ["Close friends", "Tech crowd"]
    report = _engine().analyze("test joke", audiences=auds)
    assert [a.audience for a in report.audiences] == auds
    for a in report.audiences:
        assert a.verdict in {"kills", "lands", "polite chuckle", "bombs"}


def test_analyze_is_deterministic():
    joke = "consistent joke text"
    r1 = _engine().analyze(joke)
    r2 = _engine().analyze(joke)
    assert r1.funniness == r2.funniness
    assert [d.score for d in r1.dimensions] == [d.score for d in r2.dimensions]


def test_empty_joke_raises():
    try:
        _engine().analyze("   ")
    except ValueError:
        return
    assert False, "expected ValueError for empty joke"


def test_punch_up_produces_rewrite():
    engine = _engine()
    report = engine.analyze("a mediocre joke about pizza")
    punch = engine.punch_up(report, "Tech crowd")
    assert punch.audience == "Tech crowd"
    assert punch.rewrite.strip() != ""


def test_extract_json_handles_code_fence():
    text = 'here you go:\n```json\n{"a": 1, "b": [2,3]}\n```\nthanks'
    assert _extract_json(text) == {"a": 1, "b": [2, 3]}


def test_extract_json_handles_trailing_comma():
    assert _extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_extract_json_bare_object():
    assert _extract_json('prose {"x": 5} more prose') == {"x": 5}


def test_report_roundtrips_to_dict():
    report = _engine().analyze("roundtrip joke")
    d = report.to_dict()
    assert json.dumps(d)  # serializable
    assert d["joke"] == "roundtrip joke"
    assert len(d["dimensions"]) == len(GENOME_AXES)
    assert "punchlines" in d and "improvements" in d


def test_punchlines_detected():
    report = _engine().analyze(
        "Why did the scarecrow win an award? Because he was outstanding in his field."
    )
    assert len(report.punchlines) >= 1
    for p in report.punchlines:
        assert p.text.strip() != ""
        assert 0 <= p.strength <= 10


def test_tag_topper_detected():
    report = _engine().analyze(
        "I bought a treadmill to get fit. Now it's a coat rack — the most expensive one I own."
    )
    kinds = [p.kind for p in report.punchlines]
    assert "punchline" in kinds
    # the em-dash trailing clause should be picked up as a second laugh line
    assert len(report.punchlines) >= 2


def test_weak_joke_gets_improvements_and_flag():
    report = _engine().analyze("I like pizza. Pizza is good. Do you like pizza too?")
    assert report.needs_work is True
    assert len(report.improvements) >= 2
    for s in report.improvements:
        assert s.suggestion.strip() != ""


def test_improvements_target_weakest_axes():
    report = _engine().analyze("a fairly generic joke about mondays being bad")
    weakest = sorted(report.dimensions, key=lambda d: d.score)[:2]
    weak_names = {d.name for d in weakest}
    # each suggestion issue references one of the two weakest axes
    mentioned = " ".join(s.issue for s in report.improvements)
    assert any(name in mentioned for name in weak_names)


def test_scores_clamped_from_bad_model_output():
    # simulate a backend that returns out-of-range values
    class BadClient(GemmaClient):
        def __init__(self):
            self.config = GemmaConfig(backend="mock")
            self.active_backend = "mock"

        def generate(self, prompt, system=None):
            return json.dumps({
                "dimensions": [{"name": "surprise", "score": 99}],
                "funniness": -5,
                "audiences": [],
            })

    engine = HumorGenomeEngine(client=BadClient())
    report = engine.analyze("x")
    assert report.dimension("surprise") == 10.0
    assert report.funniness == 0.0
