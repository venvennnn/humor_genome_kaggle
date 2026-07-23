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
