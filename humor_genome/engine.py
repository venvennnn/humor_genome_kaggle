"""The humor genome analysis engine.

Orchestrates: build prompt -> call Gemma -> parse JSON -> validate/normalize into
the dataclasses in ``genome.py``. Robust JSON extraction is important because
instruction-tuned models occasionally wrap JSON in prose or code fences.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from .gemma_client import GemmaClient, GemmaConfig
from .genome import (
    GENOME_AXES,
    GenomeReport,
    GenomeDimension,
    AudienceFit,
    PunchUp,
)
from .prompts import SYSTEM_PROMPT, analysis_prompt, punchup_prompt

DEFAULT_AUDIENCES = [
    "Close friends",
    "Tech crowd",
    "General public",
    "Corporate all-hands",
]


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response.

    Handles code fences and leading/trailing prose. Raises ValueError if no
    parseable object is found.
    """
    if not text:
        raise ValueError("empty model response")

    # strip common markdown code fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None

    if candidate is None:
        # find the outermost {...} span
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object found in model response")
        candidate = text[start : end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # last-ditch: remove trailing commas
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(cleaned)


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class HumorGenomeEngine:
    def __init__(
        self,
        client: Optional[GemmaClient] = None,
        config: Optional[GemmaConfig] = None,
    ):
        self.client = client or GemmaClient(config)

    @property
    def backend(self) -> str:
        return self.client.active_backend

    def describe_backend(self) -> str:
        return self.client.describe()

    # ---------------------------------------------------------------- analyze
    def analyze(
        self, joke: str, audiences: Optional[List[str]] = None
    ) -> GenomeReport:
        joke = (joke or "").strip()
        if not joke:
            raise ValueError("joke text is empty")
        audiences = audiences or DEFAULT_AUDIENCES

        prompt = analysis_prompt(joke, audiences)
        raw = self.client.generate(prompt, system=SYSTEM_PROMPT)

        try:
            data = _extract_json(raw)
        except ValueError:
            # If a real backend returns unparseable text, degrade gracefully to
            # a minimal report rather than crashing the UI.
            return GenomeReport(
                joke=joke,
                one_line_explanation="(could not parse model output)",
                raw_model_output=raw,
            )

        return self._build_report(joke, data, audiences, raw)

    def _build_report(
        self, joke: str, data: dict, audiences: List[str], raw: str
    ) -> GenomeReport:
        dims_in = {d.get("name"): d for d in data.get("dimensions", []) if isinstance(d, dict)}
        dimensions = []
        for axis in GENOME_AXES:
            d = dims_in.get(axis, {})
            dimensions.append(
                GenomeDimension(
                    name=axis,
                    score=max(0.0, min(10.0, _num(d.get("score")))),
                    note=str(d.get("note", "")),
                )
            )

        audiences_out = []
        for a in data.get("audiences", []):
            if not isinstance(a, dict):
                continue
            audiences_out.append(
                AudienceFit(
                    audience=str(a.get("audience", "")),
                    verdict=str(a.get("verdict", "")),
                    score=max(0.0, min(10.0, _num(a.get("score")))),
                    reasoning=str(a.get("reasoning", "")),
                )
            )

        def _as_list(v) -> List[str]:
            if isinstance(v, list):
                return [str(x) for x in v]
            if isinstance(v, str) and v:
                return [v]
            return []

        return GenomeReport(
            joke=joke,
            setup=str(data.get("setup", "")),
            expectation=str(data.get("expectation", "")),
            violation=str(data.get("violation", "")),
            payoff_mechanism=str(data.get("payoff_mechanism", "")),
            mechanisms=_as_list(data.get("mechanisms")),
            dimensions=dimensions,
            cultural_assumptions=_as_list(data.get("cultural_assumptions")),
            timing_notes=str(data.get("timing_notes", "")),
            failure_modes=_as_list(data.get("failure_modes")),
            audiences=audiences_out,
            one_line_explanation=str(data.get("one_line_explanation", "")),
            funniness=max(0.0, min(10.0, _num(data.get("funniness")))),
            raw_model_output=raw,
        )

    # ---------------------------------------------------------------- punch up
    def punch_up(
        self,
        report: GenomeReport,
        audience: str,
        n_weak_axes: int = 2,
    ) -> PunchUp:
        weak_axes = [
            d.name
            for d in sorted(report.dimensions, key=lambda x: x.score)[:n_weak_axes]
        ]
        prompt = punchup_prompt(report.joke, audience, weak_axes)
        raw = self.client.generate(prompt, system=SYSTEM_PROMPT)
        try:
            data = _extract_json(raw)
        except ValueError:
            data = {"rewrite": raw.strip()}

        return PunchUp(
            audience=audience,
            rewrite=str(data.get("rewrite", "")).strip(),
            mechanism_changed=str(data.get("mechanism_changed", "")),
            why_better=str(data.get("why_better", "")),
        )
