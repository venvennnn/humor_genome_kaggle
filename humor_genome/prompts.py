"""Prompt templates for the humor genome engine.

We ask Gemma to return strict JSON so results can be rendered as structured UI
cards and radar charts rather than a wall of prose. The schema mirrors the
dataclasses in ``genome.py``.
"""

from __future__ import annotations

from typing import List

from .genome import GENOME_AXES, KNOWN_MECHANISMS

SYSTEM_PROMPT = (
    "You are a comedy theorist and writers'-room script doctor. You analyze WHY "
    "jokes work by decomposing their structure, surprise, and cultural context. "
    "You are precise, honest about when something is not funny, and you always "
    "answer with valid JSON only — no markdown, no commentary outside the JSON."
)


def _axes_list() -> str:
    return ", ".join(GENOME_AXES)


def _mechanisms_list() -> str:
    return ", ".join(KNOWN_MECHANISMS)


def analysis_prompt(joke: str, audiences: List[str]) -> str:
    """Prompt that asks Gemma to produce a full GenomeReport as JSON."""
    audiences_str = ", ".join(f'"{a}"' for a in audiences)
    return f"""Analyze the humor of the following joke and return ONLY a JSON object.

JOKE:
\"\"\"{joke}\"\"\"

Decompose the joke's "humor genome". Return JSON with EXACTLY these keys:

{{
  "setup": "the premise / what frame the joke establishes",
  "expectation": "what the setup makes the listener expect will happen",
  "violation": "how the payoff breaks that expectation",
  "payoff_mechanism": "the specific 'aha' that triggers the laugh",
  "mechanisms": ["subset of: {_mechanisms_list()}"],
  "dimensions": [
    {{"name": "<one of: {_axes_list()}>", "score": <0-10 number>, "note": "<short reason>"}}
  ],
  "cultural_assumptions": ["knowledge or context the listener must share to get it"],
  "timing_notes": "notes on rhythm, word economy, where the pause/beat lands",
  "failure_modes": ["ways this joke could bomb or who it would confuse"],
  "audiences": [
    {{"audience": "<one of: {audiences_str}>", "verdict": "kills|lands|polite chuckle|bombs", "score": <0-10>, "reasoning": "<why>"}}
  ],
  "one_line_explanation": "a single crisp sentence explaining why it is (or isn't) funny",
  "funniness": <overall 0-10 number>
}}

Rules:
- Include one dimensions entry for EACH of: {_axes_list()}.
- Include one audiences entry for EACH of: {audiences_str}.
- Be candid: if the joke is weak, say so and score it low.
- Return ONLY the JSON object, nothing else."""


def punchup_prompt(joke: str, audience: str, weak_axes: List[str]) -> str:
    """Prompt that asks Gemma to rewrite a joke for a target audience."""
    weak = ", ".join(weak_axes) if weak_axes else "overall punch"
    return f"""Rewrite the following joke so it lands harder specifically with this audience: "{audience}".

ORIGINAL JOKE:
\"\"\"{joke}\"\"\"

The original is weakest on: {weak}. Improve those while keeping the core premise recognizable.

Return ONLY a JSON object with EXACTLY these keys:
{{
  "rewrite": "the improved joke",
  "mechanism_changed": "what comedic mechanism you added or sharpened",
  "why_better": "one sentence on why this version lands harder for {audience}"
}}

Return ONLY the JSON object, nothing else."""
