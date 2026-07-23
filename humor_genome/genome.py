"""Structured data models for a joke's "humor genome".

These dataclasses define the shape of everything the engine produces. Keeping
them separate from the prompting/model code means the UI, CLI, and tests can all
depend on a stable schema regardless of which Gemma backend is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


# The six axes we score every joke on. These are the "genes" of the humor
# genome: each is a 0-10 intensity, and together they form a fingerprint that
# can be compared across jokes and audiences.
GENOME_AXES: List[str] = [
    "surprise",       # how much the payoff violates the setup's expectation
    "specificity",    # concrete, vivid detail vs. generic
    "cleverness",     # wordplay / logical elegance of the mechanism
    "relatability",   # how broadly the premise is shared
    "edge",           # taboo / risk / transgression
    "warmth",         # affection vs. cruelty in the target
]

# Recognized comedic mechanisms. The engine is asked to map a joke onto these
# so results are consistent and explainable rather than free-form prose.
KNOWN_MECHANISMS: List[str] = [
    "incongruity",
    "misdirection",
    "wordplay",
    "callback",
    "act-out",
    "exaggeration",
    "understatement",
    "taboo",
    "superiority",
    "absurdism",
    "self-deprecation",
    "rule-of-three",
    "irony",
    "observational",
]


@dataclass
class GenomeDimension:
    """One scored axis of the humor genome."""

    name: str
    score: float  # 0-10
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AudienceFit:
    """Predicted reaction for a named audience segment."""

    audience: str
    verdict: str          # e.g. "kills", "lands", "polite chuckle", "bombs"
    score: float          # 0-10 predicted laugh strength
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenomeReport:
    """The full decomposition of a single joke."""

    joke: str
    setup: str = ""
    expectation: str = ""       # what the setup makes you expect
    violation: str = ""         # how the payoff breaks that expectation
    payoff_mechanism: str = ""  # the "aha" that triggers the laugh
    mechanisms: List[str] = field(default_factory=list)
    dimensions: List[GenomeDimension] = field(default_factory=list)
    cultural_assumptions: List[str] = field(default_factory=list)
    timing_notes: str = ""
    failure_modes: List[str] = field(default_factory=list)
    audiences: List[AudienceFit] = field(default_factory=list)
    one_line_explanation: str = ""
    funniness: float = 0.0      # overall 0-10 estimate
    raw_model_output: str = ""  # kept for transparency / debugging

    def dimension(self, name: str) -> float:
        for d in self.dimensions:
            if d.name == name:
                return d.score
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class PunchUp:
    """A rewrite of a joke targeted at a specific audience."""

    audience: str
    rewrite: str
    mechanism_changed: str = ""
    why_better: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
