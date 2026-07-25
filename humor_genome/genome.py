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
class Punchline:
    """A single laugh trigger detected in a joke.

    A joke can have more than one: a main punchline plus tags/toppers,
    act-outs, or callbacks that each earn their own laugh.
    """

    text: str
    kind: str = "punchline"   # punchline | tag/topper | callback | act-out
    mechanism: str = ""       # the comedic mechanism it uses
    strength: float = 0.0     # 0-10 how hard this specific line hits

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImprovementSuggestion:
    """One concrete, actionable way to make the joke funnier."""

    issue: str        # what's holding the joke back
    suggestion: str   # how to fix it
    example: str = ""  # optional rewritten fragment demonstrating the fix

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
    punchlines: List[Punchline] = field(default_factory=list)
    dimensions: List[GenomeDimension] = field(default_factory=list)
    cultural_assumptions: List[str] = field(default_factory=list)
    timing_notes: str = ""
    failure_modes: List[str] = field(default_factory=list)
    audiences: List[AudienceFit] = field(default_factory=list)
    improvements: List[ImprovementSuggestion] = field(default_factory=list)
    one_line_explanation: str = ""
    funniness: float = 0.0      # overall 0-10 estimate
    needs_work: bool = False    # true when the joke is weak enough to flag fixes
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


# ---------------------------------------------------------------------- video


@dataclass
class ReactionMoment:
    """A detected audience reaction (laughter/applause) in a video's audio."""

    start_s: float
    end_s: float
    intensity: float  # 0-1 normalized energy of the burst
    kind: str = "laughter/applause"

    @property
    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoBeat:
    """One comedic beat/moment in a clip and how the audience responded."""

    start_s: float
    end_s: float
    moment: str = ""            # what happens / the line delivered
    is_joke: bool = True
    landed: bool = False        # did the audience actually laugh?
    audience_reaction: float = 0.0  # 0-10 measured reaction strength
    mechanism: str = ""         # comedic mechanism, if it is a joke
    explanation: str = ""       # WHY it landed or fell flat

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictedLaugh:
    """Where Gemma predicts a laugh SHOULD land, reading the transcript alone
    (blind to the real audience reaction). Used for predicted-vs-actual overlay.
    """

    time_s: float
    quote: str = ""            # the line expected to get the laugh
    expected_intensity: float = 0.0  # 0-1 how big a laugh is expected
    why: str = ""              # why the model expects a laugh here

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LaughGap:
    """One entry in the predicted-vs-actual comparison."""

    time_s: float
    kind: str  # "matched" | "bombed" (predicted, silent) | "surprise" (laugh, unpredicted)
    predicted_intensity: float = 0.0
    measured_intensity: float = 0.0
    quote: str = ""
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoHumorReport:
    """Full analysis of a comedy video clip."""

    source: str
    duration_s: float = 0.0
    transcript: str = ""
    has_transcript: bool = False
    frames_analyzed: int = 0
    multimodal_used: bool = False
    reactions: List[ReactionMoment] = field(default_factory=list)
    envelope_t: List[float] = field(default_factory=list)  # waveform time axis (s)
    envelope_v: List[float] = field(default_factory=list)  # 0-1 reaction energy
    laugh_coverage: float = 0.0   # fraction of clip time under a reaction
    biggest_laugh_s: float = 0.0  # timestamp of the strongest reaction
    beats: List[VideoBeat] = field(default_factory=list)
    overall_summary: str = ""
    what_worked: List[str] = field(default_factory=list)
    what_fell_flat: List[str] = field(default_factory=list)
    # predicted-vs-actual ("theory tested against reality")
    predicted_laughs: List[PredictedLaugh] = field(default_factory=list)
    laugh_gaps: List[LaughGap] = field(default_factory=list)
    prediction_summary: str = ""
    prediction_hit_rate: float = 0.0  # fraction of predicted laughs that landed
    notes: List[str] = field(default_factory=list)  # warnings/diagnostics for the UI
    raw_model_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------- set / special analysis


@dataclass
class SetJoke:
    """One bit/joke within a full set, with its genome fingerprint."""

    index: int
    text: str
    time_s: float = 0.0
    funniness: float = 0.0
    mechanisms: List[str] = field(default_factory=list)
    axes: Dict[str, float] = field(default_factory=dict)  # axis name -> 0-10
    cluster: int = -1

    def vector(self, axis_order: List[str]) -> List[float]:
        return [float(self.axes.get(a, 0.0)) for a in axis_order]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StyleCluster:
    """A cluster of bits that share a comedic style (a 'fingerprint')."""

    label: int
    name: str                      # human-readable, e.g. "Warm & relatable"
    size: int
    centroid: Dict[str, float] = field(default_factory=dict)
    dominant_axes: List[str] = field(default_factory=list)
    exemplar: str = ""             # representative bit text
    member_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CallbackLink:
    """A callback that pays off an earlier setup, with attributed 'yield'."""

    setup_index: int
    callback_index: int
    setup_text: str = ""
    callback_text: str = ""
    note: str = ""
    yield_value: float = 0.0  # laughter/funniness the callback owes to the setup

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SetReport:
    """Analysis of a full set / special transcript."""

    source: str
    jokes: List[SetJoke] = field(default_factory=list)
    clusters: List[StyleCluster] = field(default_factory=list)
    callbacks: List[CallbackLink] = field(default_factory=list)
    pca_coords: List[List[float]] = field(default_factory=list)  # per-joke [x, y]
    summary: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
