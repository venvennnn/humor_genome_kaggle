"""Why'd They Laugh? — a Gemma-powered humor understanding engine.

This package decomposes a joke into its "humor genome": the structural and
cultural mechanisms that make (or fail to make) people laugh, and predicts how
it will land with different audiences.

Public API:
    - GemmaClient: unified interface over Ollama / HuggingFace / offline mock.
    - HumorGenomeEngine: analyze a joke and punch it up for an audience.
    - GenomeReport, PunchUp: structured results.
"""

from .gemma_client import GemmaClient, GemmaConfig, BackendUnavailable
from .engine import HumorGenomeEngine
from .genome import (
    GenomeReport,
    GenomeDimension,
    PunchUp,
    AudienceFit,
    ReactionMoment,
    VideoBeat,
    VideoHumorReport,
)

__all__ = [
    "GemmaClient",
    "GemmaConfig",
    "BackendUnavailable",
    "HumorGenomeEngine",
    "GenomeReport",
    "GenomeDimension",
    "PunchUp",
    "AudienceFit",
    "ReactionMoment",
    "VideoBeat",
    "VideoHumorReport",
]

__version__ = "0.1.0"
