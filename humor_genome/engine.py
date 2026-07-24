"""The humor genome analysis engine.

Orchestrates: build prompt -> call Gemma -> parse JSON -> validate/normalize into
the dataclasses in ``genome.py``. Robust JSON extraction is important because
instruction-tuned models occasionally wrap JSON in prose or code fences.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

import os
import tempfile

from .gemma_client import GemmaClient, GemmaConfig
from .genome import (
    GENOME_AXES,
    GenomeReport,
    GenomeDimension,
    AudienceFit,
    PunchUp,
    Punchline,
    ImprovementSuggestion,
    ReactionMoment,
    VideoBeat,
    VideoHumorReport,
)
from .prompts import SYSTEM_PROMPT, analysis_prompt, punchup_prompt, video_prompt

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

        punchlines = []
        for p in data.get("punchlines", []):
            if not isinstance(p, dict):
                # tolerate a bare string punchline
                if isinstance(p, str) and p.strip():
                    punchlines.append(Punchline(text=p.strip()))
                continue
            text = str(p.get("text", "")).strip()
            if not text:
                continue
            punchlines.append(
                Punchline(
                    text=text,
                    kind=str(p.get("kind", "punchline")),
                    mechanism=str(p.get("mechanism", "")),
                    strength=max(0.0, min(10.0, _num(p.get("strength")))),
                )
            )

        improvements = []
        for s in data.get("improvements", []):
            if isinstance(s, str) and s.strip():
                improvements.append(ImprovementSuggestion(issue="", suggestion=s.strip()))
                continue
            if not isinstance(s, dict):
                continue
            suggestion = str(s.get("suggestion", "")).strip()
            if not suggestion and not s.get("issue"):
                continue
            improvements.append(
                ImprovementSuggestion(
                    issue=str(s.get("issue", "")),
                    suggestion=suggestion,
                    example=str(s.get("example", "")),
                )
            )

        funniness = max(0.0, min(10.0, _num(data.get("funniness"))))

        return GenomeReport(
            joke=joke,
            setup=str(data.get("setup", "")),
            expectation=str(data.get("expectation", "")),
            violation=str(data.get("violation", "")),
            payoff_mechanism=str(data.get("payoff_mechanism", "")),
            mechanisms=_as_list(data.get("mechanisms")),
            punchlines=punchlines,
            dimensions=dimensions,
            cultural_assumptions=_as_list(data.get("cultural_assumptions")),
            timing_notes=str(data.get("timing_notes", "")),
            failure_modes=_as_list(data.get("failure_modes")),
            audiences=audiences_out,
            improvements=improvements,
            one_line_explanation=str(data.get("one_line_explanation", "")),
            funniness=funniness,
            needs_work=funniness < 6.0,
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

    # ------------------------------------------------------------- video path
    def analyze_video(
        self,
        video_path: str,
        transcript: str = "",
        n_frames: int = 6,
    ) -> VideoHumorReport:
        """Analyze a comedy clip: detect audience reactions, then have Gemma
        explain why each beat did or didn't land.

        Degrades gracefully: if ffmpeg is unavailable it still runs on the
        transcript alone; frames are only used when the backend is multimodal.
        """
        from . import laughter as laughter_mod
        from . import video as video_mod

        report = VideoHumorReport(source=os.path.basename(video_path))
        tmpdir = tempfile.mkdtemp(prefix="humor_video_")

        reactions: list[ReactionMoment] = []
        frame_paths: list[str] = []
        want_frames = self.client.supports_images and n_frames > 0

        try:
            report.duration_s = video_mod.probe_duration(video_path)
        except Exception:
            report.duration_s = 0.0

        # 1) audio -> reaction timeline
        try:
            wav = video_mod.extract_audio_wav(
                video_path, os.path.join(tmpdir, "audio.wav")
            )
            waveform, rate = laughter_mod.read_wav_mono(wav)
            reactions = laughter_mod.detect_reactions(waveform, rate)
        except Exception:
            reactions = []

        # 2) frames for multimodal reasoning
        if want_frames:
            try:
                frame_paths = video_mod.extract_frames(video_path, tmpdir, n=n_frames)
            except Exception:
                frame_paths = []

        # 3) transcript (SRT/VTT/plain)
        plain, _cues = video_mod.parse_transcript(transcript)

        report.reactions = reactions
        report.frames_analyzed = len(frame_paths)
        report.multimodal_used = bool(frame_paths)
        report.transcript = plain
        report.has_transcript = bool(plain)
        report.laugh_coverage = laughter_mod.laugh_coverage(reactions, report.duration_s)
        report.biggest_laugh_s = laughter_mod.biggest_laugh(reactions)

        reactions_desc = (
            "\n".join(
                f"- {r.start_s:.2f}s-{r.end_s:.2f}s (intensity {r.intensity:.2f})"
                for r in reactions
            )
            or "- (no audience reactions detected in the audio)"
        )

        prompt = video_prompt(
            transcript=plain,
            reactions_desc=reactions_desc,
            duration_s=report.duration_s,
            n_frames=len(frame_paths),
            has_frames=bool(frame_paths),
        )
        raw = self.client.generate(
            prompt, system=SYSTEM_PROMPT, images=frame_paths or None
        )
        report.raw_model_output = raw

        try:
            data = _extract_json(raw)
        except ValueError:
            report.overall_summary = "(could not parse model output)"
            return report

        report.overall_summary = str(data.get("overall_summary", ""))

        def _as_list(v):
            if isinstance(v, list):
                return [str(x) for x in v]
            return [str(v)] if v else []

        report.what_worked = _as_list(data.get("what_worked"))
        report.what_fell_flat = _as_list(data.get("what_fell_flat"))

        for b in data.get("beats", []):
            if not isinstance(b, dict):
                continue
            start_s = _num(b.get("start_s"))
            end_s = _num(b.get("end_s"))
            beat = VideoBeat(
                start_s=start_s,
                end_s=end_s,
                moment=str(b.get("moment", "")),
                is_joke=bool(b.get("is_joke", True)),
                landed=bool(b.get("landed", False)),
                mechanism=str(b.get("mechanism", "")),
                explanation=str(b.get("explanation", "")),
            )
            beat.audience_reaction = round(
                10.0 * self._reaction_near(reactions, start_s, end_s), 1
            )
            report.beats.append(beat)

        return report

    @staticmethod
    def _reaction_near(
        reactions: list, start_s: float, end_s: float, window: float = 4.0
    ) -> float:
        """Peak reaction intensity (0-1) occurring within a beat or shortly after."""
        best = 0.0
        for r in reactions:
            if r.end_s >= start_s and r.start_s <= end_s + window:
                best = max(best, r.intensity)
        return best
