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
    PredictedLaugh,
    LaughGap,
    SetJoke,
    StyleCluster,
    CallbackLink,
    SetReport,
)
from .prompts import (
    SYSTEM_PROMPT,
    analysis_prompt,
    punchup_prompt,
    video_prompt,
    laugh_prediction_prompt,
    callback_prompt,
)

DEFAULT_AUDIENCES = [
    "Close friends",
    "Tech crowd",
    "General public",
    "Corporate all-hands",
]


def _normalize_smart_quotes(s: str) -> str:
    """Convert typographic (curly) quotes/dashes to ASCII.

    Instruction-tuned models very often emit “smart quotes” which are invalid
    JSON syntax and silently break parsing. Curly quotes are never valid JSON
    delimiters, so normalizing them is safe and fixes the common failure where a
    stray ” is used to close a string value.
    """
    return (
        s.replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2013", "-").replace("\u2014", "-")
        .replace("\u00a0", " ")
    )


def _strip_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)


def _bracket_state(s: str):
    """Scan JSON text, returning (open_bracket_stack, inside_unterminated_string)."""
    in_string = False
    escape = False
    stack: list = []
    for ch in s:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]") and stack and stack[-1] == ch:
            stack.pop()
    return stack, in_string


def _close_truncated_json(s: str) -> str:
    """Best-effort close of a truncated JSON object/array.

    Models often hit the token limit mid-value. Closes any unterminated string,
    resolves a dangling key/colon/comma, then appends the missing ``]``/``}``
    innermost-first. Pure Python — no third-party dependency required.
    """
    stack, in_string = _bracket_state(s)
    if in_string:
        # Terminate the cut-off string so its (partial) value is preserved.
        # Trailing whitespace must go first: a raw newline inside a JSON string
        # is an illegal control character.
        s = s.rstrip() + '"'

    s = s.rstrip()
    if s.endswith(":"):          # key with no value yet
        s += " null"
    s = s.rstrip()
    if s.endswith(","):          # dangling separator
        s = s[:-1]

    # Drop a dangling KEY (a quoted string sitting right after '{', '[' or ',',
    # i.e. with no ':' value). A quoted VALUE is preceded by ':' so it is kept.
    m = re.search(r'[{\[,]\s*"[^"]*"\s*$', s)
    if m:
        s = s[: m.start() + 1].rstrip()
        if s.endswith(","):
            s = s[:-1]

    stack, _ = _bracket_state(s)
    for closer in reversed(stack):
        s += closer
    return s


def _truncation_variants(s: str):
    """Yield progressively more conservative repairs of truncated JSON.

    First try closing in place (keeps the partial trailing element), then walk
    back to each earlier complete ``}`` and close from there — which reliably
    recovers all fully-written list elements.
    """
    yield _close_truncated_json(s)
    idx = len(s)
    for _ in range(60):
        idx = s.rfind("}", 0, idx)
        if idx == -1:
            break
        prefix = s[: idx + 1]
        stack, in_str = _bracket_state(prefix)
        if in_str or not stack:
            continue
        trimmed = prefix.rstrip()
        if trimmed.endswith(","):
            trimmed = trimmed[:-1]
        for closer in reversed(stack):
            trimmed += closer
        yield trimmed


def _extract_json_candidate(text: str) -> str:
    """Pull the most likely JSON object string out of a model response."""
    # complete fenced block
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)

    # fenced but truncated (opening fence, no closing) — take rest of text
    fenced_open = re.search(r"```(?:json)?\s*(\{.*)", text, re.DOTALL)
    if fenced_open:
        return fenced_open.group(1)

    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model response")

    # Prefer a balanced outer object when one exists; otherwise take to EOF
    # (truncated responses have no closing '}').
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # never balanced — truncated; take everything from the opening brace
    return text[start:]


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response.

    Robust to code fences, leading/trailing prose, trailing commas, smart
    quotes, AND truncated output (token-limit cutoffs mid-object). Falls back
    to the ``json_repair`` library when available. Raises ValueError only if
    nothing parseable can be recovered.
    """
    if not text:
        raise ValueError("empty model response")

    candidate = _extract_json_candidate(text)
    normalized = _normalize_smart_quotes(candidate)

    def _attempts():
        yield candidate
        yield _strip_trailing_commas(candidate)
        yield normalized
        yield _strip_trailing_commas(normalized)
        for variant in _truncation_variants(normalized):
            yield variant
            yield _strip_trailing_commas(variant)

    best: dict = {}
    for attempt in _attempts():
        try:
            obj = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            # keep the richest recovery (most beats survive truncation)
            if not best or len(json.dumps(obj)) > len(json.dumps(best)):
                best = obj
            return best

    # optional heavy-duty repair (handles missing quotes, unescaped chars, …)
    try:
        from json_repair import repair_json

        for text_in in (normalized, _close_truncated_json(normalized)):
            obj = repair_json(text_in, return_objects=True)
            if isinstance(obj, dict) and obj:
                return obj
    except Exception:
        pass

    if best:
        return best
    raise ValueError("could not parse JSON from model response")


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
        env_t: list[float] = []
        env_v: list[float] = []
        want_frames = self.client.supports_images and n_frames > 0

        try:
            report.duration_s = video_mod.probe_duration(video_path)
        except Exception:
            report.duration_s = 0.0

        # 1) audio -> reaction timeline + loudness envelope
        try:
            wav = video_mod.extract_audio_wav(
                video_path, os.path.join(tmpdir, "audio.wav")
            )
            waveform, rate = laughter_mod.read_wav_mono(wav)
            reactions = laughter_mod.detect_reactions(waveform, rate)
            env_t, env_v = laughter_mod.reaction_envelope(waveform, rate)
        except Exception:
            reactions = []

        # 2) frames for multimodal reasoning
        if want_frames:
            try:
                frame_paths = video_mod.extract_frames(video_path, tmpdir, n=n_frames)
            except Exception:
                frame_paths = []

        # 3) transcript (SRT/VTT/plain)
        plain, cues = video_mod.parse_transcript(transcript)
        report.envelope_t, report.envelope_v = env_t, env_v

        report.reactions = reactions
        report.frames_analyzed = len(frame_paths)
        # multimodal_used is set accurately after the model call (a vision call
        # can still fail even when frames were extracted).
        report.multimodal_used = False
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

        def _build_prompt(with_frames: bool) -> str:
            return video_prompt(
                transcript=plain,
                reactions_desc=reactions_desc,
                duration_s=report.duration_s,
                n_frames=len(frame_paths) if with_frames else 0,
                has_frames=with_frames,
            )

        # Video JSON is long (many beats + improvements). Use a higher token
        # budget so Gemma doesn't get cut off mid-object (the #1 parse failure).
        video_max_tokens = max(self.client.config.max_tokens, 4096)

        raw = ""
        if frame_paths:
            try:
                raw = self.client.generate(
                    _build_prompt(True),
                    system=SYSTEM_PROMPT,
                    images=frame_paths,
                    max_tokens=video_max_tokens,
                )
                report.multimodal_used = True
            except Exception as exc:  # vision model missing / rejects images
                report.multimodal_used = False
                report.notes.append(
                    "Frame (vision) analysis failed, so this used transcript + "
                    f"reactions only. Reason: {exc}"
                )
                raw = ""

        if not raw:
            # text-only path (no frames, or the vision call failed above)
            report.multimodal_used = False
            raw = self.client.generate(
                _build_prompt(False),
                system=SYSTEM_PROMPT,
                max_tokens=video_max_tokens,
            )

        if not report.multimodal_used:
            report.frames_analyzed = 0

        report.raw_model_output = raw

        try:
            data = _extract_json(raw)
        except ValueError:
            # Truncation / malformed JSON — keep the reaction timeline we already
            # built and surface a useful note instead of a blank report.
            looks_truncated = not raw.rstrip().endswith("}")
            report.overall_summary = (
                "(model output was truncated mid-JSON — raised the token budget; "
                "re-run the analysis. The laugh timeline above is still valid.)"
                if looks_truncated
                else "(could not parse model output — see raw output below)"
            )
            report.notes.append(
                "Gemma's JSON was incomplete or invalid, so beat-by-beat reasoning "
                "couldn't be loaded. The measured laughter timeline is still accurate. "
                "Tip: re-run, or set HUMOR_GENOME_MAX_TOKENS=4096."
            )
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
                improvement=str(b.get("improvement", "")).strip().strip('"'),
            )
            beat.audience_reaction = round(
                10.0 * self._reaction_near(reactions, start_s, end_s), 1
            )
            report.beats.append(beat)

        # When the model was truncated before writing what_worked/fell_flat,
        # synthesize them from the beats so the UI isn't empty.
        if not report.what_worked:
            report.what_worked = [
                f"{b.moment} ({b.mechanism})" for b in report.beats if b.landed and b.moment
            ][:3] or ["(no landed beats recorded)"]
        if not report.what_fell_flat:
            report.what_fell_flat = [
                (b.improvement or b.explanation or b.moment)
                for b in report.beats if (not b.landed) and b.is_joke
            ][:3] or ["(no flat joke-beats recorded)"]

        # Feature: predicted-vs-actual laughter (the model's theory, tested)
        if plain:
            try:
                self._predict_and_compare(report, plain, cues, reactions)
            except Exception as exc:
                report.notes.append(f"Laugh prediction skipped: {exc}")

        else:
            report.notes.append(
                "No transcript provided, so predicted-vs-actual is unavailable "
                "(it needs the words). The humor genome below is derived from the "
                "beats Gemma saw/heard — paste a transcript for a sharper read."
            )

        # Feature: full humor genome of the clip's material (radar, axes,
        # audience fit, punch-up) — same analysis as the text tab.
        # Prefer the real transcript; otherwise fall back to the material Gemma
        # already described (beat moments + summary) so the radar/scores and
        # improvement suggestions still appear.
        material = plain or self._material_from_beats(report)
        if material:
            try:
                report.genome = self.analyze(material, audiences=DEFAULT_AUDIENCES)
            except Exception as exc:
                report.notes.append(f"Genome analysis skipped: {exc}")

        return report

    @staticmethod
    def _material_from_beats(report: VideoHumorReport) -> str:
        """Reconstruct analyzable 'material' from a clip with no transcript."""
        parts = [b.moment.strip() for b in report.beats if b.moment.strip()]
        if not parts and report.overall_summary:
            parts = [report.overall_summary]
        return " ".join(parts).strip()

    # ------------------------------------------- predicted vs actual laughter
    def _predict_and_compare(
        self, report: VideoHumorReport, transcript: str, cues, reactions
    ) -> None:
        raw = self.client.generate(
            laugh_prediction_prompt(transcript, report.duration_s),
            system=SYSTEM_PROMPT,
        )
        try:
            data = _extract_json(raw)
        except ValueError:
            return

        report.prediction_summary = str(data.get("prediction_summary", ""))

        predicted: List[PredictedLaugh] = []
        for p in data.get("predicted_laughs", []):
            if not isinstance(p, dict):
                continue
            quote = str(p.get("quote", "")).strip()
            t = _num(p.get("time_s"))
            # if we have timed cues, snap the predicted time to the quoted line
            if cues and quote:
                t = self._time_of_quote(quote, cues, fallback=t)
            predicted.append(
                PredictedLaugh(
                    time_s=round(t, 2),
                    quote=quote,
                    expected_intensity=max(0.0, min(1.0, _num(p.get("expected_intensity")))),
                    why=str(p.get("why", "")),
                )
            )
        predicted.sort(key=lambda x: x.time_s)
        report.predicted_laughs = predicted

        gaps, hit_rate = self._compare_laughs(predicted, reactions)
        report.laugh_gaps = gaps
        report.prediction_hit_rate = hit_rate

    @staticmethod
    def _time_of_quote(quote: str, cues, fallback: float = 0.0) -> float:
        q = re.sub(r"[^a-z0-9 ]", "", quote.lower())[:40]
        best_t, best_overlap = fallback, 0
        for c in cues:
            ct = re.sub(r"[^a-z0-9 ]", "", c.text.lower())
            words = set(q.split())
            overlap = sum(1 for w in words if w and w in ct)
            if overlap > best_overlap:
                best_overlap, best_t = overlap, c.start_s
        return best_t

    @staticmethod
    def _compare_laughs(
        predicted: List[PredictedLaugh], reactions, window: float = 4.0
    ):
        gaps: List[LaughGap] = []
        used = set()
        matched = 0
        for p in predicted:
            hit = None
            for i, r in enumerate(reactions):
                if i in used:
                    continue
                center = (r.start_s + r.end_s) / 2
                if abs(center - p.time_s) <= window:
                    hit = (i, r)
                    break
            if hit is not None:
                used.add(hit[0])
                matched += 1
                gaps.append(LaughGap(
                    time_s=p.time_s, kind="matched",
                    predicted_intensity=p.expected_intensity,
                    measured_intensity=hit[1].intensity,
                    quote=p.quote,
                    explanation="Predicted here and the crowd delivered.",
                ))
            else:
                gaps.append(LaughGap(
                    time_s=p.time_s, kind="bombed",
                    predicted_intensity=p.expected_intensity,
                    measured_intensity=0.0,
                    quote=p.quote,
                    explanation="The text reads funny, but the room was silent — "
                    "a joke that bombed (delivery, timing, or wrong crowd?).",
                ))
        # measured reactions with no matching prediction = surprise laughs
        for i, r in enumerate(reactions):
            if i in used:
                continue
            gaps.append(LaughGap(
                time_s=round((r.start_s + r.end_s) / 2, 2), kind="surprise",
                predicted_intensity=0.0, measured_intensity=r.intensity,
                explanation="A real laugh the text didn't predict — likely "
                "delivery, an act-out, or physical comedy the transcript missed.",
            ))
        gaps.sort(key=lambda g: g.time_s)
        hit_rate = round(matched / len(predicted), 3) if predicted else 0.0
        return gaps, hit_rate

    # --------------------------------------------------- set / special analysis
    def analyze_set(
        self, transcript: str, max_bits: int = 40, source: str = "set"
    ) -> SetReport:
        """Analyze a whole set: genome per bit, style clusters, trajectory,
        and a setup->callback attribution graph.
        """
        from . import setsplit, clustering

        report = SetReport(source=source)
        bits = setsplit.split_into_bits(transcript, max_bits=max_bits)
        if not bits:
            report.notes.append("No bits could be parsed from the transcript.")
            return report

        for i, (text, t) in enumerate(bits):
            try:
                g = self.analyze(text, audiences=["General public"])
                axes = {d.name: d.score for d in g.dimensions}
                report.jokes.append(SetJoke(
                    index=i, text=text, time_s=round(t, 2),
                    funniness=g.funniness, mechanisms=g.mechanisms, axes=axes,
                ))
            except Exception as exc:
                report.notes.append(f"Bit {i} failed: {exc}")

        if not report.jokes:
            return report

        import numpy as np

        x = np.array([j.vector(GENOME_AXES) for j in report.jokes], dtype=float)
        k = clustering.suggest_k(len(report.jokes))
        labels, centroids = clustering.kmeans(x, k)
        coords = clustering.pca_2d(x)
        report.pca_coords = [[round(float(a), 3), round(float(b), 3)] for a, b in coords]

        for j, lab in zip(report.jokes, labels):
            j.cluster = int(lab)

        for lab in range(len(centroids)):
            members = [j for j in report.jokes if j.cluster == lab]
            if not members:
                continue
            cen = {a: round(float(centroids[lab][idx]), 2) for idx, a in enumerate(GENOME_AXES)}
            dominant = sorted(cen, key=cen.get, reverse=True)[:2]
            exemplar = max(members, key=lambda m: m.funniness)
            report.clusters.append(StyleCluster(
                label=lab,
                name=" + ".join(a.capitalize() for a in dominant),
                size=len(members),
                centroid=cen,
                dominant_axes=dominant,
                exemplar=exemplar.text,
                member_indices=[m.index for m in members],
            ))

        # callbacks / attribution
        try:
            self._detect_callbacks(report)
        except Exception as exc:
            report.notes.append(f"Callback detection skipped: {exc}")

        report.summary = (
            f"{len(report.jokes)} bits across {len(report.clusters)} style "
            f"cluster(s); {len(report.callbacks)} callback link(s) detected."
        )
        return report

    def _detect_callbacks(self, report: SetReport) -> None:
        bits_desc = "\n".join(
            f"{j.index}: {j.text[:160]}" for j in report.jokes
        )
        raw = self.client.generate(callback_prompt(bits_desc), system=SYSTEM_PROMPT)
        try:
            data = _extract_json(raw)
        except ValueError:
            return
        by_index = {j.index: j for j in report.jokes}
        for c in data.get("callbacks", []):
            if not isinstance(c, dict):
                continue
            si = int(_num(c.get("setup_index"), -1))
            ci = int(_num(c.get("callback_index"), -1))
            if si not in by_index or ci not in by_index or si >= ci:
                continue
            report.callbacks.append(CallbackLink(
                setup_index=si,
                callback_index=ci,
                setup_text=by_index[si].text,
                callback_text=by_index[ci].text,
                note=str(c.get("note", "")),
                yield_value=max(0.0, min(10.0, _num(c.get("yield")))),
            ))

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
