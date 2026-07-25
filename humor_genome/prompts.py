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
  "punchlines": [
    {{"text": "the exact laugh-line from the joke", "kind": "punchline|tag/topper|callback|act-out", "mechanism": "<one of: {_mechanisms_list()}>", "strength": <0-10 how hard this specific line hits>}}
  ],
  "dimensions": [
    {{"name": "<one of: {_axes_list()}>", "score": <0-10 number>, "note": "<short reason>"}}
  ],
  "cultural_assumptions": ["knowledge or context the listener must share to get it"],
  "timing_notes": "notes on rhythm, word economy, where the pause/beat lands",
  "failure_modes": ["ways this joke could bomb or who it would confuse"],
  "audiences": [
    {{"audience": "<one of: {audiences_str}>", "verdict": "kills|lands|polite chuckle|bombs", "score": <0-10>, "reasoning": "<why>"}}
  ],
  "improvements": [
    {{"issue": "what specifically holds this joke back", "suggestion": "a concrete, actionable fix a comedy writer could apply", "example": "optional short rewritten fragment showing the fix"}}
  ],
  "one_line_explanation": "a single crisp sentence explaining why it is (or isn't) funny",
  "funniness": <overall 0-10 number>
}}

Rules:
- Include one dimensions entry for EACH of: {_axes_list()}.
- Include one audiences entry for EACH of: {audiences_str}.
- DETECT ALL PUNCHLINES: list every distinct laugh line — the main punchline AND any tags/toppers, callbacks, or act-outs. Quote the exact text of each. If the joke genuinely has only one, return one.
- ALWAYS give at least 2 improvement suggestions. Make them specific to THIS joke (name the weak setup word, the flabby phrasing, the missing act-out) — never generic advice. For strong jokes, suggest sharper tags or alternate angles.
- Be candid: if the joke is weak, say so and score it low.
- Return ONLY the JSON object, nothing else."""


def video_prompt(
    transcript: str,
    reactions_desc: str,
    duration_s: float,
    n_frames: int,
    has_frames: bool,
) -> str:
    """Prompt for analyzing a comedy clip.

    We give Gemma the transcript (if any), the MEASURED audience-reaction
    timeline (from audio), and — for multimodal backends — sampled frames. The
    measured reactions let the model ground "did they laugh?" in real data
    instead of guessing.
    """
    frame_note = (
        f"You are also shown {n_frames} still frames sampled evenly across the clip. "
        "Use them for physical/visual comedy, facial expressions, and context."
        if has_frames
        else "No video frames are available; reason from the transcript and reactions."
    )
    transcript_block = (
        f'TRANSCRIPT:\n"""{transcript}"""'
        if transcript
        else "TRANSCRIPT: (none provided — infer content from frames/reactions where possible)"
    )
    return f"""You are analyzing a {duration_s:.0f}-second comedy video clip to explain WHY it is (or isn't) funny and WHY the audience did or didn't laugh.

{transcript_block}

MEASURED AUDIENCE REACTIONS (detected from the audio track — timestamps where the crowd laughs/applauds, with 0-1 intensity):
{reactions_desc}

{frame_note}

Break the clip into comedic BEATS. For each beat decide whether it is a joke, whether the audience actually laughed (cross-reference the measured reactions), and explain the comedic mechanism and WHY it landed or fell flat.

Return ONLY a JSON object with EXACTLY these keys:
{{
  "overall_summary": "2-3 sentences: what kind of comedy this is and how it performed",
  "beats": [
    {{
      "start_s": <number>,
      "end_s": <number>,
      "moment": "what happens / the line delivered",
      "is_joke": <true|false>,
      "landed": <true|false, grounded in the measured reactions>,
      "mechanism": "<comedic mechanism, e.g. misdirection/act-out/callback/taboo/absurdism>",
      "explanation": "WHY it landed (what expectation was violated) or WHY it fell flat"
    }}
  ],
  "what_worked": ["concrete reasons the laughs happened"],
  "what_fell_flat": ["moments that got no reaction and why"]
}}

Rules:
- Ground "landed" in the measured reactions: a beat landed if a reaction occurs at/just after it.
- If a beat is a joke but got no measured laugh, set landed=false and explain the likely reason (timing, unclear setup, wrong audience, cultural reference).
- Return ONLY the JSON object, nothing else."""


def laugh_prediction_prompt(transcript: str, duration_s: float) -> str:
    """Ask Gemma to predict where laughs SHOULD land, from the transcript alone.

    Deliberately blind to the measured audience reactions — this is the model's
    'theory of the joke', which we later test against reality.
    """
    return f"""You are reading the transcript of a ~{duration_s:.0f}-second comedy performance. You CANNOT hear the audience. Predict, from the words alone, exactly where the laughs SHOULD land and how big each should be.

TRANSCRIPT:
\"\"\"{transcript}\"\"\"

Return ONLY a JSON object with EXACTLY these keys:
{{
  "predicted_laughs": [
    {{
      "time_s": <approximate seconds into the clip where the laugh lands; spread these across 0..{duration_s:.0f}>,
      "quote": "the exact line that should trigger the laugh",
      "expected_intensity": <0-1 how big a laugh you expect>,
      "why": "why this line should get a laugh"
    }}
  ],
  "prediction_summary": "1-2 sentences on the comedic rhythm you expect"
}}

Rules:
- Base predictions ONLY on the text — do not assume anything about delivery or crowd.
- Order predicted_laughs by time_s. Estimate times proportionally to where each line sits in the transcript.
- Return ONLY the JSON object, nothing else."""


def callback_prompt(bits_desc: str) -> str:
    """Ask Gemma to find callbacks across a set and attribute their 'yield'."""
    return f"""Below is an ORDERED list of bits from a stand-up set (index: text). Comedians often 'call back' to an earlier premise for a bigger laugh later. Find every callback and link it to the earlier bit whose premise it pays off.

BITS:
{bits_desc}

Return ONLY a JSON object with EXACTLY these keys:
{{
  "callbacks": [
    {{
      "setup_index": <index of the earlier bit that planted the premise>,
      "callback_index": <index of the later bit that pays it off>,
      "note": "what premise is being called back and why it lands harder the second time",
      "yield": <0-10 how much of the callback's laugh is owed to the earlier setup>
    }}
  ]
}}

Rules:
- setup_index must be strictly less than callback_index.
- Only include genuine callbacks (shared specific premise), not generic topic overlap.
- If there are no callbacks, return an empty list.
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
