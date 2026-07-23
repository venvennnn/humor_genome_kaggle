"""Deterministic, offline stand-in for Gemma.

This is NOT trying to be smart — it exists so the whole pipeline (engine, CLI,
Streamlit app, tests) runs with zero model weights, and so judges can click
around before wiring up a real Gemma backend. It uses cheap heuristics to emit
schema-valid JSON that mirrors what Gemma returns.

The UI always labels this output clearly as mock.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import List

from .genome import GENOME_AXES, KNOWN_MECHANISMS


def _stable_unit(seed: str) -> float:
    """Deterministic float in [0, 1) derived from a string."""
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _score(seed: str, low: float = 2.0, high: float = 9.0) -> float:
    return round(low + (high - low) * _stable_unit(seed), 1)


def _extract_between(text: str, marker: str) -> str:
    """Grab the triple-quoted block following a marker like ``JOKE:``."""
    idx = text.find(marker)
    if idx == -1:
        return ""
    after = text[idx + len(marker):]
    m = re.search(r'"""(.*?)"""', after, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _extract_audiences(text: str) -> List[str]:
    m = re.search(r"audiences entry for EACH of:\s*(.+?)\.", text)
    segment = m.group(1) if m else ""
    auds = re.findall(r'"([^"]+)"', segment)
    return auds or ["General public"]


def _detect_mechanisms(joke: str) -> List[str]:
    j = joke.lower()
    found = []
    if any(w in j for w in ("why", "knock", "what do you call", "difference between")):
        found.append("misdirection")
    if re.search(r"\b(\w+)\b.*\b\1\b", j):
        found.append("callback")
    if any(p in j for p in ("!", "literally", "so ", "never")):
        found.append("exaggeration")
    if "i " in j or "my " in j:
        found.append("self-deprecation")
    # crude wordplay detector: repeated stems / homophone-ish endings
    words = re.findall(r"[a-z']+", j)
    if len({w[:4] for w in words}) < len(words) * 0.7 and len(words) > 4:
        found.append("wordplay")
    if not found:
        found.append("incongruity")
    # always seed with the workhorse of comedy
    if "incongruity" not in found:
        found.insert(0, "incongruity")
    return found[:4]


def _mock_analysis(joke: str, audiences: List[str]) -> str:
    words = re.findall(r"[A-Za-z']+", joke)
    n = max(1, len(words))
    mechanisms = _detect_mechanisms(joke)

    dims = []
    for axis in GENOME_AXES:
        note = {
            "surprise": "estimated from setup/payoff distance",
            "specificity": "based on presence of concrete nouns",
            "cleverness": "based on detected wordplay/structure",
            "relatability": "based on everyday vocabulary",
            "edge": "based on risky/taboo terms",
            "warmth": "based on target of the joke",
        }[axis]
        dims.append({"name": axis, "score": _score(joke + axis), "note": note})

    aud_entries = []
    for a in audiences:
        s = _score(joke + a)
        verdict = (
            "kills" if s >= 7.5 else
            "lands" if s >= 5.5 else
            "polite chuckle" if s >= 3.5 else
            "bombs"
        )
        aud_entries.append({
            "audience": a,
            "verdict": verdict,
            "score": s,
            "reasoning": f"[mock] predicted from lexical fit for {a}.",
        })

    funniness = round(sum(d["score"] for d in dims) / len(dims), 1)

    payload = {
        "setup": " ".join(words[: max(1, n // 2)]) + ("…" if n > 1 else ""),
        "expectation": "[mock] The setup steers you toward a literal reading.",
        "violation": "[mock] The payoff reframes an assumption from the setup.",
        "payoff_mechanism": f"[mock] Relies primarily on {mechanisms[0]}.",
        "mechanisms": mechanisms,
        "dimensions": dims,
        "cultural_assumptions": [
            "[mock] Shared familiarity with the words used in the setup.",
        ],
        "timing_notes": "[mock] Payoff should land on the final beat; trim words before it.",
        "failure_modes": [
            "[mock] Falls flat if the audience doesn't share the setup's frame.",
        ],
        "audiences": aud_entries,
        "one_line_explanation": (
            f"[mock] It works by setting up one frame and snapping to another via {mechanisms[0]}."
        ),
        "funniness": funniness,
    }
    return json.dumps(payload)


def _mock_punchup(joke: str, audience: str) -> str:
    payload = {
        "rewrite": f"{joke.rstrip('.')} — and honestly, that's the most {audience.lower()} thing I've ever admitted.",
        "mechanism_changed": "added self-deprecation + a callback tag",
        "why_better": f"[mock] Adds a targeted tag that flatters the in-group knowledge of {audience}.",
    }
    return json.dumps(payload)


def _mock_video(prompt: str) -> str:
    # parse measured reactions like "- 12.30s-13.10s (intensity 0.87)"
    reacts = re.findall(r"([\d.]+)s\s*[-–]\s*([\d.]+)s\s*\(intensity\s*([\d.]+)\)", prompt)
    transcript = _extract_between(prompt, "TRANSCRIPT:")
    words = re.findall(r"[A-Za-z']+", transcript)

    beats = []
    for idx, (start, end, inten) in enumerate(reacts):
        s, e, it = float(start), float(end), float(inten)
        # grab a slice of transcript as the "moment" if we have text
        moment = (
            " ".join(words[idx * 6 : idx * 6 + 8])
            if words else f"beat around {s:.0f}s"
        )
        beats.append({
            "start_s": max(0.0, s - 3.0),
            "end_s": e,
            "moment": moment or f"beat around {s:.0f}s",
            "is_joke": True,
            "landed": it >= 0.4,
            "mechanism": "misdirection" if idx % 2 == 0 else "act-out",
            "explanation": f"[mock] Measured a reaction at {s:.1f}s (intensity {it:.2f}); the payoff broke the setup's expectation.",
        })

    if not beats:
        beats.append({
            "start_s": 0.0,
            "end_s": 5.0,
            "moment": " ".join(words[:8]) if words else "opening",
            "is_joke": True,
            "landed": False,
            "mechanism": "observational",
            "explanation": "[mock] No audience reactions were detected in the audio; likely flat or a non-comedic segment.",
        })

    landed = sum(1 for b in beats if b["landed"])
    payload = {
        "overall_summary": (
            f"[mock] Detected {len(reacts)} audience reaction(s) across the clip; "
            f"{landed}/{len(beats)} beats landed. Heuristic analysis — connect a "
            f"multimodal Gemma (gemma3n) backend for real reasoning."
        ),
        "beats": beats,
        "what_worked": [
            f"[mock] Beats with measured laughs at "
            + ", ".join(f"{float(s):.0f}s" for s, _, _ in reacts) if reacts
            else "[mock] (no measured laughs)",
        ],
        "what_fell_flat": [
            "[mock] Segments with no detected reaction — check setup clarity/timing.",
        ],
    }
    return json.dumps(payload)


def mock_response(prompt: str) -> str:
    """Route a prompt to the right mock generator based on its content."""
    if "comedy video clip" in prompt:
        return _mock_video(prompt)
    if "Rewrite the following joke" in prompt:
        joke = _extract_between(prompt, "ORIGINAL JOKE:")
        m = re.search(r'this audience:\s*"([^"]+)"', prompt)
        audience = m.group(1) if m else "General public"
        return _mock_punchup(joke, audience)

    joke = _extract_between(prompt, "JOKE:")
    audiences = _extract_audiences(prompt)
    return _mock_analysis(joke, audiences)
