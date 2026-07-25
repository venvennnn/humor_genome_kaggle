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


def _split_sentences(joke: str) -> List[str]:
    # keep the delimiter so question setups stay readable
    parts = re.split(r"(?<=[.!?])\s+", joke.strip())
    return [p.strip() for p in parts if p.strip()]


def _detect_punchlines(joke: str, mechanisms: List[str]) -> List[dict]:
    """Heuristically locate laugh lines: the last clause, plus any trailing tag."""
    sentences = _split_sentences(joke)
    punchlines: List[dict] = []
    if not sentences:
        return punchlines

    main = sentences[-1]
    # a dash/colon inside the final line often separates a punch from a tag
    tag = ""
    m = re.split(r"\s[—–-]\s|:\s", main, maxsplit=1)
    if len(m) == 2 and len(m[1].split()) >= 2:
        main, tag = m[0].strip(), m[1].strip()

    punchlines.append({
        "text": main,
        "kind": "punchline",
        "mechanism": mechanisms[0] if mechanisms else "incongruity",
        "strength": _score(joke + "punch", 3.0, 9.0),
    })
    if tag:
        punchlines.append({
            "text": tag,
            "kind": "tag/topper",
            "mechanism": mechanisms[1] if len(mechanisms) > 1 else "callback",
            "strength": _score(joke + "tag", 2.0, 8.0),
        })
    return punchlines


def _suggest_improvements(joke: str, dims: List[dict]) -> List[dict]:
    """Turn the two weakest genome axes into concrete, actionable fixes."""
    fixes = {
        "surprise": (
            "The turn is too predictable — the payoff sits close to what the setup implies.",
            "Widen the gap: misdirect harder in the setup so the punchline reframes it.",
            "Set up an innocent assumption, then reveal it meant something else entirely.",
        ),
        "specificity": (
            "The language is generic, so no vivid picture forms.",
            "Swap vague nouns for one hyper-specific, concrete detail.",
            "Not 'furniture' but 'a Swedish flat-pack wardrobe named BJÖRKSNÄS'.",
        ),
        "cleverness": (
            "The mechanism is doing little work — there's no wordplay or logical snap.",
            "Add a double meaning, a reversal, or a rule-of-three build.",
            "End on a third item that breaks the pattern the first two set.",
        ),
        "relatability": (
            "The premise is niche, so many listeners won't have a stake in it.",
            "Anchor the setup in a near-universal everyday frustration.",
            "Open with a moment everyone has lived, then twist it.",
        ),
        "edge": (
            "It plays it safe, so there's no tension to release as a laugh.",
            "Add a small, well-aimed transgression or an honest, uncomfortable truth.",
            "Punch at the situation (or yourself), not an easy target.",
        ),
        "warmth": (
            "The tone reads a bit cold, which limits how freely people laugh.",
            "Aim the joke at yourself or the situation to keep it likeable.",
            "Reframe the target as shared human folly rather than a put-down.",
        ),
    }
    weakest = sorted(dims, key=lambda d: d["score"])[:2]
    out = []
    for d in weakest:
        issue, suggestion, example = fixes.get(
            d["name"],
            ("This axis is weak.", "Tighten and sharpen the beat.", ""),
        )
        out.append({
            "issue": f"[mock] {issue} (low {d['name']}: {d['score']}/10)",
            "suggestion": suggestion,
            "example": example,
        })
    return out


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
        "punchlines": _detect_punchlines(joke, mechanisms),
        "dimensions": dims,
        "cultural_assumptions": [
            "[mock] Shared familiarity with the words used in the setup.",
        ],
        "timing_notes": "[mock] Payoff should land on the final beat; trim words before it.",
        "failure_modes": [
            "[mock] Falls flat if the audience doesn't share the setup's frame.",
        ],
        "audiences": aud_entries,
        "improvements": _suggest_improvements(joke, dims),
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
            "improvement": "[mock] Tighten the words before the turn so the punch lands on the last beat."
            if it < 0.6 else "",
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
            "improvement": "[mock] Add a concrete, surprising turn — the setup never pays off.",
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


def _mock_prediction(prompt: str) -> str:
    transcript = _extract_between(prompt, "TRANSCRIPT:")
    m = re.search(r"~([\d.]+)-second", prompt)
    duration = float(m.group(1)) if m else 60.0
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()]
    preds = []
    # treat every other sentence as a likely laugh line
    candidates = sentences[::2] or sentences
    n = max(1, len(candidates))
    for i, s in enumerate(candidates):
        t = round(duration * (i + 0.7) / n, 1)
        preds.append({
            "time_s": t,
            "quote": s[:120],
            "expected_intensity": round(0.4 + 0.5 * _stable_unit(s), 2),
            "why": "[mock] Reads like a payoff line (turn at the end of the thought).",
        })
    return json.dumps({
        "predicted_laughs": preds,
        "prediction_summary": (
            f"[mock] Expect roughly {len(preds)} laughs, paced across the clip. "
            "Connect real Gemma for a genuine read of the rhythm."
        ),
    })


def _mock_callbacks(prompt: str) -> str:
    pairs = re.findall(r"^(\d+):\s*(.+)$", prompt, re.MULTILINE)
    bits = [(int(i), t) for i, t in pairs]
    stop = set("the a an and or but to of in on for with that this is are was were "
               "i you it my me we they he she so about like just".split())

    def keywords(text):
        return {w for w in re.findall(r"[a-z']{5,}", text.lower()) if w not in stop}

    callbacks = []
    for ci, (cidx, ctext) in enumerate(bits):
        ckw = keywords(ctext)
        for sidx, stext in bits[:ci]:
            shared = ckw & keywords(stext)
            if shared:
                callbacks.append({
                    "setup_index": sidx,
                    "callback_index": cidx,
                    "note": f"[mock] Shares premise word(s): {', '.join(sorted(shared)[:3])}.",
                    "yield": round(3 + 6 * _stable_unit(ctext + stext), 1),
                })
                break
    return json.dumps({"callbacks": callbacks[:8]})


def mock_response(prompt: str) -> str:
    """Route a prompt to the right mock generator based on its content."""
    if '"predicted_laughs"' in prompt:
        return _mock_prediction(prompt)
    if '"callbacks"' in prompt:
        return _mock_callbacks(prompt)
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
