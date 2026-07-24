"""Command-line interface for the humor genome engine.

Examples:
    python -m humor_genome.cli "Why did the scarecrow win an award?..."
    python -m humor_genome.cli --example dad-scarecrow --json
    python -m humor_genome.cli --backend mock "my joke here" --punchup "Tech crowd"
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .data import load_examples
from .engine import HumorGenomeEngine, DEFAULT_AUDIENCES
from .gemma_client import GemmaConfig
from .genome import GenomeReport


def _bar(score: float, width: int = 20) -> str:
    filled = int(round(score / 10 * width))
    return "█" * filled + "░" * (width - filled)


def _print_report(report: GenomeReport) -> None:
    print("=" * 64)
    print(f"JOKE: {report.joke}")
    print("=" * 64)
    print(f"\nOverall funniness: {report.funniness}/10  {_bar(report.funniness)}")
    print(f"\n> {report.one_line_explanation}\n")

    print(f"Setup:       {report.setup}")
    print(f"Expectation: {report.expectation}")
    print(f"Violation:   {report.violation}")
    print(f"Payoff:      {report.payoff_mechanism}")
    print(f"Mechanisms:  {', '.join(report.mechanisms)}")

    if report.punchlines:
        print("\nPunchlines detected:")
        for i, p in enumerate(report.punchlines, 1):
            print(f"  {i}. [{p.kind}] {p.strength:>4}/10 ({p.mechanism})")
            print(f"     \"{p.text}\"")

    print("\nHumor genome:")
    for d in report.dimensions:
        print(f"  {d.name:<12} {d.score:>4}/10  {_bar(d.score)}  {d.note}")

    if report.cultural_assumptions:
        print("\nCultural assumptions:")
        for c in report.cultural_assumptions:
            print(f"  - {c}")

    if report.timing_notes:
        print(f"\nTiming: {report.timing_notes}")

    if report.failure_modes:
        print("\nFailure modes:")
        for f in report.failure_modes:
            print(f"  - {f}")

    print("\nAudience fit:")
    for a in report.audiences:
        print(f"  {a.audience:<20} {a.verdict:<16} {a.score:>4}/10  {a.reasoning}")

    if report.improvements:
        header = (
            "\nHow to make it funnier:"
            if report.needs_work
            else "\nSuggestions to sharpen it further:"
        )
        print(header)
        for i, s in enumerate(report.improvements, 1):
            if s.issue:
                print(f"  {i}. Issue: {s.issue}")
                print(f"     Fix:   {s.suggestion}")
            else:
                print(f"  {i}. {s.suggestion}")
            if s.example:
                print(f"     e.g.:  {s.example}")


def _print_video_report(backend: str, report) -> None:
    print(f"[backend: {backend}]\n")
    print("=" * 64)
    print(f"CLIP: {report.source}  ({report.duration_s:.0f}s)")
    print("=" * 64)
    print(f"Laughs detected: {len(report.reactions)}  |  "
          f"coverage: {report.laugh_coverage * 100:.0f}%  |  "
          f"biggest laugh @ {report.biggest_laugh_s:.1f}s  |  "
          f"frames→Gemma: {report.frames_analyzed}")
    print(f"\n> {report.overall_summary}\n")

    for note in report.notes:
        print(f"[note] {note}\n")

    if report.reactions:
        print("Measured audience reactions:")
        for r in report.reactions:
            print(f"  {r.start_s:>6.1f}s–{r.end_s:<6.1f}s  intensity {r.intensity:.2f}  {_bar(r.intensity * 10)}")

    print("\nBeat-by-beat:")
    for b in report.beats:
        status = "LANDED " if b.landed else "no laugh"
        print(f"  [{b.start_s:>5.1f}–{b.end_s:<5.1f}s] {status}  react {b.audience_reaction:>4}/10  ({b.mechanism})")
        print(f"      moment: {b.moment}")
        print(f"      why:    {b.explanation}")

    if report.what_worked:
        print("\nWhat worked:")
        for w in report.what_worked:
            print(f"  + {w}")
    if report.what_fell_flat:
        print("\nWhat fell flat:")
        for w in report.what_fell_flat:
            print(f"  - {w}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="humor_genome",
        description="Why'd They Laugh? — decompose a joke's humor genome with Gemma.",
    )
    parser.add_argument("joke", nargs="?", help="the joke text to analyze")
    parser.add_argument("--example", help="analyze a bundled example by id")
    parser.add_argument("--list-examples", action="store_true", help="list bundled examples")
    parser.add_argument("--backend", default=None, help="auto|ollama|hf|mock")
    parser.add_argument("--punchup", metavar="AUDIENCE", help="also punch up the joke for AUDIENCE")
    parser.add_argument("--video", metavar="PATH", help="analyze a comedy video clip instead of a joke")
    parser.add_argument("--transcript", metavar="PATH", help="transcript file (.txt/.srt/.vtt) for --video")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of pretty text")
    args = parser.parse_args(argv)

    examples = load_examples()

    if args.list_examples:
        for ex in examples:
            print(f"{ex['id']:<22} {ex['label']:<28} {ex['joke'][:50]}...")
        return 0

    config = GemmaConfig(backend=args.backend) if args.backend else None

    if args.video:
        engine = HumorGenomeEngine(config=config)
        transcript = ""
        if args.transcript:
            try:
                with open(args.transcript, "r", encoding="utf-8") as f:
                    transcript = f.read()
            except OSError as e:
                print(f"Could not read transcript: {e}", file=sys.stderr)
                return 2
        report = engine.analyze_video(args.video, transcript=transcript)
        if args.json:
            print(json.dumps(
                {"backend": engine.describe_backend(), "report": report.to_dict()},
                indent=2,
            ))
        else:
            _print_video_report(engine.describe_backend(), report)
        return 0

    joke = args.joke
    if args.example:
        match = next((e for e in examples if e["id"] == args.example), None)
        if not match:
            print(f"No example with id '{args.example}'.", file=sys.stderr)
            return 2
        joke = match["joke"]

    if not joke:
        parser.print_help()
        return 2

    engine = HumorGenomeEngine(config=config)

    report = engine.analyze(joke)

    result = {"backend": engine.describe_backend(), "report": report.to_dict()}

    if args.punchup:
        punch = engine.punch_up(report, args.punchup)
        result["punchup"] = punch.to_dict()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"[backend: {engine.describe_backend()}]\n")
        _print_report(report)
        if args.punchup:
            punch = result["punchup"]
            print(f"\n--- Punch-up for '{args.punchup}' ---")
            print(f"Rewrite:  {punch['rewrite']}")
            print(f"Changed:  {punch['mechanism_changed']}")
            print(f"Why:      {punch['why_better']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
