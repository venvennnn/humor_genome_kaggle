# 🎭 Why'd They Laugh?

**A Gemma-powered humor *understanding* engine.**
Paste a joke and see its **humor genome** — the structure, surprise, cultural
context, and audience fit that decide whether people actually laugh.

Built for **Build with Gemma: Humor Genome NYC** · Track 2 — *Humor Understanding*.

---

## What it does

Most "AI + comedy" tools *generate* jokes. This one **explains why a joke works
(or doesn't)** and predicts how it will land with different crowds — the harder,
more interesting half of the problem.

It works two ways:

### 🗣️ Analyze a joke (text)

Given any joke, Gemma decomposes it into a structured report:

- **Setup → Expectation → Violation → Payoff** — the anatomy of the laugh.
- **Comedic mechanisms** — misdirection, wordplay, callback, taboo, absurdism…
- **The humor genome** — six scored axes (`surprise`, `specificity`,
  `cleverness`, `relatability`, `edge`, `warmth`) rendered as a radar chart.
- **Cultural assumptions** the listener must share to "get it."
- **Timing notes** and **failure modes** (who it confuses, when it bombs).
- **Audience fit** — predicted verdict (*kills / lands / polite chuckle /
  bombs*) per audience segment, with reasoning.
- **Punch-up** — Gemma rewrites the joke to land harder with a chosen audience,
  and tells you *which mechanism it changed and why*.

### 🎬 Analyze a comedy clip (video)

Upload a short clip and the engine answers **"why did they laugh — or why
didn't they?"** grounded in the *actual audience reaction*:

1. **Reaction detection** — we extract the audio and detect where the crowd
   laughs/applauds (sustained, broadband energy bursts) → a reaction timeline.
2. **Frames → multimodal Gemma** — evenly sampled frames are sent to a
   multimodal Gemma (**Gemma 3 / 3n**) for physical/visual comedy and context.
3. **Beat-by-beat reasoning** — Gemma splits the clip into comedic beats and,
   cross-referencing the *measured* laughs, explains **why each beat landed or
   fell flat** (timing, unclear setup, wrong audience, cultural reference).

Optionally paste a transcript (plain text, `.srt`, or `.vtt`) to align jokes to
laughs precisely.

## Why this is interesting

Humor has no ground-truth label — the same joke kills in one room and bombs in
another. Instead of asking a model "is this funny?", we ask it to make its
reasoning **explicit and structured**: what expectation is set, how it's
violated, what knowledge is assumed, and who that excludes. That turns an opaque
gut reaction into an inspectable artifact you can argue with — useful for
writers, teachers, localizers, and comedy researchers.

## How Gemma is used

Gemma is the entire reasoning core. It performs:

1. **Structured decomposition** — a single instruction-tuned prompt asks Gemma
   to return strict JSON matching our genome schema (`humor_genome/prompts.py`).
2. **Audience simulation** — the same call predicts per-audience reactions,
   effectively role-playing different crowds.
3. **Targeted rewriting** — a second prompt feeds the joke's *weakest* genome
   axes back to Gemma to punch it up for a specific audience.
4. **Multimodal clip reasoning** — for video, sampled frames are passed to a
   multimodal Gemma along with the transcript and the *measured* laugh timeline,
   so the model grounds "did they laugh?" in real data instead of guessing.

The engine (`humor_genome/engine.py`) handles robust JSON extraction and clamps
values, so the UI stays stable even when a model wraps its output in prose.

## Which Gemma? (model choice)

| Gemma | Modality | Used for | Ollama tag |
| ----- | -------- | -------- | ---------- |
| **Gemma 3** (4B/12B/27B) | text **+ image**, 128K ctx | default text analysis + frames | `gemma3` |
| **Gemma 3n** (E2B/E4B) | text **+ image + audio + video** | the video feature (multimodal) | `gemma3n` |

We deliberately moved **off Gemma 2** (which is text-only) to the **Gemma 3
family**: Gemma 3 for the text-analysis path and **Gemma 3n** for video, since it
is natively multimodal and can reason over frames (and audio). Everything is
configurable via env vars, so pointing at **Gemma 4** or a larger size is a
one-line change (see below) — no code edits required.

## Backends (runs anywhere)

The unified client (`humor_genome/gemma_client.py`) auto-selects a backend so
**anyone — including judges — can run it**:

| Backend | How | Needs |
| ------- | --- | ----- |
| `ollama` | `ollama run gemma3` (+ `ollama pull gemma3n` for video) | Ollama + a Gemma model |
| `hf` | `pip install -r requirements-model.txt` | GPU + `google/gemma-3-4b-it` / `gemma-3n-e4b` access |
| `mock` | nothing — always works | ✅ zero setup |

Selection order for `auto`: **Ollama → HuggingFace → offline mock**. The mock is
a deterministic heuristic stand-in (clearly labeled in the UI) so the full
experience is clickable before you wire up real Gemma. The video feature's
**reaction detection runs regardless of backend** (it's audio analysis); only
the frame-based reasoning needs a multimodal Gemma.

> **Video needs `ffmpeg`** on your PATH (`apt-get install ffmpeg` /
> `brew install ffmpeg`) for frame + audio extraction.

## Quickstart

```bash
pip install -r requirements.txt
# for the video feature also install ffmpeg (apt-get install ffmpeg / brew install ffmpeg)

# Recommended: real Gemma via Ollama
ollama run gemma3          # text analysis, in another terminal
ollama pull gemma3n        # (optional) multimodal model for the video feature

streamlit run app.py       # open http://localhost:8501
```

No model handy? It still runs on the offline mock:

```bash
HUMOR_GENOME_BACKEND=mock streamlit run app.py
```

### CLI

```bash
# analyze a bundled example (pretty output)
python -m humor_genome.cli --example dad-scarecrow

# analyze your own joke and punch it up, as JSON
python -m humor_genome.cli "your joke here" --punchup "Tech crowd" --json

# analyze a comedy video clip (+ optional transcript)
python -m humor_genome.cli --video clip.mp4 --transcript clip.srt

# force a backend
python -m humor_genome.cli --backend mock --list-examples
```

## Configuration

Environment variables:

- `HUMOR_GENOME_BACKEND` — `auto` (default) | `ollama` | `hf` | `mock`
- `HUMOR_GENOME_OLLAMA_MODEL` — default `gemma3` (text)
- `HUMOR_GENOME_OLLAMA_VISION_MODEL` — default `gemma3n` (video/images)
- `HUMOR_GENOME_HF_MODEL` — default `google/gemma-3-4b-it`
- `HUMOR_GENOME_HF_VISION_MODEL` — default `google/gemma-3n-e4b`
- `OLLAMA_HOST` — default `http://localhost:11434`

To use **Gemma 4** once you have access, e.g.:

```bash
export HUMOR_GENOME_OLLAMA_MODEL=gemma4
```

## Project layout

```
app.py                     Streamlit UI: text tab + video tab
humor_genome/
  gemma_client.py          Unified Ollama / HuggingFace / mock backend (+ images)
  prompts.py               JSON-schema prompts for analysis, punch-up, video
  engine.py                Orchestration + robust JSON parsing/validation
  genome.py                Dataclasses: GenomeReport, video report, reactions
  mock_brain.py            Deterministic offline stand-in for Gemma
  laughter.py              Audio audience-reaction detection (numpy)
  video.py                 ffmpeg frame/audio extraction + transcript parsing
  data.py                  Loads the curated demo joke set
  cli.py                   Command-line interface
examples/jokes.json        Curated demo jokes (incl. deliberately weak ones)
tests/test_engine.py       Text-analysis tests (mock backend)
tests/test_video.py        Reaction-detection + video tests (mock backend)
WRITEUP.md                 Kaggle writeup draft
DEMO_SCRIPT.md             2-minute demo video script
```

## Tests

```bash
pip install pytest
pytest tests/ -q
```

## License

MIT
