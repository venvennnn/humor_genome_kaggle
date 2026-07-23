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

The engine (`humor_genome/engine.py`) handles robust JSON extraction and clamps
values, so the UI stays stable even when a model wraps its output in prose.

## Backends (runs anywhere)

The unified client (`humor_genome/gemma_client.py`) auto-selects a backend so
**anyone — including judges — can run it**:

| Backend | How | Needs |
| ------- | --- | ----- |
| `ollama` | `ollama run gemma2` then run the app | Ollama + a Gemma model |
| `hf` | `pip install -r requirements-model.txt` | GPU + `google/gemma-2-2b-it` access |
| `mock` | nothing — always works | ✅ zero setup |

Selection order for `auto`: **Ollama → HuggingFace → offline mock**. The mock is
a deterministic heuristic stand-in (clearly labeled in the UI) so the full
experience is clickable before you wire up real Gemma.

## Quickstart

```bash
pip install -r requirements.txt

# Recommended: real Gemma via Ollama
ollama run gemma2          # in another terminal

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

# force a backend
python -m humor_genome.cli --backend mock --list-examples
```

## Configuration

Environment variables:

- `HUMOR_GENOME_BACKEND` — `auto` (default) | `ollama` | `hf` | `mock`
- `HUMOR_GENOME_OLLAMA_MODEL` — default `gemma2`
- `HUMOR_GENOME_HF_MODEL` — default `google/gemma-2-2b-it`
- `OLLAMA_HOST` — default `http://localhost:11434`

## Project layout

```
app.py                     Streamlit UI (radar chart, genome cards, punch-up)
humor_genome/
  gemma_client.py          Unified Ollama / HuggingFace / mock backend
  prompts.py               JSON-schema prompts for analysis + punch-up
  engine.py                Orchestration + robust JSON parsing/validation
  genome.py                Dataclasses: GenomeReport, dimensions, audiences
  mock_brain.py            Deterministic offline stand-in for Gemma
  data.py                  Loads the curated demo joke set
  cli.py                   Command-line interface
examples/jokes.json        Curated demo jokes (incl. deliberately weak ones)
tests/test_engine.py       Tests (run on the mock backend)
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
