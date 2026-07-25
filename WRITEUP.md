# Why'd They Laugh?

### Subtitle: A Gemma-powered engine that decomposes a joke's "humor genome" — then tests the AI's theory of the joke against a real audience's laughter

**Track:** Humor Understanding (Track 2)

---

## The problem: humor has no answer key

Almost every AI-and-comedy project asks a model to *write* jokes. We think the
harder and more revealing question is the opposite one: **why did people laugh —
and why, sometimes, did they not?**

Humor understanding is uniquely difficult because there is no ground-truth label.
The same line kills in a comedy club and dies at a corporate all-hands. Funniness
isn't a property of a sentence; it's a relationship between a sentence, a
listener, a moment, and a room. So instead of asking a model the unanswerable
question "is this funny?", we ask it to make its reasoning **explicit,
structured, and falsifiable**: what expectation does the setup create, how does
the payoff violate it, what shared knowledge is required, and who does that
knowledge exclude?

That reframing is the heart of *Why'd They Laugh?*. It converts an opaque gut
reaction into an artifact a human can read, argue with, and act on — which is
exactly the human-AI collaboration this event is about.

## What we built

A Streamlit application (plus a CLI and a small Python library) with three modes.

### 1. The humor genome of a joke

Paste any joke and Gemma returns a structured decomposition:

- **Setup → Expectation → Violation → Payoff** — the anatomy of the laugh.
- **Comedic mechanisms** drawn from a fixed vocabulary (misdirection, wordplay,
  callback, act-out, taboo, absurdism, rule-of-three, self-deprecation…).
- **Punchline detection** — *every* laugh line, not just the main one: the payoff
  plus any tags/toppers, callbacks or act-outs, each with its mechanism and a
  strength score.
- **Six scored genome axes** — surprise, specificity, cleverness, relatability,
  edge, warmth — rendered as a radar "fingerprint" out of 10.
- **Cultural assumptions**, **timing notes**, and **failure modes**.
- **Audience fit** — a verdict (*kills / lands / polite chuckle / bombs*) with
  reasoning for each room, including **custom personas** you define yourself
  ("40-person corporate offsite, mixed seniority, HR present").
- **Improvement suggestions** — for weak jokes, concrete fixes targeted at the
  two lowest axes, with example rewrites; strong jokes get sharpening tips.
- **Punch-up** — a one-click rewrite aimed at a chosen audience that reports
  *which mechanism it changed and why*.

### 2. Why did the room laugh? (video)

Upload a clip and the analysis is grounded in the **actual audience reaction**:

1. **Reaction detection.** We extract the audio and detect where the crowd
   laughs or applauds — sustained, broadband energy bursts sitting above the
   speech baseline — producing a laugh-intensity timeline. This is pure numpy
   signal processing, so it needs no model at all.
2. **Frames → multimodal Gemma.** Evenly sampled frames go to a multimodal
   Gemma for physical and visual comedy the words can't carry.
3. **Beat-by-beat reasoning.** Gemma splits the clip into comedic beats and,
   cross-referencing the *measured* laughs, explains why each beat landed or fell
   flat — timing, an unclear setup, the wrong crowd, a reference that didn't
   connect — plus a concrete fix for each.
4. **Interactive laugh timeline.** A synced video player sits over the laugh
   waveform, with every beat marked and colored by outcome. **Click a beat** to
   jump the video to that moment and read Gemma's verdict and fix in a side panel.
5. **Predicted vs. actual laughter — our strongest idea.** Gemma also predicts
   where laughs *should* land from the transcript **alone**, blind to the crowd.
   We overlay that prediction on the measured laughter, and the mismatches are
   where it gets interesting: a predicted laugh over **silence** is a joke that
   bombed; a **real laugh the model never predicted** exposes delivery, timing or
   physical comedy that text simply cannot encode. This turns "AI describes a
   video" into **"the AI's theory of the joke, tested against reality."**

### 3. Style fingerprint and callback attribution (full set)

Paste an entire special and the tool becomes exploratory data analysis of a
comedian's craft. Every bit's genome is extracted, and the bits are **clustered**
(numpy k-means + PCA) into distinct comedic styles. A **trajectory heatmap** shows
how the genome shifts across the set — many comics open warm and relatable to buy
the room's trust, then pivot to edge and surprise late. Finally, Gemma builds a
**setup → callback attribution graph**, linking late payoffs back to the early
premises that seeded them with an attributed "yield" — treating a set like a
revenue-attribution model, where a minute-45 punchline owes part of its laugh to
a minute-5 premise.

## How Gemma was used

Gemma is the entire reasoning core. There is no separate classifier or rule engine
doing the real work.

- **Structured decomposition.** One instruction-tuned prompt asks Gemma to return
  strict JSON matching our schema. We constrain it to a fixed axis set and
  mechanism vocabulary so results are consistent, comparable and chartable across
  jokes rather than free-form prose.
- **Audience simulation.** In the same call, Gemma role-plays each audience —
  preset or user-defined — and returns a verdict, score and reasoning.
- **Blind laugh prediction.** A deliberately reaction-free prompt asks Gemma where
  laughs should land from the words alone. We then compare that to the measured
  timeline to classify every moment as *matched*, *bombed*, or *surprise*.
- **Targeted rewriting.** Punch-up feeds the joke's two weakest axes back to Gemma
  with a target audience, closing the loop from *understanding* to *action*.
- **Callback attribution.** Given the ordered bits of a set, Gemma identifies
  genuine callbacks and estimates how much of the later laugh the earlier setup
  earned.

**Why Gemma 3 / 3n rather than Gemma 2.** Gemma 2 is text-only, so it cannot see a
comedy clip. We use **Gemma 3** (multimodal, 128K context) as the default for both
text analysis and video frames — a single `ollama pull gemma3` covers everything —
with **Gemma 3n** (natively text + image + audio + video) available as an upgrade.
Every model name is an environment variable, so moving to Gemma 4 or a larger size
is a one-line change with no code edits.

## Architecture

```
Streamlit UI ─┐
CLI          ─┼─► HumorGenomeEngine ─► GemmaClient ─► Ollama | HuggingFace | mock
Library      ─┘        │                (text + image input)
                       ├─ prompts.py    strict-JSON prompts (genome / video / prediction / callbacks)
                       ├─ genome.py     typed schema: reports, axes, beats, clusters
                       ├─ video.py      ffmpeg: frame + audio extraction, SRT/VTT parsing
                       ├─ laughter.py   numpy: reaction detection + loudness envelope
                       └─ clustering.py numpy k-means + PCA style fingerprints

Video: clip ─► ffmpeg ─► audio ─► laughter.py ─► measured laugh timeline ─┐
                      └► frames ──────────────────────────────────────────┼─► Gemma ─► beats
              transcript ─► blind prediction ─► predicted laughs ─────────┘      + genome
```

Three design principles carried the project:

**Schema-first.** All output is typed dataclasses, so the UI, CLI and tests depend
on a stable contract rather than on whichever backend produced the text.

**It must run for the judges.** The client auto-selects Ollama → HuggingFace →
a deterministic **offline mock** that needs no weights and is clearly labeled in
the UI. Reaction detection works on any backend because it's signal processing.
The whole app is clickable in seconds, then upgrades to real Gemma with one
command.

**Degrade, never crash.** If the vision model rejects images, we fall back to
text-only and say so. If no transcript is supplied, the genome is derived from the
detected beats instead. If a call fails, the measured laugh timeline still stands.

## Key challenges

**Models emit invalid JSON, and it fails silently.** Our hardest bug: real Gemma
output used typographic curly quotes (`humor.”`) and, on longer clips, was
**truncated mid-string** by the token limit. Both produced an empty report. We
raised the token budget and wrote a tolerant parser that normalizes smart quotes
and *salvages truncated JSON* — closing unterminated strings and brackets, and
walking back to the last complete element. On the real 49-second clip that broke
us, it now recovers all seven beats using only the standard library.

**No ground truth.** We chose explainability over a single score: the axes,
per-audience verdicts and named mechanisms force the model to commit to reasons,
which are far more useful — and debuggable — than one funniness number.

**Consistency.** Free-form humor analysis drifts, so we constrained Gemma to fixed
axes, a mechanism vocabulary, and strict JSON.

## What we learned

Making a model **name the violated expectation** is a surprisingly strong
diagnostic: bits that score low almost always have a vague or missing
"expectation," which matches comedians' intuition about weak setups.

**Audience is a first-class variable, not a footnote.** Forcing per-room
predictions revealed how much "funniness" is really *fit* — and Gemma's reasoning
about cultural assumptions was consistently more insightful than its scores.

Most of all: **for video, the laugh is the label.** Live audience laughter is one
of the few places where humor genuinely has ground truth. Detecting it and handing
it to Gemma lets the model explain an observed reaction instead of guessing in a
vacuum — and comparing its blind prediction against that reality is where the
project stopped describing comedy and started measuring its own understanding
of it.

## Links

- **Code repository:** _(add GitHub link)_
- **Demo video:** _(add link)_

## Try it in 30 seconds

```bash
pip install -r requirements.txt                    # + ffmpeg for the video tab
HUMOR_GENOME_BACKEND=mock streamlit run app.py     # zero-setup demo
ollama pull gemma3 && streamlit run app.py         # real Gemma
```
