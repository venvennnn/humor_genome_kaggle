# Why'd They Laugh?

### Subtitle: A Gemma-powered engine that decomposes a joke's "humor genome" and predicts how it lands with different audiences

**Track:** Humor Understanding (Track 2)

---

## What we built

*Why'd They Laugh?* is a humor **understanding** engine. Instead of generating
jokes, it explains **why a joke works — or why it bombs — and for whom.** It does
this for both **text jokes** and **comedy video clips**.

### Text: the humor genome

You paste any joke, and Gemma returns a structured decomposition we call its
**humor genome**:

- The anatomy of the laugh: **Setup → Expectation → Violation → Payoff**.
- The **comedic mechanisms** at play (misdirection, wordplay, callback, taboo,
  absurdism, self-deprecation, rule-of-three, …).
- **Every punchline**, not just the main one: the payoff line plus any
  tags/toppers, callbacks, or act-outs, each labeled with its mechanism and a
  strength score.
- **Actionable improvement suggestions**: when a joke is weak, targeted fixes
  aimed at its lowest genome axes (with example rewrites); for strong jokes,
  ways to sharpen it further.
- Six scored **genome axes** — surprise, specificity, cleverness, relatability,
  edge, warmth — visualized as a radar "fingerprint."
- The **cultural assumptions** a listener must share to get it.
- **Timing notes** and **failure modes**.
- **Audience fit**: a predicted verdict (*kills / lands / polite chuckle /
  bombs*) with reasoning for each of several audience segments.
- A one-click **punch-up** that rewrites the joke to land harder with a chosen
  audience, and reports which mechanism it changed and why.

### Video: why did the room laugh?

Upload a short comedy clip and the engine answers **"why did they laugh — or why
didn't they?"**, grounded in the *actual audience reaction*:

1. **Reaction detection.** We extract the audio and detect where the crowd
   laughs/applauds — sustained, broadband energy bursts above the speech
   baseline — producing a reaction timeline (numpy, no model needed).
2. **Frames → multimodal Gemma.** Evenly sampled frames are sent to a multimodal
   Gemma (**Gemma 3 / 3n**) for physical and visual comedy.
3. **Beat-by-beat reasoning.** Gemma splits the clip into comedic beats and,
   cross-referencing the *measured* laughs, explains **why each beat landed or
   fell flat** — timing, unclear setup, wrong audience, or a cultural reference
   that didn't connect. An optional transcript (plain/`.srt`/`.vtt`) sharpens the
   alignment of jokes to laughs.

It ships as a Streamlit app (text + video tabs), a CLI, and a small Python
library, with a curated example set (including deliberately unfunny jokes to show
the engine scoring things *low* and explaining why).

## Why humor *understanding*

Humor generation is crowded; humor *understanding* is the harder, more revealing
problem. There's no ground-truth label — the same line kills in a comedy club
and dies in a corporate all-hands. So rather than ask a model the unanswerable
"is this funny?", we ask it to make its reasoning **explicit and inspectable**:
what expectation is set up, how the payoff violates it, what shared knowledge is
required, and which audiences that knowledge excludes.

That reframing is the core insight of the project. It converts an opaque gut
reaction into a structured artifact a human can read, disagree with, and act on
— which is exactly the kind of human-AI collaboration the event is about. The
same engine is useful to stand-up writers tuning a set, teachers explaining why
a bit works, and localizers deciding whether a joke will survive translation.

## How Gemma was used

Gemma is the entire reasoning core; there is no separate classifier or rules
engine doing the real work.

1. **Structured decomposition.** A single instruction-tuned prompt asks Gemma to
   return strict JSON matching our genome schema. We constrain it to a fixed set
   of axes and a known vocabulary of comedic mechanisms so results are
   consistent and comparable across jokes rather than free-form prose.
2. **Audience simulation.** In the same call, Gemma role-plays each audience
   segment and predicts a verdict + score + reasoning — effectively simulating
   different rooms.
3. **Targeted rewriting (punch-up).** We identify the joke's two *weakest*
   genome axes and feed them back to Gemma with a target audience, asking for a
   rewrite plus an explanation of the mechanism it changed. This closes the loop
   from *understanding* to *actionable improvement*.

For video, Gemma acts as a multimodal reasoner: it receives sampled frames plus
the measured laugh timeline and produces the beat-by-beat verdicts. Crucially, we
feed the model the *detected* reactions so its "landed / fell flat" judgments are
grounded in real audience data rather than a guess about what's funny.

**Why Gemma 3 / 3n (not Gemma 2).** Gemma 2 is text-only, so it can't see a
comedy clip. We use the **Gemma 3 family**: **Gemma 3** (4B/12B/27B) for text
analysis and frame understanding, and **Gemma 3n** — natively multimodal over
text, image, audio, and video — for the video feature. Model names are fully
configurable via environment variables, so upgrading to **Gemma 4** or a larger
size is a one-line change with no code edits.

We designed for portability so judges can actually run it. A unified client
auto-selects a backend: **Ollama** (`gemma3` / `gemma3n`), **HuggingFace
transformers** (`google/gemma-3-4b-it` / `gemma-3n-e4b`), or a deterministic
**offline mock** that requires no weights and is clearly labeled in the UI. The
video reaction-detection stage runs on *any* backend because it's pure audio
analysis; only the frame reasoning needs a multimodal Gemma. This means the full
experience is clickable in seconds, then upgrades to real Gemma with one command.

## Technical approach & architecture

```
Streamlit UI  ─┐
CLI           ─┼─►  HumorGenomeEngine  ─►  GemmaClient ─► Ollama | HF | mock
Library API   ─┘         │                    (text + image input)
                         ├─ prompts.py   (JSON-schema prompts: text / punch-up / video)
                         ├─ genome.py    (typed schema: report, axes, audiences, video beats)
                         ├─ video.py     (ffmpeg: frames + audio, SRT/VTT parsing)
                         ├─ laughter.py  (numpy: audience-reaction detection)
                         └─ robust JSON extraction + validation/clamping

Video path:  clip ─► ffmpeg ─► [audio → laughter.py → reaction timeline]
                              └► [frames] ─┐
                    transcript ────────────┴─► Gemma (multimodal) ─► beat verdicts
```

- **Schema-first design** (`genome.py`): the UI, CLI, and tests all depend on
  stable dataclasses, decoupled from whichever backend produced them.
- **Robust parsing** (`engine.py`): instruction-tuned models sometimes wrap JSON
  in code fences or prose, so we extract the outermost object, strip fences,
  repair trailing commas, clamp numeric scores to `[0, 10]`, and degrade
  gracefully to a minimal report instead of crashing.
- **Backend abstraction** (`gemma_client.py`): one `generate()` interface over
  three backends with automatic fallback.

## Key challenges & design decisions

- **No ground truth.** Our answer was to prioritize *explainability over a single
  score*. The genome axes and per-audience verdicts make the model commit to
  reasons, which are far more valuable (and debuggable) than a lone funniness
  number.
- **Consistency of model output.** Free-form humor analysis drifts. Constraining
  Gemma to a fixed axis set + mechanism vocabulary + strict JSON made outputs
  comparable and chartable.
- **"It must run for the judges."** Rather than assume model access, we built the
  three-tier backend with a zero-setup mock, so the prototype is always
  demonstrable and the real Gemma path is one command away.
- **Understanding → action.** Punch-up uses the analysis (weakest axes) to guide
  the rewrite, so the tool doesn't just critique — it improves.

## What we learned about humor and intelligence

- Making a model **name the violated expectation** is a surprisingly strong lens:
  jokes that score low almost always have a vague or missing "expectation," which
  matches comedic intuition about weak setups.
- **Audience is a first-class variable, not a footnote.** Forcing per-audience
  predictions exposed how much "funniness" is really *fit* — the model's
  reasoning about cultural assumptions is often more insightful than its scores.
- Structured decomposition turns an LLM's fuzzy comedic sense into something a
  human can collaborate with — the real goal of the hackathon.
- **For video, the laugh is the label.** Real audience laughter is one of the few
  places humor *does* have ground truth. Detecting it from audio and feeding it to
  Gemma lets the model explain observed reactions instead of predicting funniness
  in a vacuum — a much more grounded, useful form of humor understanding.

## Links

- **Code repository:** _(add GitHub link here)_
- **Demo video:** _(add link here)_
- **Live demo / notebook:** _(optional)_

## Try it in 30 seconds

```bash
pip install -r requirements.txt                    # + ffmpeg for the video tab
HUMOR_GENOME_BACKEND=mock streamlit run app.py     # zero-setup demo
# or, for real Gemma:
ollama run gemma3 && ollama pull gemma3n && streamlit run app.py
```
