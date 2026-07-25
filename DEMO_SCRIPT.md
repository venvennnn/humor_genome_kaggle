# 2-Minute Live Demo Script — "Why'd They Laugh?"

Spoken content is ~300 words (≈150 wpm = 2:00). **Bold** = say it. _Italic_ = do it.

---

## ⚠️ Before you present (do this 10 minutes early)

1. Start Ollama and warm the model so it's resident:
   `ollama run gemma3 "hi"` then `/bye`
2. Launch the app: `streamlit run app.py`
3. **Run the clip analysis BEFORE you go on.** Local inference takes minutes —
   never wait for it live. Upload the clip, paste the transcript, hit Analyze,
   and leave the finished result on screen.
4. Have the **Joke tab** pre-loaded with one joke analyzed as a backup.
5. Zoom the browser to ~110% so the timeline is readable from the back.

---

## The script

### 0:00 – 0:15 · The hook
_Start on the finished Clip tab, timeline visible._

> **Every joke is a bet. You set an expectation, then you break it — and a room
> either laughs or it doesn't. That makes comedy one of the only creative fields
> with a real-time scoreboard. So instead of asking Gemma to write jokes, I asked
> it a harder question: why did these people laugh?**

### 0:15 – 0:35 · The laugh timeline
_Point at the green waveform._

> **This is a stand-up crowd-work clip. I don't tell the app where the jokes are —
> it listens. This green wave is the actual audience audio, and every shaded band
> is a detected laugh. That's signal processing, no model needed. Now Gemma has
> ground truth to reason against.**

### 0:35 – 1:05 · Click a beat
_Click a green triangle, then a red one._

> **Each triangle is a comedic beat. Green landed, red died. Watch — I click, and
> it jumps the video to that exact second and Gemma explains the mechanism: what
> expectation the setup created, and how the punchline broke it.**

_Click a red beat._

> **And here's the useful part for a comic: this one got nothing, and Gemma tells
> you why — and gives you a concrete fix.**

### 1:05 – 1:35 · Predicted vs. actual · **the money shot**
_Scroll to the purple diamonds / the matched-bombed-surprise cards._

> **This is the part I'm proudest of. Gemma also reads the transcript blind — no
> audio, no crowd — and predicts where the laughs should be. Those purple diamonds
> are its theory. The green is what actually happened.**
>
> **The mismatches are the insight. A prediction over silence is a joke that
> looked good on paper and bombed in the room. A real laugh it never predicted is
> delivery — a face, a pause, a physical bit the words can't hold. That's the gap
> between text and performance, measured.**

### 1:35 – 1:50 · Genome + fixes
_Scroll to the radar._

> **It also scores the material across six axes — surprise, edge, warmth — with
> audience-fit predictions for rooms you define, and rewrites weak lines on
> request.**

### 1:50 – 2:00 · Close
> **Gemma does all the reasoning here. It runs fully local on Gemma 3. And it
> turns "was that funny?" into something you can actually argue with. Thank you.**

---

## If you're cut to 60 seconds
Keep only: the hook (0:00), the laugh timeline (0:15), and **predicted vs.
actual** (1:05). Drop the beat clicking and the radar.

## Likely judge questions — quick answers

- **"How is Gemma used?"** → Five jobs: structured genome JSON, audience
  simulation, blind laugh prediction, punch-up rewrites, callback attribution.
  No separate classifier — Gemma is the whole reasoning core.
- **"Why Gemma 3?"** → Gemma 2 is text-only and can't see a clip. Gemma 3 is
  multimodal and one `ollama pull` covers text and frames; Gemma 3n adds audio.
- **"How do you detect laughter?"** → Sustained broadband energy bursts above the
  speech baseline (numpy). Not a trained classifier — deliberately, so it runs
  anywhere with zero setup.
- **"What's the hardest part?"** → No ground truth for "funny." Our answer:
  make the model commit to *reasons*, then test its predictions against real
  laughter.
- **"What would you do next?"** → Auto-transcription, a trained laughter
  classifier, and per-beat A/B rewriting tested against future audiences.
