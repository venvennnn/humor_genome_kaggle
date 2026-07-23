# Demo video script — "Why'd They Laugh?" (target: under 2:00)

Goal: show (1) the humor challenge, (2) the project in action, (3) how Gemma
contributes. Screen-record the Streamlit app with real Gemma via Ollama if
possible (`ollama run gemma3`, `ollama pull gemma3n`; falls back to mock if not).

There are two things to show — the **text genome** and the **video "why did they
laugh"** analysis. Two suggested cuts below: (A) balanced, (B) video-forward.
Pick based on which demo looks best on the day.

---

## 0:00–0:15 — The hook / the challenge
> "Every joke is a bet: set an expectation, then break it. But the same joke
> that kills at a comedy club dies in a corporate all-hands. So instead of
> asking AI to *write* jokes, we asked Gemma the harder question — *why* do
> people laugh, and *who* will?"

Show the app title screen: **🎭 Why'd They Laugh?**

## 0:15–0:35 — Analyze a joke
- Pick the example **"Tech in-joke"** (dark mode joke).
- Click **Analyze the genome**.
- Talk over the spinner: "Gemma decomposes it into a *humor genome*."

## 0:35–1:05 — Walk the genome
- Point at **Setup → Expectation → Violation → Payoff**.
  > "Gemma names the expectation it sets and exactly how the punchline breaks it."
- Point at the **radar chart** (six axes).
  > "Six axes — surprise, cleverness, edge, warmth — a fingerprint of the joke."
- Point at **mechanisms** chips and **cultural assumptions**.

## 1:05–1:30 — Audience fit (the key insight)
- Scroll to **Audience fit**.
  > "Here's the payoff: Gemma predicts the room. This lands with the tech crowd
  > and gets a polite chuckle at a corporate all-hands — and it tells you *why*."
- (Optional) quickly analyze the **"Deliberately weak"** pizza joke to show it
  scoring low and explaining the flat setup. Great contrast beat.

## 1:30–1:55 — Punch it up (understanding → action)
- In **Punch it up**, choose an audience and click **Rewrite with Gemma**.
  > "Because it knows the joke's weakest axes, Gemma rewrites it to land harder —
  > and tells you which mechanism it changed."
- Show the new line + "mechanism changed / why it's better."

## 1:55–2:00 — Close
> "Why'd They Laugh? turns Gemma's comedic intuition into something a writer can
> actually collaborate with. Runs on Gemma via Ollama, HuggingFace, or a
> zero-setup demo mode. Thanks!"

---

## Cut B — video-forward (recommended if you have a good clip)

### 0:00–0:15 — Hook
> "The one place humor has a real answer key is a live crowd — either they laugh
> or they don't. So we asked Gemma to watch a comedy clip and explain the laughs."

### 0:15–0:30 — Text genome (fast)
- Analyze one joke; point at the radar + **Audience fit**. Keep it to ~15s as
  proof the engine understands structure.

### 0:30–1:00 — Upload a clip
- Switch to the **🎬 Video clip** tab, upload a short stand-up/sitcom clip.
- Talk over the spinner: "We pull the audio and detect exactly where the crowd
  laughs, then send frames to multimodal **Gemma 3n**."

### 1:00–1:35 — The reaction timeline + beats
- Point at the **reaction timeline**: green = measured laughs, triangles = beats
  (green landed / red flat).
- Open one **LANDED** beat: "Gemma explains the mechanism and why it hit."
- Open one **no-laugh** beat: "And here it tells us why the room stayed quiet."

### 1:35–1:55 — What worked / fell flat
- Show the summary + "what worked / what fell flat" columns.
> "Because it's grounded in real laughter, this is humor *understanding* with an
> answer key."

### 1:55–2:00 — Close
> "Text or video, Why'd They Laugh? uses Gemma to explain the funny. Thanks!"

---

### Recording tips
- Pre-load Ollama (`ollama run gemma3`, `ollama pull gemma3n`) before recording
  for real outputs; the sidebar **Active backend** badge proves Gemma is in the loop.
- Video needs `ffmpeg` installed. Use a clip with clear, audible audience laughter.
- If short on time, the must-show beats are the **reaction timeline** and one
  **LANDED** + one **no-laugh** beat explanation.
- No clip handy? The reaction detection + mock still render the full UI.
