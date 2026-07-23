# Demo video script — "Why'd They Laugh?" (target: under 2:00)

Goal: show (1) the humor challenge, (2) the project in action, (3) how Gemma
contributes. Screen-record the Streamlit app with real Gemma via Ollama if
possible (falls back to mock if not).

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

### Recording tips
- Pre-load Ollama (`ollama run gemma2`) before recording for real outputs.
- Keep the sidebar visible once to show the **Active backend** badge (proves
  real Gemma is in the loop).
- If short on time, the two must-show beats are **Audience fit** and **Punch-up**.
