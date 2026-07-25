"""Why'd They Laugh? — Streamlit UI for the Gemma humor genome engine.

Run:  streamlit run app.py

Backend is auto-detected (Ollama -> HuggingFace -> offline mock). Force it with
the sidebar selector or the HUMOR_GENOME_BACKEND env var.
"""

from __future__ import annotations

import os
import tempfile

import plotly.graph_objects as go
import streamlit as st

from humor_genome.data import load_examples
from humor_genome.engine import HumorGenomeEngine, DEFAULT_AUDIENCES
from humor_genome.gemma_client import GemmaConfig
from humor_genome.genome import (
    GENOME_AXES,
    GenomeReport,
    VideoHumorReport,
    SetReport,
)
from humor_genome.video import ffmpeg_available

st.set_page_config(
    page_title="Why'd They Laugh?",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

PURPLE = "#7c3aed"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"

VERDICT_COLOR = {
    "kills": "#16a34a",
    "lands": "#65a30d",
    "polite chuckle": "#ca8a04",
    "bombs": "#dc2626",
}

CLUSTER_PALETTE = ["#7c3aed", "#0891b2", "#db2777", "#ca8a04", "#16a34a", "#dc2626"]


# --------------------------------------------------------------------- styling
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp {
            background:
              radial-gradient(1100px 480px at 12% -12%, #ede9ff 0%, transparent 60%),
              radial-gradient(1000px 460px at 100% -6%, #ffe4f3 0%, transparent 55%),
              #faf9ff;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #faf5ff, #f3e8ff);
            border-right: 1px solid #ece5ff;
        }
        .hg-hero {
            background: linear-gradient(115deg, #4c1d95 0%, #7c3aed 48%, #db2777 100%);
            border-radius: 20px; padding: 26px 32px; color: white; margin-bottom: 14px;
            box-shadow: 0 18px 40px rgba(124,58,237,0.30);
            position: relative; overflow: hidden;
        }
        .hg-hero:after {
            content:""; position:absolute; right:-40px; top:-40px; width:220px; height:220px;
            background: radial-gradient(circle, #ffffff33, transparent 70%); border-radius:50%;
        }
        .hg-hero h1 { color: white; margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing:-0.5px; }
        .hg-hero p { color: #f3e8ff; margin: 8px 0 0 0; font-size: 1.0rem; max-width: 780px; }
        .hg-pill { display:inline-block; background:#ffffff22; border:1px solid #ffffff55;
                   color:white; padding:3px 12px; border-radius:999px; font-size:0.74rem;
                   margin-right:7px; margin-top:10px; font-weight:600; backdrop-filter: blur(4px); }
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            font-size: 1rem; font-weight: 700; border-radius: 10px 10px 0 0; padding: 8px 16px;
        }
        .stTabs [aria-selected="true"] { color: #7c3aed; background: #f3e8ff; }
        div[data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #eee6ff; border-radius: 14px;
            padding: 12px 16px; box-shadow: 0 4px 14px rgba(124,58,237,0.06);
        }
        div[data-testid="stMetricValue"] { font-size: 1.5rem; color:#5b21b6; font-weight:800; }
        .stButton>button {
            border-radius: 10px; font-weight: 700; border: 0;
        }
        .stButton>button[kind="primary"] {
            background: linear-gradient(100deg, #7c3aed, #db2777);
            box-shadow: 0 6px 18px rgba(124,58,237,0.30);
        }
        div[data-testid="stExpander"] { border-radius: 12px; border: 1px solid #eee6ff; }
        .hg-chip { background:#ede9fe; color:#5b21b6; padding:3px 11px; border-radius:999px;
                   margin-right:6px; font-size:0.82em; white-space:nowrap; font-weight:600; }
        .hg-section { font-size:1.05rem; font-weight:800; color:#4c1d95; margin: 4px 0 2px 0; }
        .hg-card { background:#fff; border:1px solid #eee6ff; border-radius:14px; padding:14px 18px;
                   box-shadow:0 4px 14px rgba(124,58,237,0.05); margin-bottom:10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="hg-hero">
          <h1>🎭 Why'd They Laugh?</h1>
          <p>A Gemma-powered humor <b>understanding</b> engine. Drop in a comedy clip to see a
          live laugh timeline, a full humor genome, and the AI's theory of the joke tested
          against the real crowd — laugh by laugh.</p>
          <span class="hg-pill">🎬 Live laugh timeline</span>
          <span class="hg-pill">🔮 Predicted vs. actual</span>
          <span class="hg-pill">🧬 Humor genome</span>
          <span class="hg-pill">🎯 Audience personas</span>
          <span class="hg-pill">🔗 Callback attribution</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_engine(backend: str) -> HumorGenomeEngine:
    config = GemmaConfig(backend=backend) if backend != "auto" else None
    return HumorGenomeEngine(config=config)


def _backend_error_message(engine: HumorGenomeEngine, exc: Exception) -> str:
    return (
        f"The **{engine.describe_backend()}** backend returned an error:\n\n"
        f"```\n{exc}\n```\n"
        "Fixes: make sure the model is pulled (e.g. `ollama pull gemma3`), or "
        "switch the **Gemma backend** in the sidebar to **mock** to try the app "
        "with zero setup."
    )


# ------------------------------------------------------------------ text report
def radar_chart(report: GenomeReport) -> go.Figure:
    axes = [d.name for d in report.dimensions] or GENOME_AXES
    values = [report.dimension(a) for a in axes]
    axes_c = axes + [axes[0]]
    values_c = values + [values[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_c, theta=axes_c, fill="toself",
        line=dict(color=PURPLE), fillcolor="rgba(124,58,237,0.25)", name="genome",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False, margin=dict(l=40, r=40, t=20, b=20), height=360,
    )
    return fig


def render_report(report: GenomeReport) -> None:
    top = st.columns([1, 1])
    with top[0]:
        st.metric("Overall funniness", f"{report.funniness}/10")
        st.markdown(f"#### 💡 {report.one_line_explanation}")
        st.markdown(
            f"**Setup →** {report.setup}\n\n"
            f"**Expectation →** {report.expectation}\n\n"
            f"**Violation →** {report.violation}\n\n"
            f"**Payoff mechanism →** {report.payoff_mechanism}"
        )
        if report.mechanisms:
            chips = " ".join(f"<span class='hg-chip'>{m}</span>" for m in report.mechanisms)
            st.markdown(chips, unsafe_allow_html=True)
    with top[1]:
        st.plotly_chart(radar_chart(report), use_container_width=True)

    st.divider()
    st.markdown(f"##### 🎯 Punchlines detected ({len(report.punchlines)})")
    if report.punchlines:
        for p in report.punchlines:
            kind_color = PURPLE if p.kind == "punchline" else "#0891b2"
            pcols = st.columns([6, 2])
            pcols[0].markdown(
                f"<span style='background:{kind_color};color:white;padding:1px 8px;"
                f"border-radius:6px;font-size:0.75em'>{p.kind}</span> &nbsp;“{p.text}”",
                unsafe_allow_html=True,
            )
            pcols[1].markdown(
                f"<div style='text-align:right'>{p.mechanism or '—'} · <b>{p.strength}/10</b></div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No distinct punchline detected — that itself may be the problem.")

    st.divider()
    cols = st.columns(2)
    with cols[0]:
        st.markdown("##### 🌍 Cultural assumptions")
        for c in report.cultural_assumptions or ["—"]:
            st.markdown(f"- {c}")
        st.markdown("##### ⏱️ Timing")
        st.write(report.timing_notes or "—")
    with cols[1]:
        st.markdown("##### 💥 Failure modes")
        for f in report.failure_modes or ["—"]:
            st.markdown(f"- {f}")

    st.divider()
    st.markdown("##### 🎯 Audience fit")
    for a in report.audiences:
        color = VERDICT_COLOR.get(a.verdict.lower(), "#6b7280")
        bar_cols = st.columns([2, 1, 5])
        bar_cols[0].markdown(f"**{a.audience}**")
        bar_cols[1].markdown(
            f"<span style='color:{color};font-weight:700'>{a.verdict} · {a.score}/10</span>",
            unsafe_allow_html=True,
        )
        bar_cols[2].progress(min(1.0, a.score / 10), text=a.reasoning)

    if report.improvements:
        st.divider()
        if report.needs_work:
            st.markdown("##### 🛠️ How to make it funnier")
            st.caption(f"This one scored {report.funniness}/10 — here are targeted fixes.")
        else:
            st.markdown("##### ✨ Suggestions to sharpen it further")
        for s in report.improvements:
            with st.container(border=True):
                if s.issue:
                    st.markdown(f"**Issue:** {s.issue}")
                st.markdown(f"**Fix:** {s.suggestion}")
                if s.example:
                    st.markdown(f"_e.g._ {s.example}")


# ---------------------------------------------------------- interactive video
def interactive_timeline(report: VideoHumorReport) -> go.Figure:
    """Laugh-intensity waveform + measured reactions + predicted laughs + clickable beats."""
    fig = go.Figure()

    # 1) measured loudness envelope (the "laugh waveform")
    if report.envelope_t:
        fig.add_trace(go.Scatter(
            x=report.envelope_t, y=report.envelope_v, mode="lines",
            line=dict(color=GREEN, width=1), fill="tozeroy",
            fillcolor="rgba(22,163,74,0.18)", name="measured loudness",
            hoverinfo="skip",
        ))

    # 2) measured reaction bursts (shaded)
    for r in report.reactions:
        fig.add_shape(type="rect", x0=r.start_s, x1=r.end_s, y0=0, y1=1.0,
                      fillcolor="rgba(22,163,74,0.25)", line=dict(width=0), layer="below")

    # 3) predicted laughs (Gemma's blind theory) as diamonds on top lane
    if report.predicted_laughs:
        fig.add_trace(go.Scatter(
            x=[p.time_s for p in report.predicted_laughs],
            y=[0.5 + 0.45 * p.expected_intensity for p in report.predicted_laughs],
            mode="markers", marker=dict(symbol="diamond", size=11, color=PURPLE,
                                        line=dict(color="white", width=1)),
            name="predicted laugh",
            hovertext=[f"predicted: {p.quote}" for p in report.predicted_laughs],
            hoverinfo="text",
        ))

    # 4) clickable beat markers on the bottom lane (customdata = beat index)
    if report.beats:
        for b in report.beats:
            color = GREEN if b.landed else RED
            fig.add_shape(type="rect", x0=b.start_s, x1=b.end_s, y0=-0.12, y1=-0.02,
                          fillcolor=color, opacity=0.35, line=dict(width=0), layer="below")
        fig.add_trace(go.Scatter(
            x=[(b.start_s + b.end_s) / 2 for b in report.beats],
            y=[-0.07] * len(report.beats),
            mode="markers",
            marker=dict(symbol="triangle-up", size=15,
                        color=[GREEN if b.landed else RED for b in report.beats],
                        line=dict(color="white", width=1)),
            name="beat (click me)",
            customdata=list(range(len(report.beats))),
            hovertext=[f"{'😂 landed' if b.landed else '🦗 flat'}: {b.moment}" for b in report.beats],
            hoverinfo="text",
        ))

    fig.update_layout(
        height=300, margin=dict(l=30, r=20, t=30, b=30),
        xaxis_title="time (s)", yaxis=dict(range=[-0.16, 1.02], showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        clickmode="event+select",
    )
    return fig


def render_video_report(report: VideoHumorReport, engine=None, audiences=None) -> None:
    m = st.columns(5)
    m[0].metric("Duration", f"{report.duration_s:.0f}s")
    m[1].metric("Laughs detected", len(report.reactions))
    m[2].metric("Laugh coverage", f"{report.laugh_coverage * 100:.0f}%")
    m[3].metric("Biggest laugh @", f"{report.biggest_laugh_s:.1f}s")
    if report.predicted_laughs:
        m[4].metric("Prediction hit-rate", f"{report.prediction_hit_rate * 100:.0f}%")

    for note in report.notes:
        st.info(note)

    if report.overall_summary:
        st.markdown(f"#### 🎬 {report.overall_summary}")

    st.markdown("##### 🎞️ Interactive laugh timeline")
    st.caption(
        "Green wave = measured loudness · shaded green = detected laughs · "
        "purple diamonds = Gemma's *predicted* laughs · triangles = comedic beats "
        "(green landed / red flat). **Click a beat** to jump the video and read why."
    )

    left, right = st.columns([3, 2])
    with left:
        video_path = st.session_state.get("video_path")
        start = int(st.session_state.get("seek_s", 0) or 0)
        if video_path and os.path.exists(video_path):
            st.video(video_path, start_time=start)
        event = st.plotly_chart(
            interactive_timeline(report), use_container_width=True,
            on_select="rerun", key="timeline_chart",
        )
        _handle_timeline_click(event, report)

    with right:
        idx = st.session_state.get("selected_beat")
        if idx is not None and 0 <= idx < len(report.beats):
            b = report.beats[idx]
            icon = "😂" if b.landed else "🦗"
            st.markdown(f"### {icon} Beat @ {b.start_s:.1f}–{b.end_s:.1f}s")
            st.markdown(
                f"**Outcome:** {'LANDED' if b.landed else 'no laugh'} · "
                f"reaction {b.audience_reaction}/10"
            )
            st.markdown(f"**Moment:** {b.moment}")
            if b.mechanism:
                st.markdown(f"**Mechanism:** `{b.mechanism}`")
            st.markdown(f"**Why:** {b.explanation}")
            if b.improvement:
                st.markdown(f"**🛠️ How to improve:** {b.improvement}")
            if st.button("▶️ Jump video to this beat", key="jump_btn"):
                st.session_state["seek_s"] = int(b.start_s)
                st.rerun()
        else:
            st.info("👈 Click a beat triangle on the timeline to inspect it — you'll see the "
                    "joke, whether it landed, why, and how to improve it.")

    _render_prediction_section(report)

    st.divider()
    cols = st.columns(2)
    with cols[0]:
        st.markdown("##### ✅ What worked")
        for w in report.what_worked or ["—"]:
            st.markdown(f"- {w}")
    with cols[1]:
        st.markdown("##### 🪫 What fell flat")
        for w in report.what_fell_flat or ["—"]:
            st.markdown(f"- {w}")

    # All per-beat fixes in one place (no clicking required)
    fixes = [b for b in report.beats if b.improvement]
    if fixes:
        st.divider()
        st.markdown("##### 🛠️ How to improve this clip, beat by beat")
        st.caption("Gemma's concrete fix for each beat — flat beats first.")
        for b in sorted(fixes, key=lambda x: (x.landed, x.start_s)):
            icon = "😂" if b.landed else "🦗"
            with st.container(border=True):
                st.markdown(
                    f"{icon} **{b.start_s:.1f}–{b.end_s:.1f}s** · "
                    f"{'landed' if b.landed else 'no laugh'}"
                    f"{f' · `{b.mechanism}`' if b.mechanism else ''}"
                )
                if b.moment:
                    st.markdown(f"_{b.moment}_")
                st.markdown(f"**Fix:** {b.improvement}")

    # Full humor genome of the clip's material (same as the Joke tab)
    if report.genome is not None:
        st.divider()
        st.markdown("## 🧬 Humor genome of this clip")
        st.caption(
            "The same deep analysis as the Joke tab — radar of the six axes, scores "
            "out of 10, punchlines, audience fit and fixes. Run on the clip's "
            "transcript when you provide one, otherwise on the beats Gemma detected."
        )
        render_report(report.genome)
        if engine is not None:
            st.divider()
            render_punchup(engine, report.genome, audiences or ["General public"], key_prefix="video")

    with st.expander("🔎 Raw model output"):
        st.code(report.raw_model_output or "(none)", language="json")


def _handle_timeline_click(event, report: VideoHumorReport) -> None:
    try:
        points = event.selection["points"] if event and event.selection else []
    except (AttributeError, KeyError, TypeError):
        points = []
    for pt in points:
        cd = pt.get("customdata")
        if isinstance(cd, list):
            cd = cd[0] if cd else None
        if cd is not None:
            idx = int(cd)
            if st.session_state.get("selected_beat") != idx:
                st.session_state["selected_beat"] = idx
                if 0 <= idx < len(report.beats):
                    st.session_state["seek_s"] = int(report.beats[idx].start_s)
                st.rerun()
            break


def _render_prediction_section(report: VideoHumorReport) -> None:
    if not report.predicted_laughs:
        return
    st.divider()
    st.markdown("### 🔮 Predicted vs. actual — the AI's theory, tested against reality")
    if report.prediction_summary:
        st.caption(report.prediction_summary)

    bombed = [g for g in report.laugh_gaps if g.kind == "bombed"]
    surprise = [g for g in report.laugh_gaps if g.kind == "surprise"]
    matched = [g for g in report.laugh_gaps if g.kind == "matched"]

    c = st.columns(3)
    c[0].metric("✅ Matched", len(matched), help="Gemma predicted a laugh and the crowd delivered.")
    c[1].metric("💀 Bombed", len(bombed), help="Predicted funny, but silence — the interesting misses.")
    c[2].metric("🎭 Surprise laughs", len(surprise), help="Real laughs the text didn't predict — delivery / physical comedy.")

    if bombed:
        st.markdown("**💀 Jokes that read funny but bombed:**")
        for g in bombed:
            st.markdown(f"- `@{g.time_s:.1f}s` “{g.quote}” — {g.explanation}")
    if surprise:
        st.markdown("**🎭 Laughs the transcript missed (delivery / physical):**")
        for g in surprise:
            st.markdown(f"- `@{g.time_s:.1f}s` — {g.explanation}")


# ------------------------------------------------------------------- set report
def cluster_scatter(report: SetReport) -> go.Figure:
    fig = go.Figure()
    for cl in report.clusters:
        xs = [report.pca_coords[i][0] for i in cl.member_indices if i < len(report.pca_coords)]
        ys = [report.pca_coords[i][1] for i in cl.member_indices if i < len(report.pca_coords)]
        texts = [report.jokes[i].text[:80] for i in cl.member_indices if i < len(report.jokes)]
        sizes = [8 + report.jokes[i].funniness * 2 for i in cl.member_indices if i < len(report.jokes)]
        color = CLUSTER_PALETTE[cl.label % len(CLUSTER_PALETTE)]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=f"{cl.name} (n={cl.size})",
            marker=dict(size=sizes, color=color, line=dict(color="white", width=1)),
            hovertext=texts, hoverinfo="text",
        ))
    fig.update_layout(
        height=380, margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="style axis 1", yaxis_title="style axis 2",
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
    )
    return fig


def trajectory_heatmap(report: SetReport) -> go.Figure:
    z = [[j.axes.get(a, 0.0) for j in report.jokes] for a in GENOME_AXES]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{j.index}" for j in report.jokes], y=GENOME_AXES,
        colorscale="Purples", zmin=0, zmax=10, colorbar=dict(title="0-10"),
    ))
    fig.update_layout(
        height=300, margin=dict(l=20, r=20, t=20, b=30),
        xaxis_title="bit order →", yaxis_title="genome axis",
    )
    return fig


def callback_arc_chart(report: SetReport) -> go.Figure:
    fig = go.Figure()
    n = len(report.jokes)
    # baseline of bits
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=[0] * n, mode="markers+text",
        marker=dict(size=10, color="#6b7280"),
        text=[str(j.index) for j in report.jokes], textposition="bottom center",
        hovertext=[j.text[:80] for j in report.jokes], hoverinfo="text",
        showlegend=False,
    ))
    for cb in report.callbacks:
        x0, x1 = cb.setup_index, cb.callback_index
        mid = (x0 + x1) / 2
        height = 0.2 + 0.08 * (x1 - x0)
        xs = [x0, mid, x1]
        ys = [0, height, 0]
        # smooth arc via many points (quadratic bezier)
        import numpy as np
        t = np.linspace(0, 1, 30)
        bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * mid + t ** 2 * x1
        by = (1 - t) ** 2 * 0 + 2 * (1 - t) * t * height + t ** 2 * 0
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines",
            line=dict(color=PURPLE, width=1 + cb.yield_value / 2),
            hovertext=f"{cb.setup_index}→{cb.callback_index} · yield {cb.yield_value}/10 · {cb.note}",
            hoverinfo="text", showlegend=False, opacity=0.75,
        ))
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=20, b=30),
        xaxis_title="bit order →", yaxis=dict(showticklabels=False, range=[-0.1, 1.2]),
    )
    return fig


def render_set_report(report: SetReport) -> None:
    for n in report.notes:
        st.info(n)
    if not report.jokes:
        return
    st.markdown(f"#### 🎤 {report.summary}")

    st.markdown("##### 🧬 Style fingerprint (clusters)")
    st.caption("Each dot is a bit, placed by its humor genome and colored by style cluster. Bigger = funnier.")
    st.plotly_chart(cluster_scatter(report), use_container_width=True)
    for cl in report.clusters:
        st.markdown(
            f"- **{cl.name}** · {cl.size} bit(s) · dominant: "
            f"{', '.join(cl.dominant_axes)} · e.g. “{cl.exemplar[:90]}”"
        )

    st.divider()
    st.markdown("##### 📈 Set trajectory")
    st.caption("How the genome shifts across the set — watch warmth/relatability vs. edge/surprise over time.")
    st.plotly_chart(trajectory_heatmap(report), use_container_width=True)

    st.divider()
    st.markdown("##### 🔗 Setup → callback attribution")
    if report.callbacks:
        st.caption("Arcs link a later callback to the earlier setup it pays off; thickness = attributed 'yield'.")
        st.plotly_chart(callback_arc_chart(report), use_container_width=True)
        total_yield = sum(c.yield_value for c in report.callbacks)
        st.markdown(f"**Total callback yield:** {total_yield:.1f} across {len(report.callbacks)} link(s)")
        for cb in report.callbacks:
            st.markdown(
                f"- **bit {cb.setup_index} → bit {cb.callback_index}** "
                f"(yield {cb.yield_value}/10): {cb.note}"
            )
    else:
        st.caption("No callbacks detected in this set.")

    with st.expander("🔬 Per-bit genomes"):
        for j in report.jokes:
            axes_str = " · ".join(f"{a[:4]} {j.axes.get(a, 0):.0f}" for a in GENOME_AXES)
            st.markdown(f"**{j.index}.** ({j.funniness:.1f}/10) {j.text[:110]}  \n`{axes_str}`")


# ------------------------------------------------------------------------- tabs
def text_tab(engine: HumorGenomeEngine, audiences: list) -> None:
    examples = load_examples()
    ex_labels = ["— pick an example —"] + [e["label"] for e in examples]
    picked = st.selectbox("Load an example", ex_labels)
    default_text = ""
    if picked != ex_labels[0]:
        default_text = next(e["joke"] for e in examples if e["label"] == picked)

    joke = st.text_area(
        "Your joke", value=default_text, height=120,
        placeholder="Why did the scarecrow win an award? Because he was outstanding in his field.",
    )

    if st.button("🔬 Analyze the genome", type="primary") and joke.strip():
        with st.spinner("Gemma is dissecting the joke…"):
            try:
                st.session_state["report"] = engine.analyze(joke, audiences=audiences)
            except Exception as exc:
                st.error(_backend_error_message(engine, exc))
                st.session_state.pop("report", None)

    report = st.session_state.get("report")
    if report:
        render_report(report)
        st.divider()
        render_punchup(engine, report, audiences, key_prefix="text")


def render_punchup(engine, report, audiences, key_prefix: str) -> None:
    st.markdown("### ✍️ Punch it up")
    pu_cols = st.columns([2, 1])
    target = pu_cols[0].selectbox(
        "Rewrite to land harder with…", audiences, key=f"punch_target_{key_prefix}"
    )
    if pu_cols[1].button("Rewrite with Gemma", key=f"punch_btn_{key_prefix}"):
        with st.spinner("Rewriting…"):
            try:
                punch = engine.punch_up(report, target)
                st.success(punch.rewrite)
                st.caption(
                    f"**Mechanism changed:** {punch.mechanism_changed}  \n"
                    f"**Why it's better:** {punch.why_better}"
                )
            except Exception as exc:
                st.error(_backend_error_message(engine, exc))


def video_tab(engine: HumorGenomeEngine, audiences=None) -> None:
    st.markdown(
        "Upload a short comedy clip. We detect where the **audience actually laughs** "
        "(from the audio), let you **click through beats on a synced timeline**, run the "
        "**full humor genome** on the material, and pit Gemma's **predicted** laughs against "
        "the real ones."
    )
    if not ffmpeg_available():
        st.error("`ffmpeg` is not installed, so video decoding is disabled here.")

    up = st.file_uploader("Comedy clip", type=["mp4", "mov", "mkv", "webm", "avi", "m4v"])
    transcript = st.text_area(
        "Transcript — paste the clip's words to unlock predicted-vs-actual "
        "(plain text, .srt or .vtt)",
        height=110,
        help="Timestamps (SRT/VTT) let us align predicted + real laughs precisely. "
             "Leave empty and the genome is derived from detected beats instead.",
        placeholder="(empty — this grey text is just an example, not a transcript)\n"
                    "00:00:03,000 --> 00:00:06,000\nSo I tried to assemble the furniture...",
    )
    if not transcript.strip():
        st.caption(
            "ℹ️ No transcript entered — you'll still get the laugh timeline, beats, "
            "fixes and a genome, but **predicted-vs-actual needs the words**."
        )

    if st.button("🎬 Analyze the clip", type="primary") and up is not None:
        suffix = os.path.splitext(up.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(up.getbuffer())
            path = tf.name
        st.session_state["video_path"] = path
        st.session_state["selected_beat"] = None
        st.session_state["seek_s"] = 0
        with st.spinner("Detecting laughs and asking Gemma why they laughed…"):
            try:
                st.session_state["video_report"] = engine.analyze_video(path, transcript=transcript)
            except Exception as exc:
                st.error(_backend_error_message(engine, exc))
                st.session_state.pop("video_report", None)

    vr = st.session_state.get("video_report")
    if vr:
        render_video_report(vr, engine=engine, audiences=audiences)


SAMPLE_SET = """So I bought a treadmill to get in shape. It is now the most expensive coat rack I have ever owned.
My dog watched me assemble it for two hours. He has never respected me less.
I decided to try meditation instead. Turns out my brain has zero interest in silence.
The meditation app now sends me passive-aggressive notifications about my 'streak'.
Remember that treadmill? I finally used it — to dry a poncho during a thunderstorm.
And the dog? He meditates twelve hours a day. We used to call it 'sleeping'."""


def set_tab(engine: HumorGenomeEngine) -> None:
    st.markdown(
        "Paste a **full set / special transcript**. The engine extracts a humor genome "
        "for every bit, **clusters the comedian's style**, charts the set's trajectory, and "
        "maps **which early setups pay off as later callbacks**."
    )
    cola, colb = st.columns([1, 1])
    if cola.button("Load sample set"):
        st.session_state["set_text"] = SAMPLE_SET
    max_bits = colb.slider("Max bits to analyze", 4, 40, 20,
                           help="Each bit is a Gemma call; lower is faster on real models.")

    transcript = st.text_area(
        "Set transcript (plain text, .srt or .vtt)",
        value=st.session_state.get("set_text", ""), height=200,
        placeholder="Paste a stand-up transcript here…",
    )

    if st.button("🎤 Analyze the set", type="primary") and transcript.strip():
        with st.spinner("Building the genome for every bit… (this is many Gemma calls)"):
            try:
                st.session_state["set_report"] = engine.analyze_set(
                    transcript, max_bits=max_bits
                )
            except Exception as exc:
                st.error(_backend_error_message(engine, exc))
                st.session_state.pop("set_report", None)

    sr = st.session_state.get("set_report")
    if sr:
        render_set_report(sr)


def about_tab() -> None:
    st.markdown(
        """
### How Gemma powers this
- **Genome decomposition** — Gemma returns structured JSON for a joke's setup/violation/payoff,
  six scored axes, punchlines, and improvement suggestions.
- **Audience simulation** — Gemma role-plays each audience persona (including ones you define).
- **Blind laugh prediction** — Gemma predicts where laughs *should* land from the transcript
  alone; we overlay that on the *measured* laughter to find bombs and delivery-driven surprises.
- **Style clustering & callbacks** — every bit's genome is clustered (numpy k-means + PCA) and
  Gemma links callbacks to their setups with an attributed "yield".

**Backends:** Ollama → HuggingFace → offline mock (auto). Video needs `ffmpeg`.
The reaction detection is pure audio analysis and runs on any backend.
        """
    )


def sidebar_controls():
    with st.sidebar:
        st.header("⚙️ Settings")
        backend = st.selectbox(
            "Gemma backend", ["auto", "ollama", "hf", "mock"],
            help="auto tries Ollama, then HuggingFace, then an offline mock.",
        )
        engine = get_engine(backend)
        st.info(f"Active backend: **{engine.describe_backend()}**")
        if engine.backend == "mock":
            st.warning(
                "Offline **mock** brain — heuristic placeholders. "
                "`ollama pull gemma3` for real Gemma."
            )

        st.markdown("---")
        st.markdown("### 🎯 Audience personas")
        st.caption("Verdicts in the Joke tab are predicted against these rooms.")
        audiences = []
        for a in DEFAULT_AUDIENCES:
            if st.checkbox(a, value=True, key=f"aud_{a}"):
                audiences.append(a)

        extra = ""
        with st.expander("➕ Define a custom room"):
            size = st.text_input("Size / setting", placeholder="40-person corporate offsite")
            vibe = st.text_input("Make-up / notes", placeholder="mixed seniority, HR present")
            if st.button("Add persona") and (size or vibe):
                persona = ", ".join(x for x in [size, vibe] if x)
                st.session_state.setdefault("custom_personas", [])
                if persona not in st.session_state["custom_personas"]:
                    st.session_state["custom_personas"].append(persona)
            extra = st.text_area(
                "…or one persona per line", height=80,
                placeholder="Gen-Z TikTok crowd\nRetirement home social hour",
            )

        custom = list(st.session_state.get("custom_personas", []))
        if custom:
            st.markdown("**Saved custom rooms:**")
            for i, p in enumerate(custom):
                cc = st.columns([5, 1])
                cc[0].markdown(f"<span class='hg-chip'>{p}</span>", unsafe_allow_html=True)
                if cc[1].button("✕", key=f"rm_{i}"):
                    st.session_state["custom_personas"].remove(p)
                    st.rerun()

        # merge free-text lines typed in the expander
        line_personas = [ln.strip() for ln in (extra or "").splitlines() if ln.strip()]
        audiences = audiences + custom + line_personas
        audiences = list(dict.fromkeys(audiences)) or ["General public"]
    return engine, audiences


def main() -> None:
    inject_css()
    hero()
    engine, audiences = sidebar_controls()

    tabs = st.tabs(["🎬 Clip (video)", "🗣️ Joke", "🎤 Set / Special", "ℹ️ About"])
    with tabs[0]:
        video_tab(engine, audiences)
    with tabs[1]:
        text_tab(engine, audiences)
    with tabs[2]:
        set_tab(engine)
    with tabs[3]:
        about_tab()


if __name__ == "__main__":
    main()
