"""Why'd They Laugh? — Streamlit UI for the Gemma humor genome engine.

Run:  streamlit run app.py

Backend is auto-detected (Ollama -> HuggingFace -> offline mock). Force it with
the sidebar selector or the HUMOR_GENOME_BACKEND env var.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from humor_genome.data import load_examples
from humor_genome.engine import HumorGenomeEngine, DEFAULT_AUDIENCES
from humor_genome.gemma_client import GemmaConfig
from humor_genome.genome import GENOME_AXES, GenomeReport

st.set_page_config(
    page_title="Why'd They Laugh?",
    page_icon="🎭",
    layout="wide",
)

VERDICT_COLOR = {
    "kills": "#16a34a",
    "lands": "#65a30d",
    "polite chuckle": "#ca8a04",
    "bombs": "#dc2626",
}


@st.cache_resource(show_spinner=False)
def get_engine(backend: str) -> HumorGenomeEngine:
    config = GemmaConfig(backend=backend) if backend != "auto" else None
    return HumorGenomeEngine(config=config)


def radar_chart(report: GenomeReport) -> go.Figure:
    axes = [d.name for d in report.dimensions] or GENOME_AXES
    values = [report.dimension(a) for a in axes]
    # close the loop
    axes_c = axes + [axes[0]]
    values_c = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_c,
            theta=axes_c,
            fill="toself",
            line=dict(color="#7c3aed"),
            fillcolor="rgba(124,58,237,0.25)",
            name="genome",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=360,
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
            chips = " ".join(
                f"<span style='background:#ede9fe;color:#5b21b6;padding:2px 10px;"
                f"border-radius:999px;margin-right:6px;font-size:0.85em'>{m}</span>"
                for m in report.mechanisms
            )
            st.markdown(chips, unsafe_allow_html=True)
    with top[1]:
        st.plotly_chart(radar_chart(report), use_container_width=True)

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


def main() -> None:
    st.title("🎭 Why'd They Laugh?")
    st.caption(
        "A Gemma-powered humor understanding engine. Paste a joke and see its "
        "**humor genome** — the structure, surprise, culture, and audience fit "
        "that decide whether people laugh."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        backend = st.selectbox(
            "Gemma backend",
            ["auto", "ollama", "hf", "mock"],
            help="auto tries Ollama, then HuggingFace, then an offline mock so "
            "the app always runs.",
        )
        engine = get_engine(backend)
        st.info(f"Active backend: **{engine.describe_backend()}**")
        if engine.backend == "mock":
            st.warning(
                "Running on the offline **mock** brain — results are heuristic "
                "placeholders. Start `ollama run gemma2` or install "
                "`requirements-model.txt` to use real Gemma."
            )
        st.markdown("---")
        st.markdown("**Target audiences**")
        audiences = []
        for a in DEFAULT_AUDIENCES:
            if st.checkbox(a, value=True, key=f"aud_{a}"):
                audiences.append(a)
        audiences = audiences or ["General public"]

    examples = load_examples()
    ex_labels = ["— pick an example —"] + [e["label"] for e in examples]
    picked = st.selectbox("Load an example", ex_labels)

    default_text = ""
    if picked != ex_labels[0]:
        default_text = next(e["joke"] for e in examples if e["label"] == picked)

    joke = st.text_area(
        "Your joke",
        value=default_text,
        height=120,
        placeholder="Why did the scarecrow win an award? Because he was outstanding in his field.",
    )

    analyze = st.button("🔬 Analyze the genome", type="primary")

    if analyze and joke.strip():
        with st.spinner("Gemma is dissecting the joke…"):
            report = engine.analyze(joke, audiences=audiences)
        st.session_state["report"] = report

    report = st.session_state.get("report")
    if report:
        render_report(report)

        st.divider()
        st.markdown("### ✍️ Punch it up")
        pu_cols = st.columns([2, 1])
        target = pu_cols[0].selectbox(
            "Rewrite to land harder with…", audiences, key="punch_target"
        )
        if pu_cols[1].button("Rewrite with Gemma"):
            with st.spinner("Rewriting…"):
                punch = engine.punch_up(report, target)
            st.success(punch.rewrite)
            st.caption(
                f"**Mechanism changed:** {punch.mechanism_changed}  \n"
                f"**Why it's better:** {punch.why_better}"
            )

        with st.expander("🔎 Raw model output"):
            st.code(report.raw_model_output or "(none)", language="json")


if __name__ == "__main__":
    main()
