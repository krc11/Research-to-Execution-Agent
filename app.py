"""Research-to-Execution Agent v1 — Streamlit UI.

Three inputs → six-step pipeline → downloadable markdown report.
"""

from __future__ import annotations

import streamlit as st
from pipeline import Pipeline

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
)

# ── Passcode gate ────────────────────────────────────────────────────────────

def _check_passcode() -> bool:
    """Return True once the user has entered the correct passcode."""
    passcode = st.secrets.get("PASSCODE", "")
    if not passcode:
        return True  # no gate configured

    if st.session_state.get("authenticated"):
        return True

    with st.container():
        st.title("🔐 Research Agent")
        entered = st.text_input("Passcode", type="password", key="passcode_input")
        if st.button("Enter"):
            if entered == passcode:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Wrong passcode.")
    return False


if not _check_passcode():
    st.stop()


# ── Sidebar: API key ────────────────────────────────────────────────────────

api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
model = st.secrets.get("MODEL", "claude-sonnet-4-20250514")

if not api_key:
    st.error(
        "Set `ANTHROPIC_API_KEY` in `.streamlit/secrets.toml` or in Streamlit Cloud secrets."
    )
    st.stop()


# ── Main UI ──────────────────────────────────────────────────────────────────

st.title("🔬 Research-to-Execution Agent")
st.caption(
    "Paste a research question, your criteria, and permitted sources.  "
    "The agent will research, verify, score, and write a report with "
    "every claim linked to evidence."
)

col1, col2 = st.columns([3, 2])

with col1:
    question = st.text_area(
        "Research question",
        placeholder="e.g. Which vector database should we adopt for our RAG pipeline?",
        height=80,
    )

    criteria_text = st.text_area(
        "Criteria (paste text or CSV)",
        placeholder=(
            "e.g.\n"
            "Latency, must-have, weight 9: p99 query latency under 50ms\n"
            "Python SDK, must-have, weight 8: first-class Python client\n"
            "Managed hosting, nice-to-have, weight 5: available as SaaS"
        ),
        height=160,
    )

with col2:
    sources_text = st.text_area(
        "Permitted sources (domains and/or URLs, one per line — leave blank for open)",
        placeholder=(
            "e.g.\n"
            "docs.pinecone.io\n"
            "weaviate.io\n"
            "milvus.io\n"
            "https://qdrant.tech"
        ),
        height=160,
    )

    st.info(
        "💡 **Tip:** leave sources blank to allow any domain. "
        "Adding domains restricts web search AND is enforced in code.",
        icon="ℹ️",
    )

# ── Run ──────────────────────────────────────────────────────────────────────

if st.button("🚀 Run Research", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a research question.")
        st.stop()
    if not criteria_text.strip():
        st.warning("Please enter at least one criterion.")
        st.stop()

    # Status container
    status_container = st.empty()
    progress_bar = st.progress(0)
    step_counter = {"n": 0}

    def on_status(msg: str) -> None:
        status_container.markdown(f"**{msg}**")
        if msg.startswith("✅"):
            step_counter["n"] += 1
            progress_bar.progress(min(step_counter["n"] / 6, 1.0))

    pipe = Pipeline(api_key=api_key, model=model, on_status=on_status)

    try:
        report, md = pipe.run(question, criteria_text, sources_text)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.exception(e)
        st.stop()

    progress_bar.progress(1.0)
    status_container.success("✅ Report complete!")

    # ── Display report ───────────────────────────────────────────────────

    st.divider()
    st.markdown(md)

    # ── Download button ──────────────────────────────────────────────────

    st.download_button(
        label="📥 Download report (.md)",
        data=md,
        file_name="research_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # ── Raw JSON (collapsed) ─────────────────────────────────────────────

    with st.expander("🗂️ Raw report JSON"):
        st.json(report.model_dump())
