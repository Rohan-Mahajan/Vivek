"""
app.py
------
Streamlit UI only. All logic lives in recommender.py.

Run with:
    streamlit run app.py

Keep this terminal open — the server runs for as long as the app is up.
Python logs (from recommender.py) appear in the TERMINAL where you
ran this command — not in the browser. Keep your terminal visible.

NOTE on state (fixed): the text area widget's key IS "use_case" now.
We no longer keep a separate mirror key, which caused the example
buttons to update invisible state while the box looked unchanged.
"""

import streamlit as st
from recommender import get_recommendation, load_models
from prompts import EXAMPLE_PROMPTS

# ── Page config — must be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="LLM Picker",
    page_icon="🧠",
    layout="centered",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Best pick card */
    .best-card {
        background: #0f172a;
        border: 1px solid #4f46e5;
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0 20px 0;
    }
    .best-label {
        display: inline-block;
        background: #4f46e5;
        color: white;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.72em;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }
    .model-name {
        font-size: 1.5em;
        font-weight: 700;
        color: #f9fafb;
        margin: 4px 0 2px 0;
    }
    .provider-name {
        color: #9ca3af;
        font-size: 0.88em;
        margin-bottom: 14px;
    }
    .reason-text {
        color: #d1d5db;
        line-height: 1.65;
        margin-bottom: 16px;
    }
    /* Meta pills row */
    .meta-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        margin: 14px 0;
    }
    .meta-cell {
        background: #1e293b;
        border-radius: 8px;
        padding: 10px 14px;
    }
    .meta-cell-label {
        font-size: 0.67em;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 4px;
    }
    .meta-cell-value {
        font-size: 0.92em;
        font-weight: 600;
        color: #e2e8f0;
    }
    /* Strengths */
    .strengths-list {
        margin: 14px 0 0 0;
        padding: 0;
        list-style: none;
    }
    .strengths-list li {
        color: #86efac;
        font-size: 0.9em;
        padding: 3px 0;
    }
    .strengths-list li::before { content: "✓  "; }
    /* Caveat */
    .caveat {
        background: #292524;
        border-left: 3px solid #f59e0b;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        margin-top: 16px;
        font-size: 0.88em;
        color: #fde68a;
    }
    /* Alt card */
    .alt-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 10px;
    }
    .alt-label {
        font-size: 0.7em;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .alt-name {
        font-size: 1.1em;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 2px;
    }
    .alt-meta {
        font-size: 0.82em;
        color: #6b7280;
        margin-bottom: 10px;
    }
    .alt-reason { color: #9ca3af; font-size: 0.9em; margin-bottom: 8px; }
    .alt-tradeoff {
        font-size: 0.85em;
        color: #fbbf24;
    }
    /* Skip card */
    .skip-card {
        background: #111111;
        border-left: 3px solid #374151;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
    }
    .skip-name { font-size: 0.92em; font-weight: 600; color: #9ca3af; }
    .skip-reason { font-size: 0.85em; color: #6b7280; margin-top: 2px; }
    /* Summary */
    .summary-bar {
        background: #0c1929;
        border-left: 3px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 22px;
        color: #7dd3fc;
        font-size: 0.92em;
        line-height: 1.55;
    }
</style>
""", unsafe_allow_html=True)


# ── Cache model loading ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_load_models():
    return load_models("models.json")


# ── Render helpers ────────────────────────────────────────────────────────────
def render_best_pick(pick: dict):
    st.markdown(f"""
    <div class="best-card">
        <div class="best-label">✦ Best Pick</div>
        <div class="model-name">{pick['model_name']}</div>
        <div class="provider-name">by {pick['provider']}</div>
        <div class="reason-text">{pick['reason']}</div>
        <div class="meta-grid">
            <div class="meta-cell">
                <div class="meta-cell-label">Pricing</div>
                <div class="meta-cell-value">{pick.get('pricing_note', 'N/A')}</div>
            </div>
            <div class="meta-cell">
                <div class="meta-cell-label">Context Window</div>
                <div class="meta-cell-value">{pick.get('context_window_k', '?')}</div>
            </div>
            <div class="meta-cell">
                <div class="meta-cell-label">License</div>
                <div class="meta-cell-value">{pick.get('license', 'N/A')}</div>
            </div>
            <div class="meta-cell">
                <div class="meta-cell-label">Self-hostable</div>
                <div class="meta-cell-value">{'✓ Yes' if pick.get('self_hostable') else '✗ No'}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # API string — outside the HTML block so it's copyable
    if pick.get("api_model_string"):
        st.markdown("**API model string**")
        st.code(pick["api_model_string"], language=None)

    # Strengths
    strengths = pick.get("key_strengths", [])
    if strengths:
        items = "".join(f"<li>{s}</li>" for s in strengths)
        st.markdown(f"""
        <div style="margin: 4px 0 0 0;">
            <div style="font-size:0.8em;color:#6b7280;text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:6px;">Key Strengths</div>
            <ul class="strengths-list">{items}</ul>
        </div>
        """, unsafe_allow_html=True)

    # Caveat
    if pick.get("watch_out_for"):
        st.markdown(f"""
        <div class="caveat">⚠ <strong>Watch out for:</strong> {pick['watch_out_for']}</div>
        """, unsafe_allow_html=True)


def render_alternative(alt: dict, index: int):
    label = f"Alternative {index + 1}"
    st.markdown(f"""
    <div class="alt-card">
        <div class="alt-label">{label}</div>
        <div class="alt-name">{alt['model_name']}</div>
        <div class="alt-meta">
            {alt['provider']} &nbsp;·&nbsp; {alt.get('license','?')} &nbsp;·&nbsp; {alt.get('pricing_note','')}
        </div>
        <div class="alt-reason">{alt['reason']}</div>
        <div class="alt-tradeoff">⚡ Trade-off vs best pick: {alt.get('trade_off','N/A')}</div>
    </div>
    """, unsafe_allow_html=True)


def render_not_recommended(items: list):
    if not items:
        return
    with st.expander("Why certain obvious picks were skipped"):
        for item in items:
            st.markdown(f"""
            <div class="skip-card">
                <div class="skip-name">✗ {item.get('model_name','?')}</div>
                <div class="skip-reason">{item.get('reason','')}</div>
            </div>
            """, unsafe_allow_html=True)


# ── App state helpers ─────────────────────────────────────────────────────────
def reset_state():
    st.session_state.result = None
    st.session_state.error = None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Init session state
    if "result" not in st.session_state:
        st.session_state.result = None
    if "error" not in st.session_state:
        st.session_state.error = None
    if "use_case" not in st.session_state:
        st.session_state.use_case = ""

    # Header
    st.title("🧠 LLM Picker")
    st.markdown(
        "Describe what you're building. Get the best LLM for your use case — "
        "with reasoning, pricing, alternatives, and honest caveats."
    )
    st.divider()

    # Load models (cached after first load)
    try:
        models_data = cached_load_models()
    except FileNotFoundError:
        st.error("❌ `models.json` not found. Place it in the same folder as `app.py` and restart.")
        st.stop()
        return

    # ── Example prompt buttons ─────────────────────────────────────────────
    st.markdown("**Try an example:**")
    cols = st.columns(len(EXAMPLE_PROMPTS))
    for i, col in enumerate(cols):
        with col:
            if st.button(f"#{i+1}", key=f"ex_{i}", use_container_width=True,
                         help=EXAMPLE_PROMPTS[i]):
                # FIXED: write directly to the WIDGET's session-state key,
                # then rerun so the text area visibly updates. The old code
                # wrote to a separate mirror key that the widget ignored.
                st.session_state["use_case"] = EXAMPLE_PROMPTS[i]
                reset_state()
                st.rerun()

    # ── Input ──────────────────────────────────────────────────────────────
    # The widget key is "use_case" — its value IS st.session_state.use_case.
    use_case = st.text_area(
        label="Describe your use case",
        height=130,
        max_chars=2000,
        placeholder=(
            "e.g. Building a customer support chatbot in Hindi. "
            "~5000 requests/day. Budget under $30/month. "
            "Low latency needed. Data must not leave India."
        ),
        key="use_case",
        on_change=reset_state,
    )

    char_count = len(use_case)
    col_count, col_btn = st.columns([3, 1])
    with col_count:
        color = "#ef4444" if char_count > 1800 else "#6b7280"
        st.markdown(
            f'<p style="color:{color};font-size:0.8em;margin-top:6px;">'
            f'{char_count}/2000</p>',
            unsafe_allow_html=True,
        )
    with col_btn:
        submit = st.button(
            "Find Best Model →",
            type="primary",
            use_container_width=True,
            disabled=(char_count < 10),
        )

    # ── Call recommender ───────────────────────────────────────────────────
    if submit and use_case.strip():
        reset_state()
        with st.spinner("Analyzing your use case... (check terminal for logs)"):
            result = get_recommendation(use_case, models_data)

        if result["success"]:
            st.session_state.result = result["data"]
        else:
            st.session_state.error = result["error"]

    # ── Show error ─────────────────────────────────────────────────────────
    if st.session_state.error:
        st.error(f"❌ {st.session_state.error}")
        st.caption("Check the terminal for detailed error logs.")

    # ── Show results ───────────────────────────────────────────────────────
    rec = st.session_state.result
    if rec:
        # Summary
        if rec.get("use_case_summary"):
            st.markdown(f"""
            <div class="summary-bar">
                💡 <strong>What I understood:</strong> {rec['use_case_summary']}
            </div>
            """, unsafe_allow_html=True)

        # Best pick
        st.subheader("Best Pick")
        if "best_pick" in rec:
            render_best_pick(rec["best_pick"])
        else:
            st.warning("No best pick found in response. Check terminal logs.")

        # Alternatives
        alts = rec.get("alternatives", [])
        if alts:
            st.subheader("Alternatives")
            for i, alt in enumerate(alts):
                render_alternative(alt, i)

        # Not recommended
        render_not_recommended(rec.get("not_recommended", []))

        # Reset button
        st.divider()
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("Start over", use_container_width=True):
                st.session_state["use_case"] = ""
                reset_state()
                st.rerun()

        st.caption(
            "Powered by Gemini 3.1 Flash (free tier) · "
            "Based on a curated model database · "
            "Always verify pricing at the provider's official page."
        )


if __name__ == "__main__":
    main()