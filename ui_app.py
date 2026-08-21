import streamlit as st
import pandas as pd
from pathlib import Path

from main import DATA_DIR, OUTPUT_DIR, SupplyChainAgent, run_pipeline


st.set_page_config(
    page_title="Inventory Nook",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Starlink-inspired: dark, plain, high-contrast — no rainbow
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #0b0d10;
  --bg-mid: #141820;
  --panel: #1a1f28;
  --panel-2: #222833;
  --line: #3a4250;
  --text: #f5f6f7;
  --muted: #a8b0bc;
  --link: #e8eaed;
}

html, body, [class*="css"] {
  font-family: "Inter", system-ui, sans-serif;
  color: var(--text);
}

.stApp {
  background:
    radial-gradient(1200px 700px at 50% 0%, #1a2333 0%, transparent 55%),
    linear-gradient(180deg, #0b0d10 0%, #12161e 50%, #0b0d10 100%);
}

/* Main text */
.stMarkdown, .stCaption, label, p, span, div {
  color: var(--text);
}

[data-testid="stSidebar"] {
  background: #0e1117 !important;
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * {
  color: var(--text) !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: var(--muted) !important;
}

/* Links: bold, light, clearly visible on dark */
a, a:visited,
.stMarkdown a,
[data-testid="stMarkdownContainer"] a,
[data-testid="stSidebar"] a {
  color: var(--link) !important;
  font-weight: 700 !important;
  text-decoration: underline !important;
  text-underline-offset: 3px;
  text-decoration-thickness: 2px;
}
a:hover, .stMarkdown a:hover {
  color: #ffffff !important;
}

.hero {
  padding: 2.2rem 0 1.6rem;
  margin-bottom: 0.5rem;
  border-bottom: 1px solid var(--line);
}
.brand-kicker {
  margin: 0 0 0.75rem 0;
  font-size: 0.72rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 600;
}
.brand-title {
  margin: 0;
  font-size: clamp(2.2rem, 4vw, 3.2rem);
  font-weight: 700;
  letter-spacing: -0.03em;
  color: #ffffff !important;
  line-height: 1.1;
}
.brand-sub {
  margin: 0.85rem 0 0 0;
  max-width: 34rem;
  color: var(--muted) !important;
  font-size: 1.05rem;
  font-weight: 400;
  line-height: 1.5;
}

.soft-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 1.1rem 1.2rem;
  color: var(--text);
}

/* Metrics — quiet dark panels */
div[data-testid="stMetric"] {
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  padding: 1rem 1.1rem !important;
}
div[data-testid="stMetric"] label {
  color: var(--muted) !important;
  font-weight: 500 !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.72rem !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: #ffffff !important;
  font-weight: 700 !important;
}

.stButton > button {
  background: #ffffff !important;
  color: #0b0d10 !important;
  border: none !important;
  border-radius: 999px !important;
  font-weight: 700 !important;
  padding: 0.65rem 1.1rem !important;
  letter-spacing: 0.02em;
}
.stButton > button:hover {
  background: #d7dbe2 !important;
  color: #0b0d10 !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.35rem;
  border-bottom: 1px solid var(--line);
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  border: none !important;
  border-radius: 0 !important;
  font-weight: 600 !important;
  padding: 0.55rem 0.9rem !important;
}
.stTabs [aria-selected="true"] {
  color: #ffffff !important;
  border-bottom: 2px solid #ffffff !important;
}

[data-testid="stChatMessage"] {
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px;
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
  color: #ffffff !important;
  font-weight: 700 !important;
}

/* Inputs / selects readable on dark */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background-color: var(--panel-2) !important;
  color: #ffffff !important;
  border-color: var(--line) !important;
}

/* FILE UPLOAD — high visibility drop zone */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: #f5f6f7 !important;
  border: 2px solid #ffffff !important;
  border-radius: 12px !important;
  padding: 0.85rem 0.9rem 1rem !important;
  box-shadow: 0 0 0 1px #0b0d10 inset;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
  color: #0b0d10 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
[data-testid="stSidebar"] [data-testid="stFileUploader"] p {
  color: #0b0d10 !important;
  font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  background: #ffffff !important;
  border: 2px dashed #3a4250 !important;
  border-radius: 10px !important;
  padding: 1.1rem 0.75rem !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
  background: #0b0d10 !important;
  color: #ffffff !important;
  border: none !important;
  font-weight: 700 !important;
  border-radius: 999px !important;
}

.upload-label {
  margin: 1rem 0 0.45rem 0;
  font-size: 0.72rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #ffffff !important;
  font-weight: 700;
}

.footer-note {
  margin-top: 2rem;
  color: var(--muted);
  font-size: 0.85rem;
  letter-spacing: 0.04em;
}
</style>
""",
    unsafe_allow_html=True,
)


def _available_weekly_files():
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.glob("*.csv"))


def _save_uploaded_csv(uploaded_file):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getvalue())
    return destination.name


def _load_existing_summary():
    path = OUTPUT_DIR.parent / "latest_summary.csv"
    if path.exists():
        return pd.read_csv(path), path
    return None, path


def _risk_sort_key(series: pd.Series) -> pd.Series:
    order = {
        "CRITICAL (Below ROP)": 0,
        "High": 1,
        "Medium": 2,
        "Low": 3,
    }
    return series.map(lambda x: order.get(x, 9))


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### Controls")
    file_options = _available_weekly_files()
    selected_file = st.selectbox(
        "Weekly CSV",
        options=file_options,
        index=0 if file_options else None,
        placeholder="No CSV files in data/weekly",
    )

    st.markdown('<p class="upload-label">Upload file</p>', unsafe_allow_html=True)
    uploaded_csv = st.file_uploader(
        "Drag and drop a CSV here",
        type=["csv"],
        help="Upload a weekly inventory CSV to analyze.",
        label_visibility="visible",
    )

    forecast_weeks = st.slider("Forecast horizon (weeks)", 2, 16, 8)
    with_report = st.toggle("Generate LLM report", value=False)
    run_button = st.button("Run analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption(
        "Holt-Winters demand · safety stock · reorder points · ABC class"
    )


if uploaded_csv is not None:
    selected_file = _save_uploaded_csv(uploaded_csv)
    st.toast(f"Saved {selected_file}", icon="📎")


# ---------- Hero ----------
st.markdown(
    """
<div class="hero">
  <p class="brand-kicker">Supply chain intelligence</p>
  <p class="brand-title">Clear inventory risk,<br/>when you need it.</p>
  <p class="brand-sub">Forecast demand, flag reorder breaches, and ask the copilot — all from your weekly CSV.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------- Pipeline run ----------
if run_button:
    if not selected_file:
        st.error("Choose or upload a weekly CSV first.")
    else:
        with st.spinner("Running forecasts and risk checks…"):
            result = run_pipeline(
                sample_file=selected_file,
                forecast_weeks=forecast_weeks,
                generate_report=with_report,
            )
        if result is None:
            st.error("Pipeline failed. Check the CSV columns and try again.")
        else:
            st.session_state["result"] = result
            st.session_state["messages"] = []
            st.success("Analysis ready.")


result = st.session_state.get("result")
if result is None:
    summary_df, summary_path = _load_existing_summary()
    if summary_df is not None:
        report_path = OUTPUT_DIR / "Inventory_Health_Risk_Report.md"
        report_text = (
            report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        )
        result = {
            "raw_df": None,
            "full_summary": summary_df,
            "report": report_text,
            "summary_path": summary_path,
        }
        st.info("Showing the last saved summary. Run a fresh analysis anytime from the sidebar.")
    else:
        st.markdown(
            '<div class="soft-panel">No inventory summary yet. Select or upload a weekly CSV, then click <strong>Run analysis</strong>.</div>',
            unsafe_allow_html=True,
        )
        st.stop()


summary_df = result["full_summary"]
raw_df = result.get("raw_df")
report = result.get("report") or ""

critical_mask = summary_df["stockout_risk"].astype(str).str.contains(
    "CRITICAL|High", regex=True
)
critical_count = int(critical_mask.sum())
avg_days = float(summary_df["days_to_stockout"].mean())
below_rop = int(
    (summary_df["current_inventory"] <= summary_df["reorder_point"]).sum()
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("SKUs watched", len(summary_df))
c2.metric("Below reorder point", below_rop)
c3.metric("High / critical risk", critical_count)
c4.metric("Avg days to stockout", f"{avg_days:.1f}")


tabs = st.tabs(["Risk board", "Raw weeks", "Written report", "Copilot"])

with tabs[0]:
    st.markdown("##### What needs attention")
    board = summary_df.copy()
    board["_risk_rank"] = _risk_sort_key(board["stockout_risk"])
    board = board.sort_values(["_risk_rank", "days_to_stockout"]).drop(
        columns=["_risk_rank"]
    )
    st.dataframe(board, use_container_width=True, hide_index=True)

with tabs[1]:
    if raw_df is not None:
        st.dataframe(raw_df, use_container_width=True, hide_index=True)
    else:
        st.caption("Raw weekly rows appear after you run a fresh analysis in this session.")

with tabs[2]:
    if report.strip():
        st.markdown(report)
        st.download_button(
            "Download report",
            data=report,
            file_name="Inventory_Health_Risk_Report.md",
            mime="text/markdown",
        )
    else:
        st.warning("No report text yet. Enable LLM report in the sidebar and re-run.")

with tabs[3]:
    st.markdown("##### Ask the inventory copilot")
    st.caption(
        "Try: *Which items are below reorder point?* · *Show high risk items* · *Status for SKU …*"
    )

    agent = SupplyChainAgent(summary_df)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about this week's stock"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            answer = agent.query(prompt)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


st.markdown(
    '<p class="footer-note">Inventory Nook · local supply-chain pipeline</p>',
    unsafe_allow_html=True,
)
