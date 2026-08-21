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

# Dark minimal UI — high contrast on circled controls / upload / button / hero
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #0b0d10;
  --panel: #1a1f28;
  --panel-2: #262c38;
  --line: #4a5260;
  --text: #f5f6f7;
  --muted: #b6bec9;
  --link: #ffffff;
}

html, body, [class*="css"] {
  font-family: "Inter", system-ui, sans-serif;
  color: var(--text);
}

.stApp {
  background:
    radial-gradient(1100px 650px at 50% -10%, #1c2433 0%, transparent 55%),
    linear-gradient(180deg, #0b0d10 0%, #12161e 55%, #0b0d10 100%);
}

[data-testid="stSidebar"] {
  background: #0e1117 !important;
  border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
  color: #ffffff !important;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
  color: var(--muted) !important;
}

/* Links */
a, a:visited,
.stMarkdown a,
[data-testid="stMarkdownContainer"] a {
  color: var(--link) !important;
  font-weight: 700 !important;
  text-decoration: underline !important;
  text-underline-offset: 3px;
  text-decoration-thickness: 2px;
}

/* Hero — compact, small, well formatted */
.hero {
  padding: 1rem 0 0.95rem;
  margin-bottom: 0.65rem;
  border-bottom: 1px solid var(--line);
}
.brand-kicker {
  margin: 0 0 0.4rem 0;
  font-size: 0.68rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--muted) !important;
  font-weight: 600;
}
.brand-title {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: #ffffff !important;
  line-height: 1.35;
}
.brand-sub {
  margin: 0.35rem 0 0 0;
  max-width: 40rem;
  color: #c4cad3 !important;
  font-size: 0.88rem;
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

.sidebar-card {
  background: #171b22;
  border: 1px solid #5a6475;
  border-radius: 12px;
  padding: 0.85rem 0.9rem 1rem;
  margin: 0 0 0.9rem 0;
}
.sidebar-card-title {
  margin: 0 0 0.55rem 0;
  font-size: 0.72rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #ffffff !important;
  font-weight: 700;
}

/* Metrics */
div[data-testid="stMetric"] {
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  padding: 1rem 1.1rem !important;
}
div[data-testid="stMetric"] label {
  color: var(--muted) !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.72rem !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: #ffffff !important;
  font-weight: 700 !important;
}

/* Primary button — dark text on white (fix invisible label) */
[data-testid="stSidebar"] .stButton > button,
.stButton > button {
  background: #ffffff !important;
  color: #0b0d10 !important;
  border: 2px solid #ffffff !important;
  border-radius: 999px !important;
  font-weight: 800 !important;
  font-size: 1rem !important;
  padding: 0.7rem 1.1rem !important;
}
[data-testid="stSidebar"] .stButton > button *,
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span,
.stButton > button * {
  color: #0b0d10 !important;
  font-weight: 800 !important;
}
[data-testid="stSidebar"] .stButton > button:hover,
.stButton > button:hover {
  background: #e8ebf0 !important;
  color: #0b0d10 !important;
}

/* Tabs */
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

/* Select / slider readable */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background-color: #ffffff !important;
  color: #0b0d10 !important;
  border: 2px solid #ffffff !important;
  border-radius: 8px !important;
  min-height: 42px;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
  color: #0b0d10 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
  fill: #0b0d10 !important;
}

/* Toggle label readable */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  color: #ffffff !important;
  font-weight: 600 !important;
}

/* FILE UPLOADER — bright card that cannot blend into dark sidebar */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: #ffffff !important;
  border: 2px solid #ffffff !important;
  border-radius: 12px !important;
  padding: 0.75rem !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
[data-testid="stSidebar"] [data-testid="stFileUploader"] div {
  color: #0b0d10 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
  background: #f0f2f5 !important;
  border: 2px dashed #0b0d10 !important;
  border-radius: 10px !important;
  padding: 1.15rem 0.8rem !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button,
[data-testid="stSidebar"] [data-testid="stFileUploader"] button * {
  background: #0b0d10 !important;
  color: #ffffff !important;
  border: none !important;
  font-weight: 700 !important;
  border-radius: 999px !important;
}

.footer-note {
  margin-top: 2rem;
  color: var(--muted);
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: none;
  font-weight: 400;
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
    st.markdown(
        '<div class="sidebar-card"><p class="sidebar-card-title">Controls</p></div>',
        unsafe_allow_html=True,
    )
    file_options = _available_weekly_files()
    selected_file = st.selectbox(
        "Weekly CSV",
        options=file_options,
        index=0 if file_options else None,
        placeholder="No CSV files in data/weekly",
    )

    st.markdown(
        '<p class="sidebar-card-title" style="margin-top:1rem;">Upload file</p>',
        unsafe_allow_html=True,
    )
    uploaded_csv = st.file_uploader(
        "Drag & drop CSV here — or browse",
        type=["csv"],
        help="Upload a weekly inventory CSV to analyze.",
    )

    forecast_weeks = st.slider("Forecast horizon (weeks)", 2, 16, 8)
    with_report = st.toggle("Generate LLM report", value=False)
    run_button = st.button("Run analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Holt-Winters · safety stock · reorder points · ABC class")


if uploaded_csv is not None:
    selected_file = _save_uploaded_csv(uploaded_csv)
    st.toast(f"Saved {selected_file}", icon="📎")


# ---------- Hero ----------
st.markdown(
    """
<div class="hero">
  <p class="brand-kicker">Supply chain intelligence</p>
  <p class="brand-title">Clear inventory risk, when you need it.</p>
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
    '<p class="footer-note">All rights reserved — Morgan 2026</p>',
    unsafe_allow_html=True,
)
