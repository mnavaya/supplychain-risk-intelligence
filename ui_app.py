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

# Bold colourful desk — high-chroma panels + dark visible links
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;600;700&display=swap');

:root {
  --ink: #0b1f33;
  --link: #061428;
  --muted: #234058;
}

html, body, [class*="css"] {
  font-family: "Source Sans 3", sans-serif;
  color: var(--ink);
}

/* Links: bold + dark so they never wash out */
a, a:visited, .stMarkdown a, [data-testid="stMarkdownContainer"] a {
  color: var(--link) !important;
  font-weight: 700 !important;
  text-decoration: underline !important;
  text-underline-offset: 3px;
  text-decoration-thickness: 2px;
}
a:hover, .stMarkdown a:hover {
  color: #8b0000 !important;
}

.stApp {
  background:
    radial-gradient(900px 500px at 5% 0%, #7ce8ff 0%, transparent 55%),
    radial-gradient(800px 460px at 95% 5%, #ffe566 0%, transparent 50%),
    radial-gradient(700px 480px at 50% 100%, #ff8fab 0%, transparent 55%),
    radial-gradient(600px 400px at 15% 80%, #b8f2a0 0%, transparent 50%),
    linear-gradient(145deg, #4cc9f0 0%, #80ffdb 28%, #ffd60a 58%, #ff6b6b 100%);
  background-attachment: fixed;
}

[data-testid="stSidebar"] {
  background: linear-gradient(185deg, #3a0ca3 0%, #4361ee 40%, #4cc9f0 78%, #80ffdb 100%) !important;
  border-right: 4px solid #061428;
}
[data-testid="stSidebar"] * {
  color: #061428 !important;
  font-weight: 600 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
  color: #061428 !important;
}
[data-testid="stSidebar"] a {
  color: #061428 !important;
  font-weight: 800 !important;
}

.hero {
  background: linear-gradient(120deg, #560bad 0%, #f72585 35%, #ff9e00 70%, #ffd60a 100%);
  border: 4px solid #061428;
  border-radius: 22px;
  padding: 1.35rem 1.5rem 1.45rem;
  margin-bottom: 1.1rem;
  box-shadow: 8px 8px 0 #061428;
}

.brand-title {
  font-family: "Fraunces", Georgia, serif !important;
  font-size: 2.55rem;
  font-weight: 700;
  margin: 0;
  line-height: 1.1;
  color: #ffffff !important;
  text-shadow: 2px 2px 0 #061428;
}

.brand-sub {
  margin: 0.55rem 0 0 0;
  color: #fff8e7 !important;
  font-size: 1.08rem;
  font-weight: 600;
  max-width: 42rem;
}

.color-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin: 0.85rem 0 1.1rem;
}
.color-strip span {
  display: inline-block;
  padding: 0.28rem 0.75rem;
  border-radius: 999px;
  border: 2px solid #061428;
  font-weight: 700;
  font-size: 0.85rem;
  color: #061428;
  box-shadow: 2px 2px 0 #061428;
}
.chip-blue { background: #4cc9f0; }
.chip-green { background: #80ffdb; }
.chip-yellow { background: #ffd60a; }
.chip-pink { background: #ff8fab; }
.chip-orange { background: #ff9e00; }

.soft-panel {
  background: #fff3b0;
  border: 3px solid #061428;
  border-radius: 16px;
  padding: 1rem 1.15rem;
  box-shadow: 5px 5px 0 #061428;
  font-weight: 600;
}

/* Each metric gets a loud colour via nth-child */
div[data-testid="stHorizontalBlock"] > div:nth-child(1) div[data-testid="stMetric"] {
  background: #4cc9f0 !important;
}
div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stMetric"] {
  background: #ff9e00 !important;
}
div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stMetric"] {
  background: #ff6b6b !important;
}
div[data-testid="stHorizontalBlock"] > div:nth-child(4) div[data-testid="stMetric"] {
  background: #80ffdb !important;
}

div[data-testid="stMetric"] {
  border: 3px solid #061428 !important;
  border-radius: 16px !important;
  padding: 0.95rem 1rem !important;
  box-shadow: 5px 5px 0 #061428;
}
div[data-testid="stMetric"] label {
  color: #061428 !important;
  font-weight: 700 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: #061428 !important;
  font-weight: 800 !important;
}

.stButton > button {
  background: linear-gradient(135deg, #f72585 0%, #7209b7 55%, #3a0ca3 100%) !important;
  color: #ffffff !important;
  border: 3px solid #061428 !important;
  border-radius: 14px !important;
  font-weight: 800 !important;
  padding: 0.65rem 1rem !important;
  box-shadow: 4px 4px 0 #061428 !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #ffd60a 0%, #ff9e00 100%) !important;
  color: #061428 !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.5rem;
  background: transparent;
}
.stTabs [data-baseweb="tab"] {
  background: #ffffff;
  border-radius: 999px;
  padding: 0.4rem 1.05rem;
  border: 3px solid #061428 !important;
  font-weight: 700 !important;
  color: #061428 !important;
  box-shadow: 3px 3px 0 #061428;
}
.stTabs [data-baseweb="tab"]:nth-child(1) { background: #4cc9f0 !important; }
.stTabs [data-baseweb="tab"]:nth-child(2) { background: #80ffdb !important; }
.stTabs [data-baseweb="tab"]:nth-child(3) { background: #ffd60a !important; }
.stTabs [data-baseweb="tab"]:nth-child(4) { background: #ff8fab !important; }
.stTabs [aria-selected="true"] {
  transform: translateY(-2px);
  box-shadow: 4px 4px 0 #061428 !important;
}

[data-testid="stChatMessage"] {
  background: #e7f5ff !important;
  border: 3px solid #061428 !important;
  border-radius: 16px;
  padding: 0.4rem 0.65rem;
  box-shadow: 4px 4px 0 #061428;
}

/* Report / markdown headings pop */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
  color: #061428 !important;
  font-weight: 800 !important;
}

.footer-note {
  margin-top: 1.5rem;
  color: #061428;
  font-size: 0.95rem;
  font-weight: 700;
  background: #ffd60a;
  display: inline-block;
  padding: 0.35rem 0.8rem;
  border: 2px solid #061428;
  border-radius: 999px;
  box-shadow: 2px 2px 0 #061428;
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
    st.markdown("### Run controls")
    file_options = _available_weekly_files()
    selected_file = st.selectbox(
        "Weekly CSV",
        options=file_options,
        index=0 if file_options else None,
        placeholder="No CSV files in data/weekly",
    )
    uploaded_csv = st.file_uploader("Or upload a CSV", type=["csv"])
    forecast_weeks = st.slider("Forecast horizon (weeks)", 2, 16, 8)
    with_report = st.toggle("Generate LLM report", value=True)
    run_button = st.button("Brew fresh analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption(
        "Uses Holt-Winters demand, safety stock, reorder points, and ABC class from `main.py`."
    )


if uploaded_csv is not None:
    selected_file = _save_uploaded_csv(uploaded_csv)
    st.toast(f"Saved {selected_file}", icon="📎")


# ---------- Hero ----------
st.markdown(
    """
<div class="hero">
  <p class="brand-title">Inventory Nook</p>
  <p class="brand-sub">Colourful weekly stock health — forecasts, reorder alerts, and a live copilot for your shift.</p>
</div>
<div class="color-strip">
  <span class="chip-blue">Forecasts</span>
  <span class="chip-orange">Reorder points</span>
  <span class="chip-pink">Risk flags</span>
  <span class="chip-yellow">ABC classes</span>
  <span class="chip-green">Live copilot</span>
</div>
""",
    unsafe_allow_html=True,
)


# ---------- Pipeline run ----------
if run_button:
    if not selected_file:
        st.error("Choose or upload a weekly CSV first.")
    else:
        with st.spinner("Warming the kettle… running forecasts and risk checks…"):
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


# Prefer live run; otherwise fall back to last summary on disk
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
            '<div class="soft-panel">No inventory summary yet. Pick a weekly file in the sidebar and hit <strong>Brew fresh analysis</strong>.</div>',
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

    if prompt := st.chat_input("What's on your mind for this week's stock?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            answer = agent.query(prompt)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


st.markdown(
    '<p class="footer-note">Inventory Nook · powered by the local supply-chain pipeline in main.py</p>',
    unsafe_allow_html=True,
)
