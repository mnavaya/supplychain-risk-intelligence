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

# Bright sky / lemon / coral — crisp daylight desk
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&family=Source+Sans+3:wght@400;500;600&display=swap');

:root {
  --ink: #12324a;
  --muted: #4a6a80;
  --accent: #00a8e8;
  --accent-deep: #0077b6;
  --sun: #ffe566;
  --coral: #ff6b6b;
  --panel: rgba(255, 255, 255, 0.92);
  --line: rgba(0, 119, 182, 0.18);
  --glow-sky: rgba(0, 200, 255, 0.35);
  --glow-sun: rgba(255, 229, 102, 0.55);
}

html, body, [class*="css"] {
  font-family: "Source Sans 3", sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(1000px 520px at 8% -8%, var(--glow-sky), transparent 55%),
    radial-gradient(900px 480px at 92% 0%, var(--glow-sun), transparent 50%),
    radial-gradient(700px 420px at 70% 100%, rgba(255, 107, 107, 0.18), transparent 55%),
    linear-gradient(165deg, #e8f9ff 0%, #fffef5 45%, #ffe9f0 100%);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #e6f8ff 55%, #fff7c2 100%);
  border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] * {
  color: var(--ink) !important;
}

h1, h2, h3, .brand-title {
  font-family: "Fraunces", Georgia, serif !important;
  letter-spacing: -0.02em;
  color: var(--ink) !important;
}

.hero {
  padding: 0.4rem 0 1.2rem 0;
  margin-bottom: 0.5rem;
  border-bottom: 2px solid rgba(0, 168, 232, 0.25);
}

.brand-title {
  font-size: 2.35rem;
  font-weight: 650;
  margin: 0;
  line-height: 1.15;
  background: linear-gradient(90deg, #0077b6 0%, #00a8e8 45%, #ff6b6b 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent !important;
}

.brand-sub {
  margin: 0.45rem 0 0 0;
  color: var(--muted);
  font-size: 1.05rem;
  max-width: 38rem;
}

.soft-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1rem 1.15rem;
  box-shadow: 0 12px 28px rgba(0, 119, 182, 0.08);
  backdrop-filter: blur(6px);
}

div[data-testid="stMetric"] {
  background: linear-gradient(180deg, #ffffff 0%, #f0fbff 100%);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 0.85rem 1rem;
  box-shadow: 0 8px 22px rgba(0, 168, 232, 0.12);
}

div[data-testid="stMetric"] label {
  color: var(--muted) !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: var(--accent-deep) !important;
}

.stButton > button {
  background: linear-gradient(135deg, #00a8e8 0%, #0077b6 100%) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  padding: 0.55rem 1rem !important;
  box-shadow: 0 8px 18px rgba(0, 168, 232, 0.35);
}

.stButton > button:hover {
  background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%) !important;
  box-shadow: 0 8px 18px rgba(255, 107, 107, 0.35);
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.4rem;
  background: transparent;
}

.stTabs [data-baseweb="tab"] {
  background: rgba(255, 255, 255, 0.85);
  border-radius: 999px;
  padding: 0.35rem 1rem;
  border: 1px solid var(--line);
}

.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, #ffe566 0%, #ffd166 100%) !important;
  color: #12324a !important;
  font-weight: 600;
  border-color: rgba(255, 209, 102, 0.9) !important;
}

[data-testid="stChatMessage"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 0.35rem 0.6rem;
  box-shadow: 0 6px 16px rgba(0, 168, 232, 0.08);
}

.footer-note {
  margin-top: 1.5rem;
  color: var(--muted);
  font-size: 0.9rem;
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
  <p class="brand-sub">A quiet desk for weekly stock health — forecasts, reorder alerts, and a calm copilot for your shift.</p>
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
