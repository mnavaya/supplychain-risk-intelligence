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

# Soft sage / warm stone — calm ops desk, not generic purple dashboard
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&family=Source+Sans+3:wght@400;500;600&display=swap');

:root {
  --ink: #2c332e;
  --muted: #5c675f;
  --sage: #6b8f71;
  --sage-deep: #4f6f55;
  --mist: #e8efe6;
  --cream: #f3f0ea;
  --panel: rgba(255, 252, 247, 0.88);
  --line: rgba(79, 111, 85, 0.18);
  --glow: rgba(107, 143, 113, 0.22);
}

html, body, [class*="css"] {
  font-family: "Source Sans 3", sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(1200px 600px at 12% -10%, var(--glow), transparent 55%),
    radial-gradient(900px 500px at 88% 8%, rgba(196, 168, 125, 0.18), transparent 50%),
    linear-gradient(165deg, #eef3ec 0%, #f6f3ed 48%, #ebe6dc 100%);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f7f4ee 0%, #e9efe7 100%);
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
  border-bottom: 1px solid var(--line);
}

.brand-title {
  font-size: 2.35rem;
  font-weight: 650;
  margin: 0;
  line-height: 1.15;
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
  box-shadow: 0 10px 30px rgba(44, 51, 46, 0.05);
  backdrop-filter: blur(6px);
}

div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 0.85rem 1rem;
  box-shadow: 0 8px 24px rgba(44, 51, 46, 0.04);
}

div[data-testid="stMetric"] label {
  color: var(--muted) !important;
}

.stButton > button {
  background: var(--sage-deep) !important;
  color: #f7faf6 !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  padding: 0.55rem 1rem !important;
  box-shadow: 0 6px 16px rgba(79, 111, 85, 0.25);
}

.stButton > button:hover {
  background: var(--sage) !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.4rem;
  background: transparent;
}

.stTabs [data-baseweb="tab"] {
  background: rgba(255, 252, 247, 0.7);
  border-radius: 999px;
  padding: 0.35rem 1rem;
  border: 1px solid var(--line);
}

.stTabs [aria-selected="true"] {
  background: var(--mist) !important;
  color: var(--sage-deep) !important;
  font-weight: 600;
}

[data-testid="stChatMessage"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 0.35rem 0.6rem;
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
