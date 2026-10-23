import altair as alt
import streamlit as st
import pandas as pd
from pathlib import Path

from inventory_logic import (
    DEFAULT_SERVICE_LEVEL,
    MONTE_CARLO_RUNS,
    SERVICE_LEVELS,
    SHOCK_COLUMNS,
    accuracy_display_table,
    apply_service_level,
    describe_shock,
    eligible_forecast_skus,
    forecast_accuracy_check,
    format_abc_summary,
    format_risk_board,
    format_service_snapshot,
    format_shock_table,
    risk_count_frame,
    risk_heatmap_frame,
    service_level_message,
    service_level_snapshot,
    simulate_supply_shock,
)
from main import DATA_DIR, OUTPUT_DIR, SupplyChainAgent, generate_scenario_narrative, run_pipeline


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

/* Hero — big centered brand header */
.hero {
  padding: 1.6rem 0 1.2rem;
  margin-bottom: 0.75rem;
  border-bottom: 1px solid var(--line);
  text-align: center;
}
.brand-kicker {
  margin: 0 0 0.65rem 0;
  font-size: clamp(2.4rem, 6vw, 4.2rem);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #ffffff !important;
  font-weight: 800;
  line-height: 1.05;
  text-align: center;
}
.brand-title {
  margin: 0.85rem 0 0 0;
  font-size: 1.05rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: #ffffff !important;
  line-height: 1.35;
  text-align: left;
  max-width: none;
}
.brand-sub {
  margin: 0.4rem 0 0 0;
  max-width: 40rem;
  color: #c4cad3 !important;
  font-size: 0.88rem;
  font-weight: 400;
  line-height: 1.5;
  text-align: left;
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

.note-box {
  margin: 0.75rem 0 0.9rem 0;
  padding: 0.75rem 0.85rem;
  background: #1a2030;
  border: 1px solid #5a6475;
  border-left: 3px solid #ffffff;
  border-radius: 8px;
  color: #d5dae2 !important;
  font-size: 0.78rem;
  line-height: 1.45;
  font-weight: 400;
}
.note-box strong {
  color: #ffffff !important;
  font-weight: 700;
}
.note-box a {
  color: #ffffff !important;
  font-weight: 700 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def _is_inventory_csv(path):
    """True when the first line looks like a weekly inventory file."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
    except OSError:
        return False
    columns = {part.strip().lower() for part in header.split(",")}
    return {"sku", "consumption", "inventory_level"}.issubset(columns)


def _available_weekly_files():
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.glob("*.csv") if _is_inventory_csv(p))


def _save_uploaded_csv(uploaded_file):
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        destination = DATA_DIR / Path(uploaded_file.name).name
        destination.write_bytes(uploaded_file.getvalue())
        return destination.name
    except OSError:
        return None


def _load_existing_summary():
    path = OUTPUT_DIR.parent / "latest_summary.csv"
    if path.exists():
        return pd.read_csv(path), path
    return None, path


def _default_file_index(options):
    preferred = "week_2026-04-13_demo.csv"
    if preferred in options:
        return options.index(preferred)
    return 0 if options else None


def _dark_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background="#1a1f28")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#f5f6f7",
            titleColor="#f5f6f7",
            gridColor="#3a4250",
            domainColor="#5a6475",
        )
        .configure_legend(labelColor="#f5f6f7", titleColor="#f5f6f7")
        .configure_title(color="#ffffff")
    )


def _shock_bar_chart(shock: pd.DataFrame) -> alt.Chart:
    counts = risk_count_frame(shock)
    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("risk:N", sort=["High", "Medium", "Low"], title="Stockout risk"),
            y=alt.Y("skus:Q", title="SKUs"),
            color=alt.Color(
                "when:N",
                sort=["Before", "After"],
                title=None,
                scale=alt.Scale(domain=["Before", "After"], range=["#8aa0b8", "#f4f1ea"]),
            ),
            xOffset="when:N",
            tooltip=["when:N", "risk:N", "skus:Q"],
        )
        .properties(height=260, title="Risk before and after the shock")
    )
    return _dark_chart(chart)


def _shock_heatmap(shock: pd.DataFrame) -> alt.Chart:
    heat = risk_heatmap_frame(shock)
    chart = (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("when:N", sort=["Before", "After"], title=None),
            y=alt.Y("sku:N", sort=shock["sku"].tolist(), title="SKU"),
            color=alt.Color(
                "risk:N",
                title="Risk",
                scale=alt.Scale(
                    domain=["High", "Medium", "Low"],
                    range=["#d45d4e", "#e0b15a", "#3f9d6a"],
                ),
            ),
            tooltip=[
                "sku:N",
                "when:N",
                "risk:N",
                alt.Tooltip("stockout_probability:Q", format=".0%", title="Stockout probability"),
            ],
        )
        .properties(height=max(220, 18 * max(len(shock), 1)), title="Before / after risk")
    )
    return _dark_chart(chart)


def _accuracy_for(raw_df: pd.DataFrame, sku: str, source_file: str):
    """Fit the holdout once per file and SKU, not on every sidebar tweak."""
    key = (source_file, sku, id(raw_df))
    if st.session_state.get("accuracy_key") != key:
        st.session_state["accuracy_key"] = key
        st.session_state["accuracy_result"] = forecast_accuracy_check(raw_df, sku)
    return st.session_state["accuracy_result"]


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
        index=_default_file_index(file_options),
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

    service_level = st.selectbox(
        "Service level",
        options=list(SERVICE_LEVELS.keys()),
        index=list(SERVICE_LEVELS.keys()).index(DEFAULT_SERVICE_LEVEL),
        format_func=lambda level: f"{level} (Z={SERVICE_LEVELS[level]:.2f})",
        help="Safety stock = Z × weekly demand SD × sqrt(lead time in weeks). 90% uses Z=1.28, 95% uses Z=1.65, 99% uses Z=2.33.",
    )
    forecast_weeks = st.slider("Forecast horizon (weeks)", 2, 16, 8)
    with_report = st.toggle("Generate LLM report", value=False)
    st.markdown(
        """
<div class="note-box">
  <strong>Note:</strong> The written LLM report needs <strong>Ollama</strong> running locally with <strong>gemma2:2b</strong>.
  On Streamlit Cloud this is expected to fail — leave this toggle off.
  Forecasts, the risk board, metrics, and the Copilot still work without Ollama.
  For the full report: install Ollama from <a href="https://ollama.com/download" target="_blank">ollama.com/download</a>,
  run <code>ollama pull gemma2:2b</code>, then use this app on your laptop with the toggle on.
</div>
""",
        unsafe_allow_html=True,
    )
    run_button = st.button("Run analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Inventory Nook — Local supply chain pipeline")


if uploaded_csv is not None:
    saved_name = _save_uploaded_csv(uploaded_csv)
    if saved_name:
        selected_file = saved_name
        st.toast(f"Saved {selected_file}", icon="📎")
    else:
        st.error("Could not save that upload. Use the bundled sample CSV instead.")


# ---------- Hero ----------
st.markdown(
    """
<div class="hero">
  <p class="brand-kicker">Supply chain intelligence</p>
  <p class="brand-title">Clear inventory risk, when you need it.</p>
  <p class="brand-sub">Forecast demand, stress a supply shock, set service-level safety stock, and ask the copilot — all from your weekly CSV.</p>
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
                service_level=service_level,
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


summary_df = apply_service_level(result["full_summary"], service_level)
raw_df = result.get("raw_df")
report = result.get("report") or ""
unit_cost_source = result.get("unit_cost_source")

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


tabs = st.tabs(
    [
        "Risk board",
        "Supply shock",
        "Forecast accuracy",
        "ABC summary",
        "Raw weeks",
        "Written report",
        "Copilot",
    ]
)

with tabs[0]:
    st.markdown("##### Service level, safety stock, and reorder point")
    st.caption(
        "Safety stock = Z × weekly demand SD × √(lead time in weeks). "
        "Reorder point = average weekly demand × lead time in weeks + safety stock. "
        f"The risk board uses {service_level} (Z={SERVICE_LEVELS[service_level]:.2f})."
    )
    level_columns = {"weekly_demand_sd", "avg_weekly_demand", "lead_time_weeks"}
    if level_columns.issubset(summary_df.columns):
        snapshot = service_level_snapshot(summary_df)
        st.dataframe(
            format_service_snapshot(snapshot),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(service_level_message(snapshot))
    else:
        st.caption("Re-run the analysis to compare 90%, 95%, and 99% on this file.")

    st.markdown("##### What needs attention")
    st.caption("High risk is sorted first. Weeks of cover compare on-hand stock with forecast weekly demand.")
    st.dataframe(format_risk_board(summary_df), use_container_width=True, hide_index=True)

with tabs[1]:
    st.markdown("##### What-if supply shock")
    st.caption(
        "Add supplier delay, a demand spike, or both. Risk uses the same weeks-of-cover rules as the board. "
        f"Stockout probability is a {MONTE_CARLO_RUNS}-run Monte Carlo: each week is drawn around the shocked "
        "forecast using that SKU's historical demand SD. Emergency orders fill the gap up to the shocked reorder point."
    )
    shock_ready = set(SHOCK_COLUMNS).issubset(summary_df.columns) and "z_score" in summary_df.columns
    if not shock_ready:
        st.caption("Re-run the analysis to simulate a shock. This summary is missing demand or lead-time inputs.")
    else:
        shock_controls = st.columns(3)
        extra_weeks = shock_controls[0].slider("Supplier delay (extra weeks)", 0, 8, 2)
        spike_pct = shock_controls[1].slider("Demand spike (%)", 0, 200, 50, step=5)
        spike_weeks = shock_controls[2].slider("Spike length (weeks)", 1, 12, 4)
        if spike_pct == 0:
            st.caption("Demand spike is 0%, so spike length does not change the scenario.")

        shock = simulate_supply_shock(
            summary_df,
            extra_lead_weeks=extra_weeks,
            spike_pct=spike_pct,
            spike_weeks=spike_weeks,
            n_runs=MONTE_CARLO_RUNS,
        )
        flipped = int(shock["risk_flipped"].sum())
        high_after = int((shock["stockout_risk_after"] == "High").sum())
        likely = int((shock["stockout_probability"] >= 0.5).sum())
        total_order = float(shock["emergency_order_qty"].sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("SKUs flipped higher", flipped)
        m2.metric("High risk after", high_after)
        m3.metric("Stockout probability ≥ 50%", likely)
        m4.metric("Emergency units", f"{total_order:,.0f}")

        st.markdown(describe_shock(shock, extra_weeks, spike_pct, spike_weeks))
        chart_left, chart_right = st.columns([1, 1])
        chart_left.altair_chart(_shock_bar_chart(shock), use_container_width=True)
        chart_right.altair_chart(_shock_heatmap(shock), use_container_width=True)

        st.markdown("##### SKUs that flip to a higher risk")
        flips = format_shock_table(shock, flips_only=True)
        if flips.empty:
            st.caption("No SKU moves into a worse risk band under this scenario.")
        else:
            st.dataframe(flips, use_container_width=True, hide_index=True)

        st.markdown("##### Every SKU under the scenario")
        st.caption("Sorted with risk increases first, then the highest stockout probability.")
        st.dataframe(format_shock_table(shock), use_container_width=True, hide_index=True)

        narrative_key = (extra_weeks, spike_pct, spike_weeks, id(summary_df))
        if st.button("Write scenario narrative", key="shock_narrative_button"):
            with st.spinner("Asking Gemma 2…"):
                told = generate_scenario_narrative(shock, extra_weeks, spike_pct, spike_weeks)
            st.session_state["shock_narrative"] = told
            st.session_state["shock_narrative_key"] = narrative_key

        saved = st.session_state.get("shock_narrative")
        if saved and st.session_state.get("shock_narrative_key") == narrative_key:
            if saved.get("source") == "gemma2:2b":
                st.markdown("##### Gemma 2")
                st.markdown(saved.get("narrative") or "")
            else:
                st.caption(
                    "Gemma 2 is not available, so the summary above is the scenario narrative. "
                    f"({saved.get('error') or 'model unavailable'})"
                )
        elif saved:
            st.caption("Sliders changed. Write the narrative again to refresh the Gemma 2 note.")

with tabs[2]:
    st.markdown("##### Forecast accuracy check")
    st.caption(
        "For a frequently sold SKU, the last 4 weeks are held out. "
        "Holt-Winters is fit on earlier weeks only and compared with a naive baseline "
        "that repeats the previous week's demand. Test weeks are not used in training."
    )
    if raw_df is None:
        st.caption("Run a fresh analysis to check a SKU. A saved summary does not include weekly history.")
    else:
        sku_choices = eligible_forecast_skus(raw_df)
        if not sku_choices:
            st.warning(
                "No SKU in this file has enough weekly history for a 4-week holdout "
                "(at least 8 weeks with demand)."
            )
        else:
            accuracy_sku = st.selectbox(
                "Frequently sold SKU",
                options=sku_choices,
                help="Most-sold SKUs are listed first. Only SKUs with enough weekly history are shown.",
            )
            with st.spinner("Fitting Holt-Winters on earlier weeks…"):
                check = _accuracy_for(
                    raw_df,
                    accuracy_sku,
                    result.get("source_file") or "",
                )
            if check.get("status") != "ok":
                st.warning(check.get("message", "Not enough history for this SKU."))
            else:
                st.dataframe(
                    accuracy_display_table(check),
                    use_container_width=True,
                    hide_index=True,
                )
                if check["winner"] == "Holt-Winters":
                    st.success(check["message"])
                elif check["winner"] == "Naive baseline":
                    st.info(check["message"])
                else:
                    st.info(check["message"])

with tabs[3]:
    st.markdown("##### ABC classification")
    st.caption(
        "Usage value = annual demand × unit cost. SKUs are sorted from highest usage value "
        "to lowest. A is the first 80% of usage value, B the next 15%, and C the remaining 5%."
    )
    if unit_cost_source == "sample":
        st.caption(
            "This file has no unit cost column, so sample planning prices were applied "
            "(Reagent $48, Consumable $6.50, Equipment $125, Chemical $22, other $15, "
            "then a stable per-SKU adjustment). See the README."
        )
    elif unit_cost_source == "mixed":
        st.caption("Missing unit costs were filled with the documented sample prices. See the README.")
    elif unit_cost_source == "file":
        st.caption("Unit cost came from the weekly file.")
    abc_table = format_abc_summary(summary_df)
    if abc_table.empty:
        st.caption("Re-run the analysis to build the ABC summary for this file.")
    else:
        st.dataframe(abc_table, use_container_width=True, hide_index=True)
        usage_pct = pd.to_numeric(abc_table["% of usage value"], errors="coerce").sum()
        sku_pct = pd.to_numeric(abc_table["% of SKUs"], errors="coerce").sum()
        st.caption(
            f"Usage value shares sum to {usage_pct:.1f}%. SKU shares sum to {sku_pct:.1f}%."
        )

with tabs[4]:
    if raw_df is not None:
        st.dataframe(raw_df, use_container_width=True, hide_index=True)
    else:
        st.caption("Raw weekly rows appear after you run a fresh analysis in this session.")

with tabs[5]:
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

with tabs[6]:
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
