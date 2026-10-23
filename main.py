import pandas as pd
from pathlib import Path
import ollama
import warnings
import time
import datetime

from inventory_logic import (
    DEFAULT_LEAD_TIME_DAYS,
    DEFAULT_SERVICE_LEVEL,
    SERVICE_LEVELS,
    annual_demand_from_weekly,
    apply_service_level,
    attach_unit_cost,
    classify_abc,
    classify_cover_risk,
    describe_shock,
    forecast_mean_demand,
    format_weeks_of_cover,
    sort_risk_table,
    weekly_series,
    z_for_service_level,
)

warnings.filterwarnings("ignore")

# Directory setup. Cloud disks can reject writes; the UI still runs from memory.
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "weekly"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
GEMMA_MODEL = "gemma2:2b"
# A missing local model must fail fast instead of hanging a public demo.
GEMMA_TIMEOUT_SECONDS = 8.0


def _ensure_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def _write_text(path: Path, text: str) -> bool:
    if not _ensure_dir(path.parent):
        return False
    try:
        path.write_text(text, encoding="utf-8")
        return True
    except OSError:
        return False


def _write_csv(frame: pd.DataFrame, path: Path) -> bool:
    if not _ensure_dir(path.parent):
        return False
    try:
        frame.to_csv(path, index=False)
        return True
    except OSError:
        return False


_ensure_dir(OUTPUT_DIR)


def _gemma_chat(messages):
    """Call local Gemma 2. Times out quickly when Ollama is not running."""
    client = ollama.Client(timeout=GEMMA_TIMEOUT_SECONDS)
    return client.chat(model=GEMMA_MODEL, messages=messages)

# Default service-level factor (95%). Other levels live in SERVICE_LEVELS.
Z_SCORE = SERVICE_LEVELS[DEFAULT_SERVICE_LEVEL]


# ====================== DATA INGESTION & CLEANING ======================
def load_and_clean_weekly_data(filename: str):
    file_path = DATA_DIR / filename
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df = df.dropna(subset=["sku", "inventory_level", "consumption"])
    df = df.sort_values(["sku", "week"])

    df["daily_consumption"] = df["consumption"] / 7.0
    df["days_of_supply"] = df["inventory_level"] / df["daily_consumption"].replace(0, 1)

    print(f"✅ Loaded {len(df)} operational record(s).")
    return df


def _latest_number(sku_data: pd.DataFrame, column: str, default: float) -> float:
    if column not in sku_data.columns or sku_data.empty:
        return float(default)
    value = pd.to_numeric(sku_data[column], errors="coerce").iloc[-1]
    if pd.isna(value):
        return float(default)
    return float(value)


def _latest_text(sku_data: pd.DataFrame, column: str, default=""):
    if column not in sku_data.columns or sku_data.empty:
        return default
    value = sku_data[column].iloc[-1]
    if pd.isna(value):
        return default
    return value


# ====================== ADVANCED FORECASTING & METRICS ======================
def forecast_and_stockout(
    df,
    forecast_weeks: int = 8,
    service_level: str = DEFAULT_SERVICE_LEVEL,
):
    """Forecast each SKU and attach safety stock, cover risk, and ABC class.

    Safety stock and the reorder point use `service_level` (90%, 95%, or 99%).
    ABC class is assigned later from annual usage value, not from volume quantiles.
    """
    z_for_service_level(service_level)
    priced, unit_cost_source = attach_unit_cost(df)
    results = []

    for sku in priced["sku"].unique():
        sku_data = priced[priced["sku"] == sku].copy()
        sku_data["week"] = pd.to_datetime(sku_data["week"], errors="coerce")
        sku_data = sku_data.sort_values("week")
        series = weekly_series(sku_data)

        current_inv = round(_latest_number(sku_data, "inventory_level", 0.0), 1)
        # Round before cover and days-of-supply so a displayed 0.0 forecast
        # takes the zero-demand path instead of dividing by a hidden fraction.
        forecast_demand = round(
            forecast_mean_demand(series["consumption"], forecast_weeks), 1
        )
        weekly_std = (
            float(series["consumption"].std(ddof=1)) if len(series) > 1 else 0.0
        )
        if pd.isna(weekly_std):
            weekly_std = 0.0
        avg_weekly = float(series["consumption"].mean()) if len(series) else 0.0
        if pd.isna(avg_weekly):
            avg_weekly = 0.0

        lead_time = _latest_number(sku_data, "lead_time_days", DEFAULT_LEAD_TIME_DAYS)
        lt_weeks = lead_time / 7.0
        cover = classify_cover_risk(current_inv, forecast_demand, lt_weeks)

        daily_demand = forecast_demand / 7.0
        days_to_stockout = current_inv / daily_demand if daily_demand > 0 else 999.0

        annual = annual_demand_from_weekly(series["consumption"])
        unit_cost = _latest_number(sku_data, "unit_cost", 0.0)
        usage_value = max(annual, 0.0) * max(unit_cost, 0.0)

        fill_rate = _latest_number(sku_data, "vendor_fill_rate", 0.0)

        results.append(
            {
                "sku": sku,
                "current_inventory": current_inv,
                "forecast_weekly_demand": forecast_demand,
                "avg_weekly_demand": round(max(avg_weekly, 0.0), 4),
                "weekly_demand_sd": round(max(weekly_std, 0.0), 4),
                "lead_time_days": lead_time,
                "lead_time_weeks": round(lt_weeks, 4),
                "days_to_stockout": round(days_to_stockout, 1),
                "weeks_of_cover": cover["weeks_of_cover"],
                "stockout_risk": cover["stockout_risk"],
                "risk_reason": cover["risk_reason"],
                "annual_demand": round(annual, 2),
                "unit_cost": round(unit_cost, 2),
                "usage_value": round(usage_value, 2),
                "vendor_fill_rate": round(fill_rate, 2),
                "vendor_name": _latest_text(sku_data, "vendor_name", ""),
                "critical_item": _latest_text(sku_data, "critical", ""),
            }
        )

    summary = pd.DataFrame(results)
    if summary.empty:
        summary.attrs["unit_cost_source"] = unit_cost_source
        return summary

    summary = apply_service_level(summary, service_level)
    classified = classify_abc(summary[["sku", "usage_value"]])
    summary = summary.merge(
        classified[["sku", "abc_class", "usage_cum_pct"]],
        on="sku",
        how="left",
    )
    summary.attrs["unit_cost_source"] = unit_cost_source
    summary = sort_risk_table(summary)
    summary.attrs["unit_cost_source"] = unit_cost_source
    return summary


# ====================== FAST LLM REPORT GENERATION ======================
def generate_llm_report(full_summary):
    exception_summary = full_summary[
        full_summary["stockout_risk"].isin(["CRITICAL (Below ROP)", "High"])
    ]

    if exception_summary.empty:
        exception_summary = full_summary.head(10)

    summary_text = exception_summary.to_string(index=False)

    prompt = f"""You are a senior Supply Chain Inventory Management Specialist.
Analyze the following high-priority inventory exceptions (Reorder Points breached or high stockout risk).

Use ONLY this data:
{summary_text}

Write a clean Markdown report with these precise operational sections:
# Corporate Inventory Health & Optimization Report
## Executive Summary (Highlight operational health and cash flow efficiency)
## Critical Reorder Triggers (Items currently below their statistical Reorder Point)
## Stockout Risks & Supplier Vulnerabilities (Factoring in vendor fill rates)
## Prioritized Action Plan for Inventory Associates"""

    print("🤖 Processing telemetry exceptions with Gemma 2 (2B)...")
    start = time.time()
    try:
        response = _gemma_chat([{"role": "user", "content": prompt}])
        report = response["message"]["content"]
    except Exception as e:
        report = (
            f"# Operational Health Report\n\n"
            f"*Error connecting to Gemma 2 model: {str(e)}*\n\n"
            f"Forecasts and the risk board are still available without the model."
        )

    report_path = OUTPUT_DIR / "Inventory_Health_Risk_Report.md"
    if _write_text(report_path, report):
        print(f"✅ Report compiled in {time.time() - start:.1f} seconds → {report_path}")
    else:
        print(f"✅ Report ready in memory in {time.time() - start:.1f} seconds (disk write skipped).")
    return report


def generate_scenario_narrative(
    shock_df: pd.DataFrame,
    extra_lead_weeks: float,
    spike_pct: float,
    spike_weeks: float,
) -> dict:
    """Ask Gemma 2 for a short shock narrative. Fall back to the calculated summary.

    Returns source "gemma2:2b" or "rule". The rule-based text is always in
    `fallback` so the UI can show the numbers even when the model answers.
    """
    fallback = describe_shock(shock_df, extra_lead_weeks, spike_pct, spike_weeks)
    focus = shock_df.head(12)
    columns = [
        column
        for column in (
            "sku",
            "stockout_risk_before",
            "stockout_risk_after",
            "projected_stockout_week",
            "stockout_probability",
            "emergency_order_qty",
        )
        if column in focus.columns
    ]
    table = focus[columns].to_string(index=False) if len(focus) else "(no SKUs)"
    prompt = f"""You are a senior inventory planner. Write two short paragraphs about this supply shock.
Use ONLY the numbers below. Do not invent SKUs or quantities.
Name which SKUs get worse, the stockout probability, and the emergency order quantity.

Scenario: {extra_lead_weeks:.0f} extra weeks of supplier delay, {spike_pct:.0f}% demand spike for {spike_weeks:.0f} weeks.

Calculated summary:
{fallback}

Worst SKUs:
{table}
"""
    try:
        response = _gemma_chat([{"role": "user", "content": prompt}])
        narrative = (response.get("message") or {}).get("content") or ""
        narrative = narrative.strip()
        if not narrative:
            raise RuntimeError("Gemma 2 returned an empty narrative")
        return {"source": "gemma2:2b", "narrative": narrative, "fallback": fallback, "error": ""}
    except Exception as exc:
        return {
            "source": "rule",
            "narrative": fallback,
            "fallback": fallback,
            "error": str(exc),
        }


# ====================== COGNITIVE INTERACTIVE AGENT ======================
class SupplyChainAgent:
    def __init__(self, data: pd.DataFrame):
        self.df = data

    def query(self, question: str) -> str:
        q = question.lower()

        if "critical" in q or "below rop" in q or "reorder" in q:
            triggered = self.df[self.df["current_inventory"] <= self.df["reorder_point"]]
            if triggered.empty:
                return "🟢 All inventory levels are above their statistical reorder points."

            res = "⚠️ **Action Required: SKUs Breaching Reorder Point:**\n"
            for _, row in triggered.iterrows():
                res += (
                    f"- **{row['sku']}** ({row['vendor_name']}): "
                    f"Current Stock: {row['current_inventory']} | "
                    f"ROP: {row['reorder_point']} | "
                    f"Days Left: {row['days_to_stockout']}\n"
                )
            return res

        if "risk" in q or "high risk" in q:
            high_risk = self.df[self.df["stockout_risk"].str.contains("High|CRITICAL")]
            if high_risk.empty:
                return "🟢 No high stockout risks identified."
            res = "📋 **High & Critical Risk SKUs:**\n"
            for _, row in high_risk.iterrows():
                res += (
                    f"- SKU **{row['sku']}**: Status: `{row['stockout_risk']}` | "
                    f"Vendor Fill Rate: {row['vendor_fill_rate'] * 100}%\n"
                )
            return res

        if "sku" in q:
            words = q.split()
            for word in words:
                clean_word = word.strip(",.?!").upper()
                match = self.df[self.df["sku"].str.upper() == clean_word]
                if not match.empty:
                    row = match.iloc[0]
                    cover_line = ""
                    if "weeks_of_cover" in row.index and "risk_reason" in row.index:
                        cover_line = (
                            f"  • Weeks of cover: {format_weeks_of_cover(row['weeks_of_cover'])}"
                            f" | Risk: {row['stockout_risk']}\n"
                            f"  • {row['risk_reason']}\n"
                        )
                    level = row["service_level"] if "service_level" in row.index else ""
                    level_bit = f" | Service level: {level}" if level else ""
                    return (
                        f"📊 **SKU Profile: {row['sku']}** ({row['vendor_name']})\n"
                        f"  • Class: {row['abc_class']} | Critical: {row['critical_item']}\n"
                        f"  • Inventory Level: {row['current_inventory']} units\n"
                        f"  • Safety Stock: {row['safety_stock']} | ROP: {row['reorder_point']} units"
                        f"{level_bit}\n"
                        f"  • Days to Stockout: {row['days_to_stockout']} "
                        f"(Lead Time: {row['lead_time_days']} days)\n"
                        f"{cover_line}"
                    ).rstrip()
            return "🔍 Couldn't match the specific SKU ID. Try entering the precise SKU code."

        return (
            "💡 *Copilot Query Examples:*\n"
            "  - 'Which items are below reorder point?'\n"
            "  - 'Show high risk items'\n"
            "  - 'Status for SKU [SKU_ID]'"
        )


# ====================== PIPELINE (CLI + UI) ======================
def run_pipeline(
    sample_file: str = "week_2026-04-13_demo.csv",
    forecast_weeks: int = 8,
    generate_report: bool = True,
    service_level: str = DEFAULT_SERVICE_LEVEL,
):
    """Run the full inventory intelligence pipeline. Returns None on failure."""
    df = load_and_clean_weekly_data(sample_file)
    if df is None:
        return None

    forecast_df = forecast_and_stockout(
        df,
        forecast_weeks=forecast_weeks,
        service_level=service_level,
    )
    summary_path = OUTPUT_DIR.parent / "latest_summary.csv"
    if not _write_csv(forecast_df, summary_path):
        summary_path = None

    report = generate_llm_report(forecast_df) if generate_report else ""

    return {
        "raw_df": df,
        "full_summary": forecast_df,
        "report": report,
        "summary_path": summary_path,
        "unit_cost_source": forecast_df.attrs.get("unit_cost_source", "sample"),
        "service_level": service_level,
        "source_file": sample_file,
    }


# ====================== MAIN EXECUTION CONTEXT ======================
if __name__ == "__main__":
    start_time = time.time()
    print(
        f"🚀 Supply Chain Intelligence Pipeline Active - "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    result = run_pipeline("week_2026-04-13_demo.csv")

    if result is not None:
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"🎉 ENGINE EXECUTION COMPLETE in {total_time:.1f} seconds")
        print("=" * 60)

        print("\n🤖 Inventory Copilot Online! (Type 'exit' to log out)")
        agent = SupplyChainAgent(result["full_summary"])
        while True:
            try:
                user_query = input("\n💬 Copilot Command: ")
                if user_query.lower() in ["exit", "quit"]:
                    print("Logging out safely. Have a great shift! 📦")
                    break
                if user_query.strip():
                    print(agent.query(user_query))
            except (KeyboardInterrupt, EOFError):
                break
