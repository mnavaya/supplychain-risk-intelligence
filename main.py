import pandas as pd
import numpy as np
from pathlib import Path
import ollama
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import time
import datetime

warnings.filterwarnings("ignore")

# Directory setup
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "weekly"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Service level factor: Z-Score for 95% protection against stockouts
Z_SCORE = 1.645


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


# ====================== ADVANCED FORECASTING & METRICS ======================
def forecast_and_stockout(df, forecast_weeks: int = 8):
    results = []

    demand_std = df.groupby("sku")["consumption"].std().fillna(0)
    total_volume = df.groupby("sku")["consumption"].sum()
    volume_th = total_volume.quantile([0.7, 0.9])

    for sku in df["sku"].unique():
        sku_data = df[df["sku"] == sku].copy().sort_values("week")

        if len(sku_data) < 3:
            forecast_demand = sku_data["consumption"].iloc[-1] if not sku_data.empty else 0
        else:
            try:
                model = ExponentialSmoothing(
                    sku_data["consumption"], trend="add", seasonal=None
                )
                fit = model.fit()
                forecast_demand = max(0, fit.forecast(forecast_weeks).mean())
            except Exception:
                forecast_demand = (
                    sku_data["consumption"].rolling(3, min_periods=1).mean().iloc[-1]
                )

        current_inv = sku_data["inventory_level"].iloc[-1]
        weekly_std = demand_std.get(sku, 0)
        lead_time = int(sku_data["lead_time_days"].iloc[-1])

        lt_weeks = lead_time / 7.0
        safety_stock = Z_SCORE * (weekly_std * np.sqrt(lt_weeks))

        daily_demand = forecast_demand / 7.0
        lead_time_demand = daily_demand * lead_time
        reorder_point = lead_time_demand + safety_stock

        days_to_stockout = current_inv / daily_demand if daily_demand > 0 else 999

        if current_inv <= reorder_point:
            stockout_risk = "CRITICAL (Below ROP)"
        elif days_to_stockout < (lead_time + 7):
            stockout_risk = "High"
        elif days_to_stockout < 30:
            stockout_risk = "Medium"
        else:
            stockout_risk = "Low"

        sku_vol = total_volume.get(sku, 0)
        abc_cat = (
            "A"
            if sku_vol >= volume_th.iloc[1]
            else "B"
            if sku_vol >= volume_th.iloc[0]
            else "C"
        )

        results.append(
            {
                "sku": sku,
                "abc_class": abc_cat,
                "current_inventory": round(current_inv, 1),
                "forecast_weekly_demand": round(forecast_demand, 1),
                "safety_stock": round(safety_stock, 1),
                "reorder_point": round(reorder_point, 1),
                "days_to_stockout": round(days_to_stockout, 1),
                "lead_time_days": lead_time,
                "stockout_risk": stockout_risk,
                "vendor_fill_rate": round(sku_data["vendor_fill_rate"].iloc[-1], 2),
                "vendor_name": sku_data["vendor_name"].iloc[-1],
                "critical_item": sku_data["critical"].iloc[-1],
            }
        )

    return pd.DataFrame(results)


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
        response = ollama.chat(
            model="gemma2:2b", messages=[{"role": "user", "content": prompt}]
        )
        report = response["message"]["content"]
    except Exception as e:
        report = (
            f"# Operational Health Report\n\n"
            f"*Error connecting to Gemma 2 model: {str(e)}*\n\n"
            f"Data summary saved locally."
        )

    report_path = OUTPUT_DIR / "Inventory_Health_Risk_Report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ Report compiled in {time.time() - start:.1f} seconds → {report_path}")
    return report


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
                    return (
                        f"📊 **SKU Profile: {row['sku']}** ({row['vendor_name']})\n"
                        f"  • Class: {row['abc_class']} | Critical: {row['critical_item']}\n"
                        f"  • Inventory Level: {row['current_inventory']} units\n"
                        f"  • Safety Stock: {row['safety_stock']} | ROP: {row['reorder_point']} units\n"
                        f"  • Days to Stockout: {row['days_to_stockout']} "
                        f"(Lead Time: {row['lead_time_days']} days)"
                    )
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
):
    """Run the full inventory intelligence pipeline. Returns None on failure."""
    df = load_and_clean_weekly_data(sample_file)
    if df is None:
        return None

    forecast_df = forecast_and_stockout(df, forecast_weeks=forecast_weeks)
    summary_path = OUTPUT_DIR.parent / "latest_summary.csv"
    forecast_df.to_csv(summary_path, index=False)

    report = generate_llm_report(forecast_df) if generate_report else ""

    return {
        "raw_df": df,
        "full_summary": forecast_df,
        "report": report,
        "summary_path": summary_path,
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
