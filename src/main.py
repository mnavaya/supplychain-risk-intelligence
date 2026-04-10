import pandas as pd
from pathlib import Path
import ollama
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import time
import datetime

warnings.filterwarnings("ignore")

# ====================== PATHS ======================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "weekly"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean_weekly_data(filename: str):
    start = time.time()
    file_path = DATA_DIR / filename
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    print(f"✅ Loaded raw CSV with columns: {list(df.columns)}")
    
    # Ensure required columns exist
    required = ['week', 'sku', 'inventory_level', 'consumption', 'lead_time_days', 'vendor_fill_rate', 'vendor_name', 'critical']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        return None
    
    df['week'] = pd.to_datetime(df['week'], errors='coerce')
    df = df.dropna(subset=['sku', 'inventory_level', 'consumption'])
    
    df['daily_consumption'] = df['consumption'] / 7.0
    df['days_of_supply'] = df['inventory_level'] / df['daily_consumption'].replace(0, 1)
    
    print(f"✅ Data cleaned and ready ({len(df)} SKUs) in {time.time() - start:.2f} seconds")
    return df

def forecast_and_stockout(df, forecast_weeks=8):
    results = []
    for sku in df['sku'].unique():
        sku_data = df[df['sku'] == sku].copy()
        if len(sku_data) < 3:
            forecast_demand = sku_data['consumption'].iloc[-1] if not sku_data.empty else 0
        else:
            model = ExponentialSmoothing(sku_data['consumption'], trend='add', seasonal=None)
            fit = model.fit()
            forecast_demand = fit.forecast(forecast_weeks).mean()
        
        current_inv = sku_data['inventory_level'].iloc[-1]
        daily_demand = forecast_demand / 7.0
        days_to_stockout = current_inv / daily_demand if daily_demand > 0 else 999
        
        lead_time = int(sku_data['lead_time_days'].iloc[-1])
        stockout_risk = "High" if days_to_stockout < (lead_time + 7) else "Medium" if days_to_stockout < 30 else "Low"
        
        results.append({
            'sku': sku,
            'current_inventory': round(current_inv, 1),
            'forecast_weekly_demand': round(forecast_demand, 1),
            'days_to_stockout': round(days_to_stockout, 1),
            'lead_time_days': lead_time,
            'stockout_risk': stockout_risk,
            'vendor_fill_rate': round(sku_data['vendor_fill_rate'].iloc[-1], 2),
            'vendor_name': sku_data['vendor_name'].iloc[-1],
            'critical': sku_data['critical'].iloc[-1]
        })
    return pd.DataFrame(results)

def detect_anomalies_and_kpis(df, forecast_df):
    anomalies = []
    for _, row in forecast_df.iterrows():
        sku = row['sku']
        sku_data = df[df['sku'] == sku].iloc[-1]
        
        fill_rate_anomaly = "Yes" if row['vendor_fill_rate'] < 0.80 else "No"
        lead_time_anomaly = "Yes" if row['lead_time_days'] > 10 else "No"
        
        turnover_proxy = round(sku_data['consumption'] / (sku_data['inventory_level'] + 1), 2)
        slow_moving = "Yes" if row['days_to_stockout'] > 90 else "No"
        
        anomalies.append({
            'sku': sku,
            'fill_rate_anomaly': fill_rate_anomaly,
            'lead_time_anomaly': lead_time_anomaly,
            'inventory_turnover_proxy': turnover_proxy,
            'slow_moving': slow_moving
        })
    anomaly_df = pd.DataFrame(anomalies)
    return pd.merge(forecast_df, anomaly_df, on='sku')

def generate_llm_report(full_summary):
    summary_text = full_summary.to_string(index=False)
    
    prompt = f"""You are a senior Supply Chain Inventory Management Specialist at Open Healthcare.

Use ONLY this data:

{summary_text}

Strict instructions:
- Base all statements strictly on the table above.
- Do not invent totals, averages, or previous week data.
- Explain anomalies using the exact values (e.g., "fill rate 0.55 is below 0.80 threshold").
- Keep recommendations practical and specific to the data.

Write the report in this exact Markdown structure:

# Inventory Health & Risk Report

## Executive Summary
(2-3 short sentences highlighting overall health and the most urgent risk(s))

## Key Findings
(Bullet points with the most important facts)

## Identified Risks & Anomalies
(Clear explanations referencing specific SKUs and values)

## Stockout Predictions
(List each SKU with its days_to_stockout and risk level)

## Prioritized Recommended Actions
(Numbered list. Each item: specific action + reason from data + Confidence: 0.XX)

## Disclaimer
This is an AI-generated recommendation. The supply chain specialist makes all final decisions.

Start directly with the title. No extra text."""

    print("🤖 Generating report with local LLM...")
    start_time = time.time()
    
    try:
        response = ollama.chat(model='llama3.2:3b', messages=[{'role': 'user', 'content': prompt}])
        report = response['message']['content']
        
        report_path = OUTPUT_DIR / "Inventory_Health_Risk_Report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"✅ Report generated in {time.time() - start_time:.2f} seconds")
        return report
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return None

# ====================== MAIN ======================
if __name__ == "__main__":
    overall_start = time.time()
    print(f"🚀 Starting Supply Chain Risk Intelligence Pipeline - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    sample_file = "week_2026-04-13_demo.csv"   # <--- Make sure this matches your file name
    
    df = load_and_clean_weekly_data(sample_file)
    
    if df is not None:
        forecast_df = forecast_and_stockout(df)
        full_summary = detect_anomalies_and_kpis(df, forecast_df)
        
        print("\n📊 Numerical Summary Ready")
        report = generate_llm_report(full_summary)
        
        if report:
            total_time = time.time() - overall_start
            print("\n" + "="*70)
            print("🎉 PIPELINE COMPLETE")
            print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"Target: < 180 seconds → {'✅ PASSED' if total_time < 180 else '❌ OVER'}")
            print("="*70)
            print(f"Report saved: outputs/reports/Inventory_Health_Risk_Report.md")