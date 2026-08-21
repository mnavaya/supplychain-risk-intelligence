# Supply Chain Inventory Forecasting & Risk Intelligence Model

**Local LLM-Powered Hybrid System**  
**Graduate AI / Agentic Systems — SPS Corp 2026**

## Overview

A lightweight, fully **local** pipeline designed for Open Healthcare’s testing operations supply chain.

The system ingests weekly CSV files containing inventory levels, consumption, lead times, vendor fill rates, and SKU metadata. It forecasts demand, computes safety stock and reorder points, scores stockout risk, and can generate an actionable **Inventory Health & Risk Report** with a local LLM.

## Setup

```bash
pip install -r requirements.txt
```

For LLM reports, install [Ollama](https://ollama.com) and pull:

```bash
ollama pull gemma2:2b
```

## Run

**CLI pipeline + terminal copilot**

```bash
python main.py
```

**Web UI (Inventory Nook)**

```bash
streamlit run ui_app.py
```

Legacy paths still work: `python src/main.py` and `streamlit run src/ui_app.py`.

## Deploy the frontend (public URL)

1. Push this repo to GitHub (`mnavaya/supplychain-risk-intelligence`).
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. **New app** → pick this repo → **Main file path:** `ui_app.py` → Deploy.
4. Your live URL will look like: `https://<app-name>.streamlit.app`

Notes for the cloud app:
- Forecasts, risk board, and the rule-based **Copilot** work without Ollama.
- The LLM written report needs a local Ollama instance, so turn **Generate LLM report** off in the sidebar on Streamlit Cloud (or leave it on — you’ll get a friendly error in the report tab).

## What it does

- Cleans weekly inventory CSVs from `data/weekly/`
- Forecasts demand (Holt-Winters), computes safety stock & reorder points
- ABC-classifies SKUs and flags stockout risk
- Writes `outputs/latest_summary.csv` and `outputs/reports/Inventory_Health_Risk_Report.md`
- Chat with a rule-based inventory copilot in the CLI or UI

## Demo data

`data/weekly/week_2026-04-13_demo.csv` is included for a quick walkthrough of risk flags and the copilot.
