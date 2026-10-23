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

`data/weekly/week_2026-04-13_demo.csv` is included for a quick walkthrough of risk flags and the copilot. It has on-hand stock (`inventory_level`) and supplier lead time (`lead_time_days`) but no unit cost. When `unit_cost` is missing, the app fills a documented sample price so ABC still runs:

| Category | Sample unit cost (USD) |
| --- | --- |
| Reagent | 48.00 |
| Consumable | 6.50 |
| Equipment | 125.00 |
| Chemical | 22.00 |
| Anything else | 15.00 |

Each SKU then gets a stable adjustment between 0.85× and 1.25× based on its id, so two reagents are not priced identically. These are planning samples, not vendor prices. `data/weekly/week_2026-04-06.csv` already includes `unit_cost`, and those values are used as given. If lead time is ever missing, the app assumes 7 days.

Annual demand is the sum of the latest 52 weekly consumption values, or average weekly consumption × 52 when a SKU has a shorter history.

## Tests

```bash
python -m unittest discover -s tests -t .
```

## Changelog

### Forecast check, service levels, explainable risk, ABC summary

- **Forecast accuracy check.** On the Forecast accuracy tab, pick a frequently sold SKU with at least eight weeks of history. The app holds out the last four weeks, fits Holt-Winters (the same additive-trend exponential smoothing used by the pipeline) on earlier weeks only, and compares mean absolute error with a naive baseline that repeats the previous week's demand. The table shows actuals, both forecasts, and MAE, plus a line saying which one won. Holdout weeks are never passed into the model.
- **Service-level safety stock.** The sidebar offers 90% (Z=1.28), 95% (Z=1.65), and 99% (Z=2.33). Safety stock = Z × weekly demand standard deviation × √(lead time in weeks). Reorder point = average weekly demand × lead time in weeks + safety stock. The risk board shows the selected level next to safety stock and reorder point, and a three-row table shows all three levels together. A higher service level raises both values when demand varies.
- **Explainable stockout alerts.** Weeks of cover = on-hand stock ÷ forecast weekly demand, compared with supplier lead time in weeks. High means stock runs out before replenishment arrives, Medium means cover is within one week of the lead time, and Low means cover is comfortable. The risk table adds Weeks of Cover and Risk Reason, sorts high risk first, and treats zero forecast demand as no drawdown instead of dividing by zero.
- **ABC summary.** Usage value = annual demand × unit cost. SKUs are sorted descending, A is the first 80% of usage value, B the next 15%, and C the rest. The ABC tab summarizes each class with SKU count, percent of SKUs, total usage value, and percent of usage value. Those percent columns sum to about 100%.

### What-if supply shock simulator

The Supply shock tab is the planning sandbox. Three sliders set a supplier delay (extra weeks), a demand spike (percent), and how many weeks the spike lasts. Either shock can be zero, or both can be on at once. Every SKU is resimulated with the same weeks-of-cover rules and the same reorder-point formula as the risk board.

- SKUs that move into a worse risk band are listed first, with the risk before and after.
- Projected stockout week is when on-hand stock hits zero on the shocked demand path. A spike only lifts demand for its own window, then demand returns to the forecast.
- A before/after bar chart and heatmap show how the High / Medium / Low mix shifts.
- Emergency order quantity is the gap up to the shocked reorder point: safety stock uses the longer lead time, and the demand rate inside the reorder point is the average over that window (spiked weeks plus normal weeks).
- Stockout probability is a 500-run Monte Carlo. Each week is a draw from a normal distribution centered on the shocked forecast, with the SKU's historical weekly standard deviation, and floored at zero. A run counts as a stockout when demand uses up on-hand stock before the delayed replenishment arrives.
- **Write scenario narrative** asks local Gemma 2 (`gemma2:2b`) for two short paragraphs. If Ollama is not running, the tab keeps the calculated summary and says the model was unavailable.
