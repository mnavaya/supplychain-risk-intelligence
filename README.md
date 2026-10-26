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

## Deploy the frontend (Streamlit Community Cloud)

The public demo is `streamlit run ui_app.py`. Nothing else has to be started. The bundled sample file is `data/weekly/week_2026-04-13_demo.csv` (50 SKUs). The app does not fit a model until you click **Run analysis**, so the first page is ready as soon as Cloud finishes installing dependencies. `requirements.txt` pins the versions this repo was tested with, and `runtime.txt` asks for Python 3.12. No secrets and no API keys.

Exact steps:

1. Merge this branch to `main` (or deploy the preview branch `cursor/inventory-insights-3fc4` before merge).
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with the GitHub account that can see `mnavaya/supplychain-risk-intelligence`.
3. **Create app**.
4. Repository: `mnavaya/supplychain-risk-intelligence`.
5. Branch: `main` after merge, or `cursor/inventory-insights-3fc4` for a preview.
6. Main file path: `ui_app.py`.
7. Advanced settings: Python **3.12**. Leave secrets empty.
8. Deploy. The live URL is `https://<app-name>.streamlit.app`.

On the live app:

- Weekly CSV defaults to `week_2026-04-13_demo.csv`. Click **Run analysis**. Leave **Generate LLM report** off.
- Risk board, service levels, forecast accuracy, ABC, supply shock (sliders, charts, Monte Carlo, emergency orders), and the Copilot all run in the browser. No local model is required.
- **Write scenario narrative**, the written report, and free-form Copilot questions call Gemma 2 only if Ollama is on the same machine. On Community Cloud the call stops within a few seconds. The Copilot then shows the original keyword answer with a **rule-based** badge. Nothing else waits on the model.
- The file picker lists inventory CSVs only. Unrelated files that happen to sit in `data/weekly/` are skipped.

## What it does

- Cleans weekly inventory CSVs from `data/weekly/`
- Forecasts demand (Holt-Winters), computes safety stock & reorder points
- ABC-classifies SKUs and flags stockout risk
- Writes `outputs/latest_summary.csv` and `outputs/reports/Inventory_Health_Risk_Report.md`
- Answers free-form copilot questions from a small slice of the forecast. Local Gemma 2 is used only when every number in the reply checks out; otherwise the original keyword answers are used.

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

### Grounded copilot questions

The Copilot tab accepts a normal question, such as "Which A-class items run out before their next delivery?"

- The question is matched to forecast rows by SKU code, ABC class (A-class, class B), or risk words (run out, stockout, high risk, reorder, below ROP). Those filters combine. If the question names none of them, the 15 highest-risk rows are used.
- Only that compact table is sent to local Gemma 2 (`gemma2:2b`) through the same 8-second helper as the shock narrative. The prompt tells the model to answer from the table and to name the SKUs it uses.
- A grounding check reads every number in the reply. The number has to appear in the supplied rows, or be a trivial derivation: rounding to a whole number or one or two decimals, a fill rate written as a percent, a row count, a column total, or a within-row gap such as on-hand minus reorder point. A reply that names a SKU outside that table, or says a Low or Medium SKU runs out, is rejected too. A reply that passes is marked **grounded**.
- If any number does not check out, Ollama is not running, or the call fails, the original keyword answer is shown instead and marked **rule-based**. The hosted Streamlit demo has no Ollama, so it stays on that path.
- The Copilot tab shows the answer, the badge, and a collapsible **Rows used** table. The terminal copilot prints the same label after the answer.

### Forecast check, service levels, explainable risk, ABC summary

- **Forecast accuracy check.** On the Forecast accuracy tab, pick a frequently sold SKU with at least eight weeks of history. The app holds out the last four weeks, fits Holt-Winters (the same additive-trend exponential smoothing used by the pipeline) on earlier weeks only, and compares mean absolute error with a naive baseline that repeats the previous week's demand. The table shows actuals, both forecasts, and MAE, plus a line saying which one won. Holdout weeks are never passed into the model.
- **Service-level safety stock.** The sidebar offers 90% (Z=1.28), 95% (Z=1.65), and 99% (Z=2.33). Safety stock = Z × weekly demand standard deviation × √(lead time in weeks). Reorder point = average weekly demand × lead time in weeks + safety stock. The risk board shows the selected level next to safety stock and reorder point, and a three-row table shows all three levels together. A higher service level raises both values when demand varies.
- **Explainable stockout alerts.** Weeks of cover = on-hand stock ÷ forecast weekly demand, compared with supplier lead time in weeks. High means stock runs out before replenishment arrives, Medium means cover is within one week of the lead time, and Low means cover is comfortable. The risk table adds Weeks of Cover and Risk Reason, sorts high risk first, and treats zero forecast demand as no drawdown instead of dividing by zero.
- **ABC summary.** Usage value = annual demand × unit cost. SKUs are sorted descending, A is the first 80% of usage value, B the next 15%, and C the rest. The ABC tab summarizes each class with SKU count, percent of SKUs, total usage value, and percent of usage value. Those percent columns sum to about 100%.

### What-if supply shock simulator

The Supply shock tab is the planning sandbox. The view remembers which tab is open, so moving a slider stays on this chart. Three sliders set a supplier delay (extra weeks), a demand spike (percent), and how many weeks the spike lasts. Either shock can be zero, or both can be on at once. Every SKU is resimulated with the same weeks-of-cover rules and the same reorder-point formula as the risk board.

- SKUs that move into a worse risk band are listed first, with the risk before and after.
- Projected stockout week is when on-hand stock hits zero on the shocked demand path. A spike only lifts demand for its own window, then demand returns to the forecast.
- A before/after bar chart and heatmap show how the High / Medium / Low mix shifts.
- Emergency order quantity is the gap up to the shocked reorder point: safety stock uses the longer lead time, and the demand rate inside the reorder point is the average over that window (spiked weeks plus normal weeks).
- Stockout probability is a 500-run Monte Carlo. Each week is a draw from a normal distribution centered on the shocked forecast, with the SKU's historical weekly standard deviation, and floored at zero. A run counts as a stockout when demand uses up on-hand stock before the delayed replenishment arrives.
- **Write scenario narrative** asks local Gemma 2 (`gemma2:2b`) for two short paragraphs. If Ollama is not running, the call times out in a few seconds, the tab keeps the calculated summary, and it says the model was unavailable.
- **Public demo.** `requirements.txt`, `runtime.txt` (`python-3.12`), and `.streamlit/config.toml` are set for Streamlit Community Cloud. The entry point is `ui_app.py`, the sample file ships in the repo, and a missing Gemma model does not block startup or the shock simulator. See **Deploy the frontend** above.
