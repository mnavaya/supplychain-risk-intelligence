# Project Report

## Local LLM-Powered Supply Chain Inventory Forecasting & Risk Intelligence Model  
### “Inventory Nook” — Graduate AI / Agentic Systems — SPS Corp 2026

**Course context:** SPS Corp 2026 — Graduate AI / Agentic Systems  
**Repository:** https://github.com/mnavaya/supplychain-risk-intelligence  
**Live web application:** https://supplychain-risk-intelligence-ftw2mpf4jdpkqthyaeyloc.streamlit.app/  
**Date:** August 21, 2026  

---

## 1. Executive Summary

This project delivers an end-to-end **hybrid AI system** for weekly inventory risk intelligence, designed for Open Healthcare–style testing operations supply chains. The system combines:

1. **Classical statistical forecasting and inventory science** (Holt–Winters exponential smoothing, safety stock, reorder points, ABC classification, stockout risk scoring), and  
2. An optional **local Large Language Model (LLM)** layer (Gemma 2 2B via Ollama) that turns exception rows into a structured Markdown operations report, and  
3. An interactive **rule-based inventory copilot** plus a polished **Streamlit web UI** (“Inventory Nook”) deployed publicly on Streamlit Community Cloud.

The pipeline is designed to run **fully locally** on a standard laptop (CPU-only), with no requirement to send inventory data to a paid cloud LLM. A public demo URL is available for review of the forecasting, risk board, metrics, and copilot. The narrative LLM report is generated when Ollama is running locally.

---

## 2. Problem Statement

Supply chain associates must review weekly inventory files and decide which SKUs need reorder action. Manual review is slow and easy to miss when:

- Demand is changing week to week,  
- Lead times and vendor fill rates vary,  
- Critical items must be prioritized.

**Goal:** Build a reproducible, privacy-preserving tool that:

- Ingests a weekly CSV,  
- Forecasts demand and computes operational thresholds,  
- Flags stockout / reorder risk,  
- Produces an actionable report and supports interactive Q&A.

---

## 3. Objectives and Success Criteria

| Objective | Status |
|---|---|
| Demand forecasting engine (Exponential Smoothing / Holt–Winters) | Completed |
| Safety stock & reorder-point (ROP) calculations | Completed |
| ABC classification and stockout risk tiers | Completed |
| Local LLM narrative report (Ollama) | Completed (local) |
| Interactive agent / copilot | Completed (CLI + UI) |
| Web UI for non-technical users | Completed (Streamlit) |
| Public deployment for demonstration | Completed (Streamlit Cloud) |
| Source control on GitHub | Completed |
| Prefer local/offline execution; synthetic data only | Completed |

---

## 4. System Architecture

### 4.1 High-level pipeline

```
Weekly CSV
    → Data cleaning & feature engineering
    → Holt–Winters demand forecast (per SKU)
    → Safety stock + Reorder Point + ABC + Risk score
    → latest_summary.csv
    → (Optional) Local LLM report → Inventory_Health_Risk_Report.md
    → Interactive Copilot (CLI / Streamlit)
```

### 4.2 Hybrid design (Minimal Tier)

| Layer | Technology | Role |
|---|---|---|
| **Traditional ML / Ops Research** | `pandas`, `numpy`, `statsmodels` (`ExponentialSmoothing`) | Forecast demand; compute safety stock, ROP, days-to-stockout, ABC class, risk labels |
| **Risk Intelligence (LLM)** | Ollama + **Gemma 2 (2B)** | Convert high-priority exception rows into a structured Markdown report |
| **Agent / Copilot** | Rule-based `SupplyChainAgent` in `main.py` | Answer questions such as “below ROP?”, “high risk?”, “status for SKU X” without hallucination of numbers |
| **Presentation** | Streamlit (`ui_app.py`) | Dark, minimal “Inventory Nook” UI; file upload; metrics; tabs; chat |

### 4.3 Key repository files

| File | Purpose |
|---|---|
| `main.py` | Core pipeline: load → forecast → report → CLI copilot; exports `run_pipeline()` |
| `ui_app.py` | Streamlit frontend (“Inventory Nook”) |
| `agents.py` | Compatibility shim re-exporting the agent from `main.py` |
| `data/weekly/` | Weekly input CSVs (demo files included) |
| `outputs/latest_summary.csv` | Numerical SKU summary used by UI / downstream steps |
| `outputs/reports/Inventory_Health_Risk_Report.md` | LLM-generated narrative report |
| `requirements.txt` | Python dependencies |
| `README.md` | Setup, run, and deploy instructions |
| `validation.py` | Supporting validation / backtest utilities |

---

## 5. Methods

### 5.1 Data ingestion and cleaning

- Read weekly CSV from `data/weekly/`.  
- Parse `week` as datetime; drop incomplete rows missing `sku`, `inventory_level`, or `consumption`.  
- Sort by SKU and week.  
- Engineer:
  - `daily_consumption = consumption / 7`
  - `days_of_supply = inventory_level / daily_consumption`

### 5.2 Demand forecasting

For each SKU:

- If fewer than 3 historical points → use last observed consumption.  
- Else fit **Holt–Winters / Exponential Smoothing** with additive trend (`statsmodels`).  
- Forecast horizon defaults to **8 weeks** (configurable in the UI).  
- Forecast weekly demand is clipped at ≥ 0; on model failure, fall back to a 3-period rolling mean.

### 5.3 Safety stock and reorder point

Using a **95% service level** factor:

\[
Z = 1.645
\]

\[
\text{Safety Stock} = Z \times \sigma_{\text{weekly}} \times \sqrt{L/7}
\]

\[
\text{ROP} = (\text{daily demand} \times \text{lead time days}) + \text{Safety Stock}
\]

Where \(L\) is lead time in days and \(\sigma_{\text{weekly}}\) is the SKU consumption standard deviation.

### 5.4 Risk scoring and ABC analysis

**Stockout risk tiers:**

- **CRITICAL (Below ROP)** if current inventory ≤ ROP  
- **High** if days-to-stockout < lead time + 7  
- **Medium** if days-to-stockout < 30  
- **Low** otherwise  

**ABC class** is based on consumption-volume quantiles (top ~10% → A, next band → B, remainder → C).

### 5.5 Local LLM report generation

- Only **exception rows** (CRITICAL / High) are passed to the LLM to keep inference fast on CPU.  
- Model: **`gemma2:2b`** through Ollama.  
- Prompt instructs the model to use **only** the provided table and to write fixed sections (executive summary, reorder triggers, supplier vulnerabilities, prioritized actions).  
- If Ollama is unavailable, the pipeline still saves numerical outputs and returns a clear error note (important for Streamlit Cloud, where local Ollama is not present).

### 5.6 Interactive inventory copilot

The `SupplyChainAgent` answers:

- Items below reorder point  
- High / critical risk SKUs  
- Per-SKU profile (class, inventory, safety stock, ROP, days to stockout, lead time)

This agent is **deterministic / rule-based** over the forecast summary table (fast, auditable, and suitable for demos without an LLM).

---

## 6. User Interface — Inventory Nook

### 6.1 Design goals

- Professional, dark, minimal aesthetic suitable for demonstration  
- High-contrast controls (especially CSV upload and **Run analysis**)  
- Clear note that LLM report requires local Ollama  

### 6.2 Features

- Select weekly CSV or **upload** a new CSV  
- Forecast horizon slider (2–16 weeks)  
- Optional **Generate LLM report** toggle (default off on cloud)  
- KPI metrics: SKUs watched, below ROP, high/critical risk, average days to stockout  
- Tabs: **Risk board**, **Raw weeks**, **Written report**, **Copilot**  
- Footer credit and sidebar branding  

### 6.3 Deployment

- Source of truth: GitHub `main` branch  
- Hosted on **Streamlit Community Cloud**  
- Live URL: https://supplychain-risk-intelligence-ftw2mpf4jdpkqthyaeyloc.streamlit.app/  

**Cloud vs local:**

| Capability | Streamlit Cloud | Local laptop |
|---|---|---|
| Forecast + risk metrics | Yes | Yes |
| Risk board / tables | Yes | Yes |
| Copilot | Yes | Yes |
| Gemma 2 written report | No (no Ollama on cloud) | Yes (with Ollama running) |

---

## 7. How to Run (Reproduction)

### 7.1 Environment

```bash
pip install -r requirements.txt
```

### 7.2 Optional local LLM

```bash
# Install Ollama from https://ollama.com/download, then:
ollama pull gemma2:2b
```

### 7.3 CLI

```bash
python main.py
```

Produces:

- `outputs/latest_summary.csv`  
- `outputs/reports/Inventory_Health_Risk_Report.md` (if Ollama is available)  
- Interactive terminal copilot  

### 7.4 Web UI

```bash
streamlit run ui_app.py
```

Or open the deployed Cloud URL above.

---

## 8. Results and Demonstration Narrative

### 8.1 Numerical intelligence layer

On each run, the system produces a per-SKU summary including:

- ABC class  
- Current inventory  
- Forecast weekly demand  
- Safety stock and reorder point  
- Days to stockout  
- Lead time, vendor fill rate, vendor name, critical flag  
- Stockout risk label  

These results drive the UI metrics and risk board.

### 8.2 LLM report (local)

When Ollama + `gemma2:2b` are available, the system writes a Markdown report focused on exceptions, intended for inventory associates. Example structure:

- Corporate / Inventory Health title  
- Executive summary  
- Critical reorder triggers  
- Stockout risks & supplier vulnerabilities  
- Prioritized action plan  

### 8.3 Privacy and data

- Synthetic / demo weekly CSVs only  
- No PHI or clinical patient data  
- Local LLM path keeps data on-device  

### 8.4 Performance posture

The design targets practical laptop use: small local model, exception-only prompting, and a lightweight Streamlit front end. Classical forecasting completes quickly; LLM time dominates when the report toggle is enabled.

---

## 9. Limitations and Honest Assessment

1. **LLM factuality:** Small local models can still misstate numbers; prompting restricts inputs to the exception table, but human review remains required (disclaimer principle).  
2. **History depth:** Synthetic weekly history limits deep forecast validation (e.g., long MAPE backtests).  
3. **Cloud LLM gap:** Streamlit Cloud cannot reach a laptop Ollama instance without additional remote hosting or a cloud API (intentionally avoided to keep the solution free/local).  
4. **Copilot scope:** The interactive agent is rule-based (reliable for demo Q&A) rather than a free-form LLM chat over arbitrary English.

---

## 10. What Was Delivered (Checklist)

- [x] End-to-end Python pipeline in `main.py`  
- [x] Forecasting + safety stock + ROP + ABC + risk tiers  
- [x] Local Gemma 2 report generation via Ollama  
- [x] CLI interactive copilot  
- [x] Streamlit “Inventory Nook” UI with upload, metrics, tabs, chat  
- [x] GitHub repository with documentation  
- [x] Public Streamlit Cloud deployment  
- [x] In-app note explaining local LLM requirements  

---

## 11. Conclusion

This project demonstrates a practical **agentic / hybrid AI** pattern for supply chain operations: classical models provide trustworthy numbers; a local LLM optionally narrates exceptions; a UI and copilot make the system usable for demonstration and weekly workflows. The solution aligns with Minimal Tier goals—local execution, structured outputs, and human-in-the-loop decision making—while adding a polished public demo.

---

## 12. Links for Review

| Resource | URL / Path |
|---|---|
| GitHub repository | https://github.com/mnavaya/supplychain-risk-intelligence |
| Live Inventory Nook UI | https://supplychain-risk-intelligence-ftw2mpf4jdpkqthyaeyloc.streamlit.app/ |
| Core pipeline | `main.py` |
| Web UI | `ui_app.py` |
| Demo data | `data/weekly/week_2026-04-13_demo.csv` |
| Sample report output | `outputs/reports/Inventory_Health_Risk_Report.md` |

---

*All rights reserved — Morgan 2026*
