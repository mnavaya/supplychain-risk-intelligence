# Project Summary & Postmortem

**Local LLM-Powered Supply Chain Inventory Forecasting & Risk Intelligence Model**  
**Graduate AI / Agentic Systems — SPS Corp 2026**  
**Students:** Uyen, Christabel, Morgan  
**Date:** April 10, 2026

## Project Overview
We developed a lightweight, fully local hybrid system for Open Healthcare’s testing operations supply chain. The tool ingests weekly CSV files and generates an actionable **Inventory Health & Risk Report** using a local LLM as the Risk Intelligence Layer.

The implementation strictly followed the **Minimal Tier** architecture specified in the project document.

## Architecture
- **Traditional ML Layer**: Exponential Smoothing (statsmodels) for demand forecasting, stockout prediction, and statistical anomaly detection (fill rate, lead time, consumption).
- **Risk Intelligence Layer**: Llama 3.2 3B via Ollama for narrative insights, root-cause explanations, prioritized recommendations, and confidence scoring.
- **Pipeline**: CSV → Cleaning → Forecasting → Anomalies/KPIs → LLM Prompt → Markdown Report.

## Key Results
- **Runtime**: 84–87 seconds on Intel i7 + 16 GB RAM CPU-only hardware → Well under 3-minute target
- **Locality & Privacy**: 100% offline, synthetic data only, no PHI/clinical data
- **Report Quality**: Professional Markdown with Executive Summary, Key Findings, Risks & Anomalies, Stockout Predictions, Prioritized Actions, and clear disclaimer
- **Demo**: Successfully detects vendor fill-rate drops (e.g., 0.55) and lead-time spikes on critical SKUs

## Postmortem

**What Went Well**
- The hybrid ML + local LLM approach worked reliably within our hardware and time constraints (~10 hours/week).
- Ollama with the 3B model proved fast enough for practical weekly use.
- Strict structured prompting helped produce usable reports.
- Incremental development (data → forecast → anomalies → LLM) kept the project manageable for beginners.

**Challenges & Limitations**
- Small 3B model occasionally hallucinates minor numbers or averages (mitigated with tight prompts but not eliminated).
- Limited historical weeks in synthetic data restricted full MAPE validation.
- Early Git and CSV formatting issues (resolved).

**Lessons Learned**
- Clear, restrictive prompts are essential when working with small local LLMs.
- Starting simple with statistical methods before adding LLM intelligence is very effective.
- Importance of proper file handling and virtual environments for reproducible pipelines.

## Success Criteria Alignment
- Forecast engine implemented (Exponential Smoothing)
- Anomaly detection and KPIs working
- Full pipeline completes in < 3 minutes
- Local/offline execution achieved
- LLM-generated recommendations with confidence scores and disclaimer included

## Conclusion
We successfully delivered a functional, local LLM-powered supply chain risk intelligence prototype that transforms raw weekly data into clear, actionable insights for the Supply Chain Specialist. 

This project gave us valuable hands-on experience with hybrid AI systems, prompt engineering, and building end-to-end local pipelines.

---

**Repository**: https://github.com/mnavaya/supplychain-risk-intelligence
