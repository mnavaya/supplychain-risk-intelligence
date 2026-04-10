# Supply Chain Inventory Forecasting & Risk Intelligence Model

**Local LLM-Powered Hybrid System**  
**Graduate AI / Agentic Systems — SPS Corp 2026**  
**Team:** Uyen, Christabel, Morgan

## Overview

A lightweight, fully **local** pipeline designed for Open Healthcare’s testing operations supply chain. 

The system ingests weekly CSV files containing inventory levels, consumption, lead times, vendor fill rates, and SKU metadata. It then uses traditional ML for forecasting and anomaly detection, combined with a local LLM (Llama 3.2 3B via Ollama) as the **Risk Intelligence Layer**, to generate clear, actionable **Inventory Health & Risk Reports**.

Fully offline, runs on standard CPU-only laptops (Intel i7 + 16 GB RAM), and completes in under **90 seconds**.

## Key Features

- Demand forecasting using Exponential Smoothing
- Stockout prediction and risk scoring
- Statistical anomaly detection (low fill rate, lead time spikes, consumption anomalies)
- Inventory health KPIs (days of supply, turnover proxy, slow-moving flags)
- Professional Markdown report with Executive Summary, Key Findings, Risks & Anomalies, Stockout Predictions, and **Prioritized Recommended Actions** with confidence scores
- Clear disclaimer ensuring human-in-the-loop decision making

## Demo Scenario

The included `week_2026-04-13_demo.csv` simulates a real-world issue:  
→ Vendor A fill rate drops sharply to **0.55**  
→ Lead time spikes to **14 days** on a critical SKU  

The system correctly identifies the risk and provides prioritized corrective actions.

## Setup & Run (Fully Local)

1. **Install Ollama** and pull the model:
   ```powershell
   ollama pull llama3.2:3b
