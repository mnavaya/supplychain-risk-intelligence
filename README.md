# Supply Chain Inventory Forecasting & Risk Intelligence Model

**Local LLM-Powered Hybrid System**  
**Graduate AI / Agentic Systems — SPS Corp 2026**  
**Team:** Uyen, Christabel, Morgan

## Overview
A lightweight, fully local pipeline that processes weekly inventory CSVs for healthcare testing operations. 

It combines **traditional ML** (Exponential Smoothing via statsmodels) for forecasting and anomaly detection with a **local LLM** (Llama 3.2 3B via Ollama) as the Risk Intelligence Layer to generate clear, actionable **Inventory Health & Risk Reports**.

**Key Achievements:**
- Fully offline execution on CPU-only hardware (Intel i7 + 16 GB RAM)
- End-to-end runtime: **~85 seconds** (well under 3-minute target)
- Professional Markdown reports with prioritized recommendations and confidence scores
- Human-in-the-loop design with clear disclaimers

## Setup Instructions

1. Install Ollama and pull the model:
   ```powershell
   ollama pull llama3.2:3b
