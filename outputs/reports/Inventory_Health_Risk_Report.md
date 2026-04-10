# Inventory Health & Risk Report

## Executive Summary
The current inventory levels for most SKUs are below optimal thresholds, indicating potential stockout risks. Specifically, TEST-SKU-003 has a high risk level due to its low vendor fill rate and critical fill rate anomaly.

## Key Findings
* Most SKUs have relatively low forecasted weekly demand compared to the available inventory.
* The days_to_stockout metric is high for all tested SKUs, indicating potential supply chain risks.
* The slow_moving metric shows no anomalies in most cases.

## Identified Risks & Anomalities
* TEST-SKU-003 has a critical fill rate anomaly (0.36) which is below the 0.80 threshold, indicating that it's likely to experience stockouts if not addressed.
* Vendor A's fill rates for both TEST-SKU-001 and TEST-SKU-003 are above 0.90, indicating potential reliability issues.

## Stockout Predictions
* TEST-SKU-001: 48.6 days_to_stockout (Low risk)
* TEST-SKU-002: 62.6 days_to_stockout (Low risk)
* TEST-SKU-003: 19.6 days_to_stockout (High risk)

## Prioritized Recommended Actions
1. Implement stock replenishment strategies for TEST-SKU-001 and TEST-SKU-002 to reduce their high days_to_stockout values.
2. Investigate the reliability of Vendor A's supply chain, focusing on TEST-SKU-001 and TEST-SKU-003, to address the critical fill rate anomaly.
3. Develop a plan to improve Vendor B's performance, addressing the low vendor fill rate (0.98) for TEST-SKU-002.

## Disclaimer
This is an AI-generated recommendation. The supply chain specialist makes all final decisions.