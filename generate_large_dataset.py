import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

# ====================== CONFIG ======================
NUM_ROWS = 59898
NUM_SKUS = 50                     # Changed to 50 as you requested
START_DATE = datetime(2024, 1, 1)
OUTPUT_PATH = Path("data/weekly/week_2026-04-13_demo.csv")

# Create output directory
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"Generating {NUM_ROWS:,} rows with {NUM_SKUS} SKUs...")

# Generate SKUs
skus = [f"SKU-{str(i).zfill(5)}" for i in range(1, NUM_SKUS + 1)]

# Vendors and categories
vendors = ["VendorA", "VendorB", "VendorC", "VendorD"]
categories = ["Reagent", "Consumable", "Equipment", "Chemical"]
critical_flags = [True, False]

# Generate weekly dates
dates = []
current_date = START_DATE
while len(dates) < (NUM_ROWS // NUM_SKUS) + 20:
    dates.append(current_date.strftime("%Y-%m-%d"))
    current_date += timedelta(days=7)

# Generate data
np.random.seed(42)

data = []

for sku in skus:
    is_critical = np.random.choice(critical_flags, p=[0.35, 0.65])
    vendor = np.random.choice(vendors)
    category = np.random.choice(categories)
    
    base_consumption = np.random.randint(50, 800)
    
    for week in range(len(dates)):
        if len(data) >= NUM_ROWS:
            break
            
        consumption = int(base_consumption * (1 + 0.15 * np.sin(week / 8) + np.random.normal(0, 0.25)))
        consumption = max(10, consumption)
        
        inventory = int(consumption * np.random.uniform(3, 12))
        
        lead_time = max(2, min(21, int(np.random.normal(7, 4))))
        
        if is_critical and np.random.random() < 0.28:
            fill_rate = round(np.random.uniform(0.45, 0.82), 2)
        else:
            fill_rate = round(np.random.uniform(0.85, 0.99), 2)
        
        data.append({
            'week': dates[week % len(dates)],
            'sku': sku,
            'inventory_level': inventory,
            'consumption': consumption,
            'lead_time_days': lead_time,
            'vendor_fill_rate': fill_rate,
            'vendor_name': vendor,
            'category': category,
            'critical': 'Yes' if is_critical else 'No'
        })

df = pd.DataFrame(data)
df = df.head(NUM_ROWS)

df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Successfully generated {len(df):,} rows with {NUM_SKUS} SKUs")
print(f"📁 Saved to: {OUTPUT_PATH.absolute()}")
print(f"   Date range: {df['week'].min()} → {df['week'].max()}")
print(f"   File size: {os.path.getsize(OUTPUT_PATH) / (1024*1024):.2f} MB")