"""Rebuild data/weekly/week_2026-04-13_demo.csv.

The first 50 SKUs are the original smooth series (NumPy seed 42, 59,898 rows).
That block is left alone. SKU-00051 and later are appended from fixed weekly
patterns so the file can be regenerated without reshuffling the original SKUs.

Patterns, on the same 1,217-week calendar as the long original series:

- Intermittent spare parts: a small steady quantity every few weeks, zeros
  in between (ADI at least 4, CV² near 0).
- Lumpy orders: long quiet stretches, then a large order whose size jumps
  around (ADI above 1.32 and CV² above 0.49).
- Erratic demand: something ships every week, but the size swings hard
  (ADI 1, CV² above 0.49).

Run: python generate_large_dataset.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_ROWS = 59898
BASELINE_SKUS = 50
START_DATE = datetime(2024, 1, 1)
OUTPUT_PATH = Path("data/weekly/week_2026-04-13_demo.csv")

COLUMNS = [
    "week",
    "sku",
    "inventory_level",
    "consumption",
    "lead_time_days",
    "vendor_fill_rate",
    "vendor_name",
    "category",
    "critical",
]


def baseline_week_count() -> int:
    """Weeks in the original calendar: 59,898 // 50 + 20."""
    return (BASELINE_ROWS // BASELINE_SKUS) + 20


def week_labels(count: int, start: datetime = START_DATE) -> list[str]:
    labels = []
    current = start
    for _ in range(count):
        labels.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=7)
    return labels


def build_baseline_frame(
    num_rows: int = BASELINE_ROWS,
    num_skus: int = BASELINE_SKUS,
) -> pd.DataFrame:
    """Original smooth demo rows. Matches the checked-in prefix byte for byte.

    Uses the global NumPy generator the file was first built with, then puts
    that generator back so importing this module does not change later draws.
    """
    skus = [f"SKU-{i:05d}" for i in range(1, num_skus + 1)]
    vendors = ["VendorA", "VendorB", "VendorC", "VendorD"]
    categories = ["Reagent", "Consumable", "Equipment", "Chemical"]
    critical_flags = [True, False]
    dates = []
    current_date = START_DATE
    while len(dates) < (num_rows // num_skus) + 20:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=7)

    state = np.random.get_state()
    data = []
    try:
        np.random.seed(42)
        for sku in skus:
            is_critical = np.random.choice(critical_flags, p=[0.35, 0.65])
            vendor = np.random.choice(vendors)
            category = np.random.choice(categories)
            base_consumption = np.random.randint(50, 800)
            for week in range(len(dates)):
                if len(data) >= num_rows:
                    break
                consumption = int(
                    base_consumption
                    * (1 + 0.15 * np.sin(week / 8) + np.random.normal(0, 0.25))
                )
                consumption = max(10, consumption)
                inventory = int(consumption * np.random.uniform(3, 12))
                lead_time = max(2, min(21, int(np.random.normal(7, 4))))
                if is_critical and np.random.random() < 0.28:
                    fill_rate = round(np.random.uniform(0.45, 0.82), 2)
                else:
                    fill_rate = round(np.random.uniform(0.85, 0.99), 2)
                data.append(
                    {
                        "week": dates[week % len(dates)],
                        "sku": sku,
                        "inventory_level": inventory,
                        "consumption": consumption,
                        "lead_time_days": lead_time,
                        "vendor_fill_rate": fill_rate,
                        "vendor_name": vendor,
                        "category": category,
                        "critical": "Yes" if is_critical else "No",
                    }
                )
    finally:
        np.random.set_state(state)

    frame = pd.DataFrame(data, columns=COLUMNS)
    return frame.head(num_rows).reset_index(drop=True)


def _tiled(pattern: list[int], n_weeks: int) -> list[int]:
    reps = (n_weeks + len(pattern) - 1) // len(pattern)
    return (list(pattern) * reps)[:n_weeks]


def _every(n_weeks: int, gap: int, size: int) -> list[int]:
    demand = [0] * n_weeks
    for index in range(gap - 1, n_weeks, gap):
        demand[index] = size
    return demand


def _every_sizes(n_weeks: int, gap: int, sizes: list[int]) -> list[int]:
    demand = [0] * n_weeks
    step = 0
    for index in range(gap - 1, n_weeks, gap):
        demand[index] = sizes[step % len(sizes)]
        step += 1
    return demand


# Fixed patterns. Demand does not use the baseline random stream.
# pattern is the Syntetos-Boylan label these weeks are built to land in.
PATTERN_SKUS = [
    {
        "sku": "SKU-00051",
        "pattern": "intermittent",
        "story": "Spare filter kit, four units every fourth week.",
        "demand": lambda n: _every(n, 4, 4),
        "category": "Equipment",
        "vendor": "VendorC",
        "critical": "Yes",
        "lead_time_days": 21,
        "fill_rate": 0.62,
        "on_hand": 4,
    },
    {
        "sku": "SKU-00052",
        "pattern": "intermittent",
        "story": "Gasket set, two units every sixth week.",
        "demand": lambda n: _every(n, 6, 2),
        "category": "Equipment",
        "vendor": "VendorA",
        "critical": "No",
        "lead_time_days": 14,
        "fill_rate": 0.96,
        "on_hand": 24,
    },
    {
        "sku": "SKU-00053",
        "pattern": "intermittent",
        "story": "Sensor spare, three units every eighth week.",
        "demand": lambda n: _every(n, 8, 3),
        "category": "Consumable",
        "vendor": "VendorB",
        "critical": "No",
        "lead_time_days": 14,
        "fill_rate": 0.93,
        "on_hand": 8,
    },
    {
        "sku": "SKU-00054",
        "pattern": "intermittent",
        "story": "Seal kit, two or three units every sixth week.",
        "demand": lambda n: _every_sizes(n, 6, [2, 3, 2, 2, 3]),
        "category": "Equipment",
        "vendor": "VendorD",
        "critical": "No",
        "lead_time_days": 21,
        "fill_rate": 0.91,
        "on_hand": 6,
    },
    {
        "sku": "SKU-00055",
        "pattern": "lumpy",
        "story": "Bulk chemical drops of 10, 40, or 180 after quiet weeks.",
        "demand": lambda n: _tiled([0, 0, 0, 10, 0, 0, 0, 0, 40, 0, 0, 0, 0, 180], n),
        "category": "Chemical",
        "vendor": "VendorB",
        "critical": "No",
        "lead_time_days": 21,
        "fill_rate": 0.88,
        "on_hand": 40,
    },
    {
        "sku": "SKU-00056",
        "pattern": "lumpy",
        "story": "Reagent campaign: a small lot, then a 150-unit order.",
        "demand": lambda n: _tiled([0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 150], n),
        "category": "Reagent",
        "vendor": "VendorC",
        "critical": "Yes",
        "lead_time_days": 14,
        "fill_rate": 0.70,
        "on_hand": 15,
    },
    {
        "sku": "SKU-00057",
        "pattern": "lumpy",
        "story": "Overhaul orders of 8, 60, or 200 with long gaps.",
        "demand": lambda n: _tiled([0, 0, 0, 0, 0, 8, 0, 0, 0, 60, 0, 0, 200], n),
        "category": "Chemical",
        "vendor": "VendorA",
        "critical": "No",
        "lead_time_days": 21,
        "fill_rate": 0.95,
        "on_hand": 80,
    },
    {
        "sku": "SKU-00058",
        "pattern": "lumpy",
        "story": "Project buys of 20, 80, or 250 units.",
        "demand": lambda n: _tiled([0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 80, 0, 250], n),
        "category": "Reagent",
        "vendor": "VendorD",
        "critical": "No",
        "lead_time_days": 21,
        "fill_rate": 0.90,
        "on_hand": 25,
    },
    {
        "sku": "SKU-00059",
        "pattern": "erratic",
        "story": "Weekly consumable whose order size swings from a handful to over 100.",
        "demand": lambda n: _tiled([5, 8, 140, 6, 110, 4, 90, 7], n),
        "category": "Consumable",
        "vendor": "VendorA",
        "critical": "No",
        "lead_time_days": 7,
        "fill_rate": 0.97,
        "on_hand": 40,
    },
    {
        "sku": "SKU-00060",
        "pattern": "erratic",
        "story": "Weekly demand alternating between a trickle and a large draw.",
        "demand": lambda n: _tiled([2, 80, 3, 120, 4, 60, 1, 150], n),
        "category": "Consumable",
        "vendor": "VendorB",
        "critical": "No",
        "lead_time_days": 7,
        "fill_rate": 0.94,
        "on_hand": 150,
    },
    {
        "sku": "SKU-00061",
        "pattern": "erratic",
        "story": "Weekly chemical issues, usually small, sometimes a full drum.",
        "demand": lambda n: _tiled([10, 12, 200, 8, 180, 15, 9, 160], n),
        "category": "Chemical",
        "vendor": "VendorC",
        "critical": "No",
        "lead_time_days": 10,
        "fill_rate": 0.89,
        "on_hand": 180,
    },
]


def build_pattern_frame(dates: list[str] | None = None) -> pd.DataFrame:
    """SKU-00051 onward. On-hand is the shelf stock, including weeks with no demand."""
    if dates is None:
        dates = week_labels(baseline_week_count())
    rows = []
    n_weeks = len(dates)
    for spec in PATTERN_SKUS:
        demand = list(spec["demand"](n_weeks))
        if len(demand) != n_weeks:
            raise ValueError(f"{spec['sku']} demand length {len(demand)} != {n_weeks}")
        for week, consumption in zip(dates, demand):
            rows.append(
                {
                    "week": week,
                    "sku": spec["sku"],
                    "inventory_level": int(spec["on_hand"]),
                    "consumption": int(consumption),
                    "lead_time_days": int(spec["lead_time_days"]),
                    "vendor_fill_rate": float(spec["fill_rate"]),
                    "vendor_name": spec["vendor"],
                    "category": spec["category"],
                    "critical": spec["critical"],
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def build_demo_frame() -> pd.DataFrame:
    baseline = build_baseline_frame()
    extra = build_pattern_frame(week_labels(baseline_week_count()))
    return pd.concat([baseline, extra], ignore_index=True)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = build_demo_frame()
    frame.to_csv(OUTPUT_PATH, index=False)
    n_skus = frame["sku"].nunique()
    print(f"Wrote {len(frame):,} rows, {n_skus} SKUs → {OUTPUT_PATH}")
    print(f"   Baseline SKUs: {BASELINE_SKUS}. Added patterns: {n_skus - BASELINE_SKUS}.")
    print(f"   Date range: {frame['week'].min()} → {frame['week'].max()}")
    print(f"   File size: {os.path.getsize(OUTPUT_PATH) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
