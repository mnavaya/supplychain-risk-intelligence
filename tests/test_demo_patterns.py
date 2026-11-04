"""Demo file: original SKUs stay put, and the added series survive the pipeline."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_large_dataset import (  # noqa: E402
    BASELINE_ROWS,
    BASELINE_SKUS,
    OUTPUT_PATH,
    PATTERN_SKUS,
    build_baseline_frame,
    build_pattern_frame,
)
from inventory_logic import (  # noqa: E402
    abc_summary,
    classify_demand_pattern,
    format_risk_board,
    simulate_supply_shock,
)
from main import SupplyChainAgent, forecast_and_stockout  # noqa: E402


class DemoPatternTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extra = build_pattern_frame()
        cls.summary = forecast_and_stockout(cls.extra, forecast_weeks=8, service_level="95%")

    def test_baseline_block_matches_the_checked_in_prefix(self):
        disk = pd.read_csv(ROOT / OUTPUT_PATH)
        built = build_baseline_frame()
        prefix = disk.iloc[:BASELINE_ROWS].reset_index(drop=True)
        self.assertEqual(len(built), BASELINE_ROWS)
        self.assertEqual(prefix["sku"].nunique(), BASELINE_SKUS)
        self.assertTrue(built.equals(prefix))
        added = sorted(set(disk["sku"]) - set(built["sku"]))
        self.assertEqual(added, [spec["sku"] for spec in PATTERN_SKUS])

    def test_added_skus_land_in_the_planned_quadrants(self):
        counts = {"intermittent": 0, "lumpy": 0, "erratic": 0}
        n_weeks = int(self.extra["week"].nunique())
        for spec in PATTERN_SKUS:
            part = self.extra.loc[self.extra["sku"] == spec["sku"], "consumption"]
            classified = classify_demand_pattern(part)
            self.assertEqual(classified["pattern"], spec["pattern"], spec["sku"])
            zeros = int((part == 0).sum())
            if spec["pattern"] == "erratic":
                self.assertEqual(zeros, 0, spec["sku"])
            else:
                self.assertGreater(zeros, n_weeks // 2, spec["sku"])
            counts[spec["pattern"]] += 1
        self.assertGreaterEqual(counts["intermittent"], 3)
        self.assertGreaterEqual(counts["lumpy"], 3)
        self.assertGreaterEqual(counts["erratic"], 2)

    def test_zero_weeks_keep_stock_math_and_the_copilot_finite(self):
        summary = self.summary
        self.assertEqual(len(summary), len(PATTERN_SKUS))
        for column in ("safety_stock", "reorder_point", "forecast_weekly_demand", "days_to_stockout"):
            self.assertTrue(np.isfinite(summary[column]).all(), column)
        self.assertTrue(set(summary["stockout_risk"]) <= {"High", "Medium", "Low"})
        self.assertTrue(set(summary["abc_class"]) <= {"A", "B", "C"})
        shares = abc_summary(summary)
        self.assertAlmostEqual(float(shares["pct_of_usage_value"].sum()), 100.0)
        self.assertAlmostEqual(float(shares["pct_of_skus"].sum()), 100.0)

        # A zero consumption week must not divide days of supply by zero.
        raw = self.extra.copy()
        raw["daily_consumption"] = raw["consumption"] / 7.0
        cover_days = raw["inventory_level"] / raw["daily_consumption"].replace(0, 1)
        self.assertTrue(np.isfinite(cover_days).all())
        self.assertGreater(int((raw["consumption"] == 0).sum()), 0)

        low = forecast_and_stockout(self.extra, forecast_weeks=4, service_level="90%")
        high = forecast_and_stockout(self.extra, forecast_weeks=4, service_level="99%")
        merged = low[["sku", "safety_stock"]].merge(
            high[["sku", "safety_stock"]], on="sku", suffixes=("_90", "_99")
        )
        self.assertTrue((merged["safety_stock_99"] >= merged["safety_stock_90"]).all())
        self.assertTrue((merged["safety_stock_99"] > merged["safety_stock_90"]).any())

        shock = simulate_supply_shock(
            summary, extra_lead_weeks=2, spike_pct=40, spike_weeks=3, n_runs=40
        )
        self.assertEqual(len(shock), len(summary))
        self.assertTrue(shock["stockout_probability"].between(0, 1).all())

        import pyarrow as pa

        pa.Table.from_pandas(format_risk_board(summary))

        agent = SupplyChainAgent(summary)
        self.assertTrue(agent.query("show high risk items"))
        self.assertTrue(agent.query("which items are below reorder point?"))
        status = agent.query("status for SKU-00051")
        self.assertIn("SKU-00051", status)
        self.assertIn("Safety Stock", status)

        erratic = summary[summary["demand_pattern"] == "erratic"]
        self.assertTrue((erratic["chosen_model"] == "Holt-Winters").all())
        eligible = summary[summary["demand_pattern"].isin(["intermittent", "lumpy"])]
        self.assertTrue((eligible["chosen_model"].isin(["Holt-Winters", "Croston SBA"])).all())
        self.assertTrue((eligible["chosen_model"] == "Croston SBA").any())
