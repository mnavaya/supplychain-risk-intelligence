"""Tests for holdout accuracy, service-level stock, cover alerts, and ABC."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inventory_logic import (  # noqa: E402
    abc_summary,
    accuracy_display_table,
    apply_service_level,
    attach_unit_cost,
    classify_abc,
    classify_cover_risk,
    compute_reorder_point,
    compute_safety_stock,
    eligible_forecast_skus,
    forecast_accuracy_check,
    format_abc_summary,
    format_risk_board,
    holt_winters_forecast,
    sample_unit_cost,
    sort_risk_table,
    split_train_test,
    weekly_series,
)
from main import forecast_and_stockout  # noqa: E402
from validation import backtest_forecast  # noqa: E402


def _weeks(start="2024-01-01", count=16):
    start = pd.Timestamp(start)
    return [start + pd.Timedelta(weeks=i) for i in range(count)]


def _history(sku="SKU-A", demands=None, stock=20, lead_days=14, unit_cost=10, category="Reagent"):
    if demands is None:
        demands = list(range(1, 17))
    rows = []
    for week, demand in zip(_weeks(count=len(demands)), demands):
        rows.append(
            {
                "week": week,
                "sku": sku,
                "inventory_level": stock,
                "consumption": demand,
                "lead_time_days": lead_days,
                "vendor_fill_rate": 0.95,
                "vendor_name": "VendorA",
                "category": category,
                "critical": "No",
                "unit_cost": unit_cost,
            }
        )
    return pd.DataFrame(rows)


class ForecastAccuracyTests(unittest.TestCase):
    def test_holdout_weeks_never_reach_the_model(self):
        demands = list(range(1, 21))
        frame = _history(demands=demands)
        seen = {}

        def fake(consumption, horizon):
            seen["values"] = [float(value) for value in consumption]
            seen["horizon"] = int(horizon)
            return np.ones(horizon)

        with patch("inventory_logic.holt_winters_forecast", fake):
            result = forecast_accuracy_check(frame, "SKU-A")

        series = weekly_series(frame, "SKU-A")
        train = [float(value) for value in series.iloc[:-4]["consumption"]]
        test = [float(value) for value in series.iloc[-4:]["consumption"]]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(seen["horizon"], 4)
        self.assertEqual(seen["values"], train)
        for value in test:
            self.assertNotIn(value, seen["values"])
        self.assertLess(result["train_end"], result["test_start"])
        self.assertEqual(result["naive_value"], train[-1])
        self.assertTrue((result["detail"]["naive_baseline"] == train[-1]).all())

    def test_unsorted_history_still_holds_out_the_latest_weeks(self):
        frame = _history(demands=list(range(1, 17))).sample(frac=1, random_state=3)
        result = forecast_accuracy_check(frame, "SKU-A")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["test_start"][:10], "2024-03-25")
        self.assertEqual(result["train_end"][:10], "2024-03-18")
        self.assertEqual(result["naive_value"], 12.0)

    def test_naive_mae_and_winner_message(self):
        frame = _history(demands=list(range(1, 21)))

        def fake(consumption, horizon):
            return np.ones(horizon)

        with patch("inventory_logic.holt_winters_forecast", fake):
            result = forecast_accuracy_check(frame, "SKU-A")

        # Holdout actuals are 17, 18, 19, 20. Naive repeats 16.
        self.assertAlmostEqual(result["mae_naive"], 2.5)
        self.assertAlmostEqual(result["mae_holt_winters"], np.mean([16, 17, 18, 19]))
        self.assertEqual(result["winner"], "Naive baseline")
        self.assertIn("Naive baseline won", result["message"])
        table = accuracy_display_table(result)
        self.assertEqual(list(table.columns), ["Week", "Actual", "Holt-Winters", "Naive baseline"])
        self.assertEqual(table.iloc[-1]["Week"], "MAE")
        self.assertEqual(table.iloc[-1]["Naive baseline"], 2.5)
        self.assertEqual(table.iloc[-1]["Holt-Winters"], round(result["mae_holt_winters"], 2))
        self.assertTrue(pd.isna(table.iloc[-1]["Actual"]))
        # Streamlit serializes dataframes through Arrow. Mixed text in a
        # numeric column drops the forecast values on screen.
        import pyarrow as pa

        arrow = pa.Table.from_pandas(table)
        self.assertIn("Holt-Winters", arrow.column_names)

    def test_short_history_is_not_scored(self):
        frame = _history(demands=[5, 6, 7, 8, 9])
        result = forecast_accuracy_check(frame, "SKU-A")
        self.assertEqual(result["status"], "insufficient_history")
        self.assertEqual(eligible_forecast_skus(frame), [])

    def test_duplicate_weeks_are_collapsed_before_the_split(self):
        frame = _history(demands=list(range(1, 13)))
        extra = frame.iloc[[0]].copy()
        extra["consumption"] = 100
        combined = pd.concat([frame, extra], ignore_index=True)
        series = weekly_series(combined, "SKU-A")
        self.assertEqual(len(series), 12)
        self.assertEqual(float(series.iloc[0]["consumption"]), 101.0)

    def test_split_rejects_a_week_that_appears_on_both_sides(self):
        series = pd.DataFrame(
            {
                "week": pd.to_datetime(
                    ["2024-01-01", "2024-02-05", "2024-02-05", "2024-02-05", "2024-02-05"]
                ),
                "consumption": [1, 2, 3, 4, 5],
            }
        )
        with self.assertRaises(ValueError):
            split_train_test(series, 3)

    def test_real_holt_winters_fit_stays_on_training_rows(self):
        frame = _history(demands=[10 + (i % 4) for i in range(20)])
        result = forecast_accuracy_check(frame, "SKU-A")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["test_weeks"], 4)
        self.assertEqual(result["train_weeks"], 16)
        self.assertGreaterEqual(result["mae_holt_winters"], 0)
        self.assertIn(result["winner"], {"Holt-Winters", "Naive baseline", "Tie"})
        backtest = backtest_forecast(frame, "SKU-A", forecast_horizon=4)
        self.assertEqual(backtest["status"], "Success")
        self.assertIn("MAE_naive", backtest)


class ServiceLevelTests(unittest.TestCase):
    def test_formula_and_monotonic_service_levels(self):
        # SD 10, lead time 4 weeks. sqrt(4) = 2.
        self.assertAlmostEqual(compute_safety_stock(10, 4, 1.28), 25.6)
        self.assertAlmostEqual(compute_safety_stock(10, 4, 1.65), 33.0)
        self.assertAlmostEqual(compute_safety_stock(10, 4, 2.33), 46.6)
        self.assertAlmostEqual(compute_reorder_point(20, 4, 25.6), 105.6)
        self.assertAlmostEqual(compute_reorder_point(20, 4, 33.0), 113.0)
        self.assertAlmostEqual(compute_reorder_point(20, 4, 46.6), 126.6)

        safety = [compute_safety_stock(10, 4, z) for z in (1.28, 1.65, 2.33)]
        reorder = [compute_reorder_point(20, 4, ss) for ss in safety]
        self.assertEqual(safety, sorted(safety))
        self.assertEqual(reorder, sorted(reorder))
        self.assertGreater(safety[2], safety[1])
        self.assertGreater(safety[1], safety[0])
        self.assertGreater(reorder[2], reorder[0])

    def test_pipeline_service_level_raises_safety_stock(self):
        frame = pd.concat(
            [
                _history("SKU-A", demands=[10, 14, 11, 18, 12, 16, 13, 19, 15, 17, 12, 20], stock=8, lead_days=21, unit_cost=40),
                _history("SKU-B", demands=[30, 28, 33, 31, 36, 29, 34, 32, 37, 30, 35, 33], stock=400, lead_days=7, unit_cost=8, category="Consumable"),
            ],
            ignore_index=True,
        )
        low = forecast_and_stockout(frame, forecast_weeks=4, service_level="90%")
        mid = forecast_and_stockout(frame, forecast_weeks=4, service_level="95%")
        high = forecast_and_stockout(frame, forecast_weeks=4, service_level="99%")
        merged = low[["sku", "safety_stock", "reorder_point"]].merge(
            mid[["sku", "safety_stock", "reorder_point"]],
            on="sku",
            suffixes=("_90", "_95"),
        ).merge(
            high[["sku", "safety_stock", "reorder_point"]].rename(
                columns={
                    "safety_stock": "safety_stock_99",
                    "reorder_point": "reorder_point_99",
                }
            ),
            on="sku",
        )
        self.assertTrue((merged["safety_stock_95"] >= merged["safety_stock_90"]).all())
        self.assertTrue((merged["safety_stock_99"] >= merged["safety_stock_95"]).all())
        self.assertTrue((merged["reorder_point_99"] >= merged["reorder_point_95"]).all())
        self.assertTrue((merged["safety_stock_99"] > merged["safety_stock_90"]).any())
        self.assertTrue((mid["service_level"] == "95%").all())
        self.assertTrue((mid["z_score"] == 1.65).all())


class CoverRiskTests(unittest.TestCase):
    def test_high_medium_low_and_zero_demand(self):
        high = classify_cover_risk(10, 10, 2)
        medium = classify_cover_risk(25, 10, 2)
        low = classify_cover_risk(40, 10, 2)
        exact_lead = classify_cover_risk(20, 10, 2)
        exact_buffer = classify_cover_risk(30, 10, 2)
        zero = classify_cover_risk(10, 0, 2)

        self.assertEqual(high["stockout_risk"], "High")
        self.assertIn("Runs out before replenishment", high["risk_reason"])
        self.assertAlmostEqual(high["weeks_of_cover"], 1.0)

        self.assertEqual(medium["stockout_risk"], "Medium")
        self.assertIn("Within one week", medium["risk_reason"])
        self.assertEqual(exact_lead["stockout_risk"], "Medium")

        self.assertEqual(low["stockout_risk"], "Low")
        self.assertIn("Comfortably covers", low["risk_reason"])
        self.assertEqual(exact_buffer["stockout_risk"], "Low")

        self.assertEqual(zero["stockout_risk"], "Low")
        self.assertTrue(np.isinf(zero["weeks_of_cover"]))
        self.assertIn("zero", zero["risk_reason"])

    def test_risk_table_sorts_high_first_and_names_the_columns(self):
        rows = pd.DataFrame(
            [
                {
                    "sku": "L",
                    "stockout_risk": "Low",
                    "weeks_of_cover": 12,
                    "risk_reason": "Comfortably covers the supplier lead time.",
                    "current_inventory": 40,
                    "forecast_weekly_demand": 3,
                },
                {
                    "sku": "H",
                    "stockout_risk": "High",
                    "weeks_of_cover": 0.5,
                    "risk_reason": "Runs out before replenishment arrives.",
                    "current_inventory": 5,
                    "forecast_weekly_demand": 10,
                },
                {
                    "sku": "M",
                    "stockout_risk": "Medium",
                    "weeks_of_cover": 2.2,
                    "risk_reason": "Within one week of the supplier lead time.",
                    "current_inventory": 22,
                    "forecast_weekly_demand": 10,
                },
                {
                    "sku": "Z",
                    "stockout_risk": "Low",
                    "weeks_of_cover": np.inf,
                    "risk_reason": "Forecast weekly demand is zero, so on-hand stock is not being drawn down.",
                    "current_inventory": 8,
                    "forecast_weekly_demand": 0,
                },
            ]
        )
        ordered = sort_risk_table(rows)
        self.assertEqual(list(ordered["sku"]), ["H", "M", "L", "Z"])
        board = format_risk_board(rows)
        self.assertEqual(list(board["SKU"]), ["H", "M", "L", "Z"])
        self.assertIn("Weeks of Cover", board.columns)
        self.assertIn("Risk Reason", board.columns)
        self.assertEqual(board.loc[board["SKU"] == "Z", "Weeks of Cover"].iloc[0], "No demand")
        import pyarrow as pa

        pa.Table.from_pandas(board)

    def test_zero_forecast_from_the_pipeline(self):
        quiet = holt_winters_forecast(np.zeros(12), 4)
        self.assertTrue(np.all(quiet == 0))
        frame = _history("SKU-Z", demands=[0] * 12, stock=15, lead_days=14, unit_cost=5)
        summary = forecast_and_stockout(frame, forecast_weeks=4, service_level="95%")
        row = summary.iloc[0]
        self.assertEqual(row["forecast_weekly_demand"], 0.0)
        self.assertTrue(np.isinf(row["weeks_of_cover"]))
        self.assertEqual(row["stockout_risk"], "Low")
        self.assertIn("zero", row["risk_reason"])


class AbcTests(unittest.TestCase):
    def test_pareto_cutoffs_and_summary_shares(self):
        usage = pd.DataFrame(
            {"sku": ["a", "b", "c", "d"], "usage_value": [50, 30, 15, 5]}
        )
        classified = classify_abc(usage)
        classes = dict(zip(classified["sku"], classified["abc_class"]))
        self.assertEqual(classes, {"a": "A", "b": "A", "c": "B", "d": "C"})

        summary = abc_summary(classified)
        shares = dict(zip(summary["abc_class"], summary["pct_of_usage_value"]))
        self.assertAlmostEqual(shares["A"], 80.0)
        self.assertAlmostEqual(shares["B"], 15.0)
        self.assertAlmostEqual(shares["C"], 5.0)
        self.assertAlmostEqual(summary["pct_of_usage_value"].sum(), 100.0)
        self.assertAlmostEqual(summary["pct_of_skus"].sum(), 100.0)
        self.assertEqual(int(summary["sku_count"].sum()), 4)

        display = format_abc_summary(usage)
        self.assertAlmostEqual(display["% of usage value"].sum(), 100.0, places=2)
        self.assertAlmostEqual(display["% of SKUs"].sum(), 100.0, places=2)

    def test_item_that_crosses_eighty_percent_stays_in_a(self):
        usage = pd.DataFrame({"sku": ["a", "b", "c"], "usage_value": [70, 20, 10]})
        classes = dict(zip(*[classify_abc(usage)[col] for col in ("sku", "abc_class")]))
        self.assertEqual(classes["a"], "A")
        self.assertEqual(classes["b"], "A")
        self.assertEqual(classes["c"], "B")

    def test_missing_unit_cost_uses_documented_sample(self):
        frame = pd.DataFrame({"sku": ["SKU-1", "SKU-2"], "category": ["Reagent", "Consumable"]})
        priced, source = attach_unit_cost(frame)
        self.assertEqual(source, "sample")
        self.assertEqual(priced.loc[0, "unit_cost"], sample_unit_cost("SKU-1", "Reagent"))
        self.assertNotEqual(priced.loc[0, "unit_cost"], priced.loc[1, "unit_cost"])

        given = frame.copy()
        given["unit_cost"] = [12.5, None]
        mixed, mixed_source = attach_unit_cost(given)
        self.assertEqual(mixed_source, "mixed")
        self.assertEqual(mixed.loc[0, "unit_cost"], 12.5)
        self.assertEqual(mixed.loc[1, "unit_cost"], sample_unit_cost("SKU-2", "Consumable"))

    def test_pipeline_summary_shares_sum_to_100(self):
        frame = pd.concat(
            [
                _history("SKU-A", demands=[20, 22, 21, 24, 23, 25, 22, 26, 24, 27, 23, 28], unit_cost=50, stock=10, lead_days=28),
                _history("SKU-B", demands=[8, 9, 7, 8, 10, 9, 8, 11, 9, 8, 10, 9], unit_cost=12, stock=80, lead_days=7, category="Consumable"),
                _history("SKU-C", demands=[2, 2, 3, 2, 2, 1, 2, 2, 3, 2, 2, 2], unit_cost=4, stock=40, lead_days=7, category="Chemical"),
                _history("SKU-D", demands=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], unit_cost=2, stock=100, lead_days=7, category="Equipment"),
            ],
            ignore_index=True,
        )
        summary = forecast_and_stockout(frame, forecast_weeks=4, service_level="95%")
        self.assertEqual(summary.attrs.get("unit_cost_source"), "file")
        shares = abc_summary(summary)
        self.assertAlmostEqual(shares["pct_of_usage_value"].sum(), 100.0)
        self.assertAlmostEqual(shares["pct_of_skus"].sum(), 100.0)
        self.assertTrue(set(summary["abc_class"]).issubset({"A", "B", "C"}))
        ranks = {"High": 0, "Medium": 1, "Low": 2}
        order = [ranks[label] for label in summary["stockout_risk"]]
        self.assertEqual(order, sorted(order))

    def test_reorder_note_follows_the_selected_service_level(self):
        summary = pd.DataFrame(
            [
                {
                    "sku": "SKU-A",
                    "current_inventory": 30,
                    "avg_weekly_demand": 20,
                    "weekly_demand_sd": 5,
                    "lead_time_weeks": 2,
                    "risk_reason": "Comfortably covers the supplier lead time (8.0 weeks of cover vs 2.0 weeks lead time).",
                }
            ]
        )
        low = apply_service_level(summary, "90%")
        high = apply_service_level(summary, "99%")
        self.assertGreater(high.iloc[0]["safety_stock"], low.iloc[0]["safety_stock"])
        self.assertGreater(high.iloc[0]["reorder_point"], low.iloc[0]["reorder_point"])
        # 90%: SS = 1.28 * 5 * sqrt(2) ≈ 9.05, ROP ≈ 40 + 9.05 > 30, so both are below ROP.
        self.assertIn("reorder point", low.iloc[0]["risk_reason"])
        again = apply_service_level(low, "90%")
        self.assertEqual(again.iloc[0]["risk_reason"].count("reorder point"), 1)

        # Stock sits between the 90% reorder point and the 99% reorder point.
        crossing = summary.copy()
        crossing["current_inventory"] = 60
        crossing["weekly_demand_sd"] = 8
        at_90 = apply_service_level(crossing, "90%")
        at_99 = apply_service_level(crossing, "99%")
        self.assertNotIn("reorder point", at_90.iloc[0]["risk_reason"])
        self.assertIn("reorder point", at_99.iloc[0]["risk_reason"])
        self.assertGreater(at_99.iloc[0]["reorder_point"], 60)
        self.assertLess(at_90.iloc[0]["reorder_point"], 60)


if __name__ == "__main__":
    unittest.main()
