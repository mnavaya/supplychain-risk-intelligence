"""Tests for holdout accuracy, service-level stock, cover alerts, and ABC."""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inventory_logic import (  # noqa: E402
    MONTE_CARLO_RUNS,
    abc_summary,
    accuracy_display_table,
    apply_service_level,
    attach_unit_cost,
    classify_abc,
    classify_cover_risk,
    compute_reorder_point,
    compute_safety_stock,
    describe_shock,
    effective_weekly_demand,
    eligible_forecast_skus,
    emergency_order_quantity,
    forecast_accuracy_check,
    format_abc_summary,
    format_risk_board,
    format_shock_table,
    holt_winters_forecast,
    projected_stockout_week,
    risk_count_frame,
    risk_heatmap_frame,
    risk_label_from_cover,
    sample_unit_cost,
    simulate_supply_shock,
    sort_risk_table,
    split_train_test,
    stockout_probabilities,
    weekly_series,
)
from main import forecast_and_stockout, generate_scenario_narrative  # noqa: E402
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


def _shock_row(**overrides):
    row = {
        "sku": "SKU-A",
        "current_inventory": 30.0,
        "forecast_weekly_demand": 10.0,
        "avg_weekly_demand": 10.0,
        "weekly_demand_sd": 0.0,
        "lead_time_weeks": 1.0,
        "z_score": 1.65,
        "abc_class": "A",
    }
    row.update(overrides)
    return row


class SupplyShockTests(unittest.TestCase):
    def test_zero_shock_matches_the_risk_board(self):
        frame = pd.DataFrame(
            [
                _shock_row(sku="LOW", current_inventory=40, forecast_weekly_demand=10, lead_time_weeks=2),
                _shock_row(sku="MED", current_inventory=25, forecast_weekly_demand=10, lead_time_weeks=2),
                _shock_row(sku="HIGH", current_inventory=10, forecast_weekly_demand=10, lead_time_weeks=2),
                _shock_row(sku="ZERO", current_inventory=8, forecast_weekly_demand=0, avg_weekly_demand=0, lead_time_weeks=2),
            ]
        )
        shock = simulate_supply_shock(frame, extra_lead_weeks=0, spike_pct=0, spike_weeks=4, n_runs=50)
        by_sku = shock.set_index("sku")
        for sku, row in frame.set_index("sku").iterrows():
            baseline = classify_cover_risk(
                row["current_inventory"], row["forecast_weekly_demand"], row["lead_time_weeks"]
            )
            self.assertEqual(by_sku.loc[sku, "stockout_risk_before"], baseline["stockout_risk"])
            self.assertEqual(by_sku.loc[sku, "stockout_risk_after"], baseline["stockout_risk"])
            self.assertFalse(bool(by_sku.loc[sku, "risk_flipped"]))
        self.assertEqual(by_sku.loc["ZERO", "stockout_risk_after"], "Low")
        self.assertTrue(np.isinf(by_sku.loc["ZERO", "projected_stockout_week"]))
        self.assertEqual(by_sku.loc["ZERO", "stockout_probability"], 0.0)
        self.assertIn("No stockout", format_shock_table(shock.loc[shock["sku"] == "ZERO"]).iloc[0]["Projected stockout week"])

    def test_delay_and_spike_can_flip_risk_and_both_is_at_least_as_severe(self):
        # Cover is 3 weeks against a 1-week lead: Low.
        base = pd.DataFrame([_shock_row()])
        self.assertEqual(risk_label_from_cover(3, 1), "Low")

        delayed = simulate_supply_shock(base, extra_lead_weeks=3, spike_pct=0, spike_weeks=4)
        spiked = simulate_supply_shock(base, extra_lead_weeks=0, spike_pct=100, spike_weeks=8)
        both = simulate_supply_shock(base, extra_lead_weeks=3, spike_pct=100, spike_weeks=8)

        self.assertEqual(delayed.iloc[0]["stockout_risk_after"], "High")
        self.assertEqual(spiked.iloc[0]["stockout_risk_after"], "Medium")
        self.assertEqual(both.iloc[0]["stockout_risk_after"], "High")
        self.assertTrue(bool(delayed.iloc[0]["risk_flipped"]))
        self.assertTrue(bool(spiked.iloc[0]["risk_flipped"]))
        self.assertTrue(bool(both.iloc[0]["risk_flipped"]))
        self.assertAlmostEqual(spiked.iloc[0]["projected_stockout_week"], 1.5)
        self.assertLess(both.iloc[0]["projected_stockout_week"], delayed.iloc[0]["projected_stockout_week"])
        self.assertGreaterEqual(both.iloc[0]["emergency_order_qty"], delayed.iloc[0]["emergency_order_qty"])
        self.assertGreaterEqual(both.iloc[0]["emergency_order_qty"], spiked.iloc[0]["emergency_order_qty"])

    def test_emergency_order_uses_the_reorder_point_formula(self):
        # 100% spike for the whole 4-week lead. Effective demand is 20.
        # SS = 1.65 * 4 * sqrt(4) = 13.2. ROP = 20 * 4 + 13.2 = 93.2. Order = 93.2 - 30.
        quote = emergency_order_quantity(30, 10, 4, 4, 1.65, spike_pct=100, spike_weeks=4)
        self.assertAlmostEqual(effective_weekly_demand(10, 100, 4, 4), 20.0)
        self.assertAlmostEqual(quote["safety_stock"], compute_safety_stock(4, 4, 1.65))
        self.assertAlmostEqual(
            quote["reorder_point"],
            compute_reorder_point(20, 4, quote["safety_stock"]),
        )
        self.assertAlmostEqual(quote["emergency_order_qty"], 93.2 - 30)

        no_gap = emergency_order_quantity(100, 10, 0, 1, 1.65, spike_pct=0, spike_weeks=0)
        self.assertEqual(no_gap["emergency_order_qty"], 0.0)

    def test_monte_carlo_probability_bounds_and_speed(self):
        certain = stockout_probabilities([10], [10], [0], [2], n_runs=MONTE_CARLO_RUNS, seed=1)
        safe = stockout_probabilities([100], [10], [0], [2], n_runs=MONTE_CARLO_RUNS, seed=1)
        partial = stockout_probabilities([4], [10], [0], [0.5], n_runs=20, seed=1)
        exact = stockout_probabilities([10], [10], [0], [1], n_runs=20, seed=1)
        noisy = stockout_probabilities([20], [10], [4], [2], n_runs=MONTE_CARLO_RUNS, seed=7)
        self.assertEqual(certain[0], 1.0)
        self.assertEqual(safe[0], 0.0)
        self.assertEqual(partial[0], 1.0)
        self.assertEqual(exact[0], 0.0)
        self.assertGreater(noisy[0], 0.0)
        self.assertLess(noisy[0], 1.0)

        longer = stockout_probabilities([25.0], [10.0], [0.0], [3.0], n_runs=20, seed=1)
        shorter = stockout_probabilities([25.0], [10.0], [0.0], [2.0], n_runs=20, seed=1)
        self.assertGreater(float(longer[0]), float(shorter[0]))

        catalog = pd.DataFrame(
            [
                _shock_row(
                    sku=f"SKU-{i:02d}",
                    current_inventory=20 + i,
                    forecast_weekly_demand=8 + (i % 5),
                    avg_weekly_demand=8 + (i % 5),
                    weekly_demand_sd=2 + (i % 3),
                    lead_time_weeks=0.5 + (i % 4),
                )
                for i in range(50)
            ]
        )
        started = time.perf_counter()
        shock = simulate_supply_shock(
            catalog, extra_lead_weeks=2, spike_pct=40, spike_weeks=3, n_runs=MONTE_CARLO_RUNS
        )
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 1.5)
        self.assertEqual(len(shock), 50)
        self.assertTrue(((shock["stockout_probability"] >= 0) & (shock["stockout_probability"] <= 1)).all())
        counts = risk_count_frame(shock)
        self.assertEqual(int(counts["skus"].sum()), 100)  # before + after
        heat = risk_heatmap_frame(shock)
        self.assertEqual(len(heat), 100)
        self.assertEqual(set(heat["when"]), {"Before", "After"})
        # Flips are sorted ahead of SKUs that stay put.
        if shock["risk_flipped"].any() and (~shock["risk_flipped"]).any():
            self.assertTrue(bool(shock.iloc[0]["risk_flipped"]))
            self.assertFalse(bool(shock.iloc[-1]["risk_flipped"]))

    def test_projected_week_respects_a_short_spike(self):
        # Double demand is 20. Stock of 50 runs out 2.5 weeks into a long spike.
        self.assertAlmostEqual(projected_stockout_week(50, 10, spike_pct=100, spike_weeks=4), 2.5)
        # A 2-week spike consumes 40, then the last 10 units take one normal week.
        self.assertAlmostEqual(projected_stockout_week(50, 10, spike_pct=100, spike_weeks=2), 3.0)
        self.assertAlmostEqual(projected_stockout_week(100, 10, spike_pct=100, spike_weeks=2), 8.0)

    def test_narrative_falls_back_when_gemma_is_down(self):
        shock = simulate_supply_shock(
            pd.DataFrame([_shock_row()]),
            extra_lead_weeks=3,
            spike_pct=0,
            spike_weeks=1,
            n_runs=10,
        )
        text = describe_shock(shock, 3, 0, 1)
        self.assertIn("higher stockout risk", text)
        self.assertIn("SKU-A", text)

        with patch("main._gemma_chat", side_effect=ConnectionError("connection refused")):
            told = generate_scenario_narrative(shock, 3, 0, 1)
        self.assertEqual(told["source"], "rule")
        self.assertEqual(told["narrative"], text)
        self.assertIn("connection refused", told["error"])

        with patch(
            "main._gemma_chat",
            return_value={"message": {"content": " Order SKU-A before the delay lands. "}},
        ):
            told = generate_scenario_narrative(shock, 3, 0, 1)
        self.assertEqual(told["source"], "gemma2:2b")
        self.assertIn("Order SKU-A", told["narrative"])
        self.assertEqual(told["fallback"], text)

    def test_gemma_call_uses_a_short_timeout(self):
        import main

        with patch("main.ollama.Client") as client_cls:
            client_cls.return_value.chat.return_value = {"message": {"content": "ok"}}
            main._gemma_chat([{"role": "user", "content": "hi"}])
        client_cls.assert_called_once_with(timeout=main.GEMMA_TIMEOUT_SECONDS)
        self.assertLessEqual(main.GEMMA_TIMEOUT_SECONDS, 8)


if __name__ == "__main__":
    unittest.main()
