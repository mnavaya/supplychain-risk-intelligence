"""Grounded copilot answers, and the rule-based fallback when Gemma is unavailable."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import (  # noqa: E402
    SupplyChainAgent,
    check_answer_grounding,
    select_copilot_rows,
)


def _summary():
    return pd.DataFrame(
        [
            {
                "sku": "SKU-00001",
                "abc_class": "A",
                "stockout_risk": "High",
                "weeks_of_cover": 1.2,
                "risk_reason": "Runs out before replenishment arrives (1.2 weeks of cover vs 1.7 weeks lead time).",
                "current_inventory": 10,
                "forecast_weekly_demand": 8.5,
                "reorder_point": 18,
                "safety_stock": 4,
                "days_to_stockout": 8.8,
                "lead_time_days": 12,
                "lead_time_weeks": 1.7,
                "vendor_fill_rate": 0.86,
                "vendor_name": "VendorA",
                "critical_item": "No",
                "service_level": "95%",
            },
            {
                "sku": "SKU-00002",
                "abc_class": "A",
                "stockout_risk": "Low",
                "weeks_of_cover": 6.0,
                "risk_reason": "Comfortably covers the supplier lead time (6.0 weeks of cover vs 1.0 weeks lead time).",
                "current_inventory": 48,
                "forecast_weekly_demand": 8.0,
                "reorder_point": 12,
                "safety_stock": 3,
                "days_to_stockout": 42.0,
                "lead_time_days": 7,
                "lead_time_weeks": 1.0,
                "vendor_fill_rate": 0.95,
                "vendor_name": "VendorB",
                "critical_item": "Yes",
                "service_level": "95%",
            },
            {
                "sku": "SKU-00003",
                "abc_class": "C",
                "stockout_risk": "High",
                "weeks_of_cover": 0.4,
                "risk_reason": "Runs out before replenishment arrives (0.4 weeks of cover vs 2.0 weeks lead time).",
                "current_inventory": 2,
                "forecast_weekly_demand": 5.0,
                "reorder_point": 20,
                "safety_stock": 6,
                "days_to_stockout": 2.8,
                "lead_time_days": 14,
                "lead_time_weeks": 2.0,
                "vendor_fill_rate": 0.70,
                "vendor_name": "VendorC",
                "critical_item": "No",
                "service_level": "95%",
            },
        ]
    )


class GroundingCheckTests(unittest.TestCase):
    def test_grounding_check_catches_a_fabricated_number(self):
        rows = pd.DataFrame(
            [
                {
                    "sku": "SKU-00001",
                    "current_inventory": 12.46,
                    "reorder_point": 20,
                    "vendor_fill_rate": 0.86,
                    "weeks_of_cover": 1.24,
                }
            ]
        )
        grounded = check_answer_grounding(
            "SKU-00001 has 12.5 on hand (about 12), reorder point 20, fill rate 86%, and 1.2 weeks of cover.",
            rows,
        )
        self.assertEqual(grounded, "grounded")

        fabricated = check_answer_grounding(
            "SKU-00001 has 12 on hand but needs 424242 units.",
            rows,
        )
        self.assertIsInstance(fabricated, list)
        self.assertIn(424242, fabricated)
        self.assertNotIn(12, fabricated)

    def test_grounding_accepts_counts_sums_gaps_and_thousands(self):
        rows = pd.DataFrame(
            [
                {"sku": "WIDGET", "current_inventory": 10, "reorder_point": 17},
                {"sku": "GADGET", "current_inventory": 15, "reorder_point": 9},
            ]
        )
        self.assertEqual(check_answer_grounding("On hand is 1,220.", pd.DataFrame([{"sku": "WIDGET", "current_inventory": 1220}])), "grounded")
        self.assertEqual(check_answer_grounding("1 item has 10 on hand.", rows.iloc[[0]]), "grounded")
        self.assertEqual(check_answer_grounding("Together that is 25.", rows), "grounded")
        self.assertEqual(check_answer_grounding("WIDGET is short by 7.", rows.iloc[[0]]), "grounded")
        self.assertIn(424242, check_answer_grounding("Order 424242.", rows))


class CopilotFallbackTests(unittest.TestCase):
    def test_fallback_runs_when_ollama_is_off(self):
        frame = _summary()
        agent = SupplyChainAgent(frame)
        question = "Which items are below reorder point?"
        with patch("main._gemma_chat", side_effect=ConnectionError("Ollama is not running")):
            reply = agent.query(question)

        self.assertEqual(reply.source, "rule-based")
        self.assertEqual(str(reply), agent._keyword_answer(question))
        self.assertIn("SKU-00001", str(reply))
        self.assertIn("SKU-00003", str(reply))
        self.assertNotIn("SKU-00002", str(reply))
        self.assertEqual(reply.unverified, [])
        self.assertIn("Ollama", reply.error)
        self.assertEqual(set(reply.rows["sku"]), {"SKU-00001", "SKU-00003"})

    def test_fallback_when_the_model_invents_a_number(self):
        frame = _summary()
        agent = SupplyChainAgent(frame)

        def fake(_messages):
            return {"message": {"content": "Buy 424242 units of SKU-00001 immediately."}}

        with patch("main._gemma_chat", fake):
            reply = agent.query("Show high risk items")

        self.assertEqual(reply.source, "rule-based")
        self.assertEqual(reply.unverified, [424242])
        self.assertNotIn("424242", str(reply))
        self.assertIn("SKU-00001", str(reply))
        self.assertIn("SKU-00003", str(reply))

    def test_grounded_answer_names_only_the_selected_rows(self):
        frame = _summary()
        agent = SupplyChainAgent(frame)
        question = "Which A-class items run out before their next delivery?"
        captured = {}

        def fake(messages):
            captured["prompt"] = messages[0]["content"]
            return {
                "message": {
                    "content": (
                        "SKU-00001 runs out before the next delivery: "
                        "10 on hand, reorder point 18, and 1.2 weeks of cover."
                    )
                }
            }

        with patch("main._gemma_chat", fake):
            reply = agent.query(question)

        self.assertEqual(reply.source, "grounded")
        self.assertIn("SKU-00001", reply.answer)
        self.assertEqual(list(reply.rows["sku"]), ["SKU-00001"])
        self.assertIn("ONLY", captured["prompt"])
        self.assertIn("SKU-00001", captured["prompt"])
        self.assertNotIn("SKU-00002", captured["prompt"])
        self.assertNotIn("SKU-00003", captured["prompt"])
        self.assertIn(question, captured["prompt"])

    def test_a_class_runout_question_selects_those_rows_only(self):
        selected = select_copilot_rows(
            _summary(),
            "Which A-class items run out before their next delivery?",
        )
        self.assertEqual(list(selected["sku"]), ["SKU-00001"])

    def test_question_without_a_signal_keeps_the_top_15_by_risk(self):
        rows = []
        for index in range(18):
            high = index >= 16
            rows.append(
                {
                    "sku": f"SKU-{index:05d}",
                    "abc_class": "C",
                    "stockout_risk": "High" if high else "Low",
                    "weeks_of_cover": 20 - index,
                    "current_inventory": 10 + index,
                    "reorder_point": 5,
                }
            )
        selected = select_copilot_rows(
            pd.DataFrame(rows),
            "What should the buyer review this morning?",
        )
        self.assertEqual(len(selected), 15)
        self.assertEqual(list(selected["sku"].head(2)), ["SKU-00017", "SKU-00016"])


if __name__ == "__main__":
    unittest.main()
