from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.models import BudgetOperator, IntentMode


CATALOG = [
    {
        "parent_asin": "RUN",
        "title": "Red running shoe",
        "features": ["cotton", "waterproof trail cushioning"],
        "details": {"color": "red", "department": "womens"},
        "description": ["comfortable footwear for running and jogging"],
        "categories": ["Clothing", "Shoes"],
        "store": "Stride",
        "average_rating": 4.4,
        "rating_number": 120,
        "price": 25.0,
    },
    {
        "parent_asin": "FORMAL",
        "title": "Black formal leather shoe",
        "features": ["leather", "office dress style"],
        "details": {"color": "black", "department": "mens"},
        "description": ["professional business footwear"],
        "categories": ["Clothing", "Shoes"],
        "store": "Executive",
        "average_rating": 4.9,
        "rating_number": 20000,
        "price": 220.0,
    },
    {
        "parent_asin": "WALK",
        "title": "Blue walking shoe",
        "features": ["nylon", "soft everyday comfort"],
        "details": {"color": "blue", "department": "unisex"},
        "description": ["casual walking footwear"],
        "categories": ["Clothing", "Shoes"],
        "store": "Daily",
        "average_rating": 4.2,
        "rating_number": 300,
        "price": 75.0,
    },
]


class AgentBehaviorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.catalog_path = Path(cls.temp_directory.name) / "catalog.jsonl"
        cls.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in CATALOG),
            encoding="utf-8",
        )
        cls.agent = Agent(cls.catalog_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_directory.cleanup()

    def reset(self, session_id: str) -> None:
        self.agent.reset(session_id, {
            "preference_tags": ["comfort", "durability"],
            "summary": "Prior purchases emphasize comfort and durability.",
            "rating_style": "usually positive",
        })

    def test_routes_browsing_and_buying_separately(self) -> None:
        self.reset("browse")
        self.agent.respond("browse", "I'm looking for Shoes, but I'm still exploring.", 1, 10)
        browse_state = self.agent._sessions["browse"]
        self.assertEqual(browse_state.intent, IntentMode.BROWSING)
        self.assertEqual(browse_state.strategy, "cold_start_prior")

        self.reset("buy")
        self.agent.respond("buy", "I'm looking for Shoes. A key requirement is: leather.", 1, 10)
        buy_state = self.agent._sessions["buy"]
        self.assertEqual(buy_state.intent, IntentMode.BUYING)
        self.assertTrue(buy_state.strategy.startswith("constraint_filter"))

    def test_paraphrased_constraint_is_extracted(self) -> None:
        self.reset("paraphrase")
        response = self.agent.respond(
            "paraphrase",
            "I'm looking for Shoes. It must satisfy this requirement: waterproof.",
            1,
            10,
        )
        self.assertEqual(response["recommendations"], [{"parent_asin": "RUN"}])
        self.assertIn("waterproof", self.agent._sessions["paraphrase"].active_keys())

    def test_budget_constraints_change_the_recommendation(self) -> None:
        self.reset("cheap")
        cheap = self.agent.respond(
            "cheap", "I'm looking for Shoes. A key requirement is: under $50.", 1, 10
        )
        self.assertEqual(cheap["recommendations"], [{"parent_asin": "RUN"}])
        self.assertEqual(
            self.agent._sessions["cheap"].budget.operator,
            BudgetOperator.MAXIMUM,
        )

        self.reset("premium")
        premium = self.agent.respond(
            "premium", "I'm looking for Shoes. A key requirement is: at least $150.", 1, 10
        )
        self.assertEqual(premium["recommendations"], [{"parent_asin": "FORMAL"}])
        self.assertEqual(
            self.agent._sessions["premium"].budget.operator,
            BudgetOperator.MINIMUM,
        )

    def test_override_revokes_the_conflicting_slot(self) -> None:
        self.reset("override")
        self.agent.respond(
            "override", "I'm looking for Shoes. A key requirement is: cotton.", 1, 10
        )
        response = self.agent.respond(
            "override",
            "Actually, ignore my earlier preference. What I need is: leather.",
            2,
            10,
        )
        state = self.agent._sessions["override"]
        self.assertNotIn("cotton", state.active_keys())
        self.assertIn("leather", state.active_keys())
        self.assertEqual(response["recommendations"], [{"parent_asin": "FORMAL"}])

    def test_confirmed_fact_survives_unrelated_override(self) -> None:
        self.reset("provenance")
        self.agent.respond("provenance", "I'm looking for Shoes. waterproof", 1, 10)
        self.agent.respond(
            "provenance", "For that, what matters is: waterproof.", 2, 10
        )
        self.agent.respond(
            "provenance",
            "Actually, ignore my earlier preference. What I need is: cotton.",
            3,
            10,
        )
        state = self.agent._sessions["provenance"]
        self.assertIn("waterproof", state.active_keys())
        self.assertIn("cotton", state.active_keys())

    def test_boundary_decline_is_not_repeated(self) -> None:
        self.reset("boundary")
        first = self.agent.respond(
            "boundary", "I'm looking for Shoes, but I'm still exploring.", 1, 10
        )
        second = self.agent.respond(
            "boundary",
            "I don't have a preference for other; please use your judgment.",
            2,
            10,
        )
        self.assertEqual(first["ask_attribute"], "other")
        self.assertNotEqual(second["ask_attribute"], "other")
        self.assertIn("other", self.agent._sessions["boundary"].declined)
        self.assertTrue(self.agent._sessions["boundary"].boundary_declined)

    def test_semantic_synonym_route_matches_running(self) -> None:
        self.reset("semantic")
        response = self.agent.respond(
            "semantic",
            "I'm looking for Shoes. It must satisfy this requirement: jogging.",
            1,
            10,
        )
        self.assertEqual(response["recommendations"], [{"parent_asin": "RUN"}])

    def test_returns_one_recommendation_before_final_turn(self) -> None:
        self.reset("limit")
        response = self.agent.respond(
            "limit", "I'm looking for Shoes, but I'm still exploring.", 1, 10
        )
        self.assertEqual(len(response["recommendations"]), 1)


if __name__ == "__main__":
    unittest.main()
