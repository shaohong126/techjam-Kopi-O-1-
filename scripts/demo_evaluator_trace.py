"""Show one official-format evaluator session without printing hidden labels.

This is a presentation helper, not part of the agent's inference path.  The
official evaluator owns the private sample and simulates the customer; the
agent receives only the same user messages and catalog access allowed during
evaluation.  Sample IDs and target labels are never printed.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator import local_evaluator as official  # noqa: E402
from starter.agent import Agent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "public_set.jsonl")
    parser.add_argument(
        "--scenario",
        choices=("buying", "browsing", "intent_override", "boundary"),
        default="intent_override",
    )
    args = parser.parse_args()

    samples = official.load_jsonl(args.dataset)
    sample = next(item for item in samples if item["scenario_type"] == args.scenario)
    catalog_ids, categories, products = official.catalog_index(args.catalog)
    agent = Agent(args.catalog)
    session_id = f"video_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])

    # Hidden fields remain inside this evaluator-side simulator. They are never
    # printed or made available to the agent.
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = official.materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = official.initial_message(
        effective,
        official.coarse_category(categories.get(target, [])),
        disclosed,
    )

    print(f"OFFICIAL-FORMAT SESSION TRACE | scenario={args.scenario}")
    print("Hidden target and sample ID are intentionally not displayed.")
    for turn in range(1, official.MAX_TURNS + 1):
        response = agent.respond(session_id, user_message, turn, official.TOP_K)
        state = agent._sessions[session_id]
        ranked = official.normalize_recommendations(
            response.get("recommendations"), catalog_ids
        )
        active = [f"{slot.attribute}={slot.key}" for slot in state.active_slots()]

        print(f"\nTURN {turn}")
        print(f"User: {user_message}")
        print(
            f"State: intent={state.intent.value}; strategy={state.strategy}; "
            f"candidate_count={state.last_candidate_count}; active={active}"
        )
        print(f"Agent question: {response['message']}")
        print(f"ask_attribute={response.get('ask_attribute')!r}")
        print("Recommendation:")
        for rank, asin in enumerate(ranked, start=1):
            title = str(products.get(asin, {}).get("title", "Untitled"))
            print(f"  {rank}. {title} [{asin}]")

        if override_applied and target in ranked:
            print("\nEvaluator: conversion detected; session complete.")
            break
        if turn == official.MAX_TURNS:
            print("\nEvaluator: turn budget exhausted.")
            break

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(
                override.get("message", "Actually, please ignore my earlier preference.")
            )
        else:
            user_message, boundary_used = official.customer_reply(
                effective,
                response.get("ask_attribute"),
                disclosed,
                boundary_used,
            )


if __name__ == "__main__":
    main()
