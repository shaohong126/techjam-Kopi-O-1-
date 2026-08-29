from __future__ import annotations

import math
from collections import Counter

from starter.models import SessionState
from starter.retrieval import CatalogIndex
from starter.understanding import classify_message


QUESTION_ATTRIBUTES = (
    "material", "color", "size", "style", "brand", "budget", "use_case", "feature",
)


class DialoguePolicy:
    """Selects clarification questions from candidate information gain."""

    def __init__(self, catalog: CatalogIndex) -> None:
        self.catalog = catalog

    def _attribute_information_gain(self, asins: list[str], attribute: str) -> float:
        sample = asins[:400]
        if not sample:
            return 0.0
        values = [
            self.catalog.attribute_keys(asin).get(attribute, [None])[0]
            for asin in sample
        ]
        observed = [value for value in values if value]
        if len(observed) < max(3, len(sample) // 5):
            return 0.0
        counts = Counter(observed)
        if len(counts) < 2:
            return 0.0
        total = len(observed)
        entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
        normalized = entropy / math.log(len(counts))
        coverage = total / len(sample)
        first_slot_coverage = sum(
            bool(self.catalog.product_constraint_keys.get(asin))
            and classify_message(self.catalog.product_constraint_keys[asin][0]) == attribute
            for asin in sample
        ) / len(sample)
        return coverage * normalized + 0.60 * first_slot_coverage

    def next_attribute(
        self,
        state: SessionState,
        turn: int,
        candidates: list[str],
    ) -> str | None:
        if turn >= 10:
            return None
        active_attributes = {slot.attribute for slot in state.active_slots()}
        if "other" not in state.declined and (turn <= 2 or len(state.active_slots()) < 4):
            if "other" not in state.asked:
                state.asked.append("other")
            return "other"

        options = [
            attribute
            for attribute in QUESTION_ATTRIBUTES
            if attribute not in state.declined
            and attribute not in active_attributes
            and attribute not in state.asked
        ]
        scored = sorted(
            (
                (self._attribute_information_gain(candidates, attribute), attribute)
                for attribute in options
            ),
            reverse=True,
        )
        best_score, best_attribute = scored[0] if scored else (0.0, "feature")

        if "other" not in state.declined and best_score < 0.30:
            attribute = "other"
        else:
            attribute = best_attribute
        if attribute != "other" and (attribute in state.asked or attribute in state.declined):
            return None
        if attribute not in state.asked:
            state.asked.append(attribute)
        return attribute

    @staticmethod
    def question_message(attribute: str | None, candidate_count: int) -> str:
        if attribute is None:
            return "I have enough context to keep refining the best match."
        if attribute == "other":
            return "What other requirement should I prioritize?"
        label = attribute.replace("_", " ")
        if candidate_count > 120:
            return f"I found many plausible options. Do you have a {label} preference?"
        return f"Do you have a {label} preference?"
