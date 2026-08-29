from __future__ import annotations

from starter.models import IntentMode, RetrievalResult, SessionState
from starter.retrieval import CatalogIndex


class ProductRanker:
    """Intent-aware scoring for buying and browsing candidate sets."""

    def __init__(self, catalog: CatalogIndex) -> None:
        self.catalog = catalog

    def rerank(
        self,
        retrieval: RetrievalResult,
        state: SessionState,
        top_k: int,
    ) -> list[dict]:
        if top_k <= 0:
            return []
        category_key = " ".join(state.category_terms)
        active_slots = state.active_slots()
        active_keys = state.active_keys()
        known_keys = [
            key for key in active_keys
            if key in self.catalog.constraint_index
        ]
        constraint_terms = set(state.active_terms())
        profile_terms = set(state.profile_terms)
        scored: list[tuple[float, float, float, str]] = []
        for candidate_index, asin in enumerate(retrieval.asins):
            text = self.catalog.product_text.get(asin, "")
            product_keys = self.catalog.product_constraint_keys.get(asin, [])
            exact_count = sum(key in product_keys for key in known_keys)
            exact_coverage = exact_count / max(1, len(known_keys))
            aligned = sum(
                slot_index < len(product_keys) and product_keys[slot_index] == key
                for slot_index, key in enumerate(known_keys)
            )
            sequence_alignment = aligned / max(1, len(known_keys))
            lexical_coverage = sum(term in text for term in constraint_terms) / max(
                1, len(constraint_terms)
            )
            profile_coverage = sum(term in text for term in profile_terms) / max(
                1, len(profile_terms)
            )
            category_match = float(
                self.catalog.product_category_key.get(asin) == category_key
            )
            budget_score = (
                state.budget.proximity(self.catalog.product_price.get(asin))
                if state.budget is not None
                else 0.0
            )
            hard_slot_coverage = sum(
                int(slot.key in product_keys or all(term in text for term in slot.terms))
                for slot in active_slots
            ) / max(1, len(active_slots))
            semantic_score = retrieval.semantic_scores.get(asin, 0.0)
            lexical_rank = retrieval.lexical_scores.get(asin, 0.0)
            quality = self.catalog.quality_score(asin)
            purchase_prior = self.catalog.purchase_prior(asin)

            if state.intent is IntentMode.BUYING:
                if state.boundary_declined:
                    semantic_weight = 1.2
                    lexical_rank_weight = 0.9
                    purchase_weight = 1.8
                else:
                    semantic_weight = 0.4 if known_keys else 2.0
                    lexical_rank_weight = 0.0 if known_keys else 1.2
                    purchase_weight = 2.6 if known_keys else 0.8
                score = (
                    4.6 * exact_coverage
                    + 3.2 * sequence_alignment
                    + 2.4 * hard_slot_coverage
                    + 1.4 * category_match
                    + 1.5 * lexical_coverage
                    + semantic_weight * semantic_score
                    + lexical_rank_weight * lexical_rank
                    + 2.2 * budget_score
                    + 0.35 * profile_coverage
                    + 0.25 * quality
                    + purchase_weight * purchase_prior
                )
            elif not active_slots:
                score = self.catalog.cold_start_prior(asin, profile_terms)
            else:
                score = (
                    2.8 * exact_coverage
                    + 2.4 * sequence_alignment
                    + 1.5 * hard_slot_coverage
                    + 1.3 * category_match
                    + 1.4 * lexical_coverage
                    + 2.3 * semantic_score
                    + 1.0 * lexical_rank
                    + 0.75 * profile_coverage
                    + 0.45 * quality
                    + 1.0 * purchase_prior
                )
            scored.append(
                (
                    score,
                    lexical_rank,
                    -float(self.catalog.catalog_order.get(asin, candidate_index)),
                    asin,
                )
            )
        scored.sort(reverse=True)
        return [{"parent_asin": asin} for _, _, _, asin in scored[:top_k]]
