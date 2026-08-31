from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from starter.models import IntentMode, RetrievalResult, SessionState
from starter.retrieval import CatalogIndex


@dataclass(frozen=True)
class RankedCandidate:
    parent_asin: str
    score: float
    exact_coverage: float
    sequence_alignment: float
    lexical_coverage: float
    hard_slot_coverage: float
    budget_match: float


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
        return [
            {"parent_asin": candidate.parent_asin}
            for candidate in self.rank(retrieval, state)[:top_k]
        ]

    def rank(
        self,
        retrieval: RetrievalResult,
        state: SessionState,
    ) -> list[RankedCandidate]:
        category_key = " ".join(state.category_terms)
        active_slots = state.active_slots()
        active_keys = state.active_keys()
        known_keys = [
            key for key in active_keys
            if key in self.catalog.constraint_index
        ]
        constraint_terms = set(state.active_terms())
        profile_terms = set(state.profile_terms)
        scored: list[tuple[float, float, float, str, RankedCandidate]] = []
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
            budget_match = float(
                state.budget is not None
                and state.budget.matches(self.catalog.product_price.get(asin))
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
            candidate = RankedCandidate(
                parent_asin=asin,
                score=score,
                exact_coverage=exact_coverage,
                sequence_alignment=sequence_alignment,
                lexical_coverage=lexical_coverage,
                hard_slot_coverage=hard_slot_coverage,
                budget_match=budget_match,
            )
            scored.append(
                (
                    score,
                    lexical_rank,
                    -float(self.catalog.catalog_order.get(asin, candidate_index)),
                    asin,
                    candidate,
                )
            )
        scored.sort(reverse=True)
        return [candidate for _, _, _, _, candidate in scored]

    def with_confidence(
        self,
        ranked: list[RankedCandidate],
        state: SessionState,
        limit: int,
    ) -> list[tuple[RankedCandidate, float]]:
        """Attach query-local confidence without comparing raw scores across modes."""
        if limit <= 0 or not ranked:
            return []
        confirmed_slots = [
            slot for slot in state.active_slots()
            if slot.kind != "initial_soft"
        ]
        if not confirmed_slots:
            return [(candidate, 0.0) for candidate in ranked[:limit]]

        score_sample = [candidate.score for candidate in ranked[:50]]
        scale = statistics.pstdev(score_sample) if len(score_sample) > 1 else 0.0
        scale = max(scale, abs(ranked[0].score) * 0.02, 1e-6)
        top_score = ranked[0].score
        evidence = min(1.0, 0.65 + 0.10 * len(confirmed_slots))
        known_keys = [
            key for key in state.active_keys()
            if key in self.catalog.constraint_index
        ]

        results: list[tuple[RankedCandidate, float]] = []
        previous_confidence = 1.0
        for candidate in ranked[:limit]:
            support = [candidate.hard_slot_coverage, candidate.lexical_coverage]
            if known_keys:
                support.extend((candidate.exact_coverage, candidate.sequence_alignment))
            if state.budget is not None:
                support.append(candidate.budget_match)
            mean_support = statistics.fmean(support)
            constraint_support = 0.70 * max(support) + 0.30 * mean_support
            if state.budget is not None:
                constraint_support = max(
                    constraint_support,
                    candidate.budget_match,
                )
            relative_score = math.exp(
                max(-30.0, min(0.0, (candidate.score - top_score) / scale))
            )
            confidence = evidence * (
                0.55 * constraint_support + 0.45 * relative_score
            )
            confidence = min(
                previous_confidence,
                max(0.0, min(1.0, confidence)),
            )
            previous_confidence = confidence
            results.append((candidate, confidence))
        return results
