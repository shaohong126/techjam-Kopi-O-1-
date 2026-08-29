from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path

from starter.models import IntentMode, RetrievalResult, SessionState
from starter.understanding import (
    catalog_constraint_keys,
    classify_message,
    coarse_category_terms,
    expanded_terms,
    semantic_similarity,
    semantic_terms,
    text_value,
)


class CatalogIndex:
    """In-memory catalog indexes and multi-route candidate retrieval."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.product_text: dict[str, str] = {}
        self.product_price: dict[str, float | None] = {}
        self.product_rating_count: dict[str, int] = {}
        self.product_rating: dict[str, float] = {}
        self.product_year: dict[str, int] = {}
        self.catalog_order: dict[str, int] = {}
        self.constraint_index: dict[str, list[str]] = {}
        self.category_index: dict[str, list[str]] = {}
        self.product_constraint_keys: dict[str, list[str]] = {}
        self.product_category_key: dict[str, str] = {}
        self._attribute_cache: dict[str, dict[str, list[str]]] = {}
        self._semantic_cache: dict[str, set[str]] = {}
        self._build()

    def _build(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                self.catalog_order[parent_asin] = len(self.catalog_order)
                searchable = " ".join(
                    (
                        text_value(product.get("title")),
                        text_value(product.get("categories")),
                        text_value(product.get("features")),
                        text_value(product.get("details")),
                        text_value(product.get("store")),
                        text_value(product.get("description")),
                    )
                )
                self.product_text[parent_asin] = searchable.lower()
                self.product_price[parent_asin] = self._float_or_none(product.get("price"))
                self.product_rating_count[parent_asin] = self._rating_count(
                    product.get("rating_number")
                )
                self.product_rating[parent_asin] = self._rating(
                    product.get("average_rating")
                )
                year_match = re.search(
                    r"\b(19\d{2}|20\d{2})\b",
                    text_value(product.get("details")),
                )
                self.product_year[parent_asin] = int(year_match.group(1)) if year_match else 2000

                category_key = " ".join(coarse_category_terms(product.get("categories")))
                constraint_keys = catalog_constraint_keys(product)
                for key in constraint_keys:
                    self.constraint_index.setdefault(key, []).append(parent_asin)
                self.product_category_key[parent_asin] = category_key
                self.product_constraint_keys[parent_asin] = constraint_keys
                if category_key:
                    self.category_index.setdefault(category_key, []).append(parent_asin)
                batch.append(
                    (
                        parent_asin,
                        text_value(product.get("title")),
                        text_value(product.get("categories")),
                        text_value(product.get("features")),
                        text_value(product.get("details")),
                        text_value(product.get("store")),
                        text_value(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    @staticmethod
    def _float_or_none(value: object) -> float | None:
        try:
            return None if value in (None, "") else float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _rating_count(value: object) -> int:
        try:
            return max(0, int(float(value or 0)))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _rating(value: object) -> float:
        try:
            return min(5.0, max(0.0, float(value or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    def search(self, query_terms: list[str], limit: int) -> tuple[list[str], dict[str, float]]:
        if not query_terms:
            return [], {}
        expanded = expanded_terms(query_terms)
        expressions: list[str] = []
        if len(query_terms) <= 10:
            expressions.append(" AND ".join(f'"{term}"' for term in query_terms))
        expressions.append(" OR ".join(f'"{term}"' for term in expanded))
        results: list[str] = []
        scores: dict[str, float] = {}
        for expression in expressions:
            try:
                rows = self.connection.execute(
                    "SELECT parent_asin FROM products WHERE products MATCH ? "
                    "ORDER BY bm25(products, 0.0, 7.0, 4.5, 3.0, 2.5, 2.0, 1.0) LIMIT ?",
                    (expression, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            denominator = max(1, len(rows) - 1)
            for rank, row in enumerate(rows):
                asin = str(row[0])
                scores[asin] = max(scores.get(asin, 0.0), 1.0 - rank / denominator)
                results.append(asin)
        return list(dict.fromkeys(results)), scores

    def exact_candidates(self, state: SessionState) -> list[str]:
        category_key = " ".join(state.category_terms)
        category_candidates = self.category_index.get(category_key, [])
        exact_keys = state.active_keys()
        if not exact_keys:
            return []
        match_lists = [
            self.constraint_index[key]
            for key in exact_keys
            if key in self.constraint_index
        ]
        if not match_lists:
            return []
        smallest = min(match_lists, key=len)
        intersection = set(smallest)
        for matches in match_lists:
            intersection.intersection_update(matches)
            if not intersection:
                break
        ordered = (
            [asin for asin in smallest if asin in intersection]
            if intersection
            else list(dict.fromkeys(asin for matches in match_lists for asin in matches))
        )
        if category_candidates:
            category_set = set(category_candidates)
            in_category = [asin for asin in ordered if asin in category_set]
            if in_category:
                return in_category
        return ordered

    def semantic_product_terms(self, asin: str) -> set[str]:
        if asin not in self._semantic_cache:
            source = " ".join(
                (
                    self.product_category_key.get(asin, ""),
                    *(key[:180] for key in self.product_constraint_keys.get(asin, [])[:12]),
                )
            )
            self._semantic_cache[asin] = semantic_terms(source)
        return self._semantic_cache[asin]

    def retrieve(
        self,
        state: SessionState,
        query_terms: list[str],
        top_k: int,
    ) -> RetrievalResult:
        exact = self.exact_candidates(state)
        category = self.category_index.get(" ".join(state.category_terms), [])
        if state.intent is IntentMode.BROWSING and not state.active_slots():
            state.last_candidate_count = len(category)
            state.strategy = "cold_start_prior"
            return RetrievalResult(asins=list(category), strategy="cold_start_prior")

        lexical, lexical_scores = self.search(query_terms, max(350, top_k * 50))
        if state.intent is IntentMode.BUYING:
            strategy = "constraint_filter"
            ordered = [*exact, *lexical, *category[:800]]
        else:
            strategy = "semantic_browse"
            ordered = [*lexical, *category[:900]]
        candidates = list(dict.fromkeys(ordered))

        if state.budget is not None:
            budget_matches = [
                asin
                for asin in candidates
                if state.budget and state.budget.matches(self.product_price.get(asin))
            ]
            if budget_matches:
                candidates = budget_matches
                strategy += "+budget"

        query_semantics = semantic_terms(" ".join(query_terms))
        semantic_scores = {
            asin: semantic_similarity(query_semantics, self.semantic_product_terms(asin))
            for asin in candidates
        }
        state.last_candidate_count = len(candidates)
        state.strategy = strategy
        return RetrievalResult(
            asins=candidates,
            lexical_scores=lexical_scores,
            semantic_scores=semantic_scores,
            exact_candidate_count=len(exact),
            strategy=strategy,
        )

    def quality_score(self, asin: str) -> float:
        count = self.product_rating_count.get(asin, 0)
        rating = self.product_rating.get(asin, 0.0) / 5.0
        confidence = min(1.0, math.log1p(count) / math.log(1001.0))
        return rating * confidence

    def purchase_prior(self, asin: str) -> float:
        popularity = min(
            1.0,
            math.log1p(self.product_rating_count.get(asin, 0)) / math.log(50001.0),
        )
        recency = max(0.0, min(1.0, (self.product_year.get(asin, 2000) - 2000) / 25.0))
        return 0.80 * popularity + 0.20 * recency

    def cold_start_prior(self, asin: str, profile_terms: set[str]) -> float:
        text = self.product_text.get(asin, "")
        rare_profile_matches = sum(
            term in text
            for term in profile_terms
            if term in {"warmth", "weather", "performance", "durability"}
        )
        price = self.product_price.get(asin) or 0.0
        return (
            math.log1p(self.product_rating_count.get(asin, 0))
            + 0.025 * (self.product_year.get(asin, 2000) - 2000)
            + 0.50 * math.log1p(max(0.0, price))
            + 0.50 * rare_profile_matches
        )

    def attribute_keys(self, asin: str) -> dict[str, list[str]]:
        if asin not in self._attribute_cache:
            attributes: dict[str, list[str]] = {}
            for key in self.product_constraint_keys.get(asin, []):
                attributes.setdefault(classify_message(key), []).append(key)
            self._attribute_cache[asin] = attributes
        return self._attribute_cache[asin]
