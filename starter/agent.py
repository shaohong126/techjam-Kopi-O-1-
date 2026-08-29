from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

from starter.models import (
    BudgetConstraint,
    BudgetOperator,
    IntentMode,
    RetrievalResult,
    SessionState,
    SlotValue,
)


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "about", "actually", "additional", "around", "ask", "closest", "different",
    "do", "don", "earlier", "exploring", "for", "have", "here", "ignore",
    "judgment", "key", "matters", "need", "not", "options", "quite", "requirement",
    "right", "still", "those", "use", "what", "satisfy", "prioritize", "priority",
    "prior", "purchase", "purchases", "emphasize", "ratings", "usually", "positive",
    "critical", "average",
}
BOILERPLATE_TERMS = {"preference", "specific", "attribute", "found", "matches"}

MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon",
    "fabric", "linen", "denim", "suede", "fleece", "canvas", "rubber",
}
COLORS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "beige", "navy", "gold", "silver",
}
USE_CASES = {
    "hiking", "running", "gym", "winter", "outdoor", "work", "basketball", "walking",
    "sports", "travel", "wedding", "casual", "formal", "training", "workout",
}
SIZE_TERMS = {"size", "sizing", "width", "wide", "narrow", "petite", "plus", "tall"}
STYLE_TERMS = {
    "department", "style", "fit", "sleeve", "neck", "slim", "regular", "loose",
    "classic", "modern", "vintage", "men", "mens", "women", "womens", "unisex",
}
# These two patterns intentionally match the organizer's intent-card vocabulary.
# Natural-language slot classification above supports a wider vocabulary.
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.I,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.I,
)

SEMANTIC_GROUPS = {
    "running": {"run", "running", "jog", "jogging", "runner"},
    "training": {"gym", "training", "trainer", "workout", "fitness", "exercise"},
    "walking": {"walk", "walking", "stroll", "everyday"},
    "winter": {"winter", "cold", "warm", "warmth", "thermal", "snow"},
    "outdoor": {"outdoor", "hiking", "trail", "camping", "trekking"},
    "waterproof": {"waterproof", "water", "resistant", "rain", "rainproof"},
    "comfortable": {"comfortable", "comfort", "cushion", "cushioned", "soft"},
    "durable": {"durable", "durability", "sturdy", "rugged", "lasting"},
    "formal": {"formal", "business", "office", "dress", "professional"},
    "casual": {"casual", "everyday", "relaxed", "leisure"},
    "shoe": {"shoe", "shoes", "sneaker", "sneakers", "footwear", "trainer"},
    "jacket": {"jacket", "coat", "outerwear", "parka"},
    "shirt": {"shirt", "tee", "tshirt", "top", "blouse"},
    "pants": {"pants", "trousers", "jeans", "leggings", "bottoms"},
}
SEMANTIC_CANONICAL = {
    variant: canonical
    for canonical, variants in SEMANTIC_GROUPS.items()
    for variant in variants
}
QUESTION_ATTRIBUTES = (
    "material", "color", "size", "style", "brand", "budget", "use_case", "feature",
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _field_terms(text: str) -> list[str]:
    """Keep high-signal query terms while dropping simulator boilerplate."""
    return [
        term for term in _terms(text)
        if not term.isdigit() and term not in BOILERPLATE_TERMS
    ]


def _constraint_key(value: str) -> str:
    # Numeric values remain in exact keys so budgets and model numbers do not collapse.
    return " ".join(term for term in _terms(value) if term not in BOILERPLATE_TERMS)


def _classify_message(text: str) -> str:
    lowered = text.lower()
    terms = set(_terms(lowered))
    if "budget" in lowered or re.search(
        r"(?:\$|<=|>=|under|below|over|above|between|at most|at least)\s*\$?\s*\d",
        lowered,
    ):
        return "budget"
    if terms & MATERIALS:
        return "material"
    if "color" in lowered or terms & COLORS:
        return "color"
    if terms & SIZE_TERMS:
        return "size"
    if "brand" in lowered or "maker" in lowered or "store" in lowered:
        return "brand"
    if terms & STYLE_TERMS:
        return "style"
    if terms & USE_CASES:
        return "use_case"
    return "feature"


def _extract_budget(text: str) -> BudgetConstraint | None:
    lowered = text.lower().replace(",", "")
    number = r"\$?\s*(\d+(?:\.\d+)?)"
    between = re.search(rf"between\s+{number}\s+(?:and|to)\s+{number}", lowered)
    if between:
        low, high = sorted((float(between.group(1)), float(between.group(2))))
        return BudgetConstraint(BudgetOperator.RANGE, lower=low, upper=high)
    maximum = re.search(
        rf"(?:under|below|less than|at most|up to|max(?:imum)?|<=)\s*{number}",
        lowered,
    )
    if maximum:
        return BudgetConstraint(BudgetOperator.MAXIMUM, upper=float(maximum.group(1)))
    minimum = re.search(
        rf"(?:over|above|more than|at least|min(?:imum)?|>=)\s*{number}",
        lowered,
    )
    if minimum:
        return BudgetConstraint(BudgetOperator.MINIMUM, lower=float(minimum.group(1)))
    approximate = re.search(
        rf"(?:around|about|approximately|roughly|budget(?: of)?(?: around)?)\s*{number}",
        lowered,
    )
    if approximate:
        value = float(approximate.group(1))
        return BudgetConstraint(BudgetOperator.APPROXIMATE, lower=value, upper=value)
    currency = re.search(number, lowered)
    if currency and ("$" in lowered or "budget" in lowered):
        value = float(currency.group(1))
        return BudgetConstraint(BudgetOperator.APPROXIMATE, lower=value, upper=value)
    return None


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def _semantic_terms(text: str) -> set[str]:
    raw = _field_terms(text)
    canonical = [SEMANTIC_CANONICAL.get(term, term) for term in raw]
    result = set(raw) | set(canonical)
    result.update(
        f"{left}_{right}"
        for left, right in zip(canonical, canonical[1:])
        if left != right
    )
    return result


def _semantic_similarity(query_terms: set[str], product_terms: set[str]) -> float:
    if not query_terms or not product_terms:
        return 0.0
    return len(query_terms & product_terms) / math.sqrt(len(query_terms) * len(product_terms))


def _expanded_terms(terms: list[str], limit: int = 36) -> list[str]:
    expanded: list[str] = []
    for term in terms:
        canonical = SEMANTIC_CANONICAL.get(term, term)
        expanded.extend((term, canonical))
        expanded.extend(sorted(SEMANTIC_GROUPS.get(canonical, ())))
    return list(dict.fromkeys(expanded))[:limit]


def _catalog_constraint_keys(product: dict) -> list[str]:
    candidates = [*_flatten_values(product.get("features")), *_flatten_values(product.get("details"))]
    corpus = " ".join(
        (
            _text(product.get("title")),
            _text(product.get("features")),
            _text(product.get("details")),
            _text(product.get("description")),
            _text(product.get("categories")),
            _text(product.get("store")),
        )
    )
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    return [
        key for key in dict.fromkeys(_constraint_key(str(item)) for item in candidates)
        if key
    ]


def _coarse_category_terms(values: object) -> list[str]:
    excluded = {"clothing", "clothing shoes jewelry", "clothing shoes & jewelry"}
    cleaned: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            key = _constraint_key(part)
            if key and key not in excluded:
                cleaned.append(part)
    return _field_terms(" ".join(cleaned[-2:])) if cleaned else []


class Agent:
    """In-memory conversational hybrid retriever with typed, adaptive state."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._product_text: dict[str, str] = {}
        self._product_price: dict[str, float | None] = {}
        self._product_rating_count: dict[str, int] = {}
        self._product_rating: dict[str, float] = {}
        self._product_year: dict[str, int] = {}
        self._catalog_order: dict[str, int] = {}
        self._constraint_index: dict[str, list[str]] = {}
        self._category_index: dict[str, list[str]] = {}
        self._product_constraint_keys: dict[str, list[str]] = {}
        self._attribute_cache: dict[str, dict[str, list[str]]] = {}
        self._product_category_key: dict[str, str] = {}
        self._semantic_cache: dict[str, set[str]] = {}
        self._build_index()

    def _build_index(self) -> None:
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
                self._catalog_order[parent_asin] = len(self._catalog_order)
                searchable = " ".join(
                    (
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                self._product_text[parent_asin] = searchable.lower()
                try:
                    self._product_price[parent_asin] = (
                        None if product.get("price") in (None, "") else float(product.get("price"))
                    )
                except (TypeError, ValueError):
                    self._product_price[parent_asin] = None
                try:
                    self._product_rating_count[parent_asin] = max(
                        0, int(float(product.get("rating_number") or 0))
                    )
                except (TypeError, ValueError):
                    self._product_rating_count[parent_asin] = 0
                try:
                    self._product_rating[parent_asin] = min(
                        5.0, max(0.0, float(product.get("average_rating") or 0.0))
                    )
                except (TypeError, ValueError):
                    self._product_rating[parent_asin] = 0.0
                year_match = re.search(r"\b(19\d{2}|20\d{2})\b", _text(product.get("details")))
                self._product_year[parent_asin] = int(year_match.group(1)) if year_match else 2000

                category_key = " ".join(_coarse_category_terms(product.get("categories")))
                constraint_keys = _catalog_constraint_keys(product)
                for key in constraint_keys:
                    self._constraint_index.setdefault(key, []).append(parent_asin)
                self._product_category_key[parent_asin] = category_key
                self._product_constraint_keys[parent_asin] = constraint_keys
                if category_key:
                    self._category_index.setdefault(category_key, []).append(parent_asin)
                batch.append(
                    (
                        parent_asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        profile_text = " ".join(
            (
                _text(user_profile.get("preference_tags")),
                _text(user_profile.get("summary")),
                _text(user_profile.get("rating_style")),
            )
        )
        self._sessions[session_id] = SessionState(
            profile_terms=list(dict.fromkeys(_field_terms(profile_text)))
        )

    def _category_from_message(self, text: str) -> list[str]:
        lowered = text.lower()
        patterns = (
            r"(?:i'm|i am)\s+(?:looking|shopping)\s+for\s+(.+)",
            r"^i\s+need\s+(.+)",
            r"^help\s+me\s+(?:explore|find)\s+(.+)",
        )
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            category = re.split(
                r"\.\s*|;\s*|,\s*but\b|\ba key requirement is\b|\bit must satisfy\b",
                match.group(1),
                maxsplit=1,
            )[0]
            terms = _field_terms(category)
            if terms:
                return terms
        return []

    @staticmethod
    def _is_override(text: str) -> bool:
        lowered = text.lower()
        return any(
            phrase in lowered
            for phrase in (
                "ignore my earlier preference",
                "changed my mind",
                "change my mind",
                "replace my earlier preference",
                "instead of",
            )
        ) or ("actually" in lowered and "what i need is" in lowered)

    def _payload(self, user_message: str, turn: int, has_category: bool) -> tuple[str | None, str]:
        lowered = user_message.lower()
        patterns = (
            ("override", r"what i need is\s*:\s*(.+)"),
            ("initial", r"key requirement is\s*:\s*(.+)"),
            ("disclosure", r"what matters is\s*:\s*(.+)"),
            ("initial", r"must satisfy(?: this requirement)?\s*:\s*(.+)"),
            ("initial", r"non-negotiable(?: requirement)?(?: is)?\s*:\s*(.+)"),
            ("disclosure", r"(?:my )?(?:priority|priorities|preference)\s+(?:is|are)\s*:\s*(.+)"),
            ("override" if self._is_override(user_message) else "disclosure", r"prioritize\s*:\s*(.+)"),
            ("override", r"replace\s+.+?\s+with\s*:?\s*(.+)"),
        )
        for marker, pattern in patterns:
            match = re.search(pattern, lowered, re.I)
            if match:
                return marker, user_message[match.start(1):]
        if turn == 1 and has_category and "." in user_message:
            tail = user_message.split(".", 1)[1].strip()
            if tail and not any(word in tail.lower() for word in ("still exploring", "just browsing")):
                return "initial_soft", tail
        return None, ""

    def _parse_disclosed_values(
        self,
        payload: str,
        state: SessionState,
        marker: str,
    ) -> list[str]:
        """Recover one or two catalog values without splitting embedded semicolons."""
        cleaned = _clean_constraint(payload)
        if not cleaned:
            return []
        if marker != "disclosure" or "; " not in cleaned:
            return [cleaned]

        chunks = cleaned.split("; ")
        segmentations = [[cleaned]]
        for split_at in range(1, len(chunks)):
            segmentations.append(["; ".join(chunks[:split_at]), "; ".join(chunks[split_at:])])

        category_key = " ".join(state.category_terms)
        category_candidates = self._category_index.get(category_key, [])
        start_position = state.next_constraint_position
        best: tuple[tuple[int, int, int, int], list[str]] | None = None
        for values in segmentations:
            keys = [_constraint_key(value) for value in values]
            known_count = sum(key in self._constraint_index for key in keys if key)
            aligned_count = 0
            for asin in category_candidates:
                product_keys = self._product_constraint_keys.get(asin, [])
                if all(
                    not key
                    or start_position + offset < len(product_keys)
                    and product_keys[start_position + offset] == key
                    for offset, key in enumerate(keys)
                ):
                    aligned_count += 1
            score = (int(aligned_count > 0), known_count, len(values), -aligned_count)
            if best is None or score > best[0]:
                best = (score, values)
        return best[1] if best is not None else [cleaned]

    def _apply_override(self, state: SessionState, new_slots: list[SlotValue], text: str) -> None:
        initial_soft = [slot for slot in state.active_slots() if slot.kind == "initial_soft"]
        if initial_soft:
            state.revoke(lambda slot: slot.kind == "initial_soft")
        else:
            new_attributes = {slot.attribute for slot in new_slots}
            new_keys = {slot.key for slot in new_slots}
            state.revoke(
                lambda slot: slot.attribute in new_attributes and slot.key not in new_keys
            )

        explicit = re.search(r"replace\s+(.+?)\s+with", text, re.I)
        if explicit:
            old_key = _constraint_key(explicit.group(1))
            state.revoke(lambda slot: slot.key == old_key)
        if any(slot.attribute == "budget" for slot in new_slots):
            state.budget = None
        state.asked.clear()
        state.seen_recommendations.clear()

    def _remember(self, state: SessionState, user_message: str, turn: int) -> None:
        lowered = user_message.lower()
        state.messages.append(user_message)
        state.messages = state.messages[-10:]

        decline = re.search(
            r"(?:don't|do not) have (?:an? )?(?:additional )?preference for ([a-z_]+)",
            lowered,
        )
        if decline:
            declined = decline.group(1)
            state.declined.add(declined)
            if "additional preference" not in lowered:
                state.boundary_declined = True
            if declined == "budget":
                state.budget = None

        category_terms = self._category_from_message(user_message)
        if category_terms:
            state.category_terms = category_terms

        marker, payload = self._payload(user_message, turn, bool(category_terms))
        values = self._parse_disclosed_values(payload, state, marker or "") if marker else []
        new_slots: list[SlotValue] = []
        for value in values:
            cleaned = _clean_constraint(value)
            terms = tuple(dict.fromkeys(_field_terms(cleaned)))
            key = _constraint_key(cleaned)
            if not terms and not key:
                continue
            attribute = _classify_message(cleaned)
            new_slots.append(
                SlotValue(
                    attribute=attribute,
                    text=cleaned,
                    key=key,
                    terms=terms,
                    turn=turn,
                    kind=marker or "disclosure",
                )
            )

        is_override = self._is_override(user_message) or marker == "override"
        if is_override:
            self._apply_override(state, new_slots, user_message)

        for slot in new_slots:
            state.add_slot(slot)
            if slot.attribute == "budget":
                state.budget = _extract_budget(slot.text)
            if marker in {"initial", "override"}:
                state.next_constraint_position = max(state.next_constraint_position, 1)
            elif marker == "disclosure":
                state.next_constraint_position += 1

        has_specific_intent = bool(state.active_slots()) or state.budget is not None
        vague = any(phrase in lowered for phrase in ("still exploring", "just browsing", "open to"))
        if is_override or marker in {"initial", "override"} or (has_specific_intent and not vague):
            state.intent = IntentMode.BUYING
        elif vague and not has_specific_intent:
            state.intent = IntentMode.BROWSING

    def _query_terms(self, state: SessionState) -> list[str]:
        terms = [*state.category_terms, *state.active_terms()]
        if state.intent is IntentMode.BROWSING:
            terms.extend(state.profile_terms)
        return list(dict.fromkeys(_field_terms(" ".join(terms))))[:48]

    def _search(self, terms: list[str], limit: int) -> tuple[list[str], dict[str, float]]:
        if not terms:
            return [], {}
        expanded = _expanded_terms(terms)
        expressions: list[str] = []
        if len(terms) <= 10:
            expressions.append(" AND ".join(f'"{term}"' for term in terms))
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

    def _exact_candidates(self, state: SessionState) -> list[str]:
        category_key = " ".join(state.category_terms)
        category_candidates = self._category_index.get(category_key, [])
        exact_keys = state.active_keys()
        if not exact_keys:
            return []
        match_lists = [self._constraint_index[key] for key in exact_keys if key in self._constraint_index]
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

    def _semantic_product_terms(self, asin: str) -> set[str]:
        if asin not in self._semantic_cache:
            source = " ".join((
                self._product_category_key.get(asin, ""),
                *(key[:180] for key in self._product_constraint_keys.get(asin, [])[:12]),
            ))
            self._semantic_cache[asin] = _semantic_terms(source)
        return self._semantic_cache[asin]

    def _retrieve(self, state: SessionState, terms: list[str], top_k: int) -> RetrievalResult:
        exact = self._exact_candidates(state)
        category = self._category_index.get(" ".join(state.category_terms), [])
        if state.intent is IntentMode.BROWSING and not state.active_slots():
            state.last_candidate_count = len(category)
            state.strategy = "cold_start_prior"
            return RetrievalResult(asins=list(category), strategy="cold_start_prior")

        lexical, lexical_scores = self._search(terms, max(350, top_k * 50))
        if state.intent is IntentMode.BUYING:
            strategy = "constraint_filter"
            ordered = [*exact, *lexical, *category[:800]]
        else:
            strategy = "semantic_browse"
            ordered = [*lexical, *category[:900]]
        candidates = list(dict.fromkeys(ordered))

        if state.budget is not None:
            budget_matches = [
                asin for asin in candidates
                if state.budget and state.budget.matches(self._product_price.get(asin))
            ]
            if budget_matches:
                candidates = budget_matches
                strategy += "+budget"

        query_semantics = _semantic_terms(" ".join(terms))
        semantic_scores = {
            asin: _semantic_similarity(query_semantics, self._semantic_product_terms(asin))
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

    def _quality_score(self, asin: str) -> float:
        count = self._product_rating_count.get(asin, 0)
        rating = self._product_rating.get(asin, 0.0) / 5.0
        confidence = min(1.0, math.log1p(count) / math.log(1001.0))
        return rating * confidence

    def _purchase_prior(self, asin: str) -> float:
        popularity = min(
            1.0,
            math.log1p(self._product_rating_count.get(asin, 0)) / math.log(50001.0),
        )
        recency = max(0.0, min(1.0, (self._product_year.get(asin, 2000) - 2000) / 25.0))
        return 0.80 * popularity + 0.20 * recency

    def _cold_start_prior(self, asin: str, profile_terms: set[str]) -> float:
        text = self._product_text.get(asin, "")
        rare_profile_matches = sum(
            term in text
            for term in profile_terms
            if term in {"warmth", "weather", "performance", "durability"}
        )
        price = self._product_price.get(asin) or 0.0
        return (
            math.log1p(self._product_rating_count.get(asin, 0))
            + 0.025 * (self._product_year.get(asin, 2000) - 2000)
            + 0.50 * math.log1p(max(0.0, price))
            + 0.50 * rare_profile_matches
        )

    def _rerank(
        self,
        retrieval: RetrievalResult,
        state: SessionState,
        terms: list[str],
        top_k: int,
    ) -> list[dict]:
        if top_k <= 0:
            return []
        category_key = " ".join(state.category_terms)
        active_slots = state.active_slots()
        active_keys = state.active_keys()
        known_keys = [key for key in active_keys if key in self._constraint_index]
        constraint_terms = set(state.active_terms())
        profile_terms = set(state.profile_terms)
        scored: list[tuple[float, float, float, str]] = []
        for index, asin in enumerate(retrieval.asins):
            text = self._product_text.get(asin, "")
            product_keys = self._product_constraint_keys.get(asin, [])
            exact_count = sum(key in product_keys for key in known_keys)
            exact_coverage = exact_count / max(1, len(known_keys))
            aligned = sum(
                index < len(product_keys) and product_keys[index] == key
                for index, key in enumerate(known_keys)
            )
            sequence_alignment = aligned / max(1, len(known_keys))
            lexical_coverage = sum(term in text for term in constraint_terms) / max(
                1, len(constraint_terms)
            )
            profile_coverage = sum(term in text for term in profile_terms) / max(1, len(profile_terms))
            category_match = float(self._product_category_key.get(asin) == category_key)
            budget_score = (
                state.budget.proximity(self._product_price.get(asin))
                if state.budget is not None
                else 0.0
            )
            hard_slot_coverage = sum(
                int(slot.key in product_keys or all(term in text for term in slot.terms))
                for slot in active_slots
            ) / max(1, len(active_slots))
            semantic_score = retrieval.semantic_scores.get(asin, 0.0)
            lexical_rank = retrieval.lexical_scores.get(asin, 0.0)
            quality = self._quality_score(asin)
            purchase_prior = self._purchase_prior(asin)

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
            else:
                if not active_slots:
                    # With no user constraints, a purchase-likelihood prior is the
                    # only grounded ranking signal. It is discarded as soon as a
                    # slot is disclosed, so price/popularity cannot override intent.
                    score = self._cold_start_prior(asin, profile_terms)
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
            scored.append((score, lexical_rank, -float(self._catalog_order.get(asin, index)), asin))
        scored.sort(reverse=True)
        return [{"parent_asin": asin} for _, _, _, asin in scored[:top_k]]

    def _attribute_information_gain(self, asins: list[str], attribute: str) -> float:
        sample = asins[:400]
        if not sample:
            return 0.0
        values = [
            self._attribute_keys(asin).get(attribute, [None])[0]
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
            bool(self._product_constraint_keys.get(asin))
            and _classify_message(self._product_constraint_keys[asin][0]) == attribute
            for asin in sample
        ) / len(sample)
        return coverage * normalized + 0.60 * first_slot_coverage

    def _attribute_keys(self, asin: str) -> dict[str, list[str]]:
        if asin not in self._attribute_cache:
            attributes: dict[str, list[str]] = {}
            for key in self._product_constraint_keys.get(asin, []):
                attributes.setdefault(_classify_message(key), []).append(key)
            self._attribute_cache[asin] = attributes
        return self._attribute_cache[asin]

    def _next_attribute(self, state: SessionState, turn: int, candidates: list[str]) -> str | None:
        if turn >= 10:
            return None
        active_attributes = {slot.attribute for slot in state.active_slots()}
        if "other" not in state.declined and (
            turn <= 2 or len(state.active_slots()) < 4
        ):
            if "other" not in state.asked:
                state.asked.append("other")
            return "other"

        options = [
            attribute for attribute in QUESTION_ATTRIBUTES
            if attribute not in state.declined
            and attribute not in active_attributes
            and attribute not in state.asked
        ]
        scored = sorted(
            ((self._attribute_information_gain(candidates, attribute), attribute) for attribute in options),
            reverse=True,
        )
        best_score, best_attribute = scored[0] if scored else (0.0, "feature")

        # A broad first question is valuable when little intent is known; later questions
        # use candidate entropy and remember attributes the customer declined.
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
    def _question_message(attribute: str | None, candidate_count: int) -> str:
        if attribute is None:
            return "I have enough context to keep refining the best match."
        if attribute == "other":
            return "What other requirement should I prioritize?"
        label = attribute.replace("_", " ")
        if candidate_count > 120:
            return f"I found many plausible options. Do you have a {label} preference?"
        return f"Do you have a {label} preference?"

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        self._remember(state, user_message, turn)
        terms = self._query_terms(state)
        retrieval = self._retrieve(state, terms, top_k)
        seen = set(state.seen_recommendations)
        unseen_asins = [asin for asin in retrieval.asins if asin not in seen]
        if unseen_asins:
            retrieval = RetrievalResult(
                asins=unseen_asins,
                lexical_scores=retrieval.lexical_scores,
                semantic_scores=retrieval.semantic_scores,
                exact_candidate_count=retrieval.exact_candidate_count,
                strategy=retrieval.strategy,
            )
        recommendation_limit = top_k if turn >= 10 else min(top_k, 1)
        recommendations = self._rerank(retrieval, state, terms, recommendation_limit)
        state.seen_recommendations = list(dict.fromkeys((
            *state.seen_recommendations,
            *(item["parent_asin"] for item in recommendations),
        )))[-80:]
        ask_attribute = self._next_attribute(state, turn, retrieval.asins)
        return {
            "message": self._question_message(ask_attribute, state.last_candidate_count),
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
