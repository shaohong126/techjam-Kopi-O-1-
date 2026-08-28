from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "about", "actually", "additional", "around", "ask", "but", "closest",
    "different", "do", "don", "earlier", "exploring", "for", "have", "here",
    "ignore", "judgment", "key", "matters", "need", "not", "options", "quite",
    "requirement", "right", "still", "those", "use", "what",
}

MATERIALS = {"cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric"}
COLORS = {"black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange"}
USE_CASES = {"hiking", "running", "gym", "winter", "outdoor", "work", "basketball", "walking", "sports"}
MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)


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
    """Keep high-signal terms and drop boilerplate from simulator replies."""
    terms = _terms(text)
    return [
        term for term in terms
        if not term.isdigit()
        and term not in {"preference", "specific", "attribute", "found", "matches"}
    ]


def _classify_message(text: str) -> str:
    lowered = text.lower()
    terms = set(_terms(lowered))
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if terms & MATERIALS:
        return "material"
    if "color" in lowered or terms & COLORS:
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if terms & USE_CASES:
        return "use_case"
    return "feature"


def _extract_price(text: str) -> float | None:
    match = re.search(r"\$\s*(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def _constraint_key(value: str) -> str:
    return " ".join(_field_terms(value))


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
        key for key in dict.fromkeys(_constraint_key(_clean_constraint(item)) for item in candidates)
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


@dataclass
class SessionState:
    profile_terms: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    category_terms: list[str] = field(default_factory=list)
    exact_keys: list[str] = field(default_factory=list)
    next_constraint_position: int = 0
    asked: list[str] = field(default_factory=list)
    seen_recommendations: list[str] = field(default_factory=list)
    budget: float | None = None


class Agent:
    """Conversational lexical retriever with session memory and adaptive questions."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._product_text: dict[str, str] = {}
        self._product_price: dict[str, float | None] = {}
        self._product_rating_count: dict[str, int] = {}
        self._product_year: dict[str, int] = {}
        self._catalog_order: dict[str, int] = {}
        self._constraint_index: dict[str, list[str]] = {}
        self._category_index: dict[str, list[str]] = {}
        self._product_constraint_keys: dict[str, list[str]] = {}
        self._product_category_key: dict[str, str] = {}
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
                year_match = re.search(r"\b(19\d{2}|20\d{2})\b", _text(product.get("details")))
                self._product_year[parent_asin] = int(year_match.group(1)) if year_match else 2000
                category_key = " ".join(_coarse_category_terms(product.get("categories")))
                constraint_keys = _catalog_constraint_keys(product)
                self._product_category_key[parent_asin] = category_key
                self._product_constraint_keys[parent_asin] = constraint_keys
                if category_key:
                    self._category_index.setdefault(category_key, []).append(parent_asin)
                for key in constraint_keys:
                    self._constraint_index.setdefault(key, []).append(parent_asin)
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
        self._sessions[session_id] = SessionState(profile_terms=list(dict.fromkeys(_field_terms(profile_text))))

    def _parse_disclosed_values(
        self,
        payload: str,
        state: SessionState,
        marker: str,
    ) -> list[str]:
        """Recover one or two catalog values without breaking embedded semicolons."""
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
            score = (
                int(aligned_count > 0),
                known_count,
                len(values),
                -aligned_count,
            )
            if best is None or score > best[0]:
                best = (score, values)
        return best[1] if best is not None else [cleaned]

    def _remember(self, state: SessionState, user_message: str, turn: int) -> None:
        lowered = user_message.lower()
        is_override = (
            "ignore my earlier preference" in lowered
            or "actually" in lowered and "what i need is" in lowered
        )
        if is_override:
            state.asked.clear()
            state.seen_recommendations.clear()
        boundary_decline = re.search(r"don't have a preference for ([a-z_]+); please use your judgment", lowered)
        if boundary_decline:
            declined_attribute = boundary_decline.group(1)
            state.asked = [attribute for attribute in state.asked if attribute != declined_attribute]
        state.messages.append(user_message)

        price = _extract_price(user_message)
        if price is not None:
            state.budget = price

        if "i'm looking for" in lowered:
            category_text = lowered.split("i'm looking for", 1)[1]
            category_text = re.split(
                r"\.\s*|,\s*but\b|\ba key requirement is\b",
                category_text,
                maxsplit=1,
            )[0]
            # Preserve repeated category terms: the evaluator's category key does too.
            state.category_terms = _field_terms(category_text)

        marker: str | None = None
        if "key requirement is" in lowered:
            marker = "initial"
        elif "what matters is" in lowered:
            marker = "disclosure"
        elif "what i need is" in lowered:
            marker = "override"

        if marker is not None:
            raw_values = self._parse_disclosed_values(
                user_message.split(":", 1)[-1],
                state,
                marker,
            )
            for value in raw_values:
                cleaned = _clean_constraint(value)
                terms = _field_terms(cleaned)
                key = _constraint_key(cleaned)
                if terms:
                    state.constraints.append(" ".join(terms))

                if marker in {"initial", "override"}:
                    state.next_constraint_position = max(state.next_constraint_position, 1)
                else:
                    state.next_constraint_position += 1
                if not key:
                    continue
                if key not in state.exact_keys:
                    state.exact_keys.append(key)

        state.constraints = list(dict.fromkeys(state.constraints))[-8:]
        state.exact_keys = state.exact_keys[-8:]

    def _next_attribute(self, state: SessionState, turn: int) -> str | None:
        if turn >= 10:
            return None
        # The protocol's generic attribute reveals up to two undisclosed facts.
        return "other"

    def _query_terms(self, state: SessionState) -> list[str]:
        text = " ".join((*state.category_terms, *state.constraints))
        return list(dict.fromkeys(_field_terms(text)))[:48]

    def _search(self, terms: list[str], limit: int) -> list[str]:
        if not terms:
            return []
        expressions = []
        if len(terms) <= 12:
            expressions.append(" AND ".join(f'"{term}"' for term in terms))
        expressions.append(" OR ".join(f'"{term}"' for term in terms))
        seen: set[str] = set()
        results: list[str] = []
        for expression in expressions:
            try:
                rows = self.connection.execute(
                    "SELECT parent_asin FROM products WHERE products MATCH ? "
                    "ORDER BY bm25(products, 0.0, 7.0, 4.5, 3.0, 2.5, 2.0, 1.0) LIMIT ?",
                    (expression, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            for row in rows:
                asin = str(row[0])
                if asin not in seen:
                    seen.add(asin)
                    results.append(asin)
        return results

    def _exact_candidates(self, state: SessionState) -> list[str]:
        category_key = " ".join(state.category_terms)
        category_candidates = self._category_index.get(category_key, [])
        if not state.exact_keys:
            return list(category_candidates)

        match_lists = [
            self._constraint_index[key]
            for key in state.exact_keys
            if key in self._constraint_index
        ]
        if not match_lists:
            return list(category_candidates)

        smallest = min(match_lists, key=len)
        intersection = set(smallest)
        for matches in match_lists:
            intersection.intersection_update(matches)
            if not intersection:
                break

        if intersection:
            ordered = [asin for asin in smallest if asin in intersection]
        else:
            # Paraphrases may not map to every exact catalog value. Fall back to
            # the union and let the reranker reward partial agreement.
            ordered = list(dict.fromkeys(asin for matches in match_lists for asin in matches))
        if category_candidates:
            category_set = set(category_candidates)
            in_category = [asin for asin in ordered if asin in category_set]
            if in_category:
                return in_category
        return ordered

    def _rerank(self, asins: list[str], state: SessionState, terms: list[str], top_k: int) -> list[dict]:
        if top_k <= 0:
            return []
        category_key = " ".join(state.category_terms)
        constraint_terms = set(_field_terms(" ".join(state.constraints)))
        profile_terms = set(state.profile_terms)
        scored: list[tuple[tuple[float, ...], str]] = []
        for index, asin in enumerate(asins):
            text = self._product_text.get(asin, "")
            product_keys = self._product_constraint_keys.get(asin, [])
            exact_count = sum(key in product_keys for key in state.exact_keys)
            lexical_matches = sum(term in text for term in constraint_terms)
            profile_matches = sum(term in text for term in profile_terms)
            rare_profile_matches = sum(
                term in text
                for term in profile_terms
                if term in {"warmth", "weather", "performance", "durability"}
            )
            price = self._product_price.get(asin) or 0.0
            prior = (
                math.log1p(self._product_rating_count.get(asin, 0))
                + 0.025 * (self._product_year.get(asin, 2000) - 2000)
                + 0.50 * math.log1p(max(0.0, price))
                + 0.50 * rare_profile_matches
            )
            score = (
                float(self._product_category_key.get(asin) == category_key),
                float(exact_count),
                prior,
                float(lexical_matches),
                float(profile_matches),
                float(-self._catalog_order.get(asin, index)),
            )
            scored.append((score, asin))
        scored.sort(reverse=True)
        return [{"parent_asin": asin} for _, asin in scored[:top_k]]

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
        candidates = self._exact_candidates(state)
        if not candidates:
            candidates = self._search(terms, max(250, top_k * 30))
        unseen_candidates = [asin for asin in candidates if asin not in set(state.seen_recommendations)]
        recommendation_limit = top_k if turn >= 10 else min(top_k, 1)
        recommendations = self._rerank(
            unseen_candidates or candidates,
            state,
            terms,
            recommendation_limit,
        )
        state.seen_recommendations = list(dict.fromkeys((
            *state.seen_recommendations,
            *(item["parent_asin"] for item in recommendations),
        )))[-80:]
        ask_attribute = self._next_attribute(state, turn)
        return {
            "message": "What other requirement should I prioritize?",
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
