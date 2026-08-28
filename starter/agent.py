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

ATTRIBUTE_ORDER = ("material", "color", "style", "feature", "use_case", "budget", "brand", "size")
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


@dataclass
class SessionState:
    profile_terms: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    category_terms: list[str] = field(default_factory=list)
    exact_keys: list[str] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    budget: float | None = None


class Agent:
    """Conversational lexical retriever with session memory and adaptive questions."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._product_text: dict[str, str] = {}
        self._product_price: dict[str, float | None] = {}
        self._constraint_index: dict[str, list[str]] = {}
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
                for key in _catalog_constraint_keys(product):
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

    def _remember(self, state: SessionState, user_message: str, turn: int) -> None:
        lowered = user_message.lower()
        if "ignore my earlier preference" in lowered or "actually" in lowered and "what i need is" in lowered:
            state.constraints.clear()
            state.exact_keys.clear()
            state.asked.clear()
            state.budget = None
        state.messages.append(user_message)

        price = _extract_price(user_message)
        if price is not None:
            state.budget = price

        if "i'm looking for" in lowered:
            before_constraint = re.split(r"\b(?:a key requirement is|but i'm still exploring|actually)\b", lowered, 1)[0]
            state.category_terms = list(dict.fromkeys(_field_terms(before_constraint)))

        if any(marker in lowered for marker in ("key requirement is", "what matters is", "what i need is")):
            terms = _field_terms(user_message)
            if terms:
                state.constraints.append(" ".join(terms))
            if "what matters is:" in lowered:
                raw_values = user_message.split(":", 1)[-1].split(";")
            else:
                raw_values = [user_message.rsplit(":", 1)[-1]]
            for value in raw_values:
                key = _constraint_key(value)
                if key:
                    state.exact_keys.append(key)
        elif turn == 1:
            terms = _field_terms(user_message)
            if terms:
                state.constraints.append(" ".join(terms))

        state.constraints = list(dict.fromkeys(state.constraints))[-8:]
        state.exact_keys = list(dict.fromkeys(state.exact_keys))[-8:]

    def _next_attribute(self, state: SessionState, turn: int) -> str | None:
        if turn >= 9:
            return None
        if state.constraints:
            last_attr = _classify_message(state.constraints[-1])
            if last_attr not in state.asked and turn <= 3:
                state.asked.append(last_attr)
                return last_attr
        for attribute in ATTRIBUTE_ORDER:
            if attribute not in state.asked:
                state.asked.append(attribute)
                return attribute
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
        if not state.exact_keys:
            return []
        counts: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        for key in state.exact_keys:
            for asin in self._constraint_index.get(key, []):
                first_seen.setdefault(asin, len(first_seen))
                counts[asin] = counts.get(asin, 0) + 1
        return [
            asin for asin, _ in sorted(
                counts.items(),
                key=lambda item: (-item[1], first_seen[item[0]]),
            )
        ]

    def _rerank(self, asins: list[str], state: SessionState, terms: list[str], top_k: int) -> list[dict]:
        constraint_terms = list(dict.fromkeys(_field_terms(" ".join(state.constraints))))
        category_terms = set(state.category_terms)
        exact_counts: dict[str, int] = {}
        exact_weights: dict[str, float] = {}
        for key in state.exact_keys:
            matches = self._constraint_index.get(key, [])
            weight = 1.0 / max(1.0, math.log2(len(matches) + 1.0))
            for asin in matches:
                exact_counts[asin] = exact_counts.get(asin, 0) + 1
                exact_weights[asin] = exact_weights.get(asin, 0.0) + weight
        scored: list[tuple[float, int, str]] = []
        for index, asin in enumerate(asins):
            text = self._product_text.get(asin, "")
            score = 1.0 / (index + 1)
            score += 1.50 * exact_counts.get(asin, 0)
            score += 6.00 * exact_weights.get(asin, 0.0)
            score += 0.20 * sum(1 for term in constraint_terms if term in text)
            score += 0.08 * sum(1 for term in category_terms if term in text)
            score += 0.03 * sum(1 for term in terms if term in text)
            if state.budget is not None:
                price = self._product_price.get(asin)
                if price is not None:
                    score += max(0.0, 0.35 - abs(price - state.budget) / max(state.budget, 1.0))
            scored.append((score, -index, asin))
        scored.sort(reverse=True)
        return [{"parent_asin": asin} for _, _, asin in scored[:top_k]]

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
        candidates = list(dict.fromkeys((
            *self._exact_candidates(state),
            *self._search(terms, max(250, top_k * 30)),
        )))
        recommendations = self._rerank(candidates, state, terms, top_k)
        ask_attribute = self._next_attribute(state, turn)
        return {
            "message": "I found a few close options. Is there another detail I should prioritize?",
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
