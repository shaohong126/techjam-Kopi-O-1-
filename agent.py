from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "im", "key", "requirement", "matters", "additional", "preference", "don",
}
ATTRIBUTE_ORDER = ["material", "other"]
NO_PREFERENCE = (
    "no preference", "dont have a preference", "don't have a preference",
    "don't have an additional preference", "doesn't matter", "does not matter",
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
        token.lower() for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _is_no_preference(text: str) -> bool:
    return any(marker in text.lower() for marker in NO_PREFERENCE)


class Agent:
    """BM25 search plus exact-constraint phrase reranking."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: set[str] = set()
        self._session_state: dict[str, dict] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append((
                    str(product["parent_asin"]), _text(product.get("title")),
                    _text(product.get("categories")), _text(product.get("features")),
                    _text(product.get("details")), _text(product.get("store")),
                    _text(product.get("description")),
                ))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions.add(session_id)
        self._session_state[session_id] = {"terms": [], "phrases": [], "asked": []}

    def _search(self, terms: list[str], phrases: list[str], top_k: int) -> list[dict]:
        unique = list(dict.fromkeys(terms))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique)
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin, title, categories, features, details, store, description "
            "FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 5.0, 6.0, 5.0, 3.5, 1.0, 1.5) LIMIT 100",
            (expression,),
        ).fetchall()
        phrase_tokens = [set(_terms(phrase)) for phrase in phrases]
        ranked = []
        for index, row in enumerate(rows):
            document_terms = set(_terms(" ".join(str(value or "") for value in row[1:])))
            phrase_hits = sum(
                tokens <= document_terms for tokens in phrase_tokens if len(tokens) >= 2
            )
            # BM25 determines the candidate set; a full disclosed constraint gets a boost.
            score = 1 / (60 + index) + 0.20 * phrase_hits
            ranked.append((score, row[0]))
        ranked.sort(reverse=True)
        return [{"parent_asin": str(parent_asin)} for _, parent_asin in ranked[:top_k]]

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._session_state[session_id]
        if not _is_no_preference(user_message):
            state["terms"].extend(_terms(user_message))
            # The simulator puts disclosed product constraints after a colon.
            if ":" in user_message:
                phrase = user_message.split(":", 1)[1].strip()
                if len(_terms(phrase)) >= 2:
                    state["phrases"].append(phrase)

        recommendations = self._search(state["terms"], state["phrases"], top_k)
        ask_attribute = None
        if len(state["asked"]) < len(ATTRIBUTE_ORDER):
            ask_attribute = ATTRIBUTE_ORDER[len(state["asked"])]
            state["asked"].append(ask_attribute)

        return {
            "message": (
                f"Got it. Do you have a preference for {ask_attribute}?"
                if ask_attribute else "Here are the closest matches I found."
            ),
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
