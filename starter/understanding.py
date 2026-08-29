from __future__ import annotations

import math
import re

from starter.models import (
    BudgetConstraint,
    BudgetOperator,
    IntentMode,
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

# These patterns intentionally match the organizer's intent-card vocabulary.
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


def text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def field_terms(text: str) -> list[str]:
    return [
        term for term in terms(text)
        if not term.isdigit() and term not in BOILERPLATE_TERMS
    ]


def constraint_key(value: str) -> str:
    return " ".join(term for term in terms(value) if term not in BOILERPLATE_TERMS)


def classify_message(text: str) -> str:
    lowered = text.lower()
    message_terms = set(terms(lowered))
    if "budget" in lowered or re.search(
        r"(?:\$|<=|>=|under|below|over|above|between|at most|at least)\s*\$?\s*\d",
        lowered,
    ):
        return "budget"
    if message_terms & MATERIALS:
        return "material"
    if "color" in lowered or message_terms & COLORS:
        return "color"
    if message_terms & SIZE_TERMS:
        return "size"
    if "brand" in lowered or "maker" in lowered or "store" in lowered:
        return "brand"
    if message_terms & STYLE_TERMS:
        return "style"
    if message_terms & USE_CASES:
        return "use_case"
    return "feature"


def extract_budget(text: str) -> BudgetConstraint | None:
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


def flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def clean_constraint(value: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def semantic_terms(text: str) -> set[str]:
    raw = field_terms(text)
    canonical = [SEMANTIC_CANONICAL.get(term, term) for term in raw]
    result = set(raw) | set(canonical)
    result.update(
        f"{left}_{right}"
        for left, right in zip(canonical, canonical[1:])
        if left != right
    )
    return result


def semantic_similarity(query_terms: set[str], product_terms: set[str]) -> float:
    if not query_terms or not product_terms:
        return 0.0
    return len(query_terms & product_terms) / math.sqrt(len(query_terms) * len(product_terms))


def expanded_terms(query_terms: list[str], limit: int = 36) -> list[str]:
    expanded: list[str] = []
    for term in query_terms:
        canonical = SEMANTIC_CANONICAL.get(term, term)
        expanded.extend((term, canonical))
        expanded.extend(sorted(SEMANTIC_GROUPS.get(canonical, ())))
    return list(dict.fromkeys(expanded))[:limit]


def catalog_constraint_keys(product: dict) -> list[str]:
    candidates = [
        *flatten_values(product.get("features")),
        *flatten_values(product.get("details")),
    ]
    corpus = " ".join(
        (
            text_value(product.get("title")),
            text_value(product.get("features")),
            text_value(product.get("details")),
            text_value(product.get("description")),
            text_value(product.get("categories")),
            text_value(product.get("store")),
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
        key for key in dict.fromkeys(constraint_key(str(item)) for item in candidates)
        if key
    ]


def coarse_category_terms(values: object) -> list[str]:
    excluded = {"clothing", "clothing shoes jewelry", "clothing shoes & jewelry"}
    cleaned: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            key = constraint_key(part)
            if key and key not in excluded:
                cleaned.append(part)
    return field_terms(" ".join(cleaned[-2:])) if cleaned else []


class ConversationStateTracker:
    def __init__(
        self,
        constraint_index: dict[str, list[str]],
        category_index: dict[str, list[str]],
        product_constraint_keys: dict[str, list[str]],
    ) -> None:
        self.constraint_index = constraint_index
        self.category_index = category_index
        self.product_constraint_keys = product_constraint_keys

    @staticmethod
    def profile_terms(user_profile: dict) -> list[str]:
        profile_text = " ".join(
            (
                text_value(user_profile.get("preference_tags")),
                text_value(user_profile.get("summary")),
                text_value(user_profile.get("rating_style")),
            )
        )
        return list(dict.fromkeys(field_terms(profile_text)))

    @staticmethod
    def _category_from_message(text: str) -> list[str]:
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
            category_terms = field_terms(category)
            if category_terms:
                return category_terms
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

    def _payload(
        self,
        user_message: str,
        turn: int,
        has_category: bool,
    ) -> tuple[str | None, str]:
        lowered = user_message.lower()
        patterns = (
            ("override", r"what i need is\s*:\s*(.+)"),
            ("initial", r"key requirement is\s*:\s*(.+)"),
            ("disclosure", r"what matters is\s*:\s*(.+)"),
            ("initial", r"must satisfy(?: this requirement)?\s*:\s*(.+)"),
            ("initial", r"non-negotiable(?: requirement)?(?: is)?\s*:\s*(.+)"),
            ("disclosure", r"(?:my )?(?:priority|priorities|preference)\s+(?:is|are)\s*:\s*(.+)"),
            (
                "override" if self._is_override(user_message) else "disclosure",
                r"prioritize\s*:\s*(.+)",
            ),
            ("override", r"replace\s+.+?\s+with\s*:?\s*(.+)"),
        )
        for marker, pattern in patterns:
            match = re.search(pattern, lowered, re.I)
            if match:
                return marker, user_message[match.start(1):]
        if turn == 1 and has_category and "." in user_message:
            tail = user_message.split(".", 1)[1].strip()
            if tail and not any(
                word in tail.lower() for word in ("still exploring", "just browsing")
            ):
                return "initial_soft", tail
        return None, ""

    def _parse_disclosed_values(
        self,
        payload: str,
        state: SessionState,
        marker: str,
    ) -> list[str]:
        cleaned = clean_constraint(payload)
        if not cleaned:
            return []
        if marker != "disclosure" or "; " not in cleaned:
            return [cleaned]

        chunks = cleaned.split("; ")
        segmentations = [[cleaned]]
        for split_at in range(1, len(chunks)):
            segmentations.append(["; ".join(chunks[:split_at]), "; ".join(chunks[split_at:])])

        category_key = " ".join(state.category_terms)
        category_candidates = self.category_index.get(category_key, [])
        start_position = state.next_constraint_position
        best: tuple[tuple[int, int, int, int], list[str]] | None = None
        for values in segmentations:
            keys = [constraint_key(value) for value in values]
            known_count = sum(key in self.constraint_index for key in keys if key)
            aligned_count = 0
            for asin in category_candidates:
                product_keys = self.product_constraint_keys.get(asin, [])
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

    @staticmethod
    def _apply_override(
        state: SessionState,
        new_slots: list[SlotValue],
        text: str,
    ) -> None:
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
            old_key = constraint_key(explicit.group(1))
            state.revoke(lambda slot: slot.key == old_key)
        if any(slot.attribute == "budget" for slot in new_slots):
            state.budget = None
        state.asked.clear()
        state.seen_recommendations.clear()

    def remember(self, state: SessionState, user_message: str, turn: int) -> None:
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
            cleaned = clean_constraint(value)
            slot_terms = tuple(dict.fromkeys(field_terms(cleaned)))
            key = constraint_key(cleaned)
            if not slot_terms and not key:
                continue
            new_slots.append(
                SlotValue(
                    attribute=classify_message(cleaned),
                    text=cleaned,
                    key=key,
                    terms=slot_terms,
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
                state.budget = extract_budget(slot.text)
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

    @staticmethod
    def query_terms(state: SessionState) -> list[str]:
        query = [*state.category_terms, *state.active_terms()]
        if state.intent is IntentMode.BROWSING:
            query.extend(state.profile_terms)
        return list(dict.fromkeys(field_terms(" ".join(query))))[:48]
