from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class IntentMode(str, Enum):
    BUYING = "buying"
    BROWSING = "browsing"


class BudgetOperator(str, Enum):
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    RANGE = "range"
    APPROXIMATE = "approximate"


@dataclass(frozen=True)
class BudgetConstraint:
    operator: BudgetOperator
    lower: float | None = None
    upper: float | None = None

    def matches(self, price: float | None) -> bool:
        if price is None:
            return False
        if self.operator is BudgetOperator.MAXIMUM:
            return self.upper is not None and price <= self.upper
        if self.operator is BudgetOperator.MINIMUM:
            return self.lower is not None and price >= self.lower
        if self.operator is BudgetOperator.RANGE:
            return (
                self.lower is not None
                and self.upper is not None
                and self.lower <= price <= self.upper
            )
        center = self.lower if self.lower is not None else self.upper
        if center is None:
            return False
        tolerance = max(5.0, center * 0.25)
        return abs(price - center) <= tolerance

    def proximity(self, price: float | None) -> float:
        if price is None:
            return 0.0
        if self.matches(price):
            if self.operator is BudgetOperator.APPROXIMATE:
                center = self.lower if self.lower is not None else self.upper
                if center is None:
                    return 1.0
                return 1.0 / (1.0 + abs(price - center) / max(1.0, center))
            return 1.0
        boundary = self.upper if self.operator is BudgetOperator.MAXIMUM else self.lower
        if self.operator is BudgetOperator.RANGE:
            boundary = self.lower if price < (self.lower or 0.0) else self.upper
        if boundary is None:
            return 0.0
        return math.exp(-abs(price - boundary) / max(10.0, boundary * 0.25))


@dataclass
class SlotValue:
    attribute: str
    text: str
    key: str
    terms: tuple[str, ...]
    turn: int
    kind: str
    active: bool = True


@dataclass
class SessionState:
    profile_terms: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    slots: list[SlotValue] = field(default_factory=list)
    category_terms: list[str] = field(default_factory=list)
    next_constraint_position: int = 0
    asked: list[str] = field(default_factory=list)
    declined: set[str] = field(default_factory=set)
    boundary_declined: bool = False
    seen_recommendations: list[str] = field(default_factory=list)
    budget: BudgetConstraint | None = None
    intent: IntentMode = IntentMode.BROWSING
    context_version: int = 0
    last_candidate_count: int = 0
    strategy: str = "category_browse"

    def active_slots(self) -> list[SlotValue]:
        return [slot for slot in self.slots if slot.active]

    def active_keys(self) -> list[str]:
        return list(dict.fromkeys(slot.key for slot in self.active_slots() if slot.key))[-8:]

    def active_terms(self) -> list[str]:
        return list(dict.fromkeys(term for slot in self.active_slots() for term in slot.terms))[-48:]

    def add_slot(self, slot: SlotValue) -> None:
        for existing in self.slots:
            if (
                existing.active
                and existing.attribute == slot.attribute
                and existing.key == slot.key
                and existing.kind == slot.kind
            ):
                return
        # Keep separate provenance when a tentative preference is later confirmed.
        # An override can then revoke the original statement without erasing the
        # independently disclosed constraint.
        self.slots.append(slot)
        self.context_version += 1

    def revoke(self, predicate: Callable[[SlotValue], bool]) -> None:
        changed = False
        for slot in self.slots:
            if slot.active and predicate(slot):
                slot.active = False
                changed = True
        if changed:
            self.context_version += 1


@dataclass
class RetrievalResult:
    asins: list[str]
    lexical_scores: dict[str, float] = field(default_factory=dict)
    semantic_scores: dict[str, float] = field(default_factory=dict)
    exact_candidate_count: int = 0
    strategy: str = "category_browse"
