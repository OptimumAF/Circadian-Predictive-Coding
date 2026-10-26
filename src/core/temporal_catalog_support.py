"""Explicit TRAIN-only catalog-support recipes over the fixed temporal corpus.

The auxiliary route rule is new hand-authored supervision, not a reinterpretation
of a v2 invoice label. This module does not score, schedule, encode or train.
"""

from dataclasses import dataclass, fields
from typing import Literal

from src.core.temporal_decision_sequence import (
    CATALOG,
    PHASES,
    TemporalDecisionExample,
    policy_for_phase,
    temporal_examples,
)
from src.core.typed_decision import DecisionCandidate, DecisionRequest

SUPPORT_QUERY = "original workflow"


@dataclass(frozen=True)
class CatalogSupportExample:
    phase: str
    exposure: int
    concept: str
    domain: str
    policy_version: str
    role: Literal["train"]
    source_group: str
    request: DecisionRequest
    target_id: str
    obsolete_target_id: str | None

    def __post_init__(self):
        if type(self) is not CatalogSupportExample or set(vars(self)) != {
            field.name for field in fields(CatalogSupportExample)
        }:
            raise ValueError("expected complete catalog support record")
        if type(self.phase) is not str or self.phase not in PHASES[1:]:
            raise ValueError("catalog route support is a v2 B/return TRAIN rule")
        if type(self.exposure) is not int or not 0 <= self.exposure <= 3:
            raise ValueError("support exposure must be a native integer in0..3")
        expected = dict(
            concept="route_support",
            domain="auxiliary",
            policy_version="v2",
            role="train",
            source_group="train-catalog-route",
            target_id="route",
        )
        if (
            any(
                type(getattr(self, name)) is not str or getattr(self, name) != value
                for name, value in expected.items()
            )
            or self.obsolete_target_id is not None
        ):
            raise ValueError("catalog support fields differ from the explicit auxiliary route rule")
        if type(self.request) is not DecisionRequest or set(vars(self.request)) != {
            field.name for field in fields(DecisionRequest)
        }:
            raise ValueError("support requires an exact complete DecisionRequest")
        self.request.__post_init__()
        for candidate in self.request.allowed_candidates:
            if type(candidate) is not DecisionCandidate or set(vars(candidate)) != {
                field.name for field in fields(DecisionCandidate)
            }:
                raise ValueError("support requires exact complete allowed candidates")
        policy = policy_for_phase(self.phase)
        allowed = tuple(
            candidate for candidate in CATALOG if candidate.candidate_id in policy["allowed_ids"]
        )
        request = DecisionRequest(
            f"train-{self.phase}-{self.exposure}-route-support", SUPPORT_QUERY, allowed
        )
        if self.request != request:
            raise ValueError("support request differs from its fixed v2 TRAIN declaration")


def catalog_training_examples(
    phase: str, catalog_support: str = "none", *, exposure: int = 0
) -> tuple[TemporalDecisionExample | CatalogSupportExample, ...]:
    """Keep A unchanged; explicitly duplicate invoice or add route support in v2."""
    if type(catalog_support) is not str or catalog_support not in ("none", "repeat", "route"):
        raise ValueError("catalog_support must be exactly none, repeat or route")
    original = temporal_examples(phase, "train", exposure=exposure)
    if phase == "after_A" or catalog_support == "none":
        return original
    if catalog_support == "repeat":
        return (*original, original[0])
    first = original[0]
    return (
        *original,
        CatalogSupportExample(
            phase,
            exposure,
            "route_support",
            "auxiliary",
            "v2",
            "train",
            "train-catalog-route",
            DecisionRequest(
                f"train-{phase}-{exposure}-route-support",
                SUPPORT_QUERY,
                first.request.allowed_candidates,
            ),
            "route",
            None,
        ),
    )
