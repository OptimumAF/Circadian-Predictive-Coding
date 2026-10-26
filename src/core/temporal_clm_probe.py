"""Fixed reserved CLM probe declarations, separate from benchmark TRAIN.

Defines two withheld markers and a suffix-test template group before observation.
No encoding, scheduling, model, scoring, release or independence claim is made.
"""

from dataclasses import dataclass, fields
from typing import Literal, cast

from src.core.temporal_decision_sequence import CATALOG, CHECKPOINTS, policy_for_phase
from src.core.typed_decision import DecisionCandidate, DecisionRequest

PROBE_UPDATES = 6
PROBE_QUERIES = ("tavrix", "vemlot", "tavrix please", "vemlot please")
PROBE_GROUP_CODES = (1, 3)


@dataclass(frozen=True)
class ReservedCLMProbeExample:
    phase: str
    concept: str
    role: Literal["probe_train", "probe_test"]
    source_group: str
    request: DecisionRequest
    target_id: str

    def __post_init__(self):
        if type(self) is not ReservedCLMProbeExample or set(vars(self)) != {
            field.name for field in fields(ReservedCLMProbeExample)
        }:
            raise ValueError("expected exact complete ReservedCLMProbeExample")
        phases = tuple(phase for _, phase in CHECKPOINTS)
        if type(self.phase) is not str or self.phase not in phases:
            raise ValueError("reserved probe phase must be a fixed temporal checkpoint")
        if type(self.role) is not str or self.role not in ("probe_train", "probe_test"):
            raise ValueError("reserved examples admit probe_train or probe_test only")
        if type(self.concept) is not str or self.concept not in PROBE_QUERIES[:2]:
            raise ValueError("reserved probe concept differs from the fixed marker pair")
        train = self.role == "probe_train"
        group = "clm-probe-train-token" if train else "clm-probe-test-please"
        target = "inspect" if self.concept == "tavrix" else "route"
        if type(self.source_group) is not str or self.source_group != group:
            raise ValueError("reserved probe source group differs from its role")
        if type(self.target_id) is not str or self.target_id != target:
            raise ValueError("reserved probe target differs from its fixed declaration")
        if type(self.request) is not DecisionRequest or set(vars(self.request)) != {
            field.name for field in fields(DecisionRequest)
        }:
            raise ValueError("reserved probe requires an exact complete DecisionRequest")
        self.request.__post_init__()
        for candidate in self.request.allowed_candidates:
            if type(candidate) is not DecisionCandidate or set(vars(candidate)) != {
                field.name for field in fields(DecisionCandidate)
            }:
                raise ValueError("reserved probe requires exact complete candidates")
        policy = policy_for_phase(self.phase)
        allowed = tuple(
            candidate for candidate in CATALOG if candidate.candidate_id in policy["allowed_ids"]
        )
        query = self.concept + ("" if train else " please")
        expected = DecisionRequest(f"{self.role}-{self.phase}-{self.concept}", query, allowed)
        if self.request != expected:
            raise ValueError("reserved probe request differs from its original role/phase template")


def reserved_probe_examples(phase, role) -> tuple[ReservedCLMProbeExample, ...]:
    if type(phase) is not str or phase not in tuple(name for _, name in CHECKPOINTS):
        raise ValueError("unknown reserved probe checkpoint")
    if type(role) is not str or role not in ("probe_train", "probe_test"):
        raise ValueError("reserved examples admit probe_train or probe_test only")
    policy = policy_for_phase(phase)
    allowed = tuple(
        candidate for candidate in CATALOG if candidate.candidate_id in policy["allowed_ids"]
    )
    return tuple(
        ReservedCLMProbeExample(
            phase,
            concept,
            cast(Literal["probe_train", "probe_test"], role),
            "clm-probe-train-token" if role == "probe_train" else "clm-probe-test-please",
            DecisionRequest(
                f"{role}-{phase}-{concept}",
                concept + ("" if role == "probe_train" else " please"),
                allowed,
            ),
            "inspect" if concept == "tavrix" else "route",
        )
        for concept in PROBE_QUERIES[:2]
    )
