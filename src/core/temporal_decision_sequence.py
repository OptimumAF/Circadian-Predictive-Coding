"""Fixed temporal tool/rule corpus with observation-bound targets.

Inputs select a declared phase/role/exposure; outputs are typed local requests.
No learner, scheduling, scoring, permission execution, IO or final data here.
The union catalog is known upfront; active tools change without dimension growth.
"""

from dataclasses import dataclass
from typing import Literal

from src.core.typed_decision import DecisionCandidate, DecisionRequest

CATALOG = (
    DecisionCandidate("defer", "defer this ticket to the new workflow queue"),
    DecisionCandidate("inspect", "inspect the ticket for manual review"),
    DecisionCandidate("retire", "use the retired legacy ticket workflow"),
    DecisionCandidate("route", "route this ticket through the original workflow"),
)
VOCABULARY = ("invoice", "legacy", "parcel", "password")
PHASES = ("after_A", "after_B", "after_return")
CHECKPOINTS = ((0, "baseline"), (4, "after_A"), (8, "after_B"), (12, "after_return"))
_OLD = dict(invoice="route", legacy="retire", parcel="route", password="inspect")
_NEW = dict(invoice="inspect", legacy="defer", parcel="defer", password="inspect")


def policy_for_phase(phase: str) -> dict:
    """Own fixed rule metadata; returning to A keeps the current v2 policy."""
    if type(phase) is not str or phase not in ("baseline", *PHASES):
        raise ValueError("unknown temporal decision phase")
    old = phase in ("baseline", "after_A")
    return dict(
        version="v1" if old else "v2",
        allowed_ids=("inspect", "retire", "route") if old else ("defer", "inspect", "route"),
        rules=dict(_OLD if old else _NEW),
    )


@dataclass(frozen=True)
class TemporalDecisionExample:
    phase: str
    exposure: int
    concept: str
    domain: str
    policy_version: str
    role: Literal["train", "outer_selection"]
    source_group: str
    request: DecisionRequest
    target_id: str
    obsolete_target_id: str | None

    def __post_init__(self) -> None:
        policy = policy_for_phase(self.phase)
        if type(self.role) is not str or self.role not in ("train", "outer_selection"):
            raise ValueError("temporal task admits TRAIN/development only")
        if type(self.exposure) is not int or not 0 <= self.exposure <= 3:
            raise ValueError("exposure must be a native integer in0..3")
        if self.role == "train" and self.phase == "baseline":
            raise ValueError("baseline has no TRAIN cases")
        if self.role == "outer_selection" and self.exposure != 0:
            raise ValueError("development uses one declared exposure")
        concepts = (
            ("invoice", "legacy", "parcel")
            if self.role == "train" and self.phase == "after_B"
            else ("invoice", "legacy", "password")
            if self.role == "train"
            else VOCABULARY
        )
        if type(self.concept) is not str or self.concept not in concepts:
            raise ValueError("concept outside declared phase/role")
        allowed = tuple(c for c in CATALOG if c.candidate_id in policy["allowed_ids"])
        old_target = _OLD[self.concept]
        obsolete = old_target if old_target != policy["rules"][self.concept] else None
        identity = f"{self.role}-{self.phase}-{self.exposure}-{self.concept}"
        query = self.concept + (" please" if self.role == "outer_selection" else "")
        if (
            self.domain != ("B" if self.concept == "parcel" else "A")
            or self.policy_version != policy["version"]
            or self.source_group
            != ("train-token" if self.role == "train" else "development-please")
            or type(self.request) is not DecisionRequest
            or self.request != DecisionRequest(identity, query, allowed)
            or self.target_id != policy["rules"][self.concept]
            or self.obsolete_target_id != obsolete
        ):
            raise ValueError("temporal case differs from its observation's declared policy")
        self.request.__post_init__()


def temporal_examples(
    phase: str, role: Literal["train", "outer_selection"], *, exposure: int = 0
) -> tuple[TemporalDecisionExample, ...]:
    """Separate TRAIN/development templates and immutable historical labels."""
    policy = policy_for_phase(phase)
    if type(role) is not str or role not in ("train", "outer_selection"):
        raise ValueError("temporal task admits TRAIN/development only")
    concepts: tuple[str, ...] = (
        ("invoice", "legacy", "parcel") if phase == "after_B" else ("invoice", "legacy", "password")
    )
    if role == "outer_selection":
        concepts = VOCABULARY
    allowed = tuple(c for c in CATALOG if c.candidate_id in policy["allowed_ids"])
    return tuple(
        TemporalDecisionExample(
            phase,
            exposure,
            concept,
            "B" if concept == "parcel" else "A",
            policy["version"],
            role,
            "train-token" if role == "train" else "development-please",
            DecisionRequest(
                f"{role}-{phase}-{exposure}-{concept}",
                concept + (" please" if role == "outer_selection" else ""),
                allowed,
            ),
            policy["rules"][concept],
            _OLD[concept] if _OLD[concept] != policy["rules"][concept] else None,
        )
        for concept in concepts
    )


def validate_temporal_examples(examples: tuple[TemporalDecisionExample, ...]) -> None:
    if type(examples) is not tuple or not 3 <= len(examples) <= 4:
        raise ValueError("expected complete bounded temporal example tuple")
    for example in examples:
        if type(example) is not TemporalDecisionExample:
            raise ValueError("expected exact TemporalDecisionExample")
        example.__post_init__()
    first = examples[0]
    expected = temporal_examples(first.phase, first.role, exposure=first.exposure)
    if examples != expected:
        raise ValueError("temporal examples missing, mixed or out of fixed order")
