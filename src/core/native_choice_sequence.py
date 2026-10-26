"""Fixed exploratory native routing corpus, roles and phase order.

Outputs typed text requests with separate TRAIN/development template groups.
No model, clock, optimizer, filesystem, label release or outcome-based selection.
Groups describe synthetic templates; they do not attest empirical independence.
"""

from dataclasses import dataclass

from src.core.experience import require_identifier
from src.core.typed_decision import DecisionCandidate, DecisionRequest

INSTRUCTIONS = "Choose the customer's support intent."
CATALOG = (
    DecisionCandidate("billing", "invoice billing payment refund"),
    DecisionCandidate("password", "password reset login account"),
    DecisionCandidate("shipping", "shipping parcel delivery tracking"),
)
PHASES = (("after_A", "A"), ("after_B", "B"), ("after_return", "A"))
PASSES = 3


@dataclass(frozen=True)
class ChoiceSequenceExample:
    example_id: str
    context: str
    source_group: str
    role: str
    request: DecisionRequest
    target_id: str


def sequence_examples() -> tuple[ChoiceSequenceExample, ...]:
    result = []
    for context, code, target in (("A", "zorbex", "shipping"), ("B", "neldax", "password")):
        for role, group, queries in (
            (
                "train",
                "train_marker_templates",
                (
                    f"Case marker: {code}",
                    f"Intake code {code}. Route this request.",
                ),
            ),
            (
                "outer_selection",
                "development_marker_templates",
                (
                    f"Case marker: {code}; please assist.",
                    f"A support ticket marked {code} needs routing.",
                ),
            ),
        ):
            for index, query in enumerate(queries):
                identity = f"{role}-{context}-{index}"
                result.append(
                    ChoiceSequenceExample(
                        identity,
                        context,
                        group,
                        role,
                        DecisionRequest(identity, query, CATALOG),
                        target,
                    )
                )
    for index, (query, target) in enumerate(
        (
            ("invoice refund", "billing"),
            ("reset my password", "password"),
        )
    ):
        identity = f"anchor-{index}"
        result.append(
            ChoiceSequenceExample(
                identity,
                "anchor",
                "development_anchor_templates",
                "outer_selection",
                DecisionRequest(identity, query, CATALOG),
                target,
            )
        )
    examples = tuple(result)
    validate_sequence_examples(examples)
    return examples


def _validate_example(example: ChoiceSequenceExample) -> None:
    if type(example) is not ChoiceSequenceExample:
        raise ValueError("expected exact ChoiceSequenceExample")
    for name in ("example_id", "context", "source_group", "target_id"):
        require_identifier(getattr(example, name), name)
    if type(example.role) is not str or example.role not in ("train", "outer_selection"):
        raise ValueError("sequence admits TRAIN and outer_selection only; no final/probe roles")
    if example.context not in ("A", "B", "anchor") or (
        example.context == "anchor" and example.role != "outer_selection"
    ):
        raise ValueError("invalid sequence context/role")
    if type(example.request) is not DecisionRequest:
        raise ValueError("sequence requires exact DecisionRequest")
    example.request.__post_init__()
    if (
        example.request.request_id != example.example_id
        or example.request.allowed_candidates != CATALOG
    ):
        raise ValueError("sequence request identity/catalog changed")
    if example.target_id not in {c.candidate_id for c in CATALOG}:
        raise ValueError("sequence target outside allowed candidates")
    if (
        example.context in ("A", "B")
        and example.target_id != {"A": "shipping", "B": "password"}[example.context]
    ):
        raise ValueError("sequence context target changed")


def validate_sequence_examples(examples: tuple[ChoiceSequenceExample, ...]) -> None:
    """Refuse role/group/query overlap before any optional model work."""
    if type(examples) is not tuple or len(examples) != 10:
        raise ValueError("sequence requires exactly ten declared examples")
    for example in examples:
        _validate_example(example)
    if len({e.example_id for e in examples}) != len(examples):
        raise ValueError("sequence example IDs must be unique")
    counts = {
        (role, context): sum(e.role == role and e.context == context for e in examples)
        for role in ("train", "outer_selection")
        for context in ("A", "B", "anchor")
    }
    if counts != {
        ("train", "A"): 2,
        ("train", "B"): 2,
        ("train", "anchor"): 0,
        ("outer_selection", "A"): 2,
        ("outer_selection", "B"): 2,
        ("outer_selection", "anchor"): 2,
    }:
        raise ValueError("sequence requires two examples per declared context/role")
    train = tuple(e for e in examples if e.role == "train")
    readout = tuple(e for e in examples if e.role == "outer_selection")
    if {e.source_group for e in train} & {e.source_group for e in readout}:
        raise ValueError("TRAIN/development source groups overlap")
    if {e.request.query.casefold().strip() for e in train} & {
        e.request.query.casefold().strip() for e in readout
    }:
        raise ValueError("TRAIN/development queries overlap")
