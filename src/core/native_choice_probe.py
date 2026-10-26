"""Fixed reserved native text task and existing retention readout declarations.

Outputs existing immutable Choice example records; this module selects no dose,
model, metric, clock or outcome. Probe labels must stay in their declared roles.
Shared marker grammar is synthetic, not empirical group independence.
"""

from src.core.native_choice_sequence import CATALOG, ChoiceSequenceExample, sequence_examples
from src.core.typed_decision import DecisionRequest

PROBE_UPDATES = 6


def _declared_examples():
    result = []
    for role, queries in (
        ("probe_train", ("Case marker: tavrix", "Intake code tavrix. Route this request.")),
        (
            "probe_test",
            (
                "Case marker: tavrix; please assist.",
                "A support ticket marked tavrix needs routing.",
            ),
        ),
    ):
        for index, query in enumerate(queries):
            identity = f"{role}-C-{index}"
            result.append(
                ChoiceSequenceExample(
                    identity,
                    "C",
                    f"{role}_marker_templates",
                    role,
                    DecisionRequest(identity, query, CATALOG),
                    "billing",
                )
            )
    result.extend(
        e
        for e in sequence_examples()
        if e.role == "outer_selection" and e.context in ("A", "anchor")
    )
    return tuple(result)


def validate_probe_examples(examples) -> None:
    """No relabeling or outcome-selected requests in this fixed probe."""
    if type(examples) is not tuple or len(examples) != 8:
        raise ValueError("reserved native probe requires exactly eight declared examples")
    for example in examples:
        if (
            type(example) is not ChoiceSequenceExample
            or type(example.request) is not DecisionRequest
        ):
            raise ValueError("reserved native probe requires exact immutable Choice records")
        example.request.__post_init__()
    if examples != _declared_examples():
        raise ValueError("reserved native probe differs from frozen requests/roles/targets")


def probe_examples() -> tuple[ChoiceSequenceExample, ...]:
    examples = _declared_examples()
    validate_probe_examples(examples)
    return examples
