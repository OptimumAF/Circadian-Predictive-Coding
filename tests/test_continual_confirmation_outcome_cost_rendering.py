"""All-cell compute/memory presentation uses declared metadata fixtures only."""

from copy import deepcopy
import json
from typing import Any

import pytest

import test_continual_confirmation_outcome_costs as input_tests
from src.app.continual_confirmation_outcome_costs import _derive_outcome_costs
from src.app.continual_confirmation_outcome_cost_rendering import render_outcome_cost_presentation

fabricated_scored_json = input_tests.fabricated_scored_json
original_cost_metadata = input_tests.original_cost_metadata
fabricated_report = input_tests.fabricated_report
development_facts = input_tests.development_facts
presentation_inputs = input_tests.presentation_inputs
seal_validation = input_tests.seal_validation


def test_should_render_both_compute_and_memory_views_for_every_cell_without_changing_json_order(
    presentation_inputs: tuple[Any, Any, Any],
) -> None:
    body = _derive_outcome_costs(*presentation_inputs)
    text = render_outcome_cost_presentation(body)
    assert text == render_outcome_cost_presentation(json.loads(json.dumps(body, sort_keys=True)))
    assert text.count("every original outcome against compute") == 6
    assert text.count("every original outcome against memory and capacity") == 6
    for row in body["rows"]:
        assert text.count(f"| {row['seed']} | {row['arm']} |") == 2
    assert text.count("### ") == 60
    for phrase in (
        "null (zero_a_after_a)",
        "not_applicable",
        "disabled",
        "configured_empty",
        "retained",
        "Rejected executed replay",
        "Per-arm wall time/RSS/guard duration remain unmeasured",
        "no allocation across arms",
    ):
        assert phrase in text
    assert "-0.025" in text and "2.0" in text


def test_should_escape_failure_text_once_and_retain_every_failed_seed(
    presentation_inputs: tuple[Any, Any, Any],
) -> None:
    body = _derive_outcome_costs(*presentation_inputs)
    body["rows"][0]["outcome"]["failure"] = "<bad>|&\nline"
    body["rows"][0]["metrics"]["final_mean_task_accuracy"] = {"value": None, "reason": "<bad>|&"}
    text = render_outcome_cost_presentation(body)
    assert "&lt;bad&gt;\\|&amp;<br>line" in text
    assert "null (&lt;bad&gt;\\|&amp;)" in text
    assert "&amp;lt;bad" not in text


@pytest.mark.parametrize("change", ["schema", "rows", "contexts"])
def test_should_refuse_incomplete_render_scope(
    presentation_inputs: tuple[Any, Any, Any], change: str
) -> None:
    body = deepcopy(_derive_outcome_costs(*presentation_inputs))
    if change == "schema":
        body["schema_id"] = "unknown"
    else:
        body[change].pop()
    with pytest.raises(ValueError):
        render_outcome_cost_presentation(body)
