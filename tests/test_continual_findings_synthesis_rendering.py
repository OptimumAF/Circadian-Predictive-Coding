"""Exact display and full late-evidence/interpretation rejection behavior."""

from copy import deepcopy
from typing import Any

import pytest

from src.app import continual_findings_synthesis_rendering as module


def test_should_escape_table_content_and_preserve_exact_negative_and_null_numbers():
    table = "\n".join(
        module._table(
            ["Value", "Reason"], [[-0.12345678901234568, "a|b\nkeep"], [None, "unmeasured"]]
        )
    )
    assert "-0.12345678901234568" in table
    assert "a\\|b\\nkeep" in table
    assert "| None | unmeasured |" in table


@pytest.mark.parametrize(
    "field,value", [("status", "supported"), ("counter", 0), ("reason", "success")]
)
def test_should_rebuild_and_reject_late_interpretation_or_unobserved_counter_change(
    monkeypatch, field, value
):
    original: dict[str, Any] = {
        "original_inputs": {"pinned": True},
        "finding": {"status": "unresolved", "counter": None, "reason": "timeout"},
    }
    changed = deepcopy(original)
    changed["finding"][field] = value
    calls = []

    def rebuild(inputs):
        calls.append(inputs)
        return deepcopy(original)

    monkeypatch.setattr(module, "build_complete_findings", rebuild)
    monkeypatch.setattr(
        module,
        "_render_validated_findings",
        lambda _: pytest.fail("must not render corrupted evidence"),
    )
    with pytest.raises(ValueError, match="whole complete findings"):
        module.render_complete_findings(changed)
    assert calls == [original["original_inputs"]]


def test_should_reject_presentation_without_full_original_input_body():
    with pytest.raises(ValueError, match="presentation input"):
        module.render_complete_findings({"hypotheses": "supported"})
