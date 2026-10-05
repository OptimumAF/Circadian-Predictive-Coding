"""Every genuine development retention checkpoint has a deterministic row."""

from typing import Any

import test_continual_confirmation_retention_costs as cost_tests
from src.app.continual_confirmation_report_costs import canonical_body_identity
from src.app.continual_confirmation_retention_costs import _derive_retention_costs
from src.app.continual_confirmation_retention_rendering import render_retention_costs
from src.infra.continual_confirmation_io import parse_json
import json

development_facts = cost_tests.development_facts


def test_should_render_all_arm_and_context_stages_including_null_states(
    development_facts: tuple[Any, Any],
) -> None:
    rows, families, inventory = cost_tests.development_inputs(development_facts)
    body = _derive_retention_costs(rows, families, inventory)
    rendered = render_retention_costs(body)
    assert len([line for line in rendered.splitlines() if "/retention |" in line]) == 168
    assert "not_applicable" in rendered and "disabled" in rendered
    assert "configured_empty" in rendered and "retained" in rendered
    assert "| after_b | 3840 | 768 | 4608 |" in rendered
    assert "per-arm RSS are unmeasured" in rendered
    decoded = parse_json(json.dumps(body, indent=2, sort_keys=True) + "\n")
    assert canonical_body_identity(decoded) == canonical_body_identity(body)
    assert render_retention_costs(decoded) == rendered
    assert render_retention_costs(body) == rendered
