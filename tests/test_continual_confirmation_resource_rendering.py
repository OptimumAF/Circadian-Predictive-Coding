"""Deterministic inventory rendering, using explicit metadata only."""

from copy import deepcopy

import pytest

from src.app.continual_confirmation_resource_rendering import render_resource_inventory
from src.app.continual_confirmation_resources import RESOURCE_SCHEMA


def test_should_render_all_rows_statuses_scopes_and_raw_precision_deterministically() -> None:
    body = {
        "schema_id": RESOURCE_SCHEMA,
        "coverage": {"fixture": True},
        "contexts": [{"family": "<scope>", "seed": 1}],
        "rows": [
            {
                "family": "family|escaped",
                "seed": 1,
                "arm": "arm",
                "fields": {
                    "wall_time": {
                        "value": None,
                        "unit": "seconds",
                        "scope": "per_arm",
                        "status": "unmeasured",
                        "reason": "not_recorded",
                    },
                    "loop": {
                        "value": 0.12345678901234566,
                        "unit": "count",
                        "scope": "per_arm",
                        "status": "derived",
                        "reason": "original_formula",
                    },
                    "parameter_history": {
                        "value": [{"parameters": 33}],
                        "unit": "parameters",
                        "scope": "named_points",
                        "status": "derived",
                        "reason": "not_continuous",
                    },
                },
            }
        ],
    }
    original = deepcopy(body)
    result = render_resource_inventory(body)
    assert "0.12345678901234566" in result
    assert "unmeasured" in result and "not_continuous" in result
    assert "family\\|escaped" in result and "&lt;scope&gt;" in result
    assert "1 recorded points; see complete JSON" in result
    assert result == render_resource_inventory(body)
    assert body == original


@pytest.mark.parametrize("body", [{}, {"schema_id": "unknown"}])
def test_should_refuse_an_unknown_inventory_schema(body: dict) -> None:
    with pytest.raises(ValueError):
        render_resource_inventory(body)
