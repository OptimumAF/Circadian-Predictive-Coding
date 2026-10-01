"""Project all original cost distinctions; development fixtures are not confirmation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from typing import Any

import pytest

import test_continual_confirmation_work_validation as work_tests
from src.app import continual_confirmation_report_costs as costs
from src.app.continual_confirmation_manifest import fixed_confirmation_manifest


def _seed(name: str) -> tuple[Any, dict[str, Any]]:
    family = next(f for f in fixed_confirmation_manifest().families if f.name == name)
    family = replace(family, seeds=(family.development_seeds[0],))
    simple = name in {"gating", "replay"}
    methods = []
    checkpoints: dict[str, dict[str, Any]] = {
        stage: {} for stage in ("initial", "after_a", "after_b")
    }
    for index, arm in enumerate(family.arms):
        methods.append(
            {
                "method" if simple else "name": arm,
                "wake_updates": 24,
                "replay_updates"
                if name in {"gating", "replay", "sleep"}
                else "applied_replay_updates": index,
                "rejected_executed_replay_updates": 2 if index == 1 else 0,
                "retained_array_bytes": None,
                "sleep": {"accepted": False, "reason": "inactive", "cost": index},
            }
        )
        for stage, values in checkpoints.items():
            values[arm] = {
                "width": 8 + (stage == "after_b"),
                "parameter_count": 33 + 4 * (stage == "after_b"),
                "model_type": "fixture.Model",
                "state_sha256": "a" * 64,
                "parameter_sha256": "b" * 64,
                "state": {"private_large_state": True},
            }
    return family, {
        "family": name,
        "seed": family.seeds[0],
        **checkpoints,
        "legacy_train_facts": {
            "arms" if name == "sleep" else "methods": methods,
            "shared_memory": {"retained_array_bytes": 1234, "access": "shared_fifo"},
            "opportunities": [{"accepted": False, "reason": "guard"}],
        },
        "supplemental_guards": [{"accepted": False, "executed_updates": 2}],
        "roles": {"train": "c" * 64},
    }


@pytest.mark.parametrize("name", ["gating", "replay", "sleep", "schedule", "combined", "parent"])
def test_should_keep_raw_method_context_and_all_checkpoint_costs(name: str) -> None:
    family, row = _seed(name)
    projected = costs._project_seed_costs(row, family)
    assert [cell["arm"] for cell in projected["cells"]] == list(family.arms)
    for index, cell in enumerate(projected["cells"]):
        assert cell["wake_updates"] == 24
        assert cell["applied_replay_updates"] == index
        assert cell["rejected_executed_replay_updates"] == (2 if index == 1 else 0)
        assert cell["executed_optimizer_updates"] == 24 + index + (2 if index == 1 else 0)
        assert [c["stage"] for c in cell["checkpoints"]] == ["initial", "after_a", "after_b"]
        assert cell["checkpoints"][-1]["width"] == 9
        assert cell["checkpoints"][-1]["parameter_count"] == 37
        assert cell["method_facts"]["sleep"]["reason"] == "inactive"
        assert cell["method_facts"]["retained_array_bytes"] is None
        assert "private_large_state" not in str(cell)
    context = projected["seed_context"]
    assert context["legacy_train_context"]["shared_memory"]["retained_array_bytes"] == 1234
    assert context["legacy_train_context"]["opportunities"][0]["reason"] == "guard"
    assert context["supplemental_guards"] == row["supplemental_guards"]
    assert context["training_roles"] == row["roles"]
    assert "methods" not in context["legacy_train_context"]
    assert "arms" not in context["legacy_train_context"]
    assert row == _seed(name)[1]


def test_should_detach_all_mutable_projected_costs_from_the_reader_body() -> None:
    family, row = _seed("combined")
    projected = costs._project_seed_costs(row, family)
    before = deepcopy(projected)
    row["legacy_train_facts"]["methods"][0]["sleep"]["reason"] = "changed"
    row["legacy_train_facts"]["shared_memory"]["retained_array_bytes"] = 0
    row["supplemental_guards"].clear()
    row["roles"].clear()
    row["after_b"][family.arms[0]]["width"] = 10
    assert projected == before


@pytest.mark.parametrize("value", [{}, {"unfinished": True}, {"nonfinite": float("nan")}])
def test_should_reject_unbound_or_partial_public_cost_body_before_projection(
    value: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbid(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unbound body reached projection")

    monkeypatch.setattr(costs, "_project_seed_costs", forbid)
    with pytest.raises(ValueError):
        costs.project_confirmation_costs(value)


@pytest.mark.parametrize("value", [{}, {"z": [1, 2.5, None, True], "a": "\u2192"}])
def test_should_hash_streamed_json_identically_to_original_artifact_encoding(value: Any) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    assert costs.canonical_body_identity(value) == {
        "sha256": sha256(encoded).hexdigest(),
        "byte_count": len(encoded),
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {"bad": object()}])
def test_should_reject_nonfinite_or_unserializable_streamed_bodies(value: Any) -> None:
    with pytest.raises(ValueError, match="malformed|nonfinite"):
        costs.canonical_body_identity(value)


@pytest.mark.parametrize(
    "change", ["duplicate", "missing", "unknown", "bool_work", "negative_work"]
)
def test_should_refuse_ambiguous_cost_scope_or_work_even_in_private_projection(change: str) -> None:
    family, row = _seed("schedule")
    methods = row["legacy_train_facts"]["methods"]
    if change == "duplicate":
        methods[-1] = deepcopy(methods[0])
    elif change == "missing":
        methods.pop()
    elif change == "unknown":
        methods[-1]["name"] = "unknown"
    else:
        methods[-1]["wake_updates"] = True if change == "bool_work" else -1
    with pytest.raises(ValueError):
        costs._project_seed_costs(row, family)


def test_should_refuse_invalid_checkpoint_capacity() -> None:
    family, row = _seed("parent")
    row["after_a"][family.arms[-1]]["parameter_count"] = True
    with pytest.raises(ValueError):
        costs._project_seed_costs(row, family)


development_cost_facts = work_tests.fixtures


@pytest.mark.parametrize("name", work_tests.NAMES)
def test_should_project_every_actual_first_development_seed_after_sealing_model_and_data_calls(
    name: str, development_cost_facts: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    families, facts = development_cost_facts
    family = next(f for f in families if f.name == name)
    work_tests._seal_validation(monkeypatch)
    before = deepcopy(facts[name])
    projected = costs._project_seed_costs(facts[name], family)
    observed = work_tests.work._verify_seed_work(facts[name], family)
    assert (
        sum(c["executed_optimizer_updates"] for c in projected["cells"])
        == observed.executed_optimizer_updates
    )
    assert (
        sum(c["rejected_executed_replay_updates"] for c in projected["cells"])
        == observed.rejected_executed_replay_updates
    )
    collection = "arms" if name == "sleep" else "methods"
    name_field = "method" if name in {"gating", "replay"} else "name"
    indexed = {m[name_field]: m for m in facts[name]["legacy_train_facts"][collection]}
    for cell in projected["cells"]:
        assert cell["method_facts"] == indexed[cell["arm"]]
        for checkpoint in cell["checkpoints"]:
            original = facts[name][checkpoint["stage"]][cell["arm"]]
            assert {k: v for k, v in checkpoint.items() if k != "stage"} == {
                k: original[k] for k in checkpoint if k != "stage"
            }
    assert facts[name] == before
