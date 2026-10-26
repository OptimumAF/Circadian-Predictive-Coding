"""Pure protocol composition, fixed diagnostic costs and source detachment."""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from src.adapters import four_phase_probes as probes
from src.adapters import four_phase_replay as replay


CONTROLS = ("recent", "partitioned", "recent48", "partitioned48", "coverage")


def base(control="coverage"):
    # A descriptor-only fixture isolates this composition from the outer runner.
    return replay.describe_protocol(
        {
            "protocol_id": "key_door_four_phase_policy_evaluation_v1",
            "limits": dict(
                body_seconds=90,
                parent_seconds=120,
                process_rss_bytes=768 * 1024**2,
                json_bytes=8 * 1024**2,
                cpu_threads=1,
                gpu=0,
            ),
            "isolation": "original owned stream",
            "limitations": "original negatives retained",
            "final_evaluation": dict(
                checkpoint_updates=[0, 64, 128, 192, 256],
                task_sequence=["A", "B", "C_train", "A"],
                layouts=["A", "B", "C_train", "D"],
                final_data_creation="after original finish",
            ),
            "planned_work": dict(
                native_train_calls=256,
                training_rows=2048,
                score_forwards=142,
                score_rows=1136,
                train_fit_score_forwards=102,
                final_cached_forwards=40,
                maximum_native_forwards=622,
                maximum_recurrent_unroll_steps=3664,
                teacher_episodes=14,
                environment_constructions=54,
                maximum_environment_steps=592,
                live_episodes=40,
                maximum_live_forwards=480,
            ),
        },
        control=control,
    )


@pytest.mark.parametrize("control", CONTROLS)
def test_should_compose_existing_fixed_control_without_changing_training_replay_or_limits(control):
    original = base(control)
    before = deepcopy(original)
    result = probes.describe_protocol(original, control=control)
    assert original == before
    assert result["protocol_id"] == f"key_door_four_phase_{control}_fresh_task_probe_v1"
    assert result["classification"] == "exploratory_single_control_fresh_task_diagnostic"
    assert result["replay"] == original["replay"]
    assert result["limits"] == original["limits"]
    assert result["replay_control"] == control
    assert result["final_evaluation"]["checkpoint_updates"] == [0, 64, 128, 192, 256]
    assert result["final_evaluation"]["task_sequence"] == ["A", "B", "C_train", "A"]
    assert result["planned_work"]["live_episodes"] == 40
    assert result["planned_work"]["maximum_live_forwards"] == 480
    assert result["planned_work"]["train_fit_score_forwards"] == 102
    assert result["planned_work"]["final_cached_forwards"] == 40


def test_should_charge_all_fixed_probe_updates_predictions_unroll_and_teacher_work():
    work = probes.describe_protocol(base(), control="coverage")["planned_work"]
    expected = dict(
        native_train_calls=416,
        training_rows=3328,
        score_forwards=158,
        score_rows=1264,
        maximum_native_forwards=638,
        maximum_recurrent_unroll_steps=5072,
        teacher_episodes=32,
        teacher_steps=256,
        environment_constructions=72,
        maximum_environment_steps=736,
        probes=2,
        probe_train_calls=32,
        probe_training_rows=256,
        probe_score_forwards=16,
        probe_score_rows=128,
    )
    assert {name: work[name] for name in expected} == expected
    assert work["wake_train_calls"] == 256 and work["replay_train_calls"] == 128
    assert work["wake_training_rows"] == 2048 and work["replay_rows"] == 1024


def test_should_fix_probe_cohorts_curves_final_isolation_and_cost_scope():
    result = probes.describe_protocol(base(), control="coverage")
    probe = result["probe"]
    assert probe["layout"] == "existing E(3,3,True,0.75)"
    assert probe["source_wake_checkpoints"] == [0, 256]
    assert probe["source_native_checkpoints"] == [0, 384]
    assert probe["training_episodes"] == 16 and probe["evaluation_episodes"] == 2
    assert probe["checkpoints"] == [0, 4, 8, 16]
    assert probe["cohorts"] == 2 and probe["independent_updates_per_cohort"] == 16
    assert probe["decision_diagnostics"] is True
    assert "CE0-minus-CE16" in probe["metrics"]
    assert "negative" in probe["metrics"]
    assert "both completed probes" in result["final_evaluation"]["final_data_creation"]
    assert probe["budget"]["max_training_updates_per_cohort"] == 16
    assert "no renewed" in probe["budget"]["deadline"]
    assert probe["storage"]["reserved_input_array_bytes"] == 18 * 8 * 9 * 8
    assert "excludes" in probe["storage"]["scope"]
    assert "not generalization" in probe["limitation"]
    assert "gain alone" in result["limitations"]


def test_should_detach_every_nested_output_from_its_base_and_other_calls():
    original = base()
    result = probes.describe_protocol(original, control="coverage")
    result["limits"]["body_seconds"] = 0
    result["final_evaluation"]["layouts"].append("injected")
    result["replay"]["policy"]["retention"]["max_examples"] = 1
    result["probe"]["checkpoints"].append(99)
    assert original["limits"]["body_seconds"] == 90
    assert original["final_evaluation"]["layouts"] == ["A", "B", "C_train", "D"]
    assert original["replay"]["policy"]["retention"]["max_examples"] == 48
    assert probes.describe_protocol(original, control="coverage")["probe"]["checkpoints"] == [
        0,
        4,
        8,
        16,
    ]


class DerivedControl(str):
    pass


@pytest.mark.parametrize(
    "control",
    [
        None,
        False,
        0,
        "",
        "unknown",
        "Coverage",
        " coverage",
        DerivedControl("coverage"),
        np.str_("coverage"),
    ],
)
def test_should_reuse_strict_existing_control_admission_before_base_access(control):
    invalid_base: Any = None
    with pytest.raises(ValueError, match="four-phase replay control"):
        probes.describe_protocol(invalid_base, control=control)


@pytest.mark.parametrize(
    "kind",
    ["base_type", "protocol", "control", "work", "final", "limits", "limitations", "isolation"],
)
def test_should_refuse_unmatched_or_malformed_base_protocol_without_mutation(kind):
    original: Any = base()
    if kind == "base_type":
        original = []
    elif kind == "protocol":
        original["protocol_id"] = "other_protocol"
    elif kind == "control":
        original["replay_control"] = "recent48"
    else:
        key = {
            "work": "planned_work",
            "final": "final_evaluation",
            "limits": "limits",
            "limitations": "limitations",
            "isolation": "isolation",
        }[kind]
        original[key] = None
    before = deepcopy(original)
    with pytest.raises(ValueError, match="matching existing replay protocol"):
        probes.describe_protocol(original, control="coverage")
    assert original == before


@pytest.mark.parametrize(
    "field",
    [
        "native_train_calls",
        "training_rows",
        "score_forwards",
        "score_rows",
        "maximum_native_forwards",
        "maximum_recurrent_unroll_steps",
        "teacher_episodes",
        "environment_constructions",
        "maximum_environment_steps",
    ],
)
@pytest.mark.parametrize("malformation", ["missing", "changed", "float"])
def test_should_refuse_changed_base_cost_assumptions_instead_of_silently_underbudgeting(
    field, malformation
):
    original = base()
    if malformation == "missing":
        original["planned_work"].pop(field)
    elif malformation == "changed":
        original["planned_work"][field] -= 1
    else:
        original["planned_work"][field] = float(original["planned_work"][field])
    before = deepcopy(original)
    with pytest.raises(ValueError, match="differs from the fixed protocol"):
        probes.describe_protocol(original, control="coverage")
    assert original == before
