"""Saved original reports plus explicitly fabricated replay DTOs; no native work."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.adapters import minigrid_probe_rehearsal as option
from src.adapters.minigrid_rehearsal_comparison import compare_reports

STAGE = Path(__file__).resolve().parents[1] / "artifacts/runs/joint-catalogue-study-20261018"


def pair(method="bp"):
    """Fabricate work-only replay branches, preserving historical curves as fakes."""
    control = json.loads((STAGE / (method + ".txt")).read_text(encoding="utf8"))
    replay = deepcopy(control)
    option.describe(replay["protocol"])
    old = deepcopy(replay["training_sources"][:2])
    for episode in old:
        initial = episode["transitions"][0]["observation"]
        for transition in episode["transitions"]:
            transition["observation"] = initial + transition["observation"]
            transition["next_observation"] = initial + transition["next_observation"]
    row = replay["methods"][0]
    for slot in row["probes"]:
        slot["rehearsal_work"] = dict(
            charged_replay_examples=288,
            attempted_updates=32,
            returned_updates=32,
            committed_updates=32,
            returned_training_rows=288,
            bank_encoded_bytes=87280,
            validation_projection_rows=36,
            validation_projected_array_equivalent_bytes=85248,
            budget_binding_preserved=True,
            work_unknown=False,
        )
        slot["report"]["rehearsal"] = dict(
            bound=True,
            stopped=False,
            reason=None,
            total_updates=16,
            input_dim=296,
            wake_updates_observed=16,
            planned_events=16,
            wake_role="probe_train",
            control="fixed_train_episodes_v1",
            fixed_bank=dict(
                episode_ids=["minigrid-source-A-blue", "minigrid-source-A-red"],
                episode_count=2,
                native_transition_count=18,
                encoded_array_bytes=87280,
                construction_updates=0,
            ),
            charged_exposures=288,
            replay_budget_charged_exposures=288,
            attempted_updates=32,
            returned_updates=32,
            committed_updates=32,
            events=[
                dict(
                    updates=index,
                    event_id=f"rehearsal-{index}",
                    status="complete",
                    reason=None,
                    selected_episodes=deepcopy(old),
                    charged_exposures=18,
                    attempted_updates=2,
                    returned_updates=2,
                    committed_updates=2,
                )
                for index in range(1, 17)
            ],
        )
        slot["report"]["probe_stream"] = dict(
            training_role="probe_train", completed_updates=16, evaluation_scope="training_only"
        )
    replay["work"].update(
        native_train_calls=256,
        training_rows=8064,
        replay_rows=576,
        old_a_rehearsal=option.aggregate(replay["methods"]),
    )
    return control, replay


@pytest.mark.parametrize("method", option.METHODS)
def test_should_reduce_all_saved_control_curves_without_native_work(method):
    control, replay = pair(method)
    before = deepcopy((control, replay))
    result = compare_reports(control, replay, method=method)
    assert result["status"] == "complete"
    assert result["scientific_advantage_verified"] is False
    assert result["dose_and_compute_matched"] is False
    assert result["extra_native_calls"] == 64 and result["extra_training_rows"] == 576
    assert result["probability_anchor"]["maximum_difference"] == 0
    assert len(result["control"]["probes"]) == 2
    assert all(
        all(value == 0 for key, value in point.items() if key != "source_updates")
        for point in result["replay_minus_control"]
    )
    assert before == (control, replay)


@pytest.mark.parametrize(
    "path,value",
    [
        (("status",), "failed"),
        (("reason",), "partial"),
        (("scientific_advantage_verified",), True),
        (("protocol", "seed"), 18),
        (("protocol", "inference_steps"), 3),
        (("training_sources", 0, "role"), "probe_train"),
        (("training_sources", 0, "transitions", 0, "action"), 6),
        (("probe_source_reports", 0, "seed"), 99),
        (("probe_source_reports", 0, "closed"), False),
        (("work", "native_train_calls"), 255),
        (("work", "replay_rows"), 575),
        (("work", "failure_work_may_be_unobserved"), True),
        (("methods", 0, "final_evaluation", "status"), "complete"),
        (("methods", 0, "source_unchanged"), False),
        (("methods", 0, "joint_catalogue_acquired"), True),
        (("methods", 0, "probes", 0, "budget_binding_preserved"), False),
        (("methods", 0, "probes", 0, "return_charged_updates"), 15),
        (("methods", 0, "probes", 0, "rehearsal_work", "work_unknown"), True),
        (("methods", 0, "probes", 0, "rehearsal_work", "charged_replay_examples"), True),
        (("methods", 0, "probes", 0, "report", "rehearsal", "events", 0, "charged_exposures"), 17),
        (
            (
                "methods",
                0,
                "probes",
                0,
                "report",
                "rehearsal",
                "events",
                0,
                "selected_episodes",
                0,
                "role",
            ),
            "probe_train",
        ),
        (("methods", 0, "probes", 0, "report", "episode_scope", "training_transitions"), 143),
        (
            (
                "methods",
                0,
                "probes",
                0,
                "report",
                "checkpoints",
                0,
                "decision_diagnostics",
                "F-blue",
                "episode_id",
            ),
            "minigrid-source-A-blue",
        ),
        (("methods", 0, "probes", 0, "outcomes", "ce_gain"), 99),
        (
            (
                "methods",
                0,
                "probes",
                0,
                "report",
                "checkpoints",
                0,
                "decision_diagnostics",
                "F-blue",
                "decisions",
                0,
                "loss",
            ),
            float("nan"),
        ),
        (
            (
                "methods",
                0,
                "probes",
                0,
                "report",
                "checkpoints",
                0,
                "decision_diagnostics",
                "F-blue",
                "decisions",
                0,
                "target_class",
            ),
            6,
        ),
        (("methods", 0, "readouts", 0, "scores", 0, "probabilities", 0, 0), float("inf")),
        (("methods", 0, "readouts", 0, "scores", 0, "predicted_actions", 0), 99),
    ],
)
def test_should_refuse_corrupted_or_unqualified_saved_evidence(path, value):
    control, replay = pair()
    target = replay
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        compare_reports(control, replay, method="bp")


@pytest.mark.parametrize(
    "control,replay,method", [({}, {}, "bp"), (None, {}, "bp"), ({}, {}, "other")]
)
def test_should_refuse_missing_evidence_with_clear_value_error(control, replay, method):
    with pytest.raises(ValueError):
        compare_reports(control, replay, method=method)


def test_should_refuse_coherent_old_a_curve_substituted_for_f():
    control, replay = pair()
    slot = replay["methods"][0]["probes"][0]
    slot["report"]["checkpoints"] = deepcopy(slot["report"]["training_readouts"]["checkpoints"])
    slot["outcomes"] = deepcopy(slot["old_task_readout_outcomes"])
    with pytest.raises(ValueError, match="phase|identity|membership|episode"):
        compare_reports(control, replay, method="bp")


@pytest.mark.parametrize("fault", ["order", "content"])
def test_should_refuse_substituted_replay_episode_records(fault):
    control, replay = pair()
    selected = replay["methods"][0]["probes"][0]["report"]["rehearsal"]["events"][0][
        "selected_episodes"
    ]
    if fault == "order":
        selected.reverse()
    else:
        selected[0]["transitions"][0]["action"] = 6
    with pytest.raises(ValueError, match="membership"):
        compare_reports(control, replay, method="bp")


def test_should_refuse_probability_anchor_shift_even_if_saved_metrics_unchanged():
    control, replay = pair()
    score = replay["methods"][0]["readouts"][0]["scores"][0]
    values = score["probabilities"][0]
    others = [
        i for i in range(7) if i not in (score["target_actions"][0], score["predicted_actions"][0])
    ]
    values[others[0]] += 1e-6
    values[others[1]] -= 1e-6
    with pytest.raises(ValueError, match="probability anchors"):
        compare_reports(control, replay, method="bp")


def test_should_refuse_nested_nonmapping_evidence_as_value_error():
    control, replay = pair()
    replay["methods"][0]["probes"][0] = []
    with pytest.raises(ValueError, match="malformed"):
        compare_reports(control, replay, method="bp")
