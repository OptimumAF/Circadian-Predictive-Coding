"""Stage partitions through actual tiny streams and existing fake robot IO.

Four families plus two paid-failure BP runs; no extra forwards for grouping.
Existing canonical fixtures supply complete raw42 records, not native science.
"""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from scripts import run_continuous_robot_learning as entry
from src.adapters import continuous_learnability_demo as demo
from src.core.readiness_stage_diagnostics import ReadinessStageBinding, readiness_stage_metadata
from src.core.pick_cube_state_quaternions import (
    state_quaternion_projection,
    state_quaternion_training_episode,
)
from src.infra import maniskill_continuous as native
from test_continuous_robot_learning import assert_state
from test_state_quaternion_robot_learning import QuaternionEnvironment, episodes


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")
LABELS = ("lift", "align")  # Explicit fixture commands, not an elapsed phase template.


def binding_pair(values):
    return tuple(ReadinessStageBinding(value, LABELS) for value in values)


def projected():
    return tuple(
        state_quaternion_training_episode(v, state_quaternion_projection()) for v in episodes()
    )


def forbidden(*args, **kwargs):
    pytest.fail("refused metadata admitted fit/model/native work")


@pytest.fixture(scope="module", params=FAMILIES)
def completed(request):
    values = projected()
    observed = []
    original = demo.diagnose_residual

    def diagnose(owner, episode, labels):
        predictions = []
        method = type(owner).predict

        def predict(model, features):
            result = method(model, features)
            predictions.append(result.copy())
            return result

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(type(owner), "predict", predict)
            report = original(owner, episode, labels)
        assert len(predictions) == 1 and labels == LABELS
        targets = np.asarray([row.action for row in episode.transitions], dtype=np.float64)
        # Independent arithmetic over the already paid prediction. [-1,1] axes.
        error = predictions[0] - targets
        np.testing.assert_allclose(
            [v.summary.normalized_half_mse for v in report.residuals.stages],
            [0.5 * float(np.mean(error[i] ** 2)) for i in range(2)],
            rtol=1e-14,
        )
        assert report.residuals.overall.normalized_half_mse == pytest.approx(
            0.5 * float(np.mean(error**2))
        )
        observed.append(report)
        return report

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo, "diagnose_residual", diagnose)
        result = demo.run_continuous_learnability(
            *values,
            phase_updates=1,
            learner_family=request.param,
            a_only_control=True,
            diagnostic_stage_bindings=binding_pair(values),
        )
    assert result.report["complete"] is True, result.report.get("error")
    return request.param, values, result, observed


def test_should_partition_original_rows_with_one_existing_prediction_and_unchanged_work(completed):
    family, values, result, observed = completed
    assert len(observed) == 8 and all(v.prediction_calls == 1 for v in observed)
    assert result.report["diagnostic_stage_sources"] == readiness_stage_metadata(
        binding_pair(values)
    )
    for checkpoint in result.report["checkpoints"]:
        for name in ("warm", "new"):
            groups = checkpoint["contexts"][name]["residual"]["residuals"]["stages"]
            assert [(v["label"], v["row_indices"], v["summary"]["rows"]) for v in groups] == [
                ("lift", (0,), 1),
                ("align", (1,), 1),
            ]
    work = result.report["work"]
    assert work["optimizer_updates"] == 3 and work["training_rows"] == 6
    assert work["fit_calls"] == 1 and work["fit_rows"] == 2
    assert work["residual_calls"] == 8 and work["residual_rows"] == 16
    assert (
        work["stage_residual_summaries"] == 16 and work["stage_residual_group_index_visits"] == 32
    )
    assert result.report["settings"]["probe_reads"] == result.report["settings"]["final_reads"] == 0
    assert len(result.report["applied_updates"]) == 3
    assert result.candidate.snapshot_state().completed_updates == 3


def test_should_bind_raw_then_projected_stages_and_serve_same_frozen_candidate(
    completed,
    monkeypatch,
    tmp_path,
):
    family, values, result, _ = completed
    owner = result.candidate.fork()
    before = owner.snapshot_state()
    cached = replace(result, candidate=owner, report=deepcopy(result.report))
    captured = []

    def run(warm, new, **options):
        captured.append((warm, new, options))
        return cached

    environment = QuaternionEnvironment()
    monkeypatch.setattr(entry, "run_continuous_learnability", run)
    monkeypatch.setattr(native, "_make_environment", lambda **options: environment)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-stage-partition")
    report = entry.run_learning_rollout(
        tmp_path / "native.json",
        *episodes(),
        phase_updates=1,
        learner_family=family,
        a_only_control=True,
        canonical_state_quaternions=True,
        diagnostic_stage_bindings=binding_pair(episodes()),
    )
    assert report["complete"] is True, report.get("error")
    assert captured[0][:2] == values
    actual = captured[0][2]["diagnostic_stage_bindings"]
    assert tuple(v.episode for v in actual) == values
    assert all(v.stages == LABELS for v in actual)
    assert len(environment.commands) == 2 and environment.close_calls == 1
    assert report["candidate_unchanged"] is report["rollout"]["frozen_state_exact"] is True
    assert_state(before, owner.snapshot_state())


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "wrong_rows",
        "wrong_partition",
        "unrequested",
        "missing_checkpoint",
        "duplicated_checkpoints",
        "reordered_checkpoints",
    ],
)
def test_should_refuse_cached_changed_diagnostics_before_native(
    completed, monkeypatch, tmp_path, fault
):
    family, _, result, _ = completed
    changed = deepcopy(result.report)
    options: dict[str, Any] = dict(diagnostic_stage_bindings=binding_pair(episodes()))
    if fault == "missing":
        del changed["diagnostic_stage_sources"]
    elif fault == "wrong_rows":
        changed["diagnostic_stage_sources"]["warm"]["rows"] = 2.0
    elif fault == "wrong_partition":
        changed["checkpoints"][0]["contexts"]["warm"]["residual"]["residuals"]["stages"][0][
            "row_indices"
        ] = (True,)
    elif fault == "missing_checkpoint":
        del changed["checkpoints"][0]
    elif fault == "duplicated_checkpoints":
        changed["checkpoints"] = [deepcopy(changed["checkpoints"][-1]) for _ in range(4)]
    elif fault == "reordered_checkpoints":
        changed["checkpoints"][0], changed["checkpoints"][1] = (
            changed["checkpoints"][1],
            changed["checkpoints"][0],
        )
    else:
        options = {}
    monkeypatch.setattr(
        entry,
        "run_continuous_learnability",
        lambda *args, **kwargs: replace(result, report=changed),
    )
    monkeypatch.setattr(entry, "run_frozen_rollout", forbidden)
    observed = entry.run_learning_rollout(
        tmp_path / "refused.json",
        *episodes(),
        phase_updates=1,
        learner_family=family,
        a_only_control=True,
        canonical_state_quaternions=True,
        **options,
    )
    assert observed["complete"] is False and observed["rollout_attempted"] is False
    assert observed["training"] == changed
    assert "stage" in observed["error"]


def test_should_refuse_mismatched_input_binding_before_fit(monkeypatch):
    values = projected()
    changed = replace(values[0], episode_id="pick-cube-task-teacher-17")
    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(
            changed, values[1], diagnostic_stage_bindings=binding_pair(values)
        )


def test_should_stop_before_first_update_when_preceding_stage_metadata_changes(monkeypatch):
    values = projected()
    original = demo._checkpoint

    def checkpoint(report, *args, **options):
        original(report, *args, **options)
        report["diagnostic_stage_sources"]["warm"]["rows"] = 2.0

    monkeypatch.setattr(demo, "_checkpoint", checkpoint)
    result = demo.run_continuous_learnability(
        *values, phase_updates=1, diagnostic_stage_bindings=binding_pair(values)
    )
    assert result.candidate is None and result.report["complete"] is False
    assert result.report["progress"]["updates_completed"] == 0
    assert result.report["applied_updates"] == [] and "stage" in result.report["error"]


@pytest.mark.parametrize("fault", ["label", "deleted_checkpoints"])
def test_should_keep_all_paid_receipts_but_refuse_final_elapsed_partition_mutation(
    monkeypatch, fault
):
    values = projected()
    captured: list[dict[str, Any]] = []
    original = demo._checkpoint

    def checkpoint(report, *args, **options):
        original(report, *args, **options)
        if not captured:
            captured.append(report)

    def timer():
        if captured and "progress" in captured[0]:
            if fault == "deleted_checkpoints":
                captured[0]["checkpoints"].clear()
            else:
                captured[0]["checkpoints"][0]["contexts"]["warm"]["residual"]["residuals"][
                    "stages"
                ][0]["label"] = "grasp"
        return 0.0

    monkeypatch.setattr(demo, "_checkpoint", checkpoint)
    monkeypatch.setattr(demo, "monotonic", timer)
    result = demo.run_continuous_learnability(
        *values, phase_updates=1, diagnostic_stage_bindings=binding_pair(values)
    )
    assert result.candidate is None and result.report["complete"] is False
    assert result.report["progress"]["updates_completed"] == 3
    assert len(result.report["applied_updates"]) == 3
    assert result.report["progress"]["next_tick"] == 6
    assert "stage" in result.report["error"]


@pytest.mark.parametrize("selected", [False, True])
def test_should_forward_cli_qualified_stage_partition_only_when_selected(
    monkeypatch, tmp_path, selected
):
    values = episodes()
    calls = []
    monkeypatch.setattr(
        entry,
        "_read_readiness_teacher",
        lambda path, seed, **options: (values[0 if seed == 17 else 1], dict(stages=list(LABELS))),
    )

    def run(*args, **options):
        calls.append(options)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_learning_rollout", run)
    args = [
        "--output",
        str(tmp_path / "out.json"),
        "--warm-teacher",
        "warm.json",
        "--new-teacher",
        "new.json",
        "--readiness-teachers",
    ]
    if selected:
        args += ["--readiness-stage-residuals"]
    assert entry.main(args) == 0
    assert ("diagnostic_stage_bindings" in calls[0]) is selected
    if selected:
        assert calls[0]["diagnostic_stage_bindings"] == binding_pair(values)


def test_should_refuse_stage_cli_without_qualified_readiness_reader(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "_read_teacher", forbidden)
    with pytest.raises(SystemExit) as refusal:
        entry.main(
            [
                "--output",
                str(tmp_path / "out.json"),
                "--warm-teacher",
                "warm.json",
                "--new-teacher",
                "new.json",
                "--readiness-stage-residuals",
            ]
        )
    assert refusal.value.code == 2


def test_should_take_cli_stages_from_actual_qualified_reader_without_inventing_schedule(
    monkeypatch, tmp_path
):
    from test_readiness_stream_input import receipt, write

    warm_path = write(tmp_path / "warm.json", receipt(17))
    new_path = write(tmp_path / "new.json", receipt(19))
    original = warm_path.read_bytes(), new_path.read_bytes()
    calls = []

    def run(output, warm, new, **options):
        calls.append((warm, new, options["diagnostic_stage_bindings"]))
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_learning_rollout", run)
    assert (
        entry.main(
            [
                "--output",
                str(tmp_path / "out.json"),
                "--warm-teacher",
                str(warm_path),
                "--new-teacher",
                str(new_path),
                "--readiness-teachers",
                "--readiness-stage-residuals",
            ]
        )
        == 0
    )
    warm, new, bindings = calls[0]
    assert tuple(v.episode for v in bindings) == (warm, new)
    assert all(v.stages == ("descend", "grasp") for v in bindings)
    assert (warm_path.read_bytes(), new_path.read_bytes()) == original


def test_should_keep_default_wrapper_dispatch_without_new_keyword(monkeypatch):
    calls = []

    def run(*args, **options):
        calls.append(options)
        return demo.ContinuousLearnabilityResult({}, None)

    monkeypatch.setattr(demo, "run_continuous_learnability", run)
    demo.run_continuous_learnability_demo(*projected())
    assert calls == [dict(expected_standardization=None)]
    bindings = binding_pair(projected())
    demo.run_continuous_learnability_demo(*projected(), diagnostic_stage_bindings=bindings)
    assert calls[-1]["diagnostic_stage_bindings"] == bindings


def test_should_count_replay_endpoints_from_private_schedule_without_models():
    from types import SimpleNamespace
    from src.app.continuous_stream_rehearsal import ContinuousReplayPlan

    warm = SimpleNamespace(transitions=(0, 0))
    new = SimpleNamespace(transitions=(0, 0, 0))
    plan = ContinuousReplayPlan(every_updates=3, max_replay_examples=20)
    assert demo._readiness_checkpoint_counts(warm, new, 2, False, plan, 2) == (
        (0, 0),
        (2, 4),
        (5, 12),
        (8, 18),
    )
    assert demo._readiness_checkpoint_counts(warm, new, 2, True, plan, 2) == (
        (0, 0),
        (2, 4),
        (5, 10),
        (8, 16),
    )
