"""Typed fixed acquisition composition with genuine learning and fake native IO.

One genuine H64/D12/A4 stream initializer supplies all cached boundary cases.
Its base24 plus joint8+8+8 cost48GN/2896rows/192solves/24064 augmented rows;
the selected candidate has40updates/2776rows, a distinct exposure accounting.
Controlled stages reuse those actual results through RNG-free forks, not new
optimization. Native environments implement IO only; no SDK or physics executes.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import joint_acquisition_robot_learning as workflow
from scripts import maniskill_continuous_smoke as smoke
from scripts import run_continuous_robot_learning as legacy
from src.adapters import continuous_learnability_demo as demo
from src.adapters import joint_readiness_corrections as joint
from src.adapters import numpy_continuous_bc as bp
from src.adapters.continuous_learnability_demo import ContinuousLearnabilityResult
from src.core.readiness_stage_diagnostics import ReadinessStageBinding
from src.infra import maniskill_continuous as native
from test_joint_readiness_corrections import assert_state, episode, inputs
from test_joint_train20_rehearsal import retained
from test_maniskill_continuous import FakeEnvironment


def workflow_inputs(*, warm_rows=15, new_rows=22, b_rows=50, c_rows=50, retained_rows=15):
    warm, corrections, curated = inputs(a_rows=warm_rows, b_rows=b_rows, c_rows=c_rows)
    raw_new = episode("b", rows=new_rows)
    raw_new = replace(
        raw_new,
        episode_id="pick-cube-readiness-teacher-19",
        transitions=tuple(
            replace(row, action=(-0.25, 0.0, 0.0, 1.0)) for row in raw_new.transitions
        ),
    )
    new = ReadinessStageBinding(raw_new, ("align",) * new_rows)
    return warm, new, corrections, curated, retained(retained_rows)


def raw(index):
    values = [0.0] * 42
    values[18] = float(index > 0)
    values[36:42] = [-0.02 + index * 0.005, 0.01, -0.1, 0.03, -0.04, 0.15]
    return tuple(values)


class WorkflowEnvironment(FakeEnvironment):
    """Two sparse controls with official taskspace inputs; no physical dynamics."""

    def __init__(self, *, success=True):
        super().__init__()
        self.success = success
        self.reset_observation = np.asarray([raw(0)], dtype=np.float32)

    def step(self, command):
        returned = super().step(command)
        index = len(self.commands)
        return (
            np.asarray([raw(index)], dtype=np.float32),
            np.asarray([float(self.success and index == 2)], dtype=np.float32),
            *returned[2:],
        )


def forbidden(*args, **kwargs):
    pytest.fail("invalid workflow admitted learner, optimization, fit or native work")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        GN_attempts=0,
        GN_returns=0,
        GN_rows_attempted=0,
        GN_rows_returned=0,
        solve_attempts=0,
        solve_completed=0,
        augmented_rows=0,
        prediction_attempts=0,
        prediction_returns=0,
        prediction_rows=0,
        fake_control_attempts=0,
        fake_control_returns=0,
        conditioning_calls=0,
        conditioning_rows=0,
        conditioning_forwards=0,
        conditioning_backwards=0,
        conditioning_gradient_values=0,
        conditioning_gradient_bytes=0,
        SGD=0,
        fits=0,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    train = bp.NumpyContinuousBCLearner.train_head_gauss_newton_batch
    predict = bp.NumpyContinuousBCLearner.predict
    condition = demo.diagnose_conditioning
    step = WorkflowEnvironment.step

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (
            policy.spec.observation_dim + len(policy.spec.action_units)
        )
        return initialize(owner, policy)

    def trained(owner, features, targets):
        counts["GN_attempts"] += 1
        counts["GN_rows_attempted"] += len(features)
        try:
            returned = train(owner, features, targets)
        except Exception as error:
            work = getattr(error, "work", {})
            counts["solve_attempts"] += work.get("solve_attempts", 0)
            counts["solve_completed"] += work.get("solve_completed", 0)
            counts["augmented_rows"] += work.get("solve_rows", 0)
            raise
        counts["GN_returns"] += 1
        counts["GN_rows_returned"] += len(features)
        counts["solve_attempts"] += returned.work["solve_attempts"]
        counts["solve_completed"] += returned.work["solve_completed"]
        counts["augmented_rows"] += returned.work["solve_rows"]
        return returned

    def predicted(owner, features):
        counts["prediction_attempts"] += 1
        counts["prediction_rows"] += len(features)
        returned = predict(owner, features)
        counts["prediction_returns"] += 1
        return returned

    def conditioned(owner, record, *args, **options):
        returned = condition(owner, record, *args, **options)
        work = asdict(returned)["conditioning"]["work"]
        counts["conditioning_calls"] += 1
        counts["conditioning_rows"] += len(record.transitions)
        counts["conditioning_forwards"] += work["forward_passes"]
        counts["conditioning_backwards"] += work["backward_passes"]
        counts["conditioning_gradient_values"] += work["gradient_values"]
        counts["conditioning_gradient_bytes"] += work["gradient_numeric_bytes"]
        return returned

    def stepped(environment, command):
        counts["fake_control_attempts"] += 1
        returned = step(environment, command)
        counts["fake_control_returns"] += 1
        return returned

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", trained)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", predicted)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        patch.setattr(demo, "diagnose_conditioning", conditioned)
        patch.setattr(WorkflowEnvironment, "step", stepped)
        yield counts
    if os.environ.get("JAW_FIXTURE_LEDGER"):
        with Path(os.environ["JAW_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 1 and counts["normal_draws"] == 1024
    assert counts["GN_attempts"] <= 60 and counts["GN_rows_attempted"] <= 6000
    assert counts["solve_attempts"] <= 240 and counts["augmented_rows"] <= 40000
    assert counts["prediction_attempts"] <= 300 and counts["prediction_rows"] <= 5000


def install_sdk(patch, *, success=True):
    environments = []

    def factory(**options):
        assert options == {}
        made = WorkflowEnvironment(success=success)
        environments.append(made)
        return made

    patch.setattr(native, "_make_environment", factory)
    patch.setattr(smoke, "version", lambda _: "fake-joint-workflow")
    return environments


@pytest.fixture(scope="module")
def completed(ledger, tmp_path_factory):
    values = workflow_inputs()
    original = tuple(asdict(value) for value in values)
    captured: dict[str, Any] = dict(native=[], actors=[])
    stream = workflow.run_joint_readiness_stream
    acquire = workflow.run_joint_readiness_corrections
    frozen = workflow.run_frozen_rollout
    factory = joint.ActorShadowRuntime

    def streamed(*args, **options):
        returned = stream(*args, **options)
        captured["source_result"] = returned
        if returned.candidate is not None:
            captured["source_initial"] = returned.candidate.snapshot_state()
        return returned

    def acquired(source, *args, **options):
        captured["acquisition_options"] = deepcopy(options)
        returned = acquire(source, *args, **options)
        captured["acquisition_result"] = returned
        return returned

    def native_rollout(output, **options):
        captured["native"].append(
            (output, options["train_seed"], options["learner"], options["learner"].snapshot_state())
        )
        return frozen(output, **options)

    def runtime(model, **options):
        made = factory(model, **options)
        captured["actors"].append(made)
        return made

    output = tmp_path_factory.mktemp("joint-acquisition") / "case.json"
    with pytest.MonkeyPatch.context() as patch:
        environments = install_sdk(patch)
        patch.setattr(workflow, "run_joint_readiness_stream", streamed)
        patch.setattr(workflow, "run_joint_readiness_corrections", acquired)
        patch.setattr(workflow, "run_frozen_rollout", native_rollout)
        patch.setattr(joint, "ActorShadowRuntime", runtime)
        report = workflow.run_joint_acquisition_rollouts(output, *values)
    return report, output, captured, values, original, environments


def test_should_expose_the_typed_workflow_from_the_existing_runner():
    assert legacy.run_joint_acquisition_rollouts is workflow.run_joint_acquisition_rollouts


def test_should_run_real_stream_and_joint_acquisition_with_one_frozen_three_seed_owner(completed):
    report, output, captured, values, original, environments = completed
    assert report["complete"] is True, report.get("error")
    assert (
        report["schema_version"] == 1
        and report["protocol_id"] == "joint_acquisition_robot_train_v1"
    )
    source = captured["source_result"].candidate
    candidate = captured["acquisition_result"].candidate
    assert type(source) is type(candidate) is bp.NumpyContinuousBCLearner
    assert (
        source.snapshot_state().completed_updates,
        source.snapshot_state().training_examples,
    ) == (32, 1736)
    assert (
        candidate.snapshot_state().completed_updates,
        candidate.snapshot_state().training_examples,
    ) == (40, 2776)
    assert report["candidate_unchanged"] is True
    assert [item[1] for item in captured["native"]] == [17, 19, 20]
    assert all(item[2] is candidate for item in captured["native"])
    assert len(environments) == 3 and all(
        len(environment.commands) == 2 for environment in environments
    )
    assert [environment.reset_calls for environment in environments] == [[17], [19], [20]]
    assert all(environment.close_calls == 1 for environment in environments)
    assert [item["seed"] for item in report["rollouts"]] == [17, 19, 20]
    assert len(report["native_attempts"]) == 3
    assert [request["seed"] for request in report["native_attempts"]] == [17, 19, 20]
    assert all(request["attempted"] is True for request in report["native_attempts"])
    assert [request["output"] for request in report["native_attempts"]] == [
        item["output"] for item in report["rollouts"]
    ]
    assert json.loads(output.read_text(encoding="utf8"))["complete"] is True
    assert tuple(asdict(value) for value in values) == original


def test_should_keep_private_source_actor_hidden_parameters_and_original_paid_scopes_exact(
    completed,
):
    report, _, captured, _, _, _ = completed
    source = captured["source_result"].candidate
    candidate = captured["acquisition_result"].candidate
    before = captured["source_initial"]
    assert_state(source.snapshot_state(), before)
    assert_state(captured["actors"][-1].actor.snapshot_state().state, before)
    final = candidate.snapshot_state()
    for index in (0, 1):
        np.testing.assert_array_equal(final.parameters[index], before.parameters[index])
        np.testing.assert_array_equal(final.velocity[index], before.velocity[index])
    assert candidate is not source and candidate is not captured["actors"][-1]._candidate._learner
    assert len(report["source"]["joint_ab"]["applied_updates"]) == 8
    assert len(report["source"]["joint_abc"]["applied_updates"]) == 8
    assert len(report["acquisition"]["applied_updates"]) == 8
    assert report["acquisition"]["group_training_exposures"] == dict(
        a=120, b=400, curated_a=400, retained_train20=120
    )
    assert report["acquisition"]["work"]["training_rows"] == 1040
    assert report["acquisition"]["work"]["native_update_counts"]["solve_rows"] == 6240
    assert captured["acquisition_options"]["now"] == 1100


@pytest.mark.parametrize(
    "fault",
    [
        "warm_count",
        "new_count",
        "correction_count",
        "curated_count",
        "retained_count",
        "warm_role",
        "retained_role",
        "curated_arrival",
        "flag",
        "type",
    ],
)
def test_should_refuse_invalid_typed_profile_before_model_sdk_or_output(
    monkeypatch, tmp_path, fault
):
    options = {
        "warm_count": dict(warm_rows=14),
        "new_count": dict(new_rows=21),
        "correction_count": dict(b_rows=49),
        "curated_count": dict(c_rows=49),
        "retained_count": dict(retained_rows=14),
    }.get(fault, {})
    values: Any = list(workflow_inputs(**options))
    if fault in ("warm_role", "retained_role"):
        object.__setattr__(values[0 if fault == "warm_role" else 4].episode, "role", "probe_train")
    elif fault == "curated_arrival":
        object.__setattr__(values[3].labels[-1], "arrived_at", 200)
    elif fault == "flag":
        row = values[0].episode.transitions[0]
        observed = list(row.observation)
        observed[18] = False
        object.__setattr__(row, "observation", tuple(observed))
    elif fault == "type":
        values[0] = asdict(values[0])
    output = tmp_path / "invalid.json"
    monkeypatch.setattr(workflow, "run_joint_readiness_stream", forbidden)
    monkeypatch.setattr(workflow, "run_joint_readiness_corrections", forbidden)
    monkeypatch.setattr(workflow, "run_frozen_rollout", forbidden)
    with pytest.raises(ValueError):
        workflow.run_joint_acquisition_rollouts(output, *values)
    assert not output.exists()


@pytest.mark.parametrize("seed", [None, 17, 19, 20])
def test_should_preserve_any_existing_output_before_model_or_sdk(monkeypatch, tmp_path, seed):
    output = tmp_path / "existing.json"
    existing = (
        output if seed is None else output.with_name(output.name + f".train-{seed}.rollout.json")
    )
    existing.write_text("preserved user content", encoding="utf8")
    monkeypatch.setattr(workflow, "run_joint_readiness_stream", forbidden)
    monkeypatch.setattr(workflow, "run_frozen_rollout", forbidden)
    with pytest.raises((ValueError, FileExistsError)):
        workflow.run_joint_acquisition_rollouts(output, *workflow_inputs())
    assert existing.read_text(encoding="utf8") == "preserved user content"


def test_should_charge_additional_input_admission_projections_separately(completed):
    report = completed[0]
    work = report["input_admission_work"]
    assert work["preflight_calls"] == 3
    assert work["projection_rows"] == 347 and work["observation_projections"] == 694
    assert [work[name]["rows"] for name in ("joint_ab", "joint_abc", "retained_acquisition")] == [
        65,
        115,
        130,
    ]
    assert work["source_projection"]["history_advances"] == 37
    assert work["source_projection"]["observation_projections"] == 74
    assert report["source"]["preparation_work"]["runtime_calls_returned"] == 2
    assert report["acquisition"]["preparation_work"]["retained_reference_rows"] == 15


def scripted(monkeypatch, completed, *, failure=None):
    """Reuse actual paid reports/models as controlled ports, with zero new updates."""
    actual = completed[2]
    source = ContinuousLearnabilityResult(
        deepcopy(actual["source_result"].report), actual["source_result"].candidate.fork()
    )
    acquired = ContinuousLearnabilityResult(
        deepcopy(actual["acquisition_result"].report), actual["acquisition_result"].candidate.fork()
    )
    captured: dict[str, Any] = dict(source=source, acquisition=acquired, calls=[], native=[])
    frozen = workflow.run_frozen_rollout

    def stream(*args, **options):
        captured["calls"].append("source")
        assert options == dict(corrections_now=99, curated_now=199)
        if failure == "source_exception":
            raise RuntimeError("controlled source work unreturned")
        if failure == "source":
            return ContinuousLearnabilityResult(
                dict(source.report, complete=False, error="controlled paid source failure"), None
            )
        return source

    def acquire(owner, *args, **options):
        captured["calls"].append("acquisition")
        assert owner is source.candidate and options["now"] == 1100
        if failure == "acquisition_exception":
            raise RuntimeError("controlled acquisition work unreturned")
        if failure == "acquisition":
            return ContinuousLearnabilityResult(
                dict(acquired.report, complete=False, error="controlled paid acquisition failure"),
                None,
            )
        return acquired

    def native_rollout(*args, **options):
        captured["calls"].append(options["train_seed"])
        if failure == "native_exception":
            raise RuntimeError("controlled native work unreturned")
        returned = frozen(*args, **options)
        captured["native"].append(returned)
        return returned

    monkeypatch.setattr(workflow, "run_joint_readiness_stream", stream)
    monkeypatch.setattr(workflow, "run_joint_readiness_corrections", acquire)
    monkeypatch.setattr(workflow, "run_frozen_rollout", native_rollout)
    return captured


def test_should_continue_all_complete_negative_scenes_without_selecting_another_learner(
    completed, monkeypatch, tmp_path
):
    captured = scripted(monkeypatch, completed)
    environments = install_sdk(monkeypatch, success=False)
    report = workflow.run_joint_acquisition_rollouts(tmp_path / "negative.json", *workflow_inputs())
    assert report["complete"] is True, report.get("error")
    assert captured["calls"] == ["source", "acquisition", 17, 19, 20]
    assert len(environments) == 3
    assert all(item["rollout"]["complete"] is True for item in report["rollouts"])
    assert all(
        item["rollout"]["sparse_return"] == 0 and item["rollout"]["any_success"] is False
        for item in report["rollouts"]
    )
    assert report["candidate_unchanged"] is True


@pytest.mark.parametrize("stage", ["source", "acquisition"])
def test_should_keep_paid_training_failure_and_stop_before_next_stage(
    completed, monkeypatch, tmp_path, stage
):
    captured = scripted(monkeypatch, completed, failure=stage)
    monkeypatch.setattr(native, "_make_environment", forbidden)
    report = workflow.run_joint_acquisition_rollouts(
        tmp_path / "paid-failure.json", *workflow_inputs()
    )
    assert report["complete"] is False
    assert report[stage]["complete"] is False and "controlled paid" in report[stage]["error"]
    assert captured["calls"] == (["source"] if stage == "source" else ["source", "acquisition"])
    assert report["native_attempts"] == report["rollouts"] == []
    if stage == "acquisition":
        assert len(report["acquisition"]["applied_updates"]) == 8


@pytest.mark.parametrize("stage", ["source", "acquisition", "native"])
def test_should_record_unreturned_work_as_unknown_without_retry(
    completed, monkeypatch, tmp_path, stage
):
    captured = scripted(monkeypatch, completed, failure=stage + "_exception")
    monkeypatch.setattr(native, "_make_environment", forbidden)
    report = workflow.run_joint_acquisition_rollouts(tmp_path / "unknown.json", *workflow_inputs())
    assert report["complete"] is False and report[f"unreturned_{stage}_work_unknown"] is True
    assert "unreturned" in report["error"]
    assert (
        captured["calls"]
        == {
            "source": ["source"],
            "acquisition": ["source", "acquisition"],
            "native": ["source", "acquisition", 17],
        }[stage]
    )
    assert report["rollouts"] == []


@pytest.mark.parametrize("seed", [17, 19])
def test_should_retain_incomplete_native_prefix_without_trying_a_later_seed(
    completed, monkeypatch, tmp_path, seed
):
    captured = scripted(monkeypatch, completed)
    environments = install_sdk(monkeypatch)
    frozen = workflow.run_frozen_rollout

    def incomplete(*args, **options):
        result = frozen(*args, **options)
        if options["train_seed"] == seed:
            result.report.update(
                complete=False, error="controlled failure after paid native return"
            )
        return result

    monkeypatch.setattr(workflow, "run_frozen_rollout", incomplete)
    report = workflow.run_joint_acquisition_rollouts(tmp_path / "partial.json", *workflow_inputs())
    count = 1 if seed == 17 else 2
    assert report["complete"] is False
    assert len(report["rollouts"]) == len(report["native_attempts"]) == len(environments) == count
    assert (
        report["rollouts"][-1]["rollout"]["error"] == "controlled failure after paid native return"
    )
    assert report["rollouts"][-1]["rollout"]["work"]["completed_steps"] == 2
    assert report["unreturned_native_work_unknown"] is False
    assert captured["calls"][-1] == seed


def test_should_refuse_corrupted_history_even_when_native_exact_flag_is_true(
    completed, monkeypatch, tmp_path
):
    scripted(monkeypatch, completed)
    environments = install_sdk(monkeypatch)
    frozen = workflow.run_frozen_rollout

    def corrupted(*args, **options):
        result = frozen(*args, **options)
        result.report["action_history"]["pending"] = True
        return result

    monkeypatch.setattr(workflow, "run_frozen_rollout", corrupted)
    report = workflow.run_joint_acquisition_rollouts(tmp_path / "history.json", *workflow_inputs())
    assert report["complete"] is False and len(environments) == 1
    assert len(report["rollouts"]) == 1
    assert report["rollouts"][0]["rollout"]["action_history_exact"] is True
    assert "history" in report["error"]


@pytest.mark.parametrize(
    "fault",
    [
        "raw_input",
        "owned_input",
        "metadata_content",
        "metadata_owner",
        "source_raw",
        "native_owner",
        "candidate_state",
    ],
)
def test_should_refuse_changed_prior_evidence_after_first_scene_before_more_native_work(
    completed, monkeypatch, tmp_path, fault
):
    captured = scripted(monkeypatch, completed)
    environments = install_sdk(monkeypatch)
    values = workflow_inputs()
    own = workflow._own_inputs
    write = workflow._write
    owned = []
    mutations: list[bool] = []

    def own_inputs(*args, **options):
        returned = own(*args, **options)
        owned.append(returned)
        return returned

    def observed_write(path, report, **options):
        returned = write(path, report, **options)
        if len(report["rollouts"]) == 1 and not mutations:
            if fault == "raw_input":
                object.__setattr__(values[-1].episode, "context", "changed")
            elif fault == "owned_input":
                object.__setattr__(owned[0][0].episode, "context", "changed")
            elif fault == "metadata_content":
                report["input_admission_work"]["projection_rows"] += 1
            elif fault == "metadata_owner":
                report["input_admission_work"] = deepcopy(report["input_admission_work"])
            elif fault == "source_raw":
                captured["source"].report["joint_ab"]["progress"]["updates_completed"] += 1
            elif fault == "native_owner":
                report["rollouts"][0]["rollout"] = deepcopy(report["rollouts"][0]["rollout"])
            else:
                candidate = captured["acquisition"].candidate
                candidate._state = deepcopy(candidate._state)
            mutations.append(True)
        return returned

    monkeypatch.setattr(workflow, "_own_inputs", own_inputs)
    monkeypatch.setattr(workflow, "_write", observed_write)
    report = workflow.run_joint_acquisition_rollouts(tmp_path / "mutation.json", *values)
    assert mutations == [True] and report["complete"] is False
    assert len(environments) == len(report["native_attempts"]) == len(report["rollouts"]) == 1
    assert captured["calls"] == ["source", "acquisition", 17]
    assert report["candidate_unchanged"] is False
    assert "state_error" in report


def test_should_preserve_later_output_race_and_paid_prefix_without_second_sdk_call(
    completed, monkeypatch, tmp_path
):
    captured = scripted(monkeypatch, completed)
    environments = install_sdk(monkeypatch)
    output = tmp_path / "race.json"
    later = output.with_name(output.name + ".train-19.rollout.json")
    write = workflow._write
    created: list[bool] = []

    def racing_write(path, report, **options):
        returned = write(path, report, **options)
        if len(report["rollouts"]) == 1 and not created:
            later.write_text("concurrent user output", encoding="utf8")
            created.append(True)
        return returned

    monkeypatch.setattr(workflow, "_write", racing_write)
    report = workflow.run_joint_acquisition_rollouts(output, *workflow_inputs())
    assert report["complete"] is False and len(environments) == 1
    assert len(report["rollouts"]) == 1 and len(report["native_attempts"]) == 2
    assert captured["calls"] == ["source", "acquisition", 17]
    assert later.read_text(encoding="utf8") == "concurrent user output"
    assert report["unreturned_native_work_unknown"] is True
