"""Owned final-learner API: one shared24-SGD tiny run, zero-SGD failure fakes."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, fields
import importlib
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_continuous_learnability_demo import episode, same_state
from src.adapters import continuous_learnability_demo as demo
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.core.continuous_bc import OBJECTIVE
from src.core.learner_ports import TrainingUpdateFailed
from src.core.observation_standardization import ObservationStandardization


native_case = importlib.import_module("artifacts.runs.visual-mlp-native-20261019.native_case")


@pytest.fixture(scope="module")
def completed_result():
    """Exactly24 SGD per invocation; no standalone direct reference here."""
    observed = dict(constructors=[], forks=[], runtimes=[], readouts=[], trains=[])
    original_init = NumpyContinuousBCLearner.__init__
    original_fork = NumpyContinuousBCLearner.fork
    original_train = NumpyContinuousBCLearner.train_batch
    original_runtime = demo.ActorShadowRuntime
    original_checkpoint = demo._checkpoint
    warm, new = episode(), episode("B")
    original_inputs = asdict(warm), asdict(new)
    with pytest.MonkeyPatch.context() as patch:

        def initialize(owner, *args, **kwargs):
            observed["constructors"].append(owner)
            original_init(owner, *args, **kwargs)

        def fork(owner):
            child = original_fork(owner)
            observed["forks"].append((owner, child))
            return child

        def train(owner, features, targets):
            observed["trains"].append((owner, len(features)))
            return original_train(owner, features, targets)

        def runtime(source, **options):
            owner = original_runtime(source, **options)
            observed["runtimes"].append((source, owner, options, source.snapshot_state()))
            return owner

        def checkpoint(*args, **kwargs):
            observed["readouts"].append(args[3])
            return original_checkpoint(*args, **kwargs)

        patch.setattr(NumpyContinuousBCLearner, "__init__", initialize)
        patch.setattr(NumpyContinuousBCLearner, "fork", fork)
        patch.setattr(NumpyContinuousBCLearner, "train_batch", train)
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(demo, "_checkpoint", checkpoint)
        result = demo.run_continuous_learnability(
            warm,
            new,
            expected_standardization=ObservationStandardization((1.0, 0.0), (1.0, 1.0), 2),
        )
    assert (asdict(warm), asdict(new)) == original_inputs
    assert result.report["complete"] is True
    return result, observed, warm, new


def plain_report(value):
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            plain_report(item)
    elif type(value) in (tuple, list):
        for item in value:
            plain_report(item)
    else:
        assert type(value) in (type(None), bool, int, float, str)
        if type(value) is float:
            assert math.isfinite(value)


def test_should_return_the_existing_detached_readout_without_new_initialization_or_fork(
    completed_result,
):
    result, observed, _, _ = completed_result
    assert type(result) is demo.ContinuousLearnabilityResult
    assert tuple(field.name for field in fields(result)) == ("report", "candidate")
    assert type(result.candidate) is NumpyContinuousBCLearner
    source, runtime, options, initial = observed["runtimes"][0]
    assert observed["constructors"] == [source]
    assert len(observed["forks"]) == 3
    assert all(owner is source for owner, _ in observed["forks"])
    children = [child for _, child in observed["forks"]]
    assert {id(child) for child in children} == {
        id(runtime.actor._learner),
        id(runtime._candidate),
        id(result.candidate),
    }
    assert len({id(source), *(id(child) for child in children)}) == 4
    assert len(observed["readouts"]) == 4 and all(
        owner is result.candidate for owner in observed["readouts"]
    )
    assert result.candidate is not source and result.candidate is not runtime._candidate
    assert len(observed["trains"]) == 24
    assert all(owner is runtime._candidate for owner, _ in observed["trains"])
    assert [rows for _, rows in observed["trains"]] == [2] * 8 + [3] * 8 + [2] * 8
    assert (
        options["budget"].updates_completed == 24
        and options["budget"].replay_examples_completed == 0
    )
    same_state(initial, source.snapshot_state())
    same_state(initial, runtime.actor.snapshot_state().state)
    same_state(result.candidate.snapshot_state(), runtime.candidate_snapshot().state)


def test_should_retain_full_default_train_policy_parameters_momentum_and_final_counters(
    completed_result,
):
    result, observed, _, _ = completed_result
    state = result.candidate.snapshot_state()
    assert state.completed_updates == 24 and state.training_examples == 56
    assert state.policy.training_role == "train" and state.policy.seed == 17
    assert state.policy.hidden_dim == 64 and state.policy.learning_rate == 0.01
    assert state.policy.momentum == 0.9 and state.policy.binary_gripper_axis is None
    assert state.policy.observation_standardization == ObservationStandardization(
        (1.0, 0.0), (1.0, 1.0), 2
    )
    assert asdict(state.policy) == result.report["policy"]
    assert [value.shape for value in state.parameters] == [(2, 64), (64,), (64, 2), (2,)]
    assert [value.shape for value in state.velocity] == [(2, 64), (64,), (64, 2), (2,)]
    assert all(
        value.dtype == np.float64 and np.isfinite(value).all()
        for value in (*state.parameters, *state.velocity)
    )
    assert any(np.any(value != 0) for value in state.velocity)
    assert result.report["progress"]["updates_completed"] == 24
    assert result.report["checkpoints"][-1]["training_examples"] == 56
    same_state(state, observed["runtimes"][0][1].candidate_snapshot().state)


def test_should_keep_result_container_frozen_while_report_is_plain_mutable_owned_data(
    completed_result,
):
    result, _, _, _ = completed_result
    with pytest.raises(FrozenInstanceError):
        result.candidate = None
    with pytest.raises(FrozenInstanceError):
        result.report = {}
    plain_report(result.report)
    json.dumps(result.report, allow_nan=False)
    before = result.candidate.snapshot_state()
    original_report = deepcopy(result.report)
    try:
        result.report["policy"]["training_role"] = "final_test"
        result.report["policy"]["observation_standardization"]["means"] = (999.0, 999.0)
        result.report["checkpoints"][-1]["contexts"]["warm"]["actor_predictions"][0][0] = 999.0
        result.report["applied_updates"][0]["diagnostic"]["value"] = 999.0
        result.report["work"]["optimizer_updates"] = 999
        same_state(before, result.candidate.snapshot_state())
        assert result.candidate.policy.training_role == "train"
    finally:
        result.report.clear()
        result.report.update(original_report)


def test_should_own_snapshot_policy_and_both_array_groups_without_aliasing_returned_model(
    completed_result,
):
    result, _, _, _ = completed_result
    before = result.candidate.snapshot_state()
    borrowed = result.candidate.snapshot_state()
    for value in (*borrowed.parameters, *borrowed.velocity):
        value.fill(999.0)
    object.__setattr__(borrowed.policy.observation_standardization, "means", (999.0, 999.0))
    object.__setattr__(borrowed.policy, "training_role", "final_test")
    same_state(before, result.candidate.snapshot_state())


def test_should_allow_owned_fork_changes_without_mutating_returned_candidate_or_runtime(
    completed_result,
):
    result, observed, _, _ = completed_result
    before = result.candidate.snapshot_state()
    child = result.candidate.fork()
    modified = child.snapshot_state()
    modified.parameters[0][0, 0] += 0.5
    modified.velocity[0][0, 0] += 0.25
    child.restore_state(modified)
    assert child is not result.candidate
    assert child.snapshot_state().parameters[0][0, 0] != before.parameters[0][0, 0]
    same_state(before, result.candidate.snapshot_state())
    same_state(before, observed["runtimes"][0][1].candidate_snapshot().state)
    for group in ("parameters", "velocity"):
        for left, right in zip(
            getattr(child._state, group), getattr(result.candidate._state, group), strict=True
        ):
            assert not np.shares_memory(left, right)


@pytest.mark.parametrize("role", ["inner_guard", "probe_train", "probe_test", "final_test"])
def test_should_preserve_returned_candidate_training_role_without_releasing_other_roles(
    completed_result, monkeypatch, role
):
    result, _, _, _ = completed_result
    before = result.candidate.snapshot_state()
    monkeypatch.setattr(
        result.candidate,
        "train_batch",
        lambda *_a, **_k: pytest.fail("held-out record reached update"),
    )
    with pytest.raises(ValueError):
        result.candidate.train_episode(episode(role=role))
    same_state(before, result.candidate.snapshot_state())


@pytest.mark.parametrize("reference", [None, ObservationStandardization((1.0, 0.0), (1.0, 1.0), 2)])
def test_should_delegate_legacy_json_wrapper_once_without_model_work_or_exposure(
    monkeypatch, reference
):
    payload = {"complete": True, "schema_version": 1, "preserved": [1, 2, 3]}
    sentinel = object()
    calls = []

    def execute(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(report=payload, candidate=sentinel)

    monkeypatch.setattr(demo, "run_continuous_learnability", execute)
    monkeypatch.setattr(
        demo,
        "NumpyContinuousBCLearner",
        lambda *_a, **_k: pytest.fail("JSON wrapper initialized a model"),
    )
    warm, new = episode(), episode("B")
    report = demo.run_continuous_learnability_demo(warm, new, expected_standardization=reference)
    assert report is payload
    assert calls == [((warm, new), {"expected_standardization": reference})]
    plain_report(report)
    assert sentinel not in report.values()


@pytest.mark.parametrize(
    "failure", ["checkpoint", "stale_success", "before_update", "native_refusal", "missing_updates"]
)
def test_should_return_no_candidate_for_incomplete_execution_without_any_sgd(monkeypatch, failure):
    monkeypatch.setattr(
        NumpyContinuousBCLearner,
        "train_batch",
        lambda *_a, **_k: pytest.fail("failure fixture executed SGD"),
    )
    if failure in ("checkpoint", "stale_success"):

        def checkpoint(report, *_args, **_kwargs):
            if failure == "stale_success":
                report["complete"] = True
            raise RuntimeError("synthetic endpoint refusal")

        monkeypatch.setattr(demo, "_checkpoint", checkpoint)
    elif failure == "missing_updates":
        monkeypatch.setattr(demo, "_apply_packet", lambda *_a, **_k: None)
    else:

        def apply(*_args, **_kwargs):
            if failure == "native_refusal":
                raise TrainingUpdateFailed(
                    "synthetic no-work refusal",
                    OBJECTIVE,
                    {"gradient_evaluations_attempted": 0, "gradient_evaluations_completed": 0},
                )
            raise RuntimeError("synthetic update admission refusal")

        monkeypatch.setattr(demo, "_apply_packet", apply)
    result = demo.run_continuous_learnability(episode(), episode("B"))
    assert type(result) is demo.ContinuousLearnabilityResult
    assert result.candidate is None and result.report["complete"] is False
    assert result.report["error"]
    assert result.report["progress"]["updates_completed"] == 0
    assert result.report["applied_updates"] == []
    assert "work" not in result.report and "metrics" not in result.report
    if failure == "native_refusal":
        assert result.report["failed_native_work"] == dict(
            gradient_evaluations_attempted=0, gradient_evaluations_completed=0
        )
    elif failure == "missing_updates":
        assert "final empty poll" in result.report["error"]


def test_should_keep_admission_errors_as_exceptions_without_creating_a_result_or_model(monkeypatch):
    monkeypatch.setattr(
        demo, "fit_training_standardization", lambda *_: pytest.fail("invalid role reached fit")
    )
    monkeypatch.setattr(
        demo,
        "NumpyContinuousBCLearner",
        lambda *_a, **_k: pytest.fail("invalid role constructed model"),
    )
    with pytest.raises(ValueError, match="TRAIN"):
        demo.run_continuous_learnability(episode(), episode("B", role="inner_guard"))


@pytest.mark.parametrize("observed,expected", [(0.0, 1e-10), (0.0, -1e-10), (1.0, 1.0), (0, 0.0)])
def test_should_accept_only_the_frozen_absolute_tolerance_in_native_reproduction(
    observed, expected
):
    native_case.near(observed, expected)


@pytest.mark.parametrize(
    "observed,expected",
    [
        (0.0, 1.000001e-10),
        (1e8, 1e8 + 1e-7),
        (float("nan"), 0.0),
        (float("inf"), float("inf")),
        (True, 1.0),
        (0.0, False),
        ("1.0", 1.0),
        (np.float64(1.0), 1.0),
    ],
)
def test_should_refuse_nonfinite_non_native_or_outside_fixed_tolerance_values(observed, expected):
    with pytest.raises(ValueError, match="absolute tolerance"):
        native_case.near(observed, expected)


def test_should_reuse_synthetic_complete_report_for_native_gate_without_more_sgd(completed_result):
    result, _, _, _ = completed_result
    previous = json.loads(json.dumps(result.report, allow_nan=False))
    original = deepcopy(result.report)
    native_case.verify_reproduction(result.report, previous)
    assert result.report == original


@pytest.mark.parametrize(
    "bad",
    [
        "missing_loss",
        "changed_loss",
        "nonfinite_loss",
        "objective",
        "receipt_count",
        "event_count",
        "endpoint_count",
        "endpoint_loss",
        "metric",
        "policy",
        "source_identity",
        "reference_unchecked",
        "state_flag",
        "clock_type",
        "endpoint_identity",
    ],
)
def test_should_refuse_bad_reproduction_before_any_native_creation_without_extra_training(
    completed_result, monkeypatch, bad
):
    result, _, _, _ = completed_result
    actual = deepcopy(result.report)
    previous = json.loads(json.dumps(result.report, allow_nan=False))
    if bad == "missing_loss":
        del actual["applied_updates"][0]["diagnostic"]["value"]
    elif bad == "changed_loss":
        actual["applied_updates"][0]["diagnostic"]["value"] += 1e-7
    elif bad == "nonfinite_loss":
        actual["applied_updates"][0]["diagnostic"]["value"] = float("nan")
    elif bad == "objective":
        actual["applied_updates"][0]["diagnostic"]["definition"] = "changed_objective"
    elif bad == "receipt_count":
        actual["applied_updates"].pop()
    elif bad == "event_count":
        actual["events"].pop()
    elif bad == "endpoint_count":
        actual["checkpoints"].pop()
    elif bad == "endpoint_loss":
        actual["checkpoints"][1]["contexts"]["warm"]["residual"]["residuals"]["overall"][
            "normalized_half_mse"
        ] += 1e-7
    elif bad == "metric":
        actual["metrics"]["warm_retention_loss"] += 1e-7
    elif bad == "policy":
        actual["policy"]["momentum"] = 0.8
    elif bad == "source_identity":
        actual["source_episodes"]["warm"]["episode_id"] = "changed-source"
    elif bad == "reference_unchecked":
        actual["expected_standardization_checked"] = False
    elif bad == "state_flag":
        actual["checkpoints"][-1]["owners_exact"] = False
    elif bad == "clock_type":
        actual["applied_updates"][0]["observed_at"] = False
    else:
        actual["checkpoints"][1]["contexts"]["warm"]["episode_id"] = "changed-endpoint"
    monkeypatch.setattr(
        native_case,
        "rollout",
        lambda *_a, **_k: pytest.fail("failed reproduction admitted native work"),
    )
    with pytest.raises((ValueError, KeyError)):
        native_case.verify_reproduction(actual, previous)


def test_should_compare_returned_owner_policy_to_report_before_native_recipe_counters(
    completed_result,
):
    result, _, _, _ = completed_result
    report = deepcopy(result.report)
    report["policy"]["momentum"] = 0.8
    changed = demo.ContinuousLearnabilityResult(report, result.candidate)
    with pytest.raises(ValueError, match="policy differs"):
        native_case.require_candidate(changed)
    # A correct tiny result is still not the admitted29/30 native recipe.
    with pytest.raises(ValueError, match="counters differ"):
        native_case.require_candidate(result)


@pytest.mark.parametrize(
    "name",
    [
        "policy-rollout.json",
        "training-report.json",
        "rollout-audit.json",
        "a-rollout.json",
        "b-rollout.json",
    ],
)
def test_should_refuse_existing_native_evidence_before_loading_or_training(
    monkeypatch, tmp_path, name
):
    path = tmp_path / name
    path.write_bytes(b"preserved failed or accepted native evidence\n")
    monkeypatch.setattr(native_case, "stage", tmp_path)
    monkeypatch.setattr(sys, "argv", ["native", "--output", str(tmp_path / "policy-rollout.json")])
    monkeypatch.setattr(
        native_case,
        "run_case",
        lambda *_: pytest.fail("existing evidence admitted reads/training/native work"),
    )
    with pytest.raises(FileExistsError):
        native_case.main()
    assert path.read_bytes() == b"preserved failed or accepted native evidence\n"


def test_should_refuse_native_output_alias_before_loading_or_training(monkeypatch, tmp_path):
    monkeypatch.setattr(native_case, "stage", tmp_path)
    monkeypatch.setattr(sys, "argv", ["native", "--output", str(tmp_path / "training-report.json")])
    monkeypatch.setattr(
        native_case,
        "run_case",
        lambda *_: pytest.fail("invalid output admitted recipe/native work"),
    )
    with pytest.raises(ValueError, match="fixed summary"):
        native_case.main()
    assert not list(tmp_path.iterdir())
