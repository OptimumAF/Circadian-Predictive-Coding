"""Lighting acquisition with actual nearest/runtime and bounded fake RGB IO.

The historical source-admission seam is mocked, without claiming provenance or
physical success. Per full module: 16 constructors/restores (42 restored rows),
34 predictions, seven two-row appends, three unique public packets, and zero
fitting/SGD/RNG. Fake IO totals 234 calls: factory/reset/close each18, light reads
116, writes/refreshes each15, and controls34 (conservative bound300).
"""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import json
import sys
from typing import Any

import numpy as np
import pytest

from src.adapters.numpy_nearest_continuous_bc import (
    NearestContinuousBCPolicy,
    NearestContinuousBCSnapshot,
    NumpyNearestContinuousBCLearner,
)
from src.app.actor_shadow import StableActor
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_lighting_context import pick_cube_lighting_context
from src.infra import maniskill_continuous as port
from test_lighting_context_interfaces import qualified_result
from test_pick_cube_lighting_context import Environment


case = importlib.import_module("artifacts.runs.lighting-context-return-20261019.native_case")


def write_json(path, value):
    with path.open("x", encoding="utf8") as handle:
        json.dump(value, handle, allow_nan=False)


def equal_state(expected, actual):
    assert actual.policy == expected.policy
    assert actual.completed_updates == expected.completed_updates
    assert actual.training_examples == expected.training_examples
    np.testing.assert_array_equal(actual.prototype_features, expected.prototype_features)
    np.testing.assert_array_equal(actual.prototype_actions, expected.prototype_actions)


@pytest.fixture
def synthetic_case(monkeypatch, tmp_path):
    result = qualified_result()
    new = result.training_episode()
    warm = replace(
        new,
        context="pick-cube",
        episode_id="pick-cube-task-teacher-17",
        transitions=tuple(replace(row, action=(-1.0, 0.0, 0.0, 1.0)) for row in new.transitions),
    )
    new = replace(
        new, transitions=tuple(replace(row, action=(1.0, 0.0, 0.0, 1.0)) for row in new.transitions)
    )
    result = replace(result, candidate_episode=new)
    raw = dict(
        asdict(result),
        qualified=True,
        seed=17,
        context=new.context,
        role="train",
        task="PickCube-v1",
        controller="pd_ee_delta_pos",
        optimizer_updates=0,
    )
    policy = NearestContinuousBCPolicy(
        warm.spec, ObservationStandardization((0.0,) * 789, (1.0,) * 789, 2)
    )
    initial = NearestContinuousBCSnapshot(
        policy,
        np.asarray([row.observation for row in warm.transitions], dtype=np.float64),
        np.asarray([row.action for row in warm.transitions], dtype=np.float64),
        1,
        2,
    )
    observed: dict[str, Any] = dict(
        warm=warm,
        new=new,
        initial=initial,
        teacher=raw,
        loads=0,
        collections=0,
        constructors=[],
        restores=[],
        appends=[],
        runtime=[],
        stream=[],
        environments=[],
        rollouts=[],
        successes={},
        on_return=None,
        on_ready=None,
        load_error=None,
        teacher_error=None,
        fail_at=None,
    )
    monkeypatch.setattr(case, "stage", tmp_path)
    original_init = NumpyNearestContinuousBCLearner.__init__
    original_restore = NumpyNearestContinuousBCLearner.restore_state
    original_train = NumpyNearestContinuousBCLearner.train_batch
    original_runtime = case.ActorShadowRuntime
    original_rollout = case.smoke.run_frozen_rollout
    active: dict[str, Any] = {}

    def load(audit):
        observed["loads"] += 1
        if observed["load_error"] is not None:
            raise observed["load_error"]
        return warm, initial, dict(sparse_return=1.0, any_success=True, final_success=True)

    def collect(audit):
        observed["collections"] += 1
        if observed["teacher_error"] is not None:
            raise observed["teacher_error"]
        returned = deepcopy(observed["teacher"])
        write_json(tmp_path / "teacher-b.json", returned)
        return new, returned

    def initialize(owner, *args, **kwargs):
        observed["constructors"].append(owner)
        original_init(owner, *args, **kwargs)

    def restore(owner, state):
        observed["restores"].append(owner)
        original_restore(owner, state)

    def append(owner, features, targets):
        observed["appends"].append((owner, len(features)))
        return original_train(owner, features, targets)

    def runtime(source, **options):
        owner = original_runtime(source, **options)
        observed["runtime"].append((source, owner, options))
        for name in ("record_experience", "record_label", "train_ready"):
            original = getattr(owner, name)

            def tracked(*args, _name=name, _original=original, **kwargs):
                before = options["budget"].updates_completed
                returned = _original(*args, **kwargs)
                observed["stream"].append(
                    (_name, options["clock"].now(), before, options["budget"].updates_completed)
                )
                if _name == "train_ready" and observed["on_ready"] is not None:
                    observed["on_ready"](owner, options, returned)
                return returned

            monkeypatch.setattr(owner, name, tracked)
        return owner

    class RGBEnvironment(Environment):
        def step(self, action):
            returned = super().step(action)
            reward = np.array(
                [
                    float(
                        observed["successes"].get(active["name"], False) and len(self.commands) == 2
                    )
                ],
                dtype=np.float32,
            )
            return returned[0], reward, *returned[2:]

    def factory(**kwargs):
        assert kwargs == dict(visual_observation=True)
        environment = RGBEnvironment()
        # Default A has no refresh; give its reset the same public initial state.
        environment.reset_packet = environment.refreshed_packet
        if observed["fail_at"] == active["name"]:
            environment.step_failure_at = 1
        observed["environments"].append(environment)
        return environment

    def rollout(output, **options):
        active["name"] = output.name
        observed["rollouts"].append((output.name, options))
        returned = original_rollout(output, **options)
        if observed["on_return"] is not None:
            observed["on_return"](output.name, returned)
        return returned

    monkeypatch.setattr(case, "load_sources", load)
    monkeypatch.setattr(case, "collect_lighting_teacher", collect)
    monkeypatch.setattr(NumpyNearestContinuousBCLearner, "__init__", initialize)
    monkeypatch.setattr(NumpyNearestContinuousBCLearner, "restore_state", restore)
    monkeypatch.setattr(NumpyNearestContinuousBCLearner, "train_batch", append)
    monkeypatch.setattr(case, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(case.smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(case.smoke, "version", lambda _: "synthetic-lighting-case")
    monkeypatch.setattr(port, "_make_environment", factory)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "policy-rollout.json")])
    return observed


def audit_at(directory):
    return json.loads((directory / "rollout-audit.json").read_text(encoding="utf8"))


@pytest.mark.parametrize("warm_already_successful", [False, True])
def test_should_apply_one_delayed_private_append_and_keep_all_six_negative_outcomes(
    synthetic_case, tmp_path, warm_already_successful
):
    observed = synthetic_case
    observed["successes"]["b-before.json"] = warm_already_successful
    assert case.main() == 0
    audit = audit_at(tmp_path)
    assert audit["complete"] is True and "error" not in audit
    source, runtime, options = observed["runtime"][0]
    assert type(runtime.actor) is StableActor
    assert [name for name, _ in observed["rollouts"]] == [
        "b-before.json",
        "b-after.json",
        "policy-rollout.json",
    ]
    assert observed["rollouts"][0][1]["actor"] is runtime.actor
    for index, (_, kwargs) in enumerate(observed["rollouts"]):
        assert kwargs["visual_observation"] is True and kwargs["train_seed"] == 17
        assert kwargs.get("lighting_context") == (
            pick_cube_lighting_context() if index < 2 else None
        )
    assert observed["stream"] == [
        ("record_experience", 0, 0, 0),
        ("train_ready", 0, 0, 0),
        ("train_ready", 1, 0, 0),
        ("record_label", 2, 0, 0),
        ("train_ready", 2, 0, 1),
        ("train_ready", 2, 1, 1),
    ]
    assert options["clock"].now() == 2 and options["budget"].updates_completed == 1
    assert options["budget"].replay_examples_completed == 0
    assert options["max_experiences"] == 1 and options["max_consolidations"] == 0
    assert len(observed["appends"]) == 1 and observed["appends"][0][1] == 2
    assert observed["appends"][0][0] is not source
    equal_state(observed["initial"], source.snapshot_state())
    equal_state(observed["initial"], runtime.actor.snapshot_state().state)
    final = runtime.candidate_snapshot().state
    assert (final.completed_updates, final.training_examples) == (2, 4)
    assert final.policy == observed["initial"].policy
    np.testing.assert_array_equal(
        final.prototype_features[:2], observed["initial"].prototype_features
    )
    np.testing.assert_array_equal(
        final.prototype_actions[:2], observed["initial"].prototype_actions
    )
    for environment in observed["environments"]:
        for command in environment.commands:
            np.testing.assert_array_equal(
                command, np.array([[-1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            )
        assert environment.close_calls == 1
    assert audit["b_after_minus_before"] == {
        name: -float(warm_already_successful)
        for name in ("sparse_return", "any_success", "final_success")
    }
    assert audit["a_reference_minus_return_retention_loss"] == {
        name: 1.0 for name in ("sparse_return", "any_success", "final_success")
    }
    receipt = runtime.applied_updates[0]
    assert receipt.observed_at == 0 and receipt.arrived_at == receipt.applied_at == 2
    assert receipt.diagnostic.definition == case.OBJECTIVE and receipt.diagnostic.value == 2
    assert (receipt.actor_version, receipt.learner_version) == ("serving-0", "candidate-0")
    assert len(observed["constructors"]) == len(observed["restores"]) == 2
    assert audit["distance_candidates"] == 20 and audit["distance_feature_values"] == 20 * 789
    assert audit["normalization_rows"] == 8 and audit["normalization_values"] == 8 * 789
    assert audit["prototype_growth_numeric_bytes"] == audit["inbox_numeric_bytes"] == 2 * 793 * 8
    assert audit["permanent_update_charge"] == 1
    with np.load(tmp_path / "prototype-states.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "warm_features",
            "warm_actions",
            "candidate_features",
            "candidate_actions",
        }
        np.testing.assert_array_equal(
            archive["warm_features"], observed["initial"].prototype_features
        )
        np.testing.assert_array_equal(archive["candidate_actions"], final.prototype_actions)


@pytest.mark.parametrize("failure", ["source", "teacher"])
def test_should_stop_unqualified_prerequisites_before_construction_or_native(
    synthetic_case, tmp_path, failure
):
    observed = synthetic_case
    observed["load_error" if failure == "source" else "teacher_error"] = ValueError(
        f"synthetic {failure} refusal"
    )
    assert case.main() == 1
    assert audit_at(tmp_path)["complete"] is False
    assert (
        observed["constructors"]
        == observed["restores"]
        == observed["runtime"]
        == observed["rollouts"]
        == observed["appends"]
        == []
    )
    assert observed["collections"] == int(failure == "teacher")


@pytest.mark.parametrize("damage", ["readback", "work"])
def test_should_refuse_paid_lighting_mismatch_before_admitting_labels(
    synthetic_case, tmp_path, damage
):
    observed = synthetic_case

    def corrupt(name, returned):
        if name == "b-before.json":
            raw = returned.report
            if damage == "readback":
                raw["lighting_readback"].update(
                    source_ambient_rgb=[0.5, 0.25, 0.125], applied_ambient_rgb=[0.25, 0.125, 0.0625]
                )
            else:
                raw["lighting_work"]["refresh_returns"] = True
            (tmp_path / name).write_text(json.dumps(raw), encoding="utf8")

    observed["on_return"] = corrupt
    assert case.main() == 1
    assert audit_at(tmp_path)["complete"] is False
    assert len(observed["rollouts"]) == 1 and observed["appends"] == observed["stream"] == []
    assert observed["runtime"][0][2]["budget"].updates_completed == 0
    assert len(observed["environments"][0].commands) == 2


@pytest.mark.parametrize("name,predictions", [("b-after.json", 3), ("policy-rollout.json", 5)])
def test_should_retain_paid_append_and_failed_native_prediction(
    synthetic_case, tmp_path, name, predictions
):
    observed = synthetic_case
    observed["fail_at"] = name
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert audit["complete"] is False and audit["permanent_update_charge"] == 1
    assert len(observed["appends"]) == 1 and observed["appends"][0][1] == 2
    assert observed["rollouts"][-1][0] == name
    assert sum(len(environment.commands) for environment in observed["environments"]) == predictions
    raw = json.loads((tmp_path / name).read_text())
    assert raw["complete"] is False and raw["validated_prediction_rows"] == 1
    assert raw["work"] == dict(
        step_attempts=1,
        completed_steps=0,
        completed_control_physics_steps=0,
        possible_failed_physics_steps=5,
    )
    assert all(environment.close_calls == 1 for environment in observed["environments"])


def test_should_revalidate_saved_evidence_after_label_callback_before_paid_append(
    synthetic_case, tmp_path, monkeypatch
):
    observed = synthetic_case
    original_factory = case.ActorShadowRuntime
    mutations = []

    def runtime(source, **options):
        owner = original_factory(source, **options)
        original_label = owner.record_label

        def record_label(*args, **kwargs):
            returned = original_label(*args, **kwargs)
            (tmp_path / "teacher-b.json").write_text(
                '{"replaced_after_label_callback":true}', encoding="utf8"
            )
            mutations.append(True)
            return returned

        monkeypatch.setattr(owner, "record_label", record_label)
        return owner

    monkeypatch.setattr(case, "ActorShadowRuntime", runtime)
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert mutations == [True] and audit["complete"] is False
    assert "preceding retained evidence changed" in audit["error"]
    assert observed["stream"][-1] == ("record_label", 2, 0, 0)
    assert observed["runtime"][0][2]["budget"].updates_completed == 0
    assert audit["permanent_update_charge"] == 0 and observed["appends"] == []
    assert len(observed["rollouts"]) == len(observed["environments"]) == 1
    assert len(observed["environments"][0].commands) == 2
    assert not (tmp_path / "prototype-states.npz").exists()


def test_should_keep_permanent_receipt_after_committed_append_delivery_throws(
    synthetic_case, tmp_path
):
    observed = synthetic_case

    def fail(_runtime, _options, returned):
        if returned:
            raise RuntimeError("synthetic committed delivery failure")

    observed["on_ready"] = fail
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert audit["complete"] is False and "committed delivery" in audit["error"]
    assert audit["permanent_update_charge"] == audit["paid_updates_after_return"] == 1
    assert audit["returned_applied"] == [] and len(audit["retained_applied"]) == 1
    assert (
        audit["retained_applied"][0]["arrived_at"]
        == audit["retained_applied"][0]["applied_at"]
        == 2
    )
    assert len(observed["rollouts"]) == len(observed["appends"]) == 1
    assert not (tmp_path / "prototype-states.npz").exists()


def test_should_refuse_corrupted_receipt_during_final_empty_poll(synthetic_case, tmp_path):
    observed = synthetic_case

    def corrupt(runtime, options, returned):
        if not returned and options["budget"].updates_completed == 1:
            key, receipt = next(iter(getattr(runtime, "_inbox")._applied.items()))
            getattr(runtime, "_inbox")._applied[key] = replace(receipt, arrived_at=3)

    observed["on_ready"] = corrupt
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert audit["complete"] is False and audit["permanent_update_charge"] == 1
    assert audit["retained_applied"][0]["arrived_at"] == 2
    assert len(observed["rollouts"]) == 1 and not (tmp_path / "prototype-states.npz").exists()


def test_should_reread_preceding_trace_after_rollout_local_owner_snapshot(
    synthetic_case, tmp_path, monkeypatch
):
    observed = synthetic_case
    original = NumpyNearestContinuousBCLearner.snapshot_state
    mutated: list[bool] = []

    def snapshot(owner):
        returned = original(owner)
        if (
            len(observed["constructors"]) == 2
            and owner is observed["constructors"][-1]
            and not mutated
        ):
            (tmp_path / "teacher-b.json").write_text(
                '{"replaced_after_snapshot":true}', encoding="utf8"
            )
            mutated.append(True)
        return returned

    monkeypatch.setattr(NumpyNearestContinuousBCLearner, "snapshot_state", snapshot)
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert mutated == [True] and audit["complete"] is False
    assert "preceding retained evidence changed" in audit["error"]
    assert len(observed["rollouts"]) == len(observed["environments"]) == 1
    assert audit["permanent_update_charge"] == 1 and len(observed["appends"]) == 1


def test_should_refuse_actual_serving_actor_mutation_before_private_append(
    synthetic_case, tmp_path
):
    observed = synthetic_case

    def corrupt(name, _returned):
        if name == "b-before.json":
            actor = observed["rollouts"][0][1]["actor"]
            getattr(getattr(actor, "_learner"), "_state").prototype_actions[0, 0] = 0.5

    observed["on_return"] = corrupt
    assert case.main() == 1
    assert audit_at(tmp_path)["complete"] is False
    assert len(observed["rollouts"]) == 1 and observed["appends"] == []
    assert observed["runtime"][0][2]["budget"].updates_completed == 0


def test_should_retain_failed_teacher_result_before_qualification(tmp_path, monkeypatch):
    result = replace(qualified_result(), error="synthetic lighting teacher failure")
    monkeypatch.setattr(case, "stage", tmp_path)
    calls = []

    def collect(**options):
        calls.append(options)
        return result

    monkeypatch.setattr(case, "collect_task_teacher", collect)
    audit: dict[str, Any] = {}
    with pytest.raises(AssertionError, match="did not qualify"):
        case.collect_lighting_teacher(audit)
    retained = json.loads((tmp_path / "teacher-b.json").read_text())
    assert retained["qualified"] is False and retained["error"] == result.error
    assert retained["lighting_work"] == asdict(result.lighting_work)
    assert retained["visual_work"] == asdict(result.visual_work)
    assert calls == [dict(visual_observation=True, lighting_context=pick_cube_lighting_context())]
    assert audit["teacher_attempts"] == audit["teacher_returns"] == 1


@pytest.mark.parametrize("name", ["teacher-b.json", "prototype-states.npz"])
def test_should_preserve_existing_outputs_before_reading_sources(synthetic_case, tmp_path, name):
    output = tmp_path / name
    output.write_bytes(b"preserved evidence")
    with pytest.raises(FileExistsError):
        case.main()
    assert output.read_bytes() == b"preserved evidence"
    assert synthetic_case["loads"] == synthetic_case["collections"] == 0
    assert synthetic_case["constructors"] == synthetic_case["rollouts"] == []
