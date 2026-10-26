"""Mass acquisition orchestration with real nearest/runtime and fake native IO.

Synthetic source restoration bypasses historical admission only at load_sources;
it establishes no physical success, accepted provenance or scientific benefit.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib
import json
import sys
from zipfile import ZipFile

import numpy as np
import pytest

from src.adapters.numpy_nearest_continuous_bc import (
    NearestContinuousBCPolicy,
    NearestContinuousBCSnapshot,
    NumpyNearestContinuousBCLearner,
)
from src.app.actor_shadow import StableActor
from src.core.observation_standardization import ObservationStandardization
from src.core.continuous_demonstration import ContinuousTransition
from src.core.pick_cube_mass_context import pick_cube_mass_context
from src.infra import maniskill_continuous as port
from test_mass_context_interfaces import InterfaceEnvironment, qualified_result


case = importlib.import_module("artifacts.runs.mass-context-return-20261019.native_case")
collect_mass_teacher = case.collect_mass_teacher


def write_json(path, value):
    with path.open("x", encoding="utf8") as handle:
        json.dump(value, handle, allow_nan=False)


def equal_state(first, second):
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    np.testing.assert_array_equal(first.prototype_features, second.prototype_features)
    np.testing.assert_array_equal(first.prototype_actions, second.prototype_actions)


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
    statistics = ObservationStandardization((0.0,) * 42, (1.0,) * 42, 2)
    policy = NearestContinuousBCPolicy(warm.spec, statistics)
    initial = NearestContinuousBCSnapshot(
        policy,
        np.asarray([row.observation for row in warm.transitions], dtype=np.float64),
        np.asarray([row.action for row in warm.transitions], dtype=np.float64),
        1,
        2,
    )
    observed = dict(
        warm=warm,
        new=new,
        teacher=raw,
        initial=initial,
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
    active = {}

    def load(audit):
        observed["loads"] += 1
        if observed["load_error"] is not None:
            raise observed["load_error"]
        return warm, initial, dict(sparse_return=1, any_success=True, final_success=True)

    def collect(audit):
        observed["collections"] += 1
        # This seam represents the already qualified public teacher boundary.
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

    class Environment(InterfaceEnvironment):
        def step(self, action):
            output = super().step(action)
            success = observed["successes"].get(active["name"], False)
            reward = np.array(
                [float(success and len(self.commands) == self.steps)], dtype=np.float32
            )
            return output[0], reward, *output[2:]

    def factory(**kwargs):
        assert kwargs == {}, "fixed state mass case must not request rendering"
        environment = Environment()
        if observed["fail_at"] == active["name"]:
            environment.step_error = RuntimeError("synthetic paid native failure")
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
    monkeypatch.setattr(case, "collect_mass_teacher", collect)
    monkeypatch.setattr(NumpyNearestContinuousBCLearner, "__init__", initialize)
    monkeypatch.setattr(NumpyNearestContinuousBCLearner, "restore_state", restore)
    monkeypatch.setattr(NumpyNearestContinuousBCLearner, "train_batch", append)
    monkeypatch.setattr(case, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(case.smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(case.smoke, "version", lambda _: "synthetic-mass-native")
    monkeypatch.setattr(port, "_make_environment", factory)
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "policy-rollout.json")])
    return observed


def audit_at(directory):
    return json.loads((directory / "rollout-audit.json").read_text(encoding="utf8"))


@pytest.mark.parametrize("baseline_success", [False, True])
def test_should_use_actual_actor_delayed_single_append_and_accept_negative_outcomes(
    synthetic_case, tmp_path, baseline_success
):
    observed = synthetic_case
    observed["successes"]["b-before.json"] = baseline_success
    assert case.main() == 0
    audit = audit_at(tmp_path)
    assert audit["complete"] is True and "error" not in audit
    assert observed["loads"] == observed["collections"] == 1
    assert [name for name, _ in observed["rollouts"]] == [
        "b-before.json",
        "b-after.json",
        "policy-rollout.json",
    ]
    source, runtime, options = observed["runtime"][0]
    assert type(runtime.actor) is StableActor
    assert observed["rollouts"][0][1]["actor"] is runtime.actor
    for _, kwargs in observed["rollouts"]:
        assert kwargs["train_seed"] == 17
        assert not kwargs.get("visual_observation", False)
    assert observed["rollouts"][0][1]["mass_context"] == pick_cube_mass_context()
    assert observed["rollouts"][1][1]["mass_context"] == pick_cube_mass_context()
    assert observed["rollouts"][2][1].get("mass_context") is None
    assert options["max_experiences"] == 1 and options["max_consolidations"] == 0
    assert (
        options["budget"].updates_completed == 1
        and options["budget"].replay_examples_completed == 0
    )
    assert options["clock"].now() == 2
    assert observed["stream"] == [
        ("record_experience", 0, 0, 0),
        ("train_ready", 0, 0, 0),
        ("train_ready", 1, 0, 0),
        ("record_label", 2, 0, 0),
        ("train_ready", 2, 0, 1),
        ("train_ready", 2, 1, 1),
    ]
    assert len(observed["appends"]) == 1 and observed["appends"][0][1] == 2
    assert observed["appends"][0][0] is not source
    equal_state(observed["initial"], source.snapshot_state())
    equal_state(observed["initial"], runtime.actor.snapshot_state().state)
    final = runtime.candidate_snapshot().state
    assert final.completed_updates == 2 and final.training_examples == 4
    assert final.policy == observed["initial"].policy
    np.testing.assert_array_equal(
        final.prototype_features[:2], observed["initial"].prototype_features
    )
    np.testing.assert_array_equal(
        final.prototype_actions[:2], observed["initial"].prototype_actions
    )
    # Duplicate public states keep the earliest warm prototype even after new
    # labels arrive. A private append is not a claim of successful adaptation.
    for environment in observed["environments"]:
        for command in environment.commands:
            np.testing.assert_array_equal(
                command, np.array([[-1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            )
    assert audit["b_after_minus_before"] == {
        key: -float(baseline_success) for key in ("sparse_return", "any_success", "final_success")
    }
    assert audit["a_reference_minus_return_retention_loss"] == {
        key: 1.0 for key in ("sparse_return", "any_success", "final_success")
    }
    receipt = runtime.applied_updates[0]
    assert receipt.observed_at == 0 and receipt.arrived_at == receipt.applied_at == 2
    assert receipt.diagnostic.definition == case.OBJECTIVE and receipt.diagnostic.value == 2
    assert receipt.actor_version == "serving-0" and receipt.learner_version == "candidate-0"
    assert len(observed["constructors"]) == len(observed["restores"]) == 2
    assert len(observed["environments"]) == 3
    assert all(environment.close_calls == 1 for environment in observed["environments"])
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
        np.testing.assert_array_equal(archive["candidate_features"], final.prototype_features)
        np.testing.assert_array_equal(archive["candidate_actions"], final.prototype_actions)
    assert (tmp_path / "prototype-states.npz").stat().st_size <= 1024**2
    assert audit["normalization_rows"] == 8 and audit["normalization_values"] == 8 * 42
    assert audit["distance_candidates"] == 20 and audit["distance_feature_values"] == 20 * 42
    assert audit["source_prototype_numeric_bytes"] == 2 * 46 * 8
    assert audit["candidate_prototype_numeric_bytes"] == 4 * 46 * 8
    assert audit["prototype_growth_numeric_bytes"] == audit["inbox_numeric_bytes"] == 2 * 46 * 8
    assert audit["control_attempts"] == 8 and audit["known_control_physics_steps"] == 40


@pytest.mark.parametrize("failure", ["source", "teacher"])
def test_should_stop_failed_prerequisites_before_any_constructor_or_runtime(
    synthetic_case, tmp_path, failure
):
    observed = synthetic_case
    observed["load_error" if failure == "source" else "teacher_error"] = ValueError(
        f"synthetic {failure} evidence refused"
    )
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert audit["complete"] is False and failure in audit["error"]
    assert observed["constructors"] == observed["restores"] == observed["runtime"] == []
    assert observed["rollouts"] == observed["appends"] == []
    assert observed["collections"] == int(failure == "teacher")


@pytest.mark.parametrize(
    "damage",
    ["incomplete", "source_count", "policy", "mass_context", "mass_work", "boolean_return"],
)
def test_should_refuse_invalid_paid_baseline_before_releasing_labels(
    synthetic_case, tmp_path, damage
):
    observed = synthetic_case

    def corrupt(name, returned):
        if name != "b-before.json":
            return
        raw = returned.report
        if damage == "incomplete":
            raw["complete"] = False
        elif damage == "source_count":
            raw["source_training_rows"] += 1
        elif damage == "policy":
            raw["learner_policy"]["training_role"] = "probe_train"
        elif damage == "mass_context":
            raw["mass_context"]["seed"] = 17.0
        elif damage == "mass_work":
            raw["mass_work"]["write_returns"] = True
        else:
            raw["sparse_return"] = False
        (tmp_path / name).write_text(json.dumps(raw), encoding="utf8")

    observed["on_return"] = corrupt
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert audit["complete"] is False
    assert len(observed["rollouts"]) == 1 and len(observed["environments"][0].commands) == 2
    assert observed["appends"] == [] and observed["stream"] == []
    assert observed["runtime"][0][2]["budget"].updates_completed == 0
    assert not (tmp_path / "prototype-states.npz").exists()


@pytest.mark.parametrize(
    "name,paid_updates,predictions",
    [("b-before.json", 0, 1), ("b-after.json", 1, 3), ("policy-rollout.json", 1, 5)],
)
def test_should_keep_failed_native_attempt_and_previous_permanent_append(
    synthetic_case, tmp_path, name, paid_updates, predictions
):
    observed = synthetic_case
    observed["fail_at"] = name
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert audit["complete"] is False and "error" in audit
    assert observed["rollouts"][-1][0] == name
    assert sum(len(environment.commands) for environment in observed["environments"]) == predictions
    assert observed["runtime"][0][2]["budget"].updates_completed == paid_updates
    assert sum(rows for _, rows in observed["appends"]) == 2 * paid_updates
    raw = json.loads((tmp_path / name).read_text(encoding="utf8"))
    assert raw["complete"] is False
    assert raw["work"] == dict(
        step_attempts=1,
        completed_steps=0,
        completed_control_physics_steps=0,
        possible_failed_physics_steps=5,
    )
    assert (
        raw["prediction_attempts"]
        == raw["prediction_returns"]
        == raw["validated_prediction_rows"]
        == 1
    )
    assert raw["validated_prediction_bytes"] == 32
    assert all(environment.close_calls == 1 for environment in observed["environments"])


def test_should_reject_actual_serving_owner_mutation_before_private_append(
    synthetic_case, tmp_path
):
    observed = synthetic_case

    def corrupt(name, _returned):
        if name == "b-before.json":
            actor = observed["rollouts"][0][1]["actor"]
            actor._learner._state.prototype_actions[0, 0] = 0.5

    observed["on_return"] = corrupt
    assert case.main() == 1
    assert len(observed["rollouts"]) == 1 and observed["appends"] == []
    assert observed["runtime"][0][2]["budget"].updates_completed == 0
    assert audit_at(tmp_path)["complete"] is False


def test_should_keep_paid_append_receipt_when_delivery_throws_after_commit(
    synthetic_case, tmp_path
):
    observed = synthetic_case

    def fail_after_commit(_runtime, _options, returned):
        if returned:
            raise RuntimeError("synthetic postcommit delivery failure")

    observed["on_ready"] = fail_after_commit
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert audit["complete"] is False and "postcommit delivery" in audit["error"]
    assert audit["paid_updates_after_return"] == 1 and audit["paid_replay_rows_after_return"] == 0
    assert audit["returned_applied"] == [] and len(audit["retained_applied"]) == 1
    assert (
        audit["retained_applied"][0]["arrived_at"]
        == audit["retained_applied"][0]["applied_at"]
        == 2
    )
    assert len(observed["rollouts"]) == len(observed["appends"]) == 1
    assert not (tmp_path / "prototype-states.npz").exists()


def test_should_refuse_receipt_replacement_during_final_empty_poll(synthetic_case, tmp_path):
    observed = synthetic_case

    def corrupt(runtime, options, returned):
        if not returned and options["budget"].updates_completed == 1:
            key, receipt = next(iter(runtime._inbox._applied.items()))
            runtime._inbox._applied[key] = replace(receipt, arrived_at=3)

    observed["on_ready"] = corrupt
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert audit["complete"] is False and audit["paid_updates_after_return"] == 1
    assert audit["retained_applied"][0]["arrived_at"] == 2
    assert len(observed["rollouts"]) == 1 and not (tmp_path / "prototype-states.npz").exists()


@pytest.mark.parametrize(
    "phase,name,rollouts,charge",
    [
        ("before_baseline", "teacher-b.json", 0, 0),
        ("after_baseline", "b-before.json", 1, 0),
        ("after_candidate_B", "b-after.json", 2, 1),
    ],
)
def test_should_reread_preceding_evidence_after_owner_callbacks_before_more_work(
    synthetic_case, tmp_path, monkeypatch, phase, name, rollouts, charge
):
    observed = synthetic_case
    original = case.guard_owners

    def guard(source, runtime, initial, candidate, audit, current):
        original(source, runtime, initial, candidate, audit, current)
        if current == phase:
            (tmp_path / name).write_text('{"replaced":true}', encoding="utf8")

    monkeypatch.setattr(case, "guard_owners", guard)
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert audit["complete"] is False and "retained evidence changed" in audit["error"]
    assert len(observed["rollouts"]) == rollouts
    assert observed["runtime"][0][2]["budget"].updates_completed == charge
    assert len(observed["appends"]) == charge


def test_should_reject_detached_readout_mutation_before_return_context(synthetic_case, tmp_path):
    observed = synthetic_case

    def corrupt(name, _returned):
        if name == "b-after.json":
            owner = observed["rollouts"][-1][1]["learner"]
            owner._state.prototype_actions[0, 0] = 0.5

    observed["on_return"] = corrupt
    assert case.main() == 1
    assert len(observed["rollouts"]) == 2 and len(observed["appends"]) == 1
    assert observed["runtime"][0][2]["budget"].updates_completed == 1
    assert audit_at(tmp_path)["complete"] is False


@pytest.mark.parametrize("damage", ["trace", "state"])
def test_should_refuse_changed_evidence_after_rollout_local_owner_snapshot(
    synthetic_case, tmp_path, monkeypatch, damage
):
    observed = synthetic_case
    original = NumpyNearestContinuousBCLearner.snapshot_state
    mutated = []

    def snapshot(owner):
        state = original(owner)
        # The readout is constructed only after the private append and archive.
        # Its first snapshot is inside candidate-B rollout admission, after the
        # orchestration guards; mutate evidence while returning unchanged state.
        if (
            len(observed["constructors"]) == 2
            and owner is observed["constructors"][-1]
            and not mutated
        ):
            if damage == "trace":
                (tmp_path / "teacher-b.json").write_text(
                    '{"replaced_after_snapshot":true}', encoding="utf8"
                )
            else:
                state.prototype_actions[0, 0] = 0.5
            mutated.append(True)
        return state

    monkeypatch.setattr(NumpyNearestContinuousBCLearner, "snapshot_state", snapshot)
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert mutated == [True] and audit["complete"] is False
    assert (
        "preceding retained evidence changed" if damage == "trace" else "full learner state changed"
    ) in audit["error"]
    assert len(observed["rollouts"]) == len(observed["environments"]) == 1
    assert observed["runtime"][0][2]["budget"].updates_completed == 1
    assert len(observed["appends"]) == 1 and observed["appends"][0][1] == 2
    assert audit["permanent_update_charge"] == 1
    assert (
        audit["retained_applied"][0]["arrived_at"]
        == audit["retained_applied"][0]["applied_at"]
        == 2
    )


def test_should_refuse_wrong_returned_receipt_without_rewriting_permanent_history(
    synthetic_case, tmp_path, monkeypatch
):
    observed = synthetic_case
    original_runtime = case.ActorShadowRuntime

    def runtime(source, **options):
        owner = original_runtime(source, **options)
        original_ready = owner.train_ready

        def ready(*args, **kwargs):
            returned = original_ready(*args, **kwargs)
            if returned:
                return (replace(returned[0], episode_id="wrong-train-episode"),)
            return returned

        monkeypatch.setattr(owner, "train_ready", ready)
        return owner

    monkeypatch.setattr(case, "ActorShadowRuntime", runtime)
    assert case.main() == 1
    audit = audit_at(tmp_path)
    assert audit["complete"] is False and audit["permanent_update_charge"] == 1
    assert audit["retained_applied"][0]["episode_id"] == observed["new"].episode_id
    assert audit["returned_applied"][0]["episode_id"] == "wrong-train-episode"
    assert len(observed["rollouts"]) == 1 and not (tmp_path / "prototype-states.npz").exists()


@pytest.mark.parametrize(
    "name",
    [
        "policy-rollout.json",
        "teacher-b.json",
        "b-before.json",
        "b-after.json",
        "prototype-states.npz",
        "rollout-audit.json",
    ],
)
def test_should_preserve_existing_evidence_before_any_read_or_new_work(
    synthetic_case, tmp_path, name
):
    path = tmp_path / name
    path.write_bytes(b"preserved evidence")
    with pytest.raises(FileExistsError):
        case.main()
    assert path.read_bytes() == b"preserved evidence"
    assert synthetic_case["loads"] == synthetic_case["collections"] == 0
    assert synthetic_case["constructors"] == synthetic_case["rollouts"] == []


def test_should_refuse_alias_primary_before_reading_sources(synthetic_case, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["case", "--output", str(tmp_path / "teacher-b.json")])
    with pytest.raises((ValueError, AssertionError)):
        case.main()
    assert synthetic_case["loads"] == 0 and not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "damage", [None, "shape", "dtype", "fortran", "extra", "oversize", "nonfinite", "duplicate"]
)
def test_should_bound_two_saved_source_arrays_before_allocation_and_own_payload(
    tmp_path, monkeypatch, damage
):
    features = np.zeros((29, 42), dtype=np.float64)
    actions = np.zeros((29, 4), dtype=np.float64)
    if damage == "shape":
        features = features[:28]
    elif damage == "dtype":
        features = features.astype(np.float32)
    elif damage == "fortran":
        features = np.asfortranarray(features)
    elif damage == "nonfinite":
        features[0, 0] = float("nan")
    values = dict(features=features, actions=actions)
    if damage == "extra":
        values["extra"] = np.zeros(1)
    path = tmp_path / "source.npz"
    with path.open("xb") as handle:
        np.savez(handle, **values)
    if damage == "oversize":
        path.write_bytes(b"x" * (64 * 1024 + 1))
    elif damage == "duplicate":
        with ZipFile(path) as archive:
            payload = archive.read("features.npy")
        with pytest.warns(UserWarning, match="Duplicate name"):
            with ZipFile(path, "a") as archive:
                archive.writestr("features.npy", payload)
    original_bytes = path.read_bytes()
    original_load = np.load
    loaded = []

    def load(*args, **kwargs):
        loaded.append(True)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", load)
    audit = {}
    if damage is None:
        returned = case.read_arrays(path, audit)
        assert set(returned) == {"features", "actions"}
        np.testing.assert_array_equal(returned["features"], features)
        np.testing.assert_array_equal(returned["actions"], actions)
        assert returned["features"].dtype == returned["actions"].dtype == np.float64
        returned["features"].fill(99.0)
        assert audit["source_archive_read_returns"] == 1
        assert audit["source_archive_numeric_bytes"] == 10672
    else:
        with pytest.raises((AssertionError, ValueError)):
            case.read_arrays(path, audit)
    assert len(loaded) == int(damage is None or damage == "nonfinite")
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("qualified", [True, False])
def test_should_retain_public_teacher_result_before_qualification_and_other_work(
    tmp_path, monkeypatch, qualified
):
    result = qualified_result()
    if not qualified:
        result = replace(result, error="synthetic unqualified mass teacher")
    monkeypatch.setattr(case, "stage", tmp_path)
    calls = []

    def collect(**options):
        calls.append(options)
        return result

    monkeypatch.setattr(case, "collect_task_teacher", collect)
    audit = {}
    if qualified:
        episode, returned = collect_mass_teacher(audit)
        assert episode == result.training_episode()
        assert returned["qualified"] is True
    else:
        with pytest.raises(AssertionError, match="did not qualify"):
            collect_mass_teacher(audit)
    assert calls == [dict(mass_context=pick_cube_mass_context())]
    retained = json.loads((tmp_path / "teacher-b.json").read_text())
    assert retained["qualified"] is qualified and retained["work"] == asdict(result.work)
    assert retained["mass_work"] == asdict(result.mass_work)
    assert audit["teacher_attempts"] == audit["teacher_returns"] == 1


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "incomplete",
        "historical_type",
        "policy",
        "reference_policy",
        "reference_frozen",
        "reference_context",
        "unexpected_legacy_metadata",
        "foreign_context_metadata",
        "prototype",
    ],
)
def test_should_bind_saved_source_policy_arrays_and_reference_before_any_model(
    tmp_path, monkeypatch, damage
):
    # Thirty distinct public observations are enough for the actual29-row
    # source-admission helper; no teacher collection or prototype append occurs.
    template = qualified_result(mass=False).candidate_episode
    observations = tuple(tuple(float(index + step) for index in range(42)) for step in range(30))
    warm = replace(
        template,
        transitions=tuple(
            ContinuousTransition(
                i,
                observations[i],
                (-1.0, 0.0, 0.0, 1.0),
                float(i == 28),
                observations[i + 1],
                i == 28,
                False,
            )
            for i in range(29)
        ),
    )
    transform = ObservationStandardization((0.0,) * 42, (1.0,) * 42, 29)
    policy = NearestContinuousBCPolicy(warm.spec, transform)
    raw_policy = json.loads(json.dumps(asdict(policy)))
    source = tmp_path / "saved"
    source.mkdir()
    features = np.asarray([row.observation for row in warm.transitions], dtype=np.float64)
    actions = np.asarray([row.action for row in warm.transitions], dtype=np.float64)
    if damage == "prototype":
        features[0, 0] += 1.0
    archive_path = source / "nearest-prototypes.npz"
    with archive_path.open("xb") as handle:
        np.savez(handle, features=features, actions=actions)
    record = dict(
        complete=True,
        fit_returns=1,
        append_returns=1,
        rows=29,
        state_save_returns=1,
        prototype_numeric_bytes=10672,
        state_file_bytes=archive_path.stat().st_size,
        policy=raw_policy,
    )
    episode = asdict(replace(warm, episode_id="pick-cube-smoke-17"))
    reference = dict(
        complete=True,
        role="train",
        seed=17,
        task="PickCube-v1",
        controller="pd_ee_delta_pos",
        learner_policy=deepcopy(raw_policy),
        source_updates=1,
        source_training_rows=29,
        frozen_state_exact=True,
        original_owner_exact=True,
        episode=episode,
        sparse_return=1,
        any_success=True,
        final_success=True,
        work=dict(
            step_attempts=29,
            completed_steps=29,
            completed_control_physics_steps=145,
            possible_failed_physics_steps=0,
        ),
    )
    if damage == "incomplete":
        record["complete"] = False
    elif damage == "historical_type":
        record["append_returns"] = True
    elif damage == "policy":
        record["policy"]["training_role"] = "probe_train"
    elif damage == "reference_policy":
        reference["learner_policy"]["max_prototypes"] = 255
    elif damage == "reference_frozen":
        reference["frozen_state_exact"] = False
    elif damage == "reference_context":
        reference["episode"]["context"] = "replaced-context"
    elif damage == "unexpected_legacy_metadata":
        reference["observation_recipe"] = None
    elif damage == "foreign_context_metadata":
        reference["mass_context"] = asdict(pick_cube_mass_context())
    write_json(source / "rollout-audit.json", record)
    write_json(source / "policy-rollout.json", reference)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    monkeypatch.setattr(case, "SOURCE", source)
    monkeypatch.setattr(
        case,
        "_read_teacher",
        lambda _path, seed: warm if seed == 17 else pytest.fail("unexpected seed"),
    )
    monkeypatch.setattr(
        NumpyNearestContinuousBCLearner,
        "__init__",
        lambda *_: pytest.fail("source validation constructed learner"),
    )
    audit = {}
    if damage is None:
        returned_warm, state, returned_reference = case.load_sources(audit)
        assert (
            returned_warm == warm and state.completed_updates == 1 and state.training_examples == 29
        )
        np.testing.assert_array_equal(state.prototype_features, features)
        np.testing.assert_array_equal(state.prototype_actions, actions)
        assert returned_reference["sparse_return"] == 1
        assert audit["source_binding_normalization_rows"] == 29
        assert audit["source_binding_normalization_values"] == 1218
    else:
        with pytest.raises((AssertionError, ValueError)):
            case.load_sources(audit)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
