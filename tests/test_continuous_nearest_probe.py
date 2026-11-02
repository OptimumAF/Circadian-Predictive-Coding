"""Tiny literal probe controls; no recordings, cached models or simulator."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.adapters import continuous_probe as probe
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.adapters import numpy_nearest_continuous_bc as nearest
from src.core import continuous_bc as core
from src.core import continuous_pc as pc_core
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.observation_standardization import ObservationStandardization

WORK = dict(
    nearest_inits=0,
    appends=0,
    append_rows=0,
    retrievals=0,
    query_rows=0,
    distance_candidates=0,
    distance_features=0,
    neural_inits=0,
    neural_updates=0,
    neural_update_rows=0,
    core_forwards=0,
    core_forward_rows=0,
    gradients=0,
    gradient_rows=0,
    normal_calls=0,
    normal_values=0,
    pc_evaluations=0,
    pc_evaluation_rows=0,
)
CAPS = dict(
    nearest_inits=200,
    appends=200,
    append_rows=1000,
    retrievals=2000,
    query_rows=10000,
    distance_candidates=2560000,
    distance_features=30720000,
    neural_inits=8,
    neural_updates=8,
    neural_update_rows=8,
    core_forwards=1000,
    core_forward_rows=4000,
    gradients=200,
    gradient_rows=1000,
    normal_calls=16,
    normal_values=64,
    pc_evaluations=200,
    pc_evaluation_rows=1000,
)


def charge(key, amount=1):
    WORK[key] += amount
    assert WORK[key] <= CAPS[key], f"permanent fixture work cap exceeded: {key}"


@pytest.fixture(autouse=True)
def count_work(monkeypatch):
    def wrap(owner, name, counter, rows=None):
        original = getattr(owner, name)

        def call(*args, **kwargs):
            charge(counter)
            if rows is not None:
                charge(rows[0], len(args[rows[1]]))
            if counter == "retrievals":
                candidates = len(args[1]) * args[0]._state.training_examples
                charge("distance_candidates", candidates)
                charge("distance_features", candidates * args[0]._policy.spec.observation_dim)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, call)

    wrap(nearest.NumpyNearestContinuousBCLearner, "__init__", "nearest_inits")
    wrap(nearest.NumpyNearestContinuousBCLearner, "train_batch", "appends", ("append_rows", 1))
    wrap(
        nearest.NumpyNearestContinuousBCLearner,
        "predict_with_work",
        "retrievals",
        ("query_rows", 1),
    )
    wrap(bp.NumpyContinuousBCLearner, "__init__", "neural_inits")
    wrap(bp.NumpyContinuousBCLearner, "train_batch", "neural_updates", ("neural_update_rows", 1))
    wrap(
        pc.NumpyContinuousPCLearner,
        "train_batch_with_work",
        "neural_updates",
        ("neural_update_rows", 1),
    )
    wrap(core, "_forward", "core_forwards", ("core_forward_rows", 0))
    wrap(core, "continuous_bc_loss_and_gradients", "gradients", ("gradient_rows", 0))
    wrap(pc_core, "_evaluate", "pc_evaluations", ("pc_evaluation_rows", 1))
    original_rng = np.random.default_rng

    class Generator:
        def __init__(self, *args, **kwargs):
            self.delegate = original_rng(*args, **kwargs)

        def standard_normal(self, shape):
            charge("normal_calls")
            charge("normal_values", int(np.prod(shape)))
            return self.delegate.standard_normal(shape)

    monkeypatch.setattr(np.random, "default_rng", Generator)


def episode(name, role, position, target, *, context=None):
    spec = ContinuousActionSpec(2, ("m", "m"), (-1.0, -1.0), (1.0, 1.0))
    return ContinuousEpisode(
        name,
        context or name,
        role,
        spec,
        (
            ContinuousTransition(
                0, (position, 0.0), (target, target), 0.0, (position + 1.0, 0.0), True, False
            ),
        ),
    )


def old():
    return episode("old-A", "train", 0.0, -1.0)


def plan(**options):
    return probe.ContinuousProbePlan(
        episode("fresh-train", "probe_train", 10.0, 1.0, context="fresh"),
        episode("fresh-test", "probe_test", 10.1, 0.5, context="fresh"),
        **options,
    )


def source(*, capacity=8, fitted=False):
    policy = nearest.FixedInputNearestContinuousBCPolicy(old().spec, max_prototypes=capacity)
    if fitted:
        policy = nearest.NearestContinuousBCPolicy(
            old().spec, ObservationStandardization((0.0, 0.0), (1.0, 1.0), 1)
        )
    learner = nearest.NumpyNearestContinuousBCLearner(policy)
    learner.train_episode(old())
    return learner


def test_should_acquire_private_packet_retain_three_old_scores_and_account_costs():
    learner = source()
    before = learner.snapshot_state()
    seen = []

    def observe(readout, phase, guard):
        guard()
        seen.append((phase, readout.snapshot_state()))
        guard()

    readouts = (old(), episode("old-B", "train", 0.0, -1.0), episode("old-C", "train", 0.0, -1.0))
    result = probe.run_continuous_probe(
        learner, plan(), training_readouts=readouts, endpoint_observer=observe
    )
    assert result["complete"], result
    assert result["metrics"]["fresh_acquisition_gain"] == 1.0
    assert result["metrics"]["old_retention_loss"] == dict(old_0=0.0, old_1=0.0, old_2=0.0)
    assert probe.same_continuous_state(before, learner.snapshot_state())
    assert seen[0][1].training_examples == 1 and seen[1][1].training_examples == 2
    np.testing.assert_array_equal(seen[1][1].prototype_actions[-1], [1.0, 1.0])
    assert result["events"][0]["deferred_exact"] and result["events"][0]["success_validated"]
    assert result["progress"]["updates_completed"] == 1
    assert result["work"]["optimizer_updates"] == result["work"]["gradient_evaluations"] == 0
    assert result["work"]["prototype_numeric_bytes_per_owner_before"] == 32
    assert result["work"]["candidate_numeric_bytes_known_after_return"] == 64
    assert result["work"]["nearest_retrieval"]["distance_candidates_completed"] == 12
    assert result["work"]["nearest_retrieval"]["distance_feature_values_completed"] == 24
    assert "candidate" not in result


def test_should_work_without_endpoint_callback_and_preserve_its_original_source():
    learner = source()
    before = probe.nearest_probe.owner_identity(learner)
    result = probe.run_continuous_probe(learner, plan(), training_readouts=(old(),))
    assert result["complete"], result
    assert all(
        a is b for a, b in zip(before, probe.nearest_probe.owner_identity(learner), strict=True)
    )
    assert result["work"]["native_returns"] == 1


@pytest.mark.parametrize(
    "options", [dict(updates=2), dict(diagnostics=True), dict(zero_velocity_control=True)]
)
def test_should_refuse_unsupported_options_before_source_observation(options, monkeypatch):
    learner = source()
    monkeypatch.setattr(learner, "snapshot_state", lambda: pytest.fail("source observed"))
    with pytest.raises(ValueError, match="one append"):
        probe.run_continuous_probe(learner, plan(**options))


def test_should_refuse_capacity_before_any_fork(monkeypatch):
    learner = source(capacity=1)
    monkeypatch.setattr(learner, "fork", lambda: pytest.fail("private work admitted"))
    with pytest.raises(ValueError, match="capacity"):
        probe.run_continuous_probe(learner, plan())
    assert learner._state.training_examples == 1


def test_should_refuse_legacy_fitted_nearest_before_probe_work():
    with pytest.raises(ValueError, match="fixed-input"):
        probe.run_continuous_probe(source(fitted=True), plan())


@pytest.mark.parametrize("role", ["train", "probe_train", "final_test"])
def test_should_keep_score_only_and_final_labels_out_of_update_admission(role):
    with pytest.raises(ValueError):
        replace(plan(), evaluation=replace(plan().evaluation, role=role))


def test_should_refuse_four_old_readouts_and_context_overlap():
    learner = source()
    with pytest.raises(ValueError, match="three TRAIN"):
        probe.run_continuous_probe(learner, plan(), training_readouts=(old(),) * 4)
    with pytest.raises(ValueError, match="distinct"):
        probe.run_continuous_probe(
            learner, plan(), training_readouts=(replace(old(), context="fresh"),)
        )


@pytest.mark.parametrize("field", ["state", "policy", "features", "actions"])
def test_should_refuse_equal_source_owner_or_array_replacement(field, monkeypatch):
    learner = source()
    original = learner.snapshot_state

    def replaced():
        state = original()
        if field == "state":
            learner._state = state
        elif field == "policy":
            learner._policy = learner.policy
        else:
            key = "prototype_features" if field == "features" else "prototype_actions"
            object.__setattr__(learner._state, key, getattr(learner._state, key).copy())
        return state

    monkeypatch.setattr(learner, "snapshot_state", replaced)
    result = probe.run_continuous_probe(learner, plan())
    assert not result["complete"] and not result["native_updates"]


def test_should_refuse_replaced_preceding_score_before_append(monkeypatch):
    captured = {}
    original = probe._score

    def score(*args, **kwargs):
        original(*args, **kwargs)
        captured["report"] = args[0]

    def corrupt():
        if captured:
            points = captured["report"]["checkpoints"]
            points[0] = deepcopy(points[0])

    monkeypatch.setattr(probe, "_score", score)
    result = probe.run_continuous_probe(source(), plan(), check_boundary=corrupt)
    assert not result["complete"] and "paid evidence" in result["error"]
    assert result["progress"]["updates_completed"] == 0


def test_should_preserve_paid_append_when_native_acknowledgement_is_lost(monkeypatch):
    learner = source()
    original = nearest.NumpyNearestContinuousBCLearner.train_batch

    def lost(owner, features, targets):
        diagnostic = original(owner, features, targets)
        if owner is not learner:
            raise RuntimeError("lost acknowledgement")
        return diagnostic

    monkeypatch.setattr(nearest.NumpyNearestContinuousBCLearner, "train_batch", lost)
    result = probe.run_continuous_probe(learner, plan())
    assert not result["complete"] and result["progress"]["updates_completed"] == 0
    assert result["work"]["append_attempts_permanently_spent"] == 1
    assert result["work"]["append_slots_remaining"] == 0
    assert result["work"]["unreturned_append_possible"] is True
    assert result["work"]["training_attempts"] == 1 and not result["native_updates"]
    assert learner._state.training_examples == 1


def test_should_preserve_retrieval_attempt_on_invalid_receipt(monkeypatch):
    original = nearest.NumpyNearestContinuousBCLearner.predict_with_work

    def invalid(owner, features):
        return replace(original(owner, features), distance_candidates=True)

    monkeypatch.setattr(nearest.NumpyNearestContinuousBCLearner, "predict_with_work", invalid)
    result = probe.run_continuous_probe(source(), plan())
    assert not result["complete"]
    counts = result["work"]["nearest_retrieval"]
    assert counts["attempts"] == 1 and counts["returns"] == 0
    assert counts["distance_candidates_upper_bound"] == 1


def test_should_retain_committed_append_when_later_runtime_observation_fails(monkeypatch):
    original = probe.ActorShadowRuntime

    def runtime(*args, **kwargs):
        owner = original(*args, **kwargs)
        ready = owner.train_ready

        def lost(*args, **kwargs):
            receipts = ready(*args, **kwargs)
            if receipts:
                raise RuntimeError("postcommit observation lost")
            return receipts

        monkeypatch.setattr(owner, "train_ready", lost)
        return owner

    monkeypatch.setattr(probe, "ActorShadowRuntime", runtime)
    result = probe.run_continuous_probe(source(), plan())
    assert not result["complete"] and result["progress"]["updates_completed"] == 1
    assert result["work"]["native_returns"] == 1
    assert result["work"]["append_slots_remaining"] == 0
    assert not result["work"]["unreturned_append_possible"]


def test_should_refuse_budget_replacement_after_paid_append(monkeypatch):
    captured = {}
    original = probe.ActorShadowRuntime

    def runtime(*args, **kwargs):
        captured["runtime"] = original(*args, **kwargs)
        return captured["runtime"]

    def corrupt():
        if captured and captured["runtime"]._budget.updates_completed:
            budget = captured["runtime"]._budget
            budget.budget = replace(budget.budget, max_training_updates=2)

    monkeypatch.setattr(probe, "ActorShadowRuntime", runtime)
    result = probe.run_continuous_probe(source(), plan(), check_boundary=corrupt)
    assert not result["complete"] and result["progress"]["updates_completed"] == 1
    assert result["work"]["native_returns"] == 1
    assert result["work"]["append_slots_remaining"] == 0


def test_should_refuse_endpoint_owner_replacement_and_preserve_prior_scores():
    def replace_owner(readout, phase, guard):
        if phase == "after_probe":
            readout.restore_state(readout.snapshot_state())

    result = probe.run_continuous_probe(source(), plan(), endpoint_observer=replace_owner)
    assert not result["complete"]
    assert result["progress"]["updates_completed"] == 1
    assert result["work"]["native_returns"] == 1
    assert len(result["checkpoints"]) == 2


@pytest.mark.parametrize("phase", [0, 1])
@pytest.mark.parametrize("damage", ["value", "replacement", "retrieval_counts"])
def test_should_protect_each_returned_score_and_paid_counter_within_both_checkpoints(
    phase, damage, monkeypatch
):
    original = probe._score

    def score(report, readout, state, episodes, boundary, **kwargs):
        def corrupt():
            boundary()
            if len(report["checkpoints"]) == phase + 1:
                point = report["checkpoints"][-1]
                if "fresh" in point["scores"]:
                    if damage == "replacement":
                        point["scores"]["fresh"] = deepcopy(point["scores"]["fresh"])
                    elif damage == "value":
                        point["scores"]["fresh"]["residual"]["overall"]["normalized_half_mse"] = (
                            99.0
                        )
                    else:
                        report["work"]["nearest_retrieval"]["distance_candidates_completed"] = 0

        return original(report, readout, state, episodes, corrupt, **kwargs)

    monkeypatch.setattr(probe, "_score", score)
    result = probe.run_continuous_probe(source(), plan(), training_readouts=(old(),))
    assert not result["complete"]
    assert "metrics" not in result
    assert result["work"]["nearest_retrieval"]["distance_candidates_completed"] > 0
    assert (
        result["checkpoints"][-1]["scores"]["fresh"]["residual"]["overall"]["normalized_half_mse"]
        != 99.0
    )


def test_should_keep_negative_fresh_result_when_new_labels_tie_original_inputs():
    duplicate = replace(plan().training, transitions=old().transitions)
    duplicate = replace(
        duplicate, transitions=(replace(duplicate.transitions[0], action=(1.0, 1.0)),)
    )
    evaluation = replace(plan().evaluation, transitions=duplicate.transitions)
    result = probe.run_continuous_probe(source(), probe.ContinuousProbePlan(duplicate, evaluation))
    assert result["complete"], result
    assert result["metrics"]["fresh_acquisition_gain"] == 0.0
    assert result["work"]["candidate_rows_known_after_return"] == 2


@pytest.mark.parametrize("checkpoint", [0, 1])
def test_should_refuse_nested_equal_replacement_after_completed_checkpoint(checkpoint, monkeypatch):
    exposed = {}
    original = probe._score

    def score(report, *args, **kwargs):
        exposed["report"] = report
        return original(report, *args, **kwargs)

    def corrupt():
        report = exposed.get("report")
        if (
            report is not None
            and report["native_updates"]
            and len(report["checkpoints"]) > checkpoint
        ):
            point = report["checkpoints"][checkpoint]
            if point["complete"]:
                point["scores"]["fresh"] = deepcopy(point["scores"]["fresh"])

    monkeypatch.setattr(probe, "_score", score)
    result = probe.run_continuous_probe(source(), plan(), check_boundary=corrupt)
    assert not result["complete"] and "metrics" not in result
    assert result["work"]["native_returns"] == 1
    assert result["progress"]["updates_completed"] == 1


def test_should_recheck_paid_score_after_late_public_source_getter(monkeypatch):
    learner = source()
    exposed = {}
    original_score = probe._score
    original_snapshot = learner.snapshot_state

    def score(report, *args, **kwargs):
        exposed["report"] = report
        return original_score(report, *args, **kwargs)

    def late():
        state = original_snapshot()
        report = exposed.get("report")
        if report is not None and report["native_updates"]:
            report["checkpoints"][0]["scores"]["fresh"]["residual"]["overall"][
                "normalized_half_mse"
            ] = 99.0
        return state

    monkeypatch.setattr(probe, "_score", score)
    monkeypatch.setattr(learner, "snapshot_state", late)
    result = probe.run_continuous_probe(learner, plan())
    assert not result["complete"] and "metrics" not in result
    assert result["work"]["native_returns"] == 1
    assert result["work"]["nearest_retrieval"]["attempts"] == 1
    assert (
        result["checkpoints"][0]["scores"]["fresh"]["residual"]["overall"]["normalized_half_mse"]
        != 99.0
    )


def test_should_refuse_equal_readout_array_replacement(monkeypatch):
    learner = source()
    original = nearest.NumpyNearestContinuousBCLearner.predict_with_work

    def replaced(owner, features):
        result = original(owner, features)
        object.__setattr__(owner._state, "prototype_actions", owner._state.prototype_actions.copy())
        return result

    monkeypatch.setattr(nearest.NumpyNearestContinuousBCLearner, "predict_with_work", replaced)
    result = probe.run_continuous_probe(learner, plan())
    assert not result["complete"] and "replaced original readout" in result["error"]


@pytest.mark.parametrize("family", ["bp", "pc", "neutral-cpc", "active-cpc"])
def test_should_keep_existing_four_neural_families_on_original_shared_path(family):
    base = bp.ContinuousBCPolicy(old().spec, hidden_dim=2)
    if family == "bp":
        learner = bp.NumpyContinuousBCLearner(base)
    else:
        chemistry = (
            None
            if family == "pc"
            else CoupledCircadianConfig.matched_pc_control()
            if family == "neutral-cpc"
            else CoupledCircadianConfig()
        )
        learner = pc.NumpyContinuousPCLearner(pc.ContinuousPCPolicy(base, circadian=chemistry))
    before = learner.snapshot_state()
    result = probe.run_continuous_probe(learner, plan(), training_readouts=(old(),))
    assert result["complete"], result
    assert result["definition"] == "continuous_disposable_probe_v1"
    assert probe.same_continuous_state(before, learner.snapshot_state())
    assert "nearest_retrieval" not in result["work"]
