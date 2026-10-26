"""Disposable continuous probes preserve the benchmark and evaluation boundary.

Synthetic phase-one fixtures exercise existing learner/runtime composition.
Negative gains are valid; these tests assert ownership and accounting, not wins.

Full module: 51 cases; 21 parameter initializations, 38 PC/BP constructor entries,
5376 seeded normal values. There are 40 update attempts (39 published), exposing
82 published plus two failed admitted rows. Benchmark conditioning/residual/actor
readouts each use 72 calls/180 rows; probe predictions have at most 166 attempts/
387 rows (one failed two-row prediction). PC energy entries and latent proposals
are each bounded by 160. No extra training oracle, RNG, native or scientific run.
"""

from copy import copy
from dataclasses import asdict, replace
from typing import Any, cast
import numpy as np
import pytest

from src.adapters import continuous_learnability_demo as demo
from src.adapters import continuous_probe as probe
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
    continuous_training_pairs,
)


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")
CPC_WORK = {
    "chemistry_attempts",
    "chemistry_returns",
    "gate_applications_attempted",
    "gate_applications_returned",
}


def episode(name="A", *, role=None, rows=None, declared=None):
    geometry = (
        ContinuousActionSpec(2, ("m", "m"), (-2.0, -1.0), (6.0, 1.0))
        if declared is None
        else declared
    )
    if name == "A":
        positions, actions, context, default_role = (
            [0.0, 2.0, 4.0],
            [(-2.0, -1.0), (6.0, 1.0)],
            "context-A",
            "train",
        )
    elif name == "B":
        positions, actions, context, default_role = (
            [10.0, 11.0, 12.0, 13.0],
            [(5.0, 1.0), (4.0, 0.5), (3.0, 0.0)],
            "context-B",
            "train",
        )
    elif name == "Ftrain":
        positions, actions, context, default_role = (
            [-1.0, 1.0, 2.0],
            [(2.0, 0.5), (1.0, -0.5)],
            "fresh-F",
            "probe_train",
        )
    else:
        positions, actions, context, default_role = (
            [-0.5, 0.5, 1.5],
            [(1.5, 0.25), (1.25, -0.25)],
            "fresh-F",
            "probe_test",
        )
    if rows is not None:
        positions, actions = [float(index) for index in range(rows + 1)], [actions[0]] * rows
    observations = [
        (value, 1.0 if name == "B" else 0.0) + (0.0,) * (geometry.observation_dim - 2)
        for value in positions
    ]
    return ContinuousEpisode(
        "synthetic-" + name,
        context,
        default_role if role is None else role,
        geometry,
        tuple(
            ContinuousTransition(
                index,
                observations[index],
                action,
                0.0,
                observations[index + 1],
                index == len(actions) - 1,
                False,
            )
            for index, action in enumerate(actions)
        ),
    )


def plan(**options):
    return probe.ContinuousProbePlan(episode("Ftrain"), episode("Ftest"), **options)


def assert_state(first, second, *, common=False):
    assert type(first) is type(second)
    if common:
        assert replace(first.policy, circadian=None) == replace(second.policy, circadian=None)
    else:
        assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)
    if common:
        return
    a, b = getattr(first, "wake_state", None), getattr(second, "wake_state", None)
    if a is None:
        assert b is None
    else:
        assert b is not None
        assert a.wake_updates == b.wake_updates
        for field in (
            "chemical_decay",
            "chemical_buildup_rate",
            "plasticity_sensitivity",
            "min_plasticity",
        ):
            assert getattr(a, field) == getattr(b, field)
        np.testing.assert_array_equal(a.chemical_state, b.chemical_state)
        np.testing.assert_array_equal(a.plasticity_state, b.plasticity_state)


@pytest.mark.parametrize("updates", [0, 17, True, 1.0, None, "1"])
def test_should_refuse_invalid_probe_budgets_before_any_model(updates):
    with pytest.raises(ValueError):
        plan(updates=updates)


@pytest.mark.parametrize(
    "damage",
    [
        "train_role",
        "eval_role",
        "final_train",
        "final_eval",
        "context",
        "identity",
        "spec",
        "rows",
        "values",
    ],
)
def test_should_reject_invalid_probe_authority_and_bounded_payloads(damage):
    training, evaluation = episode("Ftrain"), episode("Ftest")
    if damage in ("train_role", "final_train"):
        training = replace(training, role="train" if damage == "train_role" else "final_test")
    elif damage in ("eval_role", "final_eval"):
        evaluation = replace(
            evaluation, role="probe_train" if damage == "eval_role" else "final_test"
        )
    elif damage == "context":
        evaluation = replace(evaluation, context="other-fresh-context")
    elif damage == "identity":
        evaluation = replace(evaluation, episode_id=training.episode_id)
    elif damage == "spec":
        evaluation = replace(evaluation, spec=replace(evaluation.spec, action_units=("other", "m")))
    elif damage == "rows":
        training = episode("Ftrain", rows=129)
    else:
        wide = replace(training.spec, observation_dim=1024)
        training, evaluation = (
            episode("Ftrain", rows=65, declared=wide),
            episode("Ftest", rows=1, declared=wide),
        )
    with pytest.raises(ValueError):
        probe.ContinuousProbePlan(training, evaluation)


def test_should_own_both_plan_episodes_and_keep_probe_test_labels_unreleased():
    training, evaluation = episode("Ftrain"), episode("Ftest")
    original = asdict(training), asdict(evaluation)
    owned = probe.ContinuousProbePlan(training, evaluation)
    assert owned.training is not training and owned.evaluation is not evaluation
    object.__setattr__(training, "context", "replaced")
    object.__setattr__(evaluation.transitions[0], "action", (6.0, 1.0))
    assert (asdict(owned.training), asdict(owned.evaluation)) == original
    with pytest.raises(ValueError):
        continuous_training_pairs(owned.evaluation, role="probe_train")
    boundary = probe.ContinuousProbePlan(
        episode("Ftrain", rows=128), episode("Ftest", rows=128), updates=16
    )
    assert boundary.updates == 16 and len(boundary.training.transitions) == 128


@pytest.mark.parametrize(
    "damage",
    ["training_id", "evaluation_id", "context_A", "context_B", "spec", "corrupt_role", "type"],
)
def test_should_revalidate_disjoint_probe_inputs_before_benchmark_fit(monkeypatch, damage):
    value = plan()
    if damage in ("training_id", "evaluation_id"):
        field = "training" if damage == "training_id" else "evaluation"
        object.__setattr__(value, field, replace(getattr(value, field), episode_id="synthetic-A"))
    elif damage.startswith("context"):
        context = "context-A" if damage == "context_A" else "context-B"
        object.__setattr__(value, "training", replace(value.training, context=context))
        object.__setattr__(value, "evaluation", replace(value.evaluation, context=context))
    elif damage == "spec":
        declared = replace(value.training.spec, action_units=("other", "m"))
        object.__setattr__(value, "training", replace(value.training, spec=declared))
        object.__setattr__(value, "evaluation", replace(value.evaluation, spec=declared))
    elif damage == "corrupt_role":
        object.__setattr__(value.training, "role", "final_test")
    else:
        value = object()
    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_: pytest.fail("invalid probe plan fitted benchmark data"),
    )
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(episode(), episode("B"), phase_updates=1, probe_plan=value)


def test_should_omit_default_probe_argument_and_metadata_in_existing_json_wrapper(monkeypatch):
    calls = []
    report = {"complete": True}

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return demo.ContinuousLearnabilityResult(report, None)

    monkeypatch.setattr(demo, "run_continuous_learnability", run)
    assert demo.run_continuous_learnability_demo("A", "B", probe_plan=None) is report
    assert calls == [(("A", "B"), dict(expected_standardization=None))]
    calls.clear()
    value = plan()
    assert demo.run_continuous_learnability_demo("A", "B", probe_plan=value) is report
    assert calls == [(("A", "B"), dict(expected_standardization=None, probe_plan=value))]


@pytest.fixture(scope="module")
def four_arms():
    warm, new, value = episode(), episode("B"), plan()
    originals = asdict(warm), asdict(new), asdict(value)
    results = {}
    probes: dict[str, list[Any]] = {}
    benchmark: dict[str, Any] = {}
    trains: dict[str, list[Any]] = {}
    fits = []
    original_runtime = demo.ActorShadowRuntime
    original_fit = demo.fit_training_standardization
    original_bp, original_pc = (
        bp.NumpyContinuousBCLearner.train_batch,
        pc.NumpyContinuousPCLearner.train_batch_with_work,
    )
    with pytest.MonkeyPatch.context() as patch:

        def fit(record, **options):
            fits.append((record.episode_id, options))
            return original_fit(record, **options)

        def runtime(source, **options):
            owner = original_runtime(source, **options)
            benchmark[family] = owner
            return owner

        def disposable(source, **options):
            initial = source.snapshot_state()
            assert_state(initial, benchmark[family].candidate_snapshot().state)
            assert options["training_role"] == "probe_train"
            assert (
                options["max_experiences"]
                == options["budget"].budget.max_training_updates
                == value.updates
            )
            assert options["budget"].updates_completed == 0
            owner = original_runtime(source, **options)
            probes[family].append((owner, initial, options["budget"], options["clock"]))
            return owner

        def bp_train(owner, features, targets):
            trains[family].append((features.copy(), targets.copy()))
            return original_bp(owner, features, targets)

        def pc_train(owner, features, targets):
            trains[family].append((features.copy(), targets.copy()))
            return original_pc(owner, features, targets)

        patch.setattr(demo, "fit_training_standardization", fit)
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(probe, "ActorShadowRuntime", disposable)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", bp_train)
        patch.setattr(pc.NumpyContinuousPCLearner, "train_batch_with_work", pc_train)
        for family in FAMILIES:
            probes[family], trains[family] = [], []
            options: dict[str, Any] = {} if family == "bp" else dict(learner_family=family)
            results[family] = demo.run_continuous_learnability(
                warm, new, phase_updates=1, probe_plan=value, **options
            )
            assert (asdict(warm), asdict(new), asdict(value)) == originals
    return results, probes, benchmark, trains, fits, value


@pytest.mark.parametrize("family", FAMILIES)
def test_should_keep_probe_scopes_independent_with_local_one_update_and_global_continuation(
    four_arms, family
):
    results, scopes, benchmarks, trains, fits, value = four_arms
    result, rows = results[family], scopes[family]
    assert result.report["complete"] is True and result.candidate is not None
    assert fits == [("synthetic-A", {})] * 4
    assert result.report["settings"]["probe_reads"] == 8
    assert [item["benchmark_updates"] for item in result.report["probes"]] == [0, 1, 2, 3]
    assert len(rows) == 4
    for checkpoint, (runtime, initial, budget, clock) in enumerate(rows):
        assert initial.completed_updates == checkpoint
        assert initial.training_examples == [0, 2, 5, 7][checkpoint]
        final = runtime.candidate_snapshot().state
        assert final.completed_updates == checkpoint + 1
        assert final.training_examples == initial.training_examples + 2
        assert budget.updates_completed == 1 and budget.replay_examples_completed == 0
        assert_state(initial, runtime.actor.snapshot_state().state)
        if family in ("neutral-cpc", "active-cpc"):
            assert final.wake_state.wake_updates == final.completed_updates
        assert len(runtime.applied_updates) == 1
        receipt = runtime.applied_updates[0]
        assert receipt.episode_id == value.training.episode_id
        assert receipt.observed_at < receipt.arrived_at == receipt.applied_at
    final_benchmark = benchmarks[family].candidate_snapshot().state
    assert final_benchmark.completed_updates == 3 and final_benchmark.training_examples == 7
    assert_state(final_benchmark, result.candidate.snapshot_state())
    expected = ["Ftrain", "A", "Ftrain", "B", "Ftrain", "A", "Ftrain"]
    assert len(trains[family]) == 7
    for (features, targets), name in zip(trains[family], expected, strict=True):
        record = episode(name)
        np.testing.assert_array_equal(features, [row.observation for row in record.transitions])
        np.testing.assert_array_equal(targets, [row.action for row in record.transitions])
        for row in episode("Ftest").transitions:
            assert not any(np.array_equal(target, row.action) for target in targets)


def test_should_match_neutral_common_benchmark_and_probe_parameters_without_carrying_probe_updates(
    four_arms,
):
    results, scopes, _, _, _, _ = four_arms
    assert results["pc"].report["metrics"] == results["neutral-cpc"].report["metrics"]
    assert results["pc"].report["checkpoints"] == results["neutral-cpc"].report["checkpoints"]
    assert_state(
        results["pc"].candidate.snapshot_state(),
        results["neutral-cpc"].candidate.snapshot_state(),
        common=True,
    )
    for (first, pc_initial, _, _), (second, neutral_initial, _, _) in zip(
        scopes["pc"], scopes["neutral-cpc"], strict=True
    ):
        assert_state(pc_initial, neutral_initial, common=True)
        assert_state(
            first.candidate_snapshot().state, second.candidate_snapshot().state, common=True
        )
        np.testing.assert_array_equal(
            second.candidate_snapshot().state.wake_state.plasticity_state, np.ones(64)
        )
        assert np.any(second.candidate_snapshot().state.wake_state.chemical_state > 0)


@pytest.mark.parametrize("family", FAMILIES)
def test_should_account_probe_only_scores_and_exposure_separately_with_no_favorable_gain_gate(
    four_arms, family
):
    result = four_arms[0][family]
    assert result.report["settings"]["probe_plan"] == dict(
        invocations=4, updates_each=1, total_training_update_budget=4
    )
    for index, entry in enumerate(result.report["probes"]):
        report = entry["report"]
        assert report["complete"] is True
        assert report["source_updates"] == index
        assert report["source_training_examples"] == [0, 2, 5, 7][index]
        assert report["progress"] == dict(
            updates_completed=1, replay_examples_completed=0, next_tick=2
        )
        assert [point["updates"] for point in report["checkpoints"]] == [index, index + 1]
        assert all(point["complete"] for point in report["checkpoints"])
        assert len(report["events"]) == 1
        event = report["events"][0]
        assert {
            key: event[key] for key in ("index", "deferred_exact", "success_validated")
        } == dict(index=0, deferred_exact=True, success_validated=True)
        assert event["returned_receipts"] == report["applied_updates"]
        first, final = [point["scores"] for point in report["checkpoints"]]
        assert first["fresh"]["role"] == final["fresh"]["role"] == "probe_test"
        assert first["fresh"]["episode_id"] == "synthetic-Ftest"
        assert first["old_0"]["role"] == first["old_1"]["role"] == "train"
        loss = lambda point, name: point[name]["residual"]["overall"]["normalized_half_mse"]
        assert report["metrics"] == dict(
            fresh_acquisition_gain=loss(first, "fresh") - loss(final, "fresh"),
            old_retention_loss={
                name: loss(final, name) - loss(first, name) for name in ("old_0", "old_1")
            },
        )
        work = report["work"]
        assert work["prediction_attempts"] == work["prediction_returns"] == 6
        assert work["prediction_rows_attempted"] == work["prediction_rows_returned"] == 14
        assert work["training_attempts"] == 1
        assert work["training_rows_attempted"] == work["training_rows_completed"] == 2
        assert work["reserved_numeric_bytes"] == 128 and work["extra_model_owners"] == 3
        assert work["parameter_momentum_numeric_bytes_per_owner"] == 5152
        assert work["wake_numeric_bytes_per_owner"] == (1024 if family.endswith("cpc") else 0)
        assert len(report["applied_updates"]) == 1
        assert len(report["native_updates"]) == 1
        assert work["native_returns"] == 1 and work["training_rows_native_returned"] == 2
        assert work["score_rows_returned_by_role"] == {"probe_test": 4, "train": 10}
        assert work["old_readout_input_numeric_bytes"] == 160
        assert work["prediction_numeric_bytes_returned"] == 224
        assert work["derived_returned_prediction_dense_multiply_accumulates"] == 3584
        if family == "bp":
            assert report["native_updates"][0]["work"] is None
            assert work["native_update_counts"] == {}
            assert work["derived_bp_native_forward_backward_returns"] == 1
            assert work["derived_bp_training_dense_multiply_accumulates"] == 1280
        else:
            native = report["native_updates"][0]["work"]
            assert work["native_update_counts"] == native
            assert work["native_event_row_weights"] == {
                key: value * 2 for key, value in native.items()
            }
            assert work["derived_bp_native_forward_backward_returns"] == 0
            assert work["derived_bp_training_dense_multiply_accumulates"] == 0
        assert "candidate" not in report and "learner" not in report


def test_should_match_pc_neutral_probe_scores_receipts_and_common_native_work(four_arms):
    results = four_arms[0]
    for pc_entry, neutral_entry in zip(
        results["pc"].report["probes"], results["neutral-cpc"].report["probes"], strict=True
    ):
        first, second = pc_entry["report"], neutral_entry["report"]
        assert first["checkpoints"] == second["checkpoints"]
        assert first["metrics"] == second["metrics"]
        assert first["applied_updates"] == second["applied_updates"]
        pc_update, neutral_update = first["native_updates"][0], second["native_updates"][0]
        for key in ("diagnostic", "task_loss", "inference", "training_rows"):
            assert pc_update[key] == neutral_update[key]
        assert pc_update["work"] == {
            key: value for key, value in neutral_update["work"].items() if key not in CPC_WORK
        }
        assert {key: neutral_update["work"][key] for key in CPC_WORK} == {
            key: 1 for key in CPC_WORK
        }


def partial(report):
    assert report["complete"] is False
    assert report["progress"]["updates_completed"] == 0
    assert report["applied_updates"] == []
    assert len(report["probes"]) == 1
    assert "metrics" not in report and "work" not in report
    return report["probes"][0]["report"]


def test_should_preserve_failed_native_probe_work_without_advancing_the_benchmark(monkeypatch):
    monkeypatch.setattr(
        pc.energy,
        "evaluate_continuous_pc",
        lambda *_: (_ for _ in ()).throw(FloatingPointError("probe initial energy failed")),
    )
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, learner_family="pc", probe_plan=plan()
    )
    assert result.candidate is None
    report = partial(result.report)
    assert report["complete"] is False
    assert report["progress"]["updates_completed"] == 0
    assert report["native_updates"] == report["applied_updates"] == []
    assert report["failed_native_work"]["energy_evaluations"] == 1
    assert report["failed_native_work"]["published_updates"] == 0
    assert report["work"]["training_attempts"] == 1


def test_should_retain_native_return_and_local_charge_after_probe_postcommit_failure(monkeypatch):
    original = probe.ActorShadowRuntime

    def runtime(*args, **options):
        owner = original(*args, **options)
        ready = owner.train_ready

        def late(*args, **kwargs):
            result = ready(*args, **kwargs)
            if result:
                raise RuntimeError("probe postcommit callback failed")
            return result

        monkeypatch.setattr(owner, "train_ready", late)
        return owner

    monkeypatch.setattr(probe, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, learner_family="pc", probe_plan=plan()
    )
    assert result.candidate is None
    report = partial(result.report)
    assert "postcommit callback" in report["error"]
    assert report["progress"]["updates_completed"] == 1
    assert len(report["applied_updates"]) == len(report["native_updates"]) == 1
    assert report["native_updates"][0]["work"]["published_updates"] == 1
    assert report["work"]["training_rows_completed"] == 2


def test_should_charge_attempted_failed_probe_evaluation_without_training_or_partial_success(
    monkeypatch,
):
    original = pc.NumpyContinuousPCLearner.predict
    calls = []

    def fail_fresh(owner, features):
        if features[0, 0] == -0.5:
            calls.append(True)
            raise FloatingPointError("fresh evaluation prediction failed")
        return original(owner, features)

    monkeypatch.setattr(pc.NumpyContinuousPCLearner, "predict", fail_fresh)
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, learner_family="pc", probe_plan=plan()
    )
    report = partial(result.report)
    assert calls == [True]
    assert report["work"]["prediction_attempts"] == 1
    assert report["work"]["prediction_returns"] == 0
    assert report["work"]["prediction_rows_attempted"] == 2
    assert report["work"]["training_attempts"] == 0
    assert report["checkpoints"][0]["complete"] is False
    assert report["checkpoints"][0]["scores"] == {}


@pytest.mark.parametrize("damage", ["limits", "timer", "started", "logical", "budget_alias"])
def test_should_refuse_changes_to_original_benchmark_budget_and_clock_before_more_work(
    monkeypatch, damage
):
    original_runtime, original_score = demo.ActorShadowRuntime, probe._score
    parents = []

    def runtime(*args, **options):
        owner = original_runtime(*args, **options)
        parents.append((owner, options["budget"], options["clock"]))
        return owner

    def score(*args, **kwargs):
        result = original_score(*args, **kwargs)
        owner, budget, clock = parents[0]
        if damage == "limits":
            budget.budget = replace(budget.budget)
        elif damage == "timer":
            budget.clock = lambda: 0.0
        elif damage == "started":
            budget.started_at += 1.0
        elif damage == "logical":
            clock.advance_to(1)
        else:
            owner._budget = copy(budget)
        return result

    monkeypatch.setattr(demo, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(probe, "_score", score)
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, learner_family="pc", probe_plan=plan()
    )
    assert result.candidate is None
    report = partial(result.report)
    assert "budget/clock" in result.report["error"]
    assert report["complete"] is False and report["work"]["training_attempts"] == 0
    assert report["work"]["prediction_returns"] == 3
    assert report["progress"]["updates_completed"] == 0


def changed_wake(state):
    wake, config = state.wake_state, state.policy.circadian
    chemistry = wake.chemical_state + 0.01
    gates = np.clip(np.exp(-config.plasticity_sensitivity * chemistry), config.min_plasticity, 1.0)
    return replace(
        state, wake_state=replace(wake, chemical_state=chemistry, plasticity_state=gates)
    )


@pytest.mark.parametrize("target", ["source", "benchmark", "probe_actor"])
def test_should_refuse_coherent_replaced_wake_after_valid_prior_success_before_new_probe_work(
    monkeypatch, target
):
    original_parent, original_probe, original_score = (
        demo.ActorShadowRuntime,
        probe.ActorShadowRuntime,
        probe._score,
    )
    parents, scopes = [], []
    injected: list[bool] = []

    def parent(source, **options):
        owner = original_parent(source, **options)
        parents.append(owner)
        return owner

    def disposable(source, **options):
        owner = original_probe(source, **options)
        scopes.append((owner, source))
        return owner

    def score(report, readout, state, episodes, boundary):
        result = original_score(report, readout, state, episodes, boundary)
        if report["source_updates"] == 1 and not injected:
            local, source = scopes[-1]
            owner = (
                source
                if target == "source"
                else parents[0]._candidate
                if target == "benchmark"
                else local.actor._learner
            )
            owner.restore_state(changed_wake(owner.snapshot_state()))
            injected.append(True)
        return result

    monkeypatch.setattr(demo, "ActorShadowRuntime", parent)
    monkeypatch.setattr(probe, "ActorShadowRuntime", disposable)
    monkeypatch.setattr(probe, "_score", score)
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, learner_family="active-cpc", probe_plan=plan()
    )
    report = result.report
    assert injected == [True] and result.candidate is None and report["complete"] is False
    assert report["progress"]["updates_completed"] == 1
    assert len(report["applied_updates"]) == len(report["native_updates"]) == 1
    assert len(report["probes"]) == 2 and report["probes"][0]["report"]["complete"] is True
    failed = report["probes"][1]["report"]
    assert failed["complete"] is False and failed["work"]["training_attempts"] == 0
    assert failed["progress"]["updates_completed"] == 0
    assert "metrics" not in report and "work" not in report


@pytest.mark.parametrize("damage", ["budget", "clock"])
def test_should_refuse_replaced_local_probe_runtime_bindings_without_training(monkeypatch, damage):
    original = probe.ActorShadowRuntime

    def runtime(*args, **options):
        owner = original(*args, **options)
        if damage == "budget":
            owner._budget = copy(options["budget"])
        else:
            setattr(owner._inbox, "_clock", object())
        return owner

    monkeypatch.setattr(probe, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, learner_family="pc", probe_plan=plan()
    )
    assert result.candidate is None
    report = partial(result.report)
    assert report["complete"] is False
    assert report["work"]["training_attempts"] == 0
    assert report["progress"]["updates_completed"] == 0


def test_should_retain_bp_receipt_and_charge_when_postcommit_candidate_snapshot_fails(monkeypatch):
    original = probe.ActorShadowRuntime

    def runtime(*args, **options):
        owner = original(*args, **options)
        snapshot = owner.candidate_snapshot

        def fail_after_commit():
            if options["budget"].updates_completed:
                raise RuntimeError("postcommit candidate snapshot failed")
            return snapshot()

        monkeypatch.setattr(owner, "candidate_snapshot", fail_after_commit)
        return owner

    monkeypatch.setattr(probe, "ActorShadowRuntime", runtime)
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, probe_plan=plan()
    )
    assert result.candidate is None
    report = partial(result.report)
    assert report["complete"] is False
    assert "postcommit candidate snapshot" in report["error"]
    assert report["progress"]["updates_completed"] == 1
    assert len(report["applied_updates"]) == 1
    assert report["work"]["training_rows_completed"] == 2
    assert len(report["native_updates"]) == 1
    assert report["native_updates"][0]["work"] is None


@pytest.mark.parametrize("damage", ["throw", "replace"])
def test_should_keep_observed_bp_return_and_receipt_when_late_receipt_getter_fails(
    monkeypatch, damage
):
    original_runtime = probe.ActorShadowRuntime
    original_get = cast(Any, original_runtime.applied_updates).fget
    scopes, paid_reads = [], []

    def runtime(*args, **options):
        owner = original_runtime(*args, **options)
        scopes.append(owner)
        return owner

    def fail_late(owner):
        result = original_get(owner)
        if any(owner is scope for scope in scopes) and result:
            paid_reads.append(True)
            if len(paid_reads) > 1:
                if damage == "throw":
                    raise RuntimeError("late retained receipt getter failed")
                return (replace(result[0], event_id="replaced-receipt"),)
        return result

    monkeypatch.setattr(probe, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(original_runtime, "applied_updates", property(fail_late))
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, probe_plan=plan()
    )
    assert result.candidate is None
    report = partial(result.report)
    assert report["complete"] is False
    assert report["receipt_observation_complete"] is (damage != "throw")
    if damage == "throw":
        assert "late retained receipt" in report["receipt_read_error"]
    else:
        assert "receipt sequence changed" in report["error"]
        assert report["events"][0]["returned_receipts"][0]["event_id"] == "probe-label-0"
    assert report["progress"]["updates_completed"] == 1
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 1
    assert report["work"]["training_rows_native_returned"] == 2
    assert report["work"]["training_rows_completed"] == (None if damage == "throw" else 2)
    assert "metrics" not in report


def test_should_refuse_source_change_from_actor_observation_after_final_receipt_read(monkeypatch):
    original_runtime = probe.ActorShadowRuntime
    original_get = cast(Any, original_runtime.applied_updates).fget
    scopes = []
    paid_reads: list[bool] = []
    changed: list[bool] = []

    def runtime(source, **options):
        owner = original_runtime(source, **options)
        scopes.append(owner)
        snapshot = owner.actor.snapshot_state

        def observe():
            result = snapshot()
            if len(paid_reads) >= 2 and not changed:
                state = source.snapshot_state()
                state.parameters[0][0, 0] += 0.01
                source.restore_state(state)
                changed.append(True)
            return result

        monkeypatch.setattr(owner.actor, "snapshot_state", observe)
        return owner

    def retained(owner):
        result = original_get(owner)
        if any(owner is scope for scope in scopes) and result:
            paid_reads.append(True)
        return result

    monkeypatch.setattr(probe, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(original_runtime, "applied_updates", property(retained))
    result = demo.run_continuous_learnability(
        episode(), episode("B"), phase_updates=1, learner_family="pc", probe_plan=plan()
    )
    report = partial(result.report)
    assert changed == [True]
    assert result.candidate is None and report["complete"] is False
    assert "checkpoint source" in report["error"]
    assert report["progress"]["updates_completed"] == 1
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 1
    assert "metrics" not in report
