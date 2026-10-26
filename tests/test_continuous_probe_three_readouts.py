"""Three owned TRAIN readouts on one disposable probe, without training escape.

Four cached H2/D2/A2 family sources consume 32 seeded normal values altogether.
The full module plans ten SGD returns/20 rows and 78 native prediction calls/
158 rows, including predictions returned immediately before injected failures.
One read-only diagnostic gradient returns12 values/96 bytes on two rows.
PC inference uses only the three successful family probes (at most15 energy
evaluations/12 latent proposals). No benchmark fit, head solve, SDK or teacher.
"""

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from src.adapters import continuous_probe as probe
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from test_continuous_probe import assert_state, episode, plan


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")


def readouts():
    third = episode("B", rows=1)
    row = replace(
        third.transitions[0],
        observation=(30.0, 1.0),
        next_observation=(31.0, 1.0),
        action=(2.5, -0.75),
    )
    third = replace(third, episode_id="synthetic-C", context="context-C", transitions=(row,))
    return episode(), episode("B"), third


def forbidden(*args, **kwargs):
    pytest.fail("invalid readout admitted a fork, update, fit or head solve")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        SGD_attempts=0,
        SGD_returns=0,
        training_rows_attempted=0,
        training_rows_returned=0,
        prediction_attempts=0,
        prediction_returns=0,
        prediction_rows_attempted=0,
        prediction_rows_returned=0,
        PC_energy_attempts=0,
        PC_energy_returns=0,
        PC_latent_proposals=0,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    trains = (
        bp.NumpyContinuousBCLearner.train_batch,
        pc.NumpyContinuousPCLearner.train_batch_with_work,
    )
    predicts = (bp.NumpyContinuousBCLearner.predict, pc.NumpyContinuousPCLearner.predict)
    evaluate = pc.energy.evaluate_continuous_pc
    infer = pc._run_inference

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (
            policy.spec.observation_dim + len(policy.spec.action_units)
        )
        return initialize(owner, policy)

    def trained(original):
        def train(owner, features, targets):
            counts["SGD_attempts"] += 1
            counts["training_rows_attempted"] += len(features)
            returned = original(owner, features, targets)
            counts["SGD_returns"] += 1
            counts["training_rows_returned"] += len(features)
            return returned

        return train

    def predicted(original):
        def predict(owner, features):
            counts["prediction_attempts"] += 1
            counts["prediction_rows_attempted"] += len(features)
            returned = original(owner, features)
            counts["prediction_returns"] += 1
            counts["prediction_rows_returned"] += len(returned)
            return returned

        return predict

    def evaluated(*args, **options):
        counts["PC_energy_attempts"] += 1
        returned = evaluate(*args, **options)
        counts["PC_energy_returns"] += 1
        return returned

    def inferred(*args, **options):
        work = options["_work"]
        previous = work.attempted_steps
        try:
            return infer(*args, **options)
        finally:
            counts["PC_latent_proposals"] += work.attempted_steps - previous

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", trained(trains[0]))
        patch.setattr(pc.NumpyContinuousPCLearner, "train_batch_with_work", trained(trains[1]))
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", predicted(predicts[0]))
        patch.setattr(pc.NumpyContinuousPCLearner, "predict", predicted(predicts[1]))
        patch.setattr(pc.energy, "evaluate_continuous_pc", evaluated)
        patch.setattr(pc, "_run_inference", inferred)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("P3_LEDGER"):
        with Path(os.environ["P3_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["initializations"] == 4 and counts["normal_draws"] == 32
    assert counts["SGD_attempts"] <= 24 and counts["training_rows_attempted"] <= 72
    assert counts["prediction_attempts"] <= 256 and counts["prediction_rows_attempted"] <= 1024
    assert counts["PC_energy_attempts"] <= 512 and counts["PC_latent_proposals"] <= 512


@pytest.fixture(scope="module")
def owners(ledger):
    result: dict[str, Any] = {}
    for family in FAMILIES:
        base = bp.ContinuousBCPolicy(episode().spec, hidden_dim=2, seed=17)
        chemistry = (
            None
            if family == "pc"
            else CoupledCircadianConfig.matched_pc_control()
            if family == "neutral-cpc"
            else CoupledCircadianConfig()
        )
        result[family] = (
            bp.NumpyContinuousBCLearner(base)
            if family == "bp"
            else pc.NumpyContinuousPCLearner(pc.ContinuousPCPolicy(base, circadian=chemistry))
        )
    return result


@pytest.fixture(scope="module")
def successful(owners):
    outcomes = {}
    for family in FAMILIES:
        source = owners[family].fork()
        before = source.snapshot_state()
        old, value = readouts(), plan()
        input_seal = tuple(asdict(item) for item in old), asdict(value)
        captured: dict[str, Any] = {}
        predictions = []
        factory = probe.ActorShadowRuntime
        owner_type = type(source)
        predict = owner_type.predict

        def capture_runtime(*args, **options):
            made = factory(*args, **options)
            captured.update(runtime=made, budget=options["budget"])
            return made

        def capture_prediction(owner, features):
            returned = predict(owner, features)
            predictions.append((features.tolist(), returned.tolist()))
            return returned

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(probe, "ActorShadowRuntime", capture_runtime)
            patch.setattr(owner_type, "predict", capture_prediction)
            report = probe.run_continuous_probe(source, value, training_readouts=old)
        outcomes[family] = report, source, before, captured, old, value, input_seal, predictions
    return outcomes


@pytest.mark.parametrize("family", FAMILIES)
def test_should_score_all_three_owned_train_readouts_before_and_after_private_update(
    successful, family
):
    report, source, before, captured, old, value, input_seal, predictions = successful[family]
    assert report["complete"] is True, report.get("error")
    assert tuple(asdict(item) for item in old) == input_seal[0] and asdict(value) == input_seal[1]
    assert_state(source.snapshot_state(), before)
    assert_state(captured["runtime"].actor.snapshot_state().state, before)
    private = captured["runtime"].candidate_snapshot().state
    assert private.policy == before.policy
    assert (private.completed_updates, private.training_examples) == (1, 2)
    assert captured["budget"].updates_completed == 1
    assert len(predictions) == 8
    episodes = (value.evaluation,) + old
    for phase, point in enumerate(report["checkpoints"]):
        assert point["complete"] is True and point["updates"] == phase
        assert list(point["scores"]) == ["fresh", "old_0", "old_1", "old_2"]
        for index, record in enumerate(episodes):
            features, predicted = predictions[4 * phase + index]
            assert features == [list(row.observation) for row in record.transitions]
            # Independent original objective: half mean squared normalized errors.
            errors = [
                ((prediction[axis] - row.action[axis]) / scale) ** 2
                for prediction, row in zip(predicted, record.transitions, strict=True)
                for axis, scale in enumerate((4.0, 1.0))
            ]
            expected = math.fsum(errors) / (2 * len(errors))
            name = ("fresh", "old_0", "old_1", "old_2")[index]
            score = point["scores"][name]
            assert score["episode_id"] == record.episode_id and score["role"] == record.role
            assert score["residual"]["overall"]["normalized_half_mse"] == pytest.approx(
                expected, rel=1e-12, abs=1e-15
            )
    assert "candidate" not in report and "learner" not in report
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("family", FAMILIES)
def test_should_account_all_three_readout_inputs_and_signed_retention_without_training_them(
    successful, family
):
    report = successful[family][0]
    work = report["work"]
    assert work["prediction_attempts"] == work["prediction_returns"] == 8
    assert work["prediction_rows_attempted"] == work["prediction_rows_returned"] == 16
    assert (
        work["score_rows_attempted_by_role"]
        == work["score_rows_returned_by_role"]
        == dict(probe_test=4, train=12)
    )
    assert work["old_readout_input_numeric_bytes"] == 192
    assert work["reserved_numeric_bytes"] == 128
    assert work["prediction_numeric_bytes_returned"] == 256
    assert work["training_attempts"] == work["native_returns"] == 1
    assert work["training_rows_attempted"] == work["training_rows_completed"] == 2
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 1
    assert report["events"][0]["success_validated"] is True
    assert report["events"][0]["returned_receipts"] == report["applied_updates"]
    first, final = [point["scores"] for point in report["checkpoints"]]
    loss = lambda point, name: point[name]["residual"]["overall"]["normalized_half_mse"]
    assert report["metrics"] == dict(
        fresh_acquisition_gain=loss(first, "fresh") - loss(final, "fresh"),
        old_retention_loss={
            name: loss(final, name) - loss(first, name) for name in ("old_0", "old_1", "old_2")
        },
    )
    assert report["applied_updates"][0]["episode_id"] == "synthetic-Ftrain"


@pytest.mark.parametrize(
    "fault",
    [
        "type",
        "probe_train",
        "probe_test",
        "inner_guard",
        "final_test",
        "context",
        "training_id",
        "evaluation_id",
        "spec",
    ],
)
def test_should_refuse_invalid_third_readout_before_any_fork_or_train(owners, monkeypatch, fault):
    old: Any = readouts()
    third = old[2]
    if fault == "type":
        old = old[:2] + (asdict(third),)
    elif fault in ("probe_train", "probe_test", "inner_guard", "final_test"):
        object.__setattr__(third, "role", fault)
    elif fault == "context":
        object.__setattr__(third, "context", "fresh-F")
    elif fault in ("training_id", "evaluation_id"):
        object.__setattr__(
            third, "episode_id", "synthetic-Ftrain" if fault == "training_id" else "synthetic-Ftest"
        )
    else:
        object.__setattr__(third, "spec", replace(third.spec, action_units=("other", "m")))
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
    monkeypatch.setattr(probe, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError):
        probe.run_continuous_probe(owners["bp"], plan(), training_readouts=old)


@pytest.mark.parametrize("container", ["four", "list"])
def test_should_refuse_four_or_non_tuple_readouts_before_private_work(
    owners, monkeypatch, container
):
    old = readouts() + (episode(),) if container == "four" else list(readouts())
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "fork", forbidden)
    monkeypatch.setattr(probe, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError):
        probe.run_continuous_probe(owners["bp"], plan(), training_readouts=old)


@pytest.mark.parametrize("count", [0, 2])
def test_should_keep_zero_and_two_readout_schema_and_scores_compatible(owners, successful, count):
    options = {} if count == 0 else dict(training_readouts=readouts()[:2])
    report = probe.run_continuous_probe(owners["bp"], plan(), **options)
    assert report["complete"] is True, report.get("error")
    names = ["fresh"] + [f"old_{index}" for index in range(count)]
    assert all(list(point["scores"]) == names for point in report["checkpoints"])
    assert set(report) == set(successful["bp"][0])
    for index, point in enumerate(report["checkpoints"]):
        for name in names:
            assert (
                point["scores"][name] == successful["bp"][0]["checkpoints"][index]["scores"][name]
            )
    assert report["work"]["old_readout_input_numeric_bytes"] == (160 if count else 0)


def test_should_own_third_readout_before_caller_mutates_it_from_boundary(owners, successful):
    old = readouts()
    called = False

    def mutate_original():
        nonlocal called
        if not called:
            object.__setattr__(old[2], "context", "replaced-after-copy")
            object.__setattr__(old[2].transitions[0], "action", (6.0, 1.0))
            called = True

    report = probe.run_continuous_probe(
        owners["bp"], plan(), training_readouts=old, check_boundary=mutate_original
    )
    assert called and report["complete"] is True, report.get("error")
    assert report["metrics"] == successful["bp"][0]["metrics"]
    assert [point["scores"]["old_2"] for point in report["checkpoints"]] == [
        point["scores"]["old_2"] for point in successful["bp"][0]["checkpoints"]
    ]


@pytest.mark.parametrize("after_update", [False, True])
def test_should_preserve_attempts_and_paid_prefix_when_third_scoring_fails(
    owners, monkeypatch, after_update
):
    source = owners["bp"].fork()
    before = source.snapshot_state()
    predict = bp.NumpyContinuousBCLearner.predict
    calls = 0
    failure_call = 8 if after_update else 4

    def fail_after_native_return(owner, features):
        nonlocal calls
        calls += 1
        returned = predict(owner, features)
        if calls == failure_call:
            raise RuntimeError("third readout failed after paid prediction")
        return returned

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "predict", fail_after_native_return)
    report = probe.run_continuous_probe(source, plan(), training_readouts=readouts())
    paid = int(after_update)
    assert report["complete"] is False and "third readout" in report["error"]
    assert calls == report["work"]["prediction_attempts"] == failure_call
    assert report["work"]["prediction_returns"] == failure_call - 1
    assert report["work"]["prediction_rows_attempted"] == 8 * (1 + paid)
    assert report["work"]["prediction_rows_returned"] == 8 * (1 + paid) - 1
    assert len(report["native_updates"]) == len(report["applied_updates"]) == paid
    assert report["work"]["training_attempts"] == paid
    assert report["work"]["training_rows_completed"] == 2 * paid
    assert list(report["checkpoints"][-1]["scores"]) == ["fresh", "old_0", "old_1"]
    assert report["checkpoints"][-1]["complete"] is False
    assert "metrics" not in report
    assert report["progress"]["updates_completed"] == paid
    if paid:
        assert report["events"][0]["success_validated"] is True
        assert report["events"][0]["returned_receipts"] == report["applied_updates"]
    assert_state(source.snapshot_state(), before)


def test_should_observe_validated_endpoints_with_three_readouts(owners, successful):
    observations = []

    def observe(readout, phase, guard):
        snapshots = []
        guard(lambda: snapshots.append(readout.snapshot_state()))
        state = snapshots[0]
        observations.append((phase, state.completed_updates, state.training_examples))

    report = probe.run_continuous_probe(
        owners["bp"], plan(), training_readouts=readouts(), endpoint_observer=observe
    )
    assert report["complete"] is True, report.get("error")
    assert observations == [("before_probe", 0, 0), ("after_probe", 1, 2)]
    assert report["metrics"] == successful["bp"][0]["metrics"]
    assert report["work"]["prediction_returns"] == 8
    assert report["endpoint_observations"] == [
        dict(phase=phase, attempted=True, completed=True) for phase, _, _ in observations
    ]


def test_should_forward_three_readouts_to_selected_velocity_pair_without_extra_authority(
    monkeypatch,
):
    old = readouts()
    seen = []

    def paired(source, value, **options):
        seen.append(options)
        return dict(complete=False)

    monkeypatch.setattr(probe, "run_zero_velocity_pair", paired)
    probe.run_continuous_probe(object(), plan(zero_velocity_control=True), training_readouts=old)
    assert len(seen) == 1 and seen[0]["training_readouts"] is old


def test_should_keep_three_old_scores_and_charge_diagnostics_when_enabled(owners, successful):
    report = probe.run_continuous_probe(
        owners["bp"], plan(diagnostics=True), training_readouts=readouts()
    )
    assert report["complete"] is True, report.get("error")
    ordinary = successful["bp"][0]
    for observed, previous in zip(report["checkpoints"], ordinary["checkpoints"], strict=True):
        assert list(observed["scores"]) == ["fresh", "old_0", "old_1", "old_2", "fresh_training"]
        for name, score in previous["scores"].items():
            assert observed["scores"][name] == score
        assert observed["scores"]["fresh_training"]["role"] == "probe_train"
    assert report["metrics"]["old_retention_loss"] == ordinary["metrics"]["old_retention_loss"]
    assert (
        report["metrics"]["fresh_acquisition_gain"] == ordinary["metrics"]["fresh_acquisition_gain"]
    )
    before, after = [
        point["scores"]["fresh_training"]["residual"]["overall"]["normalized_half_mse"]
        for point in report["checkpoints"]
    ]
    assert report["metrics"]["fresh_training_gain"] == before - after
    assert report["work"]["prediction_attempts"] == report["work"]["prediction_returns"] == 10
    assert report["work"]["prediction_rows_returned"] == 20
    diagnostic = report["work"]["step_diagnostics"]
    assert diagnostic["gradient_rows_returned"] == 2
    assert diagnostic["gradient_values_returned"] == 12
    assert diagnostic["gradient_numeric_bytes_returned"] == 96
