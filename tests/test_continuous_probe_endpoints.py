"""Synchronous disposable endpoint observations with latched refusal.

Four tiny cached family owners supply all cases. Tests observe native snapshots
and existing paid reports; they neither retain borrowed learners nor score SDK IO.
"""

from copy import copy, deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.adapters import continuous_probe as probe
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.app.toy_execution_budget import ToyExecutionProgress
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from test_continuous_probe import assert_state, episode, plan


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")


def forbidden(*args, **kwargs):
    pytest.fail("invalid endpoint request admitted model or paired-probe work")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        SGD_attempts=0,
        SGD_returns=0,
        training_rows=0,
        prediction_calls=0,
        prediction_rows=0,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    trains = (
        bp.NumpyContinuousBCLearner.train_batch,
        pc.NumpyContinuousPCLearner.train_batch_with_work,
    )
    predicts = (bp.NumpyContinuousBCLearner.predict, pc.NumpyContinuousPCLearner.predict)

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (
            policy.spec.observation_dim + len(policy.spec.action_units)
        )
        return initialize(owner, policy)

    def trained(original):
        def train(owner, features, targets):
            counts["SGD_attempts"] += 1
            counts["training_rows"] += len(features)
            returned = original(owner, features, targets)
            counts["SGD_returns"] += 1
            return returned

        return train

    def predicted(original):
        def predict(owner, features):
            counts["prediction_calls"] += 1
            counts["prediction_rows"] += len(features)
            return original(owner, features)

        return predict

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", trained(trains[0]))
        patch.setattr(pc.NumpyContinuousPCLearner, "train_batch_with_work", trained(trains[1]))
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", predicted(predicts[0]))
        patch.setattr(pc.NumpyContinuousPCLearner, "predict", predicted(predicts[1]))
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("NPE_ENDPOINT_LEDGER"):
        with Path(os.environ["NPE_ENDPOINT_LEDGER"]).open("x", encoding="utf8") as handle:
            json.dump(counts, handle, sort_keys=True)
            handle.write("\n")
    assert counts["initializations"] == 4 and counts["normal_draws"] == 32
    assert counts["SGD_attempts"] <= 24 and counts["training_rows"] <= 72
    assert counts["prediction_calls"] <= 64 and counts["prediction_rows"] <= 128


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


def capture_scope(monkeypatch):
    captured: dict[str, Any] = {}
    score = probe._score
    runtime = probe.ActorShadowRuntime

    def observed_score(report, *args, **kwargs):
        captured["report"] = report
        return score(report, *args, **kwargs)

    def observed_runtime(*args, **options):
        captured["budget"] = options["budget"]
        captured["clock"] = options["clock"]
        owner = runtime(*args, **options)
        captured["runtime"] = owner
        return owner

    monkeypatch.setattr(probe, "_score", observed_score)
    monkeypatch.setattr(probe, "ActorShadowRuntime", observed_runtime)
    return captured


@pytest.fixture(scope="module")
def successful(owners):
    result = {}
    for family in FAMILIES:
        source = owners[family].fork()
        before = source.snapshot_state()
        value = plan()
        original_inputs = asdict(value)
        observations = []

        def observer(readout, phase, guard):
            guard()
            values = []

            def read_ports():
                values.append(readout.snapshot_state())

            guard(read_ports)
            observed = values[0]
            observations.append((phase, observed.completed_updates, observed.training_examples))

        report = probe.run_continuous_probe(source, value, endpoint_observer=observer)
        assert report["complete"] is True, report.get("error")
        assert_state(before, source.snapshot_state())
        assert asdict(value) == original_inputs
        result[family] = report, observations
    return result


@pytest.mark.parametrize("family", FAMILIES)
def test_should_observe_validated_before_and_after_private_readouts_without_leaking_learner(
    successful, family
):
    report, observations = successful[family]
    assert observations == [("before_probe", 0, 0), ("after_probe", 1, 2)]
    assert report["endpoint_observations"] == [
        dict(phase=phase, attempted=True, completed=True)
        for phase in ("before_probe", "after_probe")
    ]
    assert report["events"][0]["success_validated"] is True
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 1
    assert report["work"]["training_rows_completed"] == 2
    assert report["work"]["prediction_returns"] == 2
    assert [point["updates"] for point in report["checkpoints"]] == [0, 1]
    assert "candidate" not in report and "learner" not in report
    json.dumps(report, allow_nan=False)


def test_should_preserve_default_schema_and_same_imitation_result(owners, successful):
    default = probe.run_continuous_probe(owners["bp"].fork(), plan(), endpoint_observer=None)
    assert default["complete"] is True and "endpoint_observations" not in default
    observed = successful["bp"][0]
    omitted = {"elapsed_seconds", "endpoint_observations"}
    assert {key: value for key, value in default.items() if key not in omitted} == {
        key: value for key, value in observed.items() if key not in omitted
    }


def test_should_omit_none_observer_keyword_at_existing_dispatch_boundary(monkeypatch):
    calls = []

    def single(*args, **options):
        calls.append(options)
        return dict(complete=False)

    monkeypatch.setattr(probe, "_run_single_probe", single)
    probe.run_continuous_probe(object(), object())
    probe.run_continuous_probe(object(), object(), endpoint_observer=None)
    assert calls == [dict(training_readouts=(), check_boundary=None)] * 2


@pytest.mark.parametrize("observer", [False, 1, object(), []])
def test_should_refuse_noncallable_observers_before_probe_work(monkeypatch, observer):
    monkeypatch.setattr(probe, "_run_single_probe", forbidden)
    monkeypatch.setattr(probe, "run_zero_velocity_pair", forbidden)
    with pytest.raises(ValueError):
        probe.run_continuous_probe(object(), plan(), endpoint_observer=observer)


def test_should_refuse_ambiguous_zero_velocity_pair_with_observer_before_work(monkeypatch):
    monkeypatch.setattr(probe, "_run_single_probe", forbidden)
    monkeypatch.setattr(probe, "run_zero_velocity_pair", forbidden)
    with pytest.raises(ValueError):
        probe.run_continuous_probe(
            object(),
            plan(zero_velocity_control=True),
            endpoint_observer=lambda readout, phase, guard: None,
        )


@pytest.mark.parametrize("phase", ["before_probe", "after_probe"])
@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "readout",
        "report_value",
        "report_identity",
        "budget_counter",
        "clock",
        "exception",
        "non_none_return",
    ],
)
def test_should_refuse_endpoint_mutation_and_preserve_trusted_paid_evidence(
    owners, monkeypatch, phase, fault
):
    source = owners["bp"].fork()
    before = source.snapshot_state()
    captured = capture_scope(monkeypatch)
    called = []

    def observer(readout, current, guard):
        called.append(current)
        if current != phase:
            guard()
            return None
        if fault == "source":
            source._state.parameters[3][0] += 0.125
        elif fault == "readout":
            readout._state.parameters[3][0] += 0.125
        elif fault == "report_value":
            captured["report"]["work"]["prediction_returns"] += 100
        elif fault == "report_identity":
            captured["report"]["checkpoints"] = deepcopy(captured["report"]["checkpoints"])
        elif fault == "budget_counter":
            captured["budget"].updates_completed += 1
        elif fault == "clock":
            captured["clock"].advance_to(100)
        elif fault == "exception":
            raise RuntimeError("endpoint observer failed")
        elif fault == "non_none_return":
            return dict(success=True)
        return None

    report = probe.run_continuous_probe(source, plan(), endpoint_observer=observer)
    paid = int(phase == "after_probe")
    assert report["complete"] is False
    assert called == (["before_probe"] if not paid else ["before_probe", "after_probe"])
    assert len(report["native_updates"]) == len(report["applied_updates"]) == paid
    assert report["work"]["training_rows_completed"] == 2 * paid
    assert report["work"]["prediction_returns"] == 1 + paid
    assert report["endpoint_observations"][-1] == dict(phase=phase, attempted=True, completed=False)
    assert len(report["endpoint_observations"]) == 1 + paid
    if paid:
        assert report["events"][0]["success_validated"] is True
        assert report["events"][0]["returned_receipts"] == report["applied_updates"]
        first, final = (
            point["scores"]["fresh"]["residual"]["overall"]["normalized_half_mse"]
            for point in report["checkpoints"]
        )
        assert report["metrics"]["fresh_acquisition_gain"] == first - final
    else:
        assert "metrics" not in report
    if fault == "source":
        assert source._state.parameters[3][0] == before.parameters[3][0] + 0.125
    else:
        assert_state(before, source.snapshot_state())


@pytest.mark.parametrize("fault", ["readout_repaired", "ports_throw", "ports_non_none"])
def test_should_latch_guard_refusal_even_when_observer_catches_and_repairs_it(owners, fault):
    source = owners["bp"].fork()
    called = []

    def observer(readout, phase, guard):
        called.append(phase)
        assert phase == "before_probe"
        original = readout._state.parameters[3][0]
        if fault == "readout_repaired":
            readout._state.parameters[3][0] += 0.125
            try:
                with pytest.raises(Exception):
                    guard()
            finally:
                readout._state.parameters[3][0] = original
        else:

            def ports():
                if fault == "ports_throw":
                    raise RuntimeError("native observation failed")
                return dict(unexpected="port result")

            with pytest.raises(Exception):
                guard(ports)
        # Even a callback that swallows every guard exception cannot resume work.

    report = probe.run_continuous_probe(source, plan(), endpoint_observer=observer)
    assert report["complete"] is False and called == ["before_probe"]
    assert report["native_updates"] == report["applied_updates"] == []
    assert report["work"]["training_attempts"] == 0
    assert report["endpoint_observations"] == [
        dict(phase="before_probe", attempted=True, completed=False)
    ]


@pytest.mark.parametrize(
    "fault",
    [
        "source_spoof",
        "progress",
        "sampler",
        "actor",
        "inbox",
        "candidate_port",
        "candidate_native",
    ],
)
def test_should_refuse_replaced_authority_before_first_endpoint(owners, monkeypatch, fault):
    source = owners["bp"].fork()
    original = source.snapshot_state()
    captured = capture_scope(monkeypatch)
    score = probe._score
    called = []

    def initial_score(*args, **kwargs):
        result = score(*args, **kwargs)
        runtime = captured["runtime"]
        if fault == "source_spoof":
            source._state.parameters[3][0] += 0.125
            source.snapshot_state = lambda: deepcopy(original)
        elif fault == "progress":
            captured["budget"].progress = ToyExecutionProgress()
        elif fault == "sampler":
            captured["budget"].process_rss_sampler = object()
        elif fault == "actor":
            runtime._actor = copy(runtime._actor)
        elif fault == "inbox":
            runtime._inbox = copy(runtime._inbox)
        elif fault == "candidate_port":
            runtime._candidate = copy(runtime._candidate)
            runtime._inbox._learner = runtime._candidate
        else:
            runtime._candidate.learner = runtime._candidate.learner.fork()
        return result

    monkeypatch.setattr(probe, "_score", initial_score)
    report = probe.run_continuous_probe(
        source, plan(), endpoint_observer=lambda *_: called.append(True)
    )
    assert report["complete"] is False and called == []
    assert report["native_updates"] == report["applied_updates"] == []
    assert report["work"]["training_attempts"] == 0
    assert report["work"]["prediction_returns"] == 1


def test_should_reprove_runtime_after_last_endpoint_readout_getter(owners, monkeypatch):
    source = owners["bp"].fork()
    captured = capture_scope(monkeypatch)

    def observer(readout, phase, guard):
        if phase == "after_probe":
            getter = readout.snapshot_state

            def replaced_runtime():
                runtime = captured["runtime"]
                runtime._candidate.learner = runtime._candidate.learner.fork()
                return getter()

            readout.snapshot_state = replaced_runtime

    report = probe.run_continuous_probe(source, plan(), endpoint_observer=observer)
    assert report["complete"] is False
    assert report["endpoint_observations"][-1]["completed"] is False
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 1
    assert report["work"]["training_rows_completed"] == 2


def test_should_refuse_native_replacement_from_final_elapsed_clock(owners, monkeypatch):
    source = owners["bp"].fork()
    captured = capture_scope(monkeypatch)
    timer = probe.monotonic

    def observed_time():
        report = captured.get("report", {})
        points = report.get("endpoint_observations", [])
        if len(points) == 2 and points[-1]["completed"]:
            runtime = captured["runtime"]
            runtime._candidate.learner = runtime._candidate.learner.fork()
        return timer()

    monkeypatch.setattr(probe, "monotonic", observed_time)
    report = probe.run_continuous_probe(source, plan(), endpoint_observer=lambda *_: None)
    assert report["complete"] is False and "final_observation_error" in report
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 1
    assert report["work"]["training_rows_completed"] == 2
