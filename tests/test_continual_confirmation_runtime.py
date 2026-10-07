"""Live resource observers count execution outside restored model snapshots."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from typing import Any

import numpy as np
import pytest

from src.app import continual_confirmation_training as training
from src.app.continual_confirmation_execution import verify_observed_updates
from src.app.continual_confirmation_manifest import fixed_confirmation_manifest
from src.app.continual_confirmation_validation import _verify_seed_envelope
from src.app.continual_confirmation_work_validation import _verify_seed_work
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.infra.continual_confirmation_runtime import ConfirmationStopped, ExecutionObserver
from src.shared.process_memory import ProcessRssSampler


def _sampler(value: Any = 100) -> ProcessRssSampler:
    return ProcessRssSampler(read_rss_bytes=lambda: value)


def _models() -> tuple[Any, ...]:
    return (
        BackpropMLP(2, 8, 41),
        PredictiveCodingNetwork(2, 8, 41),
        CircadianPredictiveCodingNetwork(
            2, 8, 41, circadian_config=CircadianConfig(replay_steps=2)
        ),
    )


def _train(model: Any) -> None:
    inputs = np.array([[0.1, 0.2], [0.3, 0.4]])
    targets = np.array([[0.0], [1.0]])
    if isinstance(model, BackpropMLP):
        model.train_epoch(inputs, targets, 0.01)
    else:
        model.train_epoch(inputs, targets, 0.01, 2, 0.01)


def test_should_count_each_model_update_once_and_restore_methods() -> None:
    originals = (
        BackpropMLP.train_epoch,
        PredictiveCodingNetwork.train_epoch,
        CircadianPredictiveCodingNetwork._run_training_step,
    )
    models = _models()
    with _sampler() as sampler:
        observer = ExecutionObserver(fixed_confirmation_manifest(), sampler, clock=lambda: 0.0)
        with observer.observe_updates():
            for model in models:
                _train(model)
    assert observer.updates() == {
        "attempted_updates": 3,
        "executed_updates": 3,
        "by_model_kind": {"backprop": 1, "pc": 1, "circadian": 1},
    }
    assert originals == (
        BackpropMLP.train_epoch,
        PredictiveCodingNetwork.train_epoch,
        CircadianPredictiveCodingNetwork._run_training_step,
    )


def test_should_stop_before_next_update_and_preserve_that_model() -> None:
    model = _models()[0]
    original = BackpropMLP.train_epoch
    manifest = replace(fixed_confirmation_manifest(), max_optimizer_updates=1)
    with _sampler() as sampler:
        observer = ExecutionObserver(manifest, sampler, clock=lambda: 0.0)
        with pytest.raises(ConfirmationStopped, match="optimizer_limit"):
            with observer.observe_updates():
                _train(model)
                before = model.weight_input_hidden.copy()
                _train(model)
    assert observer.executed_updates == 1 and observer.attempted_updates == 1
    np.testing.assert_array_equal(before, model.weight_input_hidden)
    assert BackpropMLP.train_epoch is original


@pytest.mark.parametrize(
    "kind", ["rss", "wall", "unavailable", "clock_nonfinite", "clock_backwards"]
)
def test_should_fail_before_an_update_when_live_resources_are_invalid(kind: str) -> None:
    model = _models()[0]
    before = model.weight_input_hidden.copy()
    state: dict[str, Any] = {"rss": 100, "clock": 0.0}
    with ProcessRssSampler(read_rss_bytes=lambda: state["rss"]) as sampler:
        observer = ExecutionObserver(
            fixed_confirmation_manifest(), sampler, clock=lambda: state["clock"]
        )
        state["rss"] = (
            512 * 1024 * 1024 + 1 if kind == "rss" else None if kind == "unavailable" else 100
        )
        state["clock"] = (
            601.0
            if kind == "wall"
            else float("nan")
            if kind == "clock_nonfinite"
            else -1.0
            if kind == "clock_backwards"
            else 0.0
        )
        with pytest.raises(
            (ConfirmationStopped, RuntimeError, ValueError), match="rss|RSS|wall|clock"
        ):
            with observer.observe_updates():
                _train(model)
        assert observer.executed_updates == 0 and observer.attempted_updates == 0
    np.testing.assert_array_equal(before, model.weight_input_hidden)


def test_should_keep_completed_update_when_post_update_rss_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"rss": 100}
    original = BackpropMLP.train_epoch

    def trained(model: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(model, *args, **kwargs)
        state["rss"] = 512 * 1024 * 1024 + 1
        return result

    monkeypatch.setattr(BackpropMLP, "train_epoch", trained)
    with ProcessRssSampler(read_rss_bytes=lambda: state["rss"]) as sampler:
        observer = ExecutionObserver(fixed_confirmation_manifest(), sampler, clock=lambda: 0.0)
        with pytest.raises(ConfirmationStopped, match="rss_limit"):
            with observer.observe_updates():
                _train(_models()[0])
    assert observer.executed_updates == 1 and observer.attempted_updates == 1
    assert BackpropMLP.train_epoch is trained


def test_should_keep_failed_attempt_and_restore_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _models()[0]

    def failed(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(BackpropMLP, "train_epoch", failed)
    with _sampler() as sampler:
        observer = ExecutionObserver(fixed_confirmation_manifest(), sampler, clock=lambda: 0.0)
        with pytest.raises(RuntimeError, match="injected optimizer"):
            with observer.observe_updates():
                _train(model)
    assert observer.executed_updates == 0 and observer.attempted_updates == 1
    assert BackpropMLP.train_epoch is failed


def test_should_stop_before_the_16001st_update_at_the_actual_frozen_cap() -> None:
    model = _models()[0]
    with _sampler() as sampler:
        observer = ExecutionObserver(fixed_confirmation_manifest(), sampler, clock=lambda: 0.0)
        observer.executed_updates = 16000
        observer.attempted_updates = 16000
        with pytest.raises(ConfirmationStopped, match="optimizer_limit"):
            with observer.observe_updates():
                _train(model)
    assert observer.executed_updates == 16000 and model._traffic_steps == 0


def test_should_stop_on_previously_observed_peak_after_current_rss_falls() -> None:
    state = {"rss": 100}
    with ProcessRssSampler(read_rss_bytes=lambda: state["rss"]) as sampler:
        observer = ExecutionObserver(fixed_confirmation_manifest(), sampler, clock=lambda: 0.0)
        state["rss"] = 512 * 1024 * 1024 + 1
        sampler.sample()
        state["rss"] = 100
        with pytest.raises(ConfirmationStopped, match="rss_limit"):
            observer.checkpoint()


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_optimizer_updates", True),
        ("wall_limit_seconds", float("nan")),
        ("max_process_rss_bytes", 0),
        ("rss_interval_seconds", 0.01),
    ],
)
def test_should_refuse_invalid_observer_limits_or_sampling(field: str, value: Any) -> None:
    with _sampler() as sampler:
        with pytest.raises(ValueError, match="cap|interval"):
            ExecutionObserver(
                replace(fixed_confirmation_manifest(), **{field: value}), sampler, clock=lambda: 0.0
            )


@pytest.mark.parametrize("forced_rejection", [False, True])
def test_should_count_all_six_development_families_including_rolled_back_replay(
    monkeypatch: pytest.MonkeyPatch, forced_rejection: bool
) -> None:
    families = tuple(
        replace(item, seeds=(item.development_seeds[0],))
        for item in fixed_confirmation_manifest().families
    )
    if forced_rejection:
        calls: dict[int, int] = {}

        def reject(model: Any, *args: Any, **kwargs: Any) -> float:
            count = calls.get(id(model), 0)
            calls[id(model)] = count + 1
            return 1.0 if count % 2 == 0 else 0.0

        monkeypatch.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", reject)
    with _sampler() as sampler:
        observer = ExecutionObserver(fixed_confirmation_manifest(), sampler)
        with observer.observe_updates():
            held = training._train_families(families)
    work = []
    rows = []
    for item, family in zip(held, families, strict=True):
        row = json.loads(json.dumps(asdict(item.facts), allow_nan=False))
        rows.append(row)
        _verify_seed_envelope(row, family)
        work.append(_verify_seed_work(row, family))
    verify_observed_updates(observer.updates(), tuple(work), {"seed_results": rows})
    assert sum(item.wake_updates for item in work) == 1344
    if forced_rejection:
        rejected = {item.family: item.rejected_executed_replay_updates for item in work}
        assert rejected["schedule"] == 12 and rejected["combined"] == 72
        assert rejected["sleep"] == 0 and rejected["parent"] == 0
