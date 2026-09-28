"""A combined checkpoint resumes seeded sleep at each transaction boundary."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from hashlib import sha256
import importlib
import random
from typing import Any

import numpy as np
import pytest

from src.app.circadian_checkpoint import (
    CircadianResumePosition,
    capture_circadian_checkpoint,
    restore_circadian_checkpoint,
)
from src.app.sleep_schedule import SleepRollbackCooldown, decide_sleep_attempt
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianNetworkSnapshot,
    CircadianPredictiveCodingNetwork,
)
from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead

try:
    torch: Any = importlib.import_module("torch")
except ImportError:
    torch = None


@pytest.fixture(autouse=True)
def _preserve_process_random_streams() -> Any:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone() if torch is not None else None
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    if torch is not None and torch_state is not None:
        torch.set_rng_state(torch_state)


def _model(backend: str) -> Any:
    if backend == "numpy":
        return CircadianPredictiveCodingNetwork(
            2,
            4,
            seed=181,
            min_hidden_dim=3,
            max_hidden_dim=6,
            circadian_config=CircadianConfig(
                sleep_mode="components",
                split_threshold=0.8,
                split_weight_norm_mix=0.0,
                split_importance_mix=0.0,
                max_split_per_sleep=1,
                max_prune_per_sleep=0,
                split_noise_scale=0.1,
                sleep_enable_prune=False,
                sleep_enable_homeostasis=False,
                sleep_enable_replay=False,
                sleep_enable_chemical_reset=False,
            ),
        )
    if torch is None:
        pytest.skip("Torch is optional")
    return CircadianPredictiveCodingHead(
        feature_dim=3,
        hidden_dim=4,
        num_classes=2,
        device=torch.device("cpu"),
        seed=241,
        config=CircadianHeadConfig(
            sleep_mode="components",
            split_threshold=0.8,
            split_weight_norm_mix=0.0,
            split_importance_mix=0.0,
            max_split_per_sleep=1,
            max_prune_per_sleep=0,
            split_noise_scale=0.1,
            sleep_enable_prune=False,
            sleep_enable_homeostasis=False,
            sleep_enable_chemical_reset=False,
        ),
        min_hidden_dim=3,
        max_hidden_dim=6,
    )


def _data_digest(backend: str) -> str:
    return sha256(f"fixed-seeded-{backend}-features-and-targets".encode()).hexdigest()


def _wake(model: Any, backend: str) -> float:
    if backend == "numpy":
        return model.train_epoch(
            np.array([[0.3, -0.2], [-0.1, 0.4]]),
            np.array([[1.0], [0.0]]),
            0.03,
            2,
            0.2,
        )
    return model.train_step(
        torch.tensor([[0.2, -0.3, 0.1], [-0.1, 0.4, 0.5]]),
        torch.tensor([1, 0], dtype=torch.long),
        0.03,
        2,
        0.2,
    )


def _make_split_eligible(model: Any, backend: str) -> None:
    values = [0.95] + [0.0] * (model.hidden_dim - 1)
    if backend == "numpy":
        model.set_chemical_state(np.array(values))
    else:
        model._chemical = torch.tensor(values)


def _draw_process_random(backend: str) -> tuple[float, ...]:
    values = (random.random(), float(np.random.random()))
    if backend == "torch":
        return (*values, float(torch.rand(()).item()))
    return values


def _act(name: str, model: Any, retry: SleepRollbackCooldown, backend: str) -> Any:
    if name.startswith("wake"):
        return (_wake(model, backend), _draw_process_random(backend))
    if name == "accepted_sleep":
        _make_split_eligible(model, backend)
        event = model.sleep_event(force_sleep=True)
        assert event.split_indices == (0,)
        return event
    if name == "rejected_sleep":
        _make_split_eligible(model, backend)
        before = model.snapshot_state()
        event = model.sleep_event(force_sleep=True)
        assert event.split_indices == (0,)
        model.restore_state(before)
        retry.record_rejection(
            completed_epochs=2, wake_batches=model.get_sleep_clocks().wake_batches
        )
        return event
    epoch = 3 if name == "due_epoch_three" else 4
    decision = decide_sleep_attempt(
        sleep_mode="components",
        completed_epochs=epoch,
        interval_epochs=1,
        adaptive_due=False,
        force_periodic=True,
    )
    allowed = retry.allow_due_attempt(
        decision, completed_epochs=epoch, wake_batches=model.get_sleep_clocks().wake_batches
    )
    if allowed:
        _make_split_eligible(model, backend)
        return (allowed, model.sleep_event(force_sleep=True))
    return (allowed, None)


def _same_value(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    if isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif torch is not None and torch.is_tensor(actual):
        assert torch.equal(actual, expected)
    elif isinstance(actual, np.random.Generator):
        _same_value(actual.bit_generator.state, expected.bit_generator.state)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _same_value(actual[key], expected[key])
    elif isinstance(actual, (tuple, list, deque)):
        assert len(actual) == len(expected)
        for current, other in zip(actual, expected):
            _same_value(current, other)
    elif is_dataclass(actual):
        for field in fields(actual):
            _same_value(getattr(actual, field.name), getattr(expected, field.name))
    else:
        assert actual == expected


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize(
    ("boundary", "last_action", "stage", "epoch"),
    [
        ("before_sleep", 0, "before_sleep", 1),
        ("after_accepted", 1, "after_sleep", 1),
        ("after_rejected", 3, "after_sleep", 2),
    ],
)
def test_combined_checkpoint_matches_uninterrupted_seeded_continuation(
    backend: str, boundary: str, last_action: int, stage: str, epoch: int
) -> None:
    if backend == "torch" and torch is None:
        pytest.skip("Torch is optional")
    random.seed(700)
    np.random.seed(701)
    if torch is not None:
        torch.manual_seed(702)
    actions = (
        "wake_one",
        "accepted_sleep",
        "wake_two",
        "rejected_sleep",
        "wake_three",
        "due_epoch_three",
        "wake_four",
        "due_epoch_four",
    )
    model = _model(backend)
    retry = SleepRollbackCooldown(1)
    for action in actions[: last_action + 1]:
        _act(action, model, retry, backend)
    position = CircadianResumePosition(
        completed_epoch=epoch,
        stage=stage,
        wake_batches=model.get_sleep_clocks().wake_batches,
    )
    saved = capture_circadian_checkpoint(
        model,
        retry=retry,
        position=position,
        protocol_id="checkpoint_fixture_v1",
        config=model.config,
        data_digest=_data_digest(backend),
    )
    expected_events = [_act(action, model, retry, backend) for action in actions[last_action + 1 :]]
    expected_state = model.snapshot_state()
    expected_retry = retry.snapshot_state()
    expected_draw = _draw_process_random(backend)

    resumed_model = _model(backend)
    resumed_retry = SleepRollbackCooldown(1)
    random.random()
    np.random.random()
    if backend == "torch":
        torch.rand(())
    restored_position = restore_circadian_checkpoint(
        resumed_model,
        saved,
        retry=resumed_retry,
        protocol_id="checkpoint_fixture_v1",
        config=resumed_model.config,
        data_digest=_data_digest(backend),
        expected_stage=stage,
    )
    assert restored_position == position
    actual_events = [
        _act(action, resumed_model, resumed_retry, backend) for action in actions[last_action + 1 :]
    ]
    assert actual_events == expected_events, boundary
    _same_value(resumed_model.snapshot_state(), expected_state)
    assert resumed_retry.snapshot_state() == expected_retry
    assert _draw_process_random(backend) == expected_draw


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize(
    "change",
    [
        "format",
        "backend",
        "protocol",
        "config",
        "data",
        "stage",
        "progress",
        "position_state",
        "model",
        "retry",
        "python_rng",
        "numpy_rng",
        "torch_rng",
    ],
)
def test_incompatible_checkpoint_rejects_before_live_mutation(backend: str, change: str) -> None:
    if backend == "torch" and torch is None:
        pytest.skip("Torch is optional")
    model = _model(backend)
    retry = SleepRollbackCooldown(1)
    _wake(model, backend)
    position = CircadianResumePosition(1, "before_sleep", model.get_sleep_clocks().wake_batches)
    saved = capture_circadian_checkpoint(
        model,
        retry=retry,
        position=position,
        protocol_id="checkpoint_fixture_v1",
        config=model.config,
        data_digest=_data_digest(backend),
    )
    destination = _model(backend)
    destination_retry = SleepRollbackCooldown(1)
    before_model = destination.snapshot_state()
    before_retry = destination_retry.snapshot_state()
    protocol = "checkpoint_fixture_v1"
    config = destination.config
    data_digest = _data_digest(backend)
    expected_stage = "before_sleep"
    if change == "format":
        saved = replace(saved, format_version=99)
    elif change == "backend":
        saved = replace(saved, backend="other_backend")
    elif change == "protocol":
        protocol = "other_protocol_v1"
    elif change == "config":
        config = replace(config, split_noise_scale=0.2)
    elif change == "data":
        data_digest = sha256(b"different data").hexdigest()
    elif change == "stage":
        expected_stage = "after_sleep"
    elif change == "progress":
        saved = replace(saved, position=replace(saved.position, wake_batches=99))
    elif change == "position_state":
        bad_position = deepcopy(saved.position)
        object.__setattr__(bad_position, "stage", "invalid")
        saved = replace(saved, position=bad_position)
    elif change == "model":
        bad_model = deepcopy(saved.model_state)
        if backend == "numpy":
            assert isinstance(bad_model, CircadianNetworkSnapshot)
            bad_model = replace(bad_model, format_version=99)
        else:
            assert isinstance(bad_model, dict)
            bad_model["format_version"] = 99
        saved = replace(saved, model_state=bad_model)
    elif change == "retry":
        assert saved.retry_state is not None
        saved = replace(saved, retry_state=replace(saved.retry_state, cooldown_epochs=2))
    elif change == "python_rng":
        saved = replace(saved, python_random_state=("invalid",))
    elif change == "numpy_rng":
        saved = replace(saved, numpy_random_state=("invalid",))
    else:
        bad_torch_state = (
            torch.ones(2, dtype=torch.float32) if backend == "torch" else np.array([1])
        )
        saved = replace(saved, torch_cpu_random_state=bad_torch_state)
    with pytest.raises((TypeError, ValueError), match="checkpoint|snapshot|state|incompatible"):
        restore_circadian_checkpoint(
            destination,
            saved,
            retry=destination_retry,
            protocol_id=protocol,
            config=config,
            data_digest=data_digest,
            expected_stage=expected_stage,
        )
    _same_value(destination.snapshot_state(), before_model)
    assert destination_retry.snapshot_state() == before_retry
