"""Explicit220-row composition; at most3 real SGD/660rows, zero random models/LS/SDK."""

from copy import deepcopy
from dataclasses import replace
import random
from typing import Any, cast

import numpy as np
import pytest

from scripts import push_cube_correction_weights as weights
from scripts import push_cube_training_readouts as readouts
from src.adapters import continuous_train_continuation as continuation
from src.adapters import continuous_mixed_training as mixed
from src.adapters import numpy_continuous_bc as bp
from src.adapters.continuous_learnability_modes import same_continuous_state
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.continuous_bc import Parameters

WORK = dict(sgd_attempts=0, sgd_rows=0, sgd_returns=0)


@pytest.fixture(autouse=True)
def bounded_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("complete220 checks must not initialize/solve/predict/create physics")

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "__init__", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "predict", forbidden)
    original = bp.NumpyContinuousBCLearner.train_batch

    def trained(source, x, y, **kwargs):
        WORK["sgd_attempts"] += 1
        WORK["sgd_rows"] += len(x)
        assert WORK["sgd_attempts"] <= 3 and WORK["sgd_rows"] <= 660
        result = original(source, x, y, **kwargs)
        WORK["sgd_returns"] += 1
        return result

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_batch", trained)


def episode(name, count):
    identifier = "pick-cube-readiness-teacher-19" if name == "b22" else "literal-" + name
    context = "pick-cube-train-19" if name == "b22" else "literal-context-" + name

    def observation(index):
        return (0.0, float(index) / 100, -0.25) + (0.0,) * 9

    return ContinuousEpisode(
        identifier,
        context,
        "train",
        cube_taskspace_spec(),
        tuple(
            ContinuousTransition(
                i,
                observation(i),
                (0.125, -0.25, 0.0, -1.0),
                0.0,
                observation(i + 1),
                i == count - 1,
                False,
            )
            for i in range(count)
        ),
    )


def groups():
    old = tuple(
        (name, episode(name, count))
        for name, count in zip(
            weights.AGGREGATED_GROUPS, weights.AGGREGATED_GROUP_ROWS, strict=True
        )
    )
    return weights.complete_b22_training_groups(old, episode("b22", 22))


def source():
    policy = bp.ContinuousBCPolicy(cube_taskspace_spec(), hidden_dim=2, momentum=0.2)
    parameters = (np.zeros((12, 2)), np.zeros(2), np.zeros((2, 4)), np.zeros(4))
    owner = object.__new__(bp.NumpyContinuousBCLearner)
    owner._policy = policy
    owner._state = bp.ContinuousBCSnapshot(
        policy, parameters, cast(Parameters, tuple(np.zeros_like(v) for v in parameters)), 0, 0
    )
    return owner


def test_should_preserve_all198_raw_weights_and_append_distinct_complete22_group():
    given = groups()
    before = deepcopy(given)
    prefix, _ = weights.build_aggregated_correction_weights(given[:6])
    actual, recipe = weights.build_complete_b22_correction_weights(given)
    assert len(actual) == 220 and actual[:198] == prefix
    assert actual[198:] == (98.0 / 22,) * 22
    assert recipe["group_order"] == list(weights.COMPLETE_B22_GROUPS)
    assert recipe["ideal_group_mass"] == 1 / 7
    assert all(value > 0 for value in actual)
    assert given == before and len({e.episode_id for _, e in given}) == 7
    owned, _ = readouts._admit(source(), given, corrective=True, aggregated=True, complete_b22=True)
    assert tuple(name for name, _ in owned) == weights.COMPLETE_B22_GROUPS
    assert owned == given and owned[6][1] is not given[6][1]


@pytest.mark.parametrize(
    "fault", ["role", "id", "context", "count", "spec", "duplicate", "order", "extra"]
)
def test_should_refuse_wrong_complete_demonstration_or_coverage(fault):
    given = groups()
    b22 = given[-1][1]
    changes: dict[str, Any] = {"role": "probe_train"} if fault == "role" else {}
    if fault == "id":
        changes = dict(episode_id="derived-B19-supervision")
    if fault == "context":
        changes = dict(context="pick-cube")
    if fault == "count":
        b22 = episode("b22", 21)
    if fault == "spec":
        changes = dict(
            spec=replace(b22.spec, observation_dim=13),
            transitions=tuple(
                replace(
                    row,
                    observation=(*row.observation, 0.0),
                    next_observation=(*row.next_observation, 0.0),
                )
                for row in b22.transitions
            ),
        )
    if changes:
        b22 = replace(b22, **changes)
    changed = (*given[:6], ("b22", b22))
    if fault == "duplicate":
        changed = (
            (given[0][0], replace(given[0][1], episode_id=b22.episode_id)),
            *given[1:6],
            ("b22", b22),
        )
    if fault == "order":
        changed = (*given[:5], given[6], given[5])
    if fault == "extra":
        changed = (*given, given[6])
    with pytest.raises(ValueError):
        weights.build_complete_b22_correction_weights(changed)


@pytest.mark.parametrize(
    "options",
    [
        dict(),
        dict(aggregated=True, corrective=True),
        dict(complete_b22=True),
        dict(complete_b22=1, corrective=True, aggregated=True),
    ],
)
def test_should_require_explicit_seven_group_readout_scope(options):
    with pytest.raises(ValueError):
        readouts._admit(source(), groups(), **options)


def test_should_forward_explicit_complete_scope_through_actual_readout_entry(monkeypatch):
    calls = []

    def admit(owner, given, **kwargs):
        calls.append(kwargs)
        raise ValueError("literal stop after actual dispatch")

    monkeypatch.setattr(readouts, "_admit", admit)
    report = readouts.run_training_readout(
        source(),
        groups(),
        label="complete220",
        guard=lambda: None,
        corrective=True,
        aggregated=True,
        complete_b22=True,
    )
    assert not report["complete"] and "literal stop" in report["error"]
    assert calls == [dict(corrective=True, aggregated=True, complete_b22=True)]


def test_should_complete_one_genuine220_row_update_with_frozen_source_and_no_rng_change():
    owner = source()
    before = owner.snapshot_state()
    data = groups()
    raw_weights, _ = weights.build_complete_b22_correction_weights(data)
    rng: Any = random.getstate(), np.random.get_state()
    result = continuation.run_continuous_train_continuation(
        owner,
        data[0][1],
        references=tuple(e for _, e in data[1:]),
        updates=1,
        update_rule="bp_mixed",
        row_weights=np.asarray(raw_weights),
        maximum_rows=220,
    )
    assert result.report["complete"] is True, result.report
    assert result.candidate is not None
    state = result.candidate.snapshot_state()
    assert state.completed_updates == 1 and state.training_examples == 220
    assert same_continuous_state(before, owner.snapshot_state())
    assert result.report["row_scope"]["maximum_rows"] == 220
    assert random.getstate() == rng[0]
    observed: Any = np.random.get_state()
    assert (
        observed[0] == rng[1][0]
        and np.array_equal(observed[1], rng[1][1])
        and observed[2:] == rng[1][2:]
    )


@pytest.mark.parametrize("scope", [148, 198])
def test_should_refuse220_under_old_scopes_without_update(scope):
    data = groups()
    with pytest.raises(ValueError):
        continuation.run_continuous_train_continuation(
            source(),
            data[0][1],
            references=tuple(e for _, e in data[1:]),
            update_rule="bp_mixed",
            maximum_rows=scope,
        )


def test_should_refuse220_head_gn_before_private_work():
    data = groups()
    with pytest.raises(ValueError, match="bp_mixed"):
        continuation.run_continuous_train_continuation(source(), data[0][1], maximum_rows=220)


def test_should_keep220_scope_and_weights_through_private_facade_fork():
    data = groups()
    raw_weights, _ = weights.build_complete_b22_correction_weights(data)
    original = mixed.mixed_training_source(source())
    port = mixed.RecordedContinuousMixed(
        original, [], 1, maximum_rows=220, row_weights=np.asarray(raw_weights)
    )
    fork = port.fork()
    assert fork._maximum_rows == fork._maximum_rows_original == 220
    assert not np.shares_memory(port._row_weights, fork._row_weights)
    fork._maximum_rows = 198
    with pytest.raises(RuntimeError, match="scope"):
        fork.require_reports()


def test_should_preserve_paid_unknown220_work_without_returning_candidate(monkeypatch):
    owner = source()
    before = owner.snapshot_state()
    data = groups()
    raw_weights, _ = weights.build_complete_b22_correction_weights(data)
    delegated = bp.NumpyContinuousBCLearner.train_batch

    def lose_return(model, x, y, **kwargs):
        delegated(model, x, y, **kwargs)
        raise RuntimeError("lost220 native update acknowledgement")

    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_batch", lose_return)
    result = continuation.run_continuous_train_continuation(
        owner,
        data[0][1],
        references=tuple(e for _, e in data[1:]),
        updates=1,
        update_rule="bp_mixed",
        row_weights=np.asarray(raw_weights),
        maximum_rows=220,
    )
    assert result.candidate is None and result.report["complete"] is False
    assert result.report["unreturned_training_work_unknown"] is True
    assert result.report["attempted_training_rows"] == 220
    assert same_continuous_state(before, owner.snapshot_state())
