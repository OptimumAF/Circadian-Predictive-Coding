"""Frozen TRAIN transforms, native equivalence and narrow live admission.

One module fixture/platform: two standardized plus two direct BP updates,
216training rows total. All remaining cases use pure metadata or that snapshot.
No environment, teacher, final/probe or comparative model cohort is executed.
"""

from dataclasses import replace
import pickle
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import standardized_minigrid_learner as port
from src.adapters.torch_feature_learner import TorchFeatureLearner
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.actor_shadow import ActorShadowRuntime
from src.core.experience import LogicalClock
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.core.observation_standardization import ObservationStandardization
from src.shared.torch_runtime import require_torch
from test_minigrid_train_fit_bounds import histories
from test_actor_shadow import PausableLearner


@pytest.fixture(scope="module")
def inputs():
    episodes = histories()
    features = np.asarray([s.observation for e in episodes for s in e.transitions])
    targets = np.asarray([[s.action] for e in episodes for s in e.transitions], dtype=np.float64)
    return episodes, features, targets


@pytest.mark.parametrize("replay_limit", [None, 0])
def test_should_admit_training_budget_without_claiming_runtime_replay_enforcement(replay_limit):
    source = PausableLearner()
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=128, max_replay_examples=replay_limit), lambda: 0.0
    )
    before = source.snapshot_state()
    if replay_limit is not None:
        with pytest.raises(ValueError, match="does not enforce width or replay"):
            ActorShadowRuntime(
                source,
                actor_version="a0",
                candidate_version="c0",
                clock=LogicalClock(),
                budget=budget,
                max_experiences=128,
                max_consolidations=0,
            )
    else:
        runtime = ActorShadowRuntime(
            source,
            actor_version="a0",
            candidate_version="c0",
            clock=LogicalClock(),
            budget=budget,
            max_experiences=128,
            max_consolidations=0,
        )
        assert runtime.applied_updates == ()
    assert budget.updates_completed == 0 and source.snapshot_state() == before


def test_should_fit_only_current_train_features_and_not_targets_or_caller_buffers(inputs):
    episodes, features, _ = inputs
    before = pickle.dumps(episodes)
    policy = port.fit_policy(episodes)
    canonical = np.asarray(
        [canonicalize_public_history(tuple(float(v) for v in row)) for row in features]
    )
    stats = policy.statistics
    assert stats.fitted_rows == 54
    np.testing.assert_allclose(stats.means, canonical.mean(axis=0), atol=1e-14)
    constant = np.ptp(canonical, axis=0) == 0
    assert np.asarray(stats.scales)[constant].tolist() == [1.0] * int(constant.sum())
    changed = tuple(
        replace(e, transitions=tuple(replace(s, action=(s.action + 1) % 7) for s in e.transitions))
        for e in episodes
    )
    assert port.fit_policy(changed) == policy
    assert port.copy_policy(policy) == policy and port.copy_policy(policy) is not policy
    assert pickle.dumps(episodes) == before


@pytest.mark.parametrize(
    "role", ["probe_train", "probe_test", "inner_guard", "outer_selection", "final_test"]
)
def test_should_refuse_nontrain_statistics_sources_before_native_work(inputs, role):
    episodes, _, _ = inputs
    with pytest.raises(ValueError):
        port.fit_policy((replace(episodes[0], role=role), *episodes[1:]))


@pytest.mark.parametrize(
    "fault", ["method", "width", "seed", "rate", "statistics", "encoding", "extra"]
)
def test_should_refuse_foreign_policy_or_statistics_before_constructing_native(
    inputs, fault, monkeypatch
):
    policy = port.fit_policy(inputs[0])
    monkeypatch.setattr(port, "TorchFeatureLearner", lambda p: pytest.fail("native construction"))
    with pytest.raises(ValueError):
        if fault in ("method", "width", "seed", "rate"):
            field, value = dict(
                method=("method", "pc"),
                width=("hidden_dim", 8),
                seed=("seed", 19),
                rate=("learning_rate", 0.02),
            )[fault]
            changes: dict[str, Any] = {field: value}
            port.StandardizedMiniGridLearner(
                replace(policy, native_policy=replace(policy.native_policy, **changes))
            )
        elif fault == "statistics":
            port.StandardizedMiniGridLearner(
                replace(policy, statistics=ObservationStandardization((0.0,), (1.0,), 54))
            )
        elif fault == "encoding":
            port.StandardizedMiniGridLearner(replace(policy, encoding_id="wrong"))
        else:
            object.__setattr__(policy, "unobserved", 1)
            port.StandardizedMiniGridLearner(policy)


@pytest.fixture(scope="module")
def native_result(inputs):
    _, features, targets = inputs
    torch = require_torch()
    rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    policy = port.fit_policy(inputs[0])
    learner = port.StandardizedMiniGridLearner(policy)
    reference = TorchFeatureLearner(policy.native_policy)
    initial = learner.snapshot_state()
    canonical = np.asarray(
        [canonicalize_public_history(tuple(float(v) for v in row)) for row in features]
    )
    transformed = (canonical - np.asarray(policy.statistics.means)) / np.asarray(
        policy.statistics.scales
    )
    learner.train_batch(features, targets)
    reference.train_batch(transformed, targets)
    assert _same_state(learner.snapshot_state().native_state, reference.snapshot_state())
    child = learner.fork()
    child.restore_state(learner.snapshot_state())
    child.train_batch(features, targets)
    reference.train_batch(transformed, targets)
    assert _same_state(child.snapshot_state().native_state, reference.snapshot_state())
    np.testing.assert_array_equal(child.predict(features), reference.predict(transformed))
    assert learner.completed_updates == 1 and child.completed_updates == 2
    assert child.training_examples == 108
    assert torch.equal(rng[0], torch.get_rng_state()) and rng[1] == pickle.dumps(
        np.random.get_state()
    )
    return child, initial


def test_should_match_direct_standardized_native_math_full_state_and_continuation(native_result):
    child, initial = native_result
    assert child.snapshot_state().policy == initial.policy
    owned = child.snapshot_state()
    owned.native_state.head_state["weight_feature_hidden"].fill_(9)
    assert not _same_state(owned, child.snapshot_state())


@pytest.mark.parametrize("fault", ["later_raw_row", "nan", "transform", "foreign_snapshot"])
def test_should_refuse_malformed_input_or_restore_atomically(native_result, inputs, fault):
    child, initial = native_result
    before = child.snapshot_state()
    features = inputs[1].copy()
    with pytest.raises(ValueError):
        if fault in ("later_raw_row", "nan"):
            features[-1, -1] = 0.123 if fault == "later_raw_row" else np.nan
            child.train_batch(features, inputs[2])
        elif fault == "foreign_snapshot":
            child.restore_state(cast(Any, initial.native_state))
        else:
            stats = replace(initial.policy.statistics, means=(1.0,) * 296)
            child.restore_state(replace(initial, policy=replace(initial.policy, statistics=stats)))
    assert _same_state(before, child.snapshot_state())


def test_should_route_only_two_train_cues_through_original_evaluator(native_result, monkeypatch):
    child, initial = native_result
    source = port.StandardizedMiniGridLearner(initial.policy)
    source.restore_state(initial)
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=0, max_replay_examples=0), lambda: 0.0
    )
    captured = {}

    def inspect(*args, **kwargs):
        captured.update(kwargs)
        assert args == (source, initial)
        assert kwargs["memory_copier"](initial).policy == initial.policy
        with pytest.raises(ValueError):
            kwargs["memory_copier"](child.snapshot_state())  # native2 is not a declared endpoint
        return {"intercepted": True}

    monkeypatch.setattr(evaluation, "_run_checkpoint", inspect)
    assert evaluation.evaluate_standardized_neural_checkpoint(
        source, initial, updates=0, budget=budget, check_boundary=lambda: None
    ) == {"intercepted": True}
    assert [(layout.context, cue, role) for layout, cue, role in captured["live_plan"]] == [
        ("A", "blue", "train"),
        ("A", "red", "train"),
    ]
    assert captured["sources"] == () and captured["require_initial_source"]
    assert captured["representation_encoding"] == port.ENCODING_ID


@pytest.mark.parametrize("updates", [True, 1, 127, 128, 129])
def test_should_refuse_wrong_endpoint_or_clock_before_evaluator(
    native_result, monkeypatch, updates
):
    _, initial = native_result
    monkeypatch.setattr(
        evaluation, "_run_checkpoint", lambda *a, **kw: pytest.fail("evaluator access")
    )
    with pytest.raises(ValueError):
        evaluation.evaluate_standardized_neural_checkpoint(
            None, initial, updates=updates, budget=cast(Any, None), check_boundary=lambda: None
        )
