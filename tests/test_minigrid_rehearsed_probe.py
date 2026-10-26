"""Four original native methods: isolated synthetic F16+old-A replay+A16 return.

Planned256 updates2304rows/128predictions1152rows. No game/teacher/final or
benchmark updates; fixture labels establish routing, not acquired-task benefit.
"""

from dataclasses import asdict
import pickle

import numpy as np
import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters.minigrid_fixed_rehearsal import build_minigrid_old_a_rehearsal
from src.adapters.minigrid_return_fit import return_training_sources
from src.adapters.torch_feature_learner import TorchFeatureLearner
from src.app.actor_shadow import StableActor
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.torch_runtime import require_torch
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import bound, train_sources
from test_torch_feature_learner import assert_tree


@pytest.fixture(scope="module")
def native_work():
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    patch = pytest.MonkeyPatch()
    train, predict = TorchFeatureLearner.train_batch, TorchFeatureLearner.predict
    counts = dict(updates=0, rows=0, predictions=0, prediction_rows=0)

    def training(owner, features, targets):
        counts["updates"] += 1
        counts["rows"] += len(features)
        assert counts["updates"] <= 266 and counts["rows"] <= 2756
        return train(owner, features, targets)

    def prediction(owner, features):
        counts["predictions"] += 1
        counts["prediction_rows"] += len(features)
        assert counts["predictions"] <= 148 and counts["prediction_rows"] <= 1368
        return predict(owner, features)

    patch.setattr(TorchFeatureLearner, "train_batch", training)
    patch.setattr(TorchFeatureLearner, "predict", prediction)
    yield torch, counts
    patch.undo()


def session(*, updates=None, replay=None):
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=updates, max_replay_examples=replay), lambda: 1.0
    )


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_keep_rehearsal_on_the_disposable_f_learner_then_return_without_more_replay(
    native_work, method, record_property
):
    torch, work = native_work
    previous = work.copy()
    global_rng, numpy_rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    source = bound(method)
    state = source.snapshot_state()
    actor = StableActor(source, version="benchmark")
    actor_before = actor.snapshot_state()
    benchmark = session(updates=0)
    original_benchmark = (id(benchmark.budget), id(benchmark.clock), benchmark.started_at)
    raw_training, raw_evaluation = raw_probe_data()
    training, evaluation = (
        pilot._history_packets(raw_training),
        pilot._history_packets(raw_evaluation),
    )
    monitors = train_sources()[:2]
    returns = return_training_sources(monitors)
    bank = tuple(packet.episode for packet in monitors)
    original_inputs = pickle.dumps((training, evaluation, monitors, returns, bank))
    wake, replay, returning = session(updates=16), session(replay=288), session(updates=16)
    authority = tuple(
        (id(row), id(row.budget), id(row.clock), row.started_at)
        for row in (wake, replay, returning)
    )
    plans, endpoints, endpoint_owners = [], {}, []

    def factory(learner, budget, updates):
        plan = build_minigrid_old_a_rehearsal(
            learner,
            budget,
            bank,
            replay_budget=replay,
            total_updates=updates,
            wake_role="probe_train",
        )
        plans.append(plan)
        return plan.rehearsal

    def observe(learner, phase, boundary):
        def read():
            endpoint_owners.append(id(learner))
            endpoints[phase] = learner.snapshot_state()

        boundary(read)

    report = run_fresh_task_probe(
        source,
        state,
        training,
        evaluation,
        budget=wake,
        checkpoint_updates=(0, 4, 8, 16),
        timer=lambda: 1.0,
        decision_diagnostics=True,
        training_readouts=monitors,
        return_training=returns,
        return_budget=returning,
        rehearsal_factory=factory,
        endpoint_observer=observe,
    )
    assert report["status"] == "complete", report["reason"]
    assert report["return_phase"]["status"] == "complete", report["return_phase"]["reason"]
    assert report["completed_updates"] == wake.updates_completed == 16
    assert report["completed_training_examples"] == 144
    assert report["return_phase"]["completed_updates"] == returning.updates_completed == 16
    assert replay.replay_examples_completed == 288 and replay.updates_completed == 0
    assert wake.replay_examples_completed == returning.replay_examples_completed == 0
    assert endpoints["after_probe"].completed_updates == 48
    assert endpoints["after_probe"].training_examples == 432
    assert endpoints["after_return"].completed_updates == 64
    assert endpoints["after_return"].training_examples == 576
    assert endpoint_owners[0] == endpoint_owners[1]
    plan = plans[0]
    assert (plan.validation_projection_rows, plan.validation_projected_array_equivalent_bytes) == (
        36,
        85248,
    )
    helper = plan.rehearsal
    assert (
        report["rehearsal"]["wake_role"] == report["probe_stream"]["training_role"] == "probe_train"
    )
    assert report["rehearsal"]["fixed_bank"]["encoded_array_bytes"] == 87280
    assert (
        report["rehearsal"]["attempted_updates"] == report["rehearsal"]["committed_updates"] == 32
    )
    assert report["rehearsal"]["charged_exposures"] == 288
    assert len(report["probe_stream"]["applied_updates"]) == 16
    assert len(report["probe_stream"]["applied_consolidations"]) == 16
    assert report["probe_stream"]["score_forward_calls"] == 0
    assert all(row.role == "probe_train" for row in helper._runtime._inbox._experiences.values())
    assert all(row.role == "probe_train" for row in helper._runtime._inbox._labels.values())
    for event in report["rehearsal"]["events"]:
        assert event["selected_episodes"] == [asdict(packet) for packet in bank]
    for phase in (report, report["return_phase"]):
        assert phase["score_forward_calls_completed"] == 8
        assert phase["training_readouts"]["score_forward_calls_completed"] == 8
        assert [point["updates"] for point in phase["checkpoints"]] == [0, 4, 8, 16]
    assert report["return_phase"]["checkpoints"][0]["scores"] == report["checkpoints"][-1]["scores"]
    assert (
        report["return_phase"]["training_readouts"]["checkpoints"][0]["scores"]
        == report["training_readouts"]["checkpoints"][-1]["scores"]
    )
    assert authority == tuple(
        (id(row), id(row.budget), id(row.clock), row.started_at)
        for row in (wake, replay, returning)
    )
    assert benchmark.updates_completed == benchmark.replay_examples_completed == 0
    assert original_benchmark == (id(benchmark.budget), id(benchmark.clock), benchmark.started_at)
    assert_tree(torch, source.snapshot_state(), state)
    assert_tree(torch, actor.snapshot_state(), actor_before)
    assert_tree(torch, helper._learner.snapshot_state(), state)
    assert_tree(torch, helper._runtime.actor.snapshot_state().state, state)
    assert pickle.dumps((training, evaluation, monitors, returns, bank)) == original_inputs
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert pickle.dumps(np.random.get_state()) == numpy_rng
    delta = {key: work[key] - previous[key] for key in work}
    assert delta == dict(updates=64, rows=576, predictions=32, prediction_rows=288)
    for key, value in delta.items():
        record_property(key, value)
