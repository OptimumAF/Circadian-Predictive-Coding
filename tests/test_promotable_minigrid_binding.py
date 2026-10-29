"""Actual serving ports on literal frames; no fits, forwards or native physics."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from src.adapters.minigrid_serving import run_promotable_minigrid_episode
from src.adapters.promotable_minigrid_binding import bind_promotable_minigrid
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import _same_value
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.serving_promotion import PromotableActor, ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ConsolidatedState
from src.core.experience import LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.minigrid_key_door import MiniGridKeyDoorLayout
from src.core.promotion_guard import GuardBatch, PromotionPolicy
from src.core.serving_ports import ServingConfiguration
from test_affine_reference_readout import literal_state
from test_minigrid_invariant_evaluation import DiscreteEnvironment
from test_minigrid_learning_pilot import Environment

WORK = dict(
    empty_owners=0,
    snapshots=0,
    predict_calls=0,
    predict_rows=0,
    literal_episodes=0,
    literal_steps=0,
    forbidden=0,
)


def digest(value):
    return sha256(pickle.dumps(value)).hexdigest()


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = {}
    original_init = ContextLinearLearner.__init__
    original_snapshot = ContextLinearLearner.snapshot_state
    original_fork, original_restore = ContextLinearLearner.fork, ContextLinearLearner.restore_state

    def forbidden(*args, **kwargs):
        WORK["forbidden"] += 1
        raise AssertionError("genuine numerical/physics work forbidden in literal cohort")

    def empty(owner):
        WORK["empty_owners"] += 1
        assert WORK["empty_owners"] <= 112
        original_init(owner)

    def snapshot(owner):
        WORK["snapshots"] += 1
        assert WORK["snapshots"] <= 4096
        return original_snapshot(owner)

    def fork(owner):
        if work.get("no_clone"):
            forbidden()
        return original_fork(owner)

    def restore(owner, state):
        if work.get("no_clone"):
            forbidden()
        return original_restore(owner, state)

    def predict(owner, features):
        WORK["predict_calls"] += 1
        WORK["predict_rows"] += len(features)
        assert WORK["predict_calls"] <= 64 and WORK["predict_rows"] <= 576
        work.setdefault("inputs", []).append(features.copy())
        callback = work.get("on_predict")
        if callback:
            callback(owner)
        if work.get("predict_error"):
            raise RuntimeError("scripted prediction lost return")
        result = np.full((len(features), 7), 1 / 7, dtype=np.float64)
        if owner._state.completed_updates:
            result[:] = 0
            result[:, 6] = 1
        return work.get("output", result)

    class LiteralEnvironment(DiscreteEnvironment):
        def __init__(self, *args, **kwargs):
            WORK["literal_episodes"] += 1
            assert WORK["literal_episodes"] <= 10
            super().__init__(*args, **kwargs)

        def step(self, action):
            WORK["literal_steps"] += 1
            assert WORK["literal_steps"] <= 32
            return super().step(action)

    monkeypatch.setattr(ContextLinearLearner, "__init__", empty)
    monkeypatch.setattr(ContextLinearLearner, "snapshot_state", snapshot)
    monkeypatch.setattr(ContextLinearLearner, "predict", predict)
    monkeypatch.setattr(ContextLinearLearner, "fork", fork)
    monkeypatch.setattr(ContextLinearLearner, "restore_state", restore)
    monkeypatch.setattr(ContextLinearLearner, "train_batch", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(Environment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", LiteralEnvironment)

    def build(updates=0):
        source = ContextLinearLearner()
        source._state = literal_state(updates)
        actor = PromotableActor(
            source,
            version="actor-0",
            configuration=ServingConfiguration(3, 2),
            feature_digest=digest,
            metadata={"route": "known-A"},
        )
        return source, actor

    return work, build


def budget():
    return ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=0,
            max_replay_examples=0,
            max_wall_seconds=150,
            max_process_rss_bytes=512 * 1024**2,
        ),
        lambda: 0.0,
    )


def play(actor, **options):
    defaults = dict(
        cue="blue", episode_id="caller-episode", budget=budget(), check_boundary=lambda: None
    )
    defaults.update(options)
    return run_promotable_minigrid_episode(actor, **defaults)


def pin(case):
    work, build = case
    _, actor = build()
    work["no_clone"] = True
    return work, actor, bind_promotable_minigrid(actor, episode_id="direct")


def test_should_read_detached_complete_idle_snapshot_and_refuse_contention(case):
    _, build = case
    _, actor = build(2)
    before = actor.serving_snapshot()
    idle = actor.serving_snapshot_if_idle()
    assert _same_value(before, idle)
    idle.metadata["route"] = "detached"
    assert actor.serving_snapshot().metadata == {"route": "known-A"}
    with actor._payload_exclusive(), pytest.raises(ValueError, match="busy"):
        actor.serving_snapshot_if_idle()


@pytest.mark.parametrize("updates", [0, 2])
def test_should_use_actual_public_prediction_and_preserve_source_and_bundle(
    case, monkeypatch, updates
):
    work, build = case
    source, actor = build(updates)
    source_before, actor_before = source.snapshot_state(), actor.serving_snapshot()
    calls = []
    actual = PromotableActor.predict

    def public(owner, features):
        calls.append(owner)
        return actual(owner, features)

    monkeypatch.setattr(PromotableActor, "predict", public)
    work["no_clone"] = True
    reader = bind_promotable_minigrid(actor, episode_id="owned")
    inputs = np.zeros((1, 296), dtype=np.float64)
    result = reader.predict(inputs)
    assert calls == [actor] and np.argmax(result) == (6 if updates else 0)
    assert _same_value(actor.serving_snapshot(), actor_before)
    assert _same_value(source.snapshot_state(), source_before)
    assert reader.report()["predictions_returned"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "actor",
        "backend",
        "metadata",
        "wide_metadata",
        "nan_metadata",
        "cache",
        "config",
        "tick",
        "failed",
        "version",
        "generation",
    ],
)
def test_should_refuse_invalid_native_admission_before_public_snapshot(case, monkeypatch, fault):
    _, build = case
    _, actor = build()
    bundle = actor._slot.bundle
    if fault == "actor":
        actor = object()
    elif fault == "backend":
        object.__setattr__(bundle, "learner", object())
    elif fault == "metadata":
        bundle.metadata["route"] = []
    elif fault == "wide_metadata":
        object.__setattr__(bundle, "metadata", {str(i): i for i in range(17)})
    elif fault == "nan_metadata":
        bundle.metadata["route"] = float("nan")
    elif fault == "cache":
        bundle.cache["foreign"] = object()
    elif fault == "config":
        object.__setattr__(bundle, "configuration", ServingConfiguration(1025, 2))
    elif fault == "tick":
        object.__setattr__(bundle, "last_cache_tick", True)
    elif fault == "failed":
        bundle.learner._state = replace(bundle.learner._state, failed_work=(1, 9, 0, 0, True))
    elif fault == "version":
        object.__setattr__(bundle, "version", "x" * 65)
    elif fault == "generation":
        object.__setattr__(actor._slot, "generation", 2**53)

    def forbidden(*args):
        pytest.fail("invalid metadata reached public snapshot")

    monkeypatch.setattr(PromotableActor, "serving_snapshot_if_idle", forbidden)
    with pytest.raises(ValueError):
        bind_promotable_minigrid(actor, episode_id="bad")


@pytest.mark.parametrize(
    "fault", ["slot", "backend", "metadata", "clock", "configuration", "model", "policy", "version"]
)
def test_should_stop_on_changed_or_equal_replaced_prior_bundle(case, fault):
    _, actor, reader = pin(case)
    bundle = actor._slot.bundle
    if fault == "slot":
        actor._slot = replace(actor._slot)
    elif fault == "backend":
        replacement = ContextLinearLearner()
        object.__setattr__(bundle, "learner", replacement)
    elif fault == "metadata":
        bundle.metadata["route"] = "changed"
    elif fault == "clock":
        object.__setattr__(bundle, "last_cache_tick", 1)
    elif fault == "configuration":
        object.__setattr__(bundle, "configuration", ServingConfiguration(4, 2))
    elif fault == "model":
        bundle.learner._state = literal_state(2)
    elif fault == "policy":
        object.__setattr__(bundle.learner._state.policy, "encoding_id", "changed")
    elif fault == "version":
        object.__setattr__(bundle, "version", "actor-other")
    with pytest.raises(ValueError):
        reader.check()
    assert reader.stopped and reader.report()["predictions_attempted"] == 0
    with pytest.raises(ValueError, match="stopped"):
        reader.check()


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "range", "list"])
def test_should_refuse_invalid_public_input_before_prediction(case, fault):
    _, _, reader = pin(case)
    inputs: Any = np.zeros((1, 296), dtype=np.float64)
    if fault == "shape":
        inputs = np.zeros((2, 296), dtype=np.float64)
    elif fault == "dtype":
        inputs = inputs.astype(np.float32)
    elif fault == "nan":
        inputs[0, 0] = np.nan
    elif fault == "range":
        inputs[0, 0] = 2
    else:
        inputs = inputs.tolist()
    with pytest.raises(ValueError):
        reader.predict(inputs)
    assert reader.predictions_attempted == 0 and reader.stopped


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "sum", "version"])
def test_should_charge_returned_bad_predictions_and_stop(case, monkeypatch, fault):
    work, _, reader = pin(case)
    output = np.full((1, 7), 1 / 7, dtype=np.float64)
    if fault == "shape":
        output = np.ones((1, 6), dtype=np.float64)
    elif fault == "dtype":
        output = output.astype(np.float32)
    elif fault == "nan":
        output[0, 0] = np.nan
    elif fault == "sum":
        output[:] = 1
    else:
        actual = PromotableActor.predict
        monkeypatch.setattr(
            PromotableActor,
            "predict",
            lambda owner, x: replace(actual(owner, x), actor_version="foreign"),
        )
    work["output"] = output
    with pytest.raises(ValueError):
        reader.predict(np.zeros((1, 296), dtype=np.float64))
    assert reader.predictions_attempted == reader.predictions_returned == 1 and reader.stopped


def test_should_recheck_prediction_callback_mutation_before_action(case):
    work, actor, reader = pin(case)
    work["on_predict"] = lambda owner: actor._slot.bundle.metadata.update(route="changed")
    with pytest.raises(ValueError, match="changed"):
        reader.predict(np.zeros((1, 296), dtype=np.float64))
    assert reader.predictions_returned == 1 and reader.actions_attempted == 0


@pytest.mark.parametrize("fault", ["predict_error", "step_error", "close_error"])
def test_should_close_literal_environment_and_keep_paid_partial_work(case, fault):
    work, build = case
    _, actor = build()
    work["no_clone"] = True
    work[fault] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        play(actor)
    report = caught.value.report
    assert report["status"] == "failed" and work["environments"][0].closed
    assert report["serving"]["stopped"]
    if fault == "predict_error":
        assert report["serving"]["predictions_attempted"] == 1
        assert report["serving"]["predictions_returned"] == 0
    elif fault == "step_error":
        assert report["serving"]["actions_attempted"] == 1
        assert report["serving"]["actions_returned"] == 0
    else:
        assert report["serving"]["actions_returned"] == 2
    assert report["work"]["failure_work_may_be_unobserved"]


@pytest.mark.parametrize("fault", ["cue", "layout", "budget", "callback_budget", "episode"])
def test_should_refuse_out_of_scope_episode_before_literal_reset(case, fault):
    work, build = case
    _, actor = build()
    options: dict[str, Any] = {}
    if fault == "cue":
        options["cue"] = "green"
    elif fault == "layout":
        options["layout"] = MiniGridKeyDoorLayout("B")
    elif fault == "budget":
        options["budget"] = ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=1), lambda: 0.0
        )
    elif fault == "callback_budget":
        original = budget()
        options.update(budget=original, check_boundary=lambda: setattr(original, "started_at", 1.0))
    else:
        options["episode_id"] = "x" * 65
    with pytest.raises(ValueError):
        play(actor, **options)
    assert not work.get("environments")


def test_should_lease_same_actor_gate_for_each_literal_physical_action(case):
    work, build = case
    _, actor = build()
    before = deepcopy(actor.serving_snapshot())

    def locked(environment):
        assert not actor._read_gate.acquire(blocking=False)
        assert environment.seed == 17 and environment.max_steps == 16
        assert environment.role == "train" and environment.episode_id == "caller-episode"

    work["on_step"] = locked
    work["no_clone"] = True
    result = play(actor)
    assert result["status"] == "complete" and result["scores"] == []
    assert result["serving"]["predictions_returned"] == result["serving"]["actions_returned"] == 2
    assert work["environments"][0].actions == [0, 0]
    assert _same_value(actor.serving_snapshot(), before)
    assert all(row.shape == (1, 296) for row in work["inputs"])


def test_should_refuse_legal_action_callback_mutation_before_physics(case):
    _, actor, reader = pin(case)

    class ChangedLegal:
        @property
        def legal_actions(self):
            actor._slot.bundle.metadata["route"] = "changed"
            return tuple(range(7))

        def step(self, action):
            pytest.fail("changed serving values admitted physical action")

    with pytest.raises(ValueError, match="changed"):
        reader.step(ChangedLegal(), 0)
    assert reader.actions_attempted == 0


def test_should_refuse_busy_binding_without_wait_or_prediction(case):
    _, actor, reader = pin(case)
    with actor._payload_exclusive(), pytest.raises(ValueError, match="busy"):
        reader.predict(np.zeros((1, 296), dtype=np.float64))
    assert reader.stopped and reader.predictions_attempted == 0


def test_should_refuse_mutation_after_public_snapshot_copy(case, monkeypatch):
    _, actor, reader = pin(case)
    actual = PromotableActor.serving_snapshot_if_idle

    def changed(owner):
        result = actual(owner)
        owner._slot.bundle.metadata["route"] = "after-copy"
        return result

    monkeypatch.setattr(PromotableActor, "serving_snapshot_if_idle", changed)
    with pytest.raises(ValueError, match="changed"):
        reader.check()
    assert reader.stopped


@pytest.mark.parametrize("fault", ["action", "predictions", "actions", "after_step"])
def test_should_refuse_bad_or_exhausted_actions_and_keep_step_mutation_charge(case, fault):
    _, actor, reader = pin(case)

    class LiteralAction:
        legal_actions = tuple(range(7))

        def step(self, action):
            actor._slot.bundle.metadata["route"] = "during-step"
            return None

    with pytest.raises(ValueError):
        if fault == "predictions":
            reader.predictions_attempted = reader.predictions_returned = 16
            reader.predict(np.zeros((1, 296), dtype=np.float64))
        elif fault == "actions":
            reader.actions_attempted = reader.actions_returned = 16
            reader.step(LiteralAction(), 0)
        else:
            reader.step(LiteralAction(), True if fault == "action" else 0)
    assert reader.stopped
    if fault == "after_step":
        assert reader.actions_attempted == reader.actions_returned == 1


def test_should_preserve_original_fork_restore_checkpoint_path(case):
    work, build = case
    source, _ = build()
    before = source.snapshot_state()
    result = evaluation._run_checkpoint(
        source,
        before,
        method="arrived_public_cue_affine_fit",
        updates=0,
        layouts=(),
        role="literal_default_regression",
        sources=(),
        budget=budget(),
        check_boundary=lambda: None,
        public_history=True,
        memory_copier=lambda state: deepcopy(state),
        state_equal=_same_value,
        live_plan=((MiniGridKeyDoorLayout("A"), "blue", "train"),),
    )
    assert result["status"] == "complete" and len(work["environments"]) == 1
    assert work["environments"][0].closed and _same_value(source.snapshot_state(), before)


def test_should_serve_real_app_publication_then_rollback_between_literal_episodes(case):
    work, build = case
    source, actor = build()
    runtime = ActorShadowRuntime(
        source,
        actor_version="actor-0",
        candidate_version="candidate-0",
        actor=actor,
        clock=LogicalClock(),
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0),
    )
    runtime.consolidate(
        "literal",
        lambda state: ConsolidatedState(
            literal_state(2), TrainingDiagnostic("literal_no_fit", 0.0)
        ),
    )

    def builder(state):
        learner = ContextLinearLearner()
        learner.restore_state(state)
        return learner

    evaluator = PromotionGuardEvaluator(
        build_learner=builder,
        utility=lambda p, y: float(np.argmax(p[0]) == 6),
        actions=lambda p: ("allow",),
        state_valid=lambda state: True,
        prediction_valid=lambda prediction: True,
        resource_bytes=lambda state: 1,
        state_digest=digest,
        clock=lambda: 0.0,
    )
    declared = PromotionPolicy(
        "literal_accuracy", "literal_bytes", 1.0, 0.5, 0.0, 2.0, 100, ("allow",)
    )
    controller = ServingPromotionController(
        actor, policy=declared, evaluator=evaluator, build_learner=builder, state_digest=digest
    )
    new = GuardBatch(
        "new", (("new", "0"),), 1, np.zeros((1, 296)), np.array([[6.0]]), "inner_guard"
    )
    old = replace(new, task_id="old", sample_keys=(("old", "0"),))
    stale = bind_promotable_minigrid(actor, episode_id="stale")
    before = play(actor, episode_id="before")
    ticket = controller.prepare(
        runtime, new, old, now=1, training_ids=frozenset(), metadata={"route": "published"}
    )
    receipt = controller.commit(runtime, ticket)
    with pytest.raises(ValueError, match="changed"):
        stale.check()
    published = play(actor, episode_id="published", layout=MiniGridKeyDoorLayout("A", branch_row=4))
    stale_published = bind_promotable_minigrid(actor, episode_id="stale-published")
    assert controller.rollback(receipt) == 2
    with pytest.raises(ValueError, match="changed"):
        stale_published.check()
    rolled_back = play(actor, episode_id="rolled-back")
    assert [item["serving"]["generation"] for item in (before, published, rolled_back)] == [0, 1, 2]
    assert [e.actions for e in work["environments"]] == [[0, 0], [6, 6], [0, 0]]
    assert all(e.closed for e in work["environments"])
    assert actor.version == "actor-0" and source.snapshot_state().completed_updates == 0
