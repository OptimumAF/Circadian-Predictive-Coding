"""Scripted native-port boundaries only; literal fixtures are not learned games."""

from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable

import numpy as np
import pytest

from src.adapters import minigrid_public_cue_recurrent_learner as port
from src.app.coupled_attribution import AttributionStream, _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed


def data(cue="blue"):
    x = np.full((2, 148), -1.0, dtype=np.float64)
    x[0, 0] = 0.2
    x[0, 1] = -0.2 if cue == "blue" else -1.0
    return x, np.array([[2.0], [0.0 if cue == "blue" else 1.0]], dtype=np.float64)


def episode(cue="blue", role="train"):
    x, y = data(cue)
    return DemonstrationEpisode(
        "literal-" + cue,
        "A",
        role,
        tuple(
            DemonstrationTransition(
                i,
                tuple(float(v) for v in x[i]),
                int(y[i, 0]),
                0.0,
                tuple(float(v) for v in x[min(i + 1, 1)]),
                i == 1,
                7,
            )
            for i in range(2)
        ),
        7,
    )


def policy():
    return port.PublicCueRecurrentPolicy(port.native.TorchRecurrentPolicy(148, 2, 7, 17))


class ScriptedNative:
    calls: list[Any] = []

    def __init__(self, policy):
        self.policy = policy
        self.state = port.native.TorchRecurrentSnapshot(
            1,
            policy,
            {"marker": np.array([0.0])},
            {"momentum": np.array([0.0])},
            {"gradient": np.array([0.0])},
            0,
            0,
        )

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def train_batch(self, x, y):
        self.calls.append(("train", x.copy(), y.copy()))
        self.state = replace(
            self.state,
            completed_updates=self.state.completed_updates + 1,
            training_examples=self.state.training_examples + len(x),
            parameters={"marker": np.array([self.state.completed_updates + 1.0])},
        )
        return TrainingDiagnostic(port.native.TORCH_RECURRENT_CE_ID, 1.0)

    def predict(self, x):
        self.calls.append(("predict", x.copy()))
        return np.full((len(x), 7), 1 / 7, dtype=np.float64)

    def start_episode(self):
        return ScriptedSession(self.state)


class ScriptedSession:
    def __init__(self, state):
        self.state = deepcopy(state)
        self.completed_steps = 0

    def snapshot_state(self):
        return deepcopy(self.state)

    def predict(self, x):
        self.completed_steps += 1
        return np.full((1, 7), 1 / 7, dtype=np.float64)


def configure_scripted_native(monkeypatch):
    ScriptedNative.calls = []

    def snapshot(state):
        if type(state) is not port.native.TorchRecurrentSnapshot:
            raise ValueError("foreign native state")
        if type(state.completed_updates) is not int or type(state.training_examples) is not int:
            raise ValueError("native counter types")
        return deepcopy(state)

    monkeypatch.setattr(port.native, "TorchRecurrentLearner", ScriptedNative)
    monkeypatch.setattr(port.native, "_snapshot", snapshot)


@pytest.fixture
def scripted(monkeypatch):
    configure_scripted_native(monkeypatch)


def test_should_route_raw_episodes_to_independent_owned_heads_and_preserve_inputs(scripted):
    owner = port.PublicCueRecurrentLearner(policy())
    for cue in ("blue", "red", "blue"):
        x, y = data(cue)
        expected = (x.copy(), y.copy())
        owner.train_batch(x, y)
        assert _same_value(ScriptedNative.calls[-1][1:], expected)
        assert _same_value((x, y), expected)
    saved = owner.snapshot_state()
    assert (saved.completed_updates, saved.training_examples) == (3, 6)
    assert [h.completed_updates for h in saved.heads] == [2, 1]
    fork = owner.fork()
    fork.train_episode(episode())
    restored = port.PublicCueRecurrentLearner(policy())
    restored.restore_state(fork.snapshot_state())
    assert (
        _same_value(restored.snapshot_state(), fork.snapshot_state())
        and owner.completed_updates == 3
    )
    saved.heads[0].parameters["marker"][:] = 99
    assert owner.snapshot_state().heads[0].parameters["marker"][0] == 2


@pytest.mark.parametrize(
    "fault", ["dtype", "width", "too_long", "missing", "ambiguous", "mixed", "labels"]
)
def test_should_refuse_bad_episode_before_native_staging(scripted, monkeypatch, fault):
    owner = port.PublicCueRecurrentLearner(policy())
    x, y = data()
    if fault == "dtype":
        x = x.astype(np.float32)
    elif fault == "width":
        x = x[:, :147]
    elif fault == "too_long":
        x = np.tile(x, (9, 1))
        y = np.tile(y, (9, 1))
    elif fault == "missing":
        x[0, :] = -1
    elif fault == "ambiguous":
        x[0, 3] = 0.2
        x[0, 4] = -0.2
    elif fault == "mixed":
        x[1] = data("red")[0][0]
    else:
        y[0, 0] = 7
    monkeypatch.setattr(port, "_stage", lambda *a: pytest.fail("native work reached"))
    with pytest.raises(ValueError):
        owner.train_batch(x, y)
    assert owner.snapshot_state().failed_work is None


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_nontrain_record_before_native_work(scripted, monkeypatch, role):
    owner = port.PublicCueRecurrentLearner(policy())
    monkeypatch.setattr(port, "_stage", lambda *a: pytest.fail("native work reached"))
    with pytest.raises(ValueError):
        owner.train_episode(episode(role=role))


def test_should_own_bounded_session_and_choose_once_from_first_public_frame(scripted):
    owner = port.PublicCueRecurrentLearner(policy())
    before = owner.snapshot_state()
    a, b = owner.start_episode(), owner.start_episode()
    assert a.completed_steps == b.completed_steps == 0
    blue = data()[0]
    red = data("red")[0]
    a.predict(blue[:1])
    b.predict(red[:1])
    assert a._selected == 0 and b._selected == 1 and a._session is not b._session
    with pytest.raises(ValueError):
        a.predict(red[:1])
    for _ in range(15):
        a.predict(blue[1:])
    assert a.completed_steps == 16 and b.completed_steps == 1
    with pytest.raises(ValueError):
        a.predict(blue[1:])
    assert _same_value(a.snapshot_state(), before) and _same_value(owner.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["replace", "counter", "weights"])
def test_should_refuse_changed_native_session_binding_or_learned_state(scripted, fault):
    owner = port.PublicCueRecurrentLearner(policy())
    session = owner.start_episode()
    session.predict(data()[0][:1])
    if fault == "replace":
        session._session = ScriptedSession(session._session.state)
    elif fault == "counter":
        session._session.completed_steps = 0
    else:
        session._session.state.parameters["marker"][0] = 33
    with pytest.raises(ValueError):
        session.snapshot_state()


@pytest.mark.parametrize("operation", ["train", "predict", "session"])
def test_should_retain_paid_or_unknown_failed_work_and_forbid_erasure(
    scripted, monkeypatch, operation
):
    target: Any
    run: Callable[[], Any]
    owner = port.PublicCueRecurrentLearner(policy())
    before = owner.snapshot_state()
    x, y = data()
    if operation == "session":
        target = owner.start_episode()

        def fail_session(self, rows):
            self.completed_steps += 1
            raise RuntimeError("lost native return")

        monkeypatch.setattr(ScriptedSession, "predict", fail_session)
        run = lambda: target.predict(x[:1])
    else:
        target = owner

        def fail_native(self, *args):
            raise RuntimeError("lost native return")

        monkeypatch.setattr(
            ScriptedNative, "train_batch" if operation == "train" else "predict", fail_native
        )
        run = lambda: owner.train_batch(x, y) if operation == "train" else owner.predict(x)
    with pytest.raises(TrainingUpdateFailed) as error:
        run()
    failed = target.snapshot_state()
    assert failed.failed_work is not None
    assert (
        failed.failed_work.native_calls_attempted == 1
        and failed.failed_work.native_calls_returned == 0
    )
    assert error.value.work["failure_work_may_be_unobserved"] is True
    assert (failed.completed_updates, failed.training_examples) == (0, 0)
    with pytest.raises(TrainingUpdateFailed):
        run()
    if operation != "session":
        with pytest.raises(ValueError):
            owner.restore_state(before)
        with pytest.raises(TrainingUpdateFailed):
            owner.fork().predict(x)
    else:
        assert _same_value(owner.snapshot_state(), before)


def test_should_block_unknown_work_even_when_failure_follows_a_return(scripted, monkeypatch):
    owner = port.PublicCueRecurrentLearner(policy())
    x, y = data()
    original = ScriptedNative.train_batch

    def mutate(self, rows, labels):
        value = original(self, rows, labels)
        y[0, 0] = 1
        return value

    monkeypatch.setattr(ScriptedNative, "train_batch", mutate)
    with pytest.raises(TrainingUpdateFailed):
        owner.train_batch(x, y)
    work = owner.snapshot_state().failed_work
    assert work is not None
    assert work.native_calls_returned == 1 and work.native_rows_returned == 2
    assert owner.completed_updates == 0


@pytest.mark.parametrize("fault", ["owner", "input"])
def test_should_refuse_mutation_during_final_prediction_readback(scripted, monkeypatch, fault):
    owner = port.PublicCueRecurrentLearner(policy())
    armed = None
    original_predict = ScriptedNative.predict
    original_snapshot = ScriptedNative.snapshot_state

    def predict(self, rows):
        nonlocal armed
        result = original_predict(self, rows)
        armed = rows
        return result

    def snapshot(self):
        nonlocal armed
        if armed is not None:
            if fault == "owner":
                owner._state = deepcopy(owner._state)
            else:
                armed[0, 0] = -1.0
            armed = None
        return original_snapshot(self)

    monkeypatch.setattr(ScriptedNative, "predict", predict)
    monkeypatch.setattr(ScriptedNative, "snapshot_state", snapshot)
    with pytest.raises(TrainingUpdateFailed):
        owner.predict(data()[0])
    failed = owner.snapshot_state()
    assert failed.failed_work is not None and failed.failed_work.native_calls_returned == 1
    assert failed.completed_updates == 0


@pytest.mark.parametrize("fault", ["policy", "heads", "failure", "extra"])
def test_should_reject_foreign_or_corrupted_snapshots_before_restore(scripted, fault):
    owner = port.PublicCueRecurrentLearner(policy())
    before = owner.snapshot_state()
    s = owner.snapshot_state()
    if fault == "policy":
        s = replace(s, policy=port.PublicCueRecurrentPolicy())
    elif fault == "heads":
        object.__setattr__(s, "heads", s.heads[:1])
    elif fault == "failure":
        s = replace(s, failed_work=port.FailureWork("train", 0, native_calls_attempted=1))
    else:
        object.__setattr__(s, "extra", 1)
    with pytest.raises(ValueError):
        owner.restore_state(s)
    assert _same_value(owner.snapshot_state(), before)


def test_should_run_existing_delayed_stream_with_frozen_source_actor_and_complete_receipts(
    scripted,
):
    from src.adapters.minigrid_learning_pilot import _batch

    source = port.PublicCueRecurrentLearner(policy())
    initial = source.snapshot_state()
    packets = tuple(_batch(episode(c)) for c in ("blue", "red"))
    stream = AttributionStream(
        source,
        packets,
        (),
        training_only=True,
        label_delay=1,
        checkpoint_updates=(0, 2),
        budget=ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=2, max_wall_seconds=30), lambda: 0.0
        ),
    )
    stream.run_until(2)
    state = stream.candidate_snapshot()
    result = stream.finish()
    assert (state.completed_updates, state.training_examples) == (2, 4)
    assert len(result["applied_updates"]) == 2 and result["score_forward_calls"] == 0
    assert _same_value(source.snapshot_state(), initial) and _same_value(
        stream._runtime.actor.snapshot_state().state, initial
    )
