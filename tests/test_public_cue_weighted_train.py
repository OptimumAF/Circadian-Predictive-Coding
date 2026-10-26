"""Scripted class-balanced public-cue admission; no learned/game evidence."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.adapters import minigrid_public_cue_recurrent_learner as port
from src.adapters.torch_recurrent_weighted import TORCH_RECURRENT_WEIGHTED_CE_ID
from src.app.coupled_attribution import _same_value
from src.core.demonstration_class_weights import episode_class_weights
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    episode_arrays,
)
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from test_public_cue_recurrent_learner import (
    ScriptedNative,
    configure_scripted_native,
    data,
    policy,
)


def packet(cue="blue", role="train"):
    first, _ = data(cue)
    rows = np.concatenate((first[:1], np.repeat(first[1:], 3, axis=0)))
    actions = (2, 2, 2, 0 if cue == "blue" else 1)
    return DemonstrationEpisode(
        "balanced-" + cue,
        "A",
        role,
        tuple(
            DemonstrationTransition(
                i,
                tuple(float(v) for v in row),
                actions[i],
                0.0,
                tuple(float(v) for v in rows[min(i + 1, 3)]),
                i == 3,
                7,
            )
            for i, row in enumerate(rows)
        ),
        7,
    )


@pytest.fixture
def scripted(monkeypatch):
    configure_scripted_native(monkeypatch)

    def weighted(self, episode, weights):
        self.train_batch(*episode_arrays(episode))
        self.state = replace(
            self.state,
            optimizer_state={"momentum": self.state.optimizer_state["momentum"] + 2},
            gradients={"gradient": self.state.gradients["gradient"] + 3},
        )
        self.calls.append(("weighted", deepcopy(episode), weights))
        return TrainingDiagnostic(TORCH_RECURRENT_WEIGHTED_CE_ID, 0.5)

    monkeypatch.setattr(ScriptedNative, "train_weighted_episode", weighted, raising=False)


def test_should_route_fixed_weights_and_continue_both_full_heads_without_changing_source(scripted):
    source = port.PublicCueRecurrentLearner(policy())
    initial = source.snapshot_state()
    owner, actor = source.fork(), source.fork()
    for cue in ("blue", "red", "blue"):
        p = packet(cue)
        expected = deepcopy(p)
        before = owner.snapshot_state()
        selected = int(cue == "red")
        diagnostic = owner.train_class_balanced_episode(p)
        assert diagnostic.definition == TORCH_RECURRENT_WEIGHTED_CE_ID
        call = ScriptedNative.calls[-1]
        assert call[0] == "weighted" and _same_value(call[1], expected)
        assert call[2] == episode_class_weights(p)
        assert call[2][2] == 2 / 3 and call[2][0 if selected == 0 else 1] == 2.0
        state = owner.snapshot_state()
        assert _same_value(state.heads[1 - selected], before.heads[1 - selected])
        head = state.heads[selected]
        assert head.completed_updates == before.heads[selected].completed_updates + 1
        assert head.training_examples == 4 * head.completed_updates
        assert head.optimizer_state["momentum"][0] == 2 * head.completed_updates
        assert head.gradients["gradient"][0] == 3 * head.completed_updates
        assert _same_value(p, expected)
    assert (owner.completed_updates, owner.training_examples) == (3, 12)
    assert [h.completed_updates for h in owner.snapshot_state().heads] == [2, 1]
    assert _same_value(source.snapshot_state(), initial) and _same_value(
        actor.snapshot_state(), initial
    )
    restored = source.fork()
    restored.restore_state(owner.snapshot_state())
    assert _same_value(restored.snapshot_state(), owner.snapshot_state())


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_other_roles_before_any_native_staging(scripted, monkeypatch, role):
    owner = port.PublicCueRecurrentLearner(policy())
    before = owner.snapshot_state()
    monkeypatch.setattr(port, "_stage", lambda *a: pytest.fail("native work admitted"))
    with pytest.raises(ValueError):
        owner.train_class_balanced_episode(packet(role=role))
    assert _same_value(owner.snapshot_state(), before)


@pytest.mark.parametrize(
    "fault", ["extra", "transition_extra", "classes", "missing_cue", "mixed", "action", "partial"]
)
def test_should_refuse_bad_complete_packet_before_native_work(scripted, monkeypatch, fault):
    owner = port.PublicCueRecurrentLearner(policy())
    p = packet()
    if fault == "extra":
        object.__setattr__(p, "extra", 1)
    elif fault == "transition_extra":
        object.__setattr__(p.transitions[0], "extra", 1)
    elif fault == "classes":
        object.__setattr__(p, "num_classes", 6)
    elif fault == "missing_cue":
        blank = tuple(-1.0 for _ in range(148))
        object.__setattr__(p.transitions[0], "observation", blank)
    elif fault == "mixed":
        red = tuple(float(v) for v in data("red")[0][0])
        object.__setattr__(p.transitions[0], "next_observation", red)
        object.__setattr__(p.transitions[1], "observation", red)
    elif fault == "action":
        object.__setattr__(p.transitions[0], "action", 7)
    else:
        object.__setattr__(p.transitions[-1], "terminal", False)
    monkeypatch.setattr(port, "_stage", lambda *a: pytest.fail("native work admitted"))
    with pytest.raises(ValueError):
        owner.train_class_balanced_episode(p)
    assert owner.completed_updates == 0 and owner.snapshot_state().failed_work is None


@pytest.mark.parametrize("fault", ["lost", "definition", "counter", "momentum", "packet", "owner"])
def test_should_keep_paid_failed_prefix_and_refuse_further_work(scripted, monkeypatch, fault):
    owner = port.PublicCueRecurrentLearner(policy())
    owner.train_class_balanced_episode(packet("red"))
    before = owner.snapshot_state()
    original = getattr(ScriptedNative, "train_weighted_episode")

    def failed(self, p, weights):
        result = original(self, p, weights)
        if fault == "lost":
            raise RuntimeError("lost actual update return")
        if fault == "definition":
            return TrainingDiagnostic(port.native.TORCH_RECURRENT_CE_ID, 0.5)
        if fault == "counter":
            self.state = replace(self.state, training_examples=self.state.training_examples + 1)
        elif fault == "momentum":
            owner._state.heads[1].optimizer_state["momentum"][0] += 1
        elif fault == "packet":
            object.__setattr__(p.transitions[0], "action", 1)
        elif fault == "owner":
            owner._state = deepcopy(owner._state)
        return result

    monkeypatch.setattr(ScriptedNative, "train_weighted_episode", failed)
    with pytest.raises(TrainingUpdateFailed):
        owner.train_class_balanced_episode(packet())
    state = owner.snapshot_state()
    assert state.failed_work is not None and state.failed_work.operation == "train"
    assert _same_value(state.heads, before.heads)
    assert state.failed_work.native_calls_attempted == 1
    assert state.failed_work.native_calls_returned == int(fault != "lost")
    assert state.failed_work.native_rows_attempted == 4
    assert state.failed_work.failure_work_may_be_unobserved is True
    with pytest.raises(TrainingUpdateFailed):
        owner.train_class_balanced_episode(packet())
    with pytest.raises(ValueError):
        owner.restore_state(before)


def test_should_refuse_original_packet_mutation_during_staging_before_native_call(
    scripted, monkeypatch
):
    owner = port.PublicCueRecurrentLearner(policy())
    p = packet()
    stage = port._stage

    def changed(*args):
        learner = stage(*args)
        object.__setattr__(p, "role", "probe_train")
        return learner

    monkeypatch.setattr(port, "_stage", changed)
    with pytest.raises(TrainingUpdateFailed):
        owner.train_class_balanced_episode(p)
    work = owner.snapshot_state().failed_work
    assert work is not None and work.native_stages_returned == 1
    assert work.native_calls_attempted == 0


def test_should_keep_mean_ce_default_and_weighted_then_mean_continuation(scripted):
    owner = port.PublicCueRecurrentLearner(policy())
    assert owner.train_episode(packet()).definition == port.native.TORCH_RECURRENT_CE_ID
    assert owner.train_class_balanced_episode(packet()).definition == TORCH_RECURRENT_WEIGHTED_CE_ID
    assert (
        owner.train_batch(*episode_arrays(packet())).definition == port.native.TORCH_RECURRENT_CE_ID
    )
    head = owner.snapshot_state().heads[0]
    assert head.completed_updates == 3 and head.training_examples == 12
    assert head.optimizer_state["momentum"][0] == 2
    assert ScriptedNative.calls[-1][0] == "train"
