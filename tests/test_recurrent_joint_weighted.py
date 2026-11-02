"""Owned joint weighted TRAIN composition using literal native/environment ports."""

from copy import deepcopy
from dataclasses import replace
import inspect
import sys

import numpy as np
import pytest

from scripts import recurrent_joint_example as example
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import torch_train_weighted_learner as weighted
from src.app.coupled_attribution import _same_value
import test_recurrent_joint_example as original
import test_public_cue_probe_games as games
import test_public_cue_weighted_train as weighted_fixture
from test_public_cue_recurrent_learner import ScriptedNative

app = original.app
scripted = games.scripted
weighted_scripted = weighted_fixture.scripted


@pytest.fixture
def wired(app, weighted_scripted, scripted, monkeypatch):
    monkeypatch.setattr(weighted, "_same_state", _same_value)

    # The old generic fixture has eight identical actions. This literal TRAIN
    # roster uses the real task's five-forward/four-critical class distribution.
    def episodes():
        actions = (2, 2, 2, 0, 3, 1, 2, 5, 2)
        return tuple(
            replace(
                ep,
                transitions=tuple(
                    replace(t, action=actions[i]) for i, t in enumerate(ep.transitions)
                ),
            )
            for ep in original.episodes()
        )

    monkeypatch.setattr(example, "_episodes", episodes)
    # Use the real existing kernel and its scripted environment, not the old
    # app fixture's synthetic metadata evaluator, for the new integration path.
    monkeypatch.setattr(
        example,
        "evaluate_public_cue_recurrent_checkpoint",
        evaluation.evaluate_public_cue_recurrent_checkpoint,
    )
    return scripted


def test_weighted_metadata_and_cli_are_explicit_and_exclusive(monkeypatch, capsys):
    monkeypatch.setattr(example, "_episodes", lambda: pytest.fail("payload"))
    monkeypatch.setattr(sys, "argv", ["joint", "--class-balanced-joint", "--describe"])
    example.main()
    assert '"joint_train_recurrent_bc128_class_balanced_v1"' in capsys.readouterr().out
    assert example.describe(class_balanced_joint=True)["row_exposures"] == 1152
    assert "class-weightedCE" in example.describe(class_balanced_joint=True)["native"]
    assert "class_balanced_training" not in example.describe()
    for mode in ("fresh_task_probe", "rehearsal_comparison", "cold_fresh_task"):
        with pytest.raises(ValueError):
            example.describe(class_balanced_joint=True, **{mode: True})
    with pytest.raises(ValueError):
        example.describe(class_balanced_joint=1)


def test_complete_joint_sequence_uses128_weighted_calls_and_six_native_games(wired):
    result = example.run_example(class_balanced_joint=True)
    assert result["status"] == "complete", result["reason"]
    assert result["training"]["completed_updates"] == 128
    assert result["training_budget"]["updates"] == 128
    assert all(result["preservation"].values()) and "adaptation" not in result
    assert [r["updates"] for r in result["readouts"]] == [0, 128]
    assert all(r["games"]["class_balanced_training"] is True for r in result["readouts"])
    episodes = result["readouts"][-1]["games"]["episodes"]
    assert [(e["context"], e["cue"]) for e in episodes] == [
        (c, q) for c in ("A", "B", "C_train") for q in ("blue", "red")
    ]
    assert all(e["source_role"] == "train" and e["state_unchanged"] for e in episodes)
    assert len(wired["environments"]) == 6 and all(e.closed for e in wired["environments"])
    calls = [c for c in ScriptedNative.calls if c[0] == "weighted"]
    assert len(calls) == 128 and sum(len(c[1].transitions) for c in calls) == 1152
    for i, (_, episode, weights) in enumerate(calls):
        assert episode.episode_id == f"joint-rnn-arrival-{i}" and episode.role == "train"
        np.testing.assert_allclose(weights, [1.8, 1.8, 0.36, 1.8, 1.0, 1.8, 1.0])


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_weighted_joint_refuses_reserved_source_roles_before_work(wired, role):
    data = list(original.episodes())
    data[0] = replace(data[0], role=role)
    result = example.run_example(tuple(data), class_balanced_joint=True)
    assert result["status"] == "incomplete"
    assert not any(c[0] == "weighted" for c in ScriptedNative.calls)


@pytest.mark.parametrize("owner", ["source", "actor", "candidate"])
def test_equal_replaced_native_leaf_refuses_all_training(wired, monkeypatch, owner):
    evaluate = example.evaluate_public_cue_recurrent_checkpoint

    def corrupt(*a, **k):
        outer = inspect.getclosurevars(k["check_boundary"]).nonlocals
        target = (
            outer["source"]
            if owner == "source"
            else outer["stream"]._runtime.actor._learner
            if owner == "actor"
            else outer["stream"]._runtime._candidate
        )
        target._learner = target._learner.fork()
        target._native_owner = target._learner  # Even both replaced aliases cannot bless it.
        return evaluate(*a, **k)

    monkeypatch.setattr(example, "evaluate_public_cue_recurrent_checkpoint", corrupt)
    result = example.run_example(class_balanced_joint=True)
    assert result["status"] == "incomplete"
    assert not any(c[0] == "weighted" for c in ScriptedNative.calls)


def test_equal_replaced_zero_readout_refuses_next_update(wired, monkeypatch):
    qualify = example._qualify

    def corrupt(point, result):
        qualify(point, result)
        if point == 0:
            current = inspect.currentframe()
            assert current is not None and current.f_back is not None
            frame = current.f_back
            frame.f_locals["report"]["readouts"][0] = deepcopy(
                frame.f_locals["report"]["readouts"][0]
            )

    monkeypatch.setattr(example, "_qualify", corrupt)
    result = example.run_example(class_balanced_joint=True)
    assert result["status"] == "incomplete" and result["training_budget"]["updates"] == 0


def test_lost_weighted_native_return_preserves_failure_and_refuses_more_updates(wired, monkeypatch):
    train = getattr(ScriptedNative, "train_weighted_episode")

    def fail(self, episode, weights):
        train(self, episode, weights)
        raise RuntimeError("scripted weighted native lost return")

    monkeypatch.setattr(ScriptedNative, "train_weighted_episode", fail)
    result = example.run_example(class_balanced_joint=True)
    assert result["status"] == "incomplete"
    assert len([c for c in ScriptedNative.calls if c[0] == "weighted"]) == 1
    assert result["training_budget"]["updates"] == 0
    assert result["failure_work"]["native_calls_attempted"] == 1
    assert result["failure_work"]["native_calls_returned"] == 0


def test_replaced_schedule_refuses_before_another_weighted_update(wired, monkeypatch):
    train = weighted.TorchTrainWeightedLearner.train_batch

    def corrupt(self, x, y):
        if self.completed_updates == 1:
            self._episodes = tuple(list(self._episodes))
        return train(self, x, y)

    monkeypatch.setattr(weighted.TorchTrainWeightedLearner, "train_batch", corrupt)
    result = example.run_example(class_balanced_joint=True)
    assert result["status"] == "incomplete"
    assert len([c for c in ScriptedNative.calls if c[0] == "weighted"]) == 1


def test_failed_native_game_keeps_all128_paid_updates(wired):
    wired["step_error"] = True
    result = example.run_example(class_balanced_joint=True)
    assert result["status"] == "incomplete"
    assert result["training_budget"]["updates"] == 128
    assert result["training"]["completed_updates"] == 128
    assert len([c for c in ScriptedNative.calls if c[0] == "weighted"]) == 128
    assert all(e.closed for e in wired["environments"])


@pytest.mark.parametrize("target", ["source", "candidate", "candidate_wrapper"])
def test_late_actor_snapshot_cannot_replace_already_validated_owner(wired, monkeypatch, target):
    evaluate = example.evaluate_public_cue_recurrent_checkpoint

    def install(*a, **k):
        outer = inspect.getclosurevars(k["check_boundary"]).nonlocals
        source = outer["source"]
        stream = outer["stream"]
        actor = stream._runtime.actor._learner
        candidate = stream._runtime._candidate
        victim = source if target == "source" else candidate
        replacement = candidate.fork() if target == "candidate_wrapper" else victim._learner.fork()
        snapshot = actor.snapshot_state
        fired: list[bool] = []

        def corrupt():
            result = snapshot()
            if not fired:
                fired.append(True)
                if target == "candidate_wrapper":
                    stream._runtime._candidate = replacement
                else:
                    victim._learner = victim._native_owner = replacement
            return result

        monkeypatch.setattr(actor, "snapshot_state", corrupt)
        return evaluate(*a, **k)

    monkeypatch.setattr(example, "evaluate_public_cue_recurrent_checkpoint", install)
    result = example.run_example(class_balanced_joint=True)
    assert result["status"] == "incomplete"
    assert len([c for c in ScriptedNative.calls if c[0] == "weighted"]) == 0
