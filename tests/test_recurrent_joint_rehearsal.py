"""Matched old-information dispatch with real probe/stream and scripted ports."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import inspect

import pytest

from scripts import recurrent_joint_example as example
from scripts import recurrent_joint_rehearsal as paired
from scripts import recurrent_joint_probe as isolated
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_probe_rollout as live
from src.adapters import minigrid_public_cue_recurrent_learner as public
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
import test_recurrent_joint_probe as old_probe
from test_recurrent_joint_example import episodes
from test_public_cue_recurrent_learner import ScriptedNative

composition = old_probe.composition
app = old_probe.app
scripted = old_probe.scripted


@pytest.fixture
def wired(composition, monkeypatch):
    monkeypatch.setattr(paired, "_collect_probe_sources", isolated._collect_probe_sources)
    return composition


def sources():
    return example.joint_packets(episodes())[0]


def test_describe_comparison_is_explicit_and_old_modes_are_unchanged():
    meta = example.describe(rehearsal_comparison=True)
    assert meta["isolated_probe"]["probe_native_updates"] == 48
    assert meta["limits"]["readout_seconds"] == 420
    assert example.describe()["limits"]["readout_seconds"] == 200
    assert example.describe(fresh_task_probe=True)["isolated_probe"]["native_endpoints"] == [
        144,
        160,
    ]
    with pytest.raises(ValueError):
        example.describe(fresh_task_probe=True, rehearsal_comparison=True)


@pytest.mark.parametrize("consent", [False, 1, "yes"])
def test_consent_refusal_precedes_any_source_or_model_work(monkeypatch, consent):
    monkeypatch.setattr(paired, "_collect_probe_sources", lambda *a, **k: pytest.fail("collection"))
    result = paired.run_comparison(
        None,
        None,
        sources(),
        check_boundary=lambda: None,
        memory=None,
        old_training_consent=consent,
    )
    assert result["status"] == "incomplete" and result["arms"] == []
    assert result["collection"]["reports"] == []


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_foreign_bank_role_refuses_before_collection(monkeypatch, role):
    data = list(sources())
    data[0] = replace(data[0], role=role, episode=replace(data[0].episode, role=role))
    monkeypatch.setattr(paired, "_collect_probe_sources", lambda *a, **k: pytest.fail("collection"))
    result = paired.run_comparison(
        None, None, tuple(data), check_boundary=lambda: None, memory=None, old_training_consent=True
    )
    assert result["status"] == "incomplete" and result["arms"] == []


@pytest.mark.parametrize("arm", paired.ARMS)
@pytest.mark.parametrize("phase,count", [("after_probe", 176), ("after_return", 192)])
def test_selected_native_reader_requires_exact_matched_counts_and_roles(
    scripted, monkeypatch, arm, phase, count
):
    owner = public.PublicCueRecurrentLearner()
    cold = owner.snapshot_state()
    h = [
        replace(head, completed_updates=count // 2, training_examples=count * 9 // 2)
        for head in cold.heads
    ]
    state = replace(cold, heads=(h[0], h[1]))
    owner.restore_state(state)
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: k)
    result = evaluation.evaluate_public_cue_probe_checkpoint(
        owner,
        state,
        source_updates=128,
        phase=phase,
        budget=ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=0, max_replay_examples=0, max_wall_seconds=120.0
            ),
            lambda: 1.0,
        ),
        check_boundary=lambda: None,
        joint_public_cue=True,
        comparison_arm=arm,
    )
    assert result["updates"] == count and len(result["live_plan"]) == 8
    assert result["return_metadata"]["logical_probe_updates"] == (48 if arm == "extra_F" else 16)
    with pytest.raises(ValueError):
        evaluation._copy_public_cue_probe_snapshot(
            state, source_updates=128, phase=phase, joint_public_cue=True
        )  # default144/160 stays strict


def test_matched_mode_requires_explicit_arm_and_joint_source():
    with pytest.raises(ValueError):
        live.ProbeRollouts(2, public_cue=True, comparison_arm="extra_F")
    with pytest.raises(ValueError):
        live.ProbeRollouts(128, public_cue=True, joint_public_cue=True, comparison_arm="favorable")


def test_baseline420_is_only_for_explicit_comparison(scripted, monkeypatch):
    owner = public.PublicCueRecurrentLearner()
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: k)

    def read(flag):
        return evaluation.evaluate_public_cue_recurrent_checkpoint(
            owner,
            owner.snapshot_state(),
            updates=0,
            sources=sources(),
            budget=ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=0,
                    max_replay_examples=0,
                    max_wall_seconds=420,
                    max_process_rss_bytes=768 * 1024**2,
                ),
                lambda: 1.0,
            ),
            check_boundary=lambda: None,
            recurring_training=True,
            joint_comparison=flag,
        )

    assert read(True)["updates"] == 0
    with pytest.raises(ValueError):
        read(False)


def test_fixed_bank_uses_original_six_TRAIN_episodes_and_exact_storage(scripted):
    owner = public.PublicCueRecurrentLearner()
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=16), lambda: 0.0)
    replay = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=288), lambda: 0.0)
    helper = paired.build_rehearsal(owner, wake, replay, sources(), consent=True)
    report = helper.report
    assert report["fixed_bank"]["native_transition_count"] == 54
    assert report["fixed_bank"]["encoded_array_bytes"] == paired.BANK_BYTES == 133968
    assert report["coverage_keys"] == list(paired.KEYS)
    assert wake.updates_completed == replay.replay_examples_completed == 0


def test_real_comparison_matches_native_exposure_without_contaminating_benchmark(wired):
    result = example.run_example(rehearsal_comparison=True)
    assert result["status"] == "complete", result["reason"]
    assert result["training"]["completed_updates"] == 128
    assert all(result["preservation"].values())
    arms = result["adaptation"]["arms"]
    assert [a["arm"] for a in arms] == list(paired.ARMS)
    assert [a["probe_budget"]["updates"] for a in arms] == [48, 16]
    assert [a["return_budget"]["updates"] for a in arms] == [16, 16]
    assert arms[0]["replay_budget"] is None
    assert arms[1]["replay_budget"]["replay"] == 288
    assert arms[1]["replay"]["committed_updates"] == 32
    counts = Counter(
        k for event in arms[1]["replay"]["events"] for k in event["coverage_selected_keys"]
    )
    assert counts == dict(zip(paired.KEYS, [6, 6, 5, 5, 5, 5], strict=True))
    assert all(
        [e["report"]["native_training_updates"] for e in arm["endpoints"]] == [176, 192]
        for arm in arms
    )
    assert all(len(e["report"]["episodes"]) == 8 for a in arms for e in a["endpoints"])
    assert len(wired["environments"]) == 32 and all(e.closed for e in wired["environments"])
    assert len([c for c in ScriptedNative.calls if c[0] == "train"]) == 256


def test_replay_lost_return_keeps_first_arm_and_every_permanent_charge(wired, monkeypatch):
    build = paired.build_rehearsal
    train = ScriptedNative.train_batch
    calls = []

    def mark(*a, **k):
        result = build(*a, **k)
        calls.append(0)
        return result

    def fail(self, x, y):
        diagnostic = train(self, x, y)
        if calls:
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError("scripted second replay lost return")
        return diagnostic

    monkeypatch.setattr(paired, "build_rehearsal", mark)
    monkeypatch.setattr(ScriptedNative, "train_batch", fail)
    result = example.run_example(rehearsal_comparison=True)
    assert result["status"] == "incomplete"
    arms = result["adaptation"]["arms"]
    assert arms[0]["status"] == "complete"
    assert arms[1]["probe_budget"]["updates"] == 1
    assert arms[1]["return_budget"]["updates"] == 0
    assert arms[1]["replay_budget"]["replay"] == 18
    assert arms[1]["replay"]["attempted_updates"] == 2
    assert arms[1]["replay"]["returned_updates"] == 1


def test_equal_replaced_prior_arm_refuses_all_second_arm_updates(wired, monkeypatch):
    run = paired.run_fresh_task_probe
    reader = paired.ProbeRollouts
    seen = []

    def observe(*a, **k):
        seen.append(inspect.getclosurevars(k["check_boundary"]).nonlocals["report"])
        return run(*a, **k)

    def corrupt(*a, **k):
        if k["comparison_arm"] == "old_ABC":
            seen[0]["arms"][0] = deepcopy(seen[0]["arms"][0])
        return reader(*a, **k)

    monkeypatch.setattr(paired, "run_fresh_task_probe", observe)
    monkeypatch.setattr(paired, "ProbeRollouts", corrupt)
    result = example.run_example(rehearsal_comparison=True)
    assert result["status"] == "incomplete"
    assert len([c for c in ScriptedNative.calls if c[0] == "train"]) == 192
