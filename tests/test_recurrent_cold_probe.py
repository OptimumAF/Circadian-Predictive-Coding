"""Cold F48 composition and exact native mode; literal ports, no native runtime."""

from copy import deepcopy
from dataclasses import replace
import inspect
import sys

import pytest

from scripts import recurrent_joint_example as example
from scripts import recurrent_joint_rehearsal as helper
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_probe_rollout as live
from src.adapters import minigrid_public_cue_recurrent_learner as public
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learner_ports import TrainingUpdateFailed
import test_recurrent_joint_rehearsal as paired
from test_public_cue_recurrent_learner import ScriptedNative

composition = paired.composition
app = paired.app
scripted = paired.scripted


@pytest.fixture
def wired(composition, monkeypatch):
    monkeypatch.setattr(helper, "_collect_probe_sources", paired.isolated._collect_probe_sources)
    monkeypatch.setattr(helper, "_same_state", example._same_state)
    return composition


def test_cold_metadata_and_cli_require_explicit_exclusive_mode(monkeypatch, capsys):
    monkeypatch.setattr(example, "_episodes", lambda: pytest.fail("source payload"))
    monkeypatch.setattr(sys, "argv", ["joint", "--cold-fresh-task-probe", "--describe"])
    example.main()
    assert '"cold0_F48_A16_v1"' in capsys.readouterr().out
    meta = example.describe(cold_fresh_task=True)
    assert meta["row_exposures"] == 0 and meta["readout_points"] == [0]
    assert meta["isolated_probe"]["native_endpoints"] == [48, 64]
    for options in (
        dict(cold_fresh_task=1),
        dict(cold_fresh_task=True, fresh_task_probe=True),
        dict(cold_fresh_task=True, rehearsal_comparison=True),
    ):
        with pytest.raises(ValueError):
            example.describe(**options)
    assert example.describe()["row_exposures"] == 1152


@pytest.mark.parametrize("phase,count", [("after_probe", 48), ("after_return", 64)])
def test_native_reader_admits_only_exact_cold_paid_state(scripted, monkeypatch, phase, count):
    owner = public.PublicCueRecurrentLearner()
    cold = owner.snapshot_state()
    heads = [
        replace(h, completed_updates=count // 2, training_examples=9 * count // 2)
        for h in cold.heads
    ]
    state = replace(cold, heads=(heads[0], heads[1]))
    owner.restore_state(state)
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: k)
    result = evaluation.evaluate_public_cue_probe_checkpoint(
        owner,
        state,
        source_updates=0,
        phase=phase,
        budget=ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=0, max_replay_examples=0, max_wall_seconds=120.0
            ),
            lambda: 1.0,
        ),
        check_boundary=lambda: None,
        joint_public_cue=True,
        comparison_arm="cold_F",
    )
    assert result["updates"] == count and len(result["live_plan"]) == 8
    assert result["return_metadata"]["logical_probe_updates"] == 48
    for origin, arm in ((2, "cold_F"), (128, "cold_F"), (0, "extra_F"), (0, "old_ABC"), (0, None)):
        with pytest.raises(ValueError):
            evaluation._copy_public_cue_probe_snapshot(
                state, source_updates=origin, phase=phase, joint_public_cue=True, comparison_arm=arm
            )
    wrong = replace(state, heads=(replace(state.heads[0], training_examples=1), state.heads[1]))
    with pytest.raises(ValueError):
        evaluation._copy_public_cue_probe_snapshot(
            wrong, source_updates=0, phase=phase, joint_public_cue=True, comparison_arm="cold_F"
        )


@pytest.mark.parametrize("origin", [2, 128])
def test_cold_observer_refuses_other_source_labels(origin):
    with pytest.raises(ValueError):
        live.ProbeRollouts(origin, public_cue=True, joint_public_cue=True, comparison_arm="cold_F")


@pytest.mark.parametrize("flag", [1, "yes"])
def test_cold_helper_refuses_nonboolean_mode_before_collection(flag):
    result = helper.run_comparison(
        None, None, None, check_boundary=lambda: None, memory=None, cold_source=flag
    )
    assert result["status"] == "incomplete" and result["arms"] == []


def test_warm_snapshot_cannot_be_relabelled_as_cold(scripted, monkeypatch):
    monkeypatch.setattr(helper, "_collect_probe_sources", lambda *a, **k: pytest.fail("collection"))
    source = public.PublicCueRecurrentLearner()
    cold = source.snapshot_state()
    heads = [replace(h, completed_updates=64, training_examples=576) for h in cold.heads]
    warm = replace(cold, heads=(heads[0], heads[1]))
    result = helper.run_comparison(
        source, warm, paired.sources(), check_boundary=lambda: None, memory=None, cold_source=True
    )
    assert result["status"] == "incomplete" and result["collection"]["reports"] == []


def test_full_cold_sequence_uses_only64_private_updates_and_preserves_benchmark(wired):
    result = example.run_example(cold_fresh_task=True)
    assert result["status"] == "complete", result["reason"]
    assert result["training"] == dict(status="complete", completed_updates=0, applied_updates=[])
    assert result["training_budget"]["limits"]["max_training_updates"] == 0
    assert result["training_budget"]["updates"] == 0
    assert all(result["preservation"].values())
    arms = result["adaptation"]["arms"]
    assert len(arms) == 1 and arms[0]["arm"] == "cold_F"
    arm = arms[0]
    assert arm["probe_budget"]["updates"] == 48 and arm["return_budget"]["updates"] == 16
    assert arm["replay"] is None and arm["replay_budget"] is None
    assert [e["report"]["native_training_updates"] for e in arm["endpoints"]] == [48, 64]
    assert [e["report"]["native_head_updates"] for e in arm["endpoints"]] == [[24, 24], [32, 32]]
    assert all(
        [e["source_role"] for e in r["report"]["episodes"]] == ["probe_test"] * 2 + ["train"] * 6
        for r in arm["endpoints"]
    )
    assert len(wired["environments"]) == 16 and all(e.closed for e in wired["environments"])
    assert len([c for c in ScriptedNative.calls if c[0] == "train"]) == 64


def test_failed_F_game_retains48_charges_and_admits_no_A(wired):
    wired["step_error"] = True
    result = example.run_example(cold_fresh_task=True)
    assert result["status"] == "incomplete"
    arm = result["adaptation"]["arms"][0]
    assert arm["probe_budget"]["updates"] == 48 and arm["return_budget"]["updates"] == 0
    assert len([c for c in ScriptedNative.calls if c[0] == "train"]) == 48
    assert all(e.closed for e in wired["environments"])


def test_failed_F_update_keeps_permanent_native_failure_record(wired, monkeypatch):
    train = ScriptedNative.train_batch
    owned = []
    public_train = public.PublicCueRecurrentLearner.train_batch

    def capture(self, x, y):
        owned.append((self, x, y))
        return public_train(self, x, y)

    monkeypatch.setattr(public.PublicCueRecurrentLearner, "train_batch", capture)

    def fail(self, x, y):
        train(self, x, y)
        raise RuntimeError("scripted F lost native return")

    monkeypatch.setattr(ScriptedNative, "train_batch", fail)
    result = example.run_example(cold_fresh_task=True)
    assert result["status"] == "incomplete"
    arm = result["adaptation"]["arms"][0]
    # The existing budget counts committed returns, not failed native attempts.
    # Failed work stays separately recorded and permanently poisons this owner.
    assert arm["probe_budget"]["updates"] == 0 and arm["return_budget"]["updates"] == 0
    assert arm["probe"]["attempted_updates"] == 1
    assert arm["probe"]["completed_updates"] == 0
    work = arm["probe"]["failed_native_work"]["work"]
    assert (work["native_calls_attempted"], work["native_calls_returned"]) == (1, 0)
    assert (work["native_rows_attempted"], work["native_rows_returned"]) == (9, 0)
    assert work["failure_work_may_be_unobserved"] is True
    owner, x, y = owned[0]
    assert owner.snapshot_state().failed_work is not None
    with pytest.raises(TrainingUpdateFailed):
        public_train(owner, x, y)
    assert len([c for c in ScriptedNative.calls if c[0] == "train"]) == 1


@pytest.mark.parametrize("fault", ["closed", "proof", "replacement"])
def test_corrupt_or_replaced_F_evidence_refuses_all_A_updates(wired, monkeypatch, fault):
    original = helper.ProbeRollouts

    def games(*a, **k):
        reader = original(*a, **k)
        observe = reader.observe_qualified

        def corrupt(*args):
            observe(*args)
            row = reader.records[0]
            if fault == "replacement":
                row["report"] = deepcopy(row["report"])
            elif fault == "proof":
                row["report"]["episodes"][0]["state_unchanged"] = False
            else:
                row["report"]["episodes"][0]["environment"]["closed"] = False

        monkeypatch.setattr(reader, "observe_qualified", corrupt)
        return reader

    monkeypatch.setattr(helper, "ProbeRollouts", games)
    result = example.run_example(cold_fresh_task=True)
    assert result["status"] == "incomplete"
    arm = result["adaptation"]["arms"][0]
    assert arm["probe_budget"]["updates"] == 48 and arm["return_budget"]["updates"] == 0


def test_cold_benchmark_mutation_refuses_collection_and_all_updates(wired, monkeypatch):
    def corrupt(check, reports, **kwargs):
        outer = inspect.getclosurevars(check).nonlocals["check_boundary"]
        candidate = inspect.getclosurevars(outer).nonlocals["candidate"]
        candidate._state.heads[0].parameters["marker"][0] += 1
        check()

    monkeypatch.setattr(helper, "_collect_probe_sources", corrupt)
    result = example.run_example(cold_fresh_task=True)
    assert result["status"] == "incomplete" and "benchmark" in result["reason"]
    assert len([c for c in ScriptedNative.calls if c[0] == "train"]) == 0
