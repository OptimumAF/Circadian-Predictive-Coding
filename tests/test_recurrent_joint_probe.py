"""Literal new-task/return composition; genuine models and physics are forbidden."""

from copy import deepcopy
from dataclasses import replace
import inspect
from typing import Any

import pytest

from scripts import recurrent_joint_example as example
from scripts import recurrent_joint_probe as helper
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_probe_rollout as live
from src.adapters import minigrid_public_cue_recurrent_learner as port
from src.adapters.minigrid_learnability import _packet
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.coupled_attribution import _same_value
from src.app.fresh_task_probe import run_fresh_task_probe
import test_public_cue_probe_games as old_games
from test_public_cue_recurrent_learner import ScriptedNative
import test_recurrent_joint_example as old_joint
from test_recurrent_readout import nine

scripted = old_games.scripted
app = old_joint.app


def data():
    def packet(role, cue, i):
        return _packet(
            replace(nine(cue, "F-" + cue), role=role, episode_id=f"literal-F-{role}-{cue}-{i}")
        )

    return (
        tuple(packet("probe_train", ("blue", "red")[i % 2], i) for i in range(16)),
        tuple(packet("probe_test", cue, i) for i, cue in enumerate(("blue", "red"))),
    )


@pytest.fixture
def composition(app, scripted, monkeypatch):
    def collect(check, reports, **kwargs):
        assert kwargs == dict(distinct_probe=True)
        check()
        reports.extend(
            dict(role=role, cue=cue, status="complete", closed=True, environment_steps=9)
            for role in ("probe_train", "probe_test")
            for cue in ("blue", "red")
        )
        check()
        return data()

    monkeypatch.setattr(helper, "_collect_probe_sources", collect)
    return scripted


def saved(owner, phase="after_probe"):
    count = 72 if phase == "after_probe" else 80
    cold = owner.snapshot_state()
    return replace(
        cold,
        heads=tuple(
            replace(h, completed_updates=count, training_examples=9 * count) for h in cold.heads
        ),
    )


@pytest.mark.parametrize("phase,count", [("after_probe", 144), ("after_return", 160)])
def test_explicit_joint_reader_admits_exact_state_and_eight_separate_roles(
    scripted, monkeypatch, phase, count
):
    source = port.PublicCueRecurrentLearner()
    state = saved(source, phase)
    source.restore_state(state)
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *args, **kwargs: kwargs)
    result = evaluation.evaluate_public_cue_probe_checkpoint(
        source,
        state,
        source_updates=128,
        phase=phase,
        joint_public_cue=True,
        budget=ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=0, max_replay_examples=0, max_wall_seconds=80.0
            ),
            lambda: 1.0,
        ),
        check_boundary=lambda: None,
    )
    assert result["updates"] == count and result["sources"] == ()
    assert result["role"] == "probe_test_and_train"
    assert [(p.context, cue, role) for p, cue, role in result["live_plan"]] == [
        ("F-blue", "blue", "probe_test"),
        ("F-red", "red", "probe_test"),
        *[(c, q, "train") for c in ("A", "B", "C_train") for q in ("blue", "red")],
    ]
    assert result["memory_copier"](state).completed_updates == count
    changed = deepcopy(state)
    changed.heads[0].parameters["marker"][0] += 1
    with pytest.raises(ValueError):
        result["memory_copier"](changed)


@pytest.mark.parametrize("fault", ["flag", "public", "origin", "zero", "bool", "wide"])
def test_joint_mode_invalid_options_refuse_before_work(fault):
    opts: dict[str, Any] = dict(public_cue=True, joint_public_cue=True)
    origin = 128
    if fault == "flag":
        opts["joint_public_cue"] = 1
    if fault == "public":
        opts["public_cue"] = False
    if fault == "origin":
        origin = 2
    if fault == "zero":
        origin = 0
    if fault == "bool":
        origin = True
    if fault == "wide":
        opts["wide_method"] = "bp"
    with pytest.raises(ValueError):
        live.ProbeRollouts(origin, **opts)


@pytest.mark.parametrize("fault", ["source2", "rows", "heads", "failed"])
def test_joint_snapshot_refuses_unpaid_clocks_and_failure(scripted, fault):
    owner = port.PublicCueRecurrentLearner()
    state = saved(owner)
    origin = 128
    if fault == "source2":
        origin = 2
    if fault == "rows":
        state = replace(
            state, heads=(replace(state.heads[0], training_examples=647), state.heads[1])
        )
    if fault == "heads":
        state = replace(
            state, heads=(replace(state.heads[0], completed_updates=71), state.heads[1])
        )
    if fault == "failed":
        state = replace(state, failed_work=object())
    with pytest.raises((TypeError, ValueError)):
        evaluation._copy_public_cue_probe_snapshot(
            state, source_updates=origin, phase="after_probe", joint_public_cue=True
        )


def test_real_joint_stream_and_disposable_F_A_keep_benchmark_and_all_old_tasks(composition):
    result = example.run_example(fresh_task_probe=True)
    assert result["status"] == "complete", result["reason"]
    assert result["training"]["completed_updates"] == 128
    assert all(result["preservation"].values())
    adaptation = result["adaptation"]
    assert adaptation["status"] == adaptation["probe"]["status"] == "complete"
    assert adaptation["probe_budget"]["updates"] == adaptation["return_budget"]["updates"] == 16
    assert adaptation["probe"]["return_phase"]["completed_updates"] == 16
    assert [r["report"]["native_training_updates"] for r in adaptation["endpoints"]] == [144, 160]
    assert [r["report"]["native_head_updates"] for r in adaptation["endpoints"]] == [
        [72, 72],
        [80, 80],
    ]
    assert all(len(r["report"]["episodes"]) == 8 for r in adaptation["endpoints"])
    assert len(composition["environments"]) == 16 and all(
        e.closed for e in composition["environments"]
    )
    assert len([c for c in ScriptedNative.calls if c[0] == "train"]) == 160
    assert [p["updates"] for p in adaptation["probe"]["checkpoints"]] == [0, 4, 8, 16]


def test_failed_F_game_keeps_paid_F_and_refuses_all_A_return(composition):
    composition["step_error"] = True
    result = example.run_example(fresh_task_probe=True)
    assert result["status"] == "incomplete"
    paid = result["adaptation"]
    assert paid["probe_budget"]["updates"] == 16 and paid["return_budget"]["updates"] == 0
    assert len(paid["endpoints"]) == 1 and paid["endpoints"][0]["status"] == "failed"
    assert len([c for c in ScriptedNative.calls if c[0] == "train"]) == 144
    assert all(e.closed for e in composition["environments"])


def test_mutated_benchmark_refuses_collection_and_probe_updates(composition, monkeypatch):
    def corrupt(check, reports, **kwargs):
        outer = inspect.getclosurevars(check).nonlocals["check_boundary"]
        candidate = inspect.getclosurevars(outer).nonlocals["candidate"]
        candidate._state.heads[0].parameters["marker"][0] += 1
        check()
        pytest.fail("corrupted benchmark admitted")

    monkeypatch.setattr(helper, "_collect_probe_sources", corrupt)
    result = example.run_example(fresh_task_probe=True)
    assert result["status"] == "incomplete" and "benchmark" in result["reason"]
    assert result["adaptation"]["probe_budget"] is None
    assert len([c for c in ScriptedNative.calls if c[0] == "train"]) == 128


@pytest.mark.parametrize("fault", ["closed", "steps"])
def test_incomplete_source_collection_refuses_probe_fits(composition, monkeypatch, fault):
    collect = helper._collect_probe_sources

    def invalid(*args, **kwargs):
        result = collect(*args, **kwargs)
        args[1][0]["closed" if fault == "closed" else "environment_steps"] = False
        return result

    monkeypatch.setattr(helper, "_collect_probe_sources", invalid)
    result = example.run_example(fresh_task_probe=True)
    assert result["status"] == "incomplete" and result["adaptation"]["probe_budget"] is None


def test_equal_replaced_successful_F_endpoint_refuses_return(composition, monkeypatch):
    original = helper.ProbeRollouts

    def games(*args, **kwargs):
        reader = original(*args, **kwargs)
        observe = reader.observe_qualified

        def replace_paid(*a):
            observe(*a)
            reader.records[0]["report"] = deepcopy(reader.records[0]["report"])

        monkeypatch.setattr(reader, "observe_qualified", replace_paid)
        return reader

    monkeypatch.setattr(helper, "ProbeRollouts", games)
    result = example.run_example(fresh_task_probe=True)
    assert result["status"] == "incomplete"
    assert result["adaptation"]["probe_budget"]["updates"] == 16
    assert result["adaptation"]["return_budget"]["updates"] == 0


def test_describe_selected_probe_is_metadata_only_and_default_stays_joint(monkeypatch):
    monkeypatch.setattr(helper, "_collect_probe_sources", lambda *a, **k: pytest.fail("payload"))
    assert example.describe()["definition"] == "joint_train_recurrent_bc128_v1"
    selected = example.describe(fresh_task_probe=True)
    assert selected["isolated_probe"]["native_endpoints"] == [144, 160]
    with pytest.raises(ValueError):
        example.describe(fresh_task_probe=1)


@pytest.mark.parametrize("fault", ["proof", "success", "return", "clock"])
def test_contradictory_F_game_evidence_refuses_every_A_update(scripted, monkeypatch, fault):
    source = port.PublicCueRecurrentLearner()
    cold = source.snapshot_state()
    heads = [replace(h, completed_updates=64, training_examples=576) for h in cold.heads]
    state = replace(cold, heads=(heads[0], heads[1]))
    training, evaluation_batches = data()
    monitors = example.joint_packets(old_joint.episodes())[0][:2]
    templates = helper.return_training_sources(monitors)
    original = live.evaluate_public_cue_probe_checkpoint

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        first = result["episodes"][0]
        if fault == "proof":
            first["state_unchanged"] = False
        if fault == "success":
            first["success"] = 0
        if fault == "return":
            first["return"] = float("nan")
        if fault == "clock":
            result["native_training_updates"] = 143
        return result

    monkeypatch.setattr(live, "evaluate_public_cue_probe_checkpoint", corrupt)
    reader = live.ProbeRollouts(128, public_cue=True, joint_public_cue=True)

    def budget():
        return ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=16, max_replay_examples=0, max_wall_seconds=30),
            lambda: 1.0,
        )

    f_budget, a_budget = budget(), budget()

    def guard():
        assert _same_value(source.snapshot_state(), cold)
        reader.check()

    result = run_fresh_task_probe(
        source,
        state,
        training,
        evaluation_batches,
        budget=f_budget,
        checkpoint_updates=(0, 4, 8, 16),
        training_readouts=monitors,
        return_training=templates,
        return_budget=a_budget,
        endpoint_observer=reader.observe_qualified,
        check_boundary=guard,
        timer=lambda: 1.0,
    )
    assert result["status"] == "failed"
    assert f_budget.updates_completed == 16 and a_budget.updates_completed == 0
    assert reader.records[0]["status"] == "failed"
