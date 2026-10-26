"""H32 disposable endpoint games through the ordinary isolated F16/A16 path.

Only the public BP case trains:192 updates/7488 rows,94 fixed score calls/846
rows and sixteen actual games (at most256 prediction calls/steps). Observer
fixtures declare structurally valid mixed clocks without executing training;
they use fake game receipts, no native predictions or actual physics.
"""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_probe_rollout as live
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    copy_invariant_torch_feature_snapshot,
)
from src.shared.torch_runtime import require_torch
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import episode
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS, METHODS
from test_minigrid_wide_evaluation import declared_endpoint, fixed_policy
from test_minigrid_wide_learnability import EXCLUDED
from test_torch_feature_learner import assert_tree


WIDE_FLAGS = {**FLAGS, "joint_catalogue": True, "wide_neural": True}
GAME_FLAGS = {**WIDE_FLAGS, "probe_rollouts": True}


def forbidden(*args, **kwargs):
    raise AssertionError("admission must precede resource construction or teacher collection")


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)
    monkeypatch.setattr(pilot, "_collect_probe_sources", forbidden)


@pytest.mark.parametrize("method", METHODS)
def test_should_describe_fixed_h32_games_without_changing_disabled_recipe(no_resources, method):
    options = {**WIDE_FLAGS, "qualification_method": method}
    original = pilot.describe_protocol(**options)
    assert original == pilot.describe_protocol(**options, probe_rollouts=False)
    selected = pilot.describe_protocol(**options, probe_rollouts=True)
    assert selected["methods"] == original["methods"] == [f"public_invariant_{method}"]
    for key in ("seed", "hidden_dim", "learning_rate", "momentum", "checkpoint_updates"):
        assert selected[key] == original[key]
    assert selected["hidden_dim"] == 32
    rollout = selected["probe_rollout"]
    assert rollout["source_updates"] == [0, 128]
    assert rollout["native_endpoint_updates"] == [16, 32, 144, 160]
    assert rollout["phases"] == ["after_probe", "after_return"]
    work, old_work = selected["planned_work"], original["planned_work"]
    for key in ("native_train_calls", "training_rows", "score_forwards", "score_rows"):
        assert work[key] == old_work[key]
    assert work["native_train_calls"] == 192 and work["training_rows"] == 7488
    assert work["score_forwards"] == 94 and work["score_rows"] == 846
    assert work["learner_live_episodes"] == 16
    assert work["maximum_live_prediction_forwards"] == 256
    assert work["maximum_live_environment_steps"] == 256
    assert work["final_episodes"] == work["replay_rows"] == 0
    assert work["maximum_total_prediction_forwards"] == 350
    assert work["maximum_total_prediction_rows"] == 1102
    assert "no live success" not in selected["metric_direction"]
    assert "live success/return higher" in selected["metric_direction"]
    assert "target-free known A/F games" in selected["evaluation"]
    assert rollout["native_endpoint_training_rows"] == [144, 288, 7056, 7200]
    assert "40s" in rollout["budget"] and "original180s" in rollout["budget"]


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_native_game_flags_before_resources(no_resources, flag):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**GAME_FLAGS, "probe_rollouts": flag})


@pytest.mark.parametrize(
    "changed",
    [
        {"public_history": False},
        {"invariant_history": False},
        {"joint_catalogue": False},
        {"wide_neural": False},
        {"qualification_method": None},
        {"qualification_method": "recurrent_bc"},
        {"qualification_method": "pc_relax64"},
        {"qualification_method": np.str_("bp")},
        {"distinct_probe": False},
        {"old_task_readouts": False},
        {"return_to_a": False},
        *({key: True} for key in EXCLUDED if key != "probe_rollouts"),
        {"pc_update_diagnostics": True},
        {"old_a_rehearsal": True},
        {"probe_inference": "fixed-weighted"},
        {"probe_inference": "bounded-weighted"},
    ],
)
def test_should_refuse_incomplete_or_extra_wide_game_routes(no_resources, changed):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**GAME_FLAGS, **changed})


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_should_forward_cli_games_only_when_explicit(monkeypatch, capsys, describe, enabled):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete"}

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *CLI_FLAGS, "--joint-catalogue", "--wide-neural"]
        + (["--probe-rollouts"] if enabled else [])
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [{**WIDE_FLAGS, **({"probe_rollouts": True} if enabled else {})}]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        CLI_FLAGS,
        [*CLI_FLAGS, "--wide-neural"],
        [*CLI_FLAGS, "--joint-catalogue"],
        [*CLI_FLAGS, "--joint-catalogue", "--wide-neural", "--train-a-rollout"],
        [*CLI_FLAGS, "--joint-catalogue", "--wide-neural", "--memory-control"],
        [*CLI_FLAGS, "--joint-catalogue", "--wide-neural", "--deferred-probe-rollouts"],
        [*CLI_FLAGS, "--joint-catalogue", "--wide-neural", "--old-a-rehearsal"],
    ],
)
def test_should_refuse_wrong_cli_game_routes_before_dispatch(monkeypatch, arguments):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", *arguments, "--probe-rollouts", "--describe"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def declared_state(method, source_updates, relative):
    """Owned structural fixture; its update history was not executed."""
    source = InvariantTorchFeatureLearner(fixed_policy(method))
    state = declared_endpoint(source)
    updates, rows = source_updates + relative, source_updates * 54 + relative * 9
    native = state.native_state
    object.__setattr__(native, "completed_updates", updates)
    object.__setattr__(native, "training_examples", rows)
    head = native.head_state
    if method == "pc":
        head["traffic_steps"] = updates
    elif method in ("neutral_cpc", "gated_cpc"):
        head.update(traffic_steps=updates, wake_examples=rows, steps_since_sleep=updates)
        head["neuron_age"].fill_(updates)
        head["energy_history"] = [0.0] * min(
            updates, max(head["config"]["sleep_energy_window"] * 4, 16)
        )
    source.restore_state(copy_invariant_torch_feature_snapshot(state))
    return source


def paid_report():
    return dict(
        status="complete",
        work=dict(
            environment_resets_completed=4,
            live_prediction_calls_returned=4,
            environment_steps_returned=4,
            internal_projection_rows_returned=4,
            internal_projection_array_bytes_returned=9472,
            failure_work_may_be_unobserved=False,
        ),
    )


@pytest.fixture
def fake_games(monkeypatch):
    work = dict(calls=[], callback=None)
    monkeypatch.setattr(live, "monotonic", lambda: 1.0)

    def evaluate(learner, state, **options):
        work["calls"].append((learner, state, options))
        assert options["method"] in METHODS
        assert options["budget"].budget.max_training_updates == 0
        assert options["budget"].budget.max_replay_examples == 0
        assert options["budget"].budget.max_wall_seconds == 40
        options["check_boundary"]()
        return work["callback"](learner, state, options) if work["callback"] else paid_report()

    monkeypatch.setattr(live, "evaluate_wide_neural_probe_checkpoint", evaluate)
    return work


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("point", [0, 128])
def test_should_observe_both_fixed_mixed_clock_endpoints(fake_games, method, point):
    observer = live.ProbeRollouts(point, wide_method=method)
    learner = declared_state(method, point, 16)
    observer.observe(learner, "after_probe", lambda: None)
    learner.restore_state(declared_state(method, point, 32).snapshot_state())
    observer.observe(learner, "after_return", lambda: None)
    observer.check()
    assert len(observer.records) == len(fake_games["calls"]) == 2
    assert [record["status"] for record in observer.records] == ["complete", "complete"]
    for index, (owner, state, options) in enumerate(fake_games["calls"]):
        relative = 16 * (index + 1)
        assert owner is learner and options["method"] == method
        assert state.completed_updates == point + relative
        assert state.training_examples == 54 * point + 9 * relative
        assert observer.records[index]["charged_updates"] == 0
        assert observer.records[index]["replay_examples"] == 0
    assert fake_games["calls"][0][2]["budget"] is not fake_games["calls"][1][2]["budget"]


@pytest.mark.parametrize("method", [True, False, 1, "", "BP", "other", np.str_("bp")])
def test_should_refuse_noncanonical_observer_method_before_work(fake_games, method):
    with pytest.raises(ValueError):
        live.ProbeRollouts(0, wide_method=method)
    assert fake_games["calls"] == []


@pytest.mark.parametrize("point", [True, False, 128.0, 1, 127, 129])
def test_should_refuse_noncanonical_source_label_before_work(fake_games, point):
    with pytest.raises(ValueError):
        live.ProbeRollouts(point, wide_method="bp")
    assert fake_games["calls"] == []


@pytest.mark.parametrize("fault", ["count_rows", "source_rows", "clock", "bool_rows", "weights"])
def test_should_refuse_wrong_or_changed_endpoint_before_more_game_work(fake_games, fault):
    observer = live.ProbeRollouts(128, wide_method="bp")
    learner = declared_state("bp", 128, 16)
    if fault == "weights":

        def changed(owner, state, options):
            current = owner.snapshot_state()
            current.native_state.head_state["bias_hidden"][0, 0] += 1
            owner.restore_state(current)
            return paid_report()

        fake_games["callback"] = changed
    else:
        state = learner.snapshot_state()
        field = "completed_updates" if fault == "clock" else "training_examples"
        values = {"count_rows": 9 * 144, "source_rows": 6912, "clock": 128, "bool_rows": True}
        object.__setattr__(state.native_state, field, values[fault])
        # Preserve the intentionally malformed port response; restore_state itself
        # correctly refuses inconsistent native states before the observer runs.
        learner.snapshot_state = lambda: deepcopy(state)
    with pytest.raises(RuntimeError):
        observer.observe(learner, "after_probe", lambda: None)
    assert observer.records[0]["status"] == "failed"
    assert len(fake_games["calls"]) == int(fault == "weights")


@pytest.mark.parametrize("failure", ["partial", "lost", "budget", "clock"])
def test_should_preserve_paid_prefix_or_unknown_work_after_failed_wide_games(fake_games, failure):
    observer = live.ProbeRollouts(0, wide_method="bp")

    def fail(owner, state, options):
        if failure == "partial":
            report = paid_report()
            report.update(status="failed", reason="injected paid physics failure")
            raise live.MiniGridEvaluationFailed(report)
        if failure == "lost":
            raise RuntimeError("injected admitted lost report")
        if failure == "clock":
            options["budget"].last_clock += 0.5
        else:
            options["budget"].budget = replace(options["budget"].budget, max_training_updates=1)
        return paid_report()

    fake_games["callback"] = fail
    with pytest.raises(RuntimeError):
        observer.observe(declared_state("bp", 0, 16), "after_probe", lambda: None)
    record = observer.records[0]
    assert record["status"] == "failed" and record["evaluator_admitted"] is True
    work = dict(
        total_score_forwards_completed=94,
        total_score_rows_completed=846,
        source_environment_steps_returned=0,
        failure_work_may_be_unobserved=False,
    )
    live.add_work(work, [dict(probes=[dict(probe_rollouts=observer.records)])])
    if failure == "lost":
        assert record["report"] is None
        assert work["learner_live_episodes"] is None
        assert work["total_prediction_forwards_completed"] is None
    else:
        assert record["report"]["work"]["environment_steps_returned"] == 4
        assert (
            work["probe_rollouts"]["known_returned_work_lower_bounds"]["environment_steps_returned"]
            == 4
        )


@pytest.mark.parametrize("fault", ["replace", "value", "budget", "clock"])
def test_should_validate_preceding_paid_report_and_budget_before_next_phase(fake_games, fault):
    observer = live.ProbeRollouts(0, wide_method="bp")
    learner = declared_state("bp", 0, 16)
    observer.observe(learner, "after_probe", lambda: None)
    paid = deepcopy(observer.records)
    prior_budget = fake_games["calls"][0][2]["budget"]
    if fault == "replace":
        observer.records[0]["report"] = deepcopy(observer.records[0]["report"])
    elif fault == "value":
        observer.records[0]["report"]["work"]["environment_steps_returned"] += 1
    elif fault == "budget":
        prior_budget.started_at -= 0.5
    else:
        prior_budget.last_clock += 0.5
    learner.restore_state(declared_state("bp", 0, 32).snapshot_state())
    with pytest.raises(ValueError):
        observer.observe(learner, "after_return", lambda: None)
    assert len(fake_games["calls"]) == 1
    observer.restore_paid()
    assert observer.records == paid


def test_should_execute_public_bp192_isolated_chain_and_all_sixteen_native_games(
    monkeypatch, tmp_path
):
    torch = require_torch()
    raw = tuple(
        pilot._packet(episode(context, cue, "train", f"minigrid-source-{context}-{cue}"))
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )
    probe = raw_probe_data()
    before = pickle.dumps((raw, probe))
    rng = (random.getstate(), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone())
    owners, originals, arms, cohorts, endpoints = [], [], [], [], []
    input_checks, endpoint_attempts, endpoint_checks = [], [], []
    build, train, run_probe, evaluate = (
        pilot._history_learner,
        pilot.shared._train_arm,
        pilot.run_fresh_task_probe,
        live.evaluate_wide_neural_probe_checkpoint,
    )

    def learner(method, **options):
        assert options == {"wide_neural": True}
        owned = build(method, **options)
        owners.append(owned)
        originals.append(deepcopy(owned.snapshot_state()))
        return owned

    def native_arm(*args, **options):
        assert options["catalogue_rows"] == 54 and options["wide_neural"] is True
        arm = train(*args, **options)
        arms.append(arm)
        return arm

    def isolated(source, state, training, evaluation, **options):
        assert state.policy.native_policy.hidden_dim == 32
        cohorts.append((state.completed_updates, options["budget"], options["return_budget"]))
        assert options["endpoint_observer"] is not None
        inputs = pickle.dumps(
            (training, evaluation, options["training_readouts"], options["return_training"])
        )
        try:
            return run_probe(source, state, training, evaluation, **options)
        finally:
            assert (
                pickle.dumps(
                    (training, evaluation, options["training_readouts"], options["return_training"])
                )
                == inputs
            )
            input_checks.append(True)

    def games(source, state, **options):
        endpoint_attempts.append(True)
        assert_tree(torch, source.snapshot_state(), state)
        endpoints.append(
            (
                options["source_updates"],
                options["phase"],
                state.completed_updates,
                state.training_examples,
                options["budget"],
            )
        )
        try:
            return evaluate(source, state, **options)
        finally:
            assert_tree(torch, source.snapshot_state(), state)
            endpoint_checks.append(True)

    monkeypatch.setattr(pilot.shared, "_collect_sources", lambda boundary, reports: raw)
    monkeypatch.setattr(pilot, "_collect_probe_sources", lambda boundary, reports, **options: probe)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", forbidden)
    monkeypatch.setattr(pilot, "_history_learner", learner)
    monkeypatch.setattr(pilot.shared, "_train_arm", native_arm)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", isolated)
    monkeypatch.setattr(live, "evaluate_wide_neural_probe_checkpoint", games)
    report = pilot.run_pilot(**GAME_FLAGS)
    label = os.environ.get("WIDE_CHECK_LABEL")
    destination = (
        Path(os.environ.get("WIDE_CHECK_STAGE", "artifacts/runs/wide-neural-probe-games-20261018"))
        / f"{label}-public-bp.json"
        if label
        else tmp_path / "public-bp.json"
    )
    with destination.open("x", encoding="utf8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    # A failed completion assertion must not skip isolation of the paid prefix.
    assert len(owners) == len(arms) == 1
    assert_tree(torch, owners[0].snapshot_state(), originals[0])
    arms[0]["guard"]()
    assert pickle.dumps((raw, probe)) == before
    assert random.getstate() == rng[0] and pickle.dumps(np.random.get_state()) == rng[1]
    assert torch.equal(torch.get_rng_state(), rng[2])
    # The pilot catches delegate exceptions. A swallowed failed closing assertion
    # must not turn into a successful isolation receipt outside that pilot.
    assert len(input_checks) == len(cohorts)
    assert len(endpoint_checks) == len(endpoint_attempts) == len(endpoints)
    if label:
        isolation = dict(
            source_unchanged=True,
            completed_arm_guard_passed=True,
            input_bytes_unchanged=True,
            python_numpy_torch_rng_unchanged=True,
            disposable_input_cohorts_checked_in_finally=len(input_checks),
            native_endpoints_checked_in_finally=len(endpoint_checks),
            public_complete=report["status"] == "complete",
        )
        with destination.with_name(f"{label}-isolation.json").open("x", encoding="utf8") as handle:
            json.dump(isolation, handle, indent=2, allow_nan=False)
    assert report["status"] == "complete", report["reason"]
    assert report["observed_process_rss"]["peak_bytes"] > 0
    row = report["methods"][0]
    assert row["status"] == "complete" and row["method"] == "public_invariant_bp"
    assert row["charged_updates"] == row["stream_result"]["completed_updates"] == 128
    assert row["native_training_examples"] == row["stream_result"]["training_examples"] == 6912
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert [value["updates"] for value in row["readouts"]] == [0, 32, 64, 96, 128]
    assert [point for point, _, _ in cohorts] == [0, 128]
    assert [(point, phase, updates, rows) for point, phase, updates, rows, _ in endpoints] == [
        (0, "after_probe", 16, 144),
        (0, "after_return", 32, 288),
        (128, "after_probe", 144, 7056),
        (128, "after_return", 160, 7200),
    ]
    assert len({id(item[-1]) for item in endpoints}) == 4
    probe_budgets = [budget for _, fresh, returned in cohorts for budget in (fresh, returned)]
    assert len({id(budget) for budget in probe_budgets}) == 4
    assert all(
        budget.updates_completed == 16 and budget.replay_examples_completed == 0
        for budget in probe_budgets
    )
    for slot in row["probes"]:
        assert slot["status"] == "complete"
        assert slot["charged_updates"] == slot["return_charged_updates"] == 16
        assert slot["expected_training_examples"] == 54 * slot["checkpoint_updates"]
        assert slot["source_unchanged"] is slot["state_unchanged"] is True
        assert slot["return_budget_binding_preserved"] is True
        fresh, returned = slot["report"], slot["report"]["return_phase"]
        assert returned["checkpoints"][0]["scores"] == fresh["checkpoints"][-1]["scores"]
        for phase in (fresh, returned):
            assert phase["completed_training_examples"] == 144
            assert [value["updates"] for value in phase["checkpoints"]] == [0, 4, 8, 16]
            assert phase["training_readouts"]["completed_rows"] == 72
        assert len(slot["probe_rollouts"]) == 2
        for receipt in slot["probe_rollouts"]:
            assert receipt["status"] == "complete"
            assert receipt["timing_complete"] is receipt["budget_binding_preserved"] is True
            assert receipt["charged_updates"] == receipt["replay_examples"] == 0
            games_report = receipt["report"]
            assert games_report["status"] == "complete" and games_report["scores"] == []
            assert games_report["state_unchanged"] is True
            assert [
                (item["context"], item["cue"], item["source_role"])
                for item in games_report["episodes"]
            ] == [
                ("A", "blue", "train"),
                ("A", "red", "train"),
                ("F-blue", "blue", "probe_test"),
                ("F-red", "red", "probe_test"),
            ]
            for item in games_report["episodes"]:
                assert item["status"] == "complete"
                first = item["predictions"][0]["observation"]
                for decision in item["predictions"]:
                    assert decision["features"] == first + decision["observation"]
                    assert decision["action"] == int(np.argmax(decision["probabilities"]))
                    assert "target" not in decision and "teacher" not in decision
    work = report["work"]
    assert work["native_train_calls"] == 192 and work["training_rows"] == 7488
    assert (
        work["total_score_forwards_completed"] == 94 and work["total_score_rows_completed"] == 846
    )
    live_work = work["probe_rollouts"]
    assert live_work["endpoint_records"] == 4
    assert live_work["environment_resets_completed"] == work["learner_live_episodes"] == 16
    assert 0 < live_work["live_prediction_calls_returned"] <= 256
    assert live_work["environment_steps_returned"] == live_work["live_prediction_calls_returned"]
    assert live_work["charged_updates"] == live_work["replay_examples"] == 0
    assert live_work["failure_work_may_be_unobserved"] is False
    assert (
        work["total_prediction_forwards_completed"]
        == 94 + live_work["live_prediction_calls_returned"]
    )
    assert (
        work["total_prediction_rows_completed"] == 846 + live_work["live_prediction_calls_returned"]
    )
    assert report["scientific_advantage_verified"] is False and report["comparisons"] is None
    assert row["final_evaluation"]["status"] == "not_run"
