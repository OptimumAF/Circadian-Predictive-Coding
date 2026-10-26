"""Explicit H32 resource selection, with unchanged scientific work and defaults.

Admission/CLI/fake-clock checks perform no model updates or game physics. The
single separately selected native case reuses every assertion of the original
192-update/sixteen-game test, while retaining that default-budget test intact.
"""

from dataclasses import dataclass
import json
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_probe_rollout as live
from test_minigrid_invariant_learnability import episode
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS, METHODS
from test_minigrid_wide_learnability import EXCLUDED
from test_minigrid_wide_probe_games import (
    GAME_FLAGS,
    WIDE_FLAGS,
    test_should_execute_public_bp192_isolated_chain_and_all_sixteen_native_games as original_case,
)


def forbidden(*args, **kwargs):
    raise AssertionError("admission must precede resources, training and source collection")


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)
    monkeypatch.setattr(pilot, "_collect_probe_sources", forbidden)


@pytest.mark.parametrize(
    "options",
    [{}, {"public_history": True}, FLAGS, WIDE_FLAGS, GAME_FLAGS],
)
def test_should_preserve_descriptor_when_extended_budget_is_disabled(no_resources, options):
    ordinary = pilot.describe_protocol(**options)
    disabled = pilot.describe_protocol(**options, extended_wide_probe_budget=False)
    assert ordinary == disabled
    assert "extended_wide_probe_budget" not in ordinary
    if options.get("probe_rollouts"):
        assert "40s" in ordinary["probe_rollout"]["budget"]
        assert "original180s" in ordinary["probe_rollout"]["budget"]
        assert "endpoint_wall_seconds" not in ordinary["probe_rollout"]
        assert "parent_wall_seconds" not in ordinary["probe_rollout"]


@pytest.mark.parametrize("method", METHODS)
def test_should_describe_equal_explicit_resources_without_changing_scientific_recipe(
    no_resources, method
):
    options = {**GAME_FLAGS, "qualification_method": method}
    ordinary = pilot.describe_protocol(**options)
    selected = pilot.describe_protocol(**options, extended_wide_probe_budget=True)
    assert selected["methods"] == ordinary["methods"] == [f"public_invariant_{method}"]
    assert selected["extended_wide_probe_budget"] == {
        "endpoint_wall_seconds": 60.0,
        "parent_wall_seconds": 240.0,
    }
    assert ordinary["limits"]["body_seconds"] == 180
    assert ordinary["limits"]["parent_seconds"] == 210
    assert ordinary["limits"]["shared_cleanup_seconds"] == 220
    assert selected["limits"] == {
        **ordinary["limits"],
        "body_seconds": 240.0,
        "parent_seconds": 270.0,
        "shared_cleanup_seconds": 280.0,
    }
    rollout = selected["probe_rollout"]
    assert rollout["endpoint_wall_seconds"] == 60.0
    assert rollout["parent_wall_seconds"] == 240.0
    assert "60s" in rollout["budget"] and "240s" in rollout["budget"]
    assert {
        key: value
        for key, value in rollout.items()
        if key not in ("budget", "endpoint_wall_seconds", "parent_wall_seconds")
    } == {key: value for key, value in ordinary["probe_rollout"].items() if key != "budget"}
    assert {
        key: value
        for key, value in selected.items()
        if key not in ("extended_wide_probe_budget", "probe_rollout", "limits")
    } == {key: value for key, value in ordinary.items() if key not in ("probe_rollout", "limits")}
    work = selected["planned_work"]
    assert work["native_train_calls"] == 192 and work["training_rows"] == 7488
    assert work["score_forwards"] == 94 and work["score_rows"] == 846
    assert work["learner_live_episodes"] == 16
    assert work["maximum_live_prediction_forwards"] == 256
    assert work["maximum_live_environment_steps"] == 256
    assert work["final_episodes"] == work["replay_rows"] == 0


@pytest.mark.parametrize("flag", [None, 0, 1, "true", [], np.bool_(True), np.bool_(False)])
def test_should_refuse_non_native_extended_flags_before_resources(no_resources, flag):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**GAME_FLAGS, extended_wide_probe_budget=flag)


@pytest.mark.parametrize("flag", [None, 0, 1, "true", [], np.bool_(True), True])
def test_should_refuse_invalid_or_non_wide_probe_selection_before_reading_ports(no_resources, flag):
    with pytest.raises(ValueError):
        pilot._run_probe(
            {},
            {},
            object(),
            ((), ()),
            sampler=object(),
            deadline=0.0,
            check_boundary=forbidden,
            extended_wide_probe_budget=flag,
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"public_history": False},
        {"invariant_history": False},
        {"joint_catalogue": False},
        {"wide_neural": False},
        {"probe_rollouts": False},
        {"qualification_method": None},
        {"qualification_method": "recurrent_bc"},
        {"qualification_method": "pc_relax64"},
        {"qualification_method": np.str_("bp")},
        {"distinct_probe": False},
        {"old_task_readouts": False},
        {"return_to_a": False},
        *({key: True} for key in EXCLUDED),
        {"pc_update_diagnostics": True},
        {"old_a_rehearsal": True},
        {"probe_inference": "fixed-weighted"},
        {"probe_inference": "bounded-weighted"},
    ],
)
def test_should_refuse_incomplete_or_mixed_extended_recipe_before_resources(no_resources, changed):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**GAME_FLAGS, **changed}, extended_wide_probe_budget=True)


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_should_forward_cli_extended_selection_only_when_explicit(
    monkeypatch, capsys, describe, enabled
):
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
        ["pilot", *CLI_FLAGS, "--joint-catalogue", "--wide-neural", "--probe-rollouts"]
        + (["--extended-wide-probe-budget"] if enabled else [])
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [{**GAME_FLAGS, **({"extended_wide_probe_budget": True} if enabled else {})}]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--scenario", "torch-minigrid-learnability"],
        CLI_FLAGS,
        [*CLI_FLAGS, "--joint-catalogue", "--wide-neural"],
        [*CLI_FLAGS, "--joint-catalogue", "--probe-rollouts"],
        [*CLI_FLAGS, "--wide-neural", "--probe-rollouts"],
        [*CLI_FLAGS, "--joint-catalogue", "--wide-neural", "--probe-rollouts", "--train-a-rollout"],
        [*CLI_FLAGS, "--joint-catalogue", "--wide-neural", "--probe-rollouts", "--old-a-rehearsal"],
        [*CLI_FLAGS, "--joint-catalogue", "--wide-neural", "--probe-rollouts", "--memory-control"],
        [*CLI_FLAGS, "--joint-catalogue", "--wide-neural", "--probe-rollouts", "--weighted-bp"],
    ],
)
def test_should_refuse_wrong_cli_extended_routes_before_dispatch(monkeypatch, arguments, describe):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *arguments, "--extended-wide-probe-budget"]
        + (["--describe"] if describe else []),
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@dataclass
class FakeRssSnapshot:
    peak_bytes: int = 1


class FakeRssSampler:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def sample(self):
        return None

    def snapshot(self):
        return FakeRssSnapshot()


class FakeTorch:
    @staticmethod
    def set_num_threads(count):
        assert count == 1

    @staticmethod
    def get_num_interop_threads():
        return 1


@pytest.fixture
def fake_resources(monkeypatch):
    clock = [105.0]
    monkeypatch.setattr(pilot, "monotonic", lambda: clock[0])
    monkeypatch.setattr(pilot, "require_torch", FakeTorch)
    monkeypatch.setattr(pilot, "ProcessRssSampler", FakeRssSampler)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", forbidden)
    monkeypatch.setattr(pilot, "_history_learner", forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot, "_collect_probe_sources", forbidden)
    return clock


@pytest.mark.parametrize("extended", [False, True])
@pytest.mark.parametrize("elapsed", [181.0, 239.0, 241.0])
def test_should_enforce_selected_parent_resource_limit_before_collection_continues(
    monkeypatch, fake_resources, extended, elapsed
):
    admitted = []

    def collect(boundary, reports):
        fake_resources[0] += elapsed
        boundary()
        admitted.append(True)
        raise RuntimeError("fake collection stopped after selected parent check")

    monkeypatch.setattr(pilot.shared, "_collect_sources", collect)
    report = pilot.run_pilot(**GAME_FLAGS, extended_wide_probe_budget=extended)
    allowed = elapsed <= (240.0 if extended else 180.0)
    assert admitted == ([True] if allowed else [])
    assert report["status"] == "incomplete"
    assert report["total_seconds"] == elapsed
    if allowed:
        assert "fake collection stopped after selected parent check" in report["reason"]
    else:
        assert "whole-case wall-time budget exhausted" in report["reason"]
    assert report["work"]["native_train_calls"] == 0
    assert report["work"]["training_rows"] == 0


@pytest.mark.parametrize("extended", [False, True])
def test_should_pass_selected_parent_deadline_to_existing_training_runner(
    monkeypatch, fake_resources, extended
):
    sources = tuple(
        pilot._packet(episode(context, cue, "train", f"minigrid-source-{context}-{cue}"))
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )
    deadlines = []

    def train(row, packets, sampler, deadline, check_boundary, **options):
        check_boundary()
        deadlines.append(deadline)
        assert options["catalogue_rows"] == 54 and options["wide_neural"] is True
        raise RuntimeError("fake training stopped after deadline inspection")

    monkeypatch.setattr(pilot.shared, "_collect_sources", lambda boundary, reports: sources)
    monkeypatch.setattr(pilot.shared, "_train_arm", train)
    report = pilot.run_pilot(**GAME_FLAGS, extended_wide_probe_budget=extended)
    assert deadlines == [105.0 + (240.0 if extended else 180.0)]
    assert "fake training stopped after deadline inspection" in report["reason"]
    assert report["status"] == "incomplete"
    assert report["work"]["native_train_calls"] == 0
    assert report["work"]["training_rows"] == 0


def test_should_execute_public_extended_bp192_isolated_chain_and_all_sixteen_native_games(
    monkeypatch, tmp_path
):
    """One prospective case; all original isolation/work assertions stay active."""
    run_pilot, run_probe, live_budget = pilot.run_pilot, pilot._run_probe, live._live_budget
    reports, probe_options, budgets = [], [], []

    def selected_pilot(**options):
        assert "extended_wide_probe_budget" not in options
        report = run_pilot(**options, extended_wide_probe_budget=True)
        reports.append(report)
        return report

    def selected_probe(*args, **options):
        assert options["extended_wide_probe_budget"] is True
        probe_options.append(options)
        return run_probe(*args, **options)

    def selected_budget(*args, **options):
        budget, guard = live_budget(*args, **options)
        assert budget.budget.max_wall_seconds == 60.0
        assert budget.budget.max_training_updates == 0
        assert budget.budget.max_replay_examples == 0
        budgets.append(budget)
        return budget, guard

    monkeypatch.setattr(pilot, "run_pilot", selected_pilot)
    monkeypatch.setattr(pilot, "_run_probe", selected_probe)
    monkeypatch.setattr(live, "_live_budget", selected_budget)
    original_case(monkeypatch, tmp_path)
    assert len(reports) == 1 and reports[0]["status"] == "complete"
    assert reports[0]["protocol"]["extended_wide_probe_budget"] == {
        "endpoint_wall_seconds": 60.0,
        "parent_wall_seconds": 240.0,
    }
    assert len(probe_options) == 2
    assert probe_options[0]["deadline"] == probe_options[1]["deadline"]
    assert len(budgets) == 4 and len({id(budget) for budget in budgets}) == 4
