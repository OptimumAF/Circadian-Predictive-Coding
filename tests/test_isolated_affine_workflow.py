"""Fresh composition with SCRIPTED numerical/game delegates, not task evidence."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

from scripts import minigrid_affine_workflow as workflow
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_affine_probe as binding
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from test_after_c_linear_games import AfterCEnvironment, state
from test_affine_probe_games import state as probe_state
from test_catalogue_capacity import sources
from test_minigrid_minimum_change_probe import scripted as scripted
from test_minigrid_invariant_evaluation import public_frame
from test_portable_affine_workflow import complete_training
from test_retained_context_linear_games import budget

CHARGE = lambda values: None
RETURN = lambda values: None


class AllContextEnvironment(AfterCEnvironment):
    def __init__(self, *args, **kwargs):
        values = dict(script_game_episodes=1)
        CHARGE(values)
        super().__init__(*args, **kwargs)
        RETURN(values)

    def _frame(self, step):
        return public_frame(
            self.cue,
            step,
            wall_color={"grey": 5, "blue": 2, "red": 0}[self.layout.wall_color],
            rotation=self.layout.rotation,
        )

    def step(self, action):
        values = dict(script_game_steps=1)
        CHARGE(values)
        output = super().step(action)
        RETURN(values)
        return output


def roster():
    return [(f"F-{cue}", cue, "probe_test") for cue in ("blue", "red")] + [
        (context, cue, "train") for context in ("A", "B", "C_train") for cue in ("blue", "red")
    ]


def phase():
    return dict(
        status="complete",
        state_unchanged=True,
        probe_phase="after_probe",
        include_old_tasks=True,
        episodes=[
            dict(context=c, cue=q, source_role=r, status="complete", environment=dict(closed=True))
            for c, q, r in roster()
        ],
    )


@pytest.fixture
def fresh(monkeypatch, scripted):
    box: dict[str, Any] = dict(
        training=complete_training(), state=state(), sources=sources(), calls=[]
    )
    monkeypatch.setattr(workflow, "_require_minigrid", lambda: None)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: box["sources"])

    def train(episodes, **options):
        box["calls"].append("train")
        assert episodes is box["sources"]
        assert options["retain_arrived"] is options["abc_catalogue"] is True
        options["check_boundary"]()
        options["endpoint_observer"](box["state"])
        return box["training"]

    def games(source, endpoint, **options):
        box["calls"].append("games")
        assert source.snapshot_state().heads == (None, None)
        assert (endpoint.completed_updates, endpoint.training_examples) == (8, 126)
        assert options["phase"] == "after_probe" and options["include_old_tasks"] is True
        assert options["budget"].updates_completed == 0
        box.update(source=source, budget=options["budget"])
        options["check_boundary"]()
        return phase()

    monkeypatch.setattr(cli, "run_context_linear_stream", train)
    monkeypatch.setattr(evaluation, "evaluate_context_linear_probe_checkpoint", games)
    box["scripted"] = scripted
    return box


def run():
    return workflow.run_workflow(private_adaptation=True, probe_update="minimum_change")


def test_connects_actual_isolated_adapter_to_fresh_endpoint_and_all_old_games(fresh, monkeypatch):
    from scripts import minigrid_affine_probe_games as saved

    for module, name in ((cli, "_qualified_after_c_bytes"), (saved, "_qualified_probe_bytes")):
        monkeypatch.setattr(module, name, lambda *a: pytest.fail("historical source"))
    before = pickle.dumps((fresh["sources"], fresh["state"], np.random.get_state()))
    result = run()
    assert result["status"] == "complete", result["reason"]
    assert fresh["calls"] == ["train", "games"]
    assert fresh["scripted"]["fits"] == ["warm", "warm"]
    assert result["attempted_phases"] == ["training", "private_probe", "games"]
    assert result["private_comparison"] is None and result["collection"] == []
    probe = result["private_probe"]
    assert len(probe["collection"]) == len(probe["probe_data"]) == 4
    assert probe["probe"]["completed_training_examples"] == 18
    assert probe["source_fork_calls"] == 1 and probe["source_native_state_unchanged"] is True
    assert result["TRAIN_reference_unchanged"] is True
    assert result["readout_budget"]["updates_completed"] == 0
    assert result["readout_budget"]["replay_examples_completed"] == 0
    assert (
        result["readout_budget"]["started_at"]
        == result["readout_budget"]["observed_budget_started_at"]
    )
    assert pickle.dumps((fresh["sources"], fresh["state"], np.random.get_state())) == before


@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(recurring_training=True),
        dict(a4_adaptation=True),
        dict(private_adaptation=True, train_update="minimum_change"),
    ],
)
def test_refuses_unsupported_workflow_intent_before_runtime(monkeypatch, options):
    monkeypatch.setattr(workflow, "_require_minigrid", lambda: pytest.fail("runtime"))
    for action in (workflow.run_workflow, workflow.describe_workflow):
        with pytest.raises(ValueError):
            action(probe_update="minimum_change", **options)


@pytest.mark.parametrize("value", [True, "cold", 1])
def test_refuses_ambiguous_selector_without_sources(monkeypatch, value):
    monkeypatch.setattr(workflow, "_run_private_adaptation", lambda **k: pytest.fail("work"))
    with pytest.raises(ValueError, match="probe_update"):
        workflow.run_workflow(private_adaptation=True, probe_update=value)


def test_metadata_has_actual_dose_and_all_tasks_without_changing_default():
    old = workflow.describe_workflow(private_adaptation=True)
    selected = workflow.describe_workflow(private_adaptation=True, probe_update="minimum_change")
    assert old["new_fit_calls"] == 10 and old["new_fit_rows"] == 252 and "probe_update" not in old
    assert (selected["new_fit_calls"], selected["new_fit_rows"]) == (8, 126)
    assert "A/B/C_train" in selected["games"] and "returned_layout" not in selected
    assert selected["private"]["probe_update"] == "minimum_change"


@pytest.mark.parametrize("fault", ["source", "receipt", "endpoint"])
def test_refuses_incomplete_source_before_private_work(fresh, fault):
    if fault == "source":
        fresh["training"]["source_unchanged"] = False
    elif fault == "receipt":
        fresh["training"]["committed_receipts"].pop()
    else:
        fresh["state"] = replace(fresh["state"], training_examples=90)
    result = run()
    assert result["status"] == "failed" and result["private_probe"] is None
    assert fresh["scripted"]["fits"] == [] and fresh["calls"] == ["train"]


def literal_probe():
    limits = asdict(
        workflow.ToyExecutionBudget(
            max_training_updates=2,
            max_replay_examples=0,
            max_wall_seconds=90,
            max_process_rss_bytes=512 * 1024**2,
        )
    )
    return dict(
        status="complete",
        protocol=binding.describe_protocol(probe_update="minimum_change"),
        probe_update="minimum_change",
        source_native_state_unchanged=True,
        source_fork_calls=1,
        probe=dict(status="complete", completed_updates=2, completed_training_examples=18),
        budget=dict(
            limits=deepcopy(limits),
            updates_completed=2,
            replay_examples_completed=0,
            elapsed_seconds=1.0,
            started_at=10.0,
            ended_at=11.0,
        ),
        initial_probe_limits=limits,
        disposable_state=asdict(probe_state("after_probe")),
    )


@pytest.mark.parametrize("fault", ["partial", "charge", "clock", "limits", "state", "fork"])
def test_refuses_unpaid_or_incomplete_child_before_games(fresh, monkeypatch, fault):
    probe = literal_probe()
    if fault == "partial":
        probe["status"] = "failed"
    elif fault == "charge":
        probe["budget"]["updates_completed"] = 1
    elif fault == "clock":
        probe["budget"]["ended_at"] = 9
    elif fault == "limits":
        probe["budget"]["limits"]["max_wall_seconds"] = 91
    elif fault == "state":
        probe["disposable_state"]["training_examples"] = 125
    else:
        probe["source_fork_calls"] = 2
    monkeypatch.setattr(binding, "run_affine_fresh_probe", lambda *a, **k: probe)
    result = run()
    assert result["status"] == "failed" and fresh["calls"] == ["train"]
    assert result["private_probe"] == probe and result["games"] is None


@pytest.mark.parametrize(
    "fault", ["training_replace", "probe_replace", "source", "clock", "charge", "roster", "lost"]
)
def test_retains_paid_evidence_and_refuses_changed_game_boundary(fresh, monkeypatch, fault):
    probe = literal_probe()
    original = deepcopy(probe)
    monkeypatch.setattr(binding, "run_affine_fresh_probe", lambda *a, **k: probe)

    def games(source, endpoint, **options):
        guard = options["check_boundary"]
        if fault == "training_replace":
            fresh["training"]["committed_receipts"] = deepcopy(
                fresh["training"]["committed_receipts"]
            )
        elif fault == "probe_replace":
            probe["budget"] = dict(probe["budget"])
        elif fault == "source":
            source.restore_state(endpoint)
        elif fault == "clock":
            options["budget"].started_at -= 1
        elif fault == "charge":
            options["budget"].updates_completed = 1
        elif fault == "lost":
            raise evaluation.MiniGridEvaluationFailed(
                dict(status="failed", reason="lost step", episodes=[dict(paid_steps=1)])
            )
        if fault == "roster":
            output = phase()
            output["episodes"][-1]["context"] = "A"
            return output
        guard()
        pytest.fail("changed owner accepted")

    monkeypatch.setattr(evaluation, "evaluate_context_linear_probe_checkpoint", games)
    result = run()
    assert result["status"] == "failed"
    assert result["private_probe"] == original
    if fault == "charge":
        assert result["readout_budget"]["updates_completed"] == 1
    if fault == "lost":
        assert result["games"][0]["episodes"][0]["paid_steps"] == 1
    assert fresh["scripted"]["fits"] == []


def test_keeps_actual_probe_paid_failure_without_game_retry(fresh):
    def change():
        fresh["training"]["status"] = "corrupt after native commit"

    fresh["scripted"]["effect"] = change
    result = run()
    assert result["status"] == "failed" and fresh["calls"] == ["train"]
    assert fresh["scripted"]["fits"] == ["warm"]
    assert result["training"]["status"] == "complete"
    assert result["private_probe"]["disposable_state"]["completed_updates"] == 7
    assert result["private_probe"]["status"] == "failed"


def test_original_evaluator_executes_full_mixed_role_loop_without_training(monkeypatch, scripted):
    monkeypatch.setattr(AllContextEnvironment, "work", {})
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", AllContextEnvironment)
    source = ContextLinearLearner()
    result = evaluation.evaluate_context_linear_probe_checkpoint(
        source,
        probe_state("after_probe"),
        phase="after_probe",
        budget=budget(),
        check_boundary=lambda: None,
        include_old_tasks=True,
    )
    assert result["status"] == "complete", result["reason"]
    assert [(e["context"], e["cue"], e["source_role"]) for e in result["episodes"]] == roster()
    assert all(e["environment"]["closed"] for e in result["episodes"])
    assert [e.role for e in AllContextEnvironment.work["environments"]] == ["probe_test"] * 2 + [
        "train"
    ] * 6
    assert result["work"]["environment_steps_returned"] == 16
    assert source.completed_updates == 0 and scripted["fits"] == []


@pytest.mark.parametrize(
    "options",
    [
        dict(include_old_tasks=1),
        dict(phase="before_probe", include_old_tasks=True),
        dict(retained_a4=True, include_old_tasks=True),
    ],
)
def test_all_old_roster_is_not_a_general_state_admission_option(monkeypatch, options):
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("loop"))
    args = dict(phase="after_probe", include_old_tasks=True) | options
    with pytest.raises(ValueError, match="include_old_tasks"):
        evaluation.evaluate_context_linear_probe_checkpoint(
            ContextLinearLearner(),
            probe_state("after_probe"),
            budget=budget(),
            check_boundary=lambda: None,
            **args,
        )


def test_cli_describes_and_forwards_selected_private_intent_without_saved_sources(
    monkeypatch, capsys
):
    argv = ["control", "--example-affine-adaptation", "--probe-update", "minimum_change"]
    monkeypatch.setattr(sys, "argv", [*argv, "--describe"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["new_fit_calls"] == 8
    seen = []

    def execute(**options):
        seen.append(options)
        return dict(status="complete")

    monkeypatch.setattr(workflow, "run_workflow", execute)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("saved source"))
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert seen == [dict(private_adaptation=True, probe_update="minimum_change")]


def test_cli_refuses_mixed_or_unsupported_intent_before_work(monkeypatch):
    monkeypatch.setattr(workflow, "run_workflow", lambda **k: pytest.fail("work"))
    for mode in ("--example-affine-play", "--example-affine-recurring", "--example-affine-a4"):
        monkeypatch.setattr(
            sys, "argv", ["control", mode, "--probe-update", "minimum_change", "--describe"]
        )
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2
