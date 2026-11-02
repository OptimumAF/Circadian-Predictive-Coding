"""Fixed matched readout composition with scripted sources, fits and games."""

from copy import deepcopy
from dataclasses import replace
import json
import sys

import pytest

from scripts import minigrid_affine_workflow as workflow
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_affine_probe as original
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_context_linear_learner as port
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from test_affine_rehearsal_return_games import state
from test_isolated_affine_workflow import AllContextEnvironment, roster
from test_retained_context_linear_games import budget
import test_portable_affine_adaptation as composition
import test_minigrid_minimum_change_probe as delegates

scripted = composition.scripted
predictions = delegates.scripted
CHARGE = lambda values: None
RETURN = lambda values: None


def phase(arm):
    return dict(
        status="complete",
        state_unchanged=True,
        rehearsal_arm=arm,
        include_old_tasks=True,
        episodes=[
            dict(context=c, cue=q, source_role=r, status="complete", environment=dict(closed=True))
            for c, q, r in roster()
        ],
    )


@pytest.fixture
def fresh(monkeypatch, scripted):
    collect = original.reserved._collect_probe_sources

    def counted(*args, **kwargs):
        values = dict(script_collections=1, script_collection_records=4)
        CHARGE(values)
        result = collect(*args, **kwargs)
        RETURN(values)
        return result

    monkeypatch.setattr(original.reserved, "_collect_probe_sources", counted)
    return scripted


def run():
    return workflow.run_workflow(private_adaptation=True, all_old_tasks=True)


@pytest.mark.parametrize("value", [1, None, "all"])
def test_requires_exact_all_old_flag_before_runtime(monkeypatch, value):
    monkeypatch.setattr(workflow, "_require_minigrid", lambda: pytest.fail("runtime"))
    for action in (workflow.run_workflow, workflow.describe_workflow):
        with pytest.raises(ValueError, match="all_old_tasks"):
            action(private_adaptation=True, all_old_tasks=value)


@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(recurring_training=True),
        dict(a4_adaptation=True),
        dict(private_adaptation=True, probe_update="minimum_change"),
        dict(private_adaptation=True, train_update="minimum_change"),
    ],
)
def test_refuses_other_recipes_before_sources(monkeypatch, options):
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source"))
    for action in (workflow.run_workflow, workflow.describe_workflow):
        with pytest.raises(ValueError, match="all_old_tasks"):
            action(all_old_tasks=True, **options)


def test_describes_only_roster_change_with_original_matched_training_dose():
    before = workflow.describe_workflow(private_adaptation=True)
    after = workflow.describe_workflow(private_adaptation=True, all_old_tasks=True)
    assert before["games"]["games_per_phase"] == 4 and "all_old_tasks" not in before
    assert after["all_old_tasks"] is True and after["games"]["games_per_phase"] == 8
    assert after["games"]["max_steps_per_phase"] == 128
    assert after["games"]["contexts"] == tuple(c for c, _, _ in roster())
    assert after["games"]["roles"] == tuple(r for _, _, r in roster())
    assert after["training"] == before["training"] and after["private"] == before["private"]
    assert (after["new_fit_calls"], after["new_fit_rows"]) == (10, 252)
    assert after["max_workflow_seconds"] == before["max_workflow_seconds"] == 180


def test_runs_same_fresh_source_collection_and_cold_comparison_before_both_full_rosters(
    fresh, monkeypatch
):
    seen = []
    original_comparison = deepcopy(fresh["comparison"])

    def games(source, endpoint, *, arm, include_old_tasks, check_boundary, **options):
        assert include_old_tasks is True and source.completed_updates == 0
        assert (endpoint.completed_updates, endpoint.training_examples) == (8, 180)
        assert [h.fit_rows for h in endpoint.heads] == [36, 36]
        check_boundary()
        seen.append((arm, options["budget"]))
        return phase(arm)

    monkeypatch.setattr(evaluation, "evaluate_context_linear_rehearsal_checkpoint", games)
    result = run()
    assert result["status"] == "complete", result["reason"]
    assert fresh["calls"] == ["training", "collection", "private"]
    assert [a for a, _ in seen] == ["current_only", "joint_ABC_F"]
    assert seen[0][1] is seen[1][1]
    assert result["private_comparison"] == original_comparison
    assert result["TRAIN_reference_unchanged"] is True
    assert (
        result["readout_budget"]["updates_completed"]
        == result["readout_budget"]["replay_examples_completed"]
        == 0
    )
    assert (
        result["readout_budget"]["started_at"]
        == result["readout_budget"]["observed_budget_started_at"]
    )
    assert len(result["games"]) == 2 and all(len(p["episodes"]) == 8 for p in result["games"])


def test_partial_comparison_refuses_games_without_renewing_or_refitting(fresh, monkeypatch):
    fresh["comparison"]["arms"][0]["status"] = "failed"
    monkeypatch.setattr(
        evaluation,
        "evaluate_context_linear_rehearsal_checkpoint",
        lambda *a, **k: pytest.fail("game"),
    )
    result = run()
    assert result["status"] == "failed" and result["games"] is None
    assert result["private_comparison"]["budget"]["updates_completed"] == 4
    assert result["training"]["status"] == "complete"


@pytest.mark.parametrize(
    "fault",
    [
        "comparison",
        "protocol",
        "charge",
        "clock",
        "source",
        "prior_phase",
        "wrong_roster",
        "partial",
    ],
)
def test_keeps_paid_comparison_and_game_prefix_when_second_arm_boundary_fails(
    fresh, monkeypatch, fault
):
    paid = deepcopy(fresh["comparison"])
    seen = []
    first = phase("current_only")
    saved_first = deepcopy(first)

    def games(source, endpoint, *, arm, check_boundary, **options):
        seen.append(arm)
        check_boundary()
        if arm == "current_only":
            return first
        if fault == "comparison":
            fresh["comparison"]["budget"] = dict(fresh["comparison"]["budget"])
        elif fault == "protocol":
            values = dict(
                zip(
                    check_boundary.__code__.co_freevars,
                    (c.cell_contents for c in check_boundary.__closure__),
                )
            )
            values["report"]["protocol"]["all_old_tasks"] = False
        elif fault == "charge":
            options["budget"].updates_completed = 1
        elif fault == "clock":
            options["budget"].started_at -= 1
        elif fault == "source":
            source.restore_state(endpoint)
        elif fault == "prior_phase":
            first["episodes"] = deepcopy(first["episodes"])
        elif fault == "wrong_roster":
            result = phase(arm)
            result["episodes"][-1]["context"] = "A"
            return result
        else:
            raise evaluation.MiniGridEvaluationFailed(
                dict(status="failed", reason="lost step", episodes=[dict(paid_steps=1)])
            )
        check_boundary()
        pytest.fail("changed paid evidence accepted")

    monkeypatch.setattr(evaluation, "evaluate_context_linear_rehearsal_checkpoint", games)
    result = run()
    assert result["status"] == "failed" and seen == ["current_only", "joint_ABC_F"]
    assert result["games"][0] == saved_first
    assert result["private_comparison"] == paid
    assert result["protocol"]["all_old_tasks"] is True
    if fault == "charge":
        assert result["readout_budget"]["updates_completed"] == 1
    if fault == "partial":
        assert result["games"][1]["episodes"][0]["paid_steps"] == 1


@pytest.mark.parametrize(
    "options",
    [
        ["--example-affine-play"],
        ["--example-affine-recurring"],
        ["--example-affine-a4"],
        ["--example-affine-adaptation", "--probe-update", "minimum_change"],
        ["--example-affine-adaptation", "--train-update", "minimum_change"],
    ],
)
def test_cli_rejects_unsupported_choices_before_work(monkeypatch, options):
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("IO"))
    monkeypatch.setattr(workflow, "run_workflow", lambda **k: pytest.fail("work"))
    monkeypatch.setattr(sys, "argv", ["control", *options, "--all-old-tasks", "--describe"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_cli_describes_and_forwards_only_explicit_selected_mode(monkeypatch, capsys):
    args = ["control", "--example-affine-adaptation", "--all-old-tasks"]
    monkeypatch.setattr(sys, "argv", [*args, "--describe"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["games"]["games_per_phase"] == 8
    seen = []

    def execute(**options):
        seen.append(options)
        return dict(status="complete")

    monkeypatch.setattr(workflow, "run_workflow", execute)
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    assert seen == [dict(private_adaptation=True, all_old_tasks=True)]


@pytest.mark.parametrize("selected", [False, True])
def test_full_old_output_preserves_every_value_within_original_byte_cap(
    tmp_path, monkeypatch, capsys, selected
):
    # Literal metadata-only size fixture: no learner, forward or game execution.
    result = dict(status="complete", literal_rows=[[0] * 7] * 60000)
    assert len(json.dumps(result, indent=2).encode()) > 4 * 1024**2
    assert len(json.dumps(result, separators=(",", ":")).encode()) < 4 * 1024**2
    output = tmp_path / "output.json"
    monkeypatch.setattr(workflow, "run_workflow", lambda **k: result)
    args = ["control", "--example-affine-adaptation", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", args + (["--all-old-tasks"] if selected else []))
    if selected:
        cli.main()
        assert json.loads(output.read_bytes())["literal_rows"] == result["literal_rows"]
        assert len(output.read_bytes()) < 4 * 1024**2
        assert json.loads(capsys.readouterr().out) == json.loads(output.read_bytes())
    else:
        with pytest.raises(ValueError, match="exceeds4MiB"):
            cli.main()
        assert not output.exists()


def test_compact_mode_still_refuses_over_four_mib_before_output(tmp_path, monkeypatch):
    output = tmp_path / "output.json"
    monkeypatch.setattr(
        workflow, "run_workflow", lambda **k: dict(status="complete", literal="x" * (4 * 1024**2))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["control", "--example-affine-adaptation", "--all-old-tasks", "--output", str(output)],
    )
    with pytest.raises(ValueError, match="exceeds4MiB"):
        cli.main()
    assert not output.exists()


@pytest.mark.parametrize("arm", ["current_only", "joint_ABC_F"])
def test_binds_only_full_original_matched_endpoint_to_fixed_all_old_loop(monkeypatch, arm):
    owner, saved = ContextLinearLearner(), state()

    def loop(source, owned, **options):
        assert source is owner and owned == saved and owned is not saved
        assert options["layouts"] == options["sources"] == ()
        assert options["require_initial_source"] is options["public_history"] is True
        assert [(l.context, q, r) for l, q, r in options["live_plan"]] == roster()
        assert options["return_metadata"]["include_old_tasks"] is True
        options["extra_check"]()
        return dict(status="complete", public_history={})

    monkeypatch.setattr(evaluation, "_run_checkpoint", loop)
    result = evaluation.evaluate_context_linear_rehearsal_checkpoint(
        owner, saved, arm=arm, budget=budget(), check_boundary=lambda: None, include_old_tasks=True
    )
    assert "old ABC" in result["public_history"]["storage_scope"]


@pytest.mark.parametrize(
    "options",
    [
        dict(include_old_tasks=1),
        dict(retained_a4=True),
        dict(short_return_phase="after_probe"),
        dict(qualified_boundary=lambda: None),
    ],
)
def test_does_not_expand_short_retained_or_observer_admission(monkeypatch, options):
    monkeypatch.setattr(
        port, "copy_context_linear_training_snapshot", lambda *a: pytest.fail("copy")
    )
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("loop"))
    with pytest.raises(ValueError, match="include_old_tasks"):
        evaluation.evaluate_context_linear_rehearsal_checkpoint(
            ContextLinearLearner(),
            None,
            arm="current_only",
            budget=budget(),
            check_boundary=lambda: None,
            **(dict(include_old_tasks=True) | options),
        )


def test_refuses_current_isolated_8_126_geometry_before_readout(monkeypatch):
    saved = state()
    saved = replace(
        saved,
        training_examples=126,
        heads=tuple(replace(h, fit_rows=9, singular_values=(0.0,) * 9) for h in saved.heads),
    )
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("loop"))
    with pytest.raises(ValueError, match="actual8/180"):
        evaluation.evaluate_context_linear_rehearsal_checkpoint(
            ContextLinearLearner(),
            saved,
            arm="current_only",
            budget=budget(),
            check_boundary=lambda: None,
            include_old_tasks=True,
        )


def test_executes_original_loop_for_both_arms_with_scripted_predictions_only(
    monkeypatch, predictions
):
    monkeypatch.setattr(AllContextEnvironment, "work", {})
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", AllContextEnvironment)
    for arm in ("current_only", "joint_ABC_F"):
        source = ContextLinearLearner()
        result = evaluation.evaluate_context_linear_rehearsal_checkpoint(
            source,
            state(),
            arm=arm,
            budget=budget(),
            check_boundary=lambda: None,
            include_old_tasks=True,
        )
        assert result["status"] == "complete" and result["state_unchanged"] is True
        assert [(r["context"], r["cue"], r["source_role"]) for r in result["episodes"]] == roster()
        assert all(r["environment"]["closed"] for r in result["episodes"])
        assert result["work"]["environment_steps_returned"] == 16
        assert source.completed_updates == 0
    assert predictions["fits"] == []
