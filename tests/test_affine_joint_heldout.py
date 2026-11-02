"""Fixed joint-only/final isolation with literal models and scripted ports."""

from copy import deepcopy
from dataclasses import replace
import inspect
import json
import sys

import pytest

from scripts import minigrid_affine_workflow as workflow
from scripts import minigrid_affine_probe_games as saved
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_affine_rehearsal as private
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from test_after_c_linear_games import state as reference
from test_affine_rehearsal_return_games import state as joint_state
from test_retained_context_linear_games import budget
from test_catalogue_capacity import sources
from test_affine_private_rehearsal import probe_sources
from test_isolated_affine_workflow import AllContextEnvironment
from test_minigrid_invariant_evaluation import public_frame
import test_portable_affine_adaptation as old_workflow
import test_minigrid_minimum_change_probe as delegates

scripted = old_workflow.scripted
predictions = delegates.scripted
REAL_COMPARE = private.compare_private_rehearsal


class FinalEnvironment(AllContextEnvironment):
    def _frame(self, step):
        return public_frame(self.cue, step, wall_color=4, rotation=self.layout.rotation)


@pytest.fixture
def wired(scripted, predictions, monkeypatch):
    monkeypatch.setattr(private, "compare_private_rehearsal", REAL_COMPARE)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", FinalEnvironment)
    monkeypatch.setattr(FinalEnvironment, "work", {})
    return scripted, predictions


def run():
    return workflow.run_workflow(private_adaptation=True, joint_heldout=True)


def test_joint_factory_executes_two_fits_and_preserves_original_two_arm_default(predictions):
    result = private.compare_private_rehearsal(
        sources(), reference(), probe_sources(), fresh_reference=True, joint_only=True
    )
    assert result["status"] == "complete", result["reason"]
    assert [a["arm"] for a in result["arms"]] == ["joint_ABC_F"]
    assert result["budget"]["initial_limits"]["max_training_updates"] == 2
    assert result["budget"]["updates_completed"] == 2
    assert result["budget"]["replay_examples_completed"] == 54
    assert len(predictions["fits"]) == 2
    values = saved._rehearsal_states(result, joint_only=True)
    assert len(values) == 1 and (values[0].completed_updates, values[0].training_examples) == (
        8,
        180,
    )
    assert [h.fit_rows for h in values[0].heads] == [36, 36]
    with pytest.raises(ValueError):
        saved._rehearsal_states(result)
    assert private.describe_protocol()["arms"] == ("current_only", "joint_ABC_F")
    old = private.compare_private_rehearsal(
        sources(), reference(), probe_sources(), fresh_reference=True
    )
    assert old["status"] == "complete" and len(old["arms"]) == 2
    assert len(predictions["fits"]) == 6  # Two joint fits plus the unchanged four-fit comparison.


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(joint_only=1),
        dict(joint_only=True),
        dict(joint_only=True, fresh_reference=True, retained_a4=True),
    ],
)
def test_joint_only_mode_rejects_unbound_recipes_before_work(kwargs):
    with pytest.raises(ValueError):
        private.describe_protocol(**kwargs)


def test_joint_reader_refuses_unobserved_charge_and_equal_endpoint_replacement(
    predictions, monkeypatch
):
    result = private.compare_private_rehearsal(
        sources(), reference(), probe_sources(), fresh_reference=True, joint_only=True
    )
    bad = deepcopy(result)
    bad["budget"]["updates_completed"] = 1
    with pytest.raises(ValueError):
        saved._rehearsal_states(bad, joint_only=True)
    convert = cli._context_linear_training_snapshot
    monkeypatch.setattr(
        cli,
        "_context_linear_training_snapshot",
        lambda x: replace(convert(x), training_examples=181),
    )
    with pytest.raises(ValueError):
        saved._rehearsal_states(result, joint_only=True)


def test_joint_final_facade_runs_only_two_final_games_and_preserves_old_admission(
    predictions, monkeypatch
):
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", FinalEnvironment)
    monkeypatch.setattr(FinalEnvironment, "work", {})
    source = ContextLinearLearner()
    result = evaluation.evaluate_heldout_context_linear_checkpoint(
        source,
        joint_state(),
        budget=budget(),
        check_boundary=lambda: None,
        joint_private_endpoint=True,
    )
    assert result["status"] == "complete" and result["state_unchanged"] is True
    assert result["source_fit_calls"] == 8 and result["source_fit_rows"] == 180
    assert [(e["context"], e["cue"], e["source_role"]) for e in result["episodes"]] == [
        ("D", "blue", "final_test"),
        ("D", "red", "final_test"),
    ]
    assert all(e["environment"]["closed"] for e in result["episodes"])
    assert source.completed_updates == 0 and predictions["fits"] == []
    with pytest.raises(ValueError):
        evaluation.evaluate_heldout_context_linear_checkpoint(
            source, joint_state(), budget=budget(), check_boundary=lambda: None
        )
    with pytest.raises(ValueError):
        evaluation.evaluate_heldout_context_linear_checkpoint(
            source,
            replace(joint_state(), completed_updates=10, training_examples=252),
            budget=budget(),
            check_boundary=lambda: None,
            joint_private_endpoint=True,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(private_adaptation=True, all_old_tasks=True),
        dict(private_adaptation=True, probe_update="minimum_change"),
        dict(private_adaptation=True, train_update="minimum_change"),
        dict(recurring_training=True),
        dict(a4_adaptation=True),
    ],
)
def test_workflow_heldout_intent_refuses_other_modes_before_sources(monkeypatch, kwargs):
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source"))
    for action in (workflow.run_workflow, workflow.describe_workflow):
        with pytest.raises(ValueError, match="joint_heldout"):
            action(joint_heldout=True, **kwargs)


def test_cli_metadata_and_dispatch_keep_joint_fixed_before_final(monkeypatch, capsys):
    args = ["control", "--example-affine-adaptation", "--heldout-joint"]
    monkeypatch.setattr(sys, "argv", [*args, "--describe"])
    cli.main()
    meta = json.loads(capsys.readouterr().out)
    assert meta["joint_heldout"] is True and meta["private"]["arms"] == ["joint_ABC_F"]
    assert (meta["new_fit_calls"], meta["new_fit_rows"]) == (8, 180)
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        return dict(status="complete")

    monkeypatch.setattr(workflow, "run_workflow", execute)
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    assert calls == [dict(private_adaptation=True, joint_heldout=True)]
    monkeypatch.setattr(sys, "argv", ["control", "--heldout-joint", "--describe"])
    with pytest.raises(SystemExit) as bad:
        cli.main()
    assert bad.value.code == 2


def test_real_private_factory_and_final_loop_fit_only_before_final(wired, monkeypatch):
    box, predictions = wired
    before = deepcopy(box["state"])
    evaluate = evaluation.evaluate_heldout_context_linear_checkpoint
    points = []

    def observe(*a, **k):
        assert len(predictions["fits"]) == 2 and k["joint_private_endpoint"] is True
        points.append(len(predictions["fits"]))
        return evaluate(*a, **k)

    monkeypatch.setattr(evaluation, "evaluate_heldout_context_linear_checkpoint", observe)
    monkeypatch.setattr(
        evaluation,
        "evaluate_context_linear_rehearsal_checkpoint",
        lambda *a, **k: pytest.fail("known-game rerun"),
    )
    result = run()
    assert result["status"] == "complete", result["reason"]
    assert box["state"] == before and result["TRAIN_reference_unchanged"] is True
    assert result["private_comparison"]["budget"]["updates_completed"] == 2
    assert [a["arm"] for a in result["private_comparison"]["arms"]] == ["joint_ABC_F"]
    assert points == [2] and len(predictions["fits"]) == 2
    assert len(result["games"]) == 1 and len(result["games"][0]["episodes"]) == 2


@pytest.mark.parametrize("fault", ["comparison", "endpoint", "collection"])
def test_corrupt_prior_work_refuses_all_final_games(wired, monkeypatch, fault):
    box, predictions = wired
    if fault == "collection":
        collect = private.original.reserved._collect_probe_sources

        def corrupt_collection(*a, **k):
            data = collect(*a, **k)
            a[1][0]["closed"] = False
            return data

        monkeypatch.setattr(private.original.reserved, "_collect_probe_sources", corrupt_collection)
    else:
        convert = saved._rehearsal_states

        def corrupt_endpoint(report, **kwargs):
            values = convert(report, **kwargs)
            if fault == "comparison":
                report["arms"][0] = deepcopy(report["arms"][0])
            else:
                values = (replace(values[0], training_examples=181),)
            return values

        monkeypatch.setattr(saved, "_rehearsal_states", corrupt_endpoint)
    monkeypatch.setattr(
        evaluation,
        "evaluate_heldout_context_linear_checkpoint",
        lambda *a, **k: pytest.fail("final admitted"),
    )
    result = run()
    assert result["status"] == "failed" and result["games"] is None


def test_late_boundary_callback_cannot_change_paid_endpoint(wired, monkeypatch):
    evaluate = evaluation.evaluate_heldout_context_linear_checkpoint

    def corrupt(*a, **k):
        outer = inspect.getclosurevars(k["check_boundary"]).nonlocals["check_boundary"]
        values = inspect.getclosurevars(outer).nonlocals
        endpoint = values["heldout_binding"][0]
        memory = values["memory"]
        sample = memory.sample
        fired: list[bool] = []

        def changed():
            value = sample()
            if not fired:
                fired.append(True)
                object.__setattr__(endpoint, "training_examples", 181)
            return value

        monkeypatch.setattr(memory, "sample", changed)
        return evaluate(*a, **k)

    monkeypatch.setattr(evaluation, "evaluate_heldout_context_linear_checkpoint", corrupt)
    result = run()
    assert (
        result["status"] == "failed"
        and result["private_comparison"]["budget"]["updates_completed"] == 2
    )


def test_final_physics_failure_preserves_joint_prefix_without_retry(wired, monkeypatch):
    def lost(self, action):
        raise RuntimeError("scripted final step lost return")

    monkeypatch.setattr(FinalEnvironment, "step", lost)
    result = run()
    assert result["status"] == "failed"
    assert result["private_comparison"]["budget"]["updates_completed"] == 2
    assert len(wired[1]["fits"]) == 2
    assert result["games"][0]["episodes"][0]["environment"]["closed"] is True


@pytest.mark.parametrize("fault", ["success", "equal_replacement"])
def test_returned_final_phase_is_pinned_before_boundary_callbacks(wired, monkeypatch, fault):
    evaluate = evaluation.evaluate_heldout_context_linear_checkpoint
    observed = []

    def intercepted(*a, **k):
        phase = evaluate(*a, **k)
        observed.append(deepcopy(phase))
        outer = inspect.getclosurevars(k["check_boundary"]).nonlocals["check_boundary"]
        memory = inspect.getclosurevars(outer).nonlocals["memory"]
        sample = memory.sample
        fired: list[bool] = []

        def mutate():
            value = sample()
            if not fired:
                fired.append(True)
                if fault == "success":
                    phase["episodes"][0]["success"] = not phase["episodes"][0]["success"]
                else:
                    phase["episodes"] = deepcopy(phase["episodes"])
            return value

        monkeypatch.setattr(memory, "sample", mutate)
        return phase

    monkeypatch.setattr(evaluation, "evaluate_heldout_context_linear_checkpoint", intercepted)
    result = run()
    assert result["status"] == "failed" and result["final_boundary_refused"] is True
    assert result["games"] == observed
    assert len(wired[1]["fits"]) == 2
