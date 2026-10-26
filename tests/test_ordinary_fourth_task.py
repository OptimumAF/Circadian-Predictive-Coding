"""Original ordinary TRAIN G semantics and scripted integration, never outcomes."""

from dataclasses import asdict, replace
from types import ModuleType
import sys
from copy import deepcopy
from typing import Any

import pytest

from scripts import minigrid_affine_workflow as workflow
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_context_linear_learner as port
from src.core import minigrid_key_door as layouts
from src.infra import minigrid_key_door as native
from test_after_c_linear_games import state as abc_state
from test_context_linear_causal_games import budget
from test_recurring_affine_stream import sources
from test_catalogue_capacity import sources as abc_sources


def state():
    base = abc_state()
    heads = tuple(replace(h, fit_rows=36, singular_values=(0.0,) * 36) for h in base.heads)
    return replace(base, heads=heads, completed_updates=10, training_examples=252)


@pytest.mark.parametrize(
    "changes",
    [
        dict(branch_row=2),
        dict(branch_row=3),
        dict(branch_row=4),
        dict(branch_row=True),
        dict(branch_row=5.0),
        dict(rotation=1),
        dict(swap_keys=True),
        dict(wall_color="blue"),
    ],
)
def test_should_keep_g_an_exact_distinct_original_train_geometry(changes):
    declared = layouts.predefined_recurring_train_layout()
    assert asdict(declared) == dict(
        context="G_train", rotation=0, branch_row=5, swap_keys=False, wall_color="grey"
    )
    with pytest.raises(ValueError):
        replace(declared, **changes)
    with pytest.raises(ValueError):
        replace(layouts.predefined_layouts()[0], branch_row=5)
    assert all(
        item.branch_row != 5
        for item in (
            *layouts.predefined_layouts(),
            *layouts.predefined_probe_layouts(),
            *layouts.predefined_distinct_probe_layouts(),
        )
    )


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test", True, None])
def test_should_refuse_g_nontrain_roles_before_native_construction(monkeypatch, role):
    monkeypatch.setattr(
        native._MiniGridEpisode, "__init__", lambda *a, **k: pytest.fail("physics reached")
    )
    with pytest.raises(ValueError, match="TRAIN"):
        native.MiniGridKeyDoor(layouts.predefined_recurring_train_layout(), "blue", role=role)


@pytest.mark.parametrize("context,branch,cue_position", [("G_train", 5, (1, 5)), ("A", 3, (2, 5))])
def test_should_keep_both_keys_and_one_cue_in_original_grid_factory_with_scripted_native_classes(
    monkeypatch, context, branch, cue_position
):
    class Cell:
        def __init__(self, color=None, **options):
            self.color = color

    class Grid:
        def __init__(self, width, height):
            self.cells = {}

        def set(self, x, y, value):
            self.cells[x, y] = value

    class Environment:
        grid: Any

        def __init__(self, **options):
            self.options = options

        def put_obj(self, value, x, y):
            self.grid.set(x, y, value)

    cells = {name: type(name, (Cell,), {}) for name in ("Ball", "Door", "Goal", "Key", "Wall")}
    modules: dict[str, dict[str, Any]] = {
        "minigrid": {},
        "minigrid.core": {},
        "minigrid.minigrid_env": dict(MiniGridEnv=Environment),
        "minigrid.core.grid": dict(Grid=Grid),
        "minigrid.core.mission": dict(MissionSpace=lambda **k: k),
        "minigrid.core.world_object": cells,
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(native, "_require_minigrid", lambda: None)
    layout = (
        layouts.predefined_recurring_train_layout()
        if context == "G_train"
        else layouts.predefined_layouts()[0]
    )
    environment = native._make_key_door_environment(layout, "blue", 16)
    environment._gen_grid(7, 7)
    grid = environment.grid.cells
    assert environment._cue_position == cue_position
    assert type(grid[2, branch]) is cells["Key"] and grid[2, branch].color == "blue"
    assert type(grid[4, branch]) is cells["Key"] and grid[4, branch].color == "red"
    assert type(grid[cue_position]) is cells["Ball"]
    assert sum(type(value) is cells["Ball"] for value in grid.values()) == 1


def test_should_bind_actual_recurring_endpoint_to_existing_g_then_a_loop(monkeypatch):
    owner, saved = port.ContextLinearLearner(), state()

    def run(source, current, **options):
        assert source is owner and current == saved and current is not saved
        assert options["role"] == "knownTRAIN_GA_return" and options["updates"] == 10
        assert [(l.context, c, r) for l, c, r in options["live_plan"]] == [
            (context, cue, "train") for context in ("G_train", "A") for cue in ("blue", "red")
        ]
        assert options["sources"] == options["layouts"] == ()
        assert options["require_initial_source"] is options["public_history"] is True
        assert options["return_metadata"]["source_fit_rows"] == 252
        assert options["memory_copier"](owner.snapshot_state()).heads == (None, None)
        options["extra_check"]()
        return dict(status="complete")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    assert (
        evaluation.evaluate_context_linear_stream_checkpoint(
            owner, saved, budget=budget(), check_boundary=lambda: None, recurring_catalogue=True
        )["status"]
        == "complete"
    )


@pytest.mark.parametrize("fault", ["count", "rows", "head", "failed", "flag", "conflict"])
def test_should_refuse_incompatible_recurring_checkpoint_before_game_loop(monkeypatch, fault):
    saved, after = state(), False
    flag: Any = True
    if fault == "count":
        saved = replace(saved, completed_updates=8)
    elif fault == "rows":
        saved = replace(saved, training_examples=180)
    elif fault == "head":
        saved = replace(saved, heads=abc_state().heads)
    elif fault == "failed":
        saved = replace(saved, failed_work=(1, 36, 0, 0, True))
    elif fault == "flag":
        flag = 1
    else:
        after = True
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("loop reached"))
    with pytest.raises(ValueError):
        evaluation.evaluate_context_linear_stream_checkpoint(
            port.ContextLinearLearner(),
            saved,
            budget=budget(),
            check_boundary=lambda: None,
            recurring_catalogue=flag,
            after_c_capacity=after,
        )


@pytest.mark.parametrize("fault", ["clock", "counter"])
def test_should_preserve_original_zero_update_budget_binding(monkeypatch, fault):
    session = budget()

    def run(*args, **options):
        if fault == "clock":
            session.clock = lambda: 1.0
        else:
            session.updates_completed = 1
        options["extra_check"]()
        pytest.fail("changed readout authority accepted")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    with pytest.raises(ValueError, match="budget binding"):
        evaluation.evaluate_context_linear_stream_checkpoint(
            port.ContextLinearLearner(),
            state(),
            budget=session,
            check_boundary=lambda: None,
            recurring_catalogue=True,
        )


class CollectionEnvironment:
    def __init__(self, layout, cue, **options):
        assert layout == layouts.predefined_recurring_train_layout()
        assert options == dict(
            seed=17, max_steps=16, role="train", episode_id=f"minigrid-source-G_train-{cue}"
        )
        history = sources()[-2 if cue == "blue" else -1]
        self.episode = replace(
            history,
            transitions=tuple(
                replace(
                    t, observation=t.observation[:148], next_observation=t.next_observation[:148]
                )
                for t in history.transitions
            ),
        )
        self.cue, self.steps, self.closed = cue, 0, False

    @property
    def observation(self):
        return self.episode.transitions[0].observation

    @property
    def terminal(self):
        return self.steps == 9

    def step(self, action):
        assert action == 4
        self.steps += 1

    def to_demonstration(self):
        assert self.terminal
        return self.episode

    def close(self):
        self.closed = True

    @property
    def report(self):
        return dict(
            status="complete" if self.terminal else "running",
            closed=self.closed,
            role="train",
            context="G_train",
            cue=self.cue,
            episode_id=self.episode.episode_id,
            total_return=0.5,
            environment_steps=self.steps,
            environment_steps_attempted=self.steps,
        )


def scripted_collector(monkeypatch):
    class Teacher:
        def choose_action(self, observation):
            return 4

    monkeypatch.setattr(layouts, "CueKeyDoorTeacher", Teacher)
    monkeypatch.setattr(native, "MiniGridKeyDoor", CollectionEnvironment)


def test_should_collect_original_train_roles_and_close_before_history_admission(monkeypatch):
    scripted_collector(monkeypatch)
    reports: list[dict[str, Any]] = []
    checks = []
    packets = workflow._collect_recurring_train_sources(
        lambda: checks.append(len(reports)), reports
    )
    assert [e.episode_id for e in packets] == [
        f"minigrid-source-G_train-{cue}" for cue in ("blue", "red")
    ]
    assert all(
        e.role == "train"
        and e.context == "G_train"
        and len(e.transitions) == 9
        and len(e.transitions[0].observation) == 296
        for e in packets
    )
    assert all(
        r["status"] == "complete" and r["closed"] is True and r["environment_steps"] == 9
        for r in reports
    )
    assert checks[-1] == 2


@pytest.mark.parametrize("fault", ["construction", "close"])
def test_should_keep_paid_collection_prefix_and_failed_cleanup_without_admitting_source(
    monkeypatch, fault
):
    scripted_collector(monkeypatch)

    class Broken(CollectionEnvironment):
        def __init__(self, layout, cue, **options):
            if fault == "construction" and cue == "red":
                raise RuntimeError("scripted construction failed")
            super().__init__(layout, cue, **options)

        def close(self):
            if fault == "close":
                raise RuntimeError("scripted cleanup failed")
            super().close()

    monkeypatch.setattr(native, "MiniGridKeyDoor", Broken)
    reports: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError):
        workflow._collect_recurring_train_sources(lambda: None, reports)
    if fault == "construction":
        assert len(reports) == 2 and reports[0]["closed"] is True
        assert reports[1]["status"] == "construction_failed" and reports[1]["closed"] is None
        assert reports[1]["failure_work_may_be_unobserved"] is True
    else:
        assert len(reports) == 1 and reports[0]["closed"] is False
        assert "cleanup failed" in reports[0]["cleanup_error"]


@pytest.fixture
def scripted_workflow(monkeypatch):
    box = dict(
        calls=[],
        state=state(),
        training=dict(
            status="complete",
            source_unchanged=True,
            actor_unchanged=True,
            receipt_capture_error=None,
            committed_receipts=[{} for _ in range(10)],
            budget=dict(updates_completed=10),
        ),
    )
    monkeypatch.setattr(workflow, "_require_minigrid", lambda: None)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: abc_sources())

    def collect(check, reports):
        box["calls"].append("collection")
        box["collection"] = reports
        for cue in ("blue", "red"):
            check()
            reports.append(
                dict(status="complete", closed=True, role="train", context="G_train", cue=cue)
            )
        return sources()[-2:]

    def train(episodes, **options):
        box["calls"].append("training")
        assert len(episodes) == 8 and all(e.role == "train" for e in episodes)
        assert options["retain_arrived"] is options["recurring_catalogue"] is True
        assert "abc_catalogue" not in options
        options["check_boundary"]()
        options["endpoint_observer"](box["state"])
        return box["training"]

    def games(source, current, *, budget, check_boundary, **options):
        box["calls"].append("games")
        assert source.snapshot_state().completed_updates == 0 and current is box["state"]
        assert options["recurring_catalogue"] is True and budget.updates_completed == 0
        check_boundary()
        return dict(
            status="complete",
            state_unchanged=True,
            episodes=[
                dict(status="complete", environment=dict(closed=True), success=False)
                for _ in range(4)
            ],
        )

    monkeypatch.setattr(workflow, "_collect_recurring_train_sources", collect)
    monkeypatch.setattr(cli, "run_context_linear_stream", train)
    monkeypatch.setattr(workflow, "evaluate_context_linear_stream_checkpoint", games)
    return box


def test_should_compose_original_train_collection_then_observed_endpoint_and_complete_negative_games(
    scripted_workflow,
):
    result = workflow.run_workflow(recurring_training=True)
    assert result["status"] == "complete", result["reason"]
    assert (
        scripted_workflow["calls"]
        == result["attempted_phases"]
        == ["collection", "training", "games"]
    )
    assert len(result["collection"]) == 2 and len(result["games"]["episodes"]) == 4
    assert result["readout_budget"]["initial_limits"] == result["readout_budget"]["limits"]
    assert (
        result["readout_budget"]["updates_completed"]
        == result["readout_budget"]["replay_examples_completed"]
        == 0
    )
    assert result["protocol"]["new_layout"]["branch_row"] == 5
    assert result["protocol"]["returned_layout"]["branch_row"] == 3
    assert result["protocol"]["new_fit_calls"] == 10 and result["protocol"]["new_fit_rows"] == 252


@pytest.mark.parametrize("fault", ["status", "source", "actor", "receipt", "charge", "endpoint"])
def test_should_refuse_incomplete_observed_train_before_native_games(scripted_workflow, fault):
    if fault == "status":
        scripted_workflow["training"]["status"] = "failed"
    elif fault == "source":
        scripted_workflow["training"]["source_unchanged"] = False
    elif fault == "actor":
        scripted_workflow["training"]["actor_unchanged"] = False
    elif fault == "receipt":
        scripted_workflow["training"]["committed_receipts"].pop()
    elif fault == "charge":
        scripted_workflow["training"]["budget"]["updates_completed"] = 9
    else:
        scripted_workflow["state"] = replace(scripted_workflow["state"], completed_updates=8)
    result = workflow.run_workflow(recurring_training=True)
    assert result["status"] == "failed" and result["games"] is None
    assert scripted_workflow["calls"] == ["collection", "training"]


@pytest.mark.parametrize("fault", ["training", "collection_replacement", "clock", "protocol"])
def test_should_refuse_corrupted_preceding_evidence_or_readout_authority(
    scripted_workflow, monkeypatch, fault
):
    protocol = workflow.describe_workflow(recurring_training=True)
    monkeypatch.setattr(workflow, "describe_workflow", lambda **k: protocol)
    original = deepcopy(scripted_workflow["training"])

    def games(*args, budget, check_boundary, **options):
        if fault == "training":
            scripted_workflow["training"]["budget"]["updates_completed"] = 11
        elif fault == "collection_replacement":
            scripted_workflow["collection"][0] = dict(scripted_workflow["collection"][0])
        elif fault == "clock":
            budget.clock = lambda: 0.0
        else:
            protocol["new_fit_rows"] = 253
        check_boundary()
        pytest.fail("changed evidence admitted")

    monkeypatch.setattr(workflow, "evaluate_context_linear_stream_checkpoint", games)
    result = workflow.run_workflow(recurring_training=True)
    assert result["status"] == "failed"
    assert result["training"] == original
    assert all(r["closed"] is True for r in result["collection"])
    assert result["protocol"]["new_fit_rows"] == 252


def test_should_preserve_partial_native_failure_without_retry_or_paid_train_erasure(
    scripted_workflow, monkeypatch
):
    failed = dict(
        status="failed",
        reason="scripted native failure",
        state_unchanged=False,
        work=dict(failure_work_may_be_unobserved=True),
        episodes=[dict(status="complete", environment=dict(closed=True))],
    )
    calls = []

    def games(*args, **options):
        calls.append(1)
        raise evaluation.MiniGridEvaluationFailed(failed)

    monkeypatch.setattr(workflow, "evaluate_context_linear_stream_checkpoint", games)
    result = workflow.run_workflow(recurring_training=True)
    assert calls == [1] and result["status"] == "failed" and result["games"] == failed
    assert result["games"] is not failed  # Existing exception owns a detached report.
    assert result["training"]["budget"]["updates_completed"] == 10
    assert result["games"]["state_unchanged"] is False


def test_should_check_missing_optional_runtime_before_collection_or_training(monkeypatch):
    def missing():
        raise RuntimeError("missing pinned runtime")

    monkeypatch.setattr(workflow, "_require_minigrid", missing)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("data opened"))
    result = workflow.run_workflow(recurring_training=True)
    assert result["status"] == "failed" and result["attempted_phases"] == []
    assert result["training"] is result["games"] is None and result["collection"] == []


@pytest.mark.parametrize(
    "options",
    [
        dict(recurring_training=1),
        dict(recurring_training=None),
        dict(recurring_training=True, private_adaptation=True),
    ],
)
def test_should_refuse_ambiguous_workflow_modes_before_any_work(options):
    for function in (workflow.run_workflow, workflow.describe_workflow):
        with pytest.raises(ValueError):
            function(**options)
