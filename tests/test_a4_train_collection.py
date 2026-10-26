"""Fresh TRAIN collection through the real port with literal, physics-free IO."""

from dataclasses import asdict, replace
from copy import deepcopy
from typing import Any

import pytest

from scripts import minigrid_affine_workflow as workflow
from src.core import minigrid_key_door as layouts
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.infra import minigrid_key_door as native


@pytest.fixture
def collection(monkeypatch):
    box: dict[str, Any] = dict(
        environments=[],
        choices=0,
        fault=None,
        rows=8,
        layout=layouts.MiniGridKeyDoorLayout("A", branch_row=4),
    )

    class Teacher:
        def __init__(self):
            if box["fault"] == "teacher_birth":
                raise RuntimeError("teacher birth")

        def choose_action(self, observation):
            box["choices"] += 1
            if box["fault"] == "teacher_step":
                raise RuntimeError("teacher step")
            return 2

    class Environment:
        def __init__(self, layout, cue, *, seed, max_steps, role, episode_id):
            assert seed == 17 and max_steps == 16 and role == "train"
            assert layout == box["layout"]
            if box["fault"] == "constructor" and cue == "red":
                raise RuntimeError("constructor")
            self.layout, self.cue, self.identity = layout, cue, episode_id
            self.steps, self.closed = 0, False
            box["environments"].append(self)

        @property
        def observation(self):
            return (-1.0,) * 148

        @property
        def terminal(self):
            return self.steps == box["rows"]

        def step(self, action):
            assert action == 2
            self.steps += 1
            if box["fault"] == "physics_step":
                raise RuntimeError("physics step")

        def close(self):
            if box["fault"] == "close":
                raise RuntimeError("close")
            self.closed = True

        def to_demonstration(self):
            assert self.terminal
            episode = DemonstrationEpisode(
                self.identity,
                self.layout.context,
                "train",
                tuple(
                    DemonstrationTransition(
                        i,
                        self.observation,
                        2,
                        0.5 if i + 1 == self.steps else 0.0,
                        self.observation,
                        i + 1 == self.steps,
                        7,
                    )
                    for i in range(self.steps)
                ),
                7,
            )
            if box["fault"] == "episode_role":
                return replace(episode, role="probe_test")
            if box["fault"] == "episode_identity":
                return replace(episode, episode_id="replaced")
            if box["fault"] == "episode_context":
                return replace(episode, context="B")
            return episode

        @property
        def report(self):
            report = dict(
                layout=asdict(self.layout),
                context=self.layout.context,
                role="train",
                cue=self.cue,
                episode_id=self.identity,
                status="complete" if self.terminal else "running",
                terminated=self.terminal,
                truncated=False,
                closed=self.closed,
                environment_steps=self.steps,
                environment_steps_attempted=self.steps,
                total_return=0.0 if box["fault"] == "negative" else 0.5,
                failure_work_may_be_unobserved=False,
            )
            changes: dict[str, dict[str, Any]] = {
                "report_role": dict(role="probe_test"),
                "report_layout": dict(layout=asdict(layouts.MiniGridKeyDoorLayout("A"))),
                "report_steps": dict(environment_steps=self.steps + 1),
                "truncated": dict(truncated=True),
                "unknown": dict(failure_work_may_be_unobserved=True),
                "report_identity": dict(episode_id="replaced"),
                "report_context": dict(context="B"),
            }
            report.update(changes.get(box["fault"], {}))
            if self.closed and box["fault"] in ("return_after_close", "nan_after_close"):
                report["total_return"] = (
                    0.0 if box["fault"] == "return_after_close" else float("nan")
                )
            return deepcopy(report)

    monkeypatch.setattr(layouts, "CueKeyDoorTeacher", Teacher)
    monkeypatch.setattr(native, "MiniGridKeyDoor", Environment)
    return box


def collect(reports, check=lambda: None):
    return workflow._collect_recurring_train_sources(
        check, reports, layout=layouts.MiniGridKeyDoorLayout("A", branch_row=4)
    )


def test_should_collect_fresh_a4_train_for_both_cues_without_assuming_g_length(collection):
    reports: list[dict[str, Any]] = []
    packets = collect(reports)
    assert len(packets) == len(reports) == 2
    assert [p.episode_id for p in packets] == [
        f"minigrid-source-A4-train-{cue}" for cue in ("blue", "red")
    ]
    assert all(p.role == "train" and p.context == "A" and len(p.transitions) == 8 for p in packets)
    assert all(len(p.transitions[0].observation) == 296 for p in packets)
    assert all(r["closed"] is True and r["role"] == "train" for r in reports)
    assert collection["choices"] == 16


@pytest.mark.parametrize(
    "layout",
    [
        layouts.MiniGridKeyDoorLayout("A"),
        layouts.predefined_layouts()[1],
        layouts.predefined_layouts()[3],
        layouts.predefined_distinct_probe_layouts()[0],
        layouts.MiniGridKeyDoorLayout("A", branch_row=4, wall_color="blue"),
        True,
        {},
    ],
)
def test_should_refuse_non_g_a4_geometry_before_any_work(collection, layout):
    reports: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="G or"):
        workflow._collect_recurring_train_sources(
            lambda: pytest.fail("callback"), reports, layout=layout
        )
    assert not reports and not collection["environments"] and collection["choices"] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "episode_role",
        "episode_identity",
        "episode_context",
        "report_role",
        "report_layout",
        "report_steps",
        "report_identity",
        "report_context",
        "return_after_close",
        "nan_after_close",
        "truncated",
        "unknown",
        "negative",
        "teacher_birth",
        "teacher_step",
        "physics_step",
        "close",
    ],
)
def test_should_retain_paid_failed_record_and_close_before_refusing_a4(collection, fault):
    collection["fault"] = fault
    reports: list[dict[str, Any]] = []
    with pytest.raises((ValueError, RuntimeError)):
        collect(reports)
    assert len(collection["environments"]) == len(reports) == 1
    assert reports[0]["closed"] is (fault != "close")
    if fault in {
        "teacher_birth",
        "teacher_step",
        "physics_step",
        "negative",
        "episode_role",
        "episode_identity",
        "episode_context",
    }:
        assert "primary_error" in reports[0]
    if fault == "close":
        assert "cleanup_error" in reports[0]


def test_should_preserve_first_complete_source_when_second_construction_fails(collection):
    collection["fault"] = "constructor"
    reports: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError, match="constructor"):
        collect(reports)
    assert reports[0]["closed"] is True and reports[0]["environment_steps"] == 8
    assert reports[1]["status"] == "construction_failed" and reports[1]["closed"] is None
    assert reports[1]["failure_work_may_be_unobserved"] is True
    assert reports[1]["episode_id"] == "minigrid-source-A4-train-red"


def test_should_close_and_capture_on_boundary_refusal_before_more_steps(collection):
    reports: list[dict[str, Any]] = []

    def guard():
        if collection["choices"] == 1:
            raise RuntimeError("boundary")

    with pytest.raises(RuntimeError, match="boundary"):
        collect(reports, guard)
    assert len(reports) == 1 and reports[0]["closed"] is True
    assert reports[0]["environment_steps"] == 0 and collection["choices"] == 1


def test_should_preserve_original_g_nine_row_contract(collection):
    collection["layout"] = layouts.predefined_recurring_train_layout()
    reports: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="nine"):
        workflow._collect_recurring_train_sources(lambda: None, reports)
    assert len(reports) == 1 and reports[0]["closed"] is True
    assert reports[0]["environment_steps"] == 8


def test_should_detach_layout_before_caller_boundary_can_change_input(collection):
    declared = layouts.MiniGridKeyDoorLayout("A", branch_row=4)
    reports: list[dict[str, Any]] = []

    def guard():
        object.__setattr__(declared, "branch_row", 3)

    packets = workflow._collect_recurring_train_sources(guard, reports, layout=declared)
    assert declared.branch_row == 3 and len(packets) == 2
    assert all(r["layout"]["branch_row"] == 4 for r in reports)
