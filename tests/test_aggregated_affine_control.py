"""Same198 TRAIN composition and permanent attempt accounting; no source reads."""

from dataclasses import replace
import json
from typing import Any

import numpy as np
import pytest

from scripts import fit_aggregated_affine_control as app
from scripts.push_cube_correction_weights import AGGREGATED_GROUPS, AGGREGATED_GROUP_ROWS
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.continuous_affine_fit import ContinuousAffineFit
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.cube_taskspace_state import cube_taskspace_spec
from test_continuous_affine_fit import budgeted_solve as budgeted_solve


def groups():
    spec, zero = cube_taskspace_spec(), (0.0,) * 12
    return tuple(
        (
            name,
            ContinuousEpisode(
                f"literal-{name}",
                name,
                "train",
                spec,
                tuple(
                    ContinuousTransition(i, zero, (0.25,) * 4, 0.0, zero, i == rows - 1, False)
                    for i in range(rows)
                ),
            ),
        )
        for name, rows in zip(AGGREGATED_GROUPS, AGGREGATED_GROUP_ROWS, strict=True)
    )


def budget():
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=1, max_replay_examples=0, max_wall_seconds=60),
        lambda: 0.0,
    )


@pytest.fixture
def scripted(monkeypatch):
    box: dict[str, Any] = dict(calls=[], effect=None)

    def fit(x, y, weights, spec):
        box["calls"].append((x.copy(), y.copy(), weights.copy()))
        if box["effect"]:
            box["effect"]()
        return ContinuousAffineFit(
            spec, ((0.0,) * 4,) * 12 + ((0.25,) * 4,), 198, 1, (1.0,) + (0.0,) * 12, None, 1.0
        )

    monkeypatch.setattr(app, "fit_continuous_affine", fit)
    return box


def test_should_fit_all_original_groups_once_with_exact_six_group_weights_and_actual_math():
    session = budget()
    result = app.fit_groups(groups(), session)
    assert result["complete"], result["error"]
    assert result["source_inputs_unchanged"]
    assert result["work"]["fit_attempts"] == result["work"]["fit_returns"] == 1
    assert result["work"]["fit_rows_attempted"] == result["work"]["fit_rows_returned"] == 198
    assert result["model"]["rank"] == 1 and result["model"]["fit_rows"] == 198
    assert result["metrics"]["weighted_bounded_half_mse"] < 1e-28
    assert [row["rows"] for row in result["groups"]] == [50, 50, 18, 15, 50, 15]
    assert all(row["role"] == "train" for row in result["groups"])
    assert session.updates_completed == 1 and session.replay_examples_completed == 0
    assert result["work"]["input_array_bytes"] == 198 * 17 * 8


def test_should_keep_current_and_prior_correction_prefix_weights_and_inputs_owned(scripted):
    original = groups()
    session = budget()
    result = app.fit_groups(original, session)
    assert result["complete"]
    x, y, weights = scripted["calls"][0]
    assert weights[0] == weights[50] == 49 and weights[1] == weights[51] == 1
    offset = 0
    for count in AGGREGATED_GROUP_ROWS:
        assert np.sum(weights[offset : offset + count]) == pytest.approx(98)
        offset += count
    y[0, 0], x[0, 0] = -1, 100
    assert original[0][1].transitions[0].observation[0] == 0
    assert original[0][1].transitions[0].action[0] == 0.25
    assert result["budget"]["charged_fit_attempts"] == 1


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test", "inner_guard"])
def test_should_refuse_nontrain_before_solve_or_charge(scripted, role):
    original, session = groups(), budget()
    changed = ((original[0][0], replace(original[0][1], role=role)), *original[1:])
    with pytest.raises(ValueError):
        app.fit_groups(changed, session)
    assert session.updates_completed == 0 and not scripted["calls"]


@pytest.mark.parametrize(
    "fault",
    [
        "role",
        "equal_observation",
        "dictionary",
        "budget_birth",
        "equal_limits",
        "bool_charge",
        "recipe",
    ],
)
def test_should_refuse_unobserved_source_budget_or_recipe_changes_after_paid_work(
    scripted, monkeypatch, fault
):
    original, session = groups(), budget()
    if fault == "recipe":
        build = app.build_aggregated_correction_weights

        def changed(*args):
            weights, recipe = build(*args)
            scripted["recipe"] = recipe
            return weights, recipe

        monkeypatch.setattr(app, "build_aggregated_correction_weights", changed)

    def change():
        if fault == "role":
            object.__setattr__(original[0][1], "role", "final_test")
        elif fault == "equal_observation":
            row = original[0][1].transitions[0]
            object.__setattr__(row, "observation", tuple(list(row.observation)))
        elif fault == "dictionary":
            row = original[0][1]
            object.__setattr__(row, "__dict__", dict(vars(row)))
        elif fault == "budget_birth":
            session.started_at += 1
        elif fault == "equal_limits":
            session.budget = replace(session.budget)
        elif fault == "bool_charge":
            session.updates_completed = True
        else:
            scripted["recipe"]["values"][0] = 1

    scripted["effect"] = change
    result = app.fit_groups(original, session)
    assert not result["complete"] and result["model"] is None
    assert result["work"]["fit_attempts"] == result["work"]["fit_returns"] == 1
    assert result["work"]["prediction_attempts"] == 0
    assert result["budget"]["charged_fit_attempts"] == 1
    if fault == "recipe":
        assert result["weight_recipe"]["values"][0] == 49


def test_should_preserve_lost_fit_return_and_refuse_fresh_work_on_same_budget(scripted):
    def lost():
        raise RuntimeError("lost fit return")

    scripted["effect"] = lost
    session = budget()
    result = app.fit_groups(groups(), session)
    assert not result["complete"] and result["model"] is None
    assert result["work"]["fit_attempts"] == 1 and result["work"]["fit_returns"] == 0
    assert result["work"]["fit_rows_attempted"] == 198
    assert result["budget"]["charged_fit_attempts"] == session.updates_completed == 1
    with pytest.raises(ValueError, match="fresh single-fit"):
        app.fit_groups(groups(), session)
    assert len(scripted["calls"]) == 1


def test_should_retain_a_paid_slot_when_charge_acknowledgement_is_lost(scripted, monkeypatch):
    session = budget()
    original = session.record_update

    def charge():
        original()
        raise RuntimeError("lost charge acknowledgement")

    monkeypatch.setattr(session, "record_update", charge)
    result = app.fit_groups(groups(), session)
    assert not result["complete"] and not scripted["calls"]
    assert result["budget"]["charged_fit_attempts"] == session.updates_completed == 1
    assert result["work"]["fit_attempts"] == 0


def test_should_fail_an_expired_original_clock_before_attempt(scripted):
    session, ticks = budget(), [0.0]
    session.clock = lambda: ticks[0]

    def expired():
        ticks[0] = 61.0

    result = app.fit_groups(groups(), session, check_boundary=expired)
    assert not result["complete"] and not scripted["calls"]
    assert session.updates_completed == 0


@pytest.fixture
def file_ports(scripted, monkeypatch):
    box: dict[str, Any] = dict(reads=0, study_final=0, original_final=0)

    def read(*args):
        box["reads"] += 1
        return groups(), dict(
            accepted_study_read_work={}, original_correction=dict(path="literal", bytes=1)
        )

    def final(*args, **options):
        box["study_final"] += 1

    def final_original(*args):
        box["original_final"] += 1

    monkeypatch.setattr(app, "read_aggregated_inputs", read)
    monkeypatch.setattr(app, "_bound_bytes", final)
    monkeypatch.setattr(app, "_require_original_unchanged", final_original)
    return box


def test_should_capture_one_actual_app_call_exclusively_and_never_repeat_reads(
    file_ports, tmp_path
):
    output = tmp_path / "control.json"
    result = app.run_control(output)
    assert result["complete"] and result["source_bytes_unchanged"]
    assert json.loads(output.read_text())["charged_fit_attempts"] == 1
    assert file_ports == dict(reads=1, study_final=1, original_final=1)
    with pytest.raises(ValueError):
        app.run_control(output)
    output.rename(tmp_path / "preserved-control.json")
    with pytest.raises(FileExistsError):
        app.run_control(output)
    assert file_ports == dict(reads=1, study_final=1, original_final=1)


def test_should_withhold_unqualified_model_and_preserve_work_on_source_replacement(
    file_ports, monkeypatch, tmp_path
):
    def changed(*args, **options):
        raise ValueError("source bytes replaced")

    monkeypatch.setattr(app, "_bound_bytes", changed)
    result = app.run_control(tmp_path / "failed.json")
    assert not result["complete"] and result["model"] is None
    assert result["charged_fit_attempts"] == 1 and result["work"]["fit_returns"] == 1
    assert not result["source_bytes_unchanged"]


def test_should_capture_unknown_unreturned_reader_work_without_claiming_zero(
    file_ports, monkeypatch, tmp_path
):
    def failed(*args):
        raise RuntimeError("reader return lost")

    monkeypatch.setattr(app, "read_aggregated_inputs", failed)
    result = app.run_control(tmp_path / "lost-reader.json")
    assert not result["complete"] and result["unreturned_source_read_work_unknown"]
    assert result["charged_fit_attempts"] == 0 and result["work"] is None
