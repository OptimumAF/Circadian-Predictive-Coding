"""Literal/AST gates for the frozen cold matched case, no numerical imports."""

import ast
from copy import deepcopy
from functools import wraps
import gzip
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from test_isolated_affine_native import fixture as source_fixture

STAGE = (
    Path(__file__).resolve().parents[1] / "artifacts/runs/minigrid-matched-all-old-native-20261031"
)
spec = importlib.util.spec_from_file_location("matched_game_case", STAGE / "case_checks.py")
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixture(success=False):
    r, _, _, info, _ = source_fixture(success)
    p = r.pop("private_probe")
    r["collection"] = p["collection"]
    r["protocol"] = dict(
        all_old_tasks=True, new_fit_calls=10, new_fit_rows=252, games=dict(games_per_phase=8)
    )
    r["attempted_phases"] = ["training", "collection", "private_comparison", "games"]

    def budget(n, replay):
        limits = gate.limits(n, 90, replay)
        return dict(
            initial_limits=limits.copy(),
            limits=limits,
            updates_completed=n,
            replay_examples_completed=replay,
            started_at=1.0,
            observed_budget_started_at=1.0,
            ended_at=2.0,
        )

    arms, phases = [], []
    for index, name in enumerate(gate.ARMS):
        replay = 54 if index else 0
        endpoint = deepcopy(p["disposable_state"])
        endpoint["training_examples"] = 180
        endpoint["heads"] = [dict(fit_rows=36)] * 2
        probe = deepcopy(p["probe"])
        probe["completed_training_examples"] = 72
        arms.append(
            dict(
                arm=name,
                status="complete",
                source_native_state_unchanged=True,
                source_fork_calls=1,
                budget=budget(2, replay),
                updates_accounted_by_study=2,
                committed_old_TRAIN_reuse_rows=replay,
                committed_extra_F_rows=54 - replay,
                attempts=[
                    dict(
                        update=n,
                        fit_rows_attempted=36,
                        old_TRAIN_reuse_charged=replay // 2,
                        extra_F_exposures_charged=(54 - replay) // 2,
                    )
                    for n in (1, 2)
                ],
                native_endpoint=endpoint,
                probe=probe,
                old_task_readouts=deepcopy(p["old_task_readouts"]),
            )
        )
        phase = deepcopy(r["games"][0])
        phase.update(rehearsal_arm=name, source_fit_rows=180)
        phases.append(phase)
    r["games"] = phases
    r["private_comparison"] = dict(
        status="complete",
        arms=arms,
        budget=budget(4, 54),
        protocol=dict(
            reference_origin="current_workflow_actual_TRAIN_endpoint",
            upstream_paid_fits=6,
            upstream_paid_rows=108,
        ),
    )
    info["private_births"] = [dict(started_at=1.0)] * 2
    info["readout_births"] = [dict(started_at=1.0)] * 2
    info["shared_readout_owner"] = True
    counts = gate.EXACT | dict(steps=36 + 16 * (9 if success else 16))
    return r, counts, counts.copy(), info, 20


@pytest.mark.parametrize("success", [False, True])
def test_accepts_complete_all_negative_or_positive_fixed_arms(success):
    gate.validate_case(*fixture(success))


@pytest.mark.parametrize(
    "fault",
    [
        "arm_order",
        "replay_fee",
        "extra_F",
        "arm_birth",
        "handoff",
        "unreturned",
        "warm",
        "final_role",
        "seed",
        "source_reference",
        "private_rows",
        "source_clock",
        "unclosed",
        "readout_owner",
    ],
)
def test_refuses_broken_fixed_recipe_costs_and_evidence(fault):
    r, c, returned, info, closed = fixture()
    s = r["private_comparison"]
    if fault == "arm_order":
        s["arms"].reverse()
    elif fault == "replay_fee":
        s["arms"][1]["attempts"][0]["old_TRAIN_reuse_charged"] = 0
    elif fault == "extra_F":
        s["arms"][0]["committed_extra_F_rows"] = 0
    elif fault == "arm_birth":
        s["arms"][1]["budget"]["started_at"] = 0.0
    elif fault == "handoff":
        info["probe_handoff_matches"] = False
    elif fault == "unreturned":
        returned["solves"] -= 1
    elif fault == "warm":
        c["warm"] = returned["warm"] = 1
    elif fault == "final_role":
        r["games"][1]["episodes"][7]["environment"]["role"] = "final_test"
    elif fault == "seed":
        r["games"][1]["episodes"][6]["environment"]["seed"] = 18
    elif fault == "source_reference":
        r["TRAIN_reference_unchanged"] = False
    elif fault == "private_rows":
        s["arms"][1]["native_endpoint"]["training_examples"] = 126
    elif fault == "source_clock":
        r["training"]["budget"]["started_at"] = 0.0
    elif fault == "unclosed":
        closed = 19
    else:
        info["shared_readout_owner"] = False
    with pytest.raises(ValueError):
        gate.validate_case(r, c, returned, info, closed)


@pytest.mark.parametrize("replace", [False, True])
def test_saved_pair_refuses_equal_replacement_or_corruption(tmp_path, replace):
    raw = json.dumps(dict(original=dict(updates=10))).encode()
    (tmp_path / "actual.json").write_bytes(raw)
    (tmp_path / "captured.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    pins = gate.pin_saved(tmp_path, json.loads(raw))
    gate.require_saved(tmp_path, pins)
    if replace:
        new = tmp_path / "new.json"
        new.write_bytes(raw)
        new.replace(tmp_path / "actual.json")
    else:
        (tmp_path / "actual.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError):
        gate.require_saved(tmp_path, pins)


@pytest.mark.parametrize("pins", [{}, {"actual.json": ()}])
def test_saved_pair_requires_both_pins_before_reading(tmp_path, pins):
    with pytest.raises(ValueError, match="both original"):
        gate.require_saved(tmp_path, pins)


@pytest.mark.parametrize("fail", [False, True])
def test_actual_observer_keeps_precharged_failures_and_refuses_more(fail):
    tree = ast.parse((STAGE / "producer.py").read_text())
    nodes: list[ast.stmt] = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("charge", "observe")
    ]
    scope: dict[str, Any] = dict(
        CAPS=dict(fits=1, rows=9),
        counts=dict(fits=0, rows=0),
        returned=dict(fits=0, rows=0),
        wraps=wraps,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "observer", "exec"), scope)

    def fit(x):
        assert scope["counts"]["fits"] == 1
        if fail:
            raise RuntimeError("lost")
        return 1

    observed = scope["observe"](fit, "fits", ("rows", 0))
    if fail:
        with pytest.raises(RuntimeError, match="lost"):
            observed([0] * 9)
        assert scope["returned"]["fits"] == 0
    else:
        assert observed([0] * 9) == 1
    with pytest.raises(RuntimeError, match="allowance spent"):
        observed([0] * 9)


def test_current_driver_has_fixed_cold_dose_and_selected_command():
    tree = ast.parse((STAGE / "producer.py").read_text())
    values = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "CAPS"
        ):
            assert isinstance(node.value, ast.Call)
            values = {
                k.arg: ast.literal_eval(k.value)
                for k in node.value.keywords
                if k.arg
                in ("updates", "cold", "warm", "update_rows", "steps", "constructions", "forbidden")
            }
    assert values == dict(
        updates=10, cold=10, warm=0, update_rows=252, steps=292, constructions=20, forbidden=0
    )
    text = (STAGE / "producer.py").read_text()
    assert '"--all-old-tasks"' in text and '"--probe-update"' not in text
    assert '(STAGE / "native.attempted").open("x")' in text
