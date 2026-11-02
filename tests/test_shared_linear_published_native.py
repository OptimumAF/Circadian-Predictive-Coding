"""Metadata-only protocol gates. Synthetic records never claim native success."""

import ast
import gzip
import hashlib
import importlib.util
import json
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/runs/shared-linear-published-native-20261031"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load("published_native_gates", STAGE / "case_checks.py")
old = load("accepted_metadata_fixture", ROOT / "tests/test_ordinary_shared_joint_comparison.py")


@pytest.mark.parametrize("lost_return", [False, True])
def test_actual_guard_observer_counts_features_before_delegation(lost_return):
    tree = ast.parse((STAGE / "producer.py").read_text(encoding="utf8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "observe")
    caps = dict(guard_measurements=4, guard_measurement_rows=24)
    counts, returned = dict.fromkeys(caps, 0), dict.fromkeys(caps, 0)
    env: dict[str, Any] = dict(
        wraps=wraps, caps=caps, counts=counts, returned=returned, CapReached=RuntimeError
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), "actual-observer", "exec"), env)
    calls = []

    def delegate(owner, model, batch, policy, last):
        calls.append((owner, model, batch, policy, last))
        assert counts == dict(guard_measurements=1, guard_measurement_rows=6)
        if lost_return:
            raise ValueError("lost return")
        return "original return"

    observed = env["observe"](delegate, "guard_measurements", ("guard_measurement_rows", 2))
    batch = SimpleNamespace(features=[0] * 6)
    if lost_return:
        with pytest.raises(ValueError, match="lost return"):
            observed("owner", "model", batch, "policy", None)
        assert returned == dict.fromkeys(caps, 0)
    else:
        assert observed("owner", "model", batch, "policy", None) == "original return"
        assert returned == counts
    assert calls == [("owner", "model", batch, "policy", None)]


def fixture(published=True, success=False):
    r, c, t, info = old.fixture("joint")
    r["applied_updates"] = [
        dict(episode_id="phase-" + str(i), sample_id="batch-" + str(i)) for i in range(12)
    ]
    c = dict(gate.EXACT_WORK)
    c.update(
        episodes=40 if published else 24,
        probability=2,
        probability_rows=12,
        predict_ports=4,
        predict_rows=24,
        guard_measurements=4,
        guard_measurement_rows=24,
    )
    t = dict(c)
    info.update(
        publication_owners_unchanged=True,
        original_slot_restored=True,
        live_shared_row_matches=True,
        generations=[0, 2 if published else 0],
        actor_versions=["a0", "a0"],
    )
    episodes = (
        [
            dict(
                phase=phase,
                policy_version=v,
                concept=concept,
                role="outer_selection",
                episode=dict(
                    scenario=scenario,
                    complete=True,
                    success=success,
                    training_updates=0,
                    steps=[],
                    work=dict(prediction_calls_validated=0),
                ),
            )
            for phase, scenarios in (
                ("published", ("ordinary", "stale_once", "revoke_commit")),
                ("rollback", ("ordinary",)),
            )
            for scenario in scenarios
            for v, concept in gate.TASKS
        ]
        if published
        else []
    )
    guard = dict(
        policy=dict(
            metric_id="global_categorical_accuracy",
            resource_metric_id="complete_shared_linear_record_bytes",
            min_new_utility=1.0,
            min_new_gain=0.1,
            max_old_drop=0.0,
            max_prediction_seconds=0.25,
            max_resource_bytes=4096,
            allowed_actions=["commit", "defer", "inspect", "open", "retire", "route"],
        ),
        new_guard_task_id="shared-inner-new",
        old_guard_task_id="shared-inner-old",
        new_guard_keys=[["shared-inner-new", str(i)] for i in range(6)],
        old_guard_keys=[["shared-inner-old", str(i)] for i in range(6)],
        training_ids=[
            list(k) for k in sorted((u["episode_id"], u["sample_id"]) for u in r["applied_updates"])
        ],
        label_arrivals=[0, 0],
        decision=dict(accepted=published, reasons=[] if published else ["new_task_utility"]),
        evidence=dict(
            actor_new_utility=0.0,
            candidate_new_utility=1.0 if published else 0.0,
            actor_old_utility=0.0,
            candidate_old_utility=1.0,
            candidate_max_prediction_seconds=0.001,
            candidate_resource_bytes=2000,
        ),
    )
    row = dict(
        complete=True,
        status="published_and_rolled_back" if published else "rejected",
        fit_admissions=16,
        fit_returns=16,
        fit_work_unknown=False,
        guard_attempts=4,
        guard_returns=4,
        guard_rows=24,
        commit_attempts=int(published),
        commit_returns=int(published),
        commit_unknown=False,
        rollback_attempts=int(published),
        rollback_returns=int(published),
        rollback_unknown=False,
        requests=0,
        request_returns=0,
        episodes=episodes,
        guard_report=guard,
        promotion_receipt=dict(generation=1, previous_version="a0", actor_version="c0")
        if published
        else None,
        sharing=dict(
            limits=dict(max_serving_requests=1, max_admitted_updates=16, max_updates_per_poll=1),
            active_serving_requests=0,
            training_active=False,
            paused=True,
            admitted_updates=16,
            deferrals=[],
        ),
    )
    r["shared_serving"] = row
    return r, c, t, info


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("success", [False, True])
def test_accepts_fixed_complete_observation_without_demanding_success(published, success):
    r, c, t, info = fixture(published, success)
    gate.validate_arm(r, c, t, "joint", info)


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "unknown",
        "budget_alias",
        "slot",
        "row",
        "generation",
        "admissions",
        "guard_return",
        "guard_policy",
        "guard_ids",
        "guard_missing",
        "decision",
        "commit_receipt",
        "rollback_return",
        "sharing",
        "missing_episode",
        "episode_role",
        "episode_unknown",
        "episode_training",
        "work_return",
        "prior_source",
        "source_budget",
        "empty_paid_keys",
        "changed_paid_keys",
    ],
)
def test_refuses_changed_authority_unknown_work_or_missing_observation(fault):
    r, c, t, info = fixture()
    row = r["shared_serving"]
    if fault == "incomplete":
        row["complete"] = False
    elif fault == "unknown":
        row["commit_unknown"] = True
    elif fault == "budget_alias":
        info["publication_owners_unchanged"] = False
    elif fault == "slot":
        info["original_slot_restored"] = False
    elif fault == "row":
        info["live_shared_row_matches"] = False
    elif fault == "generation":
        info["generations"] = [0, 1]
    elif fault == "admissions":
        row["fit_admissions"] = 17
    elif fault == "guard_return":
        row["guard_returns"] = 3
    elif fault == "guard_policy":
        row["guard_report"]["policy"]["min_new_utility"] = 0.5
    elif fault == "guard_ids":
        row["guard_report"]["new_guard_keys"] = []
    elif fault == "guard_missing":
        row["guard_report"]["evidence"]["candidate_new_utility"] = None
    elif fault == "decision":
        row["guard_report"]["decision"]["accepted"] = False
    elif fault == "commit_receipt":
        row["promotion_receipt"]["actor_version"] = "a0"
    elif fault == "rollback_return":
        row["rollback_returns"] = 0
    elif fault == "sharing":
        row["sharing"]["admitted_updates"] = 0
    elif fault == "missing_episode":
        row["episodes"].pop()
    elif fault == "episode_role":
        row["episodes"][0]["role"] = "final_test"
    elif fault == "episode_unknown":
        row["episodes"][0]["episode"]["complete"] = False
    elif fault == "episode_training":
        row["episodes"][0]["episode"]["training_updates"] = 1
    elif fault == "work_return":
        t["predict_ports"] = 3
    elif fault == "prior_source":
        info["authority_unchanged"] = False
    elif fault == "source_budget":
        r["budget"]["limits"]["max_wall_seconds"] = 60
    elif fault == "empty_paid_keys":
        row["guard_report"]["training_ids"] = []
    elif fault == "changed_paid_keys":
        row["guard_report"]["training_ids"][0] = ["foreign", "foreign"]
    with pytest.raises(ValueError):
        gate.validate_arm(r, c, t, "joint", info)


@pytest.mark.parametrize("fault", ["corrupt", "missing", "replaced"])
def test_saved_original_refuses_replacement(fault, tmp_path):
    raw = json.dumps(dict(complete=True, charge=12)).encode()
    path = tmp_path / "joint.json"
    path.write_bytes(raw)
    (tmp_path / "joint-raw.json.gz").write_bytes(gzip.compress(raw))
    pins = gate.pin_saved_reports(tmp_path, "joint", json.loads(raw))
    gate.require_saved(tmp_path, pins)
    if fault == "corrupt":
        path.write_bytes(raw.replace(b"12", b"13"))
    elif fault == "missing":
        path.unlink()
    else:
        replacement = tmp_path / "replacement"
        replacement.write_bytes(raw)
        replacement.replace(path)
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.require_saved(tmp_path, pins)


@pytest.mark.parametrize("empty", ["source", "driver"])
def test_source_binding_refuses_empty_keysets(empty, tmp_path):
    tree = ast.parse((STAGE / "producer.py").read_text(encoding="utf8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "require_sources"
    )
    (tmp_path / "a").write_text("source")
    drivers = ("producer.py", "analysis.py", "case_checks.py")
    for name in drivers:
        (tmp_path / name).write_text("driver")
    digest = lambda name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    receipt = dict(
        complete=True,
        sources=["a"],
        source_sha256={"a": digest("a")},
        driver_sha256={n: digest(n) for n in drivers},
    )
    receipt[empty + "_sha256"] = {}
    (tmp_path / "preflight.json").write_text(json.dumps(receipt))
    env = dict(STAGE=tmp_path, ROOT=tmp_path, SOURCES=("a",), json=json, hashlib=hashlib)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "binding", "exec"), env)
    with pytest.raises(AssertionError):
        env["require_sources"]()
