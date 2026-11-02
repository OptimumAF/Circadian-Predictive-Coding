"""Pure protocol records and actual observer AST; no native/math fixtures."""

import ast
from functools import wraps
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

STAGE = (
    Path(__file__).resolve().parents[1] / "artifacts/runs/minigrid-minimum-change-native-20261031"
)
spec = importlib.util.spec_from_file_location("minimum_game_gates", STAGE / "case_checks.py")
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixture(success=False):
    limits = dict(max_training_updates=10, max_wall_seconds=90, max_process_rss_bytes=512 * 1024**2)
    read_limits = dict(
        max_training_updates=0,
        max_replay_examples=0,
        max_wall_seconds=150,
        max_process_rss_bytes=512 * 1024**2,
    )
    training = dict(
        status="complete",
        source_unchanged=True,
        actor_unchanged=True,
        train_update="minimum_change",
        protocol=dict(train_update="minimum_change"),
        budget=dict(
            initial_limits=limits.copy(),
            limits=limits.copy(),
            updates_completed=10,
            replay_examples_completed=0,
        ),
        receipt_capture_error=None,
        committed_receipts=[
            dict(update_number=i + 1, observed_at=i, arrived_at=i + 1, applied_at=i + 1)
            for i in range(10)
        ],
        catalogue=dict(
            committed_rows=dict(arrivals=90, replay=162, native=252),
            peak_bank_rows=54,
            input_array_bytes=598752,
            peak_joint_template_array_bytes=85536,
            peak_bank_observation_target_float64_equivalent_bytes=256176,
        ),
        score_calls_attempted=48,
        score_calls_completed=48,
        score_rows_attempted=432,
        score_rows_completed=432,
        stream=dict(
            evaluation_scope="training_only",
            score_forward_calls=0,
            completed_updates=10,
            training_examples=252,
        ),
        endpoints=[
            dict(
                updates=n,
                state=dict(completed_updates=n, training_examples=rows, failed_work=None),
                scores={
                    f"minigrid-source-{c}-{cue}": dict(
                        actions=[0] * 9, expected=[1] * 9, accuracy=0.0, ce=2.0
                    )
                    for c in gate.CONTEXTS
                    for cue in ("blue", "red")
                },
            )
            for n, rows in zip(gate.POINTS, gate.ROWS)
        ],
    )
    games = []
    for context, cue in gate.TASKS:
        steps, reward = (9, 0.49375) if success else (16, 0.0)
        env = dict(
            context=context,
            cue=cue,
            role="train",
            status="complete",
            closed=True,
            terminated=success,
            truncated=not success,
            total_return=reward,
            seed=17,
            max_steps=16,
            environment_steps=steps,
            transitions=[dict(action=2) for _ in range(steps)],
        )
        games.append(
            dict(
                context=context,
                cue=cue,
                source_role="train",
                status="complete",
                state_unchanged=True,
                success=success,
                return_value=reward,
                environment=env,
            )
        )
        games[-1]["return"] = games[-1].pop("return_value")
    report = dict(
        status="complete",
        reason=None,
        attempted_phases=["collection", "training", "games"],
        protocol=dict(
            train_update="minimum_change",
            new_layout=dict(
                context="G_train", rotation=0, branch_row=5, swap_keys=False, wall_color="grey"
            ),
            returned_layout=dict(
                context="A", rotation=0, branch_row=3, swap_keys=False, wall_color="grey"
            ),
        ),
        training=training,
        collection=[
            dict(
                role="train",
                context="G_train",
                cue=cue,
                episode_id="minigrid-source-G_train-" + cue,
                closed=True,
                status="complete",
                environment_steps=9,
                total_return=0.49375,
            )
            for cue in ("blue", "red")
        ],
        games=dict(
            status="complete",
            state_unchanged=True,
            source_fit_calls=10,
            source_fit_rows=252,
            new_fit_calls=0,
            new_optimizer_updates=0,
            episodes=games,
        ),
        readout_budget=dict(
            initial_limits=read_limits.copy(),
            limits=read_limits.copy(),
            started_at=1.0,
            observed_budget_started_at=1.0,
            updates_completed=0,
            replay_examples_completed=0,
        ),
        workflow_budget=dict(elapsed_seconds=10.0, sampled_peak_rss_bytes=60 * 1024**2),
    )
    counts = dict(
        gate.EXACT,
        steps=18 + sum(x["environment"]["environment_steps"] for x in games),
        teachers=18,
    )
    info = dict.fromkeys(
        (
            "source_authority_unchanged",
            "source_actor_unchanged",
            "handoff_matches",
            "readout_authority_unchanged",
            "rng_unchanged",
        ),
        True,
    )
    return report, counts, dict(counts), info


@pytest.mark.parametrize("success", [False, True])
def test_complete_fixed_negative_and_positive_games_are_both_valid(success):
    r, c, t, info = fixture(success)
    gate.validate_case(r, c, t, info, 6)


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "layout",
        "intent",
        "authority",
        "unknown_return",
        "closed",
        "source_limits",
        "source_charge",
        "clock",
        "row_charge",
        "receipts",
        "future_label",
        "diagnostics",
        "evaluation",
        "checkpoint",
        "monitor",
        "teacher_role",
        "teacher_open",
        "game_role",
        "game_open",
        "game_state",
        "game_success_type",
        "game_steps",
        "game_action",
        "readout_renewal",
        "workflow_expired",
    ],
)
def test_rejects_changed_scope_authority_unknown_or_unobserved_evidence(fault):
    r, c, t, info = fixture()
    train = r["training"]
    env = r["games"]["episodes"][0]["environment"]
    closed = 6
    if fault == "incomplete":
        r["status"] = "failed"
    elif fault == "layout":
        r["protocol"]["returned_layout"]["branch_row"] = 4
    elif fault == "intent":
        train["train_update"] = "cold"
    elif fault == "authority":
        info["source_authority_unchanged"] = False
    elif fault == "unknown_return":
        t["solves"] = 9
    elif fault == "closed":
        closed = 5
    elif fault == "source_limits":
        train["budget"]["limits"]["max_wall_seconds"] = 150
    elif fault == "source_charge":
        train["budget"]["replay_examples_completed"] = False
    elif fault == "clock":
        r["readout_budget"]["observed_budget_started_at"] = 2.0
    elif fault == "row_charge":
        train["catalogue"]["committed_rows"]["native"] = 251
    elif fault == "receipts":
        train["committed_receipts"].pop()
    elif fault == "future_label":
        train["committed_receipts"][0]["arrived_at"] = 0
    elif fault == "diagnostics":
        train["score_calls_completed"] = 47
    elif fault == "evaluation":
        train["stream"]["evaluation_scope"] = "final_test"
    elif fault == "checkpoint":
        train["endpoints"][0]["state"]["completed_updates"] = True
    elif fault == "monitor":
        train["endpoints"][0]["scores"].popitem()
    elif fault == "teacher_role":
        r["collection"][0]["role"] = "probe_train"
    elif fault == "teacher_open":
        r["collection"][0]["closed"] = False
    elif fault == "game_role":
        env["role"] = "final_test"
    elif fault == "game_open":
        env["closed"] = False
    elif fault == "game_state":
        r["games"]["episodes"][0]["state_unchanged"] = False
    elif fault == "game_success_type":
        r["games"]["episodes"][0]["success"] = 0
    elif fault == "game_steps":
        env["environment_steps"] = 17
    elif fault == "game_action":
        env["transitions"][0]["action"] = 7
    elif fault == "readout_renewal":
        r["readout_budget"]["limits"]["max_wall_seconds"] = 300
    elif fault == "workflow_expired":
        r["workflow_budget"]["elapsed_seconds"] = 291
    with pytest.raises(ValueError):
        gate.validate_case(r, c, t, info, closed)


@pytest.mark.parametrize("fault", ["corrupt", "missing", "replace"])
def test_saved_cli_raw_admission_refuses_corruption_or_equal_replacement(tmp_path, fault):
    data = dict(complete=True, charge=10)
    raw = json.dumps(data).encode()
    path = tmp_path / "actual.json"
    path.write_bytes(raw)
    (tmp_path / "captured.json.gz").write_bytes(gzip.compress(raw))
    pins = gate.pin_saved(tmp_path, data)
    gate.require_saved(tmp_path, pins)
    if fault == "corrupt":
        path.write_bytes(raw.replace(b"10", b"11"))
    elif fault == "missing":
        path.unlink()
    else:
        replacement = tmp_path / "replacement"
        replacement.write_bytes(raw)
        replacement.replace(path)
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.require_saved(tmp_path, pins)


@pytest.mark.parametrize("empty", ["source", "driver"])
def test_exact_source_binding_rejects_missing_keysets(tmp_path, empty):
    tree = ast.parse((STAGE / "producer.py").read_text(encoding="utf8"))
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "require_sources"
    )
    drivers = ("producer.py", "case_checks.py", "analysis.py", "PROTOCOL.md")
    for name in ("a", *drivers):
        (tmp_path / name).write_text(name)
    digest = lambda name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    receipt = dict(
        complete=True,
        sources=["a"],
        source_sha256=dict(a=digest("a")),
        driver_sha256={n: digest(n) for n in drivers},
    )
    receipt[empty + "_sha256"] = {}
    (tmp_path / "preflight.json").write_text(json.dumps(receipt))
    env = dict(
        ROOT=tmp_path, STAGE=tmp_path, SOURCES=("a",), DRIVERS=drivers, json=json, hashlib=hashlib
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "actual-binding", "exec"), env)
    with pytest.raises(AssertionError):
        env["require_sources"]()


@pytest.mark.parametrize("lost", [False, True])
def test_actual_observer_precharges_rows_and_preserves_lost_return(lost):
    tree = ast.parse((STAGE / "producer.py").read_text(encoding="utf8"))
    functions: list[ast.stmt] = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("charge", "observe")
    ]
    counts, returned = dict(solves=0, solve_rows=0), dict(solves=0, solve_rows=0)
    env: dict[str, Any] = dict(
        CAPS=dict(solves=1, solve_rows=2), counts=counts, returned=returned, wraps=wraps
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), "actual-observer", "exec"), env)
    calls = []

    def original(rows):
        calls.append(rows)
        assert counts == dict(solves=1, solve_rows=2)
        if lost:
            raise RuntimeError("lost return")
        return "original value"

    observe = env["observe"](original, "solves", ("solve_rows", 0))
    rows = [[1], [2]]
    if lost:
        with pytest.raises(RuntimeError, match="lost return"):
            observe(rows)
        assert returned == dict(solves=0, solve_rows=0)
    else:
        assert observe(rows) == "original value" and returned == counts
    assert calls == [rows]
