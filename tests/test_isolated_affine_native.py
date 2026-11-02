"""Pure literal/AST gates for one fixed native case; no learner imports."""

import ast
from functools import wraps
import gzip
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

STAGE = (
    Path(__file__).resolve().parents[1] / "artifacts/runs/minigrid-isolated-affine-native-20261031"
)
spec = importlib.util.spec_from_file_location("isolated_game_case", STAGE / "case_checks.py")
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixture(success=False):
    def budget(n, seconds, replay=None):
        v = gate.limits(n, seconds, replay)
        return dict(
            initial_limits=v.copy(),
            limits=v,
            updates_completed=n,
            replay_examples_completed=0,
            started_at=1.0,
            observed_budget_started_at=1.0,
            ended_at=2.0,
        )

    keys = [f"minigrid-source-{c}-{q}" for c in gate.CONTEXTS for q in ("blue", "red")]
    t = dict(
        status="complete",
        source_unchanged=True,
        actor_unchanged=True,
        receipt_capture_error=None,
        budget=budget(6, 90),
        committed_receipts=[
            dict(update_number=i + 1, observed_at=i, arrived_at=i + 1, applied_at=i + 1)
            for i in range(6)
        ],
        catalogue=dict(committed_rows=dict(arrivals=54, replay=54, native=108), peak_bank_rows=36),
        score_calls_attempted=24,
        score_calls_completed=24,
        score_rows_attempted=216,
        score_rows_completed=216,
        stream=dict(
            evaluation_scope="training_only",
            score_forward_calls=0,
            completed_updates=6,
            training_examples=108,
        ),
        endpoints=[
            dict(
                updates=n,
                state=dict(completed_updates=n, training_examples=rows, failed_work=None),
                scores=dict.fromkeys(keys, {}),
            )
            for n, rows in zip(gate.POINTS, gate.ROWS)
        ],
    )
    p = dict(
        status="complete",
        probe_update="minimum_change",
        protocol=dict(probe_update="minimum_change"),
        source_native_state_unchanged=True,
        source_fork_calls=1,
        budget=budget(2, 90, 0),
        initial_probe_limits=gate.limits(2, 90, 0),
        disposable_state=dict(
            completed_updates=8,
            training_examples=126,
            failed_work=None,
            heads=[dict(fit_rows=9)] * 2,
        ),
        probe=dict(
            status="complete",
            completed_updates=2,
            completed_training_examples=18,
            checkpoints=[dict(updates=n) for n in (0, 1, 2)],
        ),
        old_task_readouts=[
            dict(
                updates=n,
                contexts={
                    f"minigrid-source-{c}-{q}": {} for q in ("blue", "red") for c in gate.CONTEXTS
                },
            )
            for n in (0, 1, 2)
        ],
        collection=[
            dict(
                status="complete",
                closed=True,
                role=role,
                cue=q,
                context=f"F-{q}",
                environment_steps=9,
                total_return=0.49375,
            )
            for role in ("probe_train", "probe_test")
            for q in ("blue", "red")
        ],
        probe_data=[{}] * 4,
    )
    steps = 9 if success else 16
    episodes = []
    for c, q, role in gate.TASKS:
        e = dict(
            context=c,
            cue=q,
            role=role,
            status="complete",
            closed=True,
            terminated=success,
            truncated=not success,
            total_return=0.49375 if success else 0.0,
            seed=17,
            max_steps=16,
            environment_steps=steps,
            transitions=[dict(action=0)] * steps,
        )
        episodes.append(
            dict(
                context=c,
                cue=q,
                source_role=role,
                status="complete",
                state_unchanged=True,
                success=success,
                return_=e["total_return"],
                environment=e,
            )
        )
        episodes[-1]["return"] = episodes[-1].pop("return_")
    g = dict(
        status="complete",
        state_unchanged=True,
        probe_phase="after_probe",
        include_old_tasks=True,
        source_fit_calls=8,
        source_fit_rows=126,
        new_fit_calls=0,
        new_optimizer_updates=0,
        episodes=episodes,
    )
    r = dict(
        status="complete",
        reason=None,
        protocol=dict(probe_update="minimum_change", new_fit_calls=8, new_fit_rows=126),
        TRAIN_reference_unchanged=True,
        attempted_phases=["training", "private_probe", "games"],
        training=t,
        private_probe=p,
        games=[g],
        readout_budget=budget(0, 150, 0),
        workflow_budget=dict(
            max_wall_seconds=180,
            work_deadline_seconds=170,
            elapsed_seconds=3.0,
            sampled_peak_rss_bytes=90 * 1024**2,
        ),
    )
    counts = gate.EXACT | dict(steps=36 + 8 * steps)
    info: dict[str, Any] = dict.fromkeys(
        (
            "source_authority_unchanged",
            "source_actor_unchanged",
            "source_handoff_matches",
            "probe_handoff_matches",
            "probe_authority_unchanged",
            "readout_authority_unchanged",
            "rng_unchanged",
        ),
        True,
    )
    for key in ("source_birth", "probe_birth", "readout_birth"):
        info[key] = dict(started_at=1.0)
    return r, counts, counts.copy(), info, 12


@pytest.mark.parametrize("success", [False, True])
def test_accepts_both_complete_negative_and_positive_game_outcomes(success):
    gate.validate_case(*fixture(success))


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "actor",
        "receipt",
        "source_clock",
        "probe_clock",
        "probe_charge",
        "probe_endpoint",
        "probe_intent",
        "F_point",
        "old_roster",
        "teacher_role",
        "game_role",
        "game_seed",
        "old_game_roster",
        "false_success",
        "unclosed",
        "unreturned",
        "readout_birth",
        "handoff",
        "RNG",
    ],
)
def test_refuses_broken_costs_roles_owners_and_evidence(fault):
    r, c, ret, i, closed = fixture()
    t, p, g = r["training"], r["private_probe"], r["games"][0]
    if fault == "source":
        t["source_unchanged"] = False
    elif fault == "actor":
        t["actor_unchanged"] = False
    elif fault == "receipt":
        t["committed_receipts"][0]["arrived_at"] = 0
    elif fault == "source_clock":
        t["budget"]["started_at"] = 0.0
    elif fault == "probe_clock":
        p["budget"]["ended_at"] = 0.0
    elif fault == "probe_charge":
        p["budget"]["updates_completed"] = 1
    elif fault == "probe_endpoint":
        p["disposable_state"]["training_examples"] = 108
    elif fault == "probe_intent":
        p["probe_update"] = "cold"
    elif fault == "F_point":
        p["probe"]["checkpoints"].pop()
    elif fault == "old_roster":
        p["old_task_readouts"][1]["contexts"].popitem()
    elif fault == "teacher_role":
        p["collection"][0]["role"] = "train"
    elif fault == "game_role":
        g["episodes"][0]["environment"]["role"] = "final_test"
    elif fault == "game_seed":
        g["episodes"][0]["environment"]["seed"] = 18
    elif fault == "old_game_roster":
        g["episodes"][-1]["context"] = "A"
    elif fault == "false_success":
        g["episodes"][0]["success"] = True
    elif fault == "unclosed":
        closed = 11
    elif fault == "unreturned":
        ret["solves"] -= 1
    elif fault == "readout_birth":
        r["readout_budget"]["observed_budget_started_at"] = 0.0
    elif fault == "handoff":
        i["probe_handoff_matches"] = False
    else:
        i["rng_unchanged"] = False
    with pytest.raises(ValueError):
        gate.validate_case(r, c, ret, i, closed)


@pytest.mark.parametrize("replace", [False, True])
def test_rejects_replaced_or_corrupt_saved_originals(tmp_path, replace):
    value = dict(original=dict(paid_updates=8))
    raw = json.dumps(value).encode()
    (tmp_path / "actual.json").write_bytes(raw)
    (tmp_path / "captured.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    pins = gate.pin_saved(tmp_path, value)
    gate.require_saved(tmp_path, pins)
    if replace:
        replacement = tmp_path / "other.json"
        replacement.write_bytes(raw)
        replacement.replace(tmp_path / "actual.json")
    else:
        (tmp_path / "actual.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError):
        gate.require_saved(tmp_path, pins)


@pytest.mark.parametrize("fail", [False, True])
def test_requires_both_original_saved_pins(tmp_path, fail):
    with pytest.raises(ValueError, match="both original saved evidence pins"):
        gate.require_saved(tmp_path, {} if fail else {"actual.json": ()})


@pytest.mark.parametrize("fail", [False, True])
def test_actual_observer_charges_before_work_and_retains_lost_returns(fail):
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
        assert scope["counts"] == dict(fits=1, rows=9)
        if fail:
            raise RuntimeError("lost")
        return 1

    observed = scope["observe"](fit, "fits", ("rows", 0))
    if fail:
        with pytest.raises(RuntimeError, match="lost"):
            observed([0] * 9)
        assert scope["returned"]["fits"] == 0
    else:
        assert observed([0] * 9) == 1 and scope["returned"]["rows"] == 9
    with pytest.raises(RuntimeError, match="allowance spent"):
        observed([0] * 9)


def test_native_driver_keeps_exact_selected_command_and_caps_without_execution():
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
                in (
                    "updates",
                    "solves",
                    "update_rows",
                    "cold",
                    "warm",
                    "forbidden",
                    "constructions",
                    "steps",
                )
            }
    assert values["updates"] == values["solves"] == 8 and values["update_rows"] == 126
    assert values["cold"] == 6 and values["warm"] == 2 and values["forbidden"] == 0
    assert values["constructions"] == 12 and values["steps"] == 164
    text = (STAGE / "producer.py").read_text()
    assert '"--example-affine-adaptation"' in text and '"--probe-update"' in text
    assert '(STAGE / "native.attempted").open("x")' in text


def test_saved_gate_has_no_project_numeric_or_model_imports():
    for name in ("case_checks.py", "analysis.py"):
        tree = ast.parse((STAGE / name).read_text())
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert not any(
            s.startswith(("src", "scripts", "numpy", "torch", "gymnasium")) for s in imports
        )
