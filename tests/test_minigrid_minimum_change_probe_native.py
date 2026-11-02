"""Pure protocol/evidence/actual observer-AST checks, no learner/simulator work."""

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
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/minigrid-minimum-change-probe-native-20261031"
)
spec = importlib.util.spec_from_file_location("minimum_probe_case_gates", STAGE / "checks.py")
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixture(correct=False):
    loss = 0.1 if correct else 2.0
    decisions = [
        dict(target_class=1, predicted_class=1 if correct else 0, correct=correct, loss=loss)
        for _ in range(9)
    ]
    old = {
        f"minigrid-source-{c}-{cue}": dict(
            rows=9,
            actions=[1 if correct else 0] * 9,
            expected=[1] * 9,
            correct=9 if correct else 0,
            ce=loss,
        )
        for c in gate.CONTEXTS
        for cue in ("blue", "red")
    }
    r: dict[str, Any] = dict(
        status="complete",
        reason=None,
        probe_update="minimum_change",
        protocol=dict(
            probe_update="minimum_change",
            selected_native_updates=6,
            selected_native_rows=108,
            probe_updates=2,
        ),
        source_native_state_unchanged=True,
        source_fork_calls=1,
        initial_probe_limits=dict(gate.LIMITS),
        budget=dict(
            limits=dict(gate.LIMITS),
            started_at=100.0,
            elapsed_seconds=2.0,
            updates_completed=2,
            replay_examples_completed=0,
        ),
        disposable_state=dict(
            completed_updates=8,
            training_examples=126,
            failed_work=None,
            heads=[dict(fit_rows=9), dict(fit_rows=9)],
        ),
        retained_input_array_bytes=406944,
        source_bindings=dict(
            after_c_sha256=gate.INPUTS[gate.SOURCE][1],
            after_c_bytes=377836,
            training_sha256=gate.INPUTS[gate.TRAIN][1],
            training_bytes=2346812,
        ),
        probe=dict(
            status="complete",
            reason=None,
            attempted_updates=2,
            completed_updates=2,
            attempted_training_examples=18,
            completed_training_examples=18,
            score_forward_calls_attempted=6,
            score_forward_calls_completed=6,
            reserved_data_bytes=85536,
            num_classes=7,
            checkpoints=[
                dict(
                    updates=i,
                    scores={
                        c: dict(accuracy=1.0 if correct else 0.0, ce=loss)
                        for c in ("F-blue", "F-red")
                    },
                    decision_diagnostics={
                        c: dict(decisions=json.loads(json.dumps(decisions)))
                        for c in ("F-blue", "F-red")
                    },
                )
                for i in (0, 1, 2)
            ],
            training_readouts=dict(
                status="complete",
                score_forward_calls_attempted=6,
                score_forward_calls_completed=6,
                completed_rows=162,
                reserved_data_bytes=128304,
                checkpoints=[{}, {}, {}],
            ),
        ),
        old_task_readouts=[
            dict(updates=i, contexts=json.loads(json.dumps(old))) for i in (0, 1, 2)
        ],
        collection=[],
        probe_data=[],
    )
    for i in range(4):
        role, cue = ("probe_train" if i < 2 else "probe_test"), ("blue" if i % 2 == 0 else "red")
        identity = f"minigrid-probe-source-{role}-{cue}"
        r["collection"].append(
            dict(
                role=role,
                context=f"F-{cue}",
                cue=cue,
                seed=17,
                max_steps=16,
                layout=dict(
                    context=f"F-{cue}", rotation=0, branch_row=2, swap_keys=False, wall_color="grey"
                ),
                episode_id=identity,
                status="complete",
                closed=True,
                terminated=True,
                truncated=False,
                total_return=0.49375,
                environment_steps=9,
                transitions=[dict(action=2)] * 9,
            )
        )
        r["probe_data"].append(
            dict(
                role=role,
                context=f"F-{cue}",
                episode_id=identity + (f":copy:{i}" if i < 2 else ""),
                num_classes=7,
                transitions=[{}] * 9,
            )
        )
    counts = dict(gate.EXACT)
    info = dict(
        source_unchanged=True,
        authority_unchanged=True,
        rng_unchanged=True,
        birth_limits=dict(gate.LIMITS),
        birth_started_at=100.0,
    )
    return r, counts, dict(counts), info


@pytest.mark.parametrize("correct", [True, False])
def test_accepts_complete_positive_and_negative_fit_records(correct):
    r, c, v, i = fixture(correct)
    gate.validate_case(r, c, v, i, 4)


def test_refuses_training_source_id_without_the_original_release_copy_suffix():
    r, c, v, i = fixture()
    r["probe_data"][0]["episode_id"] = r["collection"][0]["episode_id"]
    with pytest.raises(ValueError, match="teacher/role/layout/closure"):
        gate.validate_case(r, c, v, i, 4)


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "intent",
        "source_geometry",
        "source_mutation",
        "authority",
        "rng",
        "closed",
        "unreturned",
        "limits",
        "clock",
        "replay",
        "fork",
        "state",
        "generic",
        "checkpoint",
        "F_metric",
        "old_metric",
        "teacher_role",
        "teacher_open",
        "layout",
        "footprint",
        "source_binding",
    ],
)
def test_refuses_changed_or_incomplete_native_evidence(fault):
    r, c, v, i = fixture()
    closed = 4
    if fault == "incomplete":
        r["status"] = "failed"
    elif fault == "intent":
        r["probe_update"] = False
    elif fault == "source_geometry":
        r["protocol"]["selected_native_updates"] = 10
    elif fault == "source_mutation":
        i["source_unchanged"] = False
    elif fault == "authority":
        i["authority_unchanged"] = False
    elif fault == "rng":
        i["rng_unchanged"] = False
    elif fault == "closed":
        closed = 3
    elif fault == "unreturned":
        v["warm_rows"] = 9
    elif fault == "limits":
        r["budget"]["limits"]["max_wall_seconds"] = 91
    elif fault == "clock":
        r["budget"]["started_at"] = 101
    elif fault == "replay":
        r["budget"]["replay_examples_completed"] = True
    elif fault == "fork":
        r["source_fork_calls"] = 2
    elif fault == "state":
        r["disposable_state"]["failed_work"] = [1, 9, 0, 0, True]
    elif fault == "generic":
        r["probe"]["completed_updates"] = 1
    elif fault == "checkpoint":
        r["probe"]["checkpoints"].pop()
    elif fault == "F_metric":
        r["probe"]["checkpoints"][1]["scores"]["F-blue"]["ce"] = 0.0
    elif fault == "old_metric":
        r["old_task_readouts"][2]["contexts"]["minigrid-source-A-blue"]["correct"] = 9
    elif fault == "teacher_role":
        r["collection"][0]["role"] = "train"
    elif fault == "teacher_open":
        r["collection"][3]["closed"] = False
    elif fault == "layout":
        r["collection"][1]["layout"]["rotation"] = 1
    elif fault == "footprint":
        r["retained_input_array_bytes"] = 0
    else:
        r["source_bindings"]["after_c_sha256"] = "foreign"
    with pytest.raises(ValueError):
        gate.validate_case(r, c, v, i, closed)


@pytest.mark.parametrize("fault", ["corrupt", "equal_replace", "omit_pin"])
def test_rejects_corrupted_or_replaced_or_missing_saved_evidence(tmp_path, fault):
    expected = dict(status="complete", negative=True)
    raw = json.dumps(expected).encode()
    (tmp_path / "actual.json").write_bytes(raw)
    (tmp_path / "captured.json.gz").write_bytes(gzip.compress(raw))
    pins = gate.pin_saved(tmp_path, expected)
    if fault == "corrupt":
        (tmp_path / "actual.json").write_bytes(b"{}")
    elif fault == "equal_replace":
        path = tmp_path / "actual.json"
        path.rename(tmp_path / "original.json")
        path.write_bytes(raw)
    else:
        pins.pop("captured.json.gz")
    with pytest.raises(ValueError):
        gate.require_saved(tmp_path, pins)


@pytest.mark.parametrize("field", ["source_sha256", "driver_sha256"])
def test_requires_exact_named_source_and_driver_keysets(tmp_path, monkeypatch, field):
    monkeypatch.setattr(gate, "INPUTS", {})
    root, stage = tmp_path / "root", tmp_path / "stage"
    root.mkdir()
    stage.mkdir()
    (root / "a.py").write_bytes(b"pass\n")
    (stage / "b.py").write_bytes(b"pass\n")
    digest = hashlib.sha256(b"pass\n").hexdigest()
    receipt: dict[str, Any] = dict(
        complete=True,
        sources=["a.py"],
        source_sha256={"a.py": digest},
        driver_sha256={"b.py": digest},
        input_pins={},
    )
    (stage / "preflight.json").write_text(json.dumps(receipt))
    gate.require_bindings(root, stage, ("a.py",), ("b.py",))
    receipt[field]["extra.py"] = digest
    (stage / "preflight.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        gate.require_bindings(root, stage, ("a.py",), ("b.py",))


@pytest.mark.parametrize("lost_return", [False, True])
def test_actual_observer_precharges_and_only_records_known_returns(lost_return):
    tree = ast.parse((STAGE / "producer.py").read_text())
    body: list[ast.stmt] = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("charge", "observe")
    ]
    caps = dict(solves=1, solve_rows=9, products=2, product_rows=18)
    namespace: dict[str, Any] = dict(
        wraps=wraps, CAPS=caps, counts=dict.fromkeys(caps, 0), returned=dict.fromkeys(caps, 0)
    )
    exec(compile(ast.Module(body=body, type_ignores=[]), "actual_observer", "exec"), namespace)

    def operation(rows):
        assert namespace["counts"] == caps
        if lost_return:
            raise RuntimeError("lost return")
        return "known"

    observed = namespace["observe"](operation, "solves", ("solve_rows", 0), 2)
    if lost_return:
        with pytest.raises(RuntimeError):
            observed([0] * 9)
        assert not any(namespace["returned"].values())
    else:
        assert observed([0] * 9) == "known" and namespace["returned"] == caps
    assert namespace["counts"] == caps
