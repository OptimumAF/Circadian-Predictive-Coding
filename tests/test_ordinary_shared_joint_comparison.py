"""Synthetic metadata gates only, never a scientific/source/model fixture."""

import ast
import gzip
import json
import importlib.util
from pathlib import Path

import pytest

STAGE = (
    Path(__file__).resolve().parents[1] / "artifacts/runs/ordinary-shared-joint-comparison-20261031"
)
spec = importlib.util.spec_from_file_location("ordinary_case_checks", STAGE / "case_checks.py")
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixture(control="current_only"):
    limits = dict(max_training_updates=12, max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2)
    r = dict(
        complete=True,
        definition="context_return_queue_rehearsal_v1",
        method="linear_fit",
        linear_train_update="minimum_change",
        linear_replay_joint=control,
        replay_mode="coverage",
        replay_consent="explicit synthetic original TRAIN packet reuse; not inbox permission renewal",
        final_examples=0,
        probe_updates=0,
        external_actions=0,
        source_unchanged=True,
        actor_unchanged=True,
        numpy_rng_unchanged=True,
        python_rng_unchanged=True,
        training_work=dict(
            attempted_updates=16,
            returned_updates=16,
            attempted_rows=120,
            unreturned_work_unknown=False,
        ),
        budget=dict(limits=limits, updates_completed=12, replay_examples_completed=0),
        replay_budget=dict(
            limits=dict(
                max_replay_examples=72, max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2
            ),
            updates_completed=0,
            replay_examples_completed=48,
        ),
        replay_examples=48,
        applied_updates=[{} for _ in range(12)],
        training_data=["fixed"],
        queue_semantics={"feature_dim": 9},
        checkpoints=[
            dict(source_updates=p, state=dict(completed_updates=u, training_examples=n))
            for p, u, n in zip(gate.POINTS, (0, 4, 10, 16), (0, 24, 72, 120))
        ],
        episodes=[
            dict(
                source_updates=p,
                policy_version=v,
                concept=c,
                role="outer_selection",
                source_group="development-context-return-workflow-traces",
                episode=dict(
                    scenario=s, complete=True, success=False, training_updates=0, steps=[]
                ),
            )
            for p, s, v, c in gate.SLOTS
        ],
    )
    events = []
    for i in range(6):
        row = dict(
            updates=2 * (i + 1),
            status="skipped" if i < 2 else "complete",
            charged_exposures=0 if i < 2 else 12,
            attempted_updates=0 if i < 2 else 1,
            returned_updates=0 if i < 2 else 1,
            committed_updates=0 if i < 2 else 1,
            proposal_applied_updates=0 if i < 2 else 1,
            proposal_application_unknown_updates=0,
            unreturned_fit_work_unknown=False,
            receipt=None if i < 2 else {"id": i},
            selected_episodes=[dict(role="train")],
            joint_shared=dict(
                current_role="train",
                retained_role="train",
                current_packet_rows=6,
                total_fit_rows=12,
                retained_train_rows_used=6 if control == "joint" else 0,
                current_rows_used=6 if control == "joint" else 12,
            ),
        )
        events.append(row)
    r["published_replay_receipts"] = [e["receipt"] for e in events[2:]]
    r["replay"] = dict(
        joint_shared=dict(policy=dict(feature_dim=9, num_classes=6, control=control)),
        events=events,
        policy={"fixed": "selector"},
    )
    info = dict(
        authority_unchanged=True,
        birth_consent=True,
        source_actor_unchanged=True,
        all_experiences_reusable=True,
        training=["hash"],
        initial={"model": None},
        replay_packets=[dict(current="hash", selected_ids=["id"])],
    )
    return r, dict(gate.EXACT_WORK), dict(gate.EXACT_WORK), info


@pytest.mark.parametrize("control", gate.ARMS)
@pytest.mark.parametrize("success", [False, True])
def test_admits_complete_fixed_case_without_a_desired_success_sign(control, success):
    r, c, t, info = fixture(control)
    for row in r["episodes"]:
        row["episode"]["success"] = success
    gate.validate_arm(r, c, t, control, info)


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "definition",
        "control",
        "consent",
        "private",
        "final",
        "final_bool",
        "source",
        "authority",
        "physical",
        "return",
        "work",
        "clock",
        "limit",
        "charge",
        "checkpoint",
        "header",
        "role",
        "missing",
        "scenario",
        "training",
        "success",
        "unknown",
        "fit_rows",
        "retained",
        "publication",
        "skip",
    ],
)
def test_refuses_corrupt_partial_foreign_or_unpaid_metadata(fault):
    r, c, t, info = fixture()
    event = r["replay"]["events"][2]
    if fault == "incomplete":
        r["complete"] = False
    elif fault == "definition":
        r["definition"] = "private"
    elif fault == "control":
        r["linear_replay_joint"] = "joint"
    elif fault == "consent":
        r["replay_consent"] = "renewed"
    elif fault == "private":
        r["reserved_probe_requested"] = True
    elif fault == "final":
        r["final_examples"] = 1
    elif fault == "final_bool":
        r["final_examples"] = False
    elif fault == "source":
        r["source_unchanged"] = False
    elif fault == "authority":
        info["authority_unchanged"] = False
    elif fault == "physical":
        c["solves"] = 0
    elif fault == "return":
        t["solves"] = 0
    elif fault == "work":
        r["training_work"]["attempted_rows"] = 108
    elif fault == "clock":
        r["budget"]["updates_completed"] = 13
    elif fault == "limit":
        r["budget"]["limits"]["max_wall_seconds"] = 90
    elif fault == "charge":
        r["replay_budget"]["replay_examples_completed"] = 0
    elif fault == "checkpoint":
        r["checkpoints"][-1]["state"]["training_examples"] = 108
    elif fault == "header":
        r["episodes"][0]["policy_version"] = "v3"
    elif fault == "role":
        r["episodes"][0]["role"] = "final_test"
    elif fault == "missing":
        r["episodes"].pop()
    elif fault == "scenario":
        r["episodes"][0]["episode"]["scenario"] = "private"
    elif fault == "training":
        r["episodes"][0]["episode"]["training_updates"] = 1
    elif fault == "success":
        r["episodes"][0]["episode"]["success"] = None
    elif fault == "unknown":
        event["proposal_application_unknown_updates"] = 1
    elif fault == "fit_rows":
        event["joint_shared"]["total_fit_rows"] = 6
    elif fault == "retained":
        event["joint_shared"]["retained_train_rows_used"] = 6
    elif fault == "publication":
        r["published_replay_receipts"] = []
    else:
        r["replay"]["events"][0]["charged_exposures"] = 12
    with pytest.raises((ValueError, KeyError)):
        gate.validate_arm(r, c, t, "current_only", info)


def test_pair_signature_preserves_information_but_excludes_outcome_control_time():
    r, _, _, info = fixture()
    s = gate.source_signature(r, info)
    other, _, _, other_info = fixture("joint")
    other["episodes"][0]["episode"]["success"] = True
    gate.require_matched(s, gate.source_signature(other, other_info))
    other_info["initial"]["model"] = "different"
    with pytest.raises(ValueError):
        gate.require_matched(s, gate.source_signature(other, other_info))


def test_failure_cannot_admit_the_second_source():
    calls = []

    def run(control):
        calls.append(control)
        return None

    def stop(result, control):
        raise ValueError("first incomplete")

    with pytest.raises(ValueError):
        gate.run_controls(run, stop)
    assert calls == ["current_only"]


@pytest.mark.parametrize("fault", ["corrupt", "missing", "replaced"])
def test_saved_original_is_required_before_another_source(fault, tmp_path):
    expected = {"complete": True, "receipt": [12]}
    original = json.dumps(expected).encode()
    path = tmp_path / "current_only.json"
    path.write_bytes(original)
    (tmp_path / "current_only-raw.json.gz").write_bytes(gzip.compress(original))
    pins = gate.pin_saved_reports(tmp_path, "current_only", expected)
    gate.require_saved(tmp_path, pins)
    if fault == "corrupt":
        path.write_bytes(original.replace(b"12", b"13"))
    elif fault == "missing":
        path.unlink()
    else:
        replacement = tmp_path / "replacement"
        replacement.write_bytes(original)
        replacement.replace(path)
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.require_saved(tmp_path, pins)


@pytest.mark.parametrize("empty", ["source", "driver"])
def test_exact_source_binding_key_sets_reject_empty_receipt_maps(empty, tmp_path):
    tree = ast.parse((STAGE / "producer.py").read_text(encoding="utf8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "require_sources"
    )
    import json, hashlib

    (tmp_path / "a").write_text("source")
    drivers = ("producer.py", "analysis.py", "case_checks.py")
    for name in drivers:
        (tmp_path / name).write_text("driver")
    digest = lambda name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    receipt = dict(
        complete=True,
        sources=["a"],
        source_sha256={"a": digest("a")},
        driver_sha256={name: digest(name) for name in drivers},
    )
    receipt[empty + "_sha256"] = {}
    (tmp_path / "preflight.json").write_text(json.dumps(receipt))
    env = dict(STAGE=tmp_path, ROOT=tmp_path, SOURCES=("a",), json=json, hashlib=hashlib)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "binding", "exec"), env)
    with pytest.raises(AssertionError):
        env["require_sources"]()
