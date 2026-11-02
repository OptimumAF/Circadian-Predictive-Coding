"""Zero-model metadata/admission checks; fixtures are not learner outcomes."""

import ast
from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

STAGE = (
    Path(__file__).resolve().parents[1] / "artifacts/runs/shared-linear-probe-comparison-20261031"
)
spec = importlib.util.spec_from_file_location("shared_case_checks", STAGE / "case_checks.py")
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def fixture(control):
    limits = dict(max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2)

    def budget(n):
        return dict(limits=limits | dict(max_training_updates=n), started_at=1, last_clock=2)

    sessions = [budget(16) | dict(updates=16) for _ in range(2)]
    events = [
        dict(
            status="complete",
            applied=True,
            phase="fresh" if i < 16 else "return",
            work=dict(
                fit_attempts=1,
                fit_returns=1,
                fit_rows=15 if i < 16 else 18,
                additional_rows_charged=12,
                application_unknown=0,
                unreturned_fit_work_unknown=0,
                native_sgd_updates=0,
                retained_train_rows_used=12 if control == "joint" else 0,
            ),
        )
        for i in range(32)
    ]
    paid = dict(
        control=control,
        attempted_updates=32,
        attempted_rows=144,
        returned_updates=32,
        applied_updates=32,
        additional_rows_charged=384,
        native_sgd_updates=0,
        application_unknown_updates=0,
        events=events,
        budget=budget(32) | dict(admissions=32, additional_rows=384),
    )
    paid["budget"]["limits"]["max_replay_examples"] = 384
    points = [dict(updates=n, scores={}) for n in (0, 4, 8, 16)]

    def phase(rows):
        return dict(
            status="complete",
            completed_updates=16,
            attempted_updates=16,
            completed_training_examples=rows,
            attempted_training_examples=rows,
            checkpoints=deepcopy(points),
            training_readouts=dict(status="complete", checkpoints=deepcopy(points)),
        )

    private = dict(
        complete=True,
        source_unchanged=True,
        final_examples=0,
        replay_examples=384,
        preserved_training_work=dict(attempted_updates=0, attempted_rows=0, returned_updates=0),
        sessions=sessions,
        preserved_linear_rehearsal_work=paid,
        probe=phase(48) | dict(return_phase=phase(96)),
        episodes={},
    )
    for name in gate.PHASES:
        private["episodes"][name] = [
            dict(
                phase=name,
                role="controller_train",
                policy_version=v,
                concept=c,
                episode=dict(complete=True, scenario="ordinary", success=False),
            )
            for v, c in gate.TASKS
        ]
    report = dict(
        complete=True,
        definition="linear_context_return_ticket_workflow_v1",
        method="linear_fit",
        linear_train_update="minimum_change",
        reserved_probe_update="minimum_change",
        reserved_probe_rehearsal=control,
        source_unchanged=True,
        actor_unchanged=True,
        numpy_rng_unchanged=True,
        python_rng_unchanged=True,
        final_examples=0,
        replay_examples=0,
        external_actions=0,
        probe_updates=32,
        probe_replay_examples=384,
        reserved_probe=private,
        probe_auxiliary_updates=deepcopy(paid),
        budget=budget(12) | dict(updates_completed=12),
        training_work=dict(
            attempted_updates=12,
            attempted_rows=72,
            returned_updates=12,
            unreturned_work_unknown=False,
            budget_committed_updates=12,
        ),
        reserved_probe_summary=dict(status="complete"),
        phases=[],
        checkpoints=[dict(source_updates=n) for n in (0, 4, 8, 12)],
        episodes=[],
    )
    for n in (0, 4, 8, 12):
        for scenario in ("ordinary", "stale_once", "revoke_commit") if n == 12 else ("ordinary",):
            report["episodes"] += [
                dict(
                    source_updates=n,
                    policy_version=v,
                    concept=c,
                    episode=dict(scenario=scenario, complete=True),
                )
                for v, c in gate.TASKS[:4]
            ]
    information = dict(
        source_train=[dict(role="train")] * 12,
        bank=[dict(role="train")] * 2,
        cohorts=[
            [dict(role="probe_train")] * 16,
            [dict(role="probe_test")] * 2,
            [dict(role="train")] * 2,
            [dict(role="train")] * 2,
        ],
    )
    return report, dict(gate.EXACT_WORK), information


@pytest.mark.parametrize("control", gate.ARMS)
def test_admits_complete_paid_arm_without_requiring_a_winning_outcome(control):
    report, work, info = fixture(control)
    gate.validate_arm(report, work, work.copy(), control, info)


@pytest.mark.parametrize(
    "change",
    [
        "incomplete",
        "source",
        "intent",
        "clock",
        "final",
        "unknown",
        "lost",
        "receipt",
        "rows",
        "role",
        "task",
    ],
)
def test_refuses_incomplete_corrupted_or_unpaid_prior_evidence(change):
    report, work, info = fixture("joint")
    if change == "incomplete":
        report["complete"] = False
    if change == "source":
        report["source_unchanged"] = False
    if change == "intent":
        report["reserved_probe_rehearsal"] = "current_only"
    if change == "clock":
        report["budget"]["last_clock"] = 32
    if change == "final":
        report["final_examples"] = 1
    if change == "unknown":
        report["reserved_probe"]["preserved_linear_rehearsal_work"][
            "application_unknown_updates"
        ] = 1
    if change == "lost":
        report["reserved_probe"]["preserved_linear_rehearsal_work"]["events"][0]["status"] = (
            "failed"
        )
    if change == "receipt":
        report["reserved_probe"]["preserved_linear_rehearsal_work"]["events"].pop()
    if change == "rows":
        work["solve_rows"] = 599
    if change == "role":
        info["bank"][0]["role"] = "probe_test"
    if change == "task":
        report["reserved_probe"]["episodes"]["after_return"].pop()
    with pytest.raises(ValueError):
        gate.validate_arm(report, work, work.copy(), "joint", info)


@pytest.mark.parametrize("fail_in", ["run", "validation"])
def test_first_failure_never_admits_second_source_or_retries(fail_in):
    calls = []

    def run(control):
        calls.append(control)
        if fail_in == "run":
            raise RuntimeError("spent")
        return control

    def validate(result, control):
        raise RuntimeError("paid prior invalid")

    with pytest.raises(RuntimeError):
        gate.run_controls(run, validate)
    assert calls == ["current_only"]


def test_pair_matches_actual_source_information_but_not_new_outcomes_or_wall_clocks():
    a, _, info = fixture("current_only")
    b, _, other = fixture("joint")
    b["budget"]["started_at"] = 100
    b["reserved_probe"]["episodes"]["after_probe"][0]["episode"]["success"] = True
    gate.require_matched(gate.source_signature(a, info), gate.source_signature(b, other))
    other["source_train"] = [dict(role="train", features_sha256="changed")] * 12
    with pytest.raises(ValueError, match="inconclusive"):
        gate.require_matched(gate.source_signature(a, info), gate.source_signature(b, other))


def test_producer_observes_actual_existing_ports_and_protocol_totals_without_importing_it():
    tree = ast.parse((STAGE / "producer.py").read_text())
    caps = ast.literal_eval(
        next(
            n.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "caps" for t in n.targets)
        )
    )
    assert (caps["solves"], caps["solve_rows"]) == (88, 1200)
    assert (caps["scheduled_training_products"], caps["scheduled_training_rows"]) == (260, 3576)
    source = (STAGE / "producer.py").read_text()
    assert "core._solve_design" in source and "core._solve_minimum_change" in source
    assert "joint_api.SourceTrainPackets.__init__ = source_information" in source
    assert "replace_aliases(original_cohorts, cohort_information)" in source


@pytest.mark.parametrize(
    "corrupt", [None, "empty_source", "missing_source", "empty_driver", "missing_driver"]
)
def test_actual_source_gate_refuses_missing_digest_bindings_before_model_imports(tmp_path, corrupt):
    tree = ast.parse((STAGE / "producer.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "require_sources"
    )
    names = ("source-a.py", "source-b.py")
    drivers = ("producer.py", "analysis.py", "case_checks.py")
    for name in names + drivers:
        (tmp_path / name).write_text(name)

    def digest(name):
        return hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()

    receipt: dict[str, Any] = dict(
        complete=True,
        sources=list(names),
        source_sha256={n: digest(n) for n in names},
        driver_sha256={n: digest(n) for n in drivers},
    )
    if corrupt == "empty_source":
        receipt["source_sha256"] = {}
    if corrupt == "missing_source":
        receipt["source_sha256"].pop(names[0])
    if corrupt == "empty_driver":
        receipt["driver_sha256"] = {}
    if corrupt == "missing_driver":
        receipt["driver_sha256"].pop(drivers[0])
    (tmp_path / "preflight-v2.json").write_text(json.dumps(receipt))
    namespace = dict(ROOT=tmp_path, STAGE=tmp_path, SOURCES=names, json=json, hashlib=hashlib)
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "actual-source-gate", "exec"),
        namespace,
    )
    if corrupt is None:
        namespace["require_sources"]()
    else:
        with pytest.raises(AssertionError):
            namespace["require_sources"]()
