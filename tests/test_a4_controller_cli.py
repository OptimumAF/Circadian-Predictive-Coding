"""Public fresh chooser command with scripted collection, fitting and games."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import sys
from typing import Any

import pytest

from scripts import minigrid_a4_adaptation as fresh
from scripts import minigrid_controller_learning as learning
from scripts import minigrid_affine_workflow as workflow
from scripts import run_minigrid_linear_fit_control as cli
from tests import test_minigrid_controller_learning as current
from tests import test_minigrid_controller_bank as bank
from tests.test_a4_fresh_workflow import bind_collection

WORK = dict(collection_envs=0, teacher_steps=0, cli_calls=0)
FLAG = "--context-linear-controller"


class Harness:
    def __init__(self, patch, mode="retain_arrived"):
        self.h = bank.BankHarness(patch) if mode == "retain_arrived" else current.Harness(patch)
        self.box = bind_collection(patch)
        self.mode = mode
        self.inputs: list[tuple[str, ...]] = []
        self.calls: list[bool] = []
        original = learning.run_controller_learning
        collect = fresh._collect_fresh_a4
        patch.setattr(fresh, "monotonic", lambda: self.h.time)

        def collection(check, records):
            self.records = records
            self.born_before_collection = self.owner
            return collect(check, records)

        self.owner: Any = None
        original_attach = fresh.ToyBudgetSession.attach_memory

        def attach(owner, sampler):
            if owner.budget.max_training_updates == 0:
                self.owner = owner
            return original_attach(owner, sampler)

        patch.setattr(fresh.ToyBudgetSession, "attach_memory", attach)
        patch.setattr(fresh, "_collect_fresh_a4", collection)

        def invoke(episodes, *, readout_budget, check_boundary, retain_arrived):
            assert readout_budget is self.owner is self.born_before_collection
            assert readout_budget.started_at == 0.0 and readout_budget.updates_completed == 0
            self.inputs.append(tuple(e.episode_id for e in episodes))
            self.calls.append(retain_arrived)
            return original(
                episodes,
                readout_budget=readout_budget,
                check_boundary=check_boundary,
                retain_arrived=retain_arrived,
            )

        patch.setattr(learning, "run_controller_learning", invoke)

    def run(self, patch, output=None):
        return invoke_cli(
            patch,
            [
                "--example-affine-a4",
                FLAG,
                self.mode,
                *(["--output", str(output)] if output is not None else []),
            ],
        )

    def finish(self):
        WORK["collection_envs"] += len(self.box["environments"])
        WORK["teacher_steps"] += self.box["teacher_calls"]
        assert all(e.closed for e in self.box["environments"])


def invoke_cli(patch, args):
    WORK["cli_calls"] += 1
    patch.setattr(sys, "argv", ["affine", *args])
    output = io.StringIO()
    code: Any = 0
    with redirect_stdout(output):
        try:
            cli.main()
        except SystemExit as error:
            code = error.code
    return code, json.loads(output.getvalue()) if output.getvalue() else None


@pytest.fixture(scope="module", params=["current_only", "retain_arrived"])
def integrated(request, tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        harness = Harness(patch, request.param)
        destination = tmp_path_factory.mktemp("controller-cli") / "report.json"
        code, report = harness.run(patch, destination)
        harness.finish()
        assert code == 0 and report["status"] == "complete", report
        return report, destination, harness


def test_should_execute_public_collection_fit_apply_and_exclusive_json(integrated):
    r, path, h = integrated
    assert json.loads(path.read_text()) == r and path.stat().st_size < 8 * 1024**2
    assert h.calls == [h.mode == "retain_arrived"]
    assert h.inputs == [tuple(e.episode_id for e in current.sources())]
    assert len(r["collection"]) == 2 and h.box["teacher_calls"] == 18
    assert all(c["closed"] and c["role"] == "train" for c in r["collection"])
    assert r["controller_work"] == dict(attempts=1, returned=1, unreturned_work_unknown=False)
    assert r["readout_budget"]["started_at"] == r["readout_budget"]["observed_started_at"] == 0.0
    assert r["readout_budget"]["updates"] == r["readout_budget"]["replay"] == 0
    assert r["readout_budget"]["limits"]["max_wall_seconds"] == 150


def test_should_project_known_costs_and_pending_without_new_scores(integrated):
    r, _, h = integrated
    s, t = r["summary"], r["training"]
    assert s["complete"] and s["private_candidate_only"] and r["games"] == []
    assert s["choices"] == t["fitted"]["choices"] and s["estimates"] == t["fitted"]["estimates"]
    assert s["warmup"]["reported_updates"] == 2
    assert [x["reported_updates"] for x in s["discarded_trials"]] == [0, 4, 0, 4]
    assert all(
        x["reported_native_rows"] == n
        for x, n in zip(
            s["discarded_trials"],
            (0, 72, 0, 72) if h.mode == "retain_arrived" else (0, 36, 0, 36),
            strict=True,
        )
    )
    if h.mode == "retain_arrived":
        assert s["reported_updates"] == 12 and s["reported_native_rows"] == 198
        assert (
            s["application"]["reported_updates"] == 2 and s["application"]["unfitted_packets"] == 2
        )
        assert s["application"]["latest_native_checkpoint"] == dict(
            relative_updates=2, native_updates=4, native_rows=54
        )
        assert s["application"]["outcomes"]["after_task"] == dict(old_A3=2, new_A4=2)
    else:
        assert s["reported_updates"] == 10 and s["reported_native_rows"] == 90
        assert (
            s["application"]["reported_updates"] == 0 and s["application"]["unfitted_packets"] == 4
        )
        assert s["application"]["outcomes"]["after_task"] == dict(old_A3=2, new_A4=0)
    assert s["application"]["window_complete"] and not s["application"]["training_complete"]
    assert t["source_unchanged"]


@pytest.mark.parametrize("mode", ["current_only", "retain_arrived"])
def test_should_describe_prospective_mode_without_readers_collection_models(monkeypatch, mode):
    def forbidden(*args, **kwargs):
        pytest.fail("describe must not execute work")

    monkeypatch.setattr(fresh, "run_fresh_adaptation", forbidden)
    monkeypatch.setattr(fresh, "_collect_fresh_a4", forbidden)
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(learning, "ContextLinearLearner", forbidden)
    code, r = invoke_cli(monkeypatch, ["--example-affine-a4", FLAG, mode, "--describe"])
    assert code == 0 and r["controller_mode"] == mode and r["historical_source_reads"] == 0
    assert r["fit_updates_upper_bound"] == 14
    assert r["fit_rows_upper_bound"] == (234 if mode == "retain_arrived" else 126)
    assert r["training"] == learning.describe_controller_learning(
        retain_arrived=mode == "retain_arrived"
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--source", "missing"],
        ["--retained-source", "missing"],
        ["--ready-feedback"],
        ["--prequential-feedback"],
        ["--reference-retain-arrived"],
        ["--fit-updates", "1"],
        ["--context-linear-reference", "periodic"],
        ["--context-linear-stream"],
        ["--describe", "--output", "unused"],
    ],
)
def test_should_refuse_conflicts_before_any_work(monkeypatch, extra):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid CLI must not execute work")

    monkeypatch.setattr(fresh, "run_fresh_adaptation", forbidden)
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    code, r = invoke_cli(monkeypatch, ["--example-affine-a4", FLAG, "retain_arrived", *extra])
    assert code == 2 and r is None


def test_should_require_fresh_flag(monkeypatch):
    code, r = invoke_cli(monkeypatch, [FLAG, "retain_arrived", "--describe"])
    assert code == 2 and r is None


@pytest.mark.parametrize("mode", [True, 1, "other"])
def test_should_refuse_invalid_api_mode_without_collection(monkeypatch, mode):
    monkeypatch.setattr(fresh, "_collect_fresh_a4", lambda *a: pytest.fail("invalid API executed"))
    with pytest.raises(ValueError, match="exact"):
        fresh.run_fresh_adaptation(controller_mode=mode)


def test_should_refuse_api_reference_controller_combination():
    from src.core.reference_controller import ReferencePolicy

    with pytest.raises(ValueError, match="without a reference"):
        fresh.run_fresh_adaptation(
            controller_mode="retain_arrived", reference_policy=ReferencePolicy("periodic")
        )


def test_should_preserve_existing_default_and_reference_dispatch(monkeypatch):
    calls: list[Any] = []

    def legacy(**kwargs):
        calls.append(kwargs)
        return dict(status="complete", legacy=True)

    monkeypatch.setattr(workflow, "run_workflow", legacy)
    code, r = invoke_cli(monkeypatch, ["--example-affine-a4"])
    assert code == 0 and r["legacy"] and calls == [dict(a4_adaptation=True)]

    def reference(args, parser):
        calls.append(args.context_linear_reference)

    monkeypatch.setattr(fresh, "handle_reference_cli", reference)
    code, r = invoke_cli(
        monkeypatch,
        [
            "--example-affine-a4",
            "--context-linear-reference",
            "drift",
            "--ready-feedback",
            "--describe",
        ],
    )
    assert code == 0 and r is None and calls[-1] == "drift"


def test_should_refuse_existing_output_before_collection(monkeypatch, tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("keep")
    monkeypatch.setattr(
        fresh, "run_fresh_adaptation", lambda **kw: pytest.fail("existing path must not execute")
    )
    with pytest.raises(FileExistsError):
        invoke_cli(
            monkeypatch, ["--example-affine-a4", FLAG, "retain_arrived", "--output", str(path)]
        )
    assert path.read_text() == "keep"


def test_should_preserve_bytes_on_exclusive_creation_race_without_retry(monkeypatch, tmp_path):
    path = tmp_path / "race.json"
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        path.write_text("racing owner")
        return dict(status="complete")

    monkeypatch.setattr(fresh, "run_fresh_adaptation", run)
    with pytest.raises(FileExistsError):
        invoke_cli(
            monkeypatch, ["--example-affine-a4", FLAG, "retain_arrived", "--output", str(path)]
        )
    assert len(calls) == 1 and path.read_text() == "racing owner"


@pytest.mark.parametrize("bad", ["large", "nonfinite"])
def test_should_refuse_oversized_or_nonfinite_serialization_without_retry(
    monkeypatch, tmp_path, bad
):
    calls = []
    path = tmp_path / "bad.json"

    def run(**kwargs):
        calls.append(kwargs)
        return dict(
            status="complete", value="x" * (8 * 1024**2) if bad == "large" else float("nan")
        )

    monkeypatch.setattr(fresh, "run_fresh_adaptation", run)
    with pytest.raises(ValueError):
        invoke_cli(
            monkeypatch, ["--example-affine-a4", FLAG, "retain_arrived", "--output", str(path)]
        )
    assert len(calls) == 1 and not path.exists()


def test_should_persist_failed_collection_and_not_invoke_controller(monkeypatch, tmp_path):
    h = Harness(monkeypatch)
    h.box["fault"] = "step"
    path = tmp_path / "failure.json"
    code, r = h.run(monkeypatch, path)
    h.finish()
    assert code == 2 and r["status"] == "failed" and not r["summary"]["complete"]
    assert r["controller_work"] == dict(attempts=0, returned=0, unreturned_work_unknown=False)
    assert h.calls == [] and h.h.train_calls == 0 and len(r["collection"]) == 1
    assert r["training"] is None and json.loads(path.read_text()) == r
    assert r["summary"]["reported_native_rows"] is None


def test_should_preserve_unknown_controller_return_without_inventing_zero_cost(monkeypatch):
    h = Harness(monkeypatch)

    def lost(*args, **kwargs):
        raise RuntimeError("lost controller return")

    monkeypatch.setattr(learning, "run_controller_learning", lost)
    code, r = h.run(monkeypatch)
    h.finish()
    assert code == 2 and r["status"] == "failed"
    assert r["controller_work"] == dict(attempts=1, returned=0, unreturned_work_unknown=True)
    assert len(r["collection"]) == 2 and r["source"] is not None
    assert not r["summary"]["complete"] and r["summary"]["reported_updates"] is None
    assert r["readout_budget"]["started_at"] == 0.0


@pytest.mark.parametrize("damage", ["value", "replacement"])
def test_should_refuse_corrupted_preceding_collection_during_child_work(monkeypatch, damage):
    h = Harness(monkeypatch)

    def corrupt(owner):
        if h.h.train_calls == 1:
            if damage == "value":
                h.records[0]["total_return"] = -1.0
            else:
                h.records[0]["transitions"] = deepcopy(h.records[0]["transitions"])

    h.h.on_train = corrupt
    code, r = h.run(monkeypatch)
    h.finish()
    assert code == 2 and r["status"] == "failed" and "capture_error" in r
    assert h.h.train_calls == 1 and not r["summary"]["complete"]
    assert r["collection"][0]["total_return"] == 0.5
    assert r["training"]["budgets"][0]["updates"] == 1


def test_should_refuse_equal_replacement_in_returned_child_evidence(monkeypatch):
    h = Harness(monkeypatch)
    original = fresh.paid_report_guard

    def guard(value):
        check, saved = original(value)
        if type(value) is dict and "fitted" in value and "trials" in value:

            def damaged():
                value["trials"][0] = deepcopy(value["trials"][0])
                check()

            return damaged, saved
        return check, saved

    monkeypatch.setattr(fresh, "paid_report_guard", guard)
    code, r = h.run(monkeypatch)
    h.finish()
    assert code == 2 and r["status"] == "failed" and "capture_error" in r
    assert r["training"]["status"] == "complete" and not r["summary"]["complete"]
    assert r["summary"]["reported_updates"] == 12 and r["controller_work"]["returned"] == 1


@pytest.mark.parametrize("status", ["complete", "failed"])
def test_should_emit_complete_nested_evidence_without_indentation_overflow(
    monkeypatch, tmp_path, status
):
    # Similar depth and numeric shape to the paid native report; no model work.
    receipt: dict[str, Any] = {"weights": [[0.125] * 296 for _ in range(7)]}
    for _ in range(8):
        receipt = {"scope": receipt}
    report = dict(status=status, receipts=[receipt] * 140, negative=-0.25, label="\u2603")
    pretty_bytes = len((json.dumps(report, indent=2) + "\n").encode("utf8"))
    assert 8 * 1024**2 < pretty_bytes <= 16 * 1024**2
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return report

    monkeypatch.setattr(fresh, "run_fresh_adaptation", run)
    path = tmp_path / "nested.json"
    code, emitted = invoke_cli(
        monkeypatch, ["--example-affine-a4", FLAG, "retain_arrived", "--output", str(path)]
    )
    assert code == (0 if status == "complete" else 2)
    assert len(calls) == 1 and emitted == json.loads(path.read_text()) == report
    assert path.stat().st_size < 8 * 1024**2


@pytest.mark.parametrize("excess", [0, 1])
def test_should_enforce_exact_encoded_byte_limit_including_final_lf(monkeypatch, tmp_path, excess):
    cap = 8 * 1024**2
    report = dict(status="complete", label="\u2603", value="")
    overhead = len((json.dumps(report, separators=(",", ":")) + "\n").encode("utf8"))
    report["value"] = "x" * (cap - overhead + excess)
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return report

    monkeypatch.setattr(fresh, "run_fresh_adaptation", run)
    path = tmp_path / "boundary.json"
    args = ["--example-affine-a4", FLAG, "retain_arrived", "--output", str(path)]
    if excess:
        with pytest.raises(ValueError, match="exceeds8MiB"):
            invoke_cli(monkeypatch, args)
        assert not path.exists()
    else:
        code, emitted = invoke_cli(monkeypatch, args)
        assert code == 0 and emitted == report
        encoded = path.read_bytes()
        assert len(encoded) == cap and encoded.endswith(b"\n") and b"\r" not in encoded
        assert json.loads(encoded) == report
    assert len(calls) == 1


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_should_reject_nested_nonfinite_values_without_output_or_retry(
    monkeypatch, tmp_path, number
):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(status="failed", evidence=[dict(value=number)])

    monkeypatch.setattr(fresh, "run_fresh_adaptation", run)
    path = tmp_path / "nonfinite.json"
    with pytest.raises(ValueError):
        invoke_cli(
            monkeypatch, ["--example-affine-a4", FLAG, "retain_arrived", "--output", str(path)]
        )
    assert len(calls) == 1 and not path.exists()
