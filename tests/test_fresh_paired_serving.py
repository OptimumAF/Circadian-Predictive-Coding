"""Owned ordinary serving from a live qualified source, without artifact IO."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from scripts import context_head_live as live
from scripts import context_head_probe as probe
from scripts import context_head_rehearsal as private
from scripts import recurring_ticket_workflow_example as workflow
from src.app.toy_execution_budget import ToyBudgetSession
from src.shared.process_memory import ProcessRssSampler
from test_context_catalog_control import initial


@pytest.fixture
def owned():
    state = initial()
    spec = probe.DecisionFeatureSpec(probe.CONTEXT_FEATURES, probe.CATALOG)
    report = dict(
        definition="separate_context_head_ticket_workflow_v1",
        source_unchanged=True,
        actor_unchanged=True,
        serving_source_training=dict(completed_updates=12, applied_update_count=12),
        summary=dict(fixture="literal qualified metadata, not a scientific run"),
        checkpoints=[dict(source_updates=n, state=asdict(state)) for n in (0, 4, 8, 12)],
        episodes=[
            dict(
                source_updates=12,
                policy_version=v,
                concept=c,
                role="outer_selection",
                episode=dict(scenario="ordinary", complete=True, success=True),
            )
            for v, c in private.probe.TASKS
        ],
    )
    with ProcessRssSampler() as memory:
        budget = ToyBudgetSession(private._limits(12), private.monotonic)
        budget.attach_memory(memory)
        # Simulate already-paid counters only. This fixture performs NO fitting.
        for _ in range(12):
            budget.before_update()
            budget.record_update()
        binding = (id(budget.budget), budget.budget, budget.started_at, id(budget.clock))

        def check_boundary():
            if binding != (id(budget.budget), budget.budget, budget.started_at, id(budget.clock)):
                raise ValueError("original source budget origin changed")

        yield state, spec, report, budget, memory, check_boundary


@pytest.mark.parametrize("value", [True, "unknown", "", 1])
def test_should_refuse_invalid_fresh_options_before_data(value, monkeypatch):
    monkeypatch.setattr(workflow, "prepare_training", lambda **kw: pytest.fail("data entered"))
    with pytest.raises(ValueError):
        workflow.run_example(paired_serving=value, context_return=True, separate_context_heads=True)


@pytest.mark.parametrize(
    "extra",
    [
        dict(),
        dict(context_return=True),
        dict(context_return=True, separate_context_heads=True, private_adaptation=True),
        dict(context_return=True, separate_context_heads=True, replay="recent"),
        dict(context_return=True, separate_context_heads=True, method="bp"),
    ],
)
def test_should_refuse_conflicting_recipe_before_data(extra, monkeypatch):
    monkeypatch.setattr(workflow, "prepare_training", lambda **kw: pytest.fail("data entered"))
    with pytest.raises(ValueError):
        workflow.run_example(paired_serving="joint", **extra)


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--context-return"],
        ["--separate-context-heads"],
        ["--context-return", "--separate-context-heads", "--private-adaptation"],
    ],
)
def test_should_refuse_bad_cli_without_output(tmp_path, monkeypatch, flags):
    monkeypatch.setattr(workflow, "run_example", lambda **kw: pytest.fail("model entered"))
    output = tmp_path / "invalid.json"
    with pytest.raises(SystemExit):
        workflow.main(["--output", str(output), "--paired-serving", "joint", *flags])
    assert not output.exists()


@pytest.mark.parametrize(
    "fault", ["failed_task", "role", "checkpoint", "schema", "budget", "clock", "limits"]
)
def test_should_refuse_invalid_owned_source_before_child_construction(owned, monkeypatch, fault):
    state, spec, report, budget, memory, check_boundary = owned
    if fault == "failed_task":
        report["episodes"][0]["episode"]["success"] = False
    elif fault == "role":
        report["episodes"][0]["role"] = "final_test"
    elif fault == "checkpoint":
        report["checkpoints"][-1]["state"]["training_examples"] += 1
    elif fault == "schema":
        spec = replace(spec, vocabulary=tuple(sorted((*spec.vocabulary, "extra"))))
    elif fault == "budget":
        budget.updates_completed += 1
    elif fault == "clock":
        budget.started_at -= 1.0
    else:
        budget.budget = replace(budget.budget, max_training_updates=13)
    monkeypatch.setattr(
        live.ContextPerceptronLearner, "__init__", lambda *a: pytest.fail("child created")
    )
    result = live.run_owned_case(
        state, spec, report, budget, control="joint", check_boundary=check_boundary, memory=memory
    )
    assert not result["complete"] and not result["owned_source_verified"]
    assert result["training_work"]["attempted_updates"] == 0
    assert "source_sha256" not in result


@pytest.mark.parametrize("fault", ["summary", "episodes", "row", "clock", "source_flag"])
def test_should_refuse_changed_or_replaced_original_evidence_after_handoff(owned, fault):
    state, spec, report, budget, memory, check_boundary = owned
    _, guard = private.prepare_owned_source(
        state,
        spec,
        report,
        budget,
        check_boundary=check_boundary,
        training_key="serving_source_training",
    )
    if fault == "summary":
        report["summary"] = deepcopy(report["summary"])
    elif fault == "episodes":
        report["episodes"] = list(report["episodes"])
    elif fault == "row":
        report["episodes"][0] = deepcopy(report["episodes"][0])
    elif fault == "clock":
        budget.before_final()  # Not an observation through the original witness.
    else:
        report["source_unchanged"] = False
    with pytest.raises(ValueError):
        guard()


def test_should_serialize_worker_observations_without_renewing_original_session(owned):
    state, spec, report, budget, memory, check_boundary = owned
    _, guard = private.prepare_owned_source(
        state,
        spec,
        report,
        budget,
        check_boundary=check_boundary,
        training_key="serving_source_training",
    )
    pin = (id(budget.budget), budget.started_at, id(budget.clock))
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: guard(), range(20))) == [None] * 20
    guard()
    assert pin == (id(budget.budget), budget.started_at, id(budget.clock))
    assert budget.updates_completed == 12 and budget.replay_examples_completed == 0


def test_should_preserve_unknown_work_when_delegated_report_is_lost(owned, monkeypatch):
    state, spec, report, budget, memory, check_boundary = owned

    def lost(*a, **kw):
        raise RuntimeError("lost delegated result after possible publication")

    monkeypatch.setattr(live, "run_case", lost)
    result = live.run_owned_case(
        state, spec, report, budget, control="joint", check_boundary=check_boundary, memory=memory
    )
    assert not result["complete"] and result["delegated_work_unknown"]
    assert result["training_work"] == dict(
        attempted_updates=None,
        attempted_rows=None,
        returned_updates=None,
        unreturned_work_unknown=True,
    )
    assert budget.updates_completed == 12 and not result["owned_source_verified"]


@pytest.mark.parametrize("child_present", [True, False])
def test_should_not_count_missing_child_work_as_zero_in_failure_totals(
    tmp_path, monkeypatch, child_present
):
    value = dict(
        complete=False,
        training_work=dict(attempted_updates=12, attempted_rows=72),
        invalid=object(),
    )
    if child_present:
        value["paired_serving"] = dict(complete=False, training_work={})
    monkeypatch.setattr(workflow, "run_example", lambda **kw: value)
    output = tmp_path / "unknown-work.json"
    assert (
        workflow.main(
            [
                "--context-return",
                "--separate-context-heads",
                "--paired-serving",
                "joint",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    saved = json.loads(output.read_text())
    assert not saved["complete"] and not saved["validated_receipts_available"]
    assert saved["observer_counts"]["train_passes"] == (None if child_present else 12)
    assert saved["observer_counts"]["training_rows"] == (None if child_present else 72)


@pytest.mark.parametrize("control", ["current", "joint"])
def test_should_train_fresh_source_then_publish_ordinary_child_with_no_historical_io(
    monkeypatch, tmp_path, control
):
    monkeypatch.setattr(live, "_source_bytes", lambda p: pytest.fail("historical source read"))
    monkeypatch.setattr(
        private, "run_owned_pair", lambda *a, **kw: pytest.fail("private probes entered")
    )
    observed = []
    original = workflow.run_example

    def run(**kw):
        r = original(**kw)
        observed.append(r)
        return r

    monkeypatch.setattr(workflow, "run_example", run)
    output = tmp_path / "fresh-serving.json"
    assert (
        workflow.main(
            [
                "--context-return",
                "--separate-context-heads",
                "--paired-serving",
                control,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    r = observed[0]
    assert r["complete"], r.get("error")
    child = r["paired_serving"]
    assert json.loads(output.read_text())["paired_serving"]["complete"]
    assert r["training_work"]["attempted_updates"] == r["training_work"]["returned_updates"] == 12
    assert r["training_work"]["attempted_rows"] == 72
    assert r["budget"]["updates_completed"] == r["budget"]["limits"]["max_training_updates"] == 12
    assert r["budget"]["limits"]["max_wall_seconds"] == 30
    assert r["stream"]["completed_updates"] == 12
    assert r["stream"]["training_examples"] == 72 and r["stream"]["score_forward_calls"] == 0
    assert len(r["applied_updates"]) == 12
    assert r["source_unchanged"] and r["actor_unchanged"] and r["numpy_rng_unchanged"]
    assert child["complete"] and child["owned_source_verified"]
    assert child["source_provenance"] == "current_workflow_train_checkpoint12"
    assert "source_sha256" not in child and "private_adaptation" not in r
    assert child["training_work"] == dict(
        attempted_updates=6, attempted_rows=48, returned_updates=6
    )
    assert child["budget"]["updates"] == 6 and child["rollback_equal_source"]
    assert child["memory_scope"] == "shared_source_and_serving"
    assert len(child["live_serving"]["records"]) == 24 and len(child["episodes"]) == 12
    assert all(e["episode"]["success"] for e in child["episodes"][:6])
    assert child["final_actor"]["model_state"] == r["checkpoints"][-1]["state"]
    assert (
        r["probe_updates"]
        == r["replay_examples"]
        == child["probe_updates"]
        == child["final_examples"]
        == 0
    )


@pytest.mark.parametrize("phase", ["source", "child"])
def test_should_preserve_lost_native_work_and_never_refund_source_budget(monkeypatch, phase):
    original = workflow.ContextPerceptronLearner.train_batch
    calls = 0

    def fail(owner, x, y):
        nonlocal calls
        calls += 1
        result = original(owner, x, y)
        if calls == (1 if phase == "source" else 13):
            raise RuntimeError("lost original native return")
        return result

    monkeypatch.setattr(workflow.ContextPerceptronLearner, "train_batch", fail)
    r = workflow.run_example(
        context_return=True, separate_context_heads=True, paired_serving="joint"
    )
    assert not r["complete"] and "lost original native return" in r["error"]
    if phase == "source":
        assert calls == r["training_work"]["attempted_updates"] == 1
        assert r["training_work"]["returned_updates"] == 0 and "paired_serving" not in r
    else:
        assert calls == 13 and r["training_work"]["attempted_updates"] == 12
        assert r["budget"]["updates_completed"] == 12
        assert r["paired_serving"]["training_work"] == dict(
            attempted_updates=1, attempted_rows=9, returned_updates=0
        )
    assert r["probe_updates"] == 0


def test_should_keep_default_source_without_entering_new_serving(monkeypatch):
    monkeypatch.setattr(live, "run_owned_case", lambda *a, **kw: pytest.fail("serving entered"))
    r = workflow.run_example(context_return=True, separate_context_heads=True)
    assert r["complete"] and "paired_serving" not in r
    assert r["training_work"]["attempted_updates"] == 12


def test_should_block_child_before_data_when_source_task_qualification_fails(monkeypatch):
    episode = workflow.run_workflow_episode

    def fail_task(state, *a, **kw):
        result = episode(state, *a, **kw)
        if state.ticket_id.startswith("task-12") and kw.get("scenario") == "ordinary":
            result["success"] = False
        return result

    monkeypatch.setattr(workflow, "run_workflow_episode", fail_task)
    monkeypatch.setattr(live, "prepare_data", lambda *a, **kw: pytest.fail("child data entered"))
    r = workflow.run_example(
        context_return=True, separate_context_heads=True, paired_serving="joint"
    )
    assert not r["complete"] and "all four qualified source tasks" in r["error"]
    assert r["training_work"]["attempted_updates"] == r["budget"]["updates_completed"] == 12
    assert r["paired_serving"]["training_work"]["attempted_updates"] == 0


def test_should_protect_existing_output_and_forward_exact_option(tmp_path, monkeypatch):
    seen = []

    def run(**kw):
        seen.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(workflow, "run_example", run)
    output = tmp_path / "output.json"
    flags = [
        "--context-return",
        "--separate-context-heads",
        "--paired-serving",
        "joint",
        "--output",
        str(output),
    ]
    assert workflow.main(flags) == 0
    assert seen == [dict(context_return=True, separate_context_heads=True, paired_serving="joint")]
    with pytest.raises(FileExistsError):
        workflow.main(flags)
    assert len(seen) == 1
