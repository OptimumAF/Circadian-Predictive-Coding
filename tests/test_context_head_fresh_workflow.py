"""Current-source private workflow; no historical research record reads."""

import json
from dataclasses import asdict, replace
from typing import Any

import pytest

from scripts import context_head_probe as head
from scripts import context_head_rehearsal as pair
from scripts import recurring_ticket_workflow_example as workflow
from test_context_head_transfer import literal_source as source_fixture


@pytest.fixture
def literal_source(monkeypatch):
    return source_fixture.__wrapped__(monkeypatch)


@pytest.mark.parametrize(
    "options",
    [
        dict(private_adaptation=1),
        dict(private_adaptation=True),
        dict(private_adaptation=True, context_return=True),
        dict(
            private_adaptation=True,
            context_return=True,
            separate_context_heads=True,
            replay="recent",
        ),
        dict(
            private_adaptation=True, context_return=True, separate_context_heads=True, method="bp"
        ),
    ],
)
def test_should_refuse_invalid_api_modes_before_data(monkeypatch, options):
    monkeypatch.setattr(workflow, "prepare_training", lambda **kw: pytest.fail("data entered"))
    with pytest.raises(ValueError):
        workflow.run_example(**options)


@pytest.mark.parametrize("flags", [[], ["--context-return"], ["--separate-context-heads"]])
def test_should_refuse_invalid_cli_before_output_or_model(tmp_path, monkeypatch, flags):
    output = tmp_path / "absent.json"
    monkeypatch.setattr(workflow, "run_example", lambda **kw: pytest.fail("model entered"))
    with pytest.raises(SystemExit):
        workflow.main(["--output", str(output), "--private-adaptation", *flags])
    assert not output.exists()


def test_should_protect_existing_output_before_model(tmp_path, monkeypatch):
    output = tmp_path / "preserve.json"
    output.write_text("preserve")
    monkeypatch.setattr(workflow, "run_example", lambda **kw: pytest.fail("model entered"))
    with pytest.raises(FileExistsError):
        workflow.main(
            [
                "--output",
                str(output),
                "--context-return",
                "--separate-context-heads",
                "--private-adaptation",
            ]
        )
    assert output.read_text() == "preserve"


def test_should_train_fresh_source_and_adapt_owned_copies_without_historical_io(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(head, "_source_bytes", lambda p: pytest.fail("historical source read"))
    original = workflow.run_example
    results = []

    def observed(**kw):
        result = original(**kw)
        results.append(result)
        return result

    monkeypatch.setattr(workflow, "run_example", observed)
    output = tmp_path / "fresh.json"
    assert (
        workflow.main(
            [
                "--output",
                str(output),
                "--context-return",
                "--separate-context-heads",
                "--private-adaptation",
            ]
        )
        == 0
    )
    r = results[0]
    assert r["complete"], r.get("error")
    assert json.loads(output.read_text())["private_adaptation"]["complete"]
    assert r["training_work"] == dict(
        attempted_updates=12,
        attempted_rows=72,
        returned_updates=12,
        unreturned_work_unknown=False,
        budget_committed_updates=12,
    )
    assert (
        r["budget"]["updates_completed"] == 12
        and r["budget"]["limits"]["max_training_updates"] == 12
    )
    assert len(r["applied_updates"]) == 12 and len(r["episodes"]) == 24
    assert r["source_unchanged"] and r["actor_unchanged"] and r["numpy_rng_unchanged"]
    private = r["private_adaptation"]
    assert (
        private["complete"]
        and private["source_provenance"] == "current_workflow_train_checkpoint12"
    )
    assert "source_sha256" not in private and "source_bytes_unchanged" not in private
    assert private["training_work"] == dict(
        attempted_updates=128,
        attempted_rows=480,
        returned_updates=128,
        unreturned_work_unknown=False,
    )
    assert r["probe_updates"] == 128 and r["replay_examples"] == 96
    assert r["final_examples"] == private["final_examples"] == private["promotions"] == 0
    assert len(private["cases"]) == 2
    for arm in private["cases"]:
        pair.require_arm(arm)
        assert len(arm["episodes"]) == 18


def test_should_preserve_historical_source_admission_with_common_pair(literal_source):
    r = pair.run_case("literal-only")
    assert r["complete"], r.get("error")
    assert r["source_sha256"] == head.HEAD_SOURCE_SHA and r["source_bytes_unchanged"]
    assert r["training_work"]["attempted_updates"] == 128
    assert r["training_work"]["attempted_rows"] == 480
    for arm in r["cases"]:
        pair.require_arm(arm)


def test_should_refuse_failed_source_task_before_private_construction(monkeypatch):
    original = workflow.run_workflow_episode
    entered = []
    original_pair = pair.run_owned_pair

    def failed(state, *a, **kw):
        result = original(state, *a, **kw)
        if state.ticket_id.startswith("task-12") and kw.get("scenario") == "ordinary":
            result["success"] = False
        return result

    def observe(*a, **kw):
        result = original_pair(*a, **kw)
        entered.append(result)
        return result

    monkeypatch.setattr(workflow, "run_workflow_episode", failed)
    monkeypatch.setattr(pair, "run_owned_pair", observe)
    r = workflow.run_example(
        context_return=True, separate_context_heads=True, private_adaptation=True
    )
    assert not r["complete"] and len(entered) == 1
    assert "all four qualified source tasks" in r["error"]
    assert entered[0]["training_work"]["attempted_updates"] == 0
    assert r["training_work"]["attempted_updates"] == 12 and r["probe_updates"] == 0


def test_should_keep_failed_source_native_work_and_never_enter_private(monkeypatch):
    original = workflow.ContextPerceptronLearner.train_batch

    def fail(owner, features, targets):
        original(owner, features, targets)
        raise RuntimeError("lost source native return")

    monkeypatch.setattr(workflow.ContextPerceptronLearner, "train_batch", fail)
    monkeypatch.setattr(pair, "run_owned_pair", lambda *a, **kw: pytest.fail("private entered"))
    r = workflow.run_example(
        context_return=True, separate_context_heads=True, private_adaptation=True
    )
    assert not r["complete"] and "lost source native return" in r["error"]
    assert r["training_work"]["attempted_updates"] == 1
    assert (
        r["training_work"]["returned_updates"] == 0
        and r["training_work"]["unreturned_work_unknown"]
    )
    assert "private_adaptation" not in r


def test_should_preserve_private_failure_and_original_source_budget(monkeypatch):
    original = pair.ContextPerceptronLearner.train_batch
    calls = []

    def fail(owner, features, targets):
        calls.append(len(features))
        answer = original(owner, features, targets)
        if len(calls) == 13:
            raise RuntimeError("lost first private return")
        return answer

    monkeypatch.setattr(pair.ContextPerceptronLearner, "train_batch", fail)
    r = workflow.run_example(
        context_return=True, separate_context_heads=True, private_adaptation=True
    )
    assert not r["complete"]
    assert r["training_work"]["attempted_updates"] == r["budget"]["updates_completed"] == 12
    assert r["private_adaptation"]["training_work"] == dict(
        attempted_updates=1, attempted_rows=3, returned_updates=0, unreturned_work_unknown=True
    )
    assert len(r["private_adaptation"]["cases"]) == 1
    assert "lost first private return" in r["error"]


def test_should_keep_default_recipe_without_private_calls(monkeypatch):
    monkeypatch.setattr(pair, "run_owned_pair", lambda *a, **kw: pytest.fail("private entered"))
    r = workflow.run_example(context_return=True, separate_context_heads=True)
    assert r["complete"] and "private_adaptation" not in r
    assert r["probe_updates"] == r["replay_examples"] == 0
    assert r["training_work"]["attempted_updates"] == 12


def test_should_reject_changed_source_budget_before_new_work(monkeypatch):
    original = pair.run_owned_pair

    def changed(state, spec, report, budget, **kw):
        budget.updates_completed += 1
        return original(state, spec, report, budget, **kw)

    monkeypatch.setattr(pair, "run_owned_pair", changed)
    r = workflow.run_example(
        context_return=True, separate_context_heads=True, private_adaptation=True
    )
    assert not r["complete"]
    assert r["private_adaptation"]["training_work"]["attempted_updates"] == 0
    assert r["training_work"]["attempted_updates"] == 12
    assert r["source_unchanged"] is None and r["actor_unchanged"] is None


@pytest.mark.parametrize("fault", ["checkpoint", "summary", "clock", "limits", "final_role"])
def test_should_refuse_corrupted_owned_evidence_before_model_work(fault, monkeypatch):
    from test_context_catalog_control import initial

    # Literal prior state/counters, not a newly trained or scientific source.
    state = initial()
    report: dict[str, Any] = dict(
        definition="separate_context_head_ticket_workflow_v1",
        source_unchanged=True,
        actor_unchanged=True,
        private_source_training=dict(completed_updates=12, applied_update_count=12),
        summary=dict(literal=True),
        checkpoints=[dict(source_updates=n, state=asdict(state)) for n in (0, 4, 8, 12)],
        episodes=[
            dict(
                source_updates=12,
                policy_version=v,
                concept=c,
                role="outer_selection",
                episode=dict(scenario="ordinary", complete=True, success=True),
            )
            for v, c in pair.probe.TASKS
        ],
    )
    spec = head.DecisionFeatureSpec(head.CONTEXT_FEATURES, head.CATALOG)
    monkeypatch.setattr(
        pair.ContextPerceptronLearner, "__init__", lambda *a: pytest.fail("model created")
    )
    with pair.ProcessRssSampler() as memory:
        budget = pair.ToyBudgetSession(pair._limits(12), pair.monotonic)
        budget.attach_memory(memory)
        for _ in range(12):
            budget.before_update()
            budget.record_update()

        def corrupt():
            if fault == "checkpoint":
                report["checkpoints"][-1]["state"]["completed_updates"] = 13
            elif fault == "summary":
                report["summary"].clear()
            elif fault == "clock":
                budget.last_clock += 1
            elif fault == "limits":
                budget.budget = replace(budget.budget, max_training_updates=13)
            else:
                report["episodes"][0]["role"] = "final_test"

        result = pair.run_owned_pair(
            state, spec, report, budget, check_boundary=corrupt, memory=memory
        )
    assert not result["complete"] and result["training_work"]["attempted_updates"] == 0
    assert result["cases"] == []
