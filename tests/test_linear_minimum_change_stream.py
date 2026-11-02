"""Single-head stream selection; tiny fits and explicitly scripted full captures."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_workflow_linear as counted
from src.adapters import linear_fit_learner as adapter
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed

SCRIPTED = dict(calls=0, rows=0)


def packet(sign=1.0, label=1.0):
    features = np.zeros((1, 9))
    features[0, 0] = sign
    return features, np.array([[label]])


@pytest.mark.parametrize("selected", [None, "minimum_change"])
def test_single_head_forks_bind_explicit_operation_and_preserve_source(selected):
    events: list[tuple[str, int | None]] = []
    kwargs = {} if selected is None else dict(train_update=selected)
    source = counted.counted_learner(
        lambda x: events.append(("before", len(x))),
        lambda: events.append(("after", None)),
        **kwargs,
    )
    first = source.train_batch(*packet())
    before = source.snapshot_state()
    child = source.fork()
    second = child.train_batch(*packet(-1.0, 0.0))
    assert source.snapshot_state() == before
    assert child.completed_updates == child.training_examples == 2
    assert events == [("before", 1), ("after", None)] * 2
    definition = adapter.MINIMUM_CHANGE_RESIDUAL_ID if selected else adapter.LINEAR_RESIDUAL_ID
    assert first.definition == second.definition == definition
    if selected:
        # Orthogonal augmented rows: this tiny fixture can retain its first row.
        assert np.argmax(
            child.predict(np.vstack((packet()[0], packet(-1.0)[0]))), axis=1
        ).tolist() == [1, 0]
    else:
        np.testing.assert_allclose(child.predict(packet()[0]), np.full((1, 6), 1 / 6), atol=1e-12)


def test_lost_single_head_fit_preserves_paid_attempt_and_original_runtime(monkeypatch):
    calls = []
    source = counted.counted_learner(
        lambda x: calls.append("before"),
        lambda: calls.append("after"),
        train_update="minimum_change",
    )
    source.train_batch(*packet())
    calls.clear()
    before = source.snapshot_state()
    x, y = packet(-1.0, 0.0)
    batch = AttributionBatch("train", "c0", ("row0",), x, y, num_classes=6)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    stream = AttributionStream(
        source,
        (batch,),
        (),
        label_delay=1,
        checkpoint_updates=(0, 1),
        budget=budget,
        training_only=True,
    )
    original = adapter.update_categorical_linear

    def lose(*args):
        original(*args)
        raise RuntimeError("lost warm fit return")

    monkeypatch.setattr(adapter, "update_categorical_linear", lose)
    with pytest.raises(TrainingUpdateFailed) as failure:
        stream.run_until(1)
    assert calls == ["before"]
    assert source.snapshot_state() == stream._runtime._candidate.snapshot_state() == before
    assert stream._runtime.actor.snapshot_state().state == before
    with pytest.raises(ValueError, match="failed or finished"):
        stream.candidate_snapshot()
    assert budget.updates_completed == 0 and not stream.applied_updates
    assert failure.value.definition == adapter.MINIMUM_CHANGE_RESIDUAL_ID
    assert failure.value.work["fit_attempts"] == failure.value.work["fit_rows_attempted"] == 1
    assert failure.value.work["fit_returns"] == failure.value.work["candidate_commits"] == 0
    assert failure.value.work["unreturned_work_unknown"] is True


@pytest.mark.parametrize("selected", [None, "minimum_change"])
def test_cli_forwards_shared_head_choice_once_and_protects_output(selected, tmp_path, monkeypatch):
    seen = []

    def run(**kwargs):
        seen.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", run)
    output = tmp_path / "capture.json"
    flags = ["--method", "linear_fit", "--context-return", "--output", str(output)]
    if selected:
        flags += ["--train-update", selected]
    assert entry.main(flags) == 0
    assert seen == [
        dict(
            method="linear_fit",
            context_return=True,
            **(dict(train_update=selected) if selected else {}),
        )
    ]
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        entry.main(flags)
    assert len(seen) == 1 and output.read_bytes() == original


@pytest.mark.parametrize(
    "change",
    [
        dict(context_return=False),
        dict(method="bp"),
        dict(paired_serving="joint"),
        dict(private_adaptation=True),
        dict(reserved_probe=True),
        dict(catalog_study=True),
        dict(record_train_fit=True),
        dict(train_update=True),
    ],
)
def test_unsupported_single_head_modes_refuse_before_source(change, monkeypatch):
    monkeypatch.setattr(entry, "training_cases", lambda **kw: pytest.fail("source access"))
    options = dict(method="linear_fit", context_return=True, train_update="minimum_change")
    options.update(change)
    with pytest.raises(ValueError):
        entry.run_example(**options)


@pytest.mark.parametrize(
    "flags",
    [
        ["--reserved-probe"],
        ["--paired-serving", "joint"],
    ],
)
def test_unsupported_cli_refuses_before_output(flags, tmp_path, monkeypatch):
    output = tmp_path / "bad.json"
    monkeypatch.setattr(entry, "run_example", lambda **kw: pytest.fail("ran"))
    with pytest.raises(SystemExit) as failed:
        entry.main(
            [
                "--method",
                "linear_fit",
                "--context-return",
                "--train-update",
                "minimum_change",
                "--output",
                str(output),
                *flags,
            ]
        )
    assert failed.value.code == 2 and not output.exists()


def scripted_runtime(monkeypatch):
    """Actual original stream and guards; zero fits and scripted task outcomes."""
    holder: dict[str, Any] = {"operations": [], "report": None}
    original_stream = entry.AttributionStream
    original_intent = counted.require_train_update

    def stream(*args, **kwargs):
        holder["stream"] = original_stream(*args, **kwargs)
        holder["budget"] = kwargs["budget"]
        holder["birth"] = kwargs["budget"].started_at
        return holder["stream"]

    def intent(report, choice):
        holder["report"] = report
        original_intent(report, choice)

    def script(self, features, targets, *, minimum_change):
        SCRIPTED["calls"] += 1
        SCRIPTED["rows"] += len(features)
        assert SCRIPTED["calls"] <= 256 and SCRIPTED["rows"] <= 4096
        inbox = holder["stream"]._runtime._inbox
        ready = inbox._ready(inbox._last_time)
        assert len(ready) == 1
        experience, label = inbox._experiences[ready[0]], inbox._labels[ready[0]]
        assert experience.role == label.role == "train" and experience.permissions.training
        assert experience.observed_at + 1 == label.arrived_at <= inbox._last_time
        assert np.array_equal(features, experience.features) and np.array_equal(
            targets, label.targets
        )
        holder["operations"].append(minimum_change)
        prior = self.snapshot_state()
        model = CategoricalLinearFit(
            9, 6, len(features), ((0.0,) * 6,) * 10, 0, (0.0,) * min(len(features), 10), 0.0
        )
        self._state = replace(
            prior,
            model=model,
            completed_updates=prior.completed_updates + 1,
            training_examples=prior.training_examples + len(features),
        )
        return TrainingDiagnostic(
            adapter.MINIMUM_CHANGE_RESIDUAL_ID if minimum_change else adapter.LINEAR_RESIDUAL_ID,
            0.0,
        )

    monkeypatch.setattr(entry, "AttributionStream", stream)
    monkeypatch.setattr(counted, "require_train_update", intent)
    monkeypatch.setattr(adapter.LinearFitLearner, "_train_batch", script)
    monkeypatch.setattr(
        entry,
        "run_workflow_episode",
        lambda *a, **kw: dict(
            complete=True,
            success=False,
            scenario=kw.get("scenario", "ordinary"),
            training_updates=0,
            work=dict(prediction_calls_validated=0),
        ),
    )
    return holder


@pytest.mark.parametrize("selected", [None, "minimum_change"])
def test_scripted_complete_capture_preserves_single_head_roles_receipts_and_costs(
    selected, monkeypatch
):
    holder = scripted_runtime(monkeypatch)
    result = entry.run_example(
        method="linear_fit",
        context_return=True,
        **(dict(train_update=selected) if selected else {}),
    )
    assert result["complete"], result.get("error")
    assert holder["operations"] == [bool(selected)] * 12
    assert result["definition"] == "linear_context_return_ticket_workflow_v1"
    assert result.get("separate_context_heads") is None
    assert ("linear_train_update" in result) == bool(selected)
    assert result["costs"]["objective"] == (
        adapter.MINIMUM_CHANGE_RESIDUAL_ID if selected else adapter.LINEAR_RESIDUAL_ID
    )
    assert result["costs"]["fitted_coefficient_bytes"] == 480
    assert result["training_work"] == dict(
        attempted_updates=12,
        attempted_rows=72,
        returned_updates=12,
        unreturned_work_unknown=False,
        budget_committed_updates=12,
    )
    assert len(result["applied_updates"]) == 12 and len(result["episodes"]) == 24
    assert all(not row["episode"]["success"] for row in result["episodes"])
    assert all(
        result[key]
        for key in (
            "source_unchanged",
            "actor_unchanged",
            "numpy_rng_unchanged",
            "python_rng_unchanged",
        )
    )
    assert result["budget"]["started_at"] == holder["birth"]
    assert result["budget"]["limits"]["max_wall_seconds"] == 30
    assert all(result[key] == 0 for key in ("final_examples", "probe_updates", "replay_examples"))


@pytest.mark.parametrize("corruption", ["intent", "preceding_checkpoint"])
def test_scripted_capture_refuses_changed_prior_evidence_before_more_work(corruption, monkeypatch):
    holder = scripted_runtime(monkeypatch)
    original = counted.checkpoint

    def corrupt(point, state):
        if point == 4:
            if corruption == "intent":
                holder["report"]["linear_train_update"] = "cold"
            else:
                holder["report"]["checkpoints"][0]["source_updates"] = 99
        return original(point, state)

    monkeypatch.setattr(counted, "checkpoint", corrupt)
    result = entry.run_example(
        method="linear_fit", context_return=True, train_update="minimum_change"
    )
    assert result["complete"] is False
    assert ("intent changed" if corruption == "intent" else "paid evidence changed") in result[
        "error"
    ]
    assert holder["operations"] == [True] * 4
    assert (
        result["training_work"]["attempted_updates"]
        == result["training_work"]["returned_updates"]
        == 4
    )
    assert result["training_work"]["attempted_rows"] == 24
    assert len(result["applied_updates"]) == result["budget"]["updates_completed"] == 4
    assert result["budget"]["started_at"] == holder["birth"]
    assert result["linear_train_update"] == "minimum_change"
    assert result["costs"]["objective"] == adapter.MINIMUM_CHANGE_RESIDUAL_ID
