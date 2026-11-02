"""Isolated minimum-change composition on the existing retained probe, no study."""

from copy import deepcopy
from typing import Any
import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import context_head_live as live
from scripts import context_retained_probe as composed
from scripts import probe_ticket_policy_learning as recipe
from scripts.ticket_linear_probe import MinimumChangeProbeLearner, bind_readouts
from scripts.context_training_streams import WITHHELD_CONTEXT_RECIPES
from src.adapters import context_linear_fit_learner as context
from src.adapters import linear_fit_learner as linear
from src.shared.process_memory import ProcessRssSampler
from test_context_retained_probe import scope_fixture
from test_context_retained_publication import qualified_literal
from test_context_catalog_replay import spec


@pytest.mark.parametrize("value", [True, 1, "cold", "unknown"])
def test_should_refuse_invalid_option_before_source_or_probe(value, monkeypatch):
    monkeypatch.setattr(entry, "training_cases", lambda **kw: pytest.fail("source"))
    with pytest.raises(ValueError, match="minimum-change"):
        entry.run_example(probe_update=value)
    with pytest.raises(ValueError, match="minimum-change"):
        live.run_case(None, _probe_update=value)
    with pytest.raises(ValueError, match="minimum-change"):
        composed.run_probe(None, None, None, None, None, {}, {}, probe_update=value)
    with pytest.raises(ValueError, match="minimum-change"):
        recipe.run_arm(None, None, (), None, None, None, "continued", linear_probe_update=value)


@pytest.mark.parametrize(
    "change",
    [
        dict(method="bp"),
        dict(context_return=False),
        dict(separate_context_heads=False),
        dict(paired_serving="retained_joint"),
        dict(probe_rehearsal="joint"),
    ],
)
def test_should_refuse_mixed_modes_before_source(change, monkeypatch):
    args = dict(
        method="linear_fit",
        context_return=True,
        separate_context_heads=True,
        paired_serving="retained_joint_probe",
        probe_update="minimum_change",
    )
    args.update(change)
    monkeypatch.setattr(entry, "training_cases", lambda **kw: pytest.fail("source"))
    with pytest.raises(ValueError, match="minimum-change"):
        entry.run_example(**args)


def test_should_pin_paid_update_intent_without_model_work():
    row = dict(
        mode="continued",
        prior_native_updates=23,
        prior_native_rows=144,
        readout_work=[],
        episodes=[],
        linear_probe_update="minimum_change",
    )
    guard, _ = bind_readouts(row, [], lambda: None)
    row["linear_probe_update"] = "cold"
    with pytest.raises(ValueError, match="paid endpoint"):
        guard()
    assert row["linear_probe_update"] == "minimum_change"
    with pytest.raises(ValueError, match="intent"):
        recipe.run_arm(None, None, (), None, None, None, "continued", row=row)


def test_should_keep_disposable_fork_choice_and_source_cold_without_training():
    runtime, _, _, _ = scope_fixture()
    source = runtime._candidate
    before = source.snapshot_state()
    private = MinimumChangeProbeLearner(before.policy)
    private.restore_state(before)
    child = private.fork()
    assert type(source.fork()) is context.ContextLinearFitLearner
    assert type(child) is MinimumChangeProbeLearner
    object.__setattr__(child._state.heads[0].model, "squared_error", 23.0)
    assert source.snapshot_state() == private.snapshot_state() == before


@pytest.mark.parametrize("phase", ["probe", "return"])
@pytest.mark.parametrize("fault", ["missing", "empty", "truncated", "definition"])
def test_should_refuse_incomplete_paid_diagnostic_intent_without_model_work(phase, fault):
    # Only the prefix reaching this gate; not an executed/completed-arm claim.
    metadata: dict[str, Any] = dict(
        status="complete",
        diagnostics=[
            dict(definition=context.CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID) for _ in range(16)
        ],
    )
    probe, returning = deepcopy(metadata), deepcopy(metadata)
    damaged = probe if phase == "probe" else returning
    if fault == "missing":
        del damaged["diagnostics"]
    elif fault == "empty":
        damaged["diagnostics"] = []
    elif fault == "truncated":
        damaged["diagnostics"].pop()
    else:
        damaged["diagnostics"][0]["definition"] = context.CONTEXT_LINEAR_RESIDUAL_ID
    probe["return_phase"] = returning
    row = dict(
        linear_probe_update="minimum_change",
        probe=probe,
        sessions=[
            dict(
                limits=recipe.asdict(recipe.HEAD_LIMITS), updates=16, started_at=0.0, last_clock=1.0
            )
            for _ in range(2)
        ],
    )
    with pytest.raises(ValueError, match="diagnostic"):
        recipe._require_arm(
            row,
            separate_context_heads=True,
            linear_fit=True,
            head_tasks=recipe.TRANSFER_HEAD_TASKS,
            linear_probe_update="minimum_change",
        )


def test_should_forward_cli_option_with_original_work_counts(monkeypatch, tmp_path):
    from scripts import context_catalog_learning as writer

    calls, observed = [], []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(
            complete=True,
            training_work=dict(attempted_updates=12, attempted_rows=72),
            paired_serving=dict(
                training_work=dict(attempted_updates=11, attempted_rows=72),
                isolated_transfer_probe=dict(
                    private_work=dict(attempted_updates=32, attempted_training_examples=144)
                ),
            ),
        )

    def write(path, report, **kwargs):
        observed.append(kwargs["observed_work"]["counts"])
        return True

    monkeypatch.setattr(entry, "run_example", run)
    monkeypatch.setattr(writer, "write_catalog_report", write)
    assert (
        entry.main(
            [
                "--method",
                "linear_fit",
                "--context-return",
                "--separate-context-heads",
                "--paired-serving",
                "retained_joint_probe",
                "--probe-update",
                "minimum_change",
                "--output",
                str(tmp_path / "fixture.json"),
            ]
        )
        == 0
    )
    assert calls[0]["probe_update"] == "minimum_change"
    assert observed == [dict(train_passes=55, training_rows=288)]
    calls.clear()
    with pytest.raises(SystemExit):
        entry.main(["--probe-update", "minimum_change"])
    assert calls == []


def test_should_execute_retained_stream_and_private_update_with_existing_budgets(
    monkeypatch, record_property
):
    cold, private = [], []
    original_cold = context.ContextLinearFitLearner.train_batch
    original_private = context.ContextLinearFitLearner.train_batch_minimum_change
    rng = linear._rng_signature()

    def cold_train(learner, x, y):
        cold.append(len(x))
        return original_cold(learner, x, y)

    def private_train(learner, x, y):
        assert type(learner) is MinimumChangeProbeLearner
        private.append(len(x))
        return original_private(learner, x, y)

    monkeypatch.setattr(context.ContextLinearFitLearner, "train_batch", cold_train)
    monkeypatch.setattr(
        context.ContextLinearFitLearner, "train_batch_minimum_change", private_train
    )
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        result = live.run_case(
            None,
            _linear_fit=True,
            _owned_source=(qualified_literal(), lambda: None),
            _memory=memory,
            _retained_joint=True,
            _probe_retained=True,
            _probe_update="minimum_change",
            stream_recipe=WITHHELD_CONTEXT_RECIPES[0],
        )
    current = result.get("isolated_transfer_probe", {})
    arm = current.get("arm", {})
    endpoints = []
    for phase in ("before_probe", "after_probe", "after_return"):
        episodes = [r["episode"] for r in arm.get("episodes", []) if r["phase"] == phase]
        endpoints.append(
            dict(
                phase=phase,
                slots=len(episodes),
                old=sum(e["success"] for e in episodes[:4]),
                selected=sum(e["success"] for e in episodes[4:]),
            )
        )
    record_property(
        "minimum_change_fixture",
        dict(
            endpoints=endpoints,
            private_work=current.get("private_work"),
            sessions=arm.get("sessions"),
            source_unchanged=result.get("source_unchanged"),
            actor_unchanged=result.get("actor_unchanged"),
            outer_error=result.get("error"),
        ),
    )
    assert result["complete"], result.get("error")
    assert result["source_unchanged"] and result["actor_unchanged"]
    assert result["candidate_state"]["completed_updates"] == 23
    assert result["candidate_state"]["training_examples"] == 144
    assert len(cold) == 11 and sum(cold) == 72
    assert private == [3] * 16 + [6] * 16
    assert current["donor_unchanged"] and current["retained_scope_verified"]
    assert current["linear_probe_update"] == arm["linear_probe_update"] == "minimum_change"
    assert current["private_work"] == dict(
        attempted_updates=32,
        completed_updates=32,
        attempted_training_examples=144,
        completed_training_examples=144,
        unreturned_work_unknown=False,
    )
    assert [(r["native_updates"], r["native_rows"]) for r in arm["readout_work"]] == [
        (23, 144),
        (39, 192),
        (55, 288),
    ]
    assert all(
        s["updates"] == 16 and s["limits"] == recipe.asdict(recipe.HEAD_LIMITS)
        for s in arm["sessions"]
    )
    assert all(
        e["role"] == "probe_test"
        and e["episode"]["complete"]
        and e["episode"]["training_updates"] == 0
        for e in arm["episodes"]
    )
    assert [e["slots"] for e in endpoints] == [
        6,
        6,
        6,
    ]  # endpoints observed, never tuned for success
    assert all(
        d["definition"] == context.CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID
        for phase in (arm["probe"], arm["probe"]["return_phase"])
        for d in phase["diagnostics"]
    )
    assert (
        result["live_serving"]["closed"]
        and result["final_examples"] == result["probe_updates"] == 0
    )
    assert linear._rng_signature() == rng
    damaged = deepcopy(arm)
    damaged["linear_probe_update"] = "cold"
    with pytest.raises(ValueError, match="intent"):
        recipe._require_arm(
            damaged,
            separate_context_heads=True,
            linear_fit=True,
            head_tasks=recipe.TRANSFER_HEAD_TASKS,
            linear_probe_update="minimum_change",
        )


def test_should_keep_lost_fit_paid_and_stop_private_probe_without_more_work(monkeypatch):
    runtime, helper, report, record = scope_fixture()
    before = runtime.candidate_snapshot()
    calls = []
    original = linear.update_categorical_linear

    def lose(*args):
        calls.append(1)
        original(*args)
        raise RuntimeError("lost minimum-change probe fit")

    monkeypatch.setattr(linear, "update_categorical_linear", lose)
    with pytest.raises(ValueError, match="paid session"):
        composed.run_probe(
            runtime,
            spec(),
            helper,
            lambda: None,
            None,
            report,
            record,
            probe_update="minimum_change",
        )
    assert calls == [1]
    assert record["arm"]["probe"]["status"] == "failed"
    assert record["arm"]["probe"]["attempted_updates"] == 1
    assert record["arm"]["probe"]["completed_updates"] == 0
    assert runtime.candidate_snapshot() == before and record["donor_unchanged"]
    assert record["retained_scope_verified"] is None
