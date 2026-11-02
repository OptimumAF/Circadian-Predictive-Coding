"""Direct checkpoint12 LS boundary: scripted history, no scientific trial."""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_native_probe as probe
from scripts.ticket_linear_probe import MinimumChangeProbeLearner
from src.app.reserved_probe_summary import summarize_reserved_probe
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.categorical_linear_state import LinearFitPolicy, LinearFitSnapshot
from src.core.context_linear_state import ContextLinearFitPolicy, ContextLinearFitSnapshot
from src.core.decision_features import DecisionFeatureSpec
from src.core.learner_ports import TrainingDiagnostic
from src.adapters.context_linear_fit_learner import CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID
from src.shared.process_memory import ProcessRssSampler


OPTIONS = dict(
    method="linear_fit",
    context_return=True,
    separate_context_heads=True,
    reserved_probe=True,
    probe_update="minimum_change",
)


def source_fixture():
    """Literal coefficients/counters, not a trained12 source or RSS qualification."""
    model = CategoricalLinearFit(7, 6, 6, ((0.0,) * 6,) * 8, 0, (0.0,) * 6, 6.0)
    head = LinearFitSnapshot(LinearFitPolicy(7, 6), model, 6, 36)
    state = ContextLinearFitSnapshot(
        ContextLinearFitPolicy(9, 6, (4, 5)), (head, deepcopy(head)), 12, 72
    )
    spec = DecisionFeatureSpec(probe.CONTEXT_FEATURES, probe.CATALOG)
    report = dict(
        definition="separate_context_linear_ticket_workflow_v1",
        method="linear_fit",
        separate_context_heads=True,
        final_examples=0,
        reserved_probe_requested=True,
        reserved_probe_update="minimum_change",
        reserved_probe_source_training=dict(completed_updates=12, applied_update_count=12),
        checkpoints=[dict(state=asdict(state))],
    )
    memory = ProcessRssSampler(read_rss_bytes=lambda: 1)
    memory.sample()
    clock = [0.0]
    budget = ToyBudgetSession(probe.LIMITS, lambda: clock[0])
    budget.attach_memory(memory)
    budget.updates_completed = 12
    return state, spec, report, budget, memory, clock


def invoke(source, **options):
    state, spec, report, budget, memory, _ = source
    return probe.run_owned_probe(
        state,
        spec,
        report,
        budget,
        memory=memory,
        check_boundary=lambda: None,
        linear_probe_update="minimum_change",
        **options,
    )


def scripted_reads(learner, phase, spec, require, row, **kwargs):
    require()
    row["episodes"][phase] = [
        dict(
            phase=phase,
            policy_version=v,
            concept=c,
            role="controller_train",
            complete=True,
            episode=dict(complete=True, success=False, scenario="ordinary", training_updates=0),
        )
        for v, c in probe.PAIRED_TASKS
    ]
    row["task_readout_work"][phase] = dict(decision_attempts=0, decision_returns=0)


@pytest.mark.parametrize("train_update", [None, "minimum_change"])
def test_direct_api_reaches_data_admission_without_serving(train_update, monkeypatch):
    def stop(*args, **kwargs):
        raise RuntimeError("data boundary reached")

    monkeypatch.setattr(entry, "prepare_training", stop)
    result = entry.run_example(**OPTIONS, train_update=train_update)
    assert result["complete"] is False and "data boundary reached" in result["error"]


@pytest.mark.parametrize(
    "change",
    [
        dict(probe_update=None),
        dict(probe_update=True),
        dict(probe_update="cold"),
        dict(paired_serving="retained_joint_probe"),
        dict(private_adaptation=True),
        dict(replay="shared"),
        dict(probe_rehearsal="joint"),
        dict(record_train_fit=True),
        dict(probe_initialization="ce"),
        dict(policy_change=True),
        dict(context_return=False),
    ],
)
def test_direct_api_refuses_foreign_compositions_before_data(change, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("data"))
    with pytest.raises(ValueError):
        entry.run_example(**(OPTIONS | change))


@pytest.mark.parametrize("train_update", [False, True])
def test_direct_cli_forwards_explicit_intent_and_never_overwrites(
    train_update, tmp_path, monkeypatch
):
    calls = []

    def record(**kw):
        calls.append(kw)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", record)
    flags = [
        "--method",
        "linear_fit",
        "--context-return",
        "--separate-context-heads",
        "--reserved-probe",
        "--probe-update",
        "minimum_change",
        "--output",
        str(tmp_path / "out.json"),
    ]
    if train_update:
        flags += ["--train-update", "minimum_change"]
    assert entry.main(flags) == 0
    assert calls == [OPTIONS | (dict(train_update="minimum_change") if train_update else {})]
    with pytest.raises(FileExistsError):
        entry.main(flags)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "extra",
    [
        ["--private-adaptation"],
        ["--paired-serving", "joint"],
        ["--policy-change"],
        ["--probe-rehearsal", "joint"],
        ["--probe-initialization", "ce"],
    ],
)
def test_direct_cli_refuses_foreign_modes_before_dispatch(extra, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "run_example", lambda **kw: pytest.fail("dispatch"))
    with pytest.raises(SystemExit):
        entry.main(
            [
                "--method",
                "linear_fit",
                "--context-return",
                "--separate-context-heads",
                "--reserved-probe",
                "--probe-update",
                "minimum_change",
                "--output",
                str(tmp_path / "out.json"),
                *extra,
            ]
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("method", "bp"),
        ("reserved_probe_requested", False),
        ("reserved_probe_update", "cold"),
        ("separate_context_heads", False),
        ("final_examples", 1),
        ("final_examples", False),
        ("linear_train_update", "cold"),
        ("definition", "retained_context_linear_transfer_probe_v1"),
    ],
)
def test_direct_source_requires_exact_checkpoint12_authority(key, value, monkeypatch):
    source = source_fixture()
    source[2][key] = value
    monkeypatch.setattr(probe, "build_head_cohorts", lambda *a, **kw: pytest.fail("data"))
    with pytest.raises(ValueError, match="checkpoint12"):
        invoke(source)


@pytest.mark.parametrize("kind", ["checkpoint", "budget", "intent", "clock", "paid"])
def test_source_or_paid_evidence_changes_refuse_work_without_restoring_it(kind, monkeypatch):
    source = source_fixture()
    row: dict[str, Any] = {}

    def data(*args, **kwargs):
        if kind == "checkpoint":
            source[2]["checkpoints"][-1] = deepcopy(source[2]["checkpoints"][-1])
        elif kind == "budget":
            source[3].budget = replace(source[3].budget)
        elif kind == "intent":
            source[2]["reserved_probe_update"] = "cold"
        elif kind == "clock":
            source[-1][0] = 31.0
        else:
            row["training_work"]["attempted_updates"] = 99
        return ((),) * 4

    monkeypatch.setattr(probe, "build_head_cohorts", data)
    result = invoke(source, row=row)
    assert result["complete"] is False
    assert result["preserved_training_work"] == dict(
        attempted_updates=0, attempted_rows=0, returned_updates=0
    )
    assert result["sessions"] == []
    if kind == "intent":
        assert source[2]["reserved_probe_update"] == "cold"  # No repair of foreign authority.


def test_scripted_complete_sequence_counts_only_private_updates_and_preserves_source(monkeypatch):
    source = source_fixture()
    pin = deepcopy(source[0])
    monkeypatch.setattr(probe, "build_head_cohorts", lambda *a, **kw: ((),) * 4)
    monkeypatch.setattr(probe, "_read_tasks", scripted_reads)
    monkeypatch.setattr(
        MinimumChangeProbeLearner,
        "train_batch",
        lambda *a: TrainingDiagnostic(CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID, 0.0),
    )
    owners: list[Any] = []

    def fake(seed, state, training, evaluation, **kw):
        child = seed.fork()
        owners.extend((seed, child))
        assert child is not seed and child.snapshot_state() == state
        for budget, rows, phase in (
            (kw["budget"], 3, "after_probe"),
            (kw["return_budget"], 6, "after_return"),
        ):
            for _ in range(16):
                child.train_batch(np.zeros((rows, 9)), np.zeros((rows, 1)))
                budget.updates_completed += 1  # Scripted chronology, not native training.
            kw["endpoint_observer"](child, phase, lambda *a, **kw: None)
        return dict(status="complete", return_phase=dict(status="complete"))

    monkeypatch.setattr(probe, "run_fresh_task_probe", fake)
    result = invoke(source)
    assert result["complete"] is True, result.get("error")
    assert result["preserved_training_work"] == dict(
        attempted_updates=32, attempted_rows=144, returned_updates=32
    )
    assert [s["updates"] for s in result["sessions"]] == [16, 16]
    assert source[0] == pin and source[3].updates_completed == 12
    assert result["source_train_update"] is None
    assert all(type(owner).__name__ == "LinearProbeLearner" for owner in owners)
    source[2]["reserved_probe"] = result
    summary = summarize_reserved_probe(source[2])
    assert summary["status"] == "complete"
    assert summary["cohorts"]["fresh"]["successes_by_phase"]["after_probe"] == 0
    assert (
        summary["reported_costs"]["task_readouts"]["after_probe"]["native_forward_attempts"] is None
    )


@pytest.mark.parametrize("throw", [False, True])
def test_one_real_tiny_private_update_or_failure_keeps_charges_and_source_detached(
    throw, monkeypatch
):
    source = source_fixture()
    monkeypatch.setattr(probe, "build_head_cohorts", lambda *a, **kw: ((),) * 4)
    monkeypatch.setattr(probe, "_read_tasks", scripted_reads)
    if throw:

        def fail(*args):
            raise RuntimeError("failed fit")

        monkeypatch.setattr(MinimumChangeProbeLearner, "train_batch", fail)
    observed = []

    def partial(seed, state, training, evaluation, **kw):
        child = seed.fork()
        x: np.ndarray = np.zeros((3, 9), dtype=np.float64)
        x[:, 4] = 1.0
        x[:, :3] = np.eye(3)
        child.train_batch(x, np.array([[0.0], [1.0], [2.0]]))
        observed.append(child.snapshot_state())
        assert seed.snapshot_state() == state
        return dict(status="incomplete")

    monkeypatch.setattr(probe, "run_fresh_task_probe", partial)
    result = invoke(source)
    assert result["complete"] is False
    assert result["preserved_training_work"] == dict(
        attempted_updates=1, attempted_rows=3, returned_updates=0 if throw else 1
    )
    assert source[3].updates_completed == 12 and source[0].completed_updates == 12
    if not throw:
        assert observed[0].completed_updates == 13
        assert observed[0].heads[1] == source[0].heads[1]
        assert observed[0].heads[0] != source[0].heads[0]


@pytest.mark.parametrize("complete", [False, True])
def test_paid_private_report_is_never_admitted_again(complete):
    row = dict(complete=complete, preserved_training_work=dict(attempted_updates=1))
    pin = deepcopy(row)
    with pytest.raises(ValueError, match="paid work cannot be renewed"):
        invoke(source_fixture(), row=row)
    assert row == pin


def test_actual_frozen_ls_readout_uses_fixed_roles_and_distinct_routing_costs():
    source = source_fixture()
    seed = MinimumChangeProbeLearner(source[0].policy)
    seed.restore_state(source[0])
    row: dict[str, Any] = {}
    probe._read_tasks(seed, "before_probe", source[1], lambda: None, row, linear=True)
    assert len(row["episodes"]["before_probe"]) == 6
    assert all(r["role"] == "controller_train" for r in row["episodes"]["before_probe"])
    work = row["task_readout_work"]["before_probe"]
    assert work["decision_attempts"] > 0 and work["routing_attempts"] > 0
    assert "native_forward_attempts" not in work
    assert seed.snapshot_state() == source[0]


@pytest.mark.parametrize("kind", ["policy", "retained23", "zero"])
def test_foreign_geometry_or_donor_history_is_refused_before_data(kind, monkeypatch):
    source = list(source_fixture())
    state = source[0]
    if kind == "policy":
        state = replace(state, policy=ContextLinearFitPolicy(9, 6, (7, 8)))
    elif kind == "retained23":
        heads = tuple(
            replace(h, completed_updates=n, training_examples=72)
            for h, n in zip(state.heads, (11, 12))
        )
        state = replace(state, heads=heads, completed_updates=23, training_examples=144)
    else:
        state = MinimumChangeProbeLearner(state.policy).snapshot_state()
    source[0] = state
    monkeypatch.setattr(probe, "build_head_cohorts", lambda *a, **kw: pytest.fail("data"))
    with pytest.raises(ValueError, match="checkpoint12"):
        invoke(source)


@pytest.mark.parametrize("kind", ["replace", "corrupt", "work"])
def test_preceding_readout_change_refuses_next_update_and_preserves_original_prefix(
    kind, monkeypatch
):
    source = source_fixture()
    row: dict[str, Any] = {}
    monkeypatch.setattr(probe, "build_head_cohorts", lambda *a, **kw: ((),) * 4)
    monkeypatch.setattr(probe, "_read_tasks", scripted_reads)
    calls = []
    monkeypatch.setattr(MinimumChangeProbeLearner, "train_batch", lambda *a: calls.append(1))

    def corrupt(seed, *args, **kw):
        if kind == "replace":
            row["episodes"]["before_probe"] = deepcopy(row["episodes"]["before_probe"])
        elif kind == "corrupt":
            row["episodes"]["before_probe"][0]["episode"]["success"] = True
        else:
            row["task_readout_work"]["before_probe"]["decision_attempts"] = 999
        seed.fork().train_batch(np.zeros((3, 9)), np.zeros((3, 1)))
        pytest.fail("further training admitted")

    monkeypatch.setattr(probe, "run_fresh_task_probe", corrupt)
    result = invoke(source, row=row)
    assert not result["complete"] and "preceding linear reserved readout" in result["error"]
    assert calls == []
    assert result["preserved_training_work"]["attempted_updates"] == 0
    prefix = result["preserved_readouts"]["before_probe"]
    assert prefix["episodes"][0]["episode"]["success"] is False
    assert prefix["work"]["decision_attempts"] == 0


@pytest.mark.parametrize("train_update", [None, "minimum_change"])
def test_ordinary_runtime_handoff_with_scripted_fits_and_episodes_preserves_source_authority(
    train_update, monkeypatch
):
    """Actual stream/guards/receipts, scripted fits/episodes; no learning evidence."""
    from src.adapters.context_linear_fit_learner import ContextLinearFitLearner
    from src.adapters.context_linear_fit_learner import CONTEXT_LINEAR_RESIDUAL_ID

    def scripted_source(self, features, targets):
        prior = self.snapshot_state()
        index = 0 if features[0, 4] == 1.0 else 1
        heads = list(prior.heads)
        head = heads[index]
        model = CategoricalLinearFit(
            7, 6, len(features), ((0.0,) * 6,) * 8, 0, (0.0,) * min(len(features), 8), 0.0
        )
        heads[index] = replace(
            head,
            model=model,
            completed_updates=head.completed_updates + 1,
            training_examples=head.training_examples + len(features),
        )
        self._state = replace(
            prior,
            heads=tuple(heads),
            completed_updates=prior.completed_updates + 1,
            training_examples=prior.training_examples + len(features),
        )
        return TrainingDiagnostic(
            CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID if train_update else CONTEXT_LINEAR_RESIDUAL_ID, 0.0
        )

    monkeypatch.setattr(ContextLinearFitLearner, "train_batch", scripted_source)
    monkeypatch.setattr(ContextLinearFitLearner, "train_batch_minimum_change", scripted_source)
    monkeypatch.setattr(
        MinimumChangeProbeLearner,
        "train_batch",
        lambda *a: TrainingDiagnostic(CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID, 0.0),
    )
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
    monkeypatch.setattr(probe, "_read_tasks", scripted_reads)

    def scripted_probe(seed, state, training, evaluation, **kw):
        for budget, rows, phase in (
            (kw["budget"], 3, "after_probe"),
            (kw["return_budget"], 6, "after_return"),
        ):
            for _ in range(16):
                seed.fork().train_batch(np.zeros((rows, 9)), np.zeros((rows, 1)))
                budget.updates_completed += 1
            kw["endpoint_observer"](seed, phase, lambda *a, **kw: None)
        return dict(status="complete", return_phase=dict(status="complete"))

    monkeypatch.setattr(probe, "run_fresh_task_probe", scripted_probe)
    result = entry.run_example(**OPTIONS, train_update=train_update)
    assert result["complete"], result.get("error")
    assert (
        result["training_work"]["attempted_updates"] == result["budget"]["updates_completed"] == 12
    )
    assert result["training_work"]["attempted_rows"] == 72
    child = result["reserved_probe"]
    assert child["complete"] and child["source_updates"] == 12 and child["source_rows"] == 72
    assert child["source_train_update"] == train_update
    assert child["preserved_training_work"] == dict(
        attempted_updates=32, attempted_rows=144, returned_updates=32
    )
    assert result["source_unchanged"] and result["actor_unchanged"]
    assert result["numpy_rng_unchanged"] and result["python_rng_unchanged"]
    assert len(result["applied_updates"]) == 12
    assert result["reserved_probe_summary"]["status"] == "complete"
    assert "paired_serving" not in result and "torch_rng_unchanged" not in result


def test_existing_return_app_refuses_shortened_protocol_before_any_private_update(monkeypatch):
    """Keep the existing sixteen-update requirement; no reduced return recipe."""
    source = source_fixture()
    build = probe.build_head_cohorts

    def tiny(*args, **kwargs):
        training, evaluation, monitors, returning = build(*args, **kwargs)
        return training[:1], evaluation, monitors, returning

    monkeypatch.setattr(probe, "build_head_cohorts", tiny)
    monkeypatch.setattr(probe, "CHECKPOINTS", (0, 1))
    monkeypatch.setattr(probe, "_read_tasks", scripted_reads)
    result = invoke(source)
    assert not result["complete"] and "requires sixteen F updates" in result["error"]
    assert "probe" not in result
    assert result["preserved_training_work"] == dict(
        attempted_updates=0, attempted_rows=0, returned_updates=0
    )
    assert [s["updates"] for s in result["sessions"]] == [0, 0]
    assert source[0].completed_updates == source[3].updates_completed == 12
    assert set(result["preserved_readouts"]) == {"before_probe"}


def test_real_existing_fixed_probe_and_return_app_from_literal_source_fixture(monkeypatch):
    """ONE software execution: literal source history, realF16/return16 math, scripted task reads."""
    source = source_fixture()
    monkeypatch.setattr(probe, "_read_tasks", scripted_reads)
    result = invoke(source)
    assert result["complete"], result.get("error")
    assert result["probe"]["status"] == result["probe"]["return_phase"]["status"] == "complete"
    assert result["preserved_training_work"] == dict(
        attempted_updates=32, attempted_rows=144, returned_updates=32
    )
    assert [s["updates"] for s in result["sessions"]] == [16, 16]
    assert source[0].completed_updates == source[3].updates_completed == 12
    assert set(result["preserved_readouts"]) == {"before_probe", "after_probe", "after_return"}
