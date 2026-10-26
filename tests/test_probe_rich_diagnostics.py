"""Scalar fake ports exercise rich probe receipts without native model work."""

from dataclasses import asdict
import pickle

import pytest

from src.core.categorical_pc_inference import CategoricalInferenceDiagnostics
from src.core.categorical_training_diagnostic import (
    BOUNDED_OBJECTIVE_ID,
    CategoricalPCTrainingDiagnostic,
    ScheduledCategoricalUpdateFailed,
)
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from test_fresh_task_probe import FakeLearner, run, session


def rich_diagnostic():
    inference = CategoricalInferenceDiagnostics(
        reason="max_steps",
        initial_energy=2.0,
        final_energy=1.0,
        final_scaled_gradient_inf_norm=0.5,
        rows=1,
        attempted_steps=2,
        accepted_steps=2,
        rejected_steps=0,
        attempted_latent_row_iterations=2,
        accepted_latent_row_iterations=2,
        energy_evaluations_attempted=3,
        energy_evaluations_completed=3,
        latent_gradient_evaluations_attempted=3,
        latent_gradient_evaluations_completed=3,
        gradient_norm_evaluations=3,
        energy_comparisons=2,
        parameter_gradient_evaluations_attempted=1,
        parameter_gradient_evaluations_completed=1,
        parameter_proposals_attempted=1,
        parameter_proposals_completed=1,
        final_energy_evaluations_attempted=1,
        final_energy_evaluations_completed=1,
        legacy_diagnostic_evaluations_attempted=1,
        legacy_diagnostic_evaluations_completed=1,
        initial_prior_evaluations_attempted=1,
        initial_prior_evaluations_completed=1,
        used_fallback=False,
        failure_work_may_be_unobserved=False,
    )
    native = dict(
        **inference.work,
        feature_projection_attempts=1,
        native_state_restores_attempted=1,
        staging_context_evaluations_attempted=1,
        native_transitions_attempted=1,
        native_transitions_completed=1,
        candidate_validations_attempted=1,
        candidate_validations_completed=1,
        published_updates=1,
    )
    stream = dict(
        candidate_forks_attempted=1,
        candidate_forks_completed=1,
        publication_restores_attempted=1,
        publication_restores_completed=1,
        publication_validations_attempted=1,
        publication_validations_completed=1,
        published_updates=1,
    )
    return CategoricalPCTrainingDiagnostic(
        BOUNDED_OBJECTIVE_ID,
        1.0,
        CoupledInferencePolicy(min_steps=1, gradient_tolerance=None),
        inference,
        tuple(native.items()),
        tuple(stream.items()),
    )


def scheduled_failure():
    receipt = rich_diagnostic()
    native = receipt.work
    native.update(
        candidate_validations_completed=0,
        published_updates=0,
        failure_work_may_be_unobserved=True,
    )
    scheduled = dict(
        candidate_forks_attempted=1,
        candidate_forks_completed=1,
        publication_restores_attempted=1,
        publication_restores_completed=0,
        publication_validations_attempted=0,
        publication_validations_completed=0,
        published_updates=0,
    )
    return ScheduledCategoricalUpdateFailed(
        "injected staged failure", BOUNDED_OBJECTIVE_ID, native, scheduled
    )


def test_should_keep_legacy_diagnostic_and_failure_wire_formats_exact():
    legacy = TrainingDiagnostic("engineering_fake", 0.25)
    report = run(FakeLearner(hooks={"train": lambda *args: legacy}))
    expected = [{"definition": "engineering_fake", "value": 0.25} for _ in range(2)]
    assert report["status"] == "complete"
    assert pickle.dumps(report["diagnostics"]) == pickle.dumps(expected)
    error = TrainingUpdateFailed("legacy failure", "legacy", {"steps": 1})

    def fail(*args):
        raise error

    failed = run(FakeLearner(hooks={"train": fail}))
    assert failed["failed_native_work"] == {"definition": "legacy", "work": {"steps": 1}}
    assert set(failed["failed_native_work"]) == {"definition", "work"}


def test_should_publish_all_rich_fields_as_owned_revalidated_scalar_records():
    receipt = rich_diagnostic()
    expected = asdict(receipt)
    source = FakeLearner(hooks={"train": lambda *args: receipt})
    budget = session()
    report = run(source, budget=budget)
    assert report["status"] == "complete", report["reason"]
    assert report["completed_updates"] == budget.updates_completed == 2
    assert report["diagnostics"] == [expected, expected]
    assert set(report["diagnostics"][0]) == {
        "definition",
        "value",
        "inference_policy",
        "inference",
        "native_work",
        "stream_work",
    }
    before = pickle.dumps(report["diagnostics"])
    object.__setattr__(receipt.inference, "final_energy", 0.0)
    object.__setattr__(receipt.inference_policy, "min_steps", 64)
    object.__setattr__(receipt, "native_work", ())
    assert pickle.dumps(report["diagnostics"]) == before
    assert source.value[0] == 99.0


@pytest.mark.parametrize("kind", ("base", "rich"))
def test_should_refuse_foreign_diagnostic_subclasses_without_erasing_paid_commit(kind):
    class ForeignBase(TrainingDiagnostic):
        pass

    class ForeignRich(CategoricalPCTrainingDiagnostic):
        pass

    original = rich_diagnostic()
    receipt = (
        ForeignBase("foreign", 1.0)
        if kind == "base"
        else ForeignRich(
            original.definition,
            original.value,
            original.inference_policy,
            original.inference,
            original.native_work,
            original.stream_work,
        )
    )
    budget = session()
    report = run(FakeLearner(hooks={"train": lambda *args: receipt}), budget=budget)
    assert report["status"] == "failed"
    assert (
        report["completed_updates"] == report["attempted_updates"] == budget.updates_completed == 1
    )
    assert report["diagnostics"] == [] and "failed_native_work" not in report
    assert [point["updates"] for point in report["checkpoints"]] == [0]


@pytest.mark.parametrize(
    "fault", ("extra", "missing", "value", "policy", "inference", "native", "stream")
)
def test_should_refuse_corrupted_rich_receipt_after_counting_actual_native_return(fault):
    receipt = rich_diagnostic()
    if fault == "extra":
        object.__setattr__(receipt, "unexpected", 0)
    elif fault == "missing":
        object.__delattr__(receipt, "native_work")
    elif fault == "value":
        object.__setattr__(receipt, "value", float("nan"))
    elif fault == "policy":
        object.__setattr__(receipt.inference_policy, "min_steps", True)
    elif fault == "inference":
        object.__setattr__(receipt.inference, "unexpected", 0)
    elif fault == "native":
        object.__setattr__(receipt, "native_work", (*receipt.native_work, ("unexpected", 1)))
    else:
        object.__setattr__(receipt, "stream_work", (*receipt.stream_work, ("unexpected", 1)))
    budget = session()
    report = run(FakeLearner(hooks={"train": lambda *args: receipt}), budget=budget)
    assert report["status"] == "failed"
    assert report["completed_updates"] == budget.updates_completed == 1
    assert report["diagnostics"] == []


def test_should_preserve_complete_native_and_scheduled_partial_failure_costs_without_commit():
    error = scheduled_failure()
    expected = {
        "definition": error.definition,
        "work": error.work,
        "scheduling_work": error.scheduling_work,
    }

    def fail(*args):
        raise error

    budget = session()
    report = run(FakeLearner(hooks={"train": fail}), budget=budget)
    assert report["status"] == "failed"
    assert report["attempted_updates"] == report["attempted_training_examples"] == 1
    assert report["completed_updates"] == budget.updates_completed == 0
    assert report["failed_native_work"] == expected
    assert len(report["failed_native_work"]["work"]) == 32
    assert len(report["failed_native_work"]["scheduling_work"]) == 7
    assert report["failed_native_work"]["work"]["failure_work_may_be_unobserved"] is True
    assert report["diagnostics"] == []
    assert [point["updates"] for point in report["checkpoints"]] == [0]
    error._work_items = ()
    error._scheduling_work = ()
    assert report["failed_native_work"] == expected


@pytest.mark.parametrize("after_native", (False, True))
def test_should_not_accept_scheduled_failure_charges_from_external_boundary(after_native):
    source = FakeLearner()
    error = scheduled_failure()

    def boundary():
        trained = any(event[0] == "train" for event in source.trace)
        if trained or not after_native:
            raise error

    budget = session()
    report = run(source, budget=budget, check_boundary=boundary)
    assert report["status"] == "failed"
    assert "failed_native_work" not in report
    assert report["completed_updates"] == budget.updates_completed == int(after_native)
    assert len(report["diagnostics"]) == int(after_native)


def test_should_not_admit_scheduling_namespace_from_foreign_exception_subclass():
    class ForeignFailure(ScheduledCategoricalUpdateFailed):
        pass

    original = scheduled_failure()
    error = ForeignFailure(
        str(original), original.definition, original.work, original.scheduling_work
    )

    def fail(*args):
        raise error

    report = run(FakeLearner(hooks={"train": fail}))
    assert report["status"] == "failed"
    assert report["failed_native_work"] == {"definition": error.definition, "work": error.work}


@pytest.mark.parametrize("fault", ("native", "stream"))
def test_should_not_publish_corrupted_scheduled_failure_namespaces(fault):
    error = scheduled_failure()
    if fault == "native":
        error._work_items = (*error._work_items[:-1], ("published_updates", -1))
    else:
        error._scheduling_work = (("published_updates", 0),)

    def fail(*args):
        raise error

    report = run(FakeLearner(hooks={"train": fail}))
    assert report["status"] == "failed" and "failed_native_work" not in report
    assert report["attempted_updates"] == 1 and report["completed_updates"] == 0


def test_should_preserve_partial_native_failure_without_inventing_unobserved_counters():
    original = scheduled_failure()
    error = ScheduledCategoricalUpdateFailed(
        "lost native receipt",
        original.definition,
        {"failure_work_may_be_unobserved": True},
        original.scheduling_work,
    )

    def fail(*args):
        raise error

    budget = session()
    report = run(FakeLearner(hooks={"train": fail}), budget=budget)
    assert report["status"] == "failed"
    assert report["failed_native_work"] == {
        "definition": error.definition,
        "work": {"failure_work_may_be_unobserved": True},
        "scheduling_work": error.scheduling_work,
    }
    assert report["attempted_updates"] == 1
    assert report["completed_updates"] == budget.updates_completed == 0
