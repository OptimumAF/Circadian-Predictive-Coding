"""TRAIN-only auxiliary CE initialization at the ordinary probe handoff.

Uses the existing source gate, fixed fitter and detached initializer. Source
budget/receipts/actor authority stay original; no source update, IO or promotion.
"""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic

from scripts import ticket_workflow_native as native
from scripts.ticket_native_probe import _require_probe_source, run_owned_probe
from src.adapters.torch_categorical_ce_readout import (
    CategoricalCEFitFailed,
    fit_categorical_ce_readout,
)
from src.adapters.torch_categorical_ce_initialization import (
    ce_initialization_record,
    initialize_categorical_ce_probe,
)
from src.adapters.torch_categorical_readout_fit import _unchanged_owner
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.probe_return_phase import paid_report_guard

LIMITS = ToyExecutionBudget(max_wall_seconds=15, max_process_rss_bytes=512 * 1024**2)


def run_initialized_probe(
    stream,
    state,
    spec,
    source_report,
    source_budget,
    *,
    check_boundary,
    memory,
    row,
    initialization_row,
    probe_update_diagnostics=False,
    probe_output_only=False,
    ce_rehearsal=False,
):
    if (
        type(ce_rehearsal) is not bool
        or ce_rehearsal
        and (probe_output_only is not False or probe_update_diagnostics is not False)
    ):
        raise ValueError("CE rehearsal cannot use native SGD diagnostics/output-only")
    if type(probe_update_diagnostics) is not bool:
        raise ValueError("probe update diagnostics require an exact bool")
    if type(probe_output_only) is not bool:
        raise ValueError("output-only probe requires an exact bool")
    initialization, prepared_guard = prepare_ce_initialization(
        stream,
        state,
        spec,
        source_report,
        source_budget,
        check_boundary=check_boundary,
        memory=memory,
        row=initialization_row,
    )
    paid_guard, _ = paid_report_guard(initialization_row)

    def ready():
        prepared_guard()
        paid_guard()

    def boundary():
        ready()
        check_boundary()
        ready()

    boundary()
    result = run_owned_probe(
        state,
        spec,
        source_report,
        source_budget,
        check_boundary=boundary,
        check_initialization=ready,
        memory=memory,
        row=row,
        ce_initialization=initialization,
        **(dict(probe_update_diagnostics=True) if probe_update_diagnostics else {}),
        **(dict(probe_output_only=True) if probe_output_only else {}),
        **(dict(ce_rehearsal=True) if ce_rehearsal else {}),
    )
    ready()
    return result


def _receipt_guard(stream):
    """Public receipts are copies; actual inbox objects retain provenance."""
    runtime, inbox = stream._runtime, stream._runtime._inbox
    ledger = inbox._applied
    pins = tuple((key, value, deepcopy(value)) for key, value in ledger.items())
    expected_view = tuple(value for _, _, value in pins)

    def owned():
        if (
            stream._runtime is not runtime
            or runtime._inbox is not inbox
            or inbox._applied is not ledger
            or len(ledger) != len(pins)
            or any(
                k != key or v is not owner or not native.same_state(v, value)
                for (k, v), (key, owner, value) in zip(ledger.items(), pins, strict=True)
            )
        ):
            raise ValueError("CE probe authoritative receipt ledger changed")

    def require():
        owned()
        view = stream.applied_updates
        owned()
        if not native.same_state(view, expected_view):
            raise ValueError("CE probe detached receipt view changed")

    require()
    return require


def prepare_ce_initialization(
    stream, state, spec, source_report, source_budget, *, check_boundary, memory, row
):
    """One paid auxiliary fit; return only a complete source-bound initialization."""
    if type(row) is not dict or row:
        raise ValueError("CE initialization requires a fresh report; paid work cannot be renewed")
    _require_probe_source(
        state, spec, source_report, source_budget, check_boundary=check_boundary, memory=memory
    )
    learner = stream._runtime._candidate
    if stream._budget is not source_budget:
        raise ValueError("CE initialization requires the original source budget")
    with _unchanged_owner(learner) as (actual, owner_unchanged):
        if not native.same_state(actual, state):
            raise ValueError("CE initialization checkpoint differs from actual source")
        receipt_unchanged = _receipt_guard(stream)
        original = (source_budget.budget, source_budget.clock, source_budget.progress, memory)
        limits, birth = deepcopy(source_budget.budget), source_budget.started_at
        checkpoint_pin, spec_pin = deepcopy(state), deepcopy(spec)
        runtime = stream._runtime
        source_training_owner = source_report["reserved_probe_source_training"]
        source_tag = deepcopy(source_training_owner)
        aux = ToyBudgetSession(LIMITS, monotonic)
        aux.attach_memory(memory)
        aux_binding = (aux.budget, aux.clock, aux.progress, aux.process_rss_sampler)
        aux_limits, aux_birth = deepcopy(aux.budget), aux.started_at
        row.update(
            definition="ordinary_ce_probe_initialization_v1",
            complete=False,
            source_updates=state.completed_updates,
            source_rows=state.training_examples,
            final_examples=0,
            work={},
        )
        work = row["work"]
        work_pin = deepcopy(work)
        fit = fit_pin = fit_binding = None
        initialization = initialization_pin = initialization_binding = None

        def charge(key):
            nonlocal work_pin
            work[key] = 1
            work_pin = deepcopy(work)

        def unchanged():
            owner_unchanged()
            receipt_unchanged()
            if (
                stream._runtime is not runtime
                or source_report.get("probe_initialization") is not row
                or row.get("work") is not work
                or work != work_pin
                or runtime._candidate is not learner
                or stream._budget is not source_budget
                or source_budget.started_at != birth
                or source_report.get("reserved_probe_source_training") is not source_training_owner
                or not native.same_state(source_training_owner, source_tag)
                or source_budget.budget != limits
                or source_budget.updates_completed != state.completed_updates
                or source_budget.replay_examples_completed != 0
                or any(
                    a is not b
                    for a, b in zip(
                        (
                            source_budget.budget,
                            source_budget.clock,
                            source_budget.progress,
                            source_budget.process_rss_sampler,
                        ),
                        original,
                        strict=True,
                    )
                )
                or aux.started_at != aux_birth
                or aux.budget != aux_limits
                or aux.updates_completed != 0
                or aux.replay_examples_completed != 0
                or any(
                    a is not b
                    for a, b in zip(
                        (aux.budget, aux.clock, aux.progress, aux.process_rss_sampler),
                        aux_binding,
                        strict=True,
                    )
                )
                or not native.same_state(state, checkpoint_pin)
                or spec != spec_pin
                or fit is not None
                and (
                    not native.same_state(fit, fit_pin)
                    or any(
                        a is not b
                        for a, b in zip(
                            (fit.source, fit.spec, fit.coefficients, fit.work),
                            fit_binding,
                            strict=True,
                        )
                    )
                )
                or initialization is not None
                and (
                    not native.same_state(initialization, initialization_pin)
                    or any(
                        a is not b
                        for a, b in zip(
                            (
                                initialization.fit,
                                initialization.state,
                                initialization.fit.source,
                                initialization.fit.spec,
                                initialization.fit.coefficients,
                                initialization.fit.work,
                            ),
                            initialization_binding,
                            strict=True,
                        )
                    )
                )
            ):
                raise ValueError("CE initialization source/budget/fit changed")

        def boundary():
            unchanged()
            check_boundary()
            unchanged()
            source_budget.before_final()
            unchanged()
            aux.before_final()
            unchanged()

        try:
            boundary()
            cases, _ = native.prepare_train_fit(spec)
            boundary()
            charge("fit_attempts")
            try:
                fit = fit_categorical_ce_readout(learner, cases, spec, check_boundary=boundary)
            except CategoricalCEFitFailed as error:
                row["fit_work"] = deepcopy(error.work)
                raise
            fit_binding = (fit.source, fit.spec, fit.coefficients, fit.work)
            fit_pin = deepcopy(fit)
            row.update(fit_work=deepcopy(fit.work), training_logits=fit.training_logits)
            charge("fit_returns")
            boundary()
            charge("initialization_attempts")
            initialization = initialize_categorical_ce_probe(learner, fit, check_boundary=boundary)
            initialization_binding = (
                initialization.fit,
                initialization.state,
                initialization.fit.source,
                initialization.fit.spec,
                initialization.fit.coefficients,
                initialization.fit.work,
            )
            initialization_pin = deepcopy(initialization)
            row["initialization"] = ce_initialization_record(initialization)
            boundary()
            charge("initialization_returns")
            row.update(complete=True, source_unchanged=True)
            # Why: the same authority must survive callbacks in the private phase.
            return initialization, unchanged
        except Exception as error:
            row.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])
            raise
        finally:
            row["preserved_work"] = deepcopy(work_pin)
            if fit_pin is not None:
                row["preserved_fit_work"] = deepcopy(fit_pin.work)
            row["session"] = dict(
                limits=asdict(aux_limits),
                started_at=aux_birth,
                last_clock=aux.last_clock,
                updates=aux.updates_completed,
            )
