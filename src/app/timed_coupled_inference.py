"""Compare fixed policies within allocated wall-time training windows.

This runner owns copied native PC/data/policy state. Complete native updates are
never preempted or retried. All in-loop checks/bookkeeping count toward the
window; heldout labels enter only deferred application metrics. Failed cases
retain known committed work and expose no initial or final heldout scores.
"""

from dataclasses import asdict
from math import isfinite
from time import perf_counter
from typing import Any, Callable

from src.app.coupled_attribution import AttributionBatch, _probability_scores, _validated_batches
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork, CoupledTrainResult


def _positive(value: float, name: str) -> float:
    if type(value) not in (float, int):
        raise ValueError(f"{name} must be a native positive finite number")
    try:
        resolved = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return resolved


class _Clock:
    def __init__(self, callback: Callable[[], float]) -> None:
        self.callback = callback
        self.last: float | None = None
        self.broken = False
        self.started = self.read()

    def read(self) -> float:
        try:
            raw = self.callback()
            if type(raw) not in (float, int):
                raise ValueError("clock must return native finite seconds")
            try:
                now = float(raw)
            except OverflowError:
                raise ValueError("clock must return finite seconds") from None
            if not isfinite(now):
                raise ValueError("clock must return finite seconds")
            if self.last is not None and (now < self.last or not isfinite(now - self.started)):
                raise ValueError("clock moved backwards or produced a nonfinite interval")
            self.last = now
            return now
        except Exception:
            self.broken = True
            raise


def _clone(initial: CoupledPredictiveCodingNetwork) -> CoupledPredictiveCodingNetwork:
    if type(initial) is not CoupledPredictiveCodingNetwork:
        raise ValueError("initial must be an exact CoupledPredictiveCodingNetwork")
    state = initial.snapshot_state()
    model = CoupledPredictiveCodingNetwork(
        initial.input_dim, initial.hidden_dims, seed=0, precisions=initial.precisions
    )
    model.restore_state(state)
    return model


def _prepare(
    initial: CoupledPredictiveCodingNetwork,
    training: AttributionBatch,
    heldout: AttributionBatch,
    training_seconds: float,
    max_updates: int,
    learning_rate: float,
    inference_steps: int,
    inference_learning_rate: float,
    policy: CoupledInferencePolicy | None,
    budget: ToyBudgetSession,
    clock: Callable[[], float],
    max_iteration_seconds: float | None,
) -> tuple[
    CoupledPredictiveCodingNetwork,
    AttributionBatch,
    AttributionBatch,
    float,
    float,
    float,
    CoupledInferencePolicy | None,
    float | None,
]:
    window = _positive(training_seconds, "training_seconds")
    if window > 60:
        raise ValueError("training_seconds must not exceed the 60-second local ceiling")
    iteration_limit = None
    if max_iteration_seconds is not None:
        iteration_limit = _positive(max_iteration_seconds, "max_iteration_seconds")
        if iteration_limit > 0.1 * window:
            raise ValueError("max_iteration_seconds must not exceed 10% of training_seconds")
    rate = _positive(learning_rate, "learning_rate")
    inference_rate = _positive(inference_learning_rate, "inference_learning_rate")
    if type(max_updates) is not int or not 1 <= max_updates <= 512:
        raise ValueError("max_updates must be a native integer in 1..512")
    if type(inference_steps) is not int or not 1 <= inference_steps <= 10000:
        raise ValueError("inference_steps must be a native integer in 1..10000")
    if type(budget) is not ToyBudgetSession or not callable(clock):
        raise ValueError("runner requires an exact ToyBudgetSession and callable clock")
    if policy is not None:
        if type(policy) is not CoupledInferencePolicy:
            raise ValueError("policy must be an exact CoupledInferencePolicy or None")
        policy = CoupledInferencePolicy(
            policy.min_steps, policy.gradient_tolerance, policy.relative_energy_tolerance
        )
        if policy.min_steps > inference_steps:
            raise ValueError("policy min_steps must not exceed inference_steps")
    copied_training, copied_heldout = _validated_batches((training,), (heldout,))
    model = _clone(initial)
    if copied_training[0].features.shape[1] != model.input_dim:
        raise ValueError("training feature width differs from the native model input width")
    return (
        model,
        copied_training[0],
        copied_heldout[0],
        window,
        rate,
        inference_rate,
        policy,
        iteration_limit,
    )


def _iteration_max(previous: float | None, start: float, end: float) -> float:
    seconds = end - start
    return seconds if previous is None else max(previous, seconds)


def _check_iteration(observed: float, limit: float) -> None:
    if observed > limit:
        raise RuntimeError("complete training iteration exceeded max_iteration_seconds")


def _record(
    result: CoupledTrainResult, updates: int, seconds: float | None, steps: int
) -> dict[str, Any]:
    diagnostic = result.inference
    return {
        "updates": updates,
        "energy": result.energy,
        "energy_definition": result.energy_definition,
        "call_seconds": seconds,
        "inference": None if diagnostic is None else asdict(diagnostic),
        "latent_steps": steps if diagnostic is None else diagnostic.attempted_steps,
        "inference_energy_evaluations": steps + 1
        if diagnostic is None
        else diagnostic.energy_evaluations,
        "parameter_energy_evaluations": 1,
        "gradient_norm_evaluations": 0
        if diagnostic is None
        else diagnostic.gradient_norm_evaluations,
        "energy_comparisons": 0 if diagnostic is None else diagnostic.energy_comparisons,
        "used_fallback": False if diagnostic is None else diagnostic.used_fallback,
    }


def _work(records: list[dict[str, Any]], unknown: bool, rows: int) -> dict[str, Any]:
    names = (
        "latent_steps",
        "inference_energy_evaluations",
        "parameter_energy_evaluations",
        "gradient_norm_evaluations",
        "energy_comparisons",
    )
    work = {name: None if unknown else sum(record[name] for record in records) for name in names}
    work["fallback_updates"] = (
        None if unknown else sum(record["used_fallback"] for record in records)
    )
    work["training_row_exposures"] = None if unknown else len(records) * rows
    work["unknown_failed_call_work"] = unknown
    return work


def run_timed_inference_case(
    initial: CoupledPredictiveCodingNetwork,
    training: AttributionBatch,
    heldout: AttributionBatch,
    *,
    training_seconds: float,
    max_updates: int,
    learning_rate: float,
    inference_steps: int,
    inference_learning_rate: float,
    policy: CoupledInferencePolicy | None,
    budget: ToyBudgetSession,
    clock: Callable[[], float] = perf_counter,
    max_iteration_seconds: float | None = None,
) -> dict[str, Any]:
    """Run whole updates until the allocated window ends, then qualify and score.

    The maximum allowed window overshoot is 10%. Reaching the update cap before
    the deadline is a failed case. Unknown failed-call work/timings remain None;
    committed records survive later budget/clock/scoring failures. Case elapsed
    time starts after preflight copying; callers may time the whole invocation.
    Call timing includes fresh per-call training-buffer copies. The last native
    iteration extends through terminal deadline checks, including their overhead.
    An optional iteration ceiling includes the first window prefix and complete
    loop bookkeeping. It qualifies observed intervals; native calls cannot be
    preempted. The default retains the original window-only timing contract.
    """
    model, training, heldout, window, rate, inference_rate, policy, iteration_limit = _prepare(
        initial,
        training,
        heldout,
        training_seconds,
        max_updates,
        learning_rate,
        inference_steps,
        inference_learning_rate,
        policy,
        budget,
        clock,
        max_iteration_seconds,
    )
    initial_state = model.snapshot_state()
    timer = _Clock(clock)
    attempted = committed = 0
    records: list[dict[str, Any]] = []
    pending: CoupledTrainResult | None = None
    iteration_active = False
    window_start: float | None = None
    window_end: float | None = None
    last_iteration: float | None = None
    last_native_iteration_start: float | None = None
    measured_iteration_start: float | None = None
    observed_max_iteration: float | None = None
    scoring_seconds: float | None = None
    initial_scores: dict[str, float] | None = None
    final_scores: dict[str, float] | None = None
    status, reason = "complete", None
    try:
        budget.before_final()
        window_start = timer.read()
        while True:
            iteration_active = True
            iteration_start = timer.read()
            if iteration_limit is not None and measured_iteration_start is not None:
                observed_max_iteration = _iteration_max(
                    observed_max_iteration, measured_iteration_start, iteration_start
                )
                _check_iteration(observed_max_iteration, iteration_limit)
            if iteration_start - window_start >= window:
                window_end = iteration_start
                iteration_active = False
                break
            if attempted >= max_updates:
                raise RuntimeError("max_updates reached before the training deadline")
            budget.before_update()
            call_start = timer.read()
            if call_start - window_start >= window:
                window_end = call_start
                last_iteration = call_start - iteration_start
                iteration_active = False
                break
            attempted += 1
            last_native_iteration_start = iteration_start
            if iteration_limit is not None:
                # Keep the initial window prefix inside the first interval. A
                # terminal loop with no admitted call does not advance this.
                measured_iteration_start = window_start if attempted == 1 else iteration_start
            if policy is None:
                result = model.train_epoch(
                    training.features.copy(),
                    training.targets.copy(),
                    rate,
                    inference_steps,
                    inference_rate,
                )
            else:
                result = model.train_epoch(
                    training.features.copy(),
                    training.targets.copy(),
                    rate,
                    inference_steps,
                    inference_rate,
                    inference_policy=policy,
                )
            committed += 1
            pending = result
            call_end = timer.read()
            records.append(_record(result, committed, call_end - call_start, inference_steps))
            pending = None
            budget.record_update()
            iteration_end = timer.read()
            last_iteration = iteration_end - iteration_start
            iteration_active = False
        # Include the terminal deadline comparison/branch, not just its input
        # timestamp. Qualification and deferred scoring happen after this bound.
        window_end = timer.read()
        observed_window = window_end - window_start
        if last_native_iteration_start is not None:
            last_iteration = window_end - last_native_iteration_start
        if iteration_limit is not None and measured_iteration_start is not None:
            observed_max_iteration = _iteration_max(
                observed_max_iteration, measured_iteration_start, window_end
            )
            _check_iteration(observed_max_iteration, iteration_limit)
        if committed == 0:
            raise RuntimeError("training window completed without a committed update")
        if observed_window > 1.1 * window:
            raise RuntimeError("training window exceeded the 10% overrun allowance")
        score_start = timer.read()
        budget.before_final()
        final_state = model.snapshot_state()
        scorer = _clone(model)
        scorer.restore_state(initial_state)
        initial_scores = _probability_scores(
            scorer.predict_proba(heldout.features.copy()), heldout.targets
        )
        budget.before_final()
        scorer.restore_state(final_state)
        final_scores = _probability_scores(
            scorer.predict_proba(heldout.features.copy()), heldout.targets
        )
        budget.before_final()
        score_end = timer.read()
        scoring_seconds = score_end - score_start
        budget.before_final()
        timer.read()
    except Exception as error:
        status, reason = "failed", f"{type(error).__name__}: {error}"
        initial_scores = final_scores = None
        if iteration_active:
            last_iteration = None
        if pending is not None:
            records.append(_record(pending, committed, None, inference_steps))
            try:
                budget.record_update()
            except Exception as budget_error:
                reason += f"; update accounting: {type(budget_error).__name__}: {budget_error}"
        if not timer.broken:
            try:
                ended = timer.read()
                if window_start is not None and window_end is None:
                    window_end = ended
                    if (
                        iteration_limit is not None
                        and measured_iteration_start is not None
                        and attempted == committed
                    ):
                        observed_max_iteration = _iteration_max(
                            observed_max_iteration, measured_iteration_start, ended
                        )
            except Exception:
                pass
    window_seconds = (
        None if window_start is None or window_end is None else window_end - window_start
    )
    assert timer.last is not None
    elapsed = None if timer.broken else timer.last - timer.started
    unknown_work = attempted > committed
    call_seconds = (
        None
        if unknown_work or any(record["call_seconds"] is None for record in records)
        else sum(record["call_seconds"] for record in records)
    )
    report = {
        "status": status,
        "reason": reason,
        "attempted_updates": attempted,
        "committed_updates": committed,
        "records": records,
        "training_window_seconds": window_seconds,
        "window_budget_seconds": window,
        "overrun_seconds": None if window_seconds is None else max(0.0, window_seconds - window),
        "last_iteration_seconds": last_iteration,
        "train_call_seconds": call_seconds,
        "scoring_seconds": scoring_seconds,
        "elapsed_seconds": elapsed,
        "initial_scores": initial_scores,
        "final_scores": final_scores,
        "work": _work(records, unknown_work, len(training.features)),
    }
    if iteration_limit is not None:
        report["max_iteration_seconds"] = iteration_limit
        report["observed_max_iteration_seconds"] = observed_max_iteration
    return report
