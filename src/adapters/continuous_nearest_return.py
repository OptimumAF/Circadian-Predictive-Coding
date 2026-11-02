"""Compose a bounded old-TRAIN return using the existing disposable probe loop.

Owns separate prospective return resources and two bounded reports. It never
trains the borrowed checkpoint, exports a child, relabels data or loads files.
"""

from dataclasses import asdict
from copy import deepcopy
import json
from time import monotonic

from src.adapters.numpy_retained_nearest_bc import pin_retained_native, require_retained_native
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.probe_endpoint_observer import _BudgetWitness
from src.app.probe_return_phase import budget_binding, paid_report_guard
from src.core.continuous_demonstration import copy_continuous_episode
from src.core.experience import LogicalClock


def run_return_probe(
    source, plan, training, *, training_readouts=(), check_boundary=None, endpoint_observer=None
):
    # Lazy dependency keeps the default single-stage probe independent of this
    # optional composition; both stages still execute the same qualified loop.
    from src.adapters.continuous_probe import (
        copy_probe_plan,
        _RetainedReturnPlan,
        _run_single_probe,
        run_retained_nearest_probe,
    )

    if type(training_readouts) is not tuple or not 1 <= len(training_readouts) <= 4:
        raise ValueError("return requires one to four declared old TRAIN readouts")
    if check_boundary is not None and not callable(check_boundary):
        raise ValueError("check_boundary must be callable or None")
    readouts = tuple(copy_continuous_episode(item) for item in training_readouts)
    owned = copy_probe_plan(plan, readouts)
    return_plan = _RetainedReturnPlan(training, owned.evaluation, owned.training)
    if sum(item == return_plan.training for item in readouts) != 1:
        raise ValueError("return packet must equal exactly one original old TRAIN readout")
    if any(item.role != "train" for item in readouts):
        raise ValueError("old return monitors must retain TRAIN role")

    # Admit both resources now, rather than starting a fresh return clock after
    # the first phase has consumed time. The prior budget is never renewed.
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), monotonic)
    clock = LogicalClock()
    binding, limits = budget_binding(budget), budget.budget
    expected_limits = deepcopy(limits)
    unspent = _BudgetWitness(budget, binding, expected_limits, 0)
    scope = (budget, clock, unspent)
    source_pin = pin_retained_native(source)
    inputs = dict(
        plan=asdict(owned), readouts=[asdict(e) for e in readouts], return_plan=asdict(return_plan)
    )
    input_seal = json.dumps(inputs, sort_keys=True, allow_nan=False)
    return_started = False
    returned = None
    return_guard = None
    return_trusted = None
    terminal_witness = None
    terminal_tick = None

    def pure():
        if (
            budget_binding(budget) != binding
            or budget.budget is not limits
            or asdict(limits) != asdict(expected_limits)
        ):
            raise RuntimeError("original return resource owner changed")
        current = dict(
            plan=asdict(owned),
            readouts=[asdict(e) for e in readouts],
            return_plan=asdict(return_plan),
        )
        if json.dumps(current, sort_keys=True, allow_nan=False) != input_seal:
            raise RuntimeError("original return inputs changed")
        if return_guard is not None:
            try:
                return_guard()
            except Exception:
                returned.clear()
                returned.update(return_trusted)
                raise
        if not return_started:
            unspent.check()
            if type(clock._time) is not int or clock._time != 0:
                raise RuntimeError("return logical clock advanced before return")
        elif terminal_witness is not None:
            terminal_witness.check()
            if type(clock._time) is not int or clock._time != terminal_tick:
                raise RuntimeError("return clock changed after its paid endpoint")

    def boundary():
        if check_boundary is not None:
            check_boundary()
        pure()

    def endpoint(readout, phase, parent_guard):
        nonlocal return_started, returned, return_guard, return_trusted
        if endpoint_observer is not None:
            endpoint_observer(readout, phase, parent_guard)
        parent_guard()
        if phase != "after_probe":
            return
        if return_started:
            raise RuntimeError("return phase already attempted")
        pure()
        return_started = True

        def returned_endpoint(child, stage, guard):
            if stage == "after_probe" and endpoint_observer is not None:
                endpoint_observer(child, "after_return", guard)

        def read_ports():
            nonlocal returned, return_guard, return_trusted, terminal_witness, terminal_tick
            # The supplied source remains frozen. The shared loop creates its
            # own detached candidate/actor/readout for the TRAIN return.
            returned = _run_single_probe(
                readout,
                return_plan,
                training_readouts=readouts,
                check_boundary=lambda: parent_guard(boundary),
                endpoint_observer=returned_endpoint if endpoint_observer is not None else None,
                retained_nearest=True,
                return_scope=scope,
            )
            return_guard, return_trusted = paid_report_guard(returned)
            count = returned["progress"]["updates_completed"]
            if type(count) is int:
                terminal_witness = _BudgetWitness(budget, binding, expected_limits, count)
                terminal_tick = clock._time

        parent_guard(read_ports)

    original = run_retained_nearest_probe(
        source,
        owned,
        training_readouts=readouts,
        check_boundary=boundary,
        endpoint_observer=endpoint,
    )
    result = dict(
        definition="continuous_retained_return_probe_v1",
        complete=False,
        error=None,
        probe=original,
        return_phase=returned,
        classification="isolated original old TRAIN return; native recovery unmeasured",
    )
    original_guard, original_trusted = paid_report_guard(original)
    try:
        # No opaque callbacks after the existing loop's final owner/clock check.
        # Close from native fields and the already admitted report/resource owners.
        pure()
        require_retained_native(source, source_pin, observe=False)
        original_guard()
        if (
            original.get("complete") is not True
            or returned is None
            or returned.get("complete") is not True
        ):
            raise RuntimeError("private acquisition or old TRAIN return incomplete")
        if terminal_witness is None:
            raise RuntimeError("return resource proof missing")
        terminal_witness.check()
        if type(clock._time) is not int or clock._time != 1:
            raise RuntimeError("return logical-clock observation changed")
        before_return = returned["checkpoints"][0]["scores"]
        after_new = original["checkpoints"][-1]["scores"]
        if json.dumps(before_return, sort_keys=True, allow_nan=False) != json.dumps(
            after_new, sort_keys=True, allow_nan=False
        ):
            raise RuntimeError("return zero checkpoint differs from paid private endpoint")
        result["complete"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        try:
            original_guard()
        except Exception:
            original.clear()
            original.update(original_trusted)
    result["work"] = dict(
        training_attempts=sum(
            r["work"]["training_attempts"] for r in (original, returned) if r is not None
        ),
        training_rows_attempted=sum(
            r["work"]["training_rows_attempted"] for r in (original, returned) if r is not None
        ),
        scope="private and old-TRAIN return attempts; source preparation and diagnostics additional",
    )
    return result
