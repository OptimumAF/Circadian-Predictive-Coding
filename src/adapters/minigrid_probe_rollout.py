"""Target-free games at paid count, H32 or explicit public-cue probe endpoints.

Inputs are a genuine probe learner, its phase and completed-work guard. Outputs
are at most two protected live reports and separate zero-update budget charges.
The existing evaluator owns games; this module supplies no training, selection,
checkpoint export, final access or IO.
"""

from collections.abc import Callable, Sequence
from copy import deepcopy
import pickle
from math import isfinite
from time import monotonic
from typing import Any

from src.adapters import minigrid_memory_control as memory
from src.adapters.minigrid_endpoint_timing import NativeMiniGridEndpointTiming
from src.adapters.invariant_action_memory_learner import copy_invariant_memory_snapshot
from src.adapters.minigrid_policy_evaluation import (
    MiniGridEvaluationFailed,
    _copy_wide_neural_probe_snapshot,
    _copy_public_cue_probe_snapshot,
    _return_budget_guard,
    evaluate_memory_probe_checkpoint,
    evaluate_wide_neural_probe_checkpoint,
    evaluate_public_cue_probe_checkpoint,
)
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.probe_endpoint_observer import EndpointBoundary
from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget

LIVE_SECONDS = 20.0
# Why: the paid H32 qualification reached42/64 steps in20s after guard
# optimizations. Fresh H32 cases get40s; the enclosing180s deadline still wins.
# Historical20s failures remain failed and their budgets are never renewed.
WIDE_LIVE_SECONDS = 40.0
# Why: the separately selected readout envelope keeps the original40/180s
# qualification intact. Limits belong to new budgets; old charges never renew.
WIDE_EXTENDED_LIVE_SECONDS = 60.0
WIDE_EXTENDED_BODY_SECONDS = 240.0
PHASES = ("after_probe", "after_return")


def validate_option(flag, memory_control, *, wide_neural: bool = False) -> None:
    if (
        type(flag) is not bool
        or type(wide_neural) is not bool
        or (flag and not (memory_control is True or wide_neural is True))
        or (wide_neural and memory_control is not False)
    ):
        raise ValueError(
            "probe_rollout requires an exact bool and the count or explicit H32 recipe"
        )


def _validate_extended_option(wide_neural: bool, extended_wide_probe_budget: bool) -> None:
    if (
        type(wide_neural) is not bool
        or type(extended_wide_probe_budget) is not bool
        or (extended_wide_probe_budget and not wide_neural)
    ):
        raise ValueError(
            "extended wide probe budget requires exact bools and the explicit H32 recipe"
        )


def _live_budget(
    *,
    wide_neural: bool = False,
    extended_wide_probe_budget: bool = False,
    public_cue: bool = False,
) -> tuple[ToyBudgetSession, Callable[[], None]]:
    """Accept elapsed-clock changes only from actual reads of this clock port."""
    _validate_extended_option(wide_neural, extended_wide_probe_budget)
    if type(public_cue) is not bool or public_cue and (wide_neural or extended_wide_probe_budget):
        raise ValueError("public-cue readout requires its explicit separate budget mode")
    owner: list[ToyBudgetSession] = []
    observed: list[float] = []

    def clock() -> float:
        if owner and (
            type(owner[0].last_clock) is not float or owner[0].last_clock != observed[-1]
        ):
            raise ValueError("live budget clock changed without an observed read")
        value = monotonic()
        if type(value) is not float or not isfinite(value):
            raise ValueError("live budget requires a finite monotonic clock")
        observed[:] = [value]
        return value

    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=0,
            max_replay_examples=0,
            max_wall_seconds=(
                WIDE_EXTENDED_LIVE_SECONDS
                if extended_wide_probe_budget
                else WIDE_LIVE_SECONDS
                if wide_neural or public_cue
                else LIVE_SECONDS
            ),
        ),
        clock,
    )
    owner.append(budget)
    bound = _return_budget_guard(budget)

    def guard() -> None:
        bound()
        if type(budget.last_clock) is not float or budget.last_clock != observed[-1]:
            raise ValueError("live budget clock changed without an observed read")

    guard()
    return budget, guard


def describe(
    protocol: dict[str, Any],
    *,
    wide_neural: bool = False,
    extended_wide_probe_budget: bool = False,
) -> None:
    _validate_extended_option(wide_neural, extended_wide_probe_budget)
    protocol["protocol_id"] = protocol["protocol_id"].removesuffix("_v1") + "_probe_rollout_v1"
    protocol["probe_rollout"] = dict(
        phases=list(PHASES),
        source_updates=[0, 128],
        native_endpoint_updates=[16, 32, 144, 160],
        games="A blue/red TRAIN then F blue/red probe_test at each same-fork endpoint",
        policy="frozen raw seven-action first argmax; no targets, teacher, mask, repair or retry",
        budget="separate20s zero training/replay updates per endpoint inside original180s/RSS cap",
        scope="known fixed task gameplay, not heldout/final generalization or circadian advantage",
    )
    if wide_neural:
        protocol["probe_rollout"].update(
            hidden_dim=32,
            native_endpoint_training_rows=[144, 288, 7056, 7200],
            exposure_law="54*source_updates + 9*relative_updates; relative16/32",
            budget="separate40s zero training/replay updates per H32 endpoint inside original180s/RSS cap; count reader retains20s",
        )
        protocol["metric_direction"] = protocol["metric_direction"].replace("; no live success", "")
    if extended_wide_probe_budget:
        protocol["probe_rollout"].update(
            endpoint_wall_seconds=WIDE_EXTENDED_LIVE_SECONDS,
            parent_wall_seconds=WIDE_EXTENDED_BODY_SECONDS,
            budget="explicit separate60s zero training/replay updates per H32 endpoint inside selected240s/RSS cap; original40s/180s gate and count20s unchanged",
        )
    protocol["planned_work"].update(
        learner_live_episodes=16,
        maximum_live_prediction_forwards=256,
        maximum_live_environment_steps=256,
        maximum_total_prediction_forwards=protocol["planned_work"]["score_forwards"] + 256,
        maximum_total_prediction_rows=protocol["planned_work"]["score_rows"] + 256,
        maximum_total_environment_steps=protocol["planned_work"]["maximum_source_environment_steps"]
        + 256,
    )
    protocol["evaluation"] = (
        protocol["evaluation"].replace("no live/final evaluation", "no final evaluation")
        + "; same-fork target-free known A/F games after F16 and A16"
    )
    protocol["limitations"] += "; live known-task games are not heldout transfer or CPC benefit"
    protocol["metric_direction"] += (
        "; live success/return higher; native valid/invalid inputs reported separately"
    )


class ProbeRollouts:
    """Keep bounded paid reports immutable while the ordinary probe learns."""

    def __init__(
        self,
        source_updates: int,
        *,
        wide_method: str | None = None,
        extended_wide_probe_budget: bool = False,
        public_cue: bool = False,
    ) -> None:
        if (
            type(public_cue) is not bool
            or public_cue
            and (wide_method is not None or extended_wide_probe_budget is not False)
        ):
            raise ValueError("public-cue games require an exact bool without wide modes")
        if type(source_updates) is not int or source_updates not in (
            (2,) if public_cue else (0, 128)
        ):
            raise ValueError("probe rollout requires the original0/128 source label")
        self.source_updates = source_updates
        if wide_method is not None and (
            type(wide_method) is not str
            or wide_method not in ("bp", "pc", "neutral_cpc", "gated_cpc")
        ):
            raise ValueError("wide probe rollout requires an exact original native method")
        _validate_extended_option(wide_method is not None, extended_wide_probe_budget)
        self.wide_method = wide_method
        self.extended_wide_probe_budget = extended_wide_probe_budget
        self.public_cue = public_cue
        self._configuration = (source_updates, wide_method, extended_wide_probe_budget, public_cue)
        self.records: list[dict[str, Any]] = []
        self._sealed: list[
            tuple[dict[str, Any], bytes, Callable[[], None], Callable[[], None]]
        ] = []

    def check(self) -> None:
        if (
            type(self.source_updates) is not int
            or (self.wide_method is not None and type(self.wide_method) is not str)
            or type(self.extended_wide_probe_budget) is not bool
            or type(self.public_cue) is not bool
            or (
                self.source_updates,
                self.wide_method,
                self.extended_wide_probe_budget,
                self.public_cue,
            )
            != self._configuration
        ):
            raise ValueError("probe rollout endpoint configuration changed")
        if len(self.records) != len(self._sealed):
            raise ValueError("paid endpoint rollout list changed")
        for actual, (owner, _, proof, binding) in zip(self.records, self._sealed, strict=True):
            if actual is not owner:
                raise ValueError("paid endpoint rollout record replaced")
            proof()
            binding()

    def restore_paid(self) -> None:
        # Why: rejected public edits cannot become the published paid history.
        self.records[:] = [pickle.loads(payload) for _, payload, _, _ in self._sealed]

    def observe(self, learner, phase: str, check_boundary: Callable[[], None]) -> None:
        """Retain the original no-argument parent callback contract."""

        def composed(
            read_ports: Callable[[], None] | None = None,
            *,
            after_clocks: Callable[[], None] | None = None,
        ) -> None:
            check_boundary()
            try:
                if read_ports is not None:
                    read_ports()
                if after_clocks is not None:
                    after_clocks()
            finally:
                check_boundary()

        self._observe(learner, phase, composed)

    def observe_qualified(self, learner, phase: str, check_boundary: EndpointBoundary) -> None:
        """Use the app's enclosure once for clock and native snapshot reads."""
        self._observe(learner, phase, check_boundary)

    def _observe(self, learner, phase: str, check_boundary: EndpointBoundary) -> None:
        self.check()
        if self.public_cue and any(r["status"] != "complete" for r in self.records):
            raise ValueError("public-cue next endpoint requires the preceding successful readout")
        if (
            type(phase) is not str
            or phase not in PHASES
            or len(self.records) >= len(PHASES)
            or phase != PHASES[len(self.records)]
        ):
            raise ValueError("probe rollout endpoint is out of fixed order")
        record: dict[str, Any] = dict(
            phase=phase,
            source_updates=self.source_updates,
            status="not_run",
            reason=None,
            report=None,
            charged_updates=0,
            replay_examples=0,
            elapsed_seconds=None,
            timing_complete=None,
            budget_binding_preserved=None,
            evaluator_admitted=False,
        )
        budget, binding = _live_budget(
            **({"public_cue": True} if self.public_cue else {}),
            **({"wide_neural": True} if self.wide_method is not None else {}),
            **({"extended_wide_probe_budget": True} if self.extended_wide_probe_budget else {}),
        )
        expected = None
        payload = None
        observed_elapsed = 0.0
        # Why: the timed-out H32 path needs a cost breakdown before another
        # experiment. This diagnostic clock grants no budget or update authority.
        timings = NativeMiniGridEndpointTiming() if self.wide_method is not None else None

        def copy_endpoint(value):
            if self.public_cue:
                return _copy_public_cue_probe_snapshot(
                    value, source_updates=self.source_updates, phase=phase
                )
            if self.wide_method is not None:
                return _copy_wide_neural_probe_snapshot(
                    value, source_updates=self.source_updates, phase=phase, method=self.wide_method
                )
            return copy_invariant_memory_snapshot(value)

        def observe_ports(native_reads: Callable[[], None] | None = None) -> None:
            nonlocal observed_elapsed
            self.check()
            binding()
            budget.before_final()
            binding()
            observed_elapsed = budget.last_clock - budget.started_at
            if expected is not None and native_reads is None:
                current = copy_endpoint(learner.snapshot_state())
                # Why: the fresh copier validates every scalar type and finite
                # key before full dataclass/tuple equality. Generic recursion
                # adds measured cost here; it remains necessary for reports.
                if not (
                    _same_value(current, expected)
                    if self.public_cue
                    else _same_state(current, expected)
                    if self.wide_method is not None
                    else current == expected
                ):
                    raise ValueError("live endpoint changed the disposable learned state")
            if native_reads is not None:
                native_reads()
            # Clock/snapshot ports may change prior records or the parent owner.
            self.check()
            binding()

        def guard(
            read_ports: Callable[[], None] | None = None,
            *,
            after_clocks: Callable[[], None] | None = None,
        ) -> None:
            # Why: the evaluator already proves the full source/scorer/checkpoint
            # group. Qualify it once with our clock instead of nesting two guards.
            if after_clocks is None:
                check_boundary(lambda: observe_ports(read_ports))
            else:

                def final_reads() -> None:
                    self.check()
                    binding()
                    after_clocks()
                    self.check()
                    binding()

                check_boundary(lambda: observe_ports(read_ports), after_clocks=final_reads)

        try:
            guard()
            expected = copy_endpoint(learner.snapshot_state())
            relative = 16 if phase == "after_probe" else 32
            if (
                self.wide_method is None
                and not self.public_cue
                and (
                    expected.policy != memory.policy()
                    or expected.completed_updates != self.source_updates + relative
                    or expected.training_examples != 9 * expected.completed_updates
                )
            ):
                raise ValueError("live endpoint is not the actual fixed paid count state")
            guard()
            try:
                record["evaluator_admitted"] = True
                if self.public_cue:
                    result = evaluate_public_cue_probe_checkpoint(
                        learner,
                        copy_endpoint(expected),
                        source_updates=self.source_updates,
                        phase=phase,
                        budget=budget,
                        check_boundary=guard,
                        qualified_boundary=guard,
                    )
                elif self.wide_method is not None:
                    result = evaluate_wide_neural_probe_checkpoint(
                        learner,
                        copy_endpoint(expected),
                        source_updates=self.source_updates,
                        phase=phase,
                        budget=budget,
                        check_boundary=guard,
                        qualified_boundary=guard,
                        method=self.wide_method,
                        timings=timings,
                    )
                else:
                    result = evaluate_memory_probe_checkpoint(
                        learner,
                        copy_endpoint(expected),
                        source_updates=self.source_updates,
                        phase=phase,
                        budget=budget,
                        check_boundary=guard,
                    )
            except MiniGridEvaluationFailed as error:
                payload = pickle.dumps(error.report, protocol=5)
                raise
            payload = pickle.dumps(result, protocol=5)
            guard()
            # The delegate may retain its returned dictionary. Only our captured
            # paid payload can authorize the next phase after external guards.
            paid = pickle.loads(payload)
            if type(paid) is not dict or paid.get("status") != "complete":
                raise ValueError("live probe endpoint readout did not complete")
            work = paid.get("work")
            keys = (
                "environment_resets_completed",
                "live_prediction_calls_returned",
                "environment_steps_returned",
                *(
                    (
                        "raw_score_forwards_completed",
                        "raw_score_rows_completed",
                        "live_sessions_started",
                    )
                    if self.public_cue
                    else (
                        "internal_projection_rows_returned",
                        "internal_projection_array_bytes_returned",
                    )
                ),
            )
            if (
                type(work) is not dict
                or any(type(work.get(key)) is not int or work[key] < 0 for key in keys)
                or work["environment_resets_completed"] != (2 if self.public_cue else 4)
                or not 0 < work["live_prediction_calls_returned"] <= (32 if self.public_cue else 64)
                or work["environment_steps_returned"] != work["live_prediction_calls_returned"]
                or (
                    not self.public_cue
                    and (
                        work["internal_projection_rows_returned"]
                        != work["live_prediction_calls_returned"]
                        or work["internal_projection_array_bytes_returned"]
                        != 2368 * work["internal_projection_rows_returned"]
                    )
                )
                or (
                    self.public_cue
                    and (
                        work["raw_score_forwards_completed"] != 0
                        or work["raw_score_rows_completed"] != 0
                        or work["live_sessions_started"] != 2
                    )
                )
            ):
                raise ValueError("live probe endpoint returned invalid paid work")
            if self.public_cue:
                role = "probe_test" if phase == "after_probe" else "train"
                contexts = ("F-blue", "F-red") if phase == "after_probe" else ("A", "A")
                episodes = paid.get("episodes", ())
                if (
                    paid.get("method") != "recurrent_public_cue"
                    or paid.get("role") != role
                    or paid.get("state_unchanged") is not True
                    or paid.get("scores") != []
                    or len(episodes) != 2
                    or any(
                        e.get("status") != "complete"
                        or e.get("source_role") != role
                        or e.get("context") != c
                        or e.get("cue") != q
                        or e.get("environment", {}).get("closed") is not True
                        for e, c, q in zip(episodes, contexts, ("blue", "red"), strict=True)
                    )
                ):
                    raise ValueError("public-cue paid game roles/order/cleanup differ")
            record["status"] = "complete"
        except Exception as error:
            record.update(status="failed", reason=f"{type(error).__name__}: {error}")
        finally:
            if payload is not None:
                record["report"] = pickle.loads(payload)
            try:
                guard()
                record["timing_complete"] = True
            except Exception as error:
                record.update(
                    status="failed",
                    reason=f"{type(error).__name__}: {error}",
                    timing_complete=False,
                )
            try:
                # An expired deadline is distinct from a changed binding.
                binding()
                record["budget_binding_preserved"] = True
            except Exception as error:
                record.update(
                    status="failed",
                    reason=f"{type(error).__name__}: {error}",
                    budget_binding_preserved=False,
                )
            record.update(
                charged_updates=budget.updates_completed,
                replay_examples=budget.replay_examples_completed,
                elapsed_seconds=observed_elapsed,
            )
            if timings is not None:
                record["phase_timings"] = timings.report()
            proof, _ = paid_report_guard(record)
            self.records.append(record)
            self._sealed.append((record, pickle.dumps(record, protocol=5), proof, binding))
        if record["status"] != "complete":
            raise RuntimeError(record["reason"])


def add_work(work: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    """Separate live work from scores; lost admitted reports stay unknown."""
    records = [
        record
        for row in rows
        for slot in row["probes"]
        for record in slot.get("probe_rollouts", [])
    ]
    totals = dict(
        environment_resets_completed=0,
        live_prediction_calls_returned=0,
        environment_steps_returned=0,
        internal_projection_rows_returned=0,
        internal_projection_array_bytes_returned=0,
    )
    unknown = False
    for record in records:
        report = record.get("report")
        unknown |= record.get("evaluator_admitted") is True and report is None
        paid = report.get("work") if type(report) is dict else None
        if type(paid) is dict:
            for key in totals:
                value = paid.get(key)
                if type(value) is int and value >= 0:
                    totals[key] += value
                else:
                    unknown = True
            unknown |= paid.get("failure_work_may_be_unobserved", False) is not False
        elif record.get("evaluator_admitted") is True:
            unknown = True
        unknown |= record.get("timing_complete") is False
    work["probe_rollouts"] = dict(
        **{key: None if unknown else value for key, value in totals.items()},
        known_returned_work_lower_bounds=deepcopy(totals),
        endpoint_records=len(records),
        charged_updates=sum(record["charged_updates"] for record in records),
        replay_examples=sum(record["replay_examples"] for record in records),
        failure_work_may_be_unobserved=unknown,
        scope="separate endpoint game costs; known lower bounds exclude admitted lost reports",
    )
    work["learner_live_episodes"] = None if unknown else totals["environment_resets_completed"]
    work["live_prediction_forwards_completed"] = (
        None if unknown else totals["live_prediction_calls_returned"]
    )
    for key, baseline, extra in (
        (
            "total_prediction_forwards_completed",
            "total_score_forwards_completed",
            "live_prediction_calls_returned",
        ),
        (
            "total_prediction_rows_completed",
            "total_score_rows_completed",
            "live_prediction_calls_returned",
        ),
        (
            "total_environment_steps_returned",
            "source_environment_steps_returned",
            "environment_steps_returned",
        ),
    ):
        work[key] = None if unknown else work.get(baseline, 0) + totals[extra]
    work["failure_work_may_be_unobserved"] |= unknown
