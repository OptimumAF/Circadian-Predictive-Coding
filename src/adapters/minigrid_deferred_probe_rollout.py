"""Capture four paid count or explicit H32 endpoints for the frozen game reader.

Private immutable payloads retain genuine typed states, never scalar reconstructions.
The original learners are not retained. The parent brackets cohort/source guards;
this adapter checks capture ownership, completed original budgets and paid games
inside the deferred phase. It grants no learning, clock renewal, IO or selection.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from math import isfinite
import pickle
from time import monotonic
from typing import Any, TypedDict, cast

from src.adapters import minigrid_memory_control as memory
from src.adapters import minigrid_probe_rollout as inline
from src.adapters.invariant_action_memory_learner import copy_invariant_memory_snapshot
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureLearner
from src.adapters.torch_feature_rehearsal_pilot import _tensor_bytes
from src.app.probe_endpoint_observer import EndpointBoundary, EndpointObserver, _BudgetWitness
from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import budget_binding
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget


class _ReaderOptions(TypedDict, total=False):
    wide_method: str
    extended_wide_probe_budget: bool


ORDER = ((0, "after_probe"), (0, "after_return"), (128, "after_probe"), (128, "after_return"))
MAX_STATE_BYTES = 256 * 1024
MAX_PAYLOAD_BYTES = 1024 * 1024
MAX_LOGICAL_BYTES = 620544


def _method(wide_method) -> None:
    if wide_method is not None and (
        type(wide_method) is not str or wide_method not in ("bp", "pc", "neutral_cpc", "gated_cpc")
    ):
        raise ValueError("deferred H32 games require an exact original native method")


def validate_option(flag, memory_control, inline, *, wide_method=None) -> None:
    _method(wide_method)
    if (
        type(flag) is not bool
        or (wide_method is not None and memory_control is not False)
        or (flag and (inline is not False or (wide_method is None and memory_control is not True)))
    ):
        raise ValueError(
            "deferred_probe_rollouts requires exact bools, count or explicit H32, and inline=False"
        )


def _state(value, origin: int, phase: str, *, wide_method=None):
    if wide_method is not None:
        return inline._copy_wide_neural_probe_snapshot(
            value, source_updates=origin, phase=phase, method=wide_method
        )
    owned = copy_invariant_memory_snapshot(value)
    updates = origin + (16 if phase == "after_probe" else 32)
    if (
        owned.policy != memory.policy()
        or type(owned.completed_updates) is not int
        or owned.completed_updates != updates
        or type(owned.training_examples) is not int
        or owned.training_examples != 9 * updates
    ):
        raise ValueError("deferred capture differs from its actual invariant count endpoint")
    return owned


def _witness(budget: ToyBudgetSession) -> tuple[_BudgetWitness, Callable[[], None]]:
    if (
        type(budget) is not ToyBudgetSession
        or type(budget.budget) is not ToyExecutionBudget
        or type(budget.budget.max_training_updates) is not int
        or budget.budget.max_training_updates != 16
    ):
        raise ValueError("deferred games require each original completed16-update budget")
    limits = replace(budget.budget)
    witness = _BudgetWitness(budget, budget_binding(budget), limits, 16)
    binding = witness.binding
    last_clock, progress, segment, session_work = (
        witness.last_clock,
        deepcopy(witness.progress),
        witness.segment_owner,
        deepcopy(witness.session_work),
    )

    def check() -> None:
        # Keep the witness's own authority fields from being replaced too.
        if (
            witness.budget is not budget
            or witness.binding is not binding
            or witness.limits is not limits
            or witness.updates != 16
            or type(witness.updates) is not int
            or not _same_value(witness.last_clock, last_clock)
            or witness.segment_owner is not segment
            or not _same_value(witness.progress, progress)
            or not _same_value(witness.session_work, session_work)
        ):
            raise ValueError("original completed budget witness was replaced")
        witness.check()

    return witness, check


class DeferredProbeRollouts:
    """Own at most four immutable captures, two readers and four paid game rows."""

    def __init__(self, *, wide_method=None, extended_wide_probe_budget=False) -> None:
        _method(wide_method)
        inline._validate_extended_option(wide_method is not None, extended_wide_probe_budget)
        self.wide_method = wide_method
        self.extended_wide_probe_budget = extended_wide_probe_budget
        captures: tuple[tuple[int, str, bytes, int], ...] = ()
        bindings: tuple[tuple[int, Callable[[], None], Callable[[], None], int, int], ...] = ()
        readers: tuple[inline.ProbeRollouts, ...] = ()
        capture_records: tuple[bytes, ...] = ()
        records: tuple[bytes, ...] = ()
        status: str = "not_run"
        reason: str | None = None
        limits = (MAX_STATE_BYTES, MAX_PAYLOAD_BYTES, MAX_LOGICAL_BYTES)
        order = ORDER
        work: dict[str, Any] = {
            key: 0
            for key in (
                "snapshot_calls_attempted",
                "snapshot_calls_returned",
                "snapshot_copy_calls_attempted",
                "snapshot_copy_calls_returned",
                "decode_calls_attempted",
                "decode_calls_returned",
                "frozen_reader_factories_attempted",
                "frozen_reader_factories_returned",
                "frozen_reader_restores_attempted",
                "frozen_reader_restores_returned",
                "reader_snapshot_calls_attempted",
                "reader_snapshot_calls_returned",
            )
        }
        work.update(
            payload_bytes_retained=0,
            logical_payload_bytes_retained=0,
            elapsed_seconds=0.0,
            preparation_elapsed_seconds=0.0,
            failure_work_may_be_unobserved=False,
        )
        if wide_method is not None:
            work.update(preparation_copy_calls_attempted=0, preparation_copy_calls_returned=0)
        self._captures, self._bindings, self._readers = captures, bindings, readers
        self._limits, self._order = limits, order

        def proof() -> None:
            if (
                self._captures is not captures
                or self._bindings is not bindings
                or self._readers is not readers
                or self._limits is not limits
                or self._order is not order
                or type(self.wide_method) is not type(wide_method)
                or self.wide_method is not wide_method
                or type(self.extended_wide_probe_budget) is not bool
                or self.extended_wide_probe_budget is not extended_wide_probe_budget
                or any(getattr(self, name) is not owner for name, owner in owners.items())
            ):
                raise ValueError("private deferred capture registry or proof was replaced")
            if (
                len(captures) > 4
                or len(capture_records) > 4
                or len(bindings) > 2
                or len(records) > 4
            ):
                raise ValueError("private deferred registry exceeds its fixed bound")
            for _, first, second, _, _ in bindings:
                first()
                second()
            for reader in readers:
                reader.check()

        def owner_proof() -> None:
            if self._proof is not proof:
                raise ValueError("private deferred capture proof was replaced")

        def charge(**increments) -> None:
            for key, value in increments.items():
                if key == "failure_work_may_be_unobserved":
                    work[key] = work[key] or value
                else:
                    work[key] += value

        def fail(error) -> None:
            nonlocal status, reason
            if status != "failed":
                status, reason = "failed", f"{type(error).__name__}: {error}"

        def capture(entry, record) -> None:
            nonlocal captures, capture_records, status
            if entry is not None:
                proof()
            if len(capture_records) >= 4:
                raise ValueError("deferred capture receipt allowance exhausted")
            capture_records += (pickle.dumps(record, protocol=5),)
            if entry is not None:
                captures += (entry,)
                self._captures = captures
                work["payload_bytes_retained"] += len(entry[2])
                work["logical_payload_bytes_retained"] += entry[3]
                if status != "failed":
                    status = "capturing"

        def bind(entry) -> None:
            nonlocal bindings
            proof()
            bindings += (entry,)
            self._bindings = bindings

        def reader(owner) -> None:
            nonlocal readers
            proof()
            readers += (owner,)
            self._readers = readers

        def save_games() -> None:
            nonlocal records
            # Existing reader payloads preserve paid/failed rows even if a
            # subsequently checked public record or binding was corrupted.
            records = tuple(payload for owner in readers for _, payload, _, _ in owner._sealed)

        def finish() -> None:
            nonlocal status
            if status != "failed":
                status = "complete"

        def report() -> dict[str, Any]:
            measured = deepcopy(work)
            measured["failure_work_may_be_unobserved"] |= any(
                work[f"{prefix}_attempted"] > work[f"{prefix}_returned"]
                for prefix in (
                    "snapshot_calls",
                    "snapshot_copy_calls",
                    "decode_calls",
                    "frozen_reader_factories",
                    "frozen_reader_restores",
                    "reader_snapshot_calls",
                )
            )
            if wide_method is not None:
                measured["failure_work_may_be_unobserved"] |= (
                    work["preparation_copy_calls_attempted"]
                    > work["preparation_copy_calls_returned"]
                )
            return dict(
                status=status,
                reason=reason,
                captures=[pickle.loads(value) for value in capture_records],
                records=[pickle.loads(value) for value in records],
                capture_work=measured,
                limits=dict(
                    maximum_captures=4,
                    maximum_state_bytes=limits[0],
                    maximum_payload_bytes=limits[1],
                    maximum_logical_payload_bytes=limits[2],
                ),
                scope="private typed captures and separately restored frozen readers; original learner not retained; training cohort checks bracket this phase in the parent; capture/decoder/reduction overhead additional to game costs",
            )

        self._proof, self._owner_proof = proof, owner_proof
        self._charge, self._fail, self._capture, self._bind = charge, fail, capture, bind
        self._reader, self._save_games, self._finish, self._report = (
            reader,
            save_games,
            finish,
            report,
        )
        # Pin evidence interpreters before invoking external state/resource ports.
        for name in ("_copy", "_same"):
            setattr(self, name, getattr(self, name))
        owners = {
            name: getattr(self, name)
            for name in (
                "_proof",
                "_owner_proof",
                "_charge",
                "_fail",
                "_capture",
                "_bind",
                "_reader",
                "_save_games",
                "_finish",
                "_report",
                "_copy",
                "_same",
            )
        }

    def _copy(self, value, origin, phase, *, capture=False):
        prefix = "snapshot_copy_calls" if capture else "preparation_copy_calls"
        charged = capture or self.wide_method is not None
        if charged:
            self._charge(**{f"{prefix}_attempted": 1})
        owned = _state(value, origin, phase, wide_method=self.wide_method)
        if charged:
            self._charge(**{f"{prefix}_returned": 1})
        return owned

    def _same(self, first, second) -> bool:
        return (
            inline._same_state(first, second) if self.wide_method is not None else first == second
        )

    def check(self) -> None:
        self._owner_proof()
        self._proof()

    def observer(self, origin: int) -> EndpointObserver:
        if type(origin) is not int or origin not in (0, 128):
            raise ValueError("deferred capture requires exact original0/128")

        def observe(learner, phase: str, check_boundary: EndpointBoundary) -> None:
            self._observe(origin, learner, phase, check_boundary)

        return observe

    def _observe(self, origin, learner, phase, boundary) -> None:
        try:
            self.check()
            if (
                self._report()["status"] == "failed"
                or type(phase) is not str
                or len(self._captures) >= 4
                or (origin, phase) != self._order[len(self._captures)]
            ):
                raise ValueError("deferred endpoint capture is out of fixed global order")
        except Exception as error:
            self._fail(error)
            raise
        record = dict(
            source_updates=origin,
            phase=phase,
            status="failed",
            reason=None,
            native_updates=None,
            native_training_examples=None,
            payload_bytes=None,
            logical_payload_bytes=None,
        )
        pending = None
        read_called = False
        port_error: BaseException | None = None
        started = monotonic()

        def capture_read() -> None:
            nonlocal pending, read_called
            if read_called:
                raise ValueError("qualified capture ports may be read only once")
            read_called = True
            self.check()
            self._charge(snapshot_calls_attempted=1)
            value = learner.snapshot_state()
            self._charge(snapshot_calls_returned=1)
            owned = self._copy(value, origin, phase, capture=True)
            payload = pickle.dumps(owned, protocol=5)
            native = owned.native_state
            logical = (
                sum(
                    _tensor_bytes(value)
                    for value in (native.head_state, native.optimizer_state, native.gradients)
                )
                if self.wide_method is not None
                else len(native.entries) * 2424
            )
            record.update(
                native_updates=owned.completed_updates,
                native_training_examples=owned.training_examples,
                payload_bytes=len(payload),
                logical_payload_bytes=logical,
            )
            if (
                len(payload) > self._limits[0]
                or sum(len(entry[2]) for entry in self._captures) + len(payload) > self._limits[1]
                or sum(entry[3] for entry in self._captures) + logical > self._limits[2]
            ):
                raise ValueError("deferred capture exceeds its bounded private payload allowance")
            self.check()
            pending = (origin, phase, payload, logical)

        def read() -> None:
            nonlocal port_error
            try:
                capture_read()
            except BaseException as error:
                if port_error is None:
                    port_error = error
                raise

        try:
            if self.wide_method is None:
                result = boundary(read)
            else:
                final_called = False

                def capture_final_read() -> None:
                    nonlocal final_called
                    if final_called or pending is None:
                        raise ValueError("qualified capture final read is out of order")
                    final_called = True
                    self.check()
                    self._charge(snapshot_calls_attempted=1)
                    value = learner.snapshot_state()
                    self._charge(snapshot_calls_returned=1)
                    observed = self._copy(value, origin, phase, capture=True)
                    expected = self._decode(pending)
                    if not self._same(observed, expected):
                        raise ValueError("deferred endpoint changed during capture resource clocks")
                    self.check()

                def final_read() -> None:
                    nonlocal port_error
                    try:
                        capture_final_read()
                    except BaseException as error:
                        if port_error is None:
                            port_error = error
                        raise

                result = boundary(read, after_clocks=final_read)
                if port_error is not None:
                    raise port_error
                if not final_called:
                    raise ValueError("qualified capture omitted its final native read")
            if port_error is not None:
                raise port_error
            if result is not None:
                raise ValueError("qualified endpoint boundary must return None")
            self.check()
            if pending is None:
                raise ValueError("qualified endpoint boundary did not execute capture ports")
            record["status"] = "complete"
            self._capture(pending, record)
        except Exception as error:
            record["reason"] = f"{type(error).__name__}: {error}"
            self._fail(error)
            self._capture(None, record)
            raise
        finally:
            ended = monotonic()
            if (
                type(started) is float
                and type(ended) is float
                and isfinite(ended - started)
                and ended >= started
            ):
                self._charge(elapsed_seconds=ended - started)
            else:
                self._charge(failure_work_may_be_unobserved=True)

    def bind_completed(
        self, origin: int, budget: ToyBudgetSession, return_budget: ToyBudgetSession
    ) -> None:
        self.check()
        if (
            type(origin) is not int
            or origin not in (0, 128)
            or budget is return_budget
            or len(self._captures) != (2 if origin == 0 else 4)
            or len(self._bindings) != (0 if origin == 0 else 1)
            or any(
                id(candidate) in entry[-2:]
                for candidate in (budget, return_budget)
                for entry in self._bindings
            )
        ):
            raise ValueError("deferred completed budget binding is out of order")
        first, first_check = _witness(budget)
        second, second_check = _witness(return_budget)
        self._bind((origin, first_check, second_check, id(budget), id(return_budget)))
        # Witnesses remain owned by the closures; these references prove no
        # original budget is replaced or observed/renewed by check().
        first.check()
        second.check()

    def _decode(self, entry):
        self.check()
        self._charge(decode_calls_attempted=1)
        owned = self._copy(pickle.loads(entry[2]), entry[0], entry[1])
        self._charge(decode_calls_returned=1)
        return owned

    def _prepare(self, entry, guard, frozen):
        started = monotonic()
        try:
            guard()
            expected = self._decode(entry)
            if entry[1] == "after_probe":
                self._charge(frozen_reader_factories_attempted=1)
                frozen = (
                    InvariantTorchFeatureLearner(expected.policy)
                    if self.wide_method is not None
                    else memory.learner()
                )
                self._charge(frozen_reader_factories_returned=1)
                guard()
                options: _ReaderOptions = {}
                if self.wide_method is not None:
                    options["wide_method"] = self.wide_method
                if self.extended_wide_probe_budget:
                    options["extended_wide_probe_budget"] = True
                self._reader(inline.ProbeRollouts(entry[0], **options))
            self._charge(frozen_reader_restores_attempted=1)
            frozen.restore_state(self._copy(expected, entry[0], entry[1]))
            self._charge(frozen_reader_restores_returned=1)
            guard()
            self._charge(reader_snapshot_calls_attempted=1)
            returned = frozen.snapshot_state()
            self._charge(reader_snapshot_calls_returned=1)
            observed = self._copy(returned, entry[0], entry[1])
            guard()
            if not self._same(observed, expected):
                raise ValueError("fresh frozen reader did not restore the captured full state")
            return frozen
        finally:
            ended = monotonic()
            if (
                type(started) is float
                and type(ended) is float
                and isfinite(ended - started)
                and ended >= started
            ):
                self._charge(preparation_elapsed_seconds=ended - started)
            else:
                self._charge(failure_work_may_be_unobserved=True)

    def run(self, *, check_boundary: Callable[[], None]) -> dict[str, Any]:
        """Compatibility entry: enclose an ordinary no-argument parent callback."""

        def legacy_guard() -> None:
            self.check()
            if not callable(check_boundary):
                raise ValueError("deferred games require a callable parent boundary")
            check_boundary()
            self.check()

        def close_native_port(port, closing) -> None:
            # Closing proofs still run, but cannot replace the first paid error.
            try:
                port()
            except BaseException:
                try:
                    closing()
                except BaseException:
                    pass
                raise
            else:
                closing()

        def composed(
            read_ports: Callable[[], None] | None = None,
            *,
            after_clocks: Callable[[], None] | None = None,
        ) -> None:
            legacy_guard()
            if read_ports is not None:
                if self.wide_method is not None:
                    close_native_port(read_ports, legacy_guard)
                else:
                    try:
                        read_ports()
                    finally:
                        legacy_guard()
            if after_clocks is not None and self.wide_method is not None:
                close_native_port(after_clocks, self.check)
            elif after_clocks is not None:
                try:
                    after_clocks()
                finally:
                    # The final native read follows the last external clock
                    # callback. Only private registry proofs may follow it.
                    self.check()

        return self._run(check_boundary=composed)

    def run_qualified(self, *, check_boundary: EndpointBoundary) -> dict[str, Any]:
        """Use the trusted application's registry/resource/read enclosure.

        The caller must prove this manager and original benchmark witnesses
        around resources and supplied synchronous reads, including on failure.
        Why: one owner can share adjoining proofs without dropping a port check.
        Ordinary callbacks must continue to use run().
        """
        return self._run(check_boundary=check_boundary)

    def _run(self, *, check_boundary: EndpointBoundary) -> dict[str, Any]:
        try:
            self.check()
            if (
                not callable(check_boundary)
                or self._report()["status"] == "failed"
                or len(self._captures) != 4
                or len(self._bindings) != 2
                or self._readers
            ):
                raise ValueError(
                    "deferred games require all four paid captures and two completed budget pairs"
                )

            frozen = None
            for entry in self._captures:
                check_boundary()
                frozen = self._prepare(entry, check_boundary, frozen)
                check_boundary()
                reader = self._readers[-1]
                boundary = check_boundary
                if self.wide_method is not None:
                    # Why: the inline reader's first snapshot must be the captured
                    # state, including when a parent resource port changes weights
                    # without changing clocks before the inline reader anchors it.
                    expected = self._decode(entry)

                    def native_read() -> None:
                        self.check()
                        self._charge(reader_snapshot_calls_attempted=1)
                        value = frozen.snapshot_state()
                        self._charge(reader_snapshot_calls_returned=1)
                        observed = self._copy(value, entry[0], entry[1])
                        if not self._same(observed, expected):
                            raise ValueError(
                                "deferred frozen reader differs from its captured full state"
                            )
                        self.check()

                    def anchored_boundary(read_ports=None, *, after_clocks=None) -> None:
                        reads_called = final_called = False
                        port_error: BaseException | None = None

                        def reads() -> None:
                            nonlocal reads_called, port_error
                            try:
                                if reads_called:
                                    raise ValueError("deferred native reads called more than once")
                                reads_called = True
                                native_read()
                                if read_ports is not None and read_ports() is not None:
                                    raise ValueError("deferred read ports must return None")
                                native_read()
                            except BaseException as error:
                                if port_error is None:
                                    port_error = error
                                raise

                        def final_reads() -> None:
                            nonlocal final_called, port_error
                            try:
                                if final_called or not reads_called:
                                    raise ValueError("deferred final native reads are out of order")
                                final_called = True
                                if after_clocks is not None and after_clocks() is not None:
                                    raise ValueError("deferred final reads must return None")
                                native_read()
                            except BaseException as error:
                                if port_error is None:
                                    port_error = error
                                raise

                        result = cast(Any, check_boundary)(reads, after_clocks=final_reads)
                        if port_error is not None:
                            raise port_error
                        if result is not None or not reads_called or not final_called:
                            raise ValueError("deferred boundary did not qualify all native reads")

                    boundary = anchored_boundary
                try:
                    reader.observe_qualified(frozen, entry[1], boundary)
                finally:
                    self._save_games()
                boundary()
            self.check()
            self._finish()
        except Exception as error:
            self._fail(error)
        return self._report()

    def report(self) -> dict[str, Any]:
        return self._report()


def describe(
    protocol: dict[str, Any],
    *,
    wide_neural: bool = False,
    extended_wide_probe_budget: bool = False,
) -> None:
    inline.describe(
        protocol,
        wide_neural=wide_neural,
        extended_wide_probe_budget=extended_wide_probe_budget,
    )
    protocol["protocol_id"] = (
        protocol["protocol_id"].removesuffix("_probe_rollout_v1") + "_deferred_probe_rollouts_v1"
    )
    protocol["deferred_probe_rollouts"] = protocol.pop("probe_rollout")
    protocol["deferred_probe_rollouts"].update(
        order=[list(pair) for pair in ORDER],
        all_training_before_games=True,
        maximum_state_bytes=MAX_STATE_BYTES,
        maximum_payload_bytes=MAX_PAYLOAD_BYTES,
        maximum_logical_payload_bytes=MAX_LOGICAL_BYTES,
        frozen_readers=2,
        frozen_restores=4,
        boundary_scope="cohorts/sources/receipts bracket the phase; capture/budget/paid-game/native guards within; refusal ordering differs from inline games",
    )


def add_work(work: dict[str, Any], phase_record: dict[str, Any]) -> None:
    report = phase_record.get("report")
    records = report.get("records", []) if type(report) is dict else []
    inline.add_work(work, [dict(probes=[dict(probe_rollouts=records)])])
    work["deferred_capture"] = (
        deepcopy(report.get("capture_work")) if type(report) is dict else None
    )
    if (
        phase_record.get("status") == "failed"
        or phase_record.get("entry_validated") is not True
        or phase_record.get("exit_validated") is not True
    ):
        work["failure_work_may_be_unobserved"] = True
    if report is None and phase_record.get("entry_validated") is True:
        work["probe_rollouts"]["failure_work_may_be_unobserved"] = True
        for key in (
            "environment_resets_completed",
            "live_prediction_calls_returned",
            "environment_steps_returned",
            "internal_projection_rows_returned",
            "internal_projection_array_bytes_returned",
        ):
            work["probe_rollouts"][key] = None
        for key in (
            "learner_live_episodes",
            "live_prediction_forwards_completed",
            "total_prediction_forwards_completed",
            "total_prediction_rows_completed",
            "total_environment_steps_returned",
        ):
            work[key] = None
    work["deferred_phase_scope"] = (
        "original cohort/source/receipt checks at phase entry/exit, not inline-equivalent timing; separate capture/decoder/factory/restore work"
    )
