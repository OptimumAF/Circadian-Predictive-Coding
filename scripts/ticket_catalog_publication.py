"""One terminal publication for the existing live catalog window.

Owns no learner, loop, budgets or promotion policy. It freezes the bounded paid
prefix, pauses the original gate, uses the caller's existing guarded publisher,
and retires the old TRAIN owner without deleting/rebinding pending evidence.
"""

from copy import deepcopy
from dataclasses import asdict, replace

from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import paid_report_guard
from src.core.serving_ports import PromotionReceipt


class CatalogWindowPublication:
    def __init__(
        self,
        control,
        runtime,
        gate,
        live,
        report,
        candidate,
        controller,
        now,
        require_environment,
        metadata,
    ):
        if "window_publication" in report or getattr(control, "_terminal_retired", False):
            raise ValueError("catalog window publication already attempted or retired")
        control.require_candidate()
        live._require_original()
        snapshot = live.snapshot()
        progress = report["action_snapshot"]["progress"]
        if (
            not snapshot["closed"]
            or snapshot["stopped"]
            or snapshot["unreturned_work_unknown"]
            or len(snapshot["records"]) != control.catalog.envelope.max_decisions
            or any(not row["complete"] for row in snapshot["records"])
            or progress["decisions_attempted"] != control.catalog.envelope.max_decisions
            or progress["decisions_returned"] != progress["decisions_attempted"]
            or progress["unreturned_work_unknown"]
        ):
            raise ValueError("terminal catalog publication requires a complete known closed window")
        self.control, self.runtime, self.gate, self.live, self.report = (
            control,
            runtime,
            gate,
            live,
            report,
        )
        self._environment, self._metadata = require_environment, deepcopy(metadata)
        self._controller = controller
        if controller._actor is not runtime.actor or controller._latest is not None:
            raise ValueError("terminal publication requires its fresh original controller")
        self._native, self._retired = deepcopy(candidate), False
        self.known_promotion = None
        # The actor shares its frozen configuration in serving views. Detach
        # that too so in-place corruption cannot change both sides of the proof.
        self._before = deepcopy(runtime.actor.serving_snapshot())
        self._record, self._candidate_owner, self._inbox = (
            control.record,
            runtime._candidate,
            runtime._inbox,
        )
        self._limits = gate._limits
        self._gate_before = gate.snapshot()
        if (
            self._gate_before.paused
            or self._gate_before.training_active
            or self._gate_before.active_serving_requests
        ):
            raise ValueError(
                "terminal catalog publication requires its original idle unpaused gate"
            )
        with runtime._exclusive():
            runtime._require_open()
            cursor = runtime._inbox._borrow_cursor()
            self._cursor = deepcopy(cursor)
            self._records = (*cursor.experiences, *cursor.labels, *cursor.applied)
            self._consolidation = deepcopy(runtime._read_consolidation_cursor())
        self._paid = {
            k: report[k]
            for k in (
                "action_snapshot",
                "applied_updates",
                "stream",
                "controller_events",
                "candidate_checkpoint",
                "candidate_episodes",
                "candidate_readout_work",
                "live_serving",
            )
        }
        self._paid["live_records"] = live.records
        applied = {(r.episode_id, r.sample_id) for r in cursor.applied}
        labels = {r.key: r for r in cursor.labels}
        pending = [
            dict(
                episode_id=r.episode_id,
                sample_id=r.sample_id,
                source_version=r.model_version,
                observed_at=r.observed_at,
                label_id=labels[r.key].event_id,
                arrived_at=labels[r.key].arrived_at,
                label_version=labels[r.key].model_version,
            )
            for r in cursor.experiences
            if r.key not in applied
        ]
        self.row = report["window_publication"] = dict(
            at=now,
            completed=False,
            accepted=False,
            publication_unknown=False,
            retired=False,
            runtime_retired=False,
            pending_unapplied=pending,
            gate_before=asdict(self._gate_before),
            receipt=None,
        )
        self._after = None
        self._publishing = False
        self._paid["terminal_pending"] = self.row["pending_unapplied"]
        self._paid["gate_before"] = self.row["gate_before"]
        self._paid_guard, _ = paid_report_guard(self._paid)
        self._row_guard = None
        self._receipt = self._receipt_record = None

    def _require_prefix(self):
        if self.report.get("window_publication") is not self.row:
            raise ValueError("original terminal publication record was replaced")
        if (
            self.control.record is not self._record
            or self.runtime._candidate is not self._candidate_owner
            or self.runtime._inbox is not self._inbox
            or self.gate._limits is not self._limits
        ):
            raise ValueError("original terminal catalog owners were replaced")
        self.control.require_pin()
        self.control.replay._budgets(replay=True)
        self._environment()
        for key, value in self._paid.items():
            current = (
                self.live.records
                if key == "live_records"
                else self.row["pending_unapplied"]
                if key == "terminal_pending"
                else self.row["gate_before"]
                if key == "gate_before"
                else self.report[key]
            )
            if current is not value:
                raise ValueError("original terminal paid report was replaced")
        try:
            self._paid_guard()
        except ValueError as error:
            raise ValueError("terminal paid evidence changed or was replaced") from error
        if self.row.get("gate_after_pause") is not self._gate_after_map or not _same_value(
            self.row["gate_after_pause"], asdict(replace(self._gate_before, paused=True))
        ):
            raise ValueError("terminal pause provenance changed")
        if self._after is None and (
            self.row["completed"] is not False
            or self.row["accepted"] is not False
            or self.row["publication_unknown"] is not self._publishing
            or self.row["receipt"] is not None
        ):
            raise ValueError("terminal publication stage evidence changed")
        if not _same_value(self.gate.snapshot(), replace(self._gate_before, paused=True)):
            raise ValueError("original terminal sharing counters or pause changed")
        if not _same_value(self.live.snapshot(), self._paid["live_serving"]):
            raise ValueError("closed live serving evidence changed")
        with self.runtime._exclusive():
            cursor = self.runtime._inbox._borrow_cursor()
            records = (*cursor.experiences, *cursor.labels, *cursor.applied)
            if (
                len(records) != len(self._records)
                or any(a is not b for a, b in zip(records, self._records))
                or not _same_value(cursor, self._cursor)
                or not _same_value(self.runtime._candidate.snapshot_state(), self._native)
                or not _same_value(
                    self.runtime._read_consolidation_cursor(),
                    replace(self._consolidation, retired=self._retired),
                )
                or self.runtime._retired is not self._retired
            ):
                raise ValueError("original terminal ledger/candidate/evidence changed")

    def _retire(self):
        # Why: end the original owner even on guard rejection/lost commit; never
        # materialize a successor, clear pending labels or renew any allowance.
        self.control._terminal_retired = True
        self.control.report["terminal_retired"] = True
        with self.runtime._exclusive():
            self.runtime._retired = True
            self.row["runtime_retired"] = True
            self.runtime._inbox._retire_ledger()
        self._retired = self.row["retired"] = True

    def publish(self, publisher):
        before = self._before
        self.gate.pause()
        self.row["gate_after_pause"] = asdict(self.gate.snapshot())
        self._gate_after_map = self.row["gate_after_pause"]
        try:
            self.require_before_commit()
            self.row["publication_unknown"] = True
            self._publishing = True
            receipt = publisher()
            self._require_prefix()
            after = self.runtime.actor.serving_snapshot()
            if receipt is None:
                if not _same_value(after, before):
                    raise ValueError("publication changed without its original commit receipt")
            else:
                if (
                    type(receipt) is not PromotionReceipt
                    or type(receipt.generation) is not int
                    or type(after.generation) is not int
                    or type(before.generation) is not int
                    or type(receipt.previous_version) is not str
                    or type(receipt.actor_version) is not str
                    or receipt.generation != before.generation + 1
                    or after.generation != receipt.generation
                    or receipt.previous_version != before.actor_version
                    or receipt.actor_version != after.actor_version
                    or after.actor_version == before.actor_version
                    or not _same_value(after.model_state, self._native)
                    or not _same_value(after.configuration, before.configuration)
                    or not _same_value(after.metadata, self._metadata)
                ):
                    raise ValueError("terminal publication receipt/model/version differs")
                if not _same_value(self.report.get("promotion_receipt"), asdict(receipt)):
                    raise ValueError("original saved promotion receipt changed")
                latest = self._controller._latest
                if (
                    latest is None
                    or receipt is not latest[0]
                    or not _same_value(receipt, latest[1])
                ):
                    raise ValueError(
                        "terminal publication did not return its original controller receipt"
                    )
                self.row["receipt"] = asdict(receipt)
                self.row["accepted"] = True
                self._receipt, self._receipt_record = receipt, self.report["promotion_receipt"]
            self._after = deepcopy(after)
            self.known_promotion = receipt is not None
            self._publishing = False
            self.row.update(
                completed=True,
                publication_unknown=False,
                generation_before=before.generation,
                generation_after=after.generation,
                actor_version_before=before.actor_version,
                actor_version_after=after.actor_version,
            )
        except BaseException as error:
            self.row["error"] = f"{type(error).__name__}: {error}"[:512]
            raise
        finally:
            try:
                self._retire()
            except Exception as error:
                self.row["retirement_error"] = f"{type(error).__name__}: {error}"[:512]
                raise
        self._row_guard, _ = paid_report_guard(self.row)
        self.require_current()

    def require_before_commit(self):
        self.control.require_candidate()
        self._require_prefix()
        if not _same_value(self.runtime.actor.serving_snapshot(), self._before):
            raise ValueError("original actor changed before terminal commit")

    def require_current(self):
        self._require_prefix()
        if self._row_guard is None:
            raise ValueError("terminal publication proof was not completed")
        self._row_guard()
        if self._receipt is not None:
            latest = self._controller._latest
            if (
                latest is None
                or latest[0] is not self._receipt
                or not _same_value(latest[0], latest[1])
                or self.report.get("promotion_receipt") is not self._receipt_record
                or not _same_value(asdict(self._receipt), self.row["receipt"])
                or not _same_value(self._receipt_record, self.row["receipt"])
            ):
                raise ValueError("verified terminal receipt was changed or replaced")
        if not self.row["retired"] or not self.row["completed"] or self.row["publication_unknown"]:
            raise ValueError("terminal publication is incomplete, unknown or not retired")
        if not _same_value(self.runtime.actor.serving_snapshot(), self._after):
            raise ValueError("verified terminal actor publication changed")
