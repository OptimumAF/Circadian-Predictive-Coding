"""Shared-head auxiliary LS callbacks using consented, arrived source TRAIN.

Owns a bounded source handoff and scalar paid evidence, not a training loop,
source budget renewal, probe bank, monitor-label conversion or publication.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

from src.adapters.linear_fit_learner import LinearFitLearner
from src.app.coupled_attribution import AttributionStream, _copy_batch, _same_value, _validate_batch
from src.app.probe_return_phase import paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_linear_state import LinearFitPolicy, LinearFitSnapshot
from src.core.probe_update import ProbeUpdateFailed, ProbeUpdateReceipt
from src.core.shared_linear_replay import compose_shared_packet

DEFINITION = "shared_linear_joint_probe_update_v1"
LIMITS = ToyExecutionBudget(
    max_training_updates=32,
    max_replay_examples=384,
    max_wall_seconds=30,
    max_process_rss_bytes=512 * 1024**2,
)
ONCE_LIMITS = replace(LIMITS, max_replay_examples=288)


def _observed_application(before, after, rows):
    return (
        type(before) is LinearFitSnapshot
        and type(after) is LinearFitSnapshot
        and after.policy == before.policy
        and after.model is not None
        and after.model.fit_rows == rows
        and after.completed_updates == before.completed_updates + 1
        and after.training_examples == before.training_examples + rows
    )


class SourceTrainPackets:
    """Two fixed packets from the same live, fully arrived consented source."""

    def __init__(self, stream, state):
        if type(stream) is not AttributionStream or type(state) is not LinearFitSnapshot:
            raise ValueError("shared rehearsal requires its live ordinary LS source")
        self._stream = self._owner = stream
        with stream._exclusive():
            self._admit(state)
            inbox = stream._runtime._inbox
            self._owners = (
                stream._training,
                stream._runtime,
                inbox._experiences,
                inbox._labels,
                inbox._applied,
            )
            self._source_clock = stream._clock
            self._records = tuple(tuple(map(id, mapping.values())) for mapping in self._owners[2:])
            self._values = deepcopy(
                (stream._training, inbox._experiences, inbox._labels, inbox._applied, state)
            )
            self._bank = tuple(_copy_batch(stream._training[i]) for i in (0, 4))
            self._bank_owner = self._bank
            self._bank_arrays = tuple((id(b), id(b.features), id(b.targets)) for b in self._bank)
        self.check()

    def _admit(self, state):
        stream = self._stream
        stream._require_open()
        stream._check_original()
        stream._budget.before_final()
        if (
            stream._replay_consent is not True
            or stream._original_replay_consent is not True
            or stream._training_role != "train"
            or not stream._training_only
            or stream._delay != 1
            or stream._clock.now() != 12
            or stream.completed_updates != 12
            or len(stream._training) != 12
            or stream._rehearsal is not None
            or state.policy != LinearFitPolicy(9, 6)
            or (state.completed_updates, state.training_examples) != (12, 72)
            or not _same_value(stream._runtime.candidate_snapshot().state, state)
        ):
            raise ValueError("shared rehearsal requires original consented TRAIN checkpoint12")
        inbox = stream._runtime._inbox
        seen: set[str] = set()
        for i, batch in enumerate(stream._training):
            _validate_batch(batch, "train", seen)
            key = ("attribution", f"batch-{i}")
            source, label, receipt = (
                inbox._experiences[key],
                inbox._labels[key],
                inbox._applied[key],
            )
            if (
                len(batch.features) != 6
                or batch.features.shape[1] != 9
                or batch.num_classes != 6
                or batch.episode is not None
                or batch.packet_episodes
                or source.role != "train"
                or label.role != "train"
                or not source.permissions.training
                or not source.permissions.replay
                or source.observed_at != i
                or label.arrived_at != i + 1
                or source.key != key
                or label.key != key
                or receipt.sample_id != f"batch-{i}"
                or receipt.episode_id != key[0]
                or receipt.event_id != label.event_id
                or (source.model_version, label.model_version, receipt.actor_version)
                != (stream._runtime._base_actor_version,) * 3
                or receipt.learner_version != inbox._learner_version
                or receipt.learner_version != stream._runtime._candidate_version
                or receipt.update_number != i + 1
                or receipt.observed_at != i
                or receipt.arrived_at != i + 1
                or receipt.applied_at < receipt.arrived_at
                or receipt.applied_at > stream._clock.now()
                or not _same_value(source.features, batch.features)
                or not _same_value(label.targets, batch.targets)
            ):
                raise ValueError("retained TRAIN differs from its original arrived paid receipt")

    def check(self):
        if self._stream is not self._owner:
            raise ValueError("original shared rehearsal source replaced")
        stream = self._stream
        with stream._exclusive():
            if stream._clock is not self._source_clock:
                raise ValueError("original source logical clock replaced")
            self._admit(self._values[-1])
            inbox = stream._runtime._inbox
            owners = (
                stream._training,
                stream._runtime,
                inbox._experiences,
                inbox._labels,
                inbox._applied,
            )
            records = tuple(tuple(map(id, m.values())) for m in owners[2:])
            values = (stream._training, inbox._experiences, inbox._labels, inbox._applied)
            if (
                any(a is not b for a, b in zip(owners, self._owners, strict=True))
                or records != self._records
                or not _same_value(values, self._values[:-1])
                or self._bank is not self._bank_owner
                or self._bank_arrays
                != tuple((id(b), id(b.features), id(b.targets)) for b in self._bank)
                or any(
                    not _same_value(b, self._values[0][i])
                    for b, i in zip(self._bank, (0, 4), strict=True)
                )
            ):
                raise ValueError("original source TRAIN or paid receipt evidence changed")

    @property
    def array_bytes(self):
        # Named array scopes only; metadata/state/report/allocator copies additional.
        return dict(retained_train_bytes=960, guard_train_array_bytes=11520)

    def require_private(self, learner):
        runtime = self._stream._runtime
        if (
            not isinstance(learner, LinearFitLearner)
            or learner.policy != LinearFitPolicy(9, 6)
            or learner is self._stream._learner
            or learner is runtime._candidate
            or runtime.actor._owns(learner)
        ):
            raise ValueError("shared rehearsal may update only a detached private LS learner")


class LinearProbeWork:
    """32 bounded auxiliary applications; permanent attempts and public evidence."""

    def __init__(self, row, require, source, budget, *, control, return_once=False):
        if (
            type(row) is not dict
            or not callable(require)
            or type(source) is not SourceTrainPackets
            or type(budget) is not ToyBudgetSession
            or type(return_once) is not bool
            or budget.budget != (ONCE_LIMITS if return_once else LIMITS)
            or budget.updates_completed != 0
            or budget.replay_examples_completed != 0
            or control not in ("joint", "current_only")
            or type(control) is not str
            or "linear_rehearsal_work" in row
            or type(row.get("replay_examples", 0)) is not int
            or row.get("replay_examples", 0) != 0
        ):
            raise ValueError("shared rehearsal requires fresh separate auxiliary authority")
        self._row, self._require, self._source, self._budget = row, require, source, budget
        self._row.setdefault("replay_examples", 0)
        self._published_exposures = 0
        self._control = self._original_control = control
        self._return_once = self._original_return_once = return_once
        self._birth = (
            budget.budget,
            budget.clock,
            budget.started_at,
            budget.process_rss_sampler,
            budget.progress,
        )
        self._limits = deepcopy(budget.budget)
        self._last_clock = budget.last_clock
        self._paid: dict[str, Any] = dict(
            definition=DEFINITION,
            control=control,
            attempted_updates=0,
            attempted_rows=0,
            returned_updates=0,
            applied_updates=0,
            application_unknown_updates=0,
            additional_rows_charged=0,
            native_sgd_updates=0,
            source_array_bytes=source.array_bytes,
            events=[],
        )
        if return_once:
            self._paid.update(
                definition="shared_linear_once_return_probe_update_v1", return_once=True
            )
        self._view: dict[str, Any] | None = None
        self._guard: Any = None
        self._refused: Exception | None = None
        self._publish()

    def check(self):
        if self._refused is not None:
            raise ValueError("preceding shared rehearsal refusal is permanent") from self._refused
        try:
            self._check()
        except Exception as error:
            self._refused = error
            raise

    def _check(self):
        self._source.check()
        budget = self._budget
        if (
            type(self._row.get("replay_examples")) is not int
            or self._row["replay_examples"] != self._published_exposures
            or self._control != self._original_control
            or self._return_once is not self._original_return_once
            or budget.budget != self._limits
            or any(
                a is not b
                for a, b in zip(
                    (budget.budget, budget.clock, budget.process_rss_sampler, budget.progress),
                    (self._birth[0], self._birth[1], self._birth[3], self._birth[4]),
                    strict=True,
                )
            )
            or budget.started_at != self._birth[2]
            or budget.last_clock != self._last_clock
            or budget.updates_completed != self._paid["attempted_updates"]
            or budget.replay_examples_completed != self._paid["additional_rows_charged"]
        ):
            raise ValueError("original shared rehearsal control or auxiliary authority changed")
        if self._view is not None:
            if self._row.get("linear_rehearsal_work") is not self._view:
                raise ValueError("paid shared rehearsal evidence replaced")
            self._guard()

    def _publish(self):
        self.check()
        # Only this observed admission can advance the public exposure counter.
        self._published_exposures = self._paid["additional_rows_charged"]
        self._row["replay_examples"] = self._published_exposures
        self._view = deepcopy(self._paid)
        self._row["linear_rehearsal_work"] = self._view
        self._guard, _ = paid_report_guard(self._view)

    @property
    def preserved(self):
        result = deepcopy(self._paid)
        result["budget"] = dict(
            limits=asdict(self._limits),
            started_at=self._birth[2],
            last_clock=self._last_clock,
            admissions=self._paid["attempted_updates"],
            additional_rows=self._paid["additional_rows_charged"],
        )
        return result

    def callback(self, learner, current, retained, *, phase, check_boundary):
        if not callable(check_boundary):
            raise ValueError("shared rehearsal requires a boundary callback")
        self._require()
        self.check()
        self._source.require_private(learner)
        if (
            phase not in ("fresh", "return")
            or phase == "fresh"
            and any(e["phase"] == "return" for e in self._paid["events"])
            or any(e["status"] != "complete" for e in self._paid["events"])
        ):
            raise ValueError("failed or returning shared rehearsal cannot admit more fresh work")
        _validate_batch(current, "probe_train" if phase == "fresh" else "train", set())
        if (
            current.num_classes != 6
            or current.features.shape[1] != 9
            or len(current.features) != (3 if phase == "fresh" else 6)
            or type(retained) is not tuple
            or any(b.role != "train" for b in retained)
        ):
            raise ValueError("shared rehearsal requires its original private phase geometry/roles")
        try:
            self._budget.before_update()
        finally:
            self._last_clock = self._budget.last_clock
        extra_rows = 6 if self._return_once and phase == "return" else 12
        if (
            self._budget.remaining_replay_examples is None
            or self._budget.remaining_replay_examples < extra_rows
        ):
            self._budget.stop_replay()
        self._paid["attempted_updates"] += 1
        self._paid["attempted_rows"] += len(current.features)
        self._paid["additional_rows_charged"] += extra_rows
        self._budget.record_update()  # Admissions stay charged even if fitting never returns.
        self._budget.record_replay(extra_rows)
        event = dict(
            phase=phase,
            current_rows=len(current.features),
            status="running",
            work_known=False,
            applied=False,
        )
        self._paid["events"].append(event)
        self._publish()
        work = dict(
            native_sgd_updates=0,
            fit_attempts=0,
            fit_returns=0,
            fit_rows=0,
            additional_rows_charged=extra_rows,
            retained_train_rows_used=extra_rows if self._control == "joint" else 0,
            current_rows_used=len(current.features)
            + (extra_rows if self._control == "current_only" else 0),
            packet_array_bytes=0,
            snapshot_attempts=0,
            snapshot_returns=0,
            application_unknown=0,
            unreturned_fit_work_unknown=0,
        )
        before = None
        after = None
        applied = False
        loss = None
        failure = None

        def boundary():
            self._require()
            check_boundary()
            self._require()
            self.check()
            try:
                self._budget.before_final()
            finally:
                self._last_clock = self._budget.last_clock

        try:
            boundary()
            # Diagnostic monitors in `retained` are deliberately NOT a fitting source.
            x, y = compose_shared_packet(
                (current.features, current.targets),
                tuple((b.features, b.targets) for b in self._source._bank),
                6,
                control=self._control,
                **(dict(current_once=True) if self._return_once and phase == "return" else {}),
            )
            work["packet_array_bytes"] = x.nbytes + y.nbytes
            boundary()
            work["snapshot_attempts"] += 1
            before = learner.snapshot_state()
            work["snapshot_returns"] += 1
            if type(before) is not LinearFitSnapshot or before.policy != learner.policy:
                raise ValueError("private LS callback requires its exact original snapshot")
            work["fit_attempts"], work["fit_rows"] = 1, len(x)
            diagnostic = learner.train_batch_minimum_change(x, y)
            work["fit_returns"] = 1
            self._paid["returned_updates"] += 1
            work["snapshot_attempts"] += 1
            after = learner.snapshot_state()
            work["snapshot_returns"] += 1
            applied = _observed_application(before, after, work["fit_rows"])
            if not applied:
                raise ValueError("returned LS callback did not apply its declared update")
            loss = diagnostic.value
            boundary()
        except Exception as error:
            failure = error
            if work["fit_attempts"] and not work["fit_returns"]:
                work["unreturned_fit_work_unknown"] = 1
            if work["fit_attempts"] and not applied:
                try:
                    if after is None:
                        if work["snapshot_attempts"] >= 2:
                            raise ValueError("private post-state read did not return")
                        work["snapshot_attempts"] += 1
                        after = learner.snapshot_state()
                        work["snapshot_returns"] += 1
                    applied = _observed_application(before, after, work["fit_rows"])
                    if not applied and not _same_value(after, before):
                        work["application_unknown"] = 1
                except Exception:
                    work["application_unknown"] = 1
        receipt = ProbeUpdateReceipt(self._paid["definition"], work, bool(applied), loss)
        event.update(
            asdict(receipt), work_known=True, status="complete" if failure is None else "failed"
        )
        self._paid["applied_updates"] += int(applied)
        self._paid["application_unknown_updates"] += work["application_unknown"]
        try:
            self._publish()
            self._require()
        except Exception as error:
            event["status"] = "failed"
            failure = error if failure is None else failure
        if failure is not None:
            raise ProbeUpdateFailed(str(failure), receipt) from failure
        return receipt

    def complete(self, probe):
        self._require()
        if tuple(
            self._paid[k]
            for k in (
                "attempted_updates",
                "attempted_rows",
                "returned_updates",
                "applied_updates",
                "additional_rows_charged",
            )
        ) != (32, 144, 32, 32, 288 if self._return_once else 384):
            raise ValueError("shared rehearsal application sequence incomplete")
        for i, (row, phase) in enumerate(
            ((probe, "fresh"), (probe.get("return_phase", {}), "return"))
        ):
            expected = self._paid["events"][16 * i : 16 * (i + 1)]
            aux = row.get("auxiliary_updates", {})
            if (
                row.get("status") != "complete"
                or aux.get("native_sgd_updates") != 0
                or not _same_value(aux.get("events"), expected)
                or any(e["phase"] != phase or e["status"] != "complete" for e in expected)
            ):
                raise ValueError("shared rehearsal phase receipts differ from paid evidence")
