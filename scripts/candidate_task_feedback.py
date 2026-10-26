"""Bounded private task diagnostics at the existing reference observation port.

Consumes an arrived ordinary TRAIN opportunity and one detached candidate.
Returns causal controller-TRAIN task counts, with paid prefixes on failure.
No fitting, guard labels, publication, acknowledgement or final/probe access.
"""

from copy import deepcopy
from time import monotonic
from typing import Any

from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import paid_report_guard
from src.core.reference_controller import ArrivedTaskFeedback


class CandidateTaskFeedback:
    """One finite schedule owns diagnostics and preceding paid evidence."""

    def __init__(self, readout, attempt_limit):
        if not callable(readout) or type(attempt_limit) is not int or not 1 <= attempt_limit <= 24:
            raise ValueError("task feedback requires bounded original reader/horizon")
        self._readout = readout
        self._limit = attempt_limit
        self._rows: list[dict[str, Any]] = []
        self._paid = []
        self._active = None
        self.report = dict(
            definition="private_controller_train_task_feedback_v1",
            attempt_limit=attempt_limit,
            typed_limit_per_attempt=64,
            records=self._rows,
        )

    def require_paid(self, report):
        if (
            report is not self.report
            or set(report) != {"definition", "attempt_limit", "typed_limit_per_attempt", "records"}
            or report["definition"] != "private_controller_train_task_feedback_v1"
            or report["attempt_limit"] != self._limit
            or report["typed_limit_per_attempt"] != 64
            or report["records"] is not self._rows
            or len(self._rows) != len(self._paid) + int(self._active is not None)
            or self._active is not None
            and self._rows[-1] is not self._active
        ):
            # Keep the paid prefix visible even if its public list was cleared.
            self._rows[:] = [row for row, _, _ in self._paid] + (
                [] if self._active is None else [self._active]
            )
            for row, _, expected in self._paid:
                row.clear()
                row.update(deepcopy(expected))
            self.report.clear()
            self.report.update(
                definition="private_controller_train_task_feedback_v1",
                attempt_limit=self._limit,
                typed_limit_per_attempt=64,
                records=self._rows,
            )
            raise ValueError("preceding paid task-feedback container changed")
        for index, (row, guard, expected) in enumerate(self._paid):
            try:
                if self._rows[index] is not row:
                    raise ValueError("preceding paid task result replaced")
                guard()
            except ValueError:
                row.clear()
                row.update(deepcopy(expected))
                self._rows[index] = row
                raise ValueError("preceding paid task result changed") from None
            if row.get("complete") is not True:
                raise ValueError("preceding paid task result incomplete")

    def read(
        self,
        runtime,
        batches,
        *,
        arrived,
        completed,
        tick,
        label_delay,
        actor,
        require_input,
    ):
        self.require_paid(self.report)
        if self._active is not None:
            raise ValueError("reentrant task feedback refused")
        if arrived <= completed:
            return None
        batch = batches[completed]
        if batch.role != "train" or completed + label_delay > tick:
            raise ValueError("task feedback requires already-arrived ordinary TRAIN")
        if len(self._rows) >= self._limit:
            raise ValueError("original task-feedback attempt allowance exhausted")
        require_input()
        actor_version = actor.version
        snapshot = runtime.candidate_snapshot()
        if snapshot.actor_version != actor_version or snapshot.wake_updates != completed:
            raise ValueError("task feedback belongs to stale candidate version/count")
        pin = deepcopy(snapshot)
        binding = dict(
            role="controller_train",
            index=completed,
            label_observed_at=completed,
            label_arrived_at=completed + label_delay,
            read_at=tick,
            base_actor_version=actor_version,
            candidate_version=snapshot.learner_version,
            wake_updates=completed,
            native_updates=snapshot.state.completed_updates,
        )
        row = dict(binding, task_outcomes=dict(before=[]), task_readout_work={}, complete=False)
        # Append before invoking work; the current row is sealed only at return.
        self._rows.append(row)
        self._active = row
        started = monotonic()

        def check():
            require_input()
            self.require_paid(self.report)
            if actor.version != actor_version:
                raise ValueError("original actor version changed during task feedback")
            if any(row.get(k) != v for k, v in binding.items()):
                raise ValueError("original task-feedback role/receipt binding changed")
            if not _same_value(snapshot, pin) or not _same_value(runtime.candidate_snapshot(), pin):
                raise ValueError("task feedback changed detached/original candidate")

        try:
            counts = self._readout(snapshot.state, "before", check, row)
            check()
            from scripts.context_controller_learning import _task_counts

            if (
                type(row.get("task_outcomes")) is not dict
                or set(row["task_outcomes"]) != {"before"}
                or type(row.get("task_readout_work")) is not dict
                or set(row["task_readout_work"]) != {"before"}
            ):
                raise ValueError("task feedback requires exact private development phase")
            actual = _task_counts(
                row["task_outcomes"]["before"], "before", pin.state.completed_updates
            )
            work = row["task_readout_work"]["before"]
            reads = sum(len(r["reads"]) for r in row["task_outcomes"]["before"])
            if (
                counts != actual
                or reads > 64
                or any(
                    work.get(k) != reads
                    for k in ("attempts", "returns", "readouts", "typed_attempts")
                )
            ):
                raise ValueError("incomplete or unknown task feedback readout work")
            check()
            feedback = ArrivedTaskFeedback(
                "controller_train",
                f"task-batch-{completed}",
                actor_version,
                tick,
                tick,
                6 - sum(actual),
                6,
            )
            row.update(complete=True, failed_tasks=feedback.failed_tasks, tasks=feedback.tasks)
            return feedback
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"[:512]
            raise
        finally:
            row["seconds"] = monotonic() - started
            self._active = None
            guard, expected = paid_report_guard(row)
            self._paid.append((row, guard, expected))
