"""Post-training cached/live evaluation using existing key/door boundaries.

Inputs are four or five owned checkpoints, the original completed training packets and
budget, and a guard bound to the completed runtime. Outputs retain all final
outcomes and partial work. No training, promotion, checkpoint selection or IO.
"""

from dataclasses import asdict
from typing import Any, Callable

from src.adapters import key_door_pilot as shared
from src.app.coupled_attribution import _score_checkpoints, _validate_batch, _validated_batches

EVALUATION_POINTS = (0, 64, 128, 192)
FOUR_PHASE_EVALUATION_POINTS = (0, 64, 128, 192, 256)


def _evaluation_layouts(four_phase: bool) -> tuple[Any, ...]:
    original = shared.layouts()
    return (*original[:2], shared.third_training_layout(), original[3]) if four_phase else original


def new_evaluation(*, four_phase: bool = False) -> dict[str, Any]:
    """Allocate bounded report slots without creating data, learners or sessions."""
    if type(four_phase) is not bool:
        raise ValueError("four_phase must be an exact bool")
    points = FOUR_PHASE_EVALUATION_POINTS if four_phase else EVALUATION_POINTS
    known = ("A", "B", "C_train") if four_phase else ("A", "B")
    return dict(
        method="recurrent_bc",
        replay_enabled=False,
        status="not_run",
        reason="not reached",
        checkpoints=[],
        final_demonstrations=[],
        outcomes=None,
        live_sessions_started=0,
        episodes=[
            dict(
                checkpoint_updates=point,
                context=layout.layout_id,
                cue=cue,
                layout_partition="known_layout" if layout.layout_id in known else "heldout_layout",
                status="not_run",
                reason="not reached",
                state_unchanged=None,
            )
            for point in points
            for layout in _evaluation_layouts(four_phase)
            for cue in (1, -1)
        ],
        work=dict(
            final_collections_attempted=0,
            final_collections_completed=0,
            cached_forwards_attempted=0,
            cached_forwards_completed=0,
            cached_rows_attempted=0,
            cached_rows_completed=0,
        ),
    )


class _CountedScorer:
    """Prediction-only accounting on the existing owned scorer port, including faults."""

    def __init__(self, learner: Any, work: dict[str, Any]) -> None:
        self.learner, self.work = learner, work

    def fork(self) -> "_CountedScorer":
        child = self.learner.fork()
        if child is self.learner:
            raise ValueError("final cached scorer must own a distinct learner")
        return _CountedScorer(child, self.work)

    def restore_state(self, state: Any) -> None:
        self.learner.restore_state(state)

    def snapshot_state(self) -> Any:
        return self.learner.snapshot_state()

    def predict(self, features: Any) -> Any:
        self.work["cached_forwards_attempted"] += 1
        self.work["cached_rows_attempted"] += len(features)
        result = self.learner.predict(features)
        self.work["cached_forwards_completed"] += 1
        self.work["cached_rows_completed"] += len(features)
        return result


def _outcomes(
    row: dict[str, Any], *, declared_layouts: tuple[Any, ...] | None = None
) -> list[dict[str, Any]]:
    outcomes = []
    for point in row["checkpoints"]:
        groups = {}
        for layout in shared.layouts() if declared_layouts is None else declared_layouts:
            name = layout.layout_id
            scores = [point["scores"][f"{name}-{cue}"] for cue in ("red", "blue")]
            live = [
                e
                for e in row["episodes"]
                if e["checkpoint_updates"] == point["updates"] and e["context"] == name
            ]
            if (
                len(live) != 2
                or {e["cue"] for e in live} != {1, -1}
                or any(e["status"] != "complete" or not e["state_unchanged"] for e in live)
            ):
                raise ValueError(
                    "policy outcome requires both complete unchanged live cue episodes"
                )
            groups[name] = dict(
                mean_ce=sum(s["ce"] for s in scores) / 2,
                mean_accuracy=sum(s["accuracy"] for s in scores) / 2,
                success_rate=sum(e["success"] for e in live) / 2,
                mean_return=sum(e["return"] for e in live) / 2,
                layout_partition=live[0]["layout_partition"],
            )
        outcomes.append(dict(updates=point["updates"], by_layout=groups))
    return outcomes


def evaluate_policy(
    row: dict[str, Any],
    source: Any,
    states: list[tuple[int, Any]],
    budget: Any,
    *,
    training: tuple[Any, ...],
    check_boundary: Callable[[], None],
    four_phase: bool = False,
) -> None:
    """Collect final packets only after the caller's completed-training guard passes."""
    if type(four_phase) is not bool:
        raise ValueError("four_phase must be an exact bool")
    updates = 256 if four_phase else 192
    points = FOUR_PHASE_EVALUATION_POINTS if four_phase else EVALUATION_POINTS
    row.update(status="running", reason=None)
    try:
        check_boundary()
        if (
            len(training) != updates
            or budget.updates_completed != updates
            or budget.budget.max_training_updates != updates
        ):
            raise ValueError(
                f"final evaluation requires the completed original{updates}update session"
            )
        if tuple(n for n, _ in states) != points:
            raise ValueError(
                "policy evaluation requires original five fixed checkpoints"
                if four_phase
                else "policy evaluation requires original four fixed checkpoints"
            )
        declared = _evaluation_layouts(four_phase)
        if four_phase:
            expected = [
                (
                    point,
                    layout.layout_id,
                    cue,
                    "known_layout" if layout.layout_id != "D" else "heldout_layout",
                )
                for point in points
                for layout in declared
                for cue in (1, -1)
            ]
            actual = [
                (slot["checkpoint_updates"], slot["context"], slot["cue"], slot["layout_partition"])
                for slot in row["episodes"]
            ]
            if actual != expected:
                raise ValueError(
                    "four-phase policy evaluation requires all40 original declared slots"
                )
        seen = {identity for b in training for identity in b.row_ids}
        final = []
        for layout in declared:
            for cue in (1, -1):
                check_boundary()
                row["work"]["final_collections_attempted"] += 1
                batch = shared._collect(
                    f"four-phase-final-{layout.layout_id}-{cue}"
                    if four_phase
                    else f"return-final-{layout.layout_id}-{cue}",
                    layout,
                    cue,
                    "final_test",
                )
                row["work"]["final_collections_completed"] += 1
                _validate_batch(batch, "final_test", seen)
                if batch.episode is None:
                    raise ValueError("final policy evaluation requires complete episodes")
                row["final_demonstrations"].append(asdict(batch.episode))
                final.append(batch)
                check_boundary()
        _, owned_final = _validated_batches(training, tuple(final))
        row["checkpoints"] = _score_checkpoints(
            _CountedScorer(source, row["work"]),
            states,
            training,
            owned_final,
            budget,
            verify_state=True,
            check_boundary=check_boundary,
        )
        check_boundary()
        shared._live(
            row,
            source,
            states,
            budget,
            task_sequence=("A", "B", "C_train", "A") if four_phase else ("A", "B", "A"),
            check_boundary=check_boundary,
            **({"declared_layouts": declared} if four_phase else {}),
        )
        check_boundary()
        row["outcomes"] = (
            _outcomes(row, declared_layouts=declared) if four_phase else _outcomes(row)
        )
        row.update(status="complete", reason=None)
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error!r}")
        raise
    finally:
        row["work"].update(
            live_callback_attempts=sum(
                e.get("prediction_calls_attempted", 0) for e in row["episodes"]
            ),
            live_callback_returns=sum(
                e.get("prediction_calls_returned", 0) for e in row["episodes"]
            ),
            live_applied_steps=sum(e.get("steps_executed", 0) for e in row["episodes"]),
            live_sessions_started=row["live_sessions_started"],
            failure_work_may_be_unobserved=row["status"] != "complete",
            scope="returned cached calls retained on failure; cached checkpoints publish only after complete scoring; partial failed native work also needs execution receipt",
        )
