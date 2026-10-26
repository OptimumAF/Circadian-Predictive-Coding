"""Twelve fixed train-role live readouts for A192/C_train256/A320.

Inputs are three owned native checkpoints and their original completed budget
and authority guard. Outputs retain both cues for both train contexts, signed
live differences and partial work. No training, prior rollouts, new evaluator,
final/probe packets, checkpoint selection, promotion or IO lives here.
"""

from copy import deepcopy
from typing import Any, Callable

from src.adapters import key_door_pilot as shared
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.app.coupled_attribution import _same_value


POINTS = (192, 256, 320)
CONTEXTS = ("A", "C_train")


def _control(value: str | None) -> None:
    if value is not None and (type(value) is not str or value not in ("recent48", "coverage")):
        raise ValueError("replay_control must be None or an exact string recent48 or coverage")


def _checkpoint_clocks() -> list[dict[str, int]]:
    return [
        dict(
            wake_updates=point,
            replay_committed_updates=point // 2,
            replay_charged_examples=8 * (point // 2),
            native_updates=point + point // 2,
            native_training_examples=8 * (point + point // 2),
        )
        for point in POINTS
    ]


def new_diagnostic(*, replay_control: str | None = None) -> dict[str, Any]:
    """Allocate the original twelve slots without creating data or resources."""
    _control(replay_control)
    result = dict(
        source_role="train",
        classification="train_resubstitution_acquired_task_return",
        method="recurrent_bc",
        replay_enabled=False,
        action_conditioning="legal_actions_v1",
        status="not_run",
        reason="not reached",
        episodes=[
            dict(
                checkpoint_updates=point,
                context=context,
                cue=cue,
                layout_partition="known_layout",
                source_role="train",
                status="not_run",
                reason="not reached",
                state_unchanged=None,
            )
            for point in POINTS
            for context in CONTEXTS
            for cue in (1, -1)
        ],
        live_sessions_started=0,
        outcomes=None,
        metrics=None,
        initial_A_acquired=None,
        interpretation="not reached",
        work={},
        scientific_advantage_verified=False,
    )
    if replay_control is not None:
        result.update(
            replay_enabled=True,
            replay_control=replay_control,
            checkpoint_clocks=_checkpoint_clocks(),
        )
    return result


def _outcomes(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for point in POINTS:
        groups = {}
        for context in CONTEXTS:
            selected = [
                slot
                for slot in episodes
                if slot["checkpoint_updates"] == point and slot["context"] == context
            ]
            if len(selected) != 2 or any(
                slot["status"] != "complete" or slot["state_unchanged"] is not True
                for slot in selected
            ):
                raise ValueError(
                    "acquired live outcomes require both complete unchanged cue episodes"
                )
            groups[context] = dict(
                success_rate=sum(slot["success"] for slot in selected) / 2,
                mean_return=sum(slot["return"] for slot in selected) / 2,
            )
        result.append(dict(updates=point, by_layout=groups))
    return result


def _differences(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    initial, learned, returned = [point["by_layout"] for point in outcomes]
    return {
        metric: dict(
            new_C_gain=learned["C_train"][metric] - initial["C_train"][metric],
            old_A_loss=initial["A"][metric] - learned["A"][metric],
            return_A_gain=returned["A"][metric] - learned["A"][metric],
            return_gap=initial["A"][metric] - returned["A"][metric],
            new_C_loss_after_return=learned["C_train"][metric] - returned["C_train"][metric],
        )
        for metric in ("success_rate", "mean_return")
    }


def evaluate_acquired_tasks(
    report: dict[str, Any],
    source: Any,
    states: list[tuple[int, Any]],
    budget: Any,
    *,
    check_boundary: Callable[[], None],
    replay_control: str | None = None,
) -> None:
    """Read all fixed states after completion, regardless of acquisition success."""
    _control(replay_control)
    if not callable(check_boundary):
        raise ValueError("check_boundary must be callable")
    replay_enabled = replay_control is not None
    expected = new_diagnostic(replay_control=replay_control) if replay_enabled else new_diagnostic()
    if type(report) is not dict or not _same_value(report, expected):
        raise ValueError("acquired diagnostic requires its original unused twelve slots")
    if (
        type(states) is not list
        or len(states) != 3
        or any(type(entry) is not tuple or len(entry) != 2 for entry in states)
        or tuple(point for point, _ in states) != POINTS
        or any(type(point) is not int for point, _ in states)
    ):
        raise ValueError("acquired diagnostic requires exact owned checkpoints192/256/320")
    for point, state in states:
        native = point + point // 2 if replay_enabled else point
        if (
            type(state.policy) is not TorchRecurrentPolicy
            or state.policy != TorchRecurrentPolicy(8, 8, 5, 17)
            or type(state.completed_updates) is not int
            or state.completed_updates != native
            or type(state.training_examples) is not int
            or state.training_examples != 8 * native
        ):
            raise ValueError("acquired checkpoint policy or native counters differ")
    if (
        type(budget.updates_completed) is not int
        or budget.updates_completed != 320
        or budget.budget.max_training_updates != 320
    ):
        raise ValueError("acquired diagnostic requires the original completed320update budget")
    frozen_states = deepcopy(states)
    state_bindings = tuple((entry, entry[1]) for entry in states)
    slots = report["episodes"]
    fields = ("checkpoint_updates", "context", "cue", "layout_partition", "source_role")
    bindings = tuple((slot, tuple(slot[field] for field in fields)) for slot in slots)
    clocks: Any = report.get("checkpoint_clocks")
    clock_bindings = tuple(clocks) if replay_enabled else ()
    frozen_clocks = deepcopy(clocks)
    original_budget, clock, birth = budget.budget, budget.clock, budget.started_at
    source_state: Any = None
    completed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    report.update(status="running", reason=None)

    def check_bindings() -> None:
        if (
            report["episodes"] is not slots
            or len(slots) != 12
            or len(states) != 3
            or any(
                actual is not bound or actual[1] is not snapshot
                for actual, (bound, snapshot) in zip(states, state_bindings, strict=True)
            )
            or any(
                actual is not bound
                or not _same_value(tuple(actual.get(field) for field in fields), metadata)
                for actual, (bound, metadata) in zip(slots, bindings, strict=True)
            )
            or report["source_role"] != "train"
            or report["method"] != "recurrent_bc"
            or report["classification"] != "train_resubstitution_acquired_task_return"
            or report["replay_enabled"] is not replay_enabled
            or (
                replay_enabled
                and (
                    type(report.get("replay_control")) is not str
                    or report["replay_control"] != replay_control
                    or report.get("checkpoint_clocks") is not clocks
                    or type(clocks) is not list
                    or len(clocks) != 3
                    or any(
                        actual is not bound
                        for actual, bound in zip(clocks, clock_bindings, strict=True)
                    )
                    or not _same_value(clocks, frozen_clocks)
                )
            )
            or report["action_conditioning"] != "legal_actions_v1"
            or budget.budget is not original_budget
            or budget.clock is not clock
            or budget.started_at != birth
            or type(budget.updates_completed) is not int
            or budget.updates_completed != 320
            or budget.budget.max_training_updates != 320
        ):
            raise ValueError(
                "acquired diagnostic fixed slots, checkpoints or budget authority changed"
            )

    def guard() -> None:
        check_bindings()
        for slot in slots:
            if slot["status"] == "complete" and not any(record is slot for record, _ in completed):
                completed.append((slot, deepcopy(slot)))
        check_boundary()
        budget.before_final()
        check_bindings()
        for record, frozen in completed:
            # Existing _live confirms this pending flag after checking the
            # returned episode's full learned state. All other evidence freezes.
            if frozen["state_unchanged"] is None and record["state_unchanged"] is True:
                frozen["state_unchanged"] = True
        if (
            not _same_state(states, frozen_states)
            or (source_state is not None and not _same_state(source_state, source.snapshot_state()))
            or any(not _same_value(record, frozen) for record, frozen in completed)
        ):
            raise ValueError("acquired source, checkpoint or preceding episode evidence changed")

    try:
        guard()
        source_state = deepcopy(source.snapshot_state())
        guard()
        shared._live(
            report,
            source,
            states,
            budget,
            task_sequence=("A", "B", "A", "C_train", "A"),
            declared_layouts=(shared.layouts()[0], shared.third_training_layout()),
            condition_on_legal_actions=True,
            check_boundary=guard,
        )
        guard()
        outcomes = _outcomes(slots)
        acquired = all(
            slot["success"] is True
            for slot in slots
            if slot["checkpoint_updates"] == 192 and slot["context"] == "A"
        )
        metrics = _differences(outcomes)
        interpretation = (
            "descriptive_train_only_retention_and_relearning_not_recovery_savings"
            if acquired
            else "inconclusive_initial_A_not_acquired"
        )
        # The callback can inspect/mutate the public report. Publish only after
        # its final invocation so derived claims always come from these locals.
        guard()
        report.update(
            outcomes=outcomes,
            metrics=metrics,
            initial_A_acquired=acquired,
            interpretation=interpretation,
            status="complete",
            reason=None,
        )
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error!r}")
        raise
    finally:
        report["work"] = dict(
            live_callback_attempts=sum(slot.get("prediction_calls_attempted", 0) for slot in slots),
            live_callback_returns=sum(slot.get("prediction_calls_returned", 0) for slot in slots),
            live_applied_steps=sum(slot.get("steps_executed", 0) for slot in slots),
            live_sessions_started=report["live_sessions_started"],
            failure_work_may_be_unobserved=report["status"] != "complete",
        )
