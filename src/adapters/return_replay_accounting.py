"""Fixed return-recipe replay readbacks; no training or policy selection.

Inputs are the existing helper, its original budget, observed wake count and
native state. Outputs bind both clocks to complete event/commit evidence. This
adapter does not replace StreamRehearsal's admission or failed-work accounting.
"""

from dataclasses import asdict
from typing import Any

from src.adapters.torch_recurrent_learner import TORCH_RECURRENT_CE_ID


def check_replay_progress(
    helper: Any,
    replay_budget: Any,
    count: int,
    state: Any,
    *,
    training_sources: dict[str, tuple[int, dict[str, Any]]],
    total_updates: int = 192,
) -> dict[str, int]:
    """Require each scheduled event before this successful checkpoint to commit."""
    if type(total_updates) is not int or total_updates not in (192, 256, 320):
        raise ValueError("return replay total_updates must be a native integer 192 or 256 or 320")
    if helper._replay_budget is not replay_budget:
        raise ValueError("return replay requires its original replay budget")
    replay_budget.before_final()
    report = helper.report
    events = report["events"]
    committed = count // 2
    expected_ids = [f"rehearsal-{n}" for n in range(2, count + 1, 2)]
    receipts = (
        []
        if helper._runtime is None
        else [asdict(r) for r in helper._runtime.applied_consolidations]
    )
    if (
        type(count) is not int
        or not 0 <= count <= total_updates
        or not report["bound"]
        or report["stopped"]
        or report["wake_updates_observed"] != count
        or report["total_updates"] != total_updates
        or len(events) != total_updates // 2
        or [e["updates"] for e in events] != list(range(2, total_updates + 1, 2))
        or [e["event_id"] for e in events[:committed]] != expected_ids
        or [r["event_id"] for r in receipts] != expected_ids
        or replay_budget.updates_completed != 0
        or replay_budget.budget.max_replay_examples != 8 * (total_updates // 2)
        or replay_budget.replay_examples_completed != 8 * committed
        or any(
            report[k] != committed
            for k in ("attempted_updates", "returned_updates", "committed_updates")
        )
        or report["charged_exposures"] != 8 * committed
        or report["replay_budget_charged_exposures"] != 8 * committed
    ):
        raise ValueError("return replay clocks, charges or consolidation receipts differ")
    for attempt, (event, receipt) in enumerate(zip(events[:committed], receipts, strict=True), 1):
        if (
            event["status"] != "complete"
            or event["reason"] is not None
            or event["receipt"] != receipt
            or event["charged_exposures"] != 8
            or any(
                event[k] != 1
                for k in ("attempted_updates", "returned_updates", "committed_updates")
            )
            or len(event["selected_ids"]) != 1
            or len(event["selected_episodes"]) != 1
            or len(event["diagnostics"]) != 1
            or event["diagnostics"][0] != receipt["diagnostic"]
            or receipt["attempt_number"] != attempt
            or receipt["actor_version"] != helper._runtime._base_actor_version
            or receipt["learner_version"] != helper._runtime._candidate_version
            or receipt["diagnostic"]["definition"] != TORCH_RECURRENT_CE_ID
        ):
            raise ValueError("return replay event differs from its complete native commit")
        episode = event["selected_episodes"][0]
        original = training_sources.get(episode["episode_id"])
        if original is None or original[0] > event["updates"] or original[1] != episode:
            raise ValueError("return replay selected episode differs from released training data")
    for event in events[committed:]:
        if (
            event["status"] != "not_run"
            or event["receipt"] is not None
            or any(
                event[k] != 0
                for k in (
                    "charged_exposures",
                    "attempted_updates",
                    "returned_updates",
                    "committed_updates",
                )
            )
        ):
            raise ValueError("return replay contains unobserved future event work")
    native = count + committed
    if (
        type(state.completed_updates) is not int
        or state.completed_updates != native
        or type(state.training_examples) is not int
        or state.training_examples != 8 * native
    ):
        raise ValueError("return replay native progression differs from wake plus replay commits")
    return dict(
        wake_updates=count,
        replay_committed_updates=committed,
        replay_charged_examples=8 * committed,
        native_updates=native,
        native_training_examples=8 * native,
    )


def replay_evidence(helper: Any, replay_budget: Any) -> dict[str, Any]:
    """Detach successful or partial costs without zero-imputing failed native work."""
    if helper._replay_budget is not replay_budget:
        raise ValueError("return replay lost its original budget")
    return dict(
        rehearsal=helper.report,
        final_retention=asdict(helper.retention),
        charged_examples=replay_budget.replay_examples_completed,
        applied_consolidations=[]
        if helper._runtime is None
        else [asdict(r) for r in helper._runtime.applied_consolidations],
        storage_scope="bounded replay array payload; event records/copies/metadata/native states included only in process RSS and total time",
        partial_work_note="attempted exposures remain permanently charged; failed native work may be unobserved, never zero-imputed",
    )
