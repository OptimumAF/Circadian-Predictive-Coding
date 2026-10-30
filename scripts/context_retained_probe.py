"""Compose the existing isolated LS probe with a completed retained donor.

Inputs are the original live runtime/spec/rehearsal/boundary/sampler and paid
report. No loader, new update loop, budget renewal, promotion or final data.
"""

from copy import deepcopy

from scripts.context_head_probe import build_head_cohorts
from scripts.probe_ticket_policy_learning import run_arm, TRANSFER_HEAD_TASKS
from scripts.run_bp_ticket_serving import _followup_actor_values
from src.adapters.context_linear_fit_learner import ContextLinearFitLearner
from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import budget_binding
from src.core.context_linear_state import ContextLinearFitSnapshot


def run_probe(
    runtime, spec, rehearsal, boundary, memory, paid_report, record, *, rehearsal_control=None
):
    """Use a detached donor; keep the completed runtime and its evidence fixed."""
    if rehearsal_control is not None and (
        type(rehearsal_control) is not str or rehearsal_control not in ("joint", "current_only")
    ):
        raise ValueError("unknown retained probe rehearsal control")
    boundary()
    candidate = runtime.candidate_snapshot()
    if (
        type(candidate.state) is not ContextLinearFitSnapshot
        or (candidate.state.completed_updates, candidate.state.training_examples) != (23, 144)
        or (
            candidate.wake_updates,
            candidate.consolidation_attempts,
            candidate.consolidations_completed,
        )
        != (6, 5, 5)
        or len(runtime.applied_updates) != 6
        or len(runtime.applied_consolidations) != 5
        or runtime.acknowledged_experiences
        or rehearsal._runtime is not runtime
    ):
        raise ValueError("isolated retained probe requires the completed original23/144 donor")
    evidence_keys = (
        "native_calls",
        "training_work",
        "joint_rehearsal",
        "candidate_return_readout",
        "candidate_state",
        "action_snapshot",
        "sharing_snapshot",
        "stream",
    )
    evidence_owners = tuple(paid_report[key] for key in evidence_keys)
    evidence = deepcopy(evidence_owners)
    record.update(
        definition="retained_context_linear_transfer_probe_v1",
        donor_native_updates=23,
        donor_native_rows=144,
        source_origin="current_owned_retained_candidate",
        scope="fixed four old/two selected probe_test tasks; legacy retention outside this scope unverified",
        donor_unchanged=None,
        retained_scope_verified=None,
        arm={},
    )
    if rehearsal_control is not None:
        record["joint_rehearsal_control"] = rehearsal_control
    arm_record = record["arm"]
    header = deepcopy({key: value for key, value in record.items() if key != "arm"})

    def scope():
        def read(current):
            cursor = runtime._inbox._borrow_cursor()
            return (
                tuple(
                    id(x)
                    for x in (
                        runtime,
                        runtime.actor,
                        runtime._candidate,
                        runtime._inbox,
                        runtime._inbox._clock,
                        runtime._budget,
                        rehearsal,
                        runtime._inbox._clock,
                        runtime._inbox._budget,
                        runtime._inbox._learner,
                    )
                ),
                current,
                cursor,
                tuple((id(x), id(x.features)) for x in cursor.experiences),
                tuple((id(x), id(x.targets)) for x in cursor.labels),
                runtime._consolidations,
                tuple(id(x) for x in (*runtime._inbox.applied_updates, *runtime._consolidations)),
                deepcopy(vars(runtime._inbox._clock)),
                _followup_actor_values(runtime.actor.serving_snapshot()),
                tuple(
                    (
                        budget_binding(s),
                        deepcopy(s.budget),
                        s.updates_completed,
                        s.replay_examples_completed,
                    )
                    for s in (runtime._budget, rehearsal._replay_budget)
                ),
                rehearsal.report,
            )

        return runtime.with_candidate_snapshot(read)

    original = deepcopy(scope())

    def guard():
        boundary()
        if (
            not _same_value(scope(), original)
            or paid_report.get("isolated_transfer_probe") is not record
            or record.get("arm") is not arm_record
            or not _same_value({key: record.get(key) for key in header}, header)
            or any(
                paid_report.get(key) is not owner
                for key, owner in zip(evidence_keys, evidence_owners, strict=True)
            )
            or not _same_value(tuple(paid_report[key] for key in evidence_keys), evidence)
        ):
            paid_report["isolated_transfer_probe"] = record
            record["arm"] = arm_record
            record.update(deepcopy(header))
            raise ValueError(
                "retained donor/runtime/actor/budget/preceding evidence changed during isolated probe"
            )

    guard()
    # Why: serving's CountedLearner hooks require a live worker timeline. A plain
    # existing LS owner avoids those retired hooks; probe itself supplies costs.
    donor = ContextLinearFitLearner(candidate.state.policy)
    donor.restore_state(candidate.state)
    joint_options = {}
    if rehearsal_control is None:
        cohorts = build_head_cohorts(spec, head_task_transfer=True)
    else:
        from scripts.context_head_rehearsal import prepare_cohorts

        cohorts, bank = prepare_cohorts(spec)
        joint_options = dict(joint_rehearsal=rehearsal_control, training_bank=bank)
    try:
        run_arm(
            donor,
            candidate.state,
            cohorts,
            spec,
            guard,
            memory,
            "continued",
            row=arm_record,
            separate_context_heads=True,
            linear_fit=True,
            head_tasks=TRANSFER_HEAD_TASKS,
            **joint_options,
        )
        guard()
        if donor.snapshot_state() != candidate.state:
            raise ValueError("isolated transfer changed its owned donor")
        record["retained_scope_verified"] = True
    finally:
        record["donor_unchanged"] = donor.snapshot_state() == candidate.state
        record["arm"] = arm_record
        phases = arm_record.get("probe", {})
        returned = phases.get("return_phase", {})
        record["private_work"] = {
            key: sum(p[key] for p in (phases, returned))
            if all(type(p.get(key)) is int for p in (phases, returned))
            else None
            for key in (
                "attempted_updates",
                "completed_updates",
                "attempted_training_examples",
                "completed_training_examples",
            )
        }
        record["private_work"]["unreturned_work_unknown"] = any(
            value is None for value in record["private_work"].values()
        )
        if rehearsal_control is not None:
            work = record["private_work"]
            replay = phases.get("rehearsal")
            record["joint_replay_charged_exposures"] = (
                replay.get("charged_exposures") if isinstance(replay, dict) else None
            )
            complete = (
                isinstance(replay, dict)
                and replay.get("attempted_updates") == 16
                and replay.get("returned_updates") == replay.get("committed_updates") == 16
                and replay.get("charged_exposures") == 144
                and len(replay.get("events", ())) == 16
                and all(e.get("status") == "complete" for e in replay.get("events", ()))
            )
            # Why: missing/failed replay readback cannot be reconstructed as zero
            # physical work. Its permanent charges remain in the original report.
            for key in (
                "attempted_updates",
                "completed_updates",
                "attempted_training_examples",
                "completed_training_examples",
            ):
                work[key] = (
                    work[key] + (16 if "updates" in key else 144)
                    if complete and type(work[key]) is int
                    else None
                )
            work["unreturned_work_unknown"] = any(
                work[k] is None
                for k in (
                    "attempted_updates",
                    "completed_updates",
                    "attempted_training_examples",
                    "completed_training_examples",
                )
            )
