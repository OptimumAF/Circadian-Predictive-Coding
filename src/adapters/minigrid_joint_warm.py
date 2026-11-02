"""Connect the current joint TRAIN endpoint to the existing reward pilot.

Inputs are the live guarded joint source/checkpoint and original TRAIN packets.
No source training, historical checkpoint loader, optimizer or runner lives here.
"""

from copy import deepcopy

from src.adapters import minigrid_public_cue_recurrent_learner as public
from src.adapters.minigrid_public_cue_reward import (
    PublicCueRewardLearner,
    _binding,
    _same_binding,
    _state_refs,
)
from src.adapters.minigrid_reward_pilot import run_pilot
from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import paid_report_guard


class JointWarmFailed(RuntimeError):
    def __init__(self, report, error):
        self.report = deepcopy(report)
        super().__init__(str(error))


def run_joint_reward(source, state, sources, *, check_boundary, memory):
    if type(source) is not public.PublicCueRecurrentLearner or not callable(check_boundary):
        raise ValueError("joint reward requires the actual guarded public-cue source")
    before = public.copy_snapshot(state)
    if (
        before.policy != public.PublicCueRecurrentPolicy()
        or before.failed_work is not None
        or (before.completed_updates, before.training_examples) != (128, 1152)
        or any((h.completed_updates, h.training_examples) != (64, 576) for h in before.heads)
    ):
        raise ValueError("joint reward requires the original complete128/1152 warm endpoint")
    source_before = source.snapshot_state()
    if source_before.completed_updates != 0 or source_before.training_examples != 0:
        raise ValueError("joint benchmark source must remain cold and separate")
    source_binding, state_binding = _binding(source), _state_refs(state)
    used = False

    def pure_proof():
        if (
            not _same_binding(source, source_binding)
            or not all(a is b for a, b in zip(_state_refs(state), state_binding, strict=True))
            or not _same_value(state, before)
            or not _same_value(source._state, source_before)
        ):
            raise ValueError("original paid joint endpoint/source changed during reward work")

    def proof():
        # Read validated owned values directly around the parent's callback.
        # A snapshot returning old values must not hide its own live mutation.
        pure_proof()
        check_boundary()
        pure_proof()

    def prepare(report, current_memory, deadline, boundary):
        nonlocal used
        proof()
        if used or type(report) is not dict or report:
            raise ValueError("joint warm handoff is single-use and requires its empty report")
        used = True
        boundary()
        proof()
        warm = PublicCueRewardLearner(before.policy)
        proof()
        warm.restore_state(before)
        proof()
        warm_owner = warm._owner()
        warm_binding = _binding(warm_owner)

        def warm_proof():
            if (
                warm._owner() is not warm_owner
                or not _same_binding(warm_owner, warm_binding)
                or not _same_value(warm_owner._state, before)
            ):
                raise ValueError("handed-off joint warm owner/full state changed")

        if not _same_value(warm.snapshot_state(), before):
            raise ValueError("joint reward handoff lost full native optimizer/gradient state")
        warm_proof()
        report.update(
            status="complete",
            definition="current_run_joint128_warm_handoff_v1",
            completed_updates=128,
            training_examples=1152,
            full_state_restored=True,
            source_reference_unchanged=True,
            # Source128 was paid by the enclosing joint stream, not refitted here.
            work=dict(charged_updates=0, observed_native_training_rows=0),
        )
        report_check, _ = paid_report_guard(report)

        def guard():
            report_check()
            warm_proof()
            proof()
            boundary()
            proof()
            warm_proof()
            report_check()

        guard()
        return dict(source=warm, guard=guard, updates=128, rows=1152, sources=sources)

    proof()
    result = run_pilot(
        warm_start=True,
        train_readouts=True,
        momentum_control=True,
        extended_control_readouts=True,
        joint_start=prepare,
    )
    result_check, paid_result = paid_report_guard(result)
    try:
        proof()
        result_check()
    except Exception as error:
        raise JointWarmFailed(paid_result, error) from error
    # Account lineage separately from the pilot's new reward/control work.
    result["joint_source_lineage"] = dict(updates=128, training_rows=1152)
    result["whole_committed_native_updates"] = 128 + result.get("total_new_native_updates", 0)
    result["lineage_accounting"] = (
        "commits only; attempted/unknown work remains in original budgets/slots"
    )
    result["whole_training_rows"] = 1152 + result.get("total_new_training_rows", 0)
    return result
