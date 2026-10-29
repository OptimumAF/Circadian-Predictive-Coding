"""Play one known TRAIN game through an actual published analytical actor.

The existing episode kernel owns observations, decisions, physics and cleanup.
This adapter admits one caller-named episode and original zero-update budget;
it neither trains, publishes, rolls back, chooses a policy nor opens evaluation.
"""

from dataclasses import replace

from src.adapters.minigrid_context_linear_learner import copy_context_linear_training_snapshot
from src.adapters.minigrid_policy_evaluation import _run_checkpoint, MiniGridEvaluationFailed
from src.adapters.promotable_minigrid_binding import bind_promotable_minigrid
from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import budget_binding
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.minigrid_key_door import MiniGridKeyDoorLayout
from src.core.minigrid_context_encoding import CONTEXT_HISTORY_ENCODING_ID


def run_promotable_minigrid_episode(
    actor, *, cue, episode_id, budget, check_boundary, layout=None, shared_runtime=None
):
    if type(cue) is not str or cue not in ("blue", "red") or not callable(check_boundary):
        raise ValueError("published MiniGrid requires a native cue and boundary callback")
    owned = MiniGridKeyDoorLayout("A") if layout is None else layout
    if type(owned) is not MiniGridKeyDoorLayout or owned not in (
        MiniGridKeyDoorLayout("A"),
        MiniGridKeyDoorLayout("A", branch_row=4),
    ):
        raise ValueError("published MiniGrid supports only declared A3/A4 TRAIN layouts")
    owned = replace(owned)
    if type(budget) is not ToyBudgetSession or type(budget.budget) is not ToyExecutionBudget:
        raise ValueError("published MiniGrid requires an original exact readout budget")
    limits, pin = replace(budget.budget), budget_binding(budget)

    def guard():
        if (
            budget_binding(budget) != pin
            or budget.budget != limits
            or limits.max_training_updates != 0
            or limits.max_replay_examples != 0
            or limits.max_wall_seconds != 150
            or limits.max_process_rss_bytes != 512 * 1024**2
            or budget.updates_completed != 0
            or budget.replay_examples_completed != 0
        ):
            raise ValueError("published MiniGrid original zero-update/readout budget changed")

    guard()
    budget.before_final()
    check_boundary()
    guard()
    reader = bind_promotable_minigrid(actor, episode_id=episode_id, shared_runtime=shared_runtime)
    state = reader.state
    try:
        result = _run_checkpoint(
            reader,
            state,
            method="arrived_public_cue_affine_fit",
            updates=state.completed_updates,
            layouts=(),
            role="knownTRAIN_published_affine_actor",
            sources=(),
            budget=budget,
            check_boundary=check_boundary,
            public_history=True,
            memory_copier=copy_context_linear_training_snapshot,
            state_equal=_same_value,
            extra_check=guard,
            live_plan=((owned, cue, "train"),),
            representation_encoding=CONTEXT_HISTORY_ENCODING_ID,
            representation_feature_dim=298,
            serving_binding=reader,
            return_metadata=dict(
                policy_source="actual_promotable_actor",
                actor_version=reader.snapshot.actor_version,
                serving_generation=reader.snapshot.generation,
                new_fit_calls=0,
                new_optimizer_updates=0,
            ),
        )
    except MiniGridEvaluationFailed as error:
        reader.stopped = True
        error.report["serving"] = reader.report()
        raise
    result["serving"] = reader.report()
    return result
