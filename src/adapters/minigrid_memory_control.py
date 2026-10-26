"""Declare the existing bounded invariant count control for the fixed assay.

This adapter supplies exact opt-in admission, its ordinary learner factory and
scalar table observations. It changes no count law, storage cap, input encoding,
stream, probe, metric, physics, persistence or scientific selection behavior.
"""

from dataclasses import asdict
from typing import Any

from src.adapters.invariant_action_memory_learner import (
    InvariantActionMemoryLearner,
    InvariantMemoryPolicy,
    copy_invariant_memory_snapshot,
)
from src.core.action_memory import ActionMemoryPolicy, ACTION_MEMORY_COUNT_DIAGNOSTIC

METHOD = "public_invariant_observation_memory"


def validate_option(
    flag,
    *,
    public_history,
    invariant_history,
    acquisition_only,
    distinct_probe,
    old_task_readouts,
    return_to_a,
    qualification_method,
    pc_qualification,
    activation_health,
    gradient_geometry,
    train_a_rollout,
) -> None:
    if type(flag) is not bool:
        raise ValueError("memory_control must be an exact bool")
    if flag and (
        any(
            value is not True
            for value in (
                public_history,
                invariant_history,
                acquisition_only,
                distinct_probe,
                old_task_readouts,
                return_to_a,
            )
        )
        or qualification_method is not None
        or any(
            value is not False
            for value in (pc_qualification, activation_health, gradient_geometry, train_a_rollout)
        )
    ):
        raise ValueError(
            "memory_control requires both history modes and A-only/distinct-probe/old-readouts/return; no neural selector, health, gradient or rollout"
        )


def policy() -> InvariantMemoryPolicy:
    return InvariantMemoryPolicy(
        ActionMemoryPolicy(feature_dim=296, num_classes=7, max_entries=64, max_payload_bytes=155136)
    )


def learner() -> InvariantActionMemoryLearner:
    return InvariantActionMemoryLearner(policy())


def table_observation(state) -> dict[str, Any]:
    owned = copy_invariant_memory_snapshot(state)
    if owned.policy != policy():
        raise ValueError("memory control requires the fixed invariant exact-count policy")
    entries = owned.native_state.entries
    return dict(
        entries=len(entries),
        logical_payload_bytes=len(entries) * 2424,
        conflicting_keys=sum(sum(count > 0 for count in entry.counts) > 1 for entry in entries),
        maximum_entries=64,
        maximum_payload_bytes=155136,
        scope="owned complete checkpoint/source table only; logical retained payload excludes Python/copies/validation/projection and RSS; no disposable endpoint observation",
    )


def describe(protocol: dict[str, Any]) -> None:
    """Replace neural-only template metadata; all doses and data scopes persist."""
    for key in (
        "hidden_dim",
        "learning_rate",
        "momentum",
        "inference_steps",
        "inference_learning_rate",
        "trainable_parameters",
        "cpc_configs",
    ):
        protocol.pop(key, None)
    protocol.update(
        protocol_id="minigrid_invariant_memory_control_A_only_distinct_probe_old_task_readouts_a_return_v1",
        methods=[METHOD],
        feature_dim_by_method={METHOD: 296},
        primary_comparisons=[],
        seed_scope="17 binds public teacher/sampler data only; exact count learner has no model RNG",
        training="six original TRAIN traces collected once; only A blue/red copied to128 alternating delayed1 packets; bounded exact action-count increments, no optimizer/SGD",
        evaluation="one A128 count stream completes before fixed F collection; all six TRAIN fits, isolated F16/old-A readouts/A16 returns from0/128; no live/final evaluation",
        memory_control=dict(
            policy=asdict(policy()),
            update_definition=ACTION_MEMORY_COUNT_DIAGNOSTIC,
            entry_payload_bytes=2424,
            maximum_entries=64,
            maximum_payload_bytes=155136,
            unseen_prediction="uniform seven classes; known key normalized cumulative action counts",
            native_updates_are_sgd=False,
            storage_scope="logical retained table only; Python/state/copies/projection/reports/RSS additional",
        ),
        limitations="fixed bounded exact-memory fit control; A/F/return teacher fit is not useful gameplay, generalization or isolated plasticity; same exposure/cap is not equal computation/total memory to neural methods; all negative/incomplete results retained",
    )
    history = protocol["public_history"]
    history.pop("parameters_by_method", None)
    history.pop("original_feedforward_parameters", None)
    history["computation"] = (
        "canonical public projection plus atomic count increments/lookups; no native Torch, optimizer or gradients; validation/copies/time/RSS additional"
    )
    protocol["probe"]["isolation"] = (
        "owned disposable count forks; probe/return updates never enter benchmark/source/actor/checkpoints"
    )
    protocol["probe"]["comparison"] = protocol["probe"]["comparison"].replace(
        "starting fit/optimizer", "starting table/count-exposure"
    )
    protocol["return_phase"]["scope"] = (
        "same disposable count learner after F16; admitted A TRAIN templates and distinct16-update budget; readonly monitors unchanged"
    )
    for section in ("return_phase", "old_task_readouts"):
        protocol[section]["input_array_bytes_by_method"] = {METHOD: 42768}
    protocol["old_task_readouts"]["direction"] = (
        "CE gain=CE0-CE16; accuracy/exact gain=16-0; negative means worse old fit; both-cue acquisition premise unchanged"
    )
    protocol["planned_work"].update(
        count_update_calls=192, count_training_rows=1728, neural_sgd_updates=0
    )
    protocol["limits"].pop("interop_threads", None)
