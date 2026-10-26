"""Fixed public cue-relative MiniGrid recipe and complete raw-history admission.

Inputs are explicit method IDs, native checkpoints, public protocol metadata and
complete raw296 episodes. Outputs are owned recipe/snapshot records or validated
raw copies. No training, role authority, budgets, task selection, IO or runner.
"""

from dataclasses import fields
from typing import Any

from src.adapters.cue_relative_torch_feature_learner import (
    CueRelativeTorchFeaturePolicy,
    CueRelativeTorchFeatureSnapshot,
    copy_cue_relative_torch_feature_snapshot,
)
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    copy_episode,
)
from src.core.minigrid_cue_relative_encoding import (
    CUE_RELATIVE_HISTORY_ENCODING_ID,
    cue_relative_public_history,
)

NATIVE_METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
METHODS = tuple(f"public_cue_relative_{method}" for method in NATIVE_METHODS)
POINTS = (0, 32, 64, 96, 128)


def policy(method: str) -> CueRelativeTorchFeaturePolicy:
    """Bind the full declared public ID to the unchanged original native recipe."""
    if type(method) is not str or method not in METHODS:
        raise ValueError("cue-relative policy requires its declared public native method")
    native = method.removeprefix("public_cue_relative_")
    return CueRelativeTorchFeaturePolicy(TorchFeaturePolicy(native, 296, 8, 17, num_classes=7))


def snapshot(
    state: Any, point: int, method: str, *, catalogue_rows: int | None = None
) -> CueRelativeTorchFeatureSnapshot:
    """Detach complete state, requiring exact family, policy and exposure clocks."""
    if type(point) is not int or (
        point not in POINTS if catalogue_rows is None else not 0 <= point <= 128
    ):
        raise ValueError("cue-relative checkpoint must follow its declared native points")
    if catalogue_rows is not None and (type(catalogue_rows) is not int or catalogue_rows != 54):
        raise ValueError("cue-relative catalogue requires exact54-row wakes")
    owned = copy_cue_relative_torch_feature_snapshot(state)
    if owned.policy != policy(method):
        raise ValueError("cue-relative snapshot differs from its fixed native policy")
    rows = 9 if catalogue_rows is None else 54
    if (
        type(owned.completed_updates) is not int
        or owned.completed_updates != point
        or type(owned.training_examples) is not int
        or owned.training_examples != rows * point
    ):
        raise ValueError("cue-relative native update/row clocks differ from declared exposure")
    return owned


def validate_histories(
    episodes: tuple[DemonstrationEpisode, ...],
) -> tuple[DemonstrationEpisode, ...]:
    """Admit complete causal raw histories, preserving role without authorizing it.

    Why: the learner validates every current row, but unused next frames and
    within-episode initial-frame consistency must be admitted before native work.
    Projection here is validation only: returned copies remain raw so a red cue
    is never accidentally projected twice at the learner boundary.
    """
    if type(episodes) is not tuple or not 1 <= len(episodes) <= 512:
        raise ValueError("cue-relative admission requires1..512 complete raw episodes")
    owned = []
    for episode in episodes:
        if (
            type(episode) is not DemonstrationEpisode
            or set(vars(episode)) != {field.name for field in fields(DemonstrationEpisode)}
            or type(episode.transitions) is not tuple
            or any(
                type(step) is not DemonstrationTransition
                or set(vars(step)) != {field.name for field in fields(DemonstrationTransition)}
                for step in episode.transitions
            )
        ):
            raise ValueError("cue-relative admission requires exact complete episode records")
        copied = copy_episode(episode)
        if copied.num_classes != 7 or len(copied.transitions[0].observation) != 296:
            raise ValueError("cue-relative admission requires raw296/K7 histories")
        initial = copied.transitions[0].observation[:148]
        if copied.transitions[0].observation[148:] != initial:
            raise ValueError("cue-relative initial history must equal its first public frame")
        for step in copied.transitions:
            if step.observation[:148] != initial or step.next_observation[:148] != initial:
                raise ValueError("cue-relative causal initial public frame changed within episode")
            cue_relative_public_history(step.observation)
            cue_relative_public_history(step.next_observation)
        owned.append(copied)
    return tuple(owned)


def describe(protocol: dict[str, Any], native_method: str) -> None:
    """Name this encoding explicitly while retaining old history transport keys."""
    if type(native_method) is not str or native_method not in NATIVE_METHODS:
        raise ValueError("cue-relative description requires one original native method")
    old_method = f"public_invariant_{native_method}"
    method = f"public_cue_relative_{native_method}"
    if protocol.get("methods") != [old_method]:
        raise ValueError("cue-relative description requires its sole declared native method")
    protocol["protocol_id"] = protocol["protocol_id"].removesuffix("_v1") + "_cue_relative_v1"
    protocol.update(
        methods=[method],
        feature_dim_by_method={method: 296},
        trainable_parameters={method: 2439},
        cue_relative_history=dict(
            encoding_id=CUE_RELATIVE_HISTORY_ENCODING_ID,
            native_method=native_method,
            method=method,
            feature_inputs="raw296 own initial/current public frames; cue projection exactly once inside bound learner; no projected transport",
            cue="exactly one initial Ball6 red0/blue2; initial cue retained; current Ball never chooses cue",
            keys="every Key5 red0/blue2 becomes match2/other0 in both frames; other task coordinates retained after original wall/direction normalization",
            support="original canonical equality/support counts valid only after full causal cue/key admission: retained initial cue makes key permutation bijective on this admitted domain",
            scope="explicit distinct encoding/complete snapshot family; native objectives, seed, hidden width, dose, labels and metrics unchanged; no benefit assumed",
        ),
    )
    protocol["invariant_history"].update(
        encoding_id=CUE_RELATIVE_HISTORY_ENCODING_ID,
        feature_inputs="raw296; cue-relative key projection exactly once inside bound learner",
        legacy_transport_key=True,
    )
    protocol["public_history"]["parameters_by_method"] = {method: 2439}
    for name in ("old_task_readouts", "return_phase"):
        metadata = protocol[name]
        previous = metadata["input_array_bytes_by_method"]
        metadata["input_array_bytes_by_method"] = {method: previous[old_method]}
    protocol["probe_support"]["representation_contract"] = (
        "measured original canonical keys; equality/support counts also describe cue-relative "
        "inputs only on the fully admitted bijective domain; old projection work is unchanged"
    )
    protocol["limitations"] += (
        "; cue-relative key encoding is a distinct public recipe; original cue retained; "
        "canonical support counts require full admitted bijective domain; no acquisition, "
        "retention, recovery or circadian advantage established"
    )
