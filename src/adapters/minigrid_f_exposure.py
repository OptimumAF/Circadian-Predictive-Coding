"""Declare an explicit F-only exposure control for the existing isolated pilot.

Inputs are original H32 pilot flags and protocol metadata. Outputs are admission
errors and a fixed cost/checkpoint description. No collection, learner, updates,
scoring, budget renewal, selection or persistence belongs here.
"""

from typing import Any

POINTS = (0, 12, 24, 48)
METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


def validate_option(flag: bool, **options: Any) -> None:
    """Keep this control distinct from every original or replay experiment."""
    if type(flag) is not bool:
        raise ValueError("extra_f_exposure must be an exact bool")
    if not flag:
        return
    required = (
        "public_history",
        "invariant_history",
        "joint_catalogue",
        "distinct_probe",
        "old_task_readouts",
        "return_to_a",
        "wide_neural",
    )
    forbidden = (
        "old_a_rehearsal",
        "probe_rollouts",
        "deferred_probe_rollouts",
        "extended_wide_probe_budget",
        "activation_health",
        "gradient_geometry",
        "prior_gradient_geometry",
        "pc_qualification",
        "acquisition_only",
        "memory_control",
        "weighted_bp",
        "weighted_pc",
        "weighted_catalogue",
        "train_action_margins",
        "pc_update_diagnostics",
        "bp_update_diagnostics",
        "train_fit_bounds",
        "cue_relative_history",
        "train_a_rollout",
    )
    if (
        type(options.get("qualification_method")) is not str
        or options["qualification_method"] not in METHODS
        or any(options.get(name) is not True for name in required)
        or any(options.get(name, False) is not False for name in forbidden)
        or options.get("probe_inference") is not None
    ):
        raise ValueError(
            "extra_f_exposure requires the explicit original native H32 joint "
            "F48/old-A/A16 route without replay, games or extra diagnostics"
        )


def describe(protocol: dict[str, Any]) -> None:
    """Match native calls/rows while preserving the original F16 reports."""
    if (
        protocol["probe"]["source_updates"] != [0, 128]
        or protocol["probe"]["updates"] != 16
        or protocol["probe"]["checkpoint_updates"] != [0, 4, 8, 16]
        or protocol["return_phase"]["updates"] != 16
        or len(protocol["methods"]) != 1
        or protocol["methods"][0] not in tuple(f"public_invariant_{m}" for m in METHODS)
    ):
        raise ValueError("F48 description requires the original two-probe return recipe")
    protocol["protocol_id"] = protocol["protocol_id"].removesuffix("_v1") + "_f48_exposure_v1"
    protocol["probe"].update(
        updates=48,
        checkpoint_updates=list(POINTS),
        training="three passes over the original16 owned alternating-cue probe_train packets; no additional input arrays or role conversion",
        comparison="CE gain=mean CE0-CE48; accuracy gain=accuracy48-accuracy0; native-count-aligned exposure control, no replay bank or selection",
    )
    protocol["extra_f_exposure"] = dict(
        input_training_batches=16,
        training_repetitions=3,
        updates_per_probe=48,
        training_rows_per_probe=432,
        checkpoint_updates=list(POINTS),
        replay_logical_checkpoint_updates=[0, 4, 8, 16],
        replay_rows=0,
        bank_encoded_bytes=0,
        scope="same native calls/rows and score counts as F16 plus32 old-A replay calls; validation/projection/bank/copy/process costs differ and remain explicit; not matched total cost or evidence of acquired retention/recovery",
    )
    for key in ("scope", "evaluation"):
        protocol["return_phase"][key] = protocol["return_phase"][key].replace("F16", "F48")
    protocol["joint_catalogue"]["isolation"] = protocol["joint_catalogue"]["isolation"].replace(
        "F16", "F48"
    )
    protocol["old_task_readouts"]["checkpoint_updates"] = list(POINTS)
    protocol["old_task_readouts"]["direction"] = (
        "CE gain=CE0-CE48; accuracy/exact gain=48-0; negative means worse old fit; "
        "no acquired-task retention claim"
    )
    work = protocol["planned_work"]
    work["probe_native_train_calls"] += 64
    work["probe_training_rows"] += 576
    work["native_train_calls"] += 64
    work["training_rows"] += 576
    protocol["limitations"] += (
        "; explicit F48-only extra-exposure control: original16 arrays repeated3times, "
        "no old-A replay; all native/diagnostic/memory/process costs reported separately"
    )
