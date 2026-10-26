"""Public MiniGrid fixed old-A replay option, protocol and observed cost reducer.

Inputs are explicit original invariant pilot flags and bounded probe work DTOs.
Outputs are protocol metadata and plain observed costs. No learner construction,
collection, updates, evaluation, budget renewal, selection or IO belongs here.
"""

from typing import Any

METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
BANK_TRANSITIONS = 18
BANK_ENCODED_BYTES = 87280
VALIDATION_PROJECTION_ROWS = 36
VALIDATION_PROJECTED_BYTES = 85248
REPLAY_CALLS_PER_PROBE = 32
REPLAY_ROWS_PER_PROBE = 288


def validate_option(flag: bool, **options: Any) -> None:
    """Admit only the declared original joint/F16/old-A/A16 composition."""
    if type(flag) is not bool:
        raise ValueError("old_a_rehearsal must be an exact bool")
    if not flag:
        return
    required = (
        "public_history",
        "invariant_history",
        "joint_catalogue",
        "distinct_probe",
        "old_task_readouts",
        "return_to_a",
    )
    forbidden = (
        "cue_relative_history",
        "pc_qualification",
        "acquisition_only",
        "memory_control",
        "weighted_bp",
        "weighted_pc",
        "weighted_catalogue",
        "activation_health",
        "gradient_geometry",
        "prior_gradient_geometry",
        "train_a_rollout",
        "probe_rollouts",
        "deferred_probe_rollouts",
        "train_action_margins",
        "pc_update_diagnostics",
    )
    if (
        type(options.get("wide_neural", False)) is not bool
        or type(options.get("qualification_method")) is not str
        or options["qualification_method"] not in METHODS
        or any(options.get(name) is not True for name in required)
        or any(options.get(name, False) is not False for name in forbidden)
        or options.get("probe_inference") is not None
    ):
        raise ValueError(
            "old_a_rehearsal requires an explicit original native joint invariant "
            "F16/old-A/A16 route without weighting, other encodings or extra diagnostics"
        )


def describe(protocol: dict[str, Any], *, wide_neural: bool = False) -> None:
    """Declare additional replay work while preserving the original controls."""
    if type(wide_neural) is not bool:
        raise ValueError("wide_neural must be an exact bool")
    methods = protocol.get("methods")
    if (
        type(methods) is not list
        or len(methods) != 1
        or methods[0] not in tuple(f"public_invariant_{method}" for method in METHODS)
        or protocol["probe"]["source_updates"] != [0, 128]
        or protocol["probe"]["updates"] != 16
        or protocol["return_phase"]["updates"] != 16
        or "joint_catalogue" not in protocol
    ):
        raise ValueError("old-A replay description requires the original two-probe joint recipe")
    protocol["protocol_id"] = protocol["protocol_id"].removesuffix("_v1") + "_old_a_rehearsal_v1"
    protocol["old_a_rehearsal"] = dict(
        control="fixed_original_a_train_bank_v1",
        role="train",
        contexts=["A"],
        cues=["blue", "red"],
        source_updates=[0, 128],
        every_f_updates=1,
        episodes_per_event=2,
        native_calls_per_probe=REPLAY_CALLS_PER_PROBE,
        replay_rows_per_probe=REPLAY_ROWS_PER_PROBE,
        bank_transitions_per_probe=BANK_TRANSITIONS,
        bank_encoded_bytes_per_probe=BANK_ENCODED_BYTES,
        validation_projection_rows_per_probe=VALIDATION_PROJECTION_ROWS,
        validation_projected_array_equivalent_bytes_per_probe=VALIDATION_PROJECTED_BYTES,
        selection="both original A-blue/A-red complete TRAIN episodes after every F wake; fixed membership and order; F never enters the bank",
        isolation="existing disposable F stream; probe_train wakes retain their role; old-A bank stays TRAIN; benchmark/source/actor/final data unchanged",
        budget="fresh separate replay allowance288 rows per probe, permanently charged before native calls; original clocks and limits are not renewed",
        return_phase="same disposable candidate and optimizer after F16; original A16 return with no further replay",
        storage="two fixed encoded episodes per probe; logical bank bytes exclude Python metadata, source/validation/native/selected-event/capture copies; actual RSS and wall time additional",
        comparison="extra old-A exposure and computation versus original no-replay control; not matched dose/compute or evidence of acquisition, retention, recovery or circadian advantage",
    )
    if wide_neural:
        # Why: the facade promotes protocol width after composing replay metadata.
        # Declare the intended native recipe here instead of inferring that order.
        protocol["old_a_rehearsal"]["native_recipe"] = dict(
            method=methods[0].removeprefix("public_invariant_"),
            feature_dim=296,
            hidden_dim=32,
            seed=17,
            num_classes=7,
            learning_rate=0.03,
            inference_steps=2,
            inference_learning_rate=0.04,
            momentum=0.9,
        )
    protocol["old_task_readouts"]["updates"] = (
        "F probe_train wakes plus separately charged fixed old-A TRAIN bank replay; "
        "original old-A monitor objects remain readonly; same-candidate A16 return follows"
    )
    work = protocol["planned_work"]
    work["native_train_calls"] += 2 * REPLAY_CALLS_PER_PROBE
    work["training_rows"] += 2 * REPLAY_ROWS_PER_PROBE
    work["replay_rows"] += 2 * REPLAY_ROWS_PER_PROBE
    work["old_a_rehearsal"] = dict(
        native_train_calls=2 * REPLAY_CALLS_PER_PROBE,
        training_rows=2 * REPLAY_ROWS_PER_PROBE,
        replay_rows=2 * REPLAY_ROWS_PER_PROBE,
        bank_transitions_per_probe=BANK_TRANSITIONS,
        bank_encoded_bytes_per_probe=BANK_ENCODED_BYTES,
        logical_bank_encoded_bytes=2 * BANK_ENCODED_BYTES,
        validation_projection_rows=2 * VALIDATION_PROJECTION_ROWS,
        validation_projected_array_equivalent_bytes=2 * VALIDATION_PROJECTED_BYTES,
        cost_scope="two probe bank constructions; logical bank bytes sum admissions, not peak RSS; event/native/projection/guard/copy allocations and time additional",
    )
    protocol["limitations"] += (
        "; fixed old-A TRAIN rehearsal adds64 native calls/576 training rows across "
        "two probes; all negative/inconclusive results retained; original no-replay "
        "controls remain separate and no scientific advantage is established"
    )


_COST_FIELDS = {
    "native_train_calls": "attempted_updates",
    "returned_native_train_calls": "returned_updates",
    "committed_native_train_calls": "committed_updates",
    "training_rows": "returned_training_rows",
    "replay_rows": "charged_replay_examples",
    "logical_bank_encoded_bytes": "bank_encoded_bytes",
    "validation_projection_rows": "validation_projection_rows",
    "validation_projected_array_equivalent_bytes": "validation_projected_array_equivalent_bytes",
}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum observed receipts; missing costs remain unknown, never inferred free.

    Why: a failed factory or native call can leave only part of its paid work
    observable. Known numeric sums are lower bounds whenever partial_work_unknown
    is true; fields with no observed value are None rather than a fabricated zero.
    """
    # Unreached slots have no admission and therefore no work record. Explicit
    # None instead means an admitted slot's evidence could not be observed.
    slots = [slot for row in rows for slot in row["probes"] if "rehearsal_work" in slot]
    values: dict[str, list[int]] = {name: [] for name in _COST_FIELDS}
    unknown_slots = 0
    bound = True
    for slot in slots:
        work = slot.get("rehearsal_work")
        partial = slot.get("work_unknown") is True
        if work is None:
            unknown_slots += 1
            bound = False
            continue
        if type(work) is not dict:
            raise ValueError("old-A replay work must be a plain observed work dictionary")
        partial |= work.get("work_unknown") is not False
        bound &= work.get("budget_binding_preserved") is True
        for name, source in _COST_FIELDS.items():
            value = work.get(source)
            if value is None:
                partial = True
            elif type(value) is not int or value < 0:
                raise ValueError(f"old-A replay observed {source} must be a nonnegative exact int")
            else:
                values[name].append(value)
        unknown_slots += int(partial)
    result: dict[str, Any] = {
        name: sum(observed) if observed or not slots else None for name, observed in values.items()
    }
    result.update(
        observed_slot_count=len(slots) - sum(slot.get("rehearsal_work") is None for slot in slots),
        unknown_slot_count=unknown_slots,
        partial_work_unknown=bool(unknown_slots),
        failure_work_may_be_unobserved=bool(unknown_slots),
        budget_binding_preserved=bound,
        scope="observed replay calls/charged exposures and returned rows only; logical bank/admission bytes sum probes, not peak RSS; validation/native/event/copy/guard time and allocations additional; partial numeric sums are lower bounds",
    )
    return result
