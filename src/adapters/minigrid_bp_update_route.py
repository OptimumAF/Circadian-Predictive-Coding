"""Original MiniGrid checkpoint128 BP observer admission, description and costs.

Inputs are explicit pilot options and saved observer work records. Outputs are
plain protocol/cost dictionaries. No learner, collection, updates, evaluation,
selection, budget renewal or IO belongs here.
"""

from typing import Any


REQUIRED = (
    "public_history",
    "invariant_history",
    "joint_catalogue",
    "distinct_probe",
    "old_task_readouts",
    "return_to_a",
)
FORBIDDEN = (
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
    "old_a_rehearsal",
)


def validate_option(flag: bool, **options: Any) -> None:
    """Require the original explicit BP joint128/F16/old-A/A16 recipe."""
    if type(flag) is not bool:
        raise ValueError("bp_update_diagnostics must be an exact bool")
    if not flag:
        return
    if (
        type(options.get("qualification_method")) is not str
        or options["qualification_method"] != "bp"
        or any(options.get(name) is not True for name in REQUIRED)
        or any(options.get(name, False) is not False for name in FORBIDDEN)
        or options.get("probe_inference") is not None
    ):
        raise ValueError(
            "bp_update_diagnostics requires original explicit BP joint invariant "
            "F16/old-A/A16 without other diagnostics, weighting, rehearsal or games"
        )


def describe(protocol: dict[str, Any]) -> None:
    """Declare a separate disposable TRAIN cycle without changing score work."""
    if (
        protocol.get("methods") != ["public_invariant_bp"]
        or protocol["probe"]["source_updates"] != [0, 128]
        or protocol["probe"]["updates"] != 16
        or protocol["return_phase"]["updates"] != 16
        or "joint_catalogue" not in protocol
    ):
        raise ValueError("BP observer description requires the original BP joint recipe")
    protocol["protocol_id"] = (
        protocol["protocol_id"].removesuffix("_v1") + "_bp_update_diagnostics_v1"
    )
    metadata = dict(
        source_update=128,
        role="train",
        wake="original complete B-blue TRAIN episode",
        replay=["original complete A-blue TRAIN episode", "original complete A-red TRAIN episode"],
        native_train_calls=3,
        training_rows=27,
        replay_rows=18,
        forward_calls=15,
        forward_rows=135,
        backward_calls=9,
        backward_rows=81,
        admission_projection_rows=54,
        admission_projection_array_equivalent_bytes=127872,
        budget="fresh independent3-update/18-replay-row allowance; charged before native attempts; original clocks and limits preserved",
        optimizer="original checkpoint128 BP SGD/momentum retained between all three calls",
        isolation="disposable snapshot before reserved F collection; zero deployed updates and no final or reserved F inputs",
        cost_scope="explicit diagnostic forwards/backwards exclude native training internals; every returned native call additionally performs one forward/backward/optimizer step and projects nine rows; snapshot validation, copies, RSS and wall time additional",
        interpretation="TRAIN counterfactual only; no causal F, acquisition, retention, recovery or circadian advantage claim",
    )
    protocol["bp_update_diagnostics"] = metadata
    work = protocol["planned_work"]
    work["native_train_calls"] += 3
    work["training_rows"] += 27
    work["replay_rows"] += 18
    work["bp_update_diagnostics"] = dict(metadata)
    protocol["limitations"] += (
        "; separate checkpoint128 BP TRAIN observer adds3 native calls/27 rows "
        "including18 replay exposures; original F/A probes and classifier score "
        "counts unchanged; negative/inconclusive results remain valid"
    )


COST_FIELDS = {
    "native_train_calls": "native_train_calls_attempted",
    "returned_native_train_calls": "native_train_calls_returned",
    "attempted_training_rows": "native_training_rows_attempted",
    "training_rows": "native_training_rows_returned",
    "replay_rows": "charged_replay_rows",
    "charged_updates": "charged_updates",
    **{
        name: name
        for name in (
            "projection_calls_attempted",
            "projection_calls_completed",
            "projection_rows_attempted",
            "projection_rows_completed",
            "projection_array_bytes_returned",
            "forward_calls_attempted",
            "forward_calls_completed",
            "forward_rows_attempted",
            "forward_rows_completed",
            "backward_calls_attempted",
            "backward_calls_completed",
            "backward_rows_attempted",
            "backward_rows_completed",
            "admission_projection_rows",
            "admission_projection_array_equivalent_bytes",
            "snapshot_validations_attempted",
            "snapshot_validations_completed",
            "validation_temporary_optimizers",
            "optimizer_steps",
            "optimizer_steps_returned",
            "native_optimizer_constructions_observed",
            "published_updates",
        )
    },
}


def aggregate(record: dict[str, Any] | None) -> dict[str, Any]:
    """Preserve observed attempt/return differences and missing paid prefixes.

    Why: None denotes no admission; an admitted report without work is unknown.
    Known counters in incomplete reports are observed lower bounds, not free work.
    """
    if record is not None and type(record) is not dict:
        raise ValueError("BP observer record must be a plain dictionary or None")
    work = None if record is None else record.get("work")
    if work is not None and type(work) is not dict:
        raise ValueError("BP observer work must be a plain dictionary or None")
    result: dict[str, Any] = {}
    unknown = record is not None and (
        work is None
        or record.get("status") != "complete"
        or work.get("failure_work_may_be_unobserved") is not False
        or work.get("budget_binding_preserved") is not True
    )
    for name, source in COST_FIELDS.items():
        value = 0 if record is None else None if work is None else work.get(source)
        if value is None:
            unknown = True
        elif type(value) is not int or value < 0:
            raise ValueError(f"BP observer observed {source} must be a nonnegative exact int")
        result[name] = value
    result.update(
        admitted=record is not None,
        partial_work_unknown=unknown,
        failure_work_may_be_unobserved=unknown,
        budget_binding_preserved=record is None
        or (work is not None and work.get("budget_binding_preserved") is True),
        scope="observed diagnostic counters; incomplete numeric counters are lower bounds; explicit forwards/backwards exclude native training internals; validation/copies/RSS/time additional",
    )
    return result


def add_work(result: dict[str, Any], record: dict[str, Any] | None) -> None:
    """Attach diagnostic costs and include native work in the existing totals."""
    extra = aggregate(record)
    result["bp_update_diagnostics"] = extra
    for name in ("native_train_calls", "training_rows", "replay_rows"):
        previous, additional = result.get(name), extra[name]
        result[name] = None if previous is None or additional is None else previous + additional
    result["failure_work_may_be_unobserved"] = bool(
        result.get("failure_work_may_be_unobserved", False)
        or extra["failure_work_may_be_unobserved"]
    )
    result["observed_work_is_lower_bound"] = bool(
        result.get("observed_work_is_lower_bound", False) or extra["partial_work_unknown"]
    )
