"""Describe the complete future study before actual source/role/request bindings.

Inputs are unchanged fixed configuration/analysis/precision factories and,
separately, complete ordered fabricated or prospective seed declarations.
Outputs preserve every matched/informative setting, role/cell/contrast, cap and
analysis/stopping requirement. Historical reservations are reference templates;
future seeds remain unset. No IO, RNG/data/model, outcome-based configuration
selection, fresh admission or original prior-usage/resource acceptance belongs here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any

from src.app.continual_confirmation_analysis_contract import fixed_analysis_contract
from src.app.continual_confirmation_checkpoints import declarations
from src.app.continual_confirmation_manifest import (
    ConfirmationManifest,
    fixed_confirmation_manifest,
    validate_confirmation_manifest,
)
from src.app.continual_precision_contract import fixed_precision_contract
from src.core.prospective_replications import (
    DeclaredReplicaStreams,
    ReplicaSeedBinding,
    ReplicaSlot,
    bind_replica_streams,
)
from src.core.seed_stream_screening import confirmation_seed_streams


def _replica_slots(manifest: ConfirmationManifest) -> tuple[ReplicaSlot, ...]:
    groups: dict[int, int] = {}
    rows = []
    for family in manifest.families:
        for index, old_seed in enumerate(family.seeds):
            group = groups.setdefault(old_seed, len(groups))
            rows.append(ReplicaSlot(family.name, index, f"planned_source_group/{group}"))
    # Why this: preserve the original declared shared-base coupling while
    # requiring a new binding. The old integer is never a future seed value.
    return tuple(rows)


def _role_requirements(
    slots: tuple[ReplicaSlot, ...], manifest: ConfirmationManifest
) -> list[dict[str, Any]]:
    names = ("train", "inner_guard", "outer_selection", "final_test")
    uses = {
        "train": "wake_training_and_declared_retention_exposure",
        "inner_guard": "declared_inner_guard_only",
        "outer_selection": "unscored_and_unavailable_to_confirmation_setting_selection",
        "final_test": "predeclared_endpoint_evaluation_after_complete_global_freeze_only",
    }
    rows = []
    for slot in slots:
        family = next(row for row in manifest.families if row.name == slot.family)
        for phase_index, phase in enumerate(("a", "b")):
            counts = manifest.role_counts[3 * phase_index : 3 * phase_index + 3] + (
                family.expected_final_counts[phase_index],
            )
            for role, count in zip(names, counts, strict=True):
                rows.append(
                    {
                        "slot": asdict(slot),
                        "phase": phase,
                        "role": role,
                        "expected_count": count,
                        "source_available_at": "complete_global_freeze"
                        if role == "final_test"
                        else f"phase_{phase}_arrival",
                        "labels_available_at": "complete_global_freeze"
                        if role == "final_test"
                        else f"phase_{phase}_arrival",
                        "allowed_use": uses[role],
                        "actual_sample_ids": None,
                        "actual_role_array_identity": None,
                    }
                )
    return rows


def _family_templates(manifest: ConfirmationManifest) -> list[dict[str, Any]]:
    rows = []
    for family in manifest.families:
        original = json.loads(family.development_manifest_json)
        # These exact historical seed fields stay in the original manifest.
        # A future configuration consumes its separately validated new bindings.
        unseeded = {
            key: value
            for key, value in original.items()
            if key not in {"seeds", "confirmation_seeds"}
        }
        arms = []
        for name, declaration in declarations(family, family.seeds[0]).items():
            full = asdict(declaration)
            arms.append(
                {
                    "name": name,
                    "configuration_without_seed": {
                        key: value for key, value in full.items() if key != "seed"
                    },
                    "seed_binding": "source_group_base_plus_original_model_initialization_offset",
                }
            )
        rows.append(
            {
                "name": family.name,
                "development_configuration_sha256": family.development_manifest_sha256,
                "configuration_without_historical_reservations": unseeded,
                "arm_templates": arms,
                "ordered_contrasts": family.contrasts,
                "wake_updates_per_replication": family.wake_updates // len(family.seeds),
                "maximum_optimizer_updates_per_replication": family.maximum_optimizer_updates
                // len(family.seeds),
                "maximum_guarded_attempts_per_replication": family.maximum_guarded_attempts
                // len(family.seeds),
            }
        )
    return rows


def fixed_prospective_design() -> dict[str, Any]:
    """Freeze the whole structural design; actual new bindings remain required."""
    manifest = fixed_confirmation_manifest()
    summary = validate_confirmation_manifest(manifest)
    analysis, objective = fixed_analysis_contract(), fixed_precision_contract()
    slots = _replica_slots(manifest)
    cell_plan = [
        {"slot": asdict(slot), "arm": arm, "endpoints": list(analysis.secondary_metrics[:3])}
        for slot in slots
        for arm in next(family for family in manifest.families if family.name == slot.family).arms
    ]
    statements = [
        {"family": family.name, "left": left, "right": right, "metric": metric}
        for family in manifest.families
        for left, right in family.contrasts
        for metric in analysis.primary_metrics
    ]
    body = {
        "schema_id": "p67_complete_prospective_confirmation_design_v1",
        "original_complete_manifest": asdict(manifest),
        "original_complete_analysis": asdict(analysis),
        "original_precision_objective": asdict(objective),
        "family_templates": _family_templates(manifest),
        "replica_slots": [asdict(row) for row in slots],
        "role_requirements": _role_requirements(slots, manifest),
        "cell_plan": cell_plan,
        "ordered_primary_statements": statements,
        "derived_stream_requirements": [
            {"name": row.name, "offset": row.value} for row in confirmation_seed_streams(0)
        ],
        "replication_policy": {
            "replications_per_family": 10,
            "family_replica_views": 60,
            "planned_shared_source_groups": len({row.source_group for row in slots}),
            "count_rationale": "largest_complete_six_family_count_within_original_additive_optimizer_cap",
            "precision_status": "exploratory_five_point_simultaneous_precision_uncertified",
            "pilot_to_final_dispersion_equality_verified": False,
            "independent_source_replications": None,
            "actual_base_seed_bindings": None,
            "copies_or_repeated_runs_add_replications": False,
            "cross_family_pooling": False,
        },
        "resource_envelope": {
            "maximum_optimizer_updates": manifest.max_optimizer_updates,
            "complete_count_optimizer_ceiling": summary["maximum_optimizer_updates"],
            "wall_limit_seconds": manifest.wall_limit_seconds,
            "max_process_rss_bytes": manifest.max_process_rss_bytes,
            "rss_interval_seconds": manifest.rss_interval_seconds,
            "future_time_and_memory_fit_measured": False,
            "every_attempt_and_failure_charged_to_declared_request": True,
        },
        "stopping_contract": {
            "planned_complete_cell_count": len(cell_plan),
            "planned_complete_endpoint_count": summary["final_evaluations"],
            "primary_statement_count": len(statements),
            "favorable_stopping_or_seed_replacement": False,
            "metrics_contrasts_or_baselines_changed_after_values": False,
            "numerical_failure_policy": "retain_all_endpoints_nulls_and_failures_under_original_analysis_rules",
            "hard_budget_failure_policy": "stop_and_preserve_failure_without_reset_retry_or_complete_success_claim",
        },
        "global_order_requirements": (
            "complete_prospective_request_and_prior_usage_gates_before_first_source",
            "all_phase_a_training_and_checkpoints_before_first_phase_b_source",
            "all_phase_b_training_and_live_state_checks_before_first_final_release",
            "all_final_releases_and_live_state_barriers_before_first_prediction",
            "all_planned_predictions_then_live_state_and_independent_readback",
        ),
        "required_actual_request_bindings": (
            "prospective_design_whole_identity",
            "new_ordered_source_group_base_seeds_and_every_derived_stream",
            "new_source_generator_configuration_and_whole_code_identities",
            "unrecycled_source_provenance_and_complete_prior_uncertainty_effects",
            "new_role_sample_id_and_source_label_availability_declarations",
            "prospective_utc_before_any_source_construction",
            "full_future_source_map_and_request_whole_identity",
            "unchanged_complete_analysis_and_stopping_identity",
            "complete_joint_resource_and_independent_repeat_envelopes",
            "exclusive_request_owner_success_failure_and_reproducibility_controls",
            "complete_b3_isolation_resource_artifact_and_independent_readback_proofs",
            "original_prior_usage_and_failed_resource_acceptance_resolution",
        ),
        "all_original_matched_informative_settings_preserved": True,
        "actual_new_request_identity": None,
        "actual_new_source_identity": None,
        "untouched_role_binding_complete": False,
        "fresh_roles_authorized": False,
        "original_p67_acceptance_complete": False,
    }
    return json.loads(json.dumps(body, sort_keys=True, allow_nan=False))


def _same_typed_value(actual: Any, expected: Any) -> None:
    if type(actual) is not type(expected):
        raise ValueError("prospective design exact field type differs")
    if type(expected) is dict:
        if set(actual) != set(expected):
            raise ValueError("prospective design complete fields differ")
        for name in expected:
            _same_typed_value(actual[name], expected[name])
    elif type(expected) is list:
        if len(actual) != len(expected):
            raise ValueError("prospective design complete ordered scope differs")
        for left, right in zip(actual, expected, strict=True):
            _same_typed_value(left, right)
    elif actual != expected:
        raise ValueError("prospective design frozen rule or value differs")


def validate_prospective_design(design: dict[str, Any]) -> None:
    """Reject any narrowed scope, changed rule/type or invented fresh authority."""
    _same_typed_value(design, fixed_prospective_design())


def prospective_design_identity(design: dict[str, Any]) -> dict[str, Any]:
    validate_prospective_design(design)
    raw = (
        json.dumps(design, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    return {"byte_count": len(raw), "sha256": sha256(raw).hexdigest()}


def declare_prospective_replica_streams(
    design: dict[str, Any], bindings: tuple[ReplicaSeedBinding, ...]
) -> DeclaredReplicaStreams:
    """Validate every fixed slot before core declaration checks; never admit roles."""
    validate_prospective_design(design)
    slots = tuple(ReplicaSlot(**row) for row in deepcopy(design["replica_slots"]))
    return bind_replica_streams(slots, bindings)
