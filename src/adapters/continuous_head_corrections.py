"""Actual successful GN correction work; no scheduling, models or solver estimates."""

import json

from src.adapters.continuous_head_learnability import _native_work
from src.core.observation_standardization import standardization_numeric_bytes


def record_head_work(report, initial, rows, native_updates, *, replay=None) -> None:
    counts, flags, weights = _native_work(native_updates)
    policy = initial.policy
    d, h, a = policy.spec.observation_dim, policy.hidden_dim, len(policy.spec.action_units)
    parameters = d * h + h + h * a + a
    exposures = sum(item["training_rows"] for item in native_updates)
    replay_updates = 0 if replay is None else replay["committed_updates"]
    replay_rows = 0 if replay is None else replay["charged_exposures"]
    if len(native_updates) != 8 + replay_updates or exposures != 8 * rows + replay_rows:
        raise RuntimeError("GN native work differs from wake and reference receipts")
    report.update(
        complete=True,
        work=dict(
            optimizer_updates=len(native_updates),
            head_gauss_newton_updates=len(native_updates),
            sgd_updates=0,
            training_gradient_evaluations=0,
            training_rows=exposures,
            unique_correction_rows=rows,
            repeated_training_rows=7 * rows,
            reference_replay_rows=replay_rows,
            fit_calls=0,
            initializations=0,
            seeded_normal_draws=0,
            native_update_counts=counts,
            native_work_flags=flags,
            native_event_row_weights=weights,
            hidden_forward_passes=counts["hidden_forward_completed"],
            hidden_forward_rows=weights["hidden_forward_completed"],
            head_forward_passes=counts["head_forward_completed"],
            head_forward_rows=weights["head_forward_completed"],
            backward_passes=0,
            backward_rows=0,
            gradient_values=0,
            gradient_numeric_bytes=0,
            normalization_calls=counts["normalization_completed"],
            normalization_rows=counts["normalization_rows"],
            normalization_values=counts["normalization_values"],
            known_dense_multiply_accumulates=counts["known_dense_multiply_accumulates"],
            solver_work_known=False,
            model_numeric_bytes_per_owner=16 * parameters,
            statistics_numeric_bytes_per_owner=standardization_numeric_bytes(
                policy.observation_standardization
            ),
            named_model_owners=4,
            inbox_feature_target_numeric_bytes=8 * rows * (d + a) * 8,
            scope="successful eight arrived GN updates; native completed counts and row weights; "
            "solver arithmetic/workspace, validation, copies, elementwise math and RSS additional",
        ),
    )

    if replay is not None:
        report["work"].update(
            wake_optimizer_updates=8,
            wake_training_rows=8 * rows,
            reference_replay_updates=replay_updates,
            reference_bank_retained_rows=replay["bank_rows"],
            reference_bank_retained_numeric_bytes=replay["retained_numeric_bytes"],
            reference_bank_signature_bytes=replay["bank_signature_bytes"],
            replay_transient_model_owners=1 if replay_updates else 0,
            replay_transient_model_numeric_bytes_per_owner=16 * parameters,
            replay_memory_scope="one temporary GN learner per event; latest native pin, snapshots, validation, copies and RSS additional",
        )


def require_replay_evidence(report, replay_payload, consolidations) -> None:
    """Reject late scalar rewrites after final owner observers; no model access."""
    if json.dumps(
        report.get("replay"), sort_keys=True, allow_nan=False
    ) != replay_payload or json.dumps(
        report.get("applied_consolidations"), sort_keys=True, allow_nan=False
    ) != json.dumps(consolidations, sort_keys=True, allow_nan=False):
        raise RuntimeError("GN final reference report or receipts changed")
