"""Write all fixed v14 guarded train-only trials to exclusive local JSON."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.app.continual_trigger_replay_runner import _parameter_digest
from src.app.continual_trigger_replay_schedule import fixed_trigger_replay_manifest
from src.app.continual_trigger_replay_training_study import (
    TriggerReplayTrainingStudy,
    preflight_trigger_replay_training_study,
    run_trigger_replay_training_study,
)


def build_payload() -> str:
    """Serialize verified train/guard decisions, excluding elapsed durations."""
    manifest = fixed_trigger_replay_manifest()
    study = run_trigger_replay_training_study(manifest)
    return serialize_training_study(study)


def serialize_training_study(study: TriggerReplayTrainingStudy) -> str:
    """Serialize a preflighted study without running its six trials again."""
    preflight_trigger_replay_training_study(study)
    manifest = study.manifest
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        pending = trial.pending
        state = pending.state
        opportunities = []
        for item in trial.opportunities:
            event = asdict(item.event)
            event.pop("durations")
            opportunities.append(
                {
                    "phase": item.phase,
                    "epoch": item.epoch,
                    "global_epoch": item.global_epoch,
                    "train_role_hash": item.train_role_hash,
                    "retention": asdict(item.retention),
                    "retained_order_ids": item.retained_order_ids,
                    "selected_ids": item.selected_ids,
                    "event": event,
                    "applied_by_method": [asdict(work) for work in item.applied_by_method],
                    "width": item.width,
                    "parameter_count": item.parameter_count,
                    "cumulative_splits": item.cumulative_splits,
                    "cumulative_prunes": item.cumulative_prunes,
                }
            )
        rows.append(
            {
                "seed": trial.seed,
                "arm": trial.arm,
                "manifest_digest": trial.manifest_digest,
                "phase_a_development_role_hashes": {
                    role: pending.phase_a.split_hashes[role]
                    for role in ("train", "inner_guard", "outer_selection")
                },
                "phase_b_development_role_hashes": {
                    role: pending.phase_b.split_hashes[role]
                    for role in ("train", "inner_guard", "outer_selection")
                },
                "initial_parameter_digests": trial.initial_parameter_digests,
                "post_a_parameter_digests": (
                    ("backprop", _parameter_digest(state.backprop_after_a)),
                    ("predictive_coding", _parameter_digest(state.predictive_after_a)),
                    ("circadian_predictive_coding", _parameter_digest(state.circadian_after_a)),
                ),
                "post_b_parameter_digests": (
                    ("backprop", _parameter_digest(state.backprop_model)),
                    ("predictive_coding", _parameter_digest(state.predictive_model)),
                    ("circadian_predictive_coding", _parameter_digest(state.circadian_model)),
                ),
                "wake_work": [asdict(work) for work in trial.wake_work],
                "guard_decisions": [asdict(item) for item in pending.audit.guard_decisions],
                "role_accesses": [asdict(item) for item in pending.audit.accesses],
                "circadian_replay_exposure": asdict(state.circadian_model.get_replay_exposure()),
                "opportunities": opportunities,
            }
        )
    return (
        json.dumps(
            {
                "protocol_id": study.protocol_id,
                "manifest_digest": study.manifest_digest,
                "resolved_manifest": asdict(manifest),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    result_path = parser.parse_args().result
    if result_path.exists():
        raise FileExistsError(f"result already exists: {result_path}")
    payload = build_payload()
    with result_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(payload)
    print(json.dumps({"result": str(result_path), "sha256": sha256(payload.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
