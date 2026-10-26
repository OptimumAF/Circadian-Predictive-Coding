"""Fixed typed joint acquisition followed by three frozen known-TRAIN returns.

Uses existing admission, private training and native ports. Trusted Python callers
supply qualified inputs; this module does not qualify files, collect labels,
select a learner, expose a candidate, tune training or grant deployment authority.
"""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np

from scripts.maniskill_continuous_smoke import FrozenRolloutResult, run_frozen_rollout
from src.adapters.continuous_learnability_demo import ContinuousLearnabilityResult
from src.adapters.continuous_learnability_modes import same_continuous_state
from src.adapters.joint_readiness_corrections import run_joint_readiness_corrections
from src.adapters.joint_readiness_data import prepare_joint_readiness_data
from src.adapters.joint_readiness_stream import _own_inputs, run_joint_readiness_stream
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner, _snapshot
from src.core.curated_readiness_corrections import CuratedReadinessBatch
from src.core.pick_cube_readiness_corrections import GoalConsistentObservedGraspCorrectionBatch
from src.core.pick_cube_train_context import pick_cube_training_context
from src.core.readiness_stage_diagnostics import ReadinessStageBinding


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _write(path, value, *, exclusive=False):
    payload = _json(value) + "\n"
    if len(payload.encode("utf8")) > 4 * 1024**2:
        raise ValueError("joint acquisition report exceeds4MiB")
    with path.open("x" if exclusive else "w", encoding="utf8") as handle:
        handle.write(payload)


def _paths(output):
    if not isinstance(output, Path):
        raise ValueError("joint acquisition output must be a Path")
    paths = (output,) + tuple(
        output.with_name(output.name + f".train-{seed}.rollout.json") for seed in (17, 19, 20)
    )
    if not output.parent.is_dir() or any(path.exists() or path.is_symlink() for path in paths):
        raise ValueError("joint acquisition requires four fresh non-symlink output paths")
    return paths


def _validate_native(raw, seed, policy, raw_spec, recipe):
    episode = raw.get("episode")
    context = pick_cube_training_context(seed)
    if (
        raw.get("complete") is not True
        or episode is None
        or type(raw["seed"]) is not int
        or raw["seed"] != seed
        or (raw["role"], raw["context"]) != ("train", context)
        or (episode["episode_id"], episode["role"], episode["context"])
        != (f"pick-cube-smoke-{seed}", "train", context)
        or _json(episode["spec"]) != _json(asdict(raw_spec))
        or raw["policy"] != "frozen_continuous_bc"
        or _json(raw["learner_policy"]) != _json(asdict(policy))
        or any(
            raw[key] is not True
            for key in ("frozen_state_exact", "original_owner_exact", "action_history_exact")
        )
        or type(raw["optimizer_updates"]) is not int
        or raw["optimizer_updates"] != 0
        or any(
            type(raw[key]) is not int or raw[key] != value
            for key, value in (("source_updates", 40), ("source_training_rows", 2776))
        )
        or raw.get("corrections") is not None
        or any(
            key in raw
            for key in (
                "probe_test_seed",
                "guard_seed",
                "scoring_only",
                "readiness_roll_in",
                "action_decoding",
                "joint_position_summary",
                "mass_context",
                "lighting_context",
            )
        )
    ):
        raise RuntimeError("joint acquisition frozen TRAIN namespace/source differs")
    rows, work = episode["transitions"], raw["work"]
    if (
        (raw["creation_attempts"], raw["creation_returns"], raw["explicit_reset"])
        != (1, 1, f"seed{seed}_if_construction_reaches_port_reset")
        or any(type(raw[key]) is not int for key in ("creation_attempts", "creation_returns"))
        or not 1 <= len(rows) <= 50
        or any(
            type(work[key]) is not int
            for key in (
                "step_attempts",
                "completed_steps",
                "completed_control_physics_steps",
                "possible_failed_physics_steps",
            )
        )
        or (
            work["step_attempts"],
            work["completed_steps"],
            work["completed_control_physics_steps"],
            work["possible_failed_physics_steps"],
        )
        != (len(rows), len(rows), 5 * len(rows), 0)
    ):
        raise RuntimeError("joint acquisition native work differs from its bounded complete prefix")
    for index, row in enumerate(rows):
        if (
            type(row["step"]) is not int
            or row["step"] != index
            or type(row["reward"]) not in (int, float)
            or row["reward"] not in (0, 1)
        ):
            raise RuntimeError("joint acquisition native step/reward differs")
        for key in ("observation", "next_observation"):
            if len(row[key]) != 42 or any(
                type(value) is not float or not math.isfinite(value) for value in row[key]
            ):
                raise RuntimeError("joint acquisition requires finite complete raw42 evidence")
        if (
            len(row["action"]) != 4
            or any(
                type(value) is not float
                or not math.isfinite(value)
                or not -1 <= value <= 1
                or struct.unpack("!f", struct.pack("!f", value))[0] != value
                for value in row["action"]
            )
            or type(row["terminated"]) is not bool
            or type(row["truncated"]) is not bool
            or (row["terminated"] or row["truncated"]) != (index == len(rows) - 1)
            or (index and _json(rows[index - 1]["next_observation"]) != _json(row["observation"]))
        ):
            raise RuntimeError("joint acquisition native32 command/seam/end flag differs")
    closing = 0
    for row in rows:
        closing = min(6, closing + 1) if row["action"][3] < 0 else 0
    history = raw["action_history"]
    if (
        history["pending"] is not False
        or _json(history["state"])
        != _json(dict(previous_action=rows[-1]["action"], closing_count=closing))
        or any(type(value) is not int for value in history["work"].values())
        or history["work"]
        != dict(
            encoding_attempts=len(rows),
            encoding_returns=len(rows),
            committed_controls=len(rows),
            source_values_validated=42 * len(rows),
            model_values_emitted=12 * len(rows),
            model_numeric_bytes_emitted=96 * len(rows),
        )
    ):
        raise RuntimeError("joint acquisition executed history/pending/work differs")
    if (
        _json(raw["action_history"]["recipe"]) != _json(asdict(recipe))
        or type(raw["any_success"]) is not bool
        or type(raw["final_success"]) is not bool
        or type(raw["sparse_return"]) not in (int, float)
        or not math.isfinite(raw["sparse_return"])
        or raw["sparse_return"] != sum(row["reward"] for row in rows)
        or raw["any_success"] is not any(row["reward"] > 0 for row in rows)
        or raw["final_success"] is not (rows[-1]["reward"] > 0)
    ):
        raise RuntimeError("joint acquisition history or primary scores differ")


def run_joint_acquisition_rollouts(
    output: Path,
    warm: ReadinessStageBinding,
    new: ReadinessStageBinding,
    corrections: GoalConsistentObservedGraspCorrectionBatch,
    curated: CuratedReadinessBatch,
    retained_reference: ReadinessStageBinding,
    *,
    input_receipts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the fixed admitted profile; partial paid work never releases a learner.

    Input/path admission raises before model/SDK/output work. Later failures return
    and save an incomplete report, including separately observed state errors.
    """
    # Receipt metadata records already paid acquisition work. It grants no data
    # authority; the same typed admission below remains mandatory.
    receipt_copy = None
    receipt_seal = None
    if input_receipts is not None:
        if type(input_receipts) is not dict:
            raise ValueError("input receipts must be an exact dictionary")
        try:
            receipt_seal = _json(input_receipts)
        except (TypeError, ValueError) as error:
            raise ValueError("input receipts must be finite JSON metadata") from error
        if len(receipt_seal.encode("utf8")) > 256 * 1024:
            raise ValueError("input receipt metadata exceeds256KiB")
        receipt_copy = deepcopy(input_receipts)
    originals = (warm, new, corrections, curated, retained_reference)
    types = (
        ReadinessStageBinding,
        ReadinessStageBinding,
        GoalConsistentObservedGraspCorrectionBatch,
        CuratedReadinessBatch,
        ReadinessStageBinding,
    )
    if any(type(value) is not kind for value, kind in zip(originals, types, strict=True)):
        raise ValueError("joint acquisition requires its five exact typed raw inputs")
    if tuple(len(value.episode.transitions) for value in originals) != (15, 22, 50, 50, 15):
        raise ValueError("joint acquisition requires fixed15/22/50/50/15 source rows")
    paths = _paths(output)
    original_seal = _json([asdict(value) for value in originals])
    owned_warm, owned_new, b, c, recipe, ab, abc, model_bindings = _own_inputs(
        warm, new, corrections, curated, 99, 199
    )
    retained = prepare_joint_readiness_data(
        owned_warm, b, c, now=1100, recipe=recipe, retained_reference=retained_reference
    )
    owned = (owned_warm, owned_new, b, c, recipe, ab, abc, *model_bindings, retained)
    owned_seal = _json([asdict(value) for value in owned])
    if _json([asdict(value) for value in originals]) != original_seal:
        raise ValueError("joint acquisition original input changed during admission")
    if input_receipts is not None and _json(input_receipts) != receipt_seal:
        raise ValueError("input receipts changed during admission")
    source_projection = dict(
        episode_calls=2,
        observation_projections=74,
        source_values_validated=84 * 37,
        projected_values=24 * 37,
        projected_numeric_bytes=192 * 37,
        history_advances=37,
    )
    report: dict[str, Any] = dict(
        schema_version=1,
        protocol_id="joint_acquisition_robot_train_v1",
        complete=False,
        source=None,
        acquisition=None,
        source_attempted=False,
        acquisition_attempted=False,
        native_attempts=[],
        rollouts=[],
        candidate_unchanged=None,
        input_admission_work=dict(
            preflight_calls=3,
            joint_ab=deepcopy(ab.work),
            joint_abc=deepcopy(abc.work),
            retained_acquisition=deepcopy(retained.work),
            source_projection=source_projection,
            projection_rows=347,
            observation_projections=694,
            scope="additional preflight projections; runtime preparations remain separately charged; copies/validation/Python storage additional",
        ),
    )
    if receipt_copy is not None:
        report["input_receipts"] = receipt_copy
    # Admission may invoke trusted validators; refuse any path that appeared meanwhile.
    _paths(output)
    metadata = report["input_admission_work"]
    metadata_seal = _json(metadata)
    paid: list[Any] = []
    pins: list[Any] = []
    native_paid: list[Any] = []
    requests: list[Any] = []
    rollouts, attempts = report["rollouts"], report["native_attempts"]
    attempted = {"source_attempted": False, "acquisition_attempted": False}
    ready = False
    flags = dict(complete=False, candidate_unchanged=None)

    def guard():
        if input_receipts is None:
            if "input_receipts" in report:
                raise RuntimeError("unexpected input receipt authority")
        elif (
            report.get("input_receipts") is not receipt_copy
            or _json(input_receipts) != receipt_seal
            or _json(receipt_copy) != receipt_seal
        ):
            raise RuntimeError("input receipt metadata owner/content changed")
        if (
            any(report[key] is not value for key, value in flags.items())
            or type(report["schema_version"]) is not int
            or report["schema_version"] != 1
            or report["protocol_id"] != "joint_acquisition_robot_train_v1"
            or _json([asdict(value) for value in originals]) != original_seal
            or _json([asdict(value) for value in owned]) != owned_seal
            or report["input_admission_work"] is not metadata
            or _json(metadata) != metadata_seal
            or any(report[key] is not value for key, value in attempted.items())
            or report["rollouts"] is not rollouts
            or len(rollouts) != len(native_paid)
            or report["native_attempts"] is not attempts
            or len(attempts) != len(requests)
        ):
            raise RuntimeError(
                "joint acquisition original input/metadata/paid-prefix owners changed"
            )
        for owner, native, policy, state, holder in pins:
            if (
                type(owner) is not NumpyContinuousBCLearner
                or holder.candidate is not owner
                or owner._state is not native
                or owner._policy is not policy
                or not same_continuous_state(state, _snapshot(owner._state, owner._policy))
            ):
                raise RuntimeError("joint acquisition original native learner changed")
        for key, holder, raw, copy, payload in paid:
            if (
                holder.report is not raw
                or report[key] is not copy
                or _json(raw) != payload
                or _json(copy) != payload
            ):
                raise RuntimeError(
                    "joint acquisition preceding raw/copied training evidence changed"
                )
        for index, (holder, raw, copy, item, payload, item_payload) in enumerate(native_paid):
            if (
                holder.report is not raw
                or rollouts[index] is not item
                or item["rollout"] is not copy
                or _json(raw) != payload
                or _json(copy) != payload
                or _json(item) != item_payload
            ):
                raise RuntimeError("joint acquisition preceding native evidence changed")
        for index, (request, payload) in enumerate(requests):
            if attempts[index] is not request or _json(request) != payload:
                raise RuntimeError("joint acquisition original native request changed")

    def pin(holder, counts):
        if (
            type(holder) is not ContinuousLearnabilityResult
            or type(holder.candidate) is not NumpyContinuousBCLearner
        ):
            raise RuntimeError("joint acquisition requires an exact native BP result")
        owner = holder.candidate
        state = _snapshot(owner._state, owner._policy)
        if (
            (state.completed_updates, state.training_examples) != counts
            or state.policy.spec != retained.spec
            or state.policy.hidden_dim != 64
            or (pins and state.policy != pins[0][3].policy)
        ):
            raise RuntimeError("joint acquisition released policy/counters differ")
        for previous, *_ in pins:
            if owner is previous or any(
                np.shares_memory(a, z)
                for a in owner._state.parameters + owner._state.velocity
                for z in previous._state.parameters + previous._state.velocity
            ):
                raise RuntimeError("joint acquisition released learner aliases an original owner")
        pins.append((owner, owner._state, owner._policy, state, holder))
        return owner

    def retain(key, holder):
        raw = holder.report
        copy = report[key] = deepcopy(raw)
        paid.append((key, holder, raw, copy, _json(raw)))
        _write(output, report)

    # Why: the main report is our exclusive reservation. Each existing native
    # entry point reserves its own fresh output before any SDK work.
    _write(output, report, exclusive=True)
    try:
        guard()
        attempted["source_attempted"] = report["source_attempted"] = True
        rebuilt = run_joint_readiness_stream(
            warm, new, corrections, curated, corrections_now=99, curated_now=199
        )
        retain("source", rebuilt)
        if rebuilt.report["complete"] is not True:
            raise RuntimeError("joint acquisition source stream incomplete")
        source = pin(rebuilt, (32, 1736))
        guard()
        attempted["acquisition_attempted"] = report["acquisition_attempted"] = True
        acquired = run_joint_readiness_corrections(
            source,
            warm,
            corrections,
            curated,
            now=1100,
            action_history_recipe=recipe,
            retained_reference=retained_reference,
        )
        retain("acquisition", acquired)
        guard()
        raw = acquired.report
        if (
            raw["complete"] is not True
            or _json(raw["rows"]) != _json(retained.rows)
            or _json(raw["groups"]) != _json(retained.groups)
            or len(raw["native_updates"]) != 8
            or len(raw["applied_updates"]) != 8
            or raw["progress"]
            != dict(updates_completed=8, replay_examples_completed=0, logical_tick=1100)
        ):
            raise RuntimeError("joint acquisition fixed eight-update report incomplete")
        for index, (native, receipt) in enumerate(
            zip(raw["native_updates"], raw["applied_updates"], strict=True)
        ):
            if (
                native["training_rows"] != 130
                or native["work"]["published_updates"] != 1
                or _json(native["diagnostic"]) != _json(receipt["diagnostic"])
                or (receipt["observed_at"], receipt["arrived_at"], receipt["applied_at"])
                != (1100, 1100, 1100)
                or receipt["update_number"] != index + 1
            ):
                raise RuntimeError("joint acquisition paid publication/receipt differs")
        candidate = pin(acquired, (40, 2776))
        guard()
        for seed, path in zip((17, 19, 20), paths[1:], strict=True):
            guard()
            request = dict(seed=seed, output=str(path), attempted=True)
            attempts.append(request)
            requests.append((request, _json(request)))
            _write(output, report)
            if path.exists() or path.is_symlink():
                raise ValueError("native output appeared after initial admission")
            rolled = run_frozen_rollout(
                path, learner=candidate, train_seed=seed, action_history_recipe=recipe
            )
            raw = rolled.report
            copy = deepcopy(raw)
            item = dict(seed=seed, output=str(path), rollout=copy)
            rollouts.append(item)
            native_paid.append((rolled, raw, copy, item, _json(raw), _json(item)))
            _write(output, report)
            guard()
            if type(rolled) is not FrozenRolloutResult or rolled.corrections is not None:
                raise RuntimeError("joint acquisition native return has foreign authority")
            _validate_native(raw, seed, candidate._policy, warm.episode.spec, recipe)
            guard()
        ready = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["unreturned_source_work_unknown"] = (
            attempted["source_attempted"] and report["source"] is None
        )
        report["unreturned_acquisition_work_unknown"] = (
            attempted["acquisition_attempted"] and report["acquisition"] is None
        )
        report["unreturned_native_work_unknown"] = len(requests) != len(native_paid)
        try:
            guard()
            flags["candidate_unchanged"] = report["candidate_unchanged"] = (
                True if len(pins) == 2 else None
            )
        except Exception as error:
            ready = False
            flags["candidate_unchanged"] = report["candidate_unchanged"] = False
            report["state_error"] = f"{type(error).__name__}: {error}"
        flags["complete"] = report["complete"] = ready
        _write(output, report)
        try:
            guard()
        except Exception as error:
            flags["complete"] = report["complete"] = False
            flags["candidate_unchanged"] = report["candidate_unchanged"] = False
            report.setdefault("state_error", f"{type(error).__name__}: {error}")
            _write(output, report)
    return deepcopy(report)
