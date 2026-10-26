"""Portable CPU reproduction of fixed TRAIN memory and recorded action replay.

Inputs are a bounded native-JSON bundle of seven labeled TRAIN episodes and four
separate recorded traces. Outputs retain the existing stream, retrieval receipts
and recorded outcomes. No simulator, teacher, new data, optimizer, role conversion,
model export or physical-success claim is provided.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np

from src.adapters.continuous_stream_sequence import run_sequence, _same
from src.adapters.numpy_nearest_continuous_bc import (
    FixedInputNearestContinuousBCPolicy,
    NumpyNearestContinuousBCLearner,
    _snapshot,
)
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.nearest_continuous_bc import NearestPrediction
from src.core.pick_cube_action_history import _validate_encoded_history
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.pick_cube_train_context import pick_cube_training_context

DEFINITION = "taskspace_memory_demo_v1"
MAX_BUNDLE_BYTES = 512 * 1024
TRAINING_GROUPS = ("a", "b", "train20", "push", "corrections", "current_corrections", "b22")
REPLAY_SCENES = (("pick19", 19), ("pick17", 17), ("pick20", 20), ("push17", 17))


def _fields(value, names, label):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(f"{label} requires exactly {', '.join(names)}")
    return value


def _unique(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate bundle field {name}")
        result[name] = value
    return result


def _reject_constant(value):
    raise ValueError(f"bundle requires finite JSON numbers, got {value}")


def _own_json(value):
    # Serialize first so cyclic direct Python inputs fail before the native-type walk.
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
        if len(encoded.encode("utf8")) > MAX_BUNDLE_BYTES:
            raise ValueError("bundle exceeds512KiB")
    except (TypeError, UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError("bundle requires bounded native UTF-8 JSON") from error
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("bundle object keys must be native strings")
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("bundle JSON numbers must be finite")
        elif item is not None and type(item) not in (str, int, bool):
            raise ValueError("bundle requires only native JSON values")
    return json.loads(encoded)


def load_bundle(path: Path) -> dict[str, Any]:
    """Read at most512KiB, rejecting duplicate keys and nonfinite JSON."""
    with path.open("rb") as handle:
        payload = handle.read(MAX_BUNDLE_BYTES + 1)
    if len(payload) > MAX_BUNDLE_BYTES:
        raise ValueError("bundle exceeds512KiB")
    try:
        value = json.loads(
            payload.decode("utf8"), object_pairs_hook=_unique, parse_constant=_reject_constant
        )
    except (UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError("bundle requires bounded native UTF-8 JSON") from error
    if type(value) is not dict:
        raise ValueError("bundle must be a native JSON object")
    return _own_json(value)


def _list(value, label):
    if type(value) is not list:
        raise ValueError(f"{label} must be a JSON list")
    return value


def _episode(value) -> ContinuousEpisode:
    data = _fields(value, ("episode_id", "context", "role", "spec", "transitions"), "episode")
    if type(data["role"]) is not str or data["role"] != "train":
        raise ValueError("demo episodes must already declare TRAIN role")
    spec = _fields(
        data["spec"], ("observation_dim", "action_units", "lower_bounds", "upper_bounds"), "spec"
    )
    declared = ContinuousActionSpec(
        spec["observation_dim"],
        tuple(_list(spec["action_units"], "units")),
        tuple(_list(spec["lower_bounds"], "bounds")),
        tuple(_list(spec["upper_bounds"], "bounds")),
    )
    if declared != cube_taskspace_spec():
        raise ValueError("demo episodes require the declared cubeD12 spec")
    records = _list(data["transitions"], "transitions")
    if not 1 <= len(records) <= 50:
        raise ValueError("each demo episode requires1..50 complete rows")
    rows = []
    for record in records:
        row = _fields(
            record,
            (
                "step",
                "observation",
                "action",
                "reward",
                "next_observation",
                "terminated",
                "truncated",
            ),
            "transition",
        )
        rows.append(
            ContinuousTransition(
                row["step"],
                tuple(_list(row["observation"], "observation")),
                tuple(_list(row["action"], "action")),
                row["reward"],
                tuple(_list(row["next_observation"], "next observation")),
                row["terminated"],
                row["truncated"],
            )
        )
    return ContinuousEpisode(data["episode_id"], data["context"], "train", declared, tuple(rows))


@dataclass(frozen=True)
class _Replay:
    scene: str
    seed: int
    episode: ContinuousEpisode
    indices: tuple[int, ...]
    outcomes: dict[str, Any]
    action_bytes: tuple[bytes, ...]


def _replay(value, scene, seed, bank_rows) -> _Replay:
    item = _fields(
        value, ("scene", "seed", "episode", "selected_indices", "recorded_outcomes"), "replay"
    )
    if (
        type(item["scene"]) is not str
        or item["scene"] != scene
        or type(item["seed"]) is not int
        or item["seed"] != seed
    ):
        raise ValueError("recorded replay order/seeds must be pick19,pick17,pick20,push17")
    episode = _episode(item["episode"])
    context = f"push-cube-train-{seed}" if scene == "push17" else pick_cube_training_context(seed)
    if episode.context != context:
        raise ValueError("recorded replay context differs from its fixed scene")
    # Training targets may differ from executed commands; only recorded traces
    # pass this causal-history validation. No public TRAIN identity is weakened.
    _validate_encoded_history(episode, PickCubeTaskSpaceHistoryRecipe(mode="executed"))
    indices = _list(item["selected_indices"], "selected indices")
    if len(indices) != len(episode.transitions) or any(
        type(index) is not int or not 0 <= index < bank_rows for index in indices
    ):
        raise ValueError("recorded selectors require one bounded native index per control")
    outcomes = _fields(
        item["recorded_outcomes"],
        ("final_success", "any_success", "sparse_return"),
        "recorded outcomes",
    )
    if (
        type(outcomes["final_success"]) is not bool
        or type(outcomes["any_success"]) is not bool
        or type(outcomes["sparse_return"]) not in (int, float)
        or (
            type(outcomes["sparse_return"]) is float
            and not math.isfinite(outcomes["sparse_return"])
        )
    ):
        raise ValueError("recorded outcomes require exact booleans and a finite sparse return")
    rows, final = episode.transitions, episode.transitions[-1]
    if (
        any(row.reward not in (0.0, 1.0) for row in rows)
        or outcomes["final_success"] != (final.reward == 1.0)
        or outcomes["any_success"] != any(row.reward == 1.0 for row in rows)
        or outcomes["sparse_return"] != sum(row.reward for row in rows)
        or final.terminated != outcomes["final_success"]
        or final.truncated == outcomes["final_success"]
    ):
        raise ValueError("recorded outcomes/end flags differ from original sparse rewards")
    return _Replay(
        scene,
        seed,
        episode,
        tuple(indices),
        deepcopy(outcomes),
        tuple(struct.pack("<4f", *row.action) for row in rows),
    )


def _admit(payload):
    data = _fields(_own_json(payload), ("definition", "training", "replay", "provenance"), "bundle")
    if (
        type(data["definition"]) is not str
        or data["definition"] != DEFINITION
        or type(data["provenance"]) is not dict
    ):
        raise ValueError("demo definition/provenance differs from the fixed bundle schema")
    training = _list(data["training"], "training")
    if len(training) != len(TRAINING_GROUPS):
        raise ValueError("demo requires seven ordered TRAIN groups")
    episodes = []
    for name, record in zip(TRAINING_GROUPS, training, strict=True):
        item = _fields(record, ("group", "episode"), "training group")
        if type(item["group"]) is not str or item["group"] != name:
            raise ValueError("demo TRAIN group order differs")
        episodes.append(_episode(item["episode"]))
    total = sum(len(episode.transitions) for episode in episodes)
    if total > 256 or len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("demo requires distinct TRAIN identities and at most256 rows")
    replays = _list(data["replay"], "replay")
    if len(replays) != len(REPLAY_SCENES):
        raise ValueError("demo requires four ordered recorded replays")
    owned = tuple(
        _replay(value, scene, seed, total)
        for value, (scene, seed) in zip(replays, REPLAY_SCENES, strict=True)
    )
    return tuple(episodes), owned, data["provenance"]


def _prediction_report(value):
    if type(value) is not NearestPrediction:
        return dict(type=type(value).__name__)
    result: dict[str, Any] = dict(type="NearestPrediction")
    for name in ("distance_candidates", "distance_feature_values", "selected_indices"):
        item = getattr(value, name)
        if type(item) is int:
            result[name] = item
        elif type(item) is tuple and len(item) <= 50 and all(type(index) is int for index in item):
            result[name] = list(item)
        else:
            result[name + "_repr"] = repr(item)[:1024]
    actions = value.actions
    if type(actions) is np.ndarray:
        result.update(shape=list(actions.shape), dtype=str(actions.dtype))
        if (
            actions.dtype == np.float64
            and actions.ndim == 2
            and actions.size <= 200
            and np.isfinite(actions).all()
        ):
            result["actions"] = actions.tolist()
        else:
            result["actions_repr"] = [repr(item) for item in actions.flat[:200]]
            result["actions_omitted"] = max(0, actions.size - 200)
    else:
        result["actions_type"] = type(actions).__name__
    return result


def _pin_readout(readout):
    if (
        type(readout) is not NumpyNearestContinuousBCLearner
        or type(readout._policy) is not FixedInputNearestContinuousBCPolicy
    ):
        raise ValueError("demo requires an exact detached fixed-input nearest readout")
    return (
        readout._state,
        readout._policy,
        readout._state.prototype_features,
        readout._state.prototype_actions,
        _snapshot(readout._state, readout._policy),
        readout._state.policy,
    )


def _require_readout(readout, pin):
    if (
        type(readout) is not NumpyNearestContinuousBCLearner
        or readout._state is not pin[0]
        or readout._policy is not pin[1]
        or readout._state.policy is not pin[5]
        or readout._state.prototype_features is not pin[2]
        or readout._state.prototype_actions is not pin[3]
        or not _same(pin[4], _snapshot(readout._state, readout._policy))
    ):
        raise RuntimeError("recorded replay changed its frozen readout")


def run_demo(payload: dict) -> dict[str, Any]:
    """Append only TRAIN inputs, then reproduce selectors/commands offline."""
    episodes, recordings, provenance = _admit(payload)
    bank_rows = sum(len(episode.transitions) for episode in episodes)
    work = dict(
        simulator_controls=0,
        prediction_attempts=0,
        prediction_returns=0,
        query_rows_attempted=0,
        query_rows_returned=0,
        distance_candidate_upper_bound=0,
        distance_feature_upper_bound=0,
        distance_candidates_completed=0,
        distance_feature_values_completed=0,
        validated_replays=0,
        unknown_unreturned_prediction_work=False,
        unknown_unvalidated_prediction_work=False,
    )
    report: dict[str, Any] = dict(
        definition=DEFINITION,
        mode="offline_recorded_replay",
        complete=False,
        error=None,
        stream=None,
        replays=[],
        provenance=provenance,
        recorded_outcomes=[
            dict(scene=item.scene, seed=item.seed, outcomes=deepcopy(item.outcomes))
            for item in recordings
        ],
        work=work,
    )
    final_seen = False

    def observer(readout, metadata):
        nonlocal final_seen
        if metadata["phase"] != "after_release" or metadata["packet_index"] != 5:
            return {}
        if final_seen:
            raise RuntimeError("demo final checkpoint was repeated")
        final_seen = True
        pin = _pin_readout(readout)
        if (
            pin[4].completed_updates != 7
            or pin[4].training_examples != bank_rows
            or metadata["completed_updates"] != 7
            or metadata["prototype_rows"] != bank_rows
            or metadata["known_episode_ids"] != [episode.episode_id for episode in episodes]
        ):
            raise RuntimeError("demo final checkpoint differs from seven original TRAIN releases")
        raw_reports: list[tuple[Any, Any, dict[str, Any], str]] = []
        metadata_seal = json.dumps(metadata, sort_keys=True, allow_nan=False)

        def guard():
            _require_readout(readout, pin)
            if json.dumps(metadata, sort_keys=True, allow_nan=False) != metadata_seal:
                raise RuntimeError("recorded replay checkpoint metadata changed")
            for raw, array, copied, seal in raw_reports:
                if (
                    type(raw) is NearestPrediction
                    and raw.actions is not array
                    or json.dumps(_prediction_report(raw), sort_keys=True, allow_nan=False) != seal
                    or json.dumps(copied, sort_keys=True, allow_nan=False) != seal
                ):
                    raise RuntimeError("preceding recorded replay receipt changed")

        for item in recordings:
            guard()
            rows = len(item.episode.transitions)
            entry: dict[str, Any] = dict(
                scene=item.scene,
                seed=item.seed,
                rows=rows,
                complete=False,
                error=None,
                prediction_attempted=True,
                prediction_returned=False,
                prediction=None,
                selected_indices_exact=False,
                native_actions_exact=False,
            )
            report["replays"].append(entry)
            work["prediction_attempts"] += 1
            work["query_rows_attempted"] += rows
            work["distance_candidate_upper_bound"] += rows * bank_rows
            work["distance_feature_upper_bound"] += rows * bank_rows * 12
            work["unknown_unreturned_prediction_work"] = True
            try:
                features = np.asarray(
                    [row.observation for row in item.episode.transitions], dtype=np.float64
                )
                returned = readout.predict_with_work(features)
                work["prediction_returns"] += 1
                work["unknown_unreturned_prediction_work"] = False
                work["unknown_unvalidated_prediction_work"] = True
                work["query_rows_returned"] += rows
                entry["prediction_returned"] = True
                copied = _prediction_report(returned)
                entry["prediction"] = copied
                seal = json.dumps(copied, sort_keys=True, allow_nan=False)
                raw_reports.append(
                    (
                        returned,
                        returned.actions if type(returned) is NearestPrediction else None,
                        copied,
                        seal,
                    )
                )
                guard()
                if (
                    type(returned) is not NearestPrediction
                    or type(returned.distance_candidates) is not int
                    or returned.distance_candidates != rows * bank_rows
                    or type(returned.distance_feature_values) is not int
                    or returned.distance_feature_values != rows * bank_rows * 12
                    or type(returned.selected_indices) is not tuple
                    or len(returned.selected_indices) != rows
                    or any(
                        type(index) is not int or not 0 <= index < bank_rows
                        for index in returned.selected_indices
                    )
                    or type(returned.actions) is not np.ndarray
                    or returned.actions.dtype != np.float64
                    or returned.actions.shape != (rows, 4)
                    or not np.isfinite(returned.actions).all()
                    or np.any(returned.actions < -1.0)
                    or np.any(returned.actions > 1.0)
                ):
                    raise ValueError("recorded replay returned invalid retrieval work/actions")
                work["distance_candidates_completed"] += returned.distance_candidates
                work["distance_feature_values_completed"] += returned.distance_feature_values
                work["unknown_unreturned_prediction_work"] = False
                work["unknown_unvalidated_prediction_work"] = False
                entry["selected_indices_exact"] = returned.selected_indices == item.indices
                actual_bytes = tuple(struct.pack("<4f", *row) for row in returned.actions)
                entry["native_actions_exact"] = actual_bytes == item.action_bytes
                if not entry["selected_indices_exact"] or not entry["native_actions_exact"]:
                    raise ValueError("recorded replay selectors or nativefloat32 commands differ")
                guard()
                entry["complete"] = True
                work["validated_replays"] += 1
            except Exception as error:
                entry["error"] = f"{type(error).__name__}: {error}"
                raise
        guard()
        return dict(offline_recorded_replays=work["validated_replays"], simulator_controls=0)

    try:
        stream = run_sequence(
            episodes[0], episodes[1:], observer, fixed_input=True, maximum_releases=6
        )
        report["stream"] = deepcopy(stream)
        if stream.get("complete") is not True or not final_seen or work["validated_replays"] != 4:
            raise RuntimeError(
                f"memory stream or recorded replay incomplete: {stream.get('error')}"
            )
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return deepcopy(report)
