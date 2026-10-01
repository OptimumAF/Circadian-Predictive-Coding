"""Independently verify saved checkpoint schemas and duplicate state views.

Inputs are JSON fingerprints and frozen family configurations. Outputs are
checked checkpoint facts; no models, datasets, scores or IO are created.
Only initial tensor fingerprints are derived from the declared local RNG.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from typing import Any

import numpy as np

from src.app.continual_confirmation_json import (
    canonical_array,
    canonical_dataclass,
    canonical_mapping,
    canonical_value,
    hash_value,
    integer,
    object_fields,
    require,
    rng_state,
    same_json,
    state_digest,
)
from src.app.continual_confirmation_manifest import ConfirmationFamily
from src.app.continual_confirmation_parameter_links import PARAMETER_CONTRACTS, REPLAY_CONTRACT
from src.app.continual_replay_factor_pilot import _circadian_config
from src.app.continual_sleep_factor_preflight import SLEEP_ARMS, _arm_config
from src.core.circadian_predictive_coding import CircadianConfig


_CPC = "src.core.circadian_predictive_coding.CircadianPredictiveCodingNetwork"
_CONTROLLED = "src.core.controlled_parent_selection.ParentControlledCircadianNetwork"
_SNAPSHOT = "src.core.circadian_predictive_coding.CircadianNetworkSnapshot"
_CONFIG = "src.core.circadian_predictive_coding.CircadianConfig"
_FIELDS = {
    "model_type",
    "parameter_sha256",
    "parameter_sha256_by_contract",
    "state_sha256",
    "width",
    "parameter_count",
    "state",
    "clocks",
    "retention",
    "lineage",
    "selector",
}
_PARAMETERS = ("weight_input_hidden", "bias_hidden", "weight_hidden_output", "bias_output")
_BASELINE_FIELDS = {
    "input_dim",
    "_hidden_weights",
    "_hidden_biases",
    *_PARAMETERS,
    "hidden_dims",
    "_traffic_sums",
    "_traffic_steps",
}
_CPC_FIELDS = {
    "input_dim",
    "config",
    "max_hidden_dim",
    "_rng",
    "_pre_hidden_weights",
    "_pre_hidden_biases",
    *_PARAMETERS,
    "hidden_dims",
    "pre_hidden_dims",
    "_hidden_chemical",
    "_hidden_chemical_fast",
    "_hidden_chemical_slow",
    "_neuron_age",
    "_traffic_sum",
    "_importance_ema",
    "_neuron_ids",
    "_parent_ids",
    "_next_neuron_id",
    "_traffic_steps",
    "_min_hidden_dim",
    "_prune_ttl",
    "_prune_marked",
    "_split_cooldown",
    "_prune_cooldown",
    "_epoch_count",
    "_epochs_since_sleep",
    "_wake_examples",
    "_replay_updates",
    "_sleep_events",
    "_energy_history",
    "_reward_error_ema",
    "_last_reward_scale",
    "_replay_memory",
}
_RETENTION_FIELDS = {
    "_replay_retention_budget",
    "_replay_retention_policy",
    "_replay_observed_ids",
    "_replay_duplicate_ids",
    "_replay_duplicate_occurrences",
    "_replay_exposed_ids",
    "_replay_exposure_updates",
}
_SELECTOR_FIELDS = {
    "_parent_selection_settings",
    "_parent_selection_rng",
    "_parent_selection_cursor",
    "_parent_selection_calls",
    "_last_parent_selection",
}


@dataclass(frozen=True)
class CheckpointDeclaration:
    model_type: str
    seed: int
    initial_width: int
    min_width: int
    max_width: int
    config: dict[str, Any] | None
    retained: bool = False
    parent_mode: str | None = None


def declarations(family: ConfirmationFamily, seed: int) -> dict[str, CheckpointDeclaration]:
    manifest = json.loads(family.development_manifest_json)
    result = {}
    for name in family.arms:
        kind = "backprop" if name.startswith("backprop") else "pc"
        width = 12 if "12" in name else 8
        low = high = width
        config = None
        mode = None
        if family.name in {"combined", "parent"}:
            spec = next(arm for arm in manifest["arms"] if arm["name"] == name)
            kind, width = spec["model_kind"], spec["width"]
            low, high, config = spec["min_width"], spec["max_width"], spec["config"]
            mode = spec.get("parent_mode")
        elif family.name == "gating" and name != "ordinary_pc":
            kind = "circadian"
            config = asdict(
                replace(
                    CircadianConfig.matched_pc_control(),
                    min_plasticity=0.2 if name == "chemical_gating" else 1.0,
                )
            )
        elif family.name == "replay" and name.startswith("circadian"):
            kind, config = "circadian", asdict(_circadian_config(name.endswith("_on")))
        elif family.name == "sleep" and name in SLEEP_ARMS:
            kind, config = "circadian", asdict(_arm_config(name))
            if name == "structure_only":
                low, high = 7, 9
        elif family.name == "schedule" and name.startswith("neutral"):
            kind, config = "circadian", manifest["configs"][name.removeprefix("neutral_")]
        model_type = (
            _CONTROLLED
            if mode is not None
            else _CPC
            if kind == "circadian"
            else "src.core.backprop_mlp.BackpropMLP"
            if kind == "backprop"
            else "src.core.predictive_coding.PredictiveCodingNetwork"
        )
        result[name] = CheckpointDeclaration(
            model_type,
            seed,
            width,
            low,
            high,
            config,
            kind == "circadian" and family.name in {"replay", "schedule", "combined", "parent"},
            mode,
        )
    return result


def _array_fingerprint(value: np.ndarray) -> dict[str, Any]:
    return {
        "array_dtype": value.dtype.str,
        "shape": list(value.shape),
        "bytes_sha256": sha256(value.tobytes(order="C")).hexdigest(),
    }


def _initial_fingerprints(seed: int, width: int) -> tuple[dict[str, Any], dict[str, str]]:
    rng = np.random.default_rng(seed + 1001)
    arrays = (
        rng.normal(0.0, 0.5, size=(2, width)),
        np.zeros((1, width)),
        rng.normal(0.0, 0.5, size=(width, 1)),
        np.zeros((1, 1)),
    )
    digests = {contract: sha256(contract.encode("ascii")) for contract in PARAMETER_CONTRACTS}
    fingerprints = {}
    for name, array in zip(_PARAMETERS, arrays, strict=True):
        fingerprints[name] = _array_fingerprint(array)
        for digest in digests.values():
            digest.update(name.encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.astype("<f8").tobytes())
    return fingerprints, {contract: digest.hexdigest() for contract, digest in digests.items()}


def verify_checkpoint(row: dict[str, Any], declared: CheckpointDeclaration, stage: str) -> None:
    object_fields(row, _FIELDS, "checkpoint")
    canonical_value(row["state"])
    require(row["model_type"] == declared.model_type, "checkpoint model kind differs")
    width = integer(row["width"], "checkpoint width", 1)
    require(
        declared.min_width <= width <= declared.max_width,
        "checkpoint width exceeds declared bounds",
    )
    require(
        integer(row["parameter_count"], "checkpoint parameters", 1) == 4 * width + 1,
        "checkpoint parameter count differs",
    )
    hash_value(row["parameter_sha256"], "checkpoint parameters")
    hashes = object_fields(
        row["parameter_sha256_by_contract"], set(PARAMETER_CONTRACTS), "parameter hash contracts"
    )
    for contract, value in hashes.items():
        hash_value(value, f"checkpoint parameters/{contract}")
    require(
        row["parameter_sha256"] == hashes[REPLAY_CONTRACT],
        "uniform parameter hash link differs",
    )
    require(
        hash_value(row["state_sha256"], "checkpoint state") == state_digest(row["state"]),
        "canonical state digest differs",
    )
    if declared.config is None:
        owner = canonical_mapping(row["state"])
        _verify_baseline(row, owner, width, declared)
    else:
        snapshot = canonical_dataclass(row["state"], _SNAPSHOT)
        expected = {
            "format_version": 2,
            "input_dim": 2,
            "initial_hidden_dims": {"tuple": [declared.initial_width]},
            "min_hidden_dim": declared.min_width,
            "max_hidden_dim": declared.max_width,
        }
        for key, value in expected.items():
            same_json(snapshot[key], value, f"snapshot/{key}")
        same_json(
            canonical_dataclass(snapshot["config"], _CONFIG),
            declared.config,
            "snapshot configuration",
        )
        owner = canonical_mapping(snapshot["state"])
        _verify_circadian(row, owner, width, declared, stage)
    _verify_parameters(owner, width)
    if stage == "initial":
        expected, initial_hashes = _initial_fingerprints(declared.seed, declared.initial_width)
        require(
            width == declared.initial_width and hashes == initial_hashes,
            "seeded initial parameter hash differs",
        )
        for name, value in expected.items():
            same_json(owner[name], value, f"seeded initial/{name}")
        _verify_initial_state(row, owner, declared)


def _verify_initial_state(
    row: dict[str, Any], owner: dict[str, Any], declared: CheckpointDeclaration
) -> None:
    same_json(owner["_traffic_steps"], 0, "initial traffic count")
    zero = _array_fingerprint(np.zeros(declared.initial_width))
    if declared.config is None:
        same_json(owner["_traffic_sums"], {"list": [zero]}, "initial baseline traffic")
        return
    for key in (
        "_hidden_chemical",
        "_hidden_chemical_fast",
        "_hidden_chemical_slow",
        "_neuron_age",
        "_traffic_sum",
        "_importance_ema",
    ):
        same_json(owner[key], zero, "initial circadian vector")
    same_json(
        row["clocks"],
        dict.fromkeys(
            (
                "wake_batches",
                "wake_examples",
                "wake_batches_since_sleep",
                "replay_updates",
                "sleep_events",
            ),
            0,
        ),
        "initial clocks",
    )
    same_json(
        rng_state(owner["_rng"]),
        np.random.default_rng(declared.seed + 11002).bit_generator.state,
        "initial noise RNG",
    )
    same_json(owner["_energy_history"], {"list": []}, "initial energy history")
    require(
        owner["_reward_error_ema"] is None and owner["_last_reward_scale"] == 1.0,
        "initial reward state differs",
    )
    lineage = canonical_dataclass(
        row["lineage"], "src.core.neuron_adaptation.NeuronLineageSnapshot"
    )
    same_json(
        lineage,
        {
            "neuron_ids": {"tuple": list(range(declared.initial_width))},
            "parent_ids": {"tuple": [None] * declared.initial_width},
            "next_neuron_id": declared.initial_width,
        },
        "initial lineage",
    )
    if declared.parent_mode is not None:
        same_json(
            rng_state(owner["_parent_selection_rng"]),
            np.random.Generator(np.random.PCG64(declared.seed + 5001)).bit_generator.state,
            "initial selector RNG",
        )
        require(
            row["selector"]["selection_calls"] == 0
            and row["selector"]["cursor_id"] == 0
            and row["selector"]["last_decision"] is None,
            "initial selector view differs",
        )


def _verify_parameters(owner: dict[str, Any], width: int) -> None:
    require(
        owner["input_dim"] == 2 and type(owner["input_dim"]) is int, "state input dimension differs"
    )
    for name, shape in zip(_PARAMETERS, ((2, width), (1, width), (width, 1), (1, 1)), strict=True):
        canonical_array(owner[name], shape)


def _verify_baseline(
    row: dict[str, Any], owner: dict[str, Any], width: int, declared: CheckpointDeclaration
) -> None:
    object_fields(owner, _BASELINE_FIELDS, "baseline state")
    require(
        all(row[key] is None for key in ("clocks", "retention", "lineage", "selector")),
        "baseline has circadian views",
    )
    same_json(owner["hidden_dims"], {"tuple": [declared.initial_width]}, "baseline initial shape")
    same_json(
        owner["_hidden_weights"],
        {"list": [owner["weight_input_hidden"]]},
        "baseline weight alias fingerprint",
    )
    same_json(
        owner["_hidden_biases"], {"list": [owner["bias_hidden"]]}, "baseline bias alias fingerprint"
    )
    traffic = object_fields(owner["_traffic_sums"], {"list"}, "baseline traffic")["list"]
    require(type(traffic) is list and len(traffic) == 1, "baseline traffic layers differ")
    canonical_array(traffic[0], (width,))
    integer(owner["_traffic_steps"], "baseline traffic steps")


def _verify_circadian(
    row: dict[str, Any],
    owner: dict[str, Any],
    width: int,
    declared: CheckpointDeclaration,
    stage: str,
) -> None:
    expected = (
        _CPC_FIELDS
        | (_RETENTION_FIELDS if declared.retained else set())
        | (_SELECTOR_FIELDS if declared.parent_mode is not None else set())
    )
    object_fields(owner, expected, "circadian state")
    same_json(canonical_dataclass(owner["config"], _CONFIG), declared.config, "state configuration")
    same_json(owner["hidden_dims"], {"tuple": [declared.initial_width]}, "state initial width")
    same_json(owner["pre_hidden_dims"], {"tuple": []}, "state pre-hidden shape")
    for key in ("_pre_hidden_weights", "_pre_hidden_biases"):
        same_json(owner[key], {"list": []}, "state shallow layers")
    same_json(owner["_min_hidden_dim"], declared.min_width, "state minimum width")
    same_json(owner["max_hidden_dim"], declared.max_width, "state maximum width")
    for key in (
        "_hidden_chemical",
        "_hidden_chemical_fast",
        "_hidden_chemical_slow",
        "_neuron_age",
        "_traffic_sum",
        "_importance_ema",
    ):
        canonical_array(owner[key], (width,))
    for key in ("_prune_ttl", "_split_cooldown", "_prune_cooldown"):
        canonical_array(owner[key], (width,), "<i4")
    canonical_array(owner["_prune_marked"], (width,), "|b1")
    integer(owner["_traffic_steps"], "circadian traffic steps")
    rng_state(owner["_rng"])
    _verify_adaptive_values(owner, declared)
    _verify_clocks(row, owner, stage)
    _verify_lineage(row, owner, width)
    _verify_retention(row, owner, declared, stage)
    _verify_selector(row, owner, declared)


def _verify_adaptive_values(owner: dict[str, Any], declared: CheckpointDeclaration) -> None:
    history = object_fields(owner["_energy_history"], {"list"}, "energy history")["list"]
    require(
        type(history) is list
        and all(type(value) in (int, float) and value >= 0 for value in history),
        "energy history values differ",
    )
    baseline = owner["_reward_error_ema"]
    require(
        baseline is None or type(baseline) in (int, float) and 0 <= baseline <= 1,
        "reward baseline value differs",
    )
    scale = owner["_last_reward_scale"]
    require(type(scale) in (int, float) and scale > 0, "reward scale type/value differs")
    assert declared.config is not None
    if declared.config["use_reward_modulated_learning"]:
        require(
            declared.config["reward_scale_min"] <= scale <= declared.config["reward_scale_max"],
            "reward scale exceeds declared bounds",
        )
    else:
        require(baseline is None and scale == 1.0, "disabled reward state differs")


def _verify_clocks(row: dict[str, Any], owner: dict[str, Any], stage: str) -> None:
    fields = {
        "wake_batches": "_epoch_count",
        "wake_examples": "_wake_examples",
        "wake_batches_since_sleep": "_epochs_since_sleep",
        "replay_updates": "_replay_updates",
        "sleep_events": "_sleep_events",
    }
    expected = {key: integer(owner[value], f"clock/{key}") for key, value in fields.items()}
    same_json(row["clocks"], expected, "clock view/state link")
    counts = {"initial": (0, 0), "after_a": (12, 864), "after_b": (24, 1296)}
    if stage in counts:
        require(
            (expected["wake_batches"], expected["wake_examples"]) == counts[stage],
            "endpoint wake clocks differ",
        )
    require(
        expected["wake_batches_since_sleep"] <= expected["wake_batches"],
        "sleep spacing exceeds wake count",
    )


def _verify_lineage(row: dict[str, Any], owner: dict[str, Any], width: int) -> None:
    lineage = canonical_dataclass(
        row["lineage"], "src.core.neuron_adaptation.NeuronLineageSnapshot"
    )
    ids = object_fields(lineage["neuron_ids"], {"tuple"}, "lineage IDs")["tuple"]
    parents = object_fields(lineage["parent_ids"], {"tuple"}, "lineage parents")["tuple"]
    require(
        type(ids) is list and type(parents) is list and len(ids) == len(parents) == width,
        "lineage width differs",
    )
    for value in ids:
        integer(value, "lineage ID")
    require(ids == sorted(set(ids)), "lineage ID order/uniqueness differs")
    for child, parent in zip(ids, parents, strict=True):
        require(
            parent is None or type(parent) is int and 0 <= parent < child, "lineage parent differs"
        )
    require(
        integer(lineage["next_neuron_id"], "next lineage ID") > max(ids), "next lineage ID differs"
    )
    same_json(owner["_next_neuron_id"], lineage["next_neuron_id"], "lineage next ID/state link")
    same_json(
        owner["_neuron_ids"],
        _array_fingerprint(np.asarray(ids, dtype="<i8")),
        "lineage array/state link",
    )
    same_json(
        owner["_parent_ids"],
        _array_fingerprint(
            np.asarray([-1 if item is None else item for item in parents], dtype="<i8")
        ),
        "parent array/state link",
    )


def _verify_retention(
    row: dict[str, Any], owner: dict[str, Any], declared: CheckpointDeclaration, stage: str
) -> None:
    memory = object_fields(owner["_replay_memory"], {"deque", "maxlen"}, "memory deque")
    require(type(memory["deque"]) is list, "memory rows differ")
    if not declared.retained:
        require(
            row["retention"] is None and memory == {"deque": [], "maxlen": 0},
            "zero-memory state/view differs",
        )
        return
    same_json(
        canonical_dataclass(
            owner["_replay_retention_budget"],
            "src.core.circadian_predictive_coding.ReplayRetentionBudget",
        ),
        {"max_examples": 8, "max_bytes": 192},
        "retention budget",
    )
    same_json(
        canonical_dataclass(
            owner["_replay_retention_policy"], "src.core.replay_retention.ReplayRetentionPolicy"
        ),
        {"name": "recent_fifo", "seed": None},
        "retention policy",
    )
    require(
        memory["maxlen"] is None and len(memory["deque"]) == (0 if stage == "initial" else 8),
        "retention row count differs",
    )
    view = object_fields(
        row["retention"], {"sample_ids", "example_count", "retained_bytes"}, "retention view"
    )
    ids = view["sample_ids"]
    require(type(ids) is list and ids == sorted(set(ids)), "retention IDs differ")
    for value in ids:
        hash_value(value, "retention ID")
    require(
        integer(view["example_count"], "retention count") == len(ids) == len(memory["deque"])
        and integer(view["retained_bytes"], "retention bytes") == 24 * len(ids),
        "retention storage/view link differs",
    )
    for item in memory["deque"]:
        data = canonical_dataclass(item, "src.core.circadian_predictive_coding.ReplaySnapshot")
        canonical_array(data["input_batch"], (1, 2))
        canonical_array(data["target_batch"], (1, 1))
        require(
            type(data["priority"]) in (int, float)
            and 0 <= data["priority"] <= 1
            and type(data["positive_fraction"]) in (int, float)
            and data["positive_fraction"] in (0, 1),
            "retained row diagnostics differ",
        )
    exposed = object_fields(owner["_replay_exposed_ids"], {"set"}, "exposed IDs")["set"]
    observed = object_fields(owner["_replay_observed_ids"], {"set"}, "observed IDs")["set"]
    duplicates = object_fields(owner["_replay_duplicate_ids"], {"set"}, "duplicate IDs")["set"]
    for value in (*observed, *exposed, *duplicates):
        hash_value(value, "exposure ID")
    require(
        set(ids) <= set(observed)
        and set(exposed) <= set(observed)
        and set(duplicates) <= set(observed),
        "retention/exposure supply differs",
    )
    integer(owner["_replay_duplicate_occurrences"], "duplicate occurrences")
    same_json(
        owner["_replay_exposure_updates"],
        row["clocks"]["replay_updates"],
        "exposure/replay clock link",
    )


def _verify_selector(
    row: dict[str, Any], owner: dict[str, Any], declared: CheckpointDeclaration
) -> None:
    if declared.parent_mode is None:
        require(row["selector"] is None, "nonselector model has selector view")
        return
    view = object_fields(
        row["selector"],
        {"settings", "cursor_id", "selection_calls", "rng_sha256", "last_decision"},
        "selector view",
    )
    settings = {"mode": declared.parent_mode, "seed": declared.seed + 5001, "initial_cursor_id": 0}
    same_json(view["settings"], settings, "selector settings")
    same_json(
        canonical_dataclass(
            owner["_parent_selection_settings"],
            "src.core.controlled_parent_selection.ParentSelectionSettings",
        ),
        settings,
        "selector settings/state link",
    )
    same_json(
        view["cursor_id"],
        integer(owner["_parent_selection_cursor"], "selector cursor"),
        "selector cursor/state link",
    )
    same_json(
        view["selection_calls"],
        integer(owner["_parent_selection_calls"], "selector calls"),
        "selector calls/state link",
    )
    require(view["selection_calls"] <= 5, "selector calls exceed planned growth")
    if declared.parent_mode != "scheduled":
        require(view["cursor_id"] == 0, "nonscheduled selector cursor differs")
    if view["selection_calls"] == 0 or declared.parent_mode != "random":
        same_json(
            rng_state(owner["_parent_selection_rng"]),
            np.random.Generator(np.random.PCG64(declared.seed + 5001)).bit_generator.state,
            "initial/nonrandom selector RNG",
        )
    encoded = json.dumps(
        rng_state(owner["_parent_selection_rng"]), sort_keys=True, allow_nan=False
    ).encode("utf-8")
    require(
        hash_value(view["rng_sha256"], "selector RNG") == sha256(encoded).hexdigest(),
        "selector RNG/state link differs",
    )
    decision = owner["_last_parent_selection"]
    if decision is None:
        require(
            view["last_decision"] is None
            and view["selection_calls"] == 0
            and view["cursor_id"] == 0,
            "initial selector decision differs",
        )
    else:
        facts = canonical_dataclass(
            decision, "src.core.controlled_parent_selection.ParentSelectionDecision"
        )
        decoded = {
            key: value["tuple"] if type(value) is dict and set(value) == {"tuple"} else value
            for key, value in facts.items()
        }
        same_json(view["last_decision"], decoded, "selector decision/state link")
        for key in ("cursor_before", "cursor_after"):
            integer(decoded[key], "selector decision cursor")
        for key in ("rng_sha256_before", "rng_sha256_after"):
            hash_value(decoded[key], "selector decision RNG")
        for key in ("eligible_parent_ids", "preferred_parent_ids", "selected_parent_ids"):
            require(type(decoded[key]) is list, "selector decision ID group differs")
            for item in decoded[key]:
                integer(item, "selector decision ID")
            require(len(set(decoded[key])) == len(decoded[key]), "selector decision duplicate IDs")
        require(
            set(decoded["preferred_parent_ids"]) <= set(decoded["eligible_parent_ids"])
            and set(decoded["selected_parent_ids"]) <= set(decoded["eligible_parent_ids"])
            and len(decoded["selected_parent_ids"]) == 1,
            "selector decision eligibility/count differs",
        )
        require(
            view["selection_calls"] > 0
            and decoded["mode"] == declared.parent_mode
            and decoded["cursor_after"] == view["cursor_id"]
            and decoded["rng_sha256_after"] == view["rng_sha256"],
            "selector latest decision link differs",
        )
