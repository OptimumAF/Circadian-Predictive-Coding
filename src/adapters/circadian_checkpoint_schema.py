"""Explicit circadian native variants and scalar/topology validation.

No persistence, native updates, callback loading or recovery authority. The
complete codec supplies detached arrays before native consistency validation.
"""

from dataclasses import fields
from math import isfinite
import numpy as np
from src.core.checkpoint_codec import CodecBinding, CodecLimits, require_codec_digest
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    WAKE_ONLY_REPLAY_SIDE_EFFECT_POLICY,
)
from src.core.replay_retention import ReplayRetentionPolicy, DEFAULT_REPLAY_RETENTION_POLICY
from src.adapters.numpy_checkpoint_frames import exact_object

CONFIG_TYPES = {
    "chemical_decay": "float",
    "chemical_buildup_rate": "float",
    "use_saturating_chemical": "bool",
    "chemical_max_value": "float",
    "chemical_saturation_gain": "float",
    "use_dual_chemical": "bool",
    "dual_fast_mix": "float",
    "slow_chemical_decay": "float",
    "slow_buildup_scale": "float",
    "plasticity_sensitivity": "float",
    "use_adaptive_plasticity_sensitivity": "bool",
    "plasticity_sensitivity_min": "float",
    "plasticity_sensitivity_max": "float",
    "plasticity_importance_mix": "float",
    "min_plasticity": "float",
    "use_reward_modulated_learning": "bool",
    "reward_baseline_decay": "float",
    "reward_difficulty_exponent": "float",
    "reward_scale_min": "float",
    "reward_scale_max": "float",
    "use_adaptive_thresholds": "bool",
    "adaptive_split_percentile": "float",
    "adaptive_prune_percentile": "float",
    "split_threshold": "float",
    "prune_threshold": "float",
    "split_hysteresis_margin": "float",
    "prune_hysteresis_margin": "float",
    "split_cooldown_epochs": "int",
    "prune_cooldown_epochs": "int",
    "split_weight_norm_mix": "float",
    "prune_weight_norm_mix": "float",
    "split_importance_mix": "float",
    "prune_importance_mix": "float",
    "importance_ema_decay": "float",
    "max_split_per_sleep": "int",
    "max_prune_per_sleep": "int",
    "split_noise_scale": "float",
    "sleep_reset_factor": "float",
    "sleep_warmup_steps": "int",
    "sleep_split_only_until_fraction": "float",
    "sleep_prune_only_after_fraction": "float",
    "sleep_max_change_fraction": "float",
    "sleep_min_change_count": "int",
    "prune_min_age_steps": "int",
    "use_adaptive_sleep_trigger": "bool",
    "min_epochs_between_sleep": "int",
    "sleep_energy_window": "int",
    "sleep_plateau_delta": "float",
    "sleep_chemical_variance_threshold": "float",
    "use_adaptive_sleep_budget": "bool",
    "adaptive_sleep_budget_min_scale": "float",
    "adaptive_sleep_budget_max_scale": "float",
    "adaptive_sleep_budget_plateau_weight": "float",
    "adaptive_sleep_budget_variance_weight": "float",
    "prune_decay_steps": "int",
    "prune_decay_factor": "float",
    "homeostatic_downscale_factor": "float",
    "homeostasis_target_input_norm": "float",
    "homeostasis_target_output_norm": "float",
    "homeostasis_strength": "float",
    "replay_steps": "int",
    "replay_memory_size": "int",
    "replay_learning_rate": "float",
    "replay_inference_steps": "int",
    "replay_inference_learning_rate": "float",
    "replay_prioritized": "bool",
    "replay_class_balanced": "bool",
    "sleep_mode": "str",
    "sleep_enable_chemical_reset": "bool",
    "sleep_enable_replay": "bool",
    "sleep_enable_homeostasis": "bool",
    "sleep_enable_split": "bool",
    "sleep_enable_prune": "bool",
}
FLOAT_VECTORS = (
    "_hidden_chemical",
    "_hidden_chemical_fast",
    "_hidden_chemical_slow",
    "_neuron_age",
    "_traffic_sum",
    "_importance_ema",
)
INT_VECTORS = {
    "_neuron_ids": "<i8",
    "_parent_ids": "<i8",
    "_prune_ttl": "<i4",
    "_prune_marked": "|b1",
    "_split_cooldown": "<i4",
    "_prune_cooldown": "<i4",
}
PARAMETERS = ("weight_input_hidden", "bias_hidden", "weight_hidden_output", "bias_output")
COUNTERS = (
    "_next_neuron_id",
    "_traffic_steps",
    "_epoch_count",
    "_epochs_since_sleep",
    "_wake_examples",
    "_replay_updates",
    "_sleep_events",
)
BASE_FIELDS = {
    "_traffic_sum",
    "_replay_memory",
    "_reward_error_ema",
    "weight_hidden_output",
    "max_hidden_dim",
    "_hidden_chemical",
    "config",
    "weight_input_hidden",
    "_energy_history",
    "_importance_ema",
    "_hidden_chemical_slow",
    "_sleep_events",
    "_traffic_steps",
    "_last_reward_scale",
    "_next_neuron_id",
    "_wake_examples",
    "_replay_updates",
    "pre_hidden_dims",
    "_pre_hidden_weights",
    "_pre_hidden_biases",
    "_prune_cooldown",
    "input_dim",
    "_prune_marked",
    "_neuron_ids",
    "_min_hidden_dim",
    "_neuron_age",
    "_rng",
    "_hidden_chemical_fast",
    "bias_hidden",
    "bias_output",
    "_split_cooldown",
    "hidden_dims",
    "_epoch_count",
    "_parent_ids",
    "_prune_ttl",
    "_epochs_since_sleep",
}
EXPOSURE = {
    "_replay_retention_policy",
    "_replay_observed_ids",
    "_replay_duplicate_ids",
    "_replay_duplicate_occurrences",
    "_replay_exposed_ids",
    "_replay_exposure_updates",
}
IDS = ("_replay_observed_ids", "_replay_duplicate_ids", "_replay_exposed_ids")


def bounded_integer(value, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("checkpoint requires bounded exact integer")
    return value


def finite_scalar(value, minimum=None, maximum=None):
    if (
        type(value) not in (int, float)
        or not isfinite(value)
        or (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ):
        raise ValueError("checkpoint requires finite supported scalar")


def validate_configuration(binding, limits):
    if type(binding) is not CodecBinding or type(limits) is not CodecLimits:
        raise ValueError("checkpoint requires exact binding/limits")
    CodecBinding.__post_init__(binding)
    CodecLimits.__post_init__(limits)


def decode_config(data):
    exact_object(data, CONFIG_TYPES.keys())
    if {f.name for f in fields(CircadianConfig)} != CONFIG_TYPES.keys():
        raise ValueError("native config changed without codec version change")
    for name, kind in CONFIG_TYPES.items():
        value = data[name]
        if kind == "float":
            finite_scalar(value)
        elif kind == "int":
            bounded_integer(value)
        elif kind == "bool" and type(value) is not bool:
            raise ValueError("config switch requires boolean")
        elif kind == "str" and type(value) is not str:
            raise ValueError("config mode requires string")
    config = CircadianConfig(**data)
    CircadianPredictiveCodingNetwork._validate_config(
        object.__new__(CircadianPredictiveCodingNetwork), config
    )
    return config


def validate_rng(data):
    exact_object(data, {"bit_generator", "state", "has_uint32", "uinteger"})
    if data["bit_generator"] != "PCG64":
        raise ValueError("unsupported native generator schema")
    exact_object(data["state"], {"state", "inc"})
    bounded_integer(data["state"]["state"], maximum=2**128 - 1)
    inc = bounded_integer(data["state"]["inc"], minimum=1, maximum=2**128 - 1)
    if not inc % 2:
        raise ValueError("PCG64 stream increment must be odd")
    bounded_integer(data["has_uint32"], maximum=1)
    bounded_integer(data["uinteger"], maximum=2**32 - 1)


def state_fields(data):
    if type(data) is not dict:
        raise ValueError("native state requires dictionary")
    expected = BASE_FIELDS.copy()
    if "_replay_retention_budget" in data:
        expected.add("_replay_retention_budget")
    if "_replay_retention_policy" in data:
        if "_replay_retention_budget" not in data:
            raise ValueError("exposure policy requires retention")
        expected |= EXPOSURE
    if "_replay_side_effect_policy" in data:
        expected.add("_replay_side_effect_policy")
    exact_object(data, expected)
    return expected


def validate_wire_scalars(meta, data, limits):
    exact_object(
        meta,
        {
            "format_version",
            "input_dim",
            "initial_hidden_dims",
            "min_hidden_dim",
            "max_hidden_dim",
            "config",
        },
    )
    bounded_integer(meta["format_version"], 2, 2)
    for name in ("input_dim", "min_hidden_dim", "max_hidden_dim"):
        bounded_integer(meta[name], 1, limits.max_dimension)
    dims = meta["initial_hidden_dims"]
    if type(dims) is not list or not 0 < len(dims) <= limits.max_layers:
        raise ValueError("unsupported topology layers")
    for value in dims:
        bounded_integer(value, 1, limits.max_dimension)
    if dims[-1] > meta["max_hidden_dim"]:
        raise ValueError("initial width exceeds native maximum")
    state_fields(data)
    if (
        type(data["input_dim"]) is not int
        or data["input_dim"] != meta["input_dim"]
        or type(data["_min_hidden_dim"]) is not int
        or data["_min_hidden_dim"] != meta["min_hidden_dim"]
        or type(data["max_hidden_dim"]) is not int
        or data["max_hidden_dim"] != meta["max_hidden_dim"]
        or data["hidden_dims"] != dims
        or data["pre_hidden_dims"] != dims[:-1]
    ):
        raise ValueError("native topology metadata differs")
    if any(type(v) is not int for v in data["hidden_dims"] + data["pre_hidden_dims"]):
        raise ValueError("native dimensions require exact integers")
    if decode_config(meta["config"]) != decode_config(data["config"]):
        raise ValueError("native configuration differs")
    for name in COUNTERS:
        bounded_integer(data[name], 1 if name == "_next_neuron_id" else 0)
    if (
        data["_epochs_since_sleep"] > data["_epoch_count"]
        or data["_wake_examples"] < data["_epoch_count"]
    ):
        raise ValueError("native wake chronology differs")
    history = data["_energy_history"]
    if type(history) is not list or len(history) > limits.max_encoded_bytes:
        raise ValueError("unsupported bounded energy history")
    for value in history:
        finite_scalar(value)
    if data["_reward_error_ema"] is not None:
        finite_scalar(data["_reward_error_ema"])
    finite_scalar(data["_last_reward_scale"], minimum=0)
    validate_rng(data["_rng"])
    if (
        "_replay_side_effect_policy" in data
        and data["_replay_side_effect_policy"] != WAKE_ONLY_REPLAY_SIDE_EFFECT_POLICY
    ):
        raise ValueError("unsupported replay side-effect policy")
    if "_replay_retention_budget" in data:
        value = exact_object(data["_replay_retention_budget"], {"max_examples", "max_bytes"})
        for v in value.values():
            bounded_integer(v, 1)
    if "_replay_retention_policy" in data:
        value = exact_object(data["_replay_retention_policy"], {"name", "seed"})
        policy = ReplayRetentionPolicy(**value)
        if policy == DEFAULT_REPLAY_RETENTION_POLICY:
            raise ValueError("default policy has no native exposure fields")
        for name in IDS:
            ids = data[name]
            if (
                type(ids) is not list
                or len(ids) > limits.max_encoded_bytes
                or ids != sorted(set(ids))
            ):
                raise ValueError("exposure IDs require bounded canonical set")
            for identity in ids:
                require_codec_digest(identity)
        for name in ("_replay_duplicate_occurrences", "_replay_exposure_updates"):
            bounded_integer(data[name])


def validate_complete_native(snapshot):
    data = snapshot.state
    candidate = object.__new__(CircadianPredictiveCodingNetwork)
    candidate.__dict__.update(data)
    candidate._validate_sleep_post_state()
    for item in data["_replay_memory"]:
        if np.any((item.target_batch < 0) | (item.target_batch > 1)):
            raise ValueError("replay targets outside original binary/soft label range")
    for name in FLOAT_VECTORS:
        if np.any(data[name] < 0):
            raise ValueError("native accumulated array cannot be negative")
    for name in ("_split_cooldown", "_prune_cooldown"):
        if np.any(data[name] < 0):
            raise ValueError("native cooldown cannot be negative")
    if candidate.hidden_dim > snapshot.max_hidden_dim:
        raise ValueError("native width exceeds original maximum")
    if "_replay_retention_budget" in data:
        retained = candidate.get_replay_retention()
        budget = data["_replay_retention_budget"]
        if (
            retained.example_count > budget.max_examples
            or retained.retained_bytes > budget.max_bytes
            or len(set(retained.sample_ids)) != retained.example_count
        ):
            raise ValueError("native retention violates original budget/identity")
    if "_replay_retention_policy" in data:
        exposure = candidate.get_replay_exposure()
        if exposure.replay_updates > data["_replay_updates"] or not set(retained.sample_ids) <= set(
            exposure.observed_ids
        ):
            raise ValueError("native exposure/retention history differs")
