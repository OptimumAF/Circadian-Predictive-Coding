"""Explicit weighted probe recipes for the existing MiniGrid R2.1 pilot.

Inputs are a declared recipe, owned warm checkpoint and released F/TRAIN packets.
Outputs are a disposable scheduled source and scalar cost metadata. No benchmark
updates, environment, final data, metric selection, budget or IO lives here.
"""

from dataclasses import asdict
from typing import Any

import numpy as np

from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureSnapshot
from src.adapters.torch_pc_inference import TORCH_BOUNDED_WEIGHTED_PC_ID
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner
from src.app.coupled_attribution import AttributionBatch, _validate_batch
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.demonstration_episode import DemonstrationEpisode, episode_arrays

MODES = ("fixed-weighted", "bounded-weighted")
METHODS = ("pc", "neutral_cpc", "pc_relax64", "neutral_cpc_relax64")


def policy(mode: str, max_steps: int) -> CoupledInferencePolicy:
    """Same kernel/objective/safety rule; only stopping configuration differs."""
    if type(mode) is not str or mode not in MODES:
        raise ValueError("probe_inference requires fixed-weighted or bounded-weighted")
    if type(max_steps) is not int or max_steps not in (2, 64):
        raise ValueError("weighted MiniGrid probe requires original native cap2 or64")
    return CoupledInferencePolicy(
        min_steps=max_steps if mode == "fixed-weighted" else 1,
        gradient_tolerance=None if mode == "fixed-weighted" else 1e-6,
        relative_energy_tolerance=None,
    )


def validate_option(mode: str | None, **options: Any) -> None:
    if mode is None:
        return
    if type(mode) is not str or mode not in MODES:
        raise ValueError("probe_inference requires an explicit weighted probe recipe")
    if options.get("wide_neural", False) is True and (
        type(options.get("qualification_method")) is not str
        or options["qualification_method"] not in ("pc", "neutral_cpc")
        or any(
            options.get(name, False) is not False
            for name in ("old_a_rehearsal", "extra_f_exposure", "extended_wide_probe_budget")
        )
    ):
        raise ValueError(
            "H32 weighted probes require original PC/neutral without replay or exposure"
        )
    required = (
        "public_history",
        "invariant_history",
        "joint_catalogue",
        "distinct_probe",
        "old_task_readouts",
        "return_to_a",
    )
    forbidden = (
        "pc_qualification",
        "acquisition_only",
        "memory_control",
        "weighted_bp",
        "weighted_pc",
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
        type(options.get("qualification_method")) is not str
        or options["qualification_method"] not in METHODS
        or any(options.get(name) is not True for name in required)
        or any(options.get(name, False) is not False for name in forbidden)
    ):
        raise ValueError(
            "probe_inference requires explicit PC/neutral joint invariant F16/readouts/A16 without extra diagnostics"
        )


def describe(
    protocol: dict[str, Any], mode: str, method: str, *, wide_neural: bool = False
) -> None:
    if type(wide_neural) is not bool:
        raise ValueError("wide_neural must be an exact bool")
    cap = 64 if method.endswith("_relax64") else 2
    configuration = policy(mode, cap)
    protocol["protocol_id"] = (
        protocol["protocol_id"].removesuffix("_v1") + "_" + mode.replace("-", "_") + "_probe_v1"
    )
    protocol["probe_inference"] = dict(
        mode=mode,
        policy=asdict(configuration),
        max_steps=cap,
        objective=TORCH_BOUNDED_WEIGHTED_PC_ID,
        weighting="current complete original probe_train episode; TRAIN episode only during explicit return; n/(present_classes*class_count), absent1",
        comparison="fixed-weighted and bounded-weighted share objective, max cap, initial state, native policy, source order and safety fallback; original unweighted controls remain separate",
        safety="fixed-weighted configures min=max with tolerances disabled; unsafe/nonfinite proposals may still stop early at the last accepted latent state",
        isolation="owned warm full-state source; existing runner trains only its fork; original benchmark/source/actor/budgets/RNG remain unchanged",
        criterion="bounded gradient1e-6 is predeclared absolute scaled-gradient tolerance, not fitted to outcomes; no convergence or learning gain assumed",
        storage="at most32 owned complete episode sources and current full native state per probe owner; existing copies/validation/publication add wall time and RSS, no success-record chain",
    )
    if wide_neural:
        protocol["probe_inference"]["native_recipe"] = dict(
            method=method,
            feature_dim=296,
            hidden_dim=32,
            num_classes=7,
            seed=17,
            learning_rate=0.03,
            momentum=0.9,
            inference_steps=2,
            inference_learning_rate=0.04,
            scope="benchmark unchanged; class-weighted bounded objective only on isolated F/A",
        )
    protocol["probe"]["objective"] = TORCH_BOUNDED_WEIGHTED_PC_ID
    protocol["return_phase"]["objective"] = TORCH_BOUNDED_WEIGHTED_PC_ID
    for key in ("weighted_catalogue", "weighted_pc"):
        if key in protocol:
            protocol[key]["probe_objective"] = TORCH_BOUNDED_WEIGHTED_PC_ID
            protocol[key]["isolation"] = protocol["probe_inference"]["isolation"]
            protocol[key]["diagnostic"] = (
                "benchmark objective unchanged; F/return report weighted CE plus hidden energy and owned stopping/native32/outer7 diagnostics"
            )
    protocol["planned_work"]["probe_inference"] = dict(
        native_train_calls=64,
        training_rows=576,
        configured_latent_row_iterations=576 * cap,
        numerical_work="actual attempted/accepted/rejected and native32/outer7 costs retained separately; copies and scalar reductions additional",
    )
    protocol["limitations"] += (
        "; weighted probe recipe changes probe objective versus original unweighted controls; only fixed-weighted versus bounded-weighted isolates stopping configuration; no final-data or circadian benefit claim"
    )


def _episodes(packets: tuple[AttributionBatch, ...], role: str) -> tuple[DemonstrationEpisode, ...]:
    seen: set[str] = set()
    result = []
    for packet in packets:
        _validate_batch(packet, role, seen)
        if packet.episode is None or packet.features.shape != (9, 296) or packet.num_classes != 7:
            raise ValueError("weighted probe requires complete nine-row public invariant episodes")
        expected = episode_arrays(packet.episode)
        if any(
            not np.array_equal(a, b)
            for a, b in zip((packet.features, packet.targets), expected, strict=True)
        ):
            raise ValueError("weighted probe arrays differ from original episode sources")
        result.append(packet.episode)
    return tuple(result)


def build_source(
    state: InvariantTorchFeatureSnapshot,
    data: tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]],
    return_training: tuple[AttributionBatch, ...],
    mode: str,
) -> TorchTrainWeightedLearner:
    if (
        type(data) is not tuple
        or len(data) != 2
        or type(data[0]) is not tuple
        or len(data[0]) != 16
    ):
        raise ValueError("weighted probe requires the original sixteen F update sources")
    if type(return_training) is not tuple or len(return_training) != 2:
        raise ValueError("weighted probe requires the original two TRAIN return templates")
    fresh = _episodes(data[0], "probe_train")
    templates = _episodes(return_training, "train")
    configuration = policy(mode, state.policy.native_policy.inference_steps)
    return TorchTrainWeightedLearner.from_probe_snapshot(
        state,
        fresh,
        inference_policy=configuration,
        return_episodes=tuple(templates[index % 2] for index in range(16)),
    )


def sum_work(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce observed numeric receipts only; scopes stay separate, never zero-fill unknown work."""
    native: dict[str, int | bool] = {}
    stream: dict[str, int | bool] = {}
    receipts = failures = 0
    unknown = False
    for row in rows:
        for slot in row["probes"]:
            unknown |= slot.get("work_unknown") is True or slot.get("return_work_unknown") is True
            report = slot.get("report") or {}
            for phase in (report, report.get("return_phase") or {}):
                unknown |= phase.get("attempted_updates", 0) > phase.get("completed_updates", 0)
                for diagnostic in phase.get("diagnostics", ()):
                    receipts += 1
                    _sum_pairs(native, diagnostic["native_work"])
                    _sum_pairs(stream, diagnostic["stream_work"])
                failed = phase.get("failed_native_work")
                if failed is not None:
                    failures += 1
                    _sum_pairs(native, failed["work"].items())
                    _sum_pairs(stream, failed.get("scheduling_work", {}).items())
    return dict(
        native_work=native,
        scheduling_work=stream,
        successful_receipts=receipts,
        failed_receipts=failures,
        failure_work_may_be_unobserved=unknown
        or native.get("failure_work_may_be_unobserved", False),
        scope="observed update receipts only, includes discarded private candidates; construction/validation/copies and unreturned work additional or unknown",
    )


def _sum_pairs(total, pairs) -> None:
    for key, value in pairs:
        total[key] = (
            (total.get(key, False) or value) if type(value) is bool else total.get(key, 0) + value
        )
