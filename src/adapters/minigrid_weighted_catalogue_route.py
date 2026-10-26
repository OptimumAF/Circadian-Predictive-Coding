"""Compose fixed weighted TRAIN catalogues through existing MiniGrid ports.

Inputs are the six original complete sources,128 joint transports and completed
stream arms. Outputs are one bound catalogue factory, explicit objective/dose
metadata and ordinary disposable probe templates. No collection, objective math,
runner, budget, evaluation, tuning, promotion or IO belongs here.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from src.adapters import minigrid_joint_catalogue_route as joint
from src.adapters import minigrid_relaxation_control as relaxation
from src.adapters import minigrid_weighted_feature_route as online
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
    InvariantTorchFeatureSnapshot,
)
from src.adapters.torch_catalogue_weighted import (
    TORCH_CATALOGUE_WEIGHTED_CE_ID,
    TORCH_CATALOGUE_WEIGHTED_PC_ID,
    prepare_train_catalogue,
)
from src.adapters.torch_feature_learner import TORCH_CE_ID, TORCH_PC_ENERGY_ID
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner, _complete_fields
from src.app.coupled_attribution import AttributionBatch, _validate_batch
from src.core.learner_ports import TrainingDiagnostic


def _method(method: str) -> str:
    if type(method) is not str or method not in ("bp", "pc", "neutral_cpc", *relaxation.METHODS):
        raise ValueError(
            "weighted catalogue requires explicit bp, pc, neutral_cpc or fixed64 control"
        )
    return f"public_invariant_{method}"


def _policy(method: str) -> InvariantTorchFeaturePolicy:
    public_method = _method(method)
    return (
        relaxation.policy(public_method)
        if method in relaxation.METHODS
        else online._fixed_policy(method)
    )


def validate_option(flag: bool, **options: Any) -> None:
    if type(flag) is not bool:
        raise ValueError("weighted_catalogue must be an exact bool")
    if not flag:
        return
    if (
        any(
            options.get(name) is not True
            for name in (
                "public_history",
                "invariant_history",
                "joint_catalogue",
                "distinct_probe",
                "old_task_readouts",
                "return_to_a",
            )
        )
        or type(options.get("qualification_method")) is not str
        or options["qualification_method"] not in ("bp", "pc", "neutral_cpc", *relaxation.METHODS)
        or any(
            options.get(name, False) is not False
            for name in (
                "weighted_bp",
                "weighted_pc",
                "pc_qualification",
                "memory_control",
                "acquisition_only",
            )
        )
        or (
            options.get("pc_update_diagnostics", False) is not False
            and (
                options.get("pc_update_diagnostics") is not True
                or options["qualification_method"] not in {"pc", "neutral_cpc"}
            )
        )
    ):
        raise ValueError(
            "weighted_catalogue requires both history modes, joint_catalogue, "
            "distinct_probe/old_task_readouts/return_to_a, explicit bp/pc/neutral_cpc or fixed64 control, "
            "without online weighting, PC alias or memory/acquisition; "
            "PC update diagnostics require explicit pc/neutral_cpc"
        )


def describe(protocol: dict[str, Any], *, method: str) -> None:
    """Apply after ordinary joint composition; keep its exposure/metric contract."""
    _method(method)
    steps = _policy(method).native_policy.inference_steps
    objective = TORCH_CATALOGUE_WEIGHTED_CE_ID if method == "bp" else TORCH_CATALOGUE_WEIGHTED_PC_ID
    probe_objective = TORCH_CE_ID if method == "bp" else TORCH_PC_ENERGY_ID
    protocol["protocol_id"] = protocol["protocol_id"].removesuffix("_v1") + "_weighted_catalogue_v1"
    protocol["weighted_catalogue"] = dict(
        method=method,
        benchmark_objective=objective,
        probe_objective=probe_objective,
        weighting="complete six-source TRAIN catalogue: N/(present_classes*class_count), absent classes1; fixed aggregate class mass, no tuning",
        exposure=f"original joint128 delayed1 independent54-row TRAIN wakes,6912rows; seed17/H8/lr.03/momentum.9/S{steps}/inference_lr.04 and checkpoint doses unchanged",
        metrics="original unweighted CE/accuracy/exact sequence and optional action margins; weighting changes training only",
        isolation="two ordinary unweighted templates proved equal to full zero-update benchmark state; restore checkpoints only on disposable ordinary forks; F/old-A/A-return never update weighted schedule",
        diagnostic="actual weighted catalogue CE for BP, weighted catalogue CE plus hidden energy for PC/neutral; ordinary probe diagnostics retain their original unweighted definitions",
        storage="one owned six-source catalogue and source spans per weighted owner plus immutable witness/current expected state; independent candidates/forks and two ordinary templates add time/RSS; no unlimited success chain",
        interpretation="separately named fixed exploratory joint weighting control, not a historical baseline replacement or acquisition/retention/recovery/circadian advantage claim",
    )
    protocol["training"] = (
        "128 delayed1 independent-row wakes; each sees all six original TRAIN sources "
        f"in fixed order,54rows; explicitly weighted catalogue {method} objective"
    )
    protocol["public_history"]["computation"] = (
        "54rows per weighted catalogue native wake; fixed aggregate TRAIN weights, "
        "original seed/policy and explicitly different training objective"
    )
    protocol["probe"]["objective"] = probe_objective
    protocol["probe"]["isolation"] += (
        "; ordinary unweighted fork, never weighted catalogue training"
    )
    if "return_phase" in protocol:
        protocol["return_phase"]["objective"] = probe_objective
    protocol["planned_work"].update(
        weighted_benchmark_native_train_calls=128,
        weighted_benchmark_training_rows=6912,
        weighted_benchmark_nominal_latent_row_iterations=0 if method == "bp" else 6912 * steps,
        unweighted_probe_templates=2,
    )


def factory(
    sources: tuple[AttributionBatch, ...],
    packets: tuple[AttributionBatch, ...],
    *,
    method: str,
) -> Callable[[str], TorchTrainWeightedLearner]:
    public_method = _method(method)
    if type(packets) is not tuple or len(packets) != 128:
        raise ValueError("weighted catalogue requires the fixed128 joint TRAIN packets")
    if (
        type(sources) is not tuple
        or len(sources) != 6
        or any(
            type(packet) is not AttributionBatch
            or packet.episode is None
            or not _complete_fields((packet.episode,))
            for packet in sources
        )
    ):
        raise ValueError("weighted catalogue requires six exact complete TRAIN sources")
    # Why: reuse the original ordered source/span transport definition rather
    # than creating an episode by joining independent catalogue rows.
    plan = joint.compose(sources)
    seen: set[str] = set()
    for packet, expected in zip(packets, plan.training, strict=True):
        _validate_batch(packet, "train", seen)
        if (
            packet.episode is not None
            or packet.context != expected.context
            or packet.num_classes != 7
            or packet.row_ids != expected.row_ids
            or not np.array_equal(packet.features, expected.features)
            or not np.array_equal(packet.targets, expected.targets)
        ):
            raise ValueError(
                "weighted catalogue packet differs from original ordered54-row sources"
            )
    episodes = tuple(packet.episode for packet in sources if packet.episode is not None)
    catalogue = prepare_train_catalogue(episodes)
    policy = _policy(method)

    def build(method: str) -> TorchTrainWeightedLearner:
        if type(method) is not str or method != public_method:
            raise ValueError("weighted catalogue factory admits only the explicit bound method")
        return TorchTrainWeightedLearner.from_catalogue(policy, catalogue, 128)

    return build


def probe_arm(arm: dict[str, Any], *, method: str) -> dict[str, Any]:
    public_method = _method(method)
    original = arm["source"]
    if type(original) is not TorchTrainWeightedLearner or arm["row"]["method"] != public_method:
        raise ValueError("weighted catalogue probe requires its original bound arm")
    before = original.snapshot_state()
    if (
        (method in relaxation.METHODS and type(before) is not InvariantTorchFeatureSnapshot)
        or original._catalogue is None
        or type(original._catalogue_updates) is not int
        or original._catalogue_updates != 128
        or type(original._catalogue_rows) is not int
        or original._catalogue_rows != 54
        or before.policy != _policy(method)
        or type(before.completed_updates) is not int
        or type(before.training_examples) is not int
        or before.completed_updates != 0
        or before.training_examples != 0
    ):
        raise ValueError("weighted catalogue probe requires fixed128x54 zero-update source")
    if method in relaxation.METHODS:
        # Why: online episode weighting deliberately admits only its S2 recipes.
        # Reuse the ordinary full-state template proof for this named S64 route.
        template = InvariantTorchFeatureLearner(before.policy)
        if not _same_state(before, template.snapshot_state()):
            raise ValueError("ordinary probe template differs from weighted source")
        if not _same_state(before, original.snapshot_state()) or arm["source"] is not original:
            raise ValueError("weighted benchmark source changed during probe construction")
        return {**arm, "source": template}
    return online.probe_arm(arm, method=method)


def check_receipts(arm: dict[str, Any], *, method: str) -> None:
    public_method = _method(method)
    objective = TORCH_CATALOGUE_WEIGHTED_CE_ID if method == "bp" else TORCH_CATALOGUE_WEIGHTED_PC_ID
    result = arm["row"]["stream_result"]
    receipts = result.get("applied_updates")
    if (
        arm["row"].get("method") != public_method
        or result.get("evaluation_scope") != "training_only"
        or type(result.get("completed_updates")) is not int
        or result.get("completed_updates") != 128
        or type(result.get("training_examples")) is not int
        or result.get("training_examples") != 6912
        or type(receipts) is not list
        or len(receipts) != 128
        or any(
            type(receipt) is not dict
            or type(receipt.get("update_number")) is not int
            or receipt.get("update_number") != index
            or type(receipt.get("diagnostic")) is not dict
            or receipt["diagnostic"].get("definition") != objective
            for index, receipt in enumerate(receipts, 1)
        )
    ):
        raise ValueError(
            "weighted catalogue stream receipts differ from fixed method/objective/dose"
        )
    for receipt in receipts:
        diagnostic = receipt["diagnostic"]
        TrainingDiagnostic(diagnostic["definition"], diagnostic.get("value"))
