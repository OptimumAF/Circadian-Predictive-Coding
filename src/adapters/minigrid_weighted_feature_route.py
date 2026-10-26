"""Opt-in weighted feature composition for the MiniGrid learnability pilot.

Inputs are the fixed encoded TRAIN packets and completed stream arms. Outputs are
a sealed weighted factory, checked objective receipts and ordinary disposable
probe templates. No collection, scheduler, metric, tuning, probe update or IO lives here.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeatureSnapshot,
)
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeaturePolicy
from src.adapters.torch_feature_learner import TORCH_CE_ID, TORCH_PC_ENERGY_ID, TorchFeaturePolicy
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_feature_weighted import TORCH_INVARIANT_WEIGHTED_CE_ID
from src.adapters.torch_pc_weighted import TORCH_INVARIANT_WEIGHTED_PC_ID
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner, _complete_fields
from src.app.coupled_attribution import AttributionBatch, _validate_batch
from src.core.demonstration_episode import copy_episode, episode_arrays
from src.core.learner_ports import TrainingDiagnostic
from src.core.minigrid_invariant_encoding import canonicalize_public_history

METHOD = "public_invariant_bp"


def _method(method: str) -> str:
    if type(method) is not str or method not in {"bp", "pc", "neutral_cpc"}:
        raise ValueError("weighted feature route requires explicit bp, pc or neutral_cpc")
    return f"public_invariant_{method}"


def _fixed_policy(method: str) -> InvariantTorchFeaturePolicy:
    _method(method)
    return InvariantTorchFeaturePolicy(
        TorchFeaturePolicy(
            method,
            296,
            8,
            17,
            learning_rate=0.03,
            inference_steps=2,
            inference_learning_rate=0.04,
            momentum=0.9,
            num_classes=7,
        )
    )


def validate_option(
    flag: bool,
    *,
    public_history: bool,
    invariant_history: bool,
    qualification_method: str | None,
    pc_qualification: bool,
    memory_control: bool,
) -> None:
    if type(flag) is not bool or (
        flag
        and (
            public_history is not True
            or invariant_history is not True
            or type(qualification_method) is not str
            or qualification_method != "bp"
            or pc_qualification is not False
            or memory_control is not False
        )
    ):
        raise ValueError(
            "weighted_bp requires an exact bool, both public/invariant history, "
            "explicit qualification_method='bp' and no PC alias or memory control"
        )


def validate_pc_option(
    flag: bool,
    *,
    public_history: bool,
    invariant_history: bool,
    qualification_method: str | None,
    pc_qualification: bool,
    memory_control: bool,
    weighted_bp: bool,
    joint_catalogue: bool,
) -> None:
    if type(flag) is not bool or (
        flag
        and (
            public_history is not True
            or invariant_history is not True
            or type(qualification_method) is not str
            or qualification_method not in {"pc", "neutral_cpc"}
            or pc_qualification is not False
            or memory_control is not False
            or weighted_bp is not False
            or joint_catalogue is not False
        )
    ):
        raise ValueError(
            "weighted_pc requires an exact bool, both public/invariant history, "
            "explicit qualification_method='pc' or 'neutral_cpc', and no PC alias, "
            "memory control, weighted BP or joint catalogue"
        )


def _describe_pc(protocol: dict[str, Any], method: str) -> None:
    protocol["protocol_id"] = protocol["protocol_id"].removesuffix("_v1") + f"_weighted_{method}_v1"
    protocol["weighted_pc"] = dict(
        method=method,
        benchmark_objective=TORCH_INVARIANT_WEIGHTED_PC_ID,
        probe_objective=TORCH_PC_ENERGY_ID,
        weighting="current complete TRAIN episode only: n/(present_classes*class_count), absent classes1; no tuning",
        energy="sum(w[y]*(CE + .5*sum((hidden-prior)^2)))/sum(observed target weights); parameter partials hold relaxed state fixed; native S2 relaxation uses batch_size*dE/dhidden",
        exposure="original128 delayed1 nine-row TRAIN packets, seed17/H8/lr.03/S2/inference_lr.04 policy and checkpoint doses unchanged; distinct weighted objective, not a relabeled historical baseline",
        metrics="original unweighted raw CE/accuracy/exact sequence and optional unweighted prior-CE gradients; no weighted evaluation metric",
        isolation="two separate ordinary unweighted source templates proved equal to original zero-update source; full original checkpoint restored only on disposable probe fork; probe and A-return updates remain unweighted",
        diagnostic="weighted benchmark receipts report actual weighted CE plus hidden energy; ordinary probe diagnostic retains legacy unweighted mean-squared errors, not that objective",
        storage="128 complete TRAIN episodes per sealed weighted owner plus current expected state; source/candidate/actor/fork copies and two ordinary probe templates add time/RSS; no success-record chain",
        interpretation="fixed exploratory weighted control with matched frozen policy/doses; comparison requires separately frozen protocol; no acquisition, plasticity, recovery or circadian advantage assumed",
    )
    protocol["training"] += f"; benchmark uses explicitly named current-episode weighted {method}"
    protocol["probe"]["objective"] = TORCH_PC_ENERGY_ID
    protocol["probe"]["isolation"] += (
        "; ordinary unweighted fork, never the weighted TRAIN schedule"
    )
    if "return_phase" in protocol:
        protocol["return_phase"]["objective"] = TORCH_PC_ENERGY_ID
    protocol["planned_work"].update(
        weighted_benchmark_native_train_calls=128,
        weighted_benchmark_training_rows=1152,
        weighted_benchmark_nominal_latent_row_iterations=2304,
        unweighted_probe_templates=2,
    )


def describe(protocol: dict[str, Any], *, method: str = "bp") -> None:
    _method(method)
    if method != "bp":
        _describe_pc(protocol, method)
        return
    protocol["protocol_id"] = protocol["protocol_id"].removesuffix("_v1") + "_weighted_bp_v1"
    protocol["weighted_bp"] = dict(
        benchmark_objective=TORCH_INVARIANT_WEIGHTED_CE_ID,
        probe_objective=TORCH_CE_ID,
        weighting="current complete TRAIN episode only: n/(present_classes*class_count), absent classes1; sum(weight*CE)/sum(observed target weights); no tuning",
        exposure="original128 delayed1 TRAIN packets, seed17/H8/native SGD policy and checkpoint doses unchanged; distinct weighted objective, not a relabeled historical baseline",
        metrics="original unweighted raw CE/accuracy/exact sequence and optional unweighted prior-CE gradients; no weighted evaluation metric",
        isolation="two separate ordinary BP source templates proved equal to original zero-update source; original full checkpoint/optimizer restored only on disposable probe fork; probe and A-return SGD remain unweighted",
        storage="128 complete TRAIN episodes per sealed weighted owner plus current expected state; source/candidate/actor/fork copies and two ordinary probe templates add time/RSS; no success-record chain",
        interpretation="one fixed exploratory weighted control; comparison requires separately frozen matched protocol; no acquisition, plasticity, recovery or circadian advantage assumed",
    )
    protocol["training"] += "; benchmark uses explicitly named current-episode weighted BP"
    protocol["probe"]["objective"] = TORCH_CE_ID
    protocol["probe"]["isolation"] += (
        "; ordinary unweighted BP fork, never the weighted TRAIN schedule"
    )
    if "return_phase" in protocol:
        protocol["return_phase"]["objective"] = TORCH_CE_ID
    protocol["planned_work"].update(
        weighted_benchmark_native_train_calls=128,
        weighted_benchmark_training_rows=1152,
        unweighted_probe_templates=2,
    )


def factory(
    packets: tuple[AttributionBatch, ...], *, method: str = "bp"
) -> Callable[[str], TorchTrainWeightedLearner]:
    public_method = _method(method)
    native_method = method
    if type(packets) is not tuple or len(packets) != 128:
        raise ValueError("weighted BP requires the fixed128 encoded TRAIN packets")
    seen: set[str] = set()
    episodes = []
    for packet in packets:
        _validate_batch(packet, "train", seen)
        if packet.episode is None:
            raise ValueError("weighted BP requires complete TRAIN episode metadata")
        if not _complete_fields((packet.episode,)):
            raise ValueError("weighted BP requires exact complete TRAIN episode records")
        episode = copy_episode(packet.episode)
        expected = episode_arrays(episode)
        if (
            packet.num_classes != 7
            or packet.features.shape != (9, 296)
            or episode.role != "train"
            or episode.context != packet.context
            or any(
                not np.array_equal(actual, wanted)
                for actual, wanted in zip((packet.features, packet.targets), expected, strict=True)
            )
        ):
            raise ValueError(
                "weighted BP packets must match complete nine-row raw296/K7 TRAIN episodes"
            )
        for transition in episode.transitions:
            canonicalize_public_history(transition.observation)
            canonicalize_public_history(transition.next_observation)
        episodes.append(episode)
    schedule = tuple(episodes)

    def build(method: str) -> TorchTrainWeightedLearner:
        if type(method) is not str or method != public_method:
            raise ValueError("weighted factory admits only the explicit bound method")
        return TorchTrainWeightedLearner(
            _fixed_policy(native_method),
            schedule,
        )

    return build


def probe_arm(arm: dict[str, Any], *, method: str = "bp") -> dict[str, Any]:
    """Keep benchmark schedule/owner out of the original unweighted probe path."""
    public_method = _method(method)
    original = arm["source"]
    if type(original) is not TorchTrainWeightedLearner or arm["row"]["method"] != public_method:
        raise ValueError("weighted probe requires the original bound weighted arm")
    before = original.snapshot_state()
    if (
        type(before) is not InvariantTorchFeatureSnapshot
        or before.policy.native_policy.method != method
        or (method != "bp" and before.policy != _fixed_policy(method))
        or (
            method != "bp"
            and (
                type(before.completed_updates) is not int
                or type(before.training_examples) is not int
            )
        )
        or before.completed_updates != 0
        or before.training_examples != 0
    ):
        raise ValueError("weighted probe source must remain the original zero-update template")
    template = InvariantTorchFeatureLearner(before.policy)
    if not _same_state(before, template.snapshot_state()):
        raise ValueError("ordinary probe template differs from weighted source")
    if not _same_state(before, original.snapshot_state()) or arm["source"] is not original:
        raise ValueError("weighted benchmark source changed during probe construction")
    # Why: ordinary probe training must accept probe_train arrays beyond the sealed
    # benchmark schedule. Existing app code restores the checkpoint on its fork.
    return {**arm, "source": template}


def check_receipts(arm: dict[str, Any], *, method: str = "bp") -> None:
    public_method = _method(method)
    objective = TORCH_INVARIANT_WEIGHTED_CE_ID if method == "bp" else TORCH_INVARIANT_WEIGHTED_PC_ID
    result = arm["row"]["stream_result"]
    receipts = result.get("applied_updates")
    if (
        (method != "bp" and arm["row"]["method"] != public_method)
        or result.get("evaluation_scope") != "training_only"
        or result.get("completed_updates") != 128
        or type(result.get("completed_updates")) is not int
        or result.get("training_examples") != 1152
        or type(result.get("training_examples")) is not int
        or type(receipts) is not list
        or len(receipts) != 128
        or any(
            type(receipt) is not dict
            or receipt.get("update_number") != index
            or type(receipt.get("update_number")) is not int
            or type(receipt.get("diagnostic")) is not dict
            or receipt["diagnostic"].get("definition") != objective
            for index, receipt in enumerate(receipts, 1)
        )
    ):
        raise ValueError(
            f"weighted {method.upper()} stream receipts differ from the fixed objective/dose"
        )
    for receipt in receipts:
        diagnostic = receipt["diagnostic"]
        TrainingDiagnostic(diagnostic["definition"], diagnostic.get("value"))
