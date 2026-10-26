"""Fixed MiniGrid checkpoint fit and disposable fresh-task diagnostics.

Inputs are the existing frozen learner recipe and public teacher packets. Outputs
are TRAIN fit, independently charged probe curves and bounded partial evidence.
Optional known-TRAIN A games use frozen forks; explicit old-A replay stays in
disposable F probes. No final D, selection,
promotion or file IO occurs.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, replace
from math import isfinite
import pickle
from time import monotonic
from typing import Any, TypedDict, cast

import numpy as np

from src.adapters import minigrid_learning_pilot as shared
from src.adapters import minigrid_bounded_probe as bounded_probe
from src.adapters import minigrid_probe_rehearsal as probe_rehearsal
from src.adapters import minigrid_f_exposure as f_exposure
from src.adapters import minigrid_bp_update_route as bp_update_route
from src.adapters.torch_bp_update_diagnostics import diagnose_bp_updates
from src.adapters.torch_recurrent_gradient_diagnostics import GradientDiagnosticFailed
from src.adapters.minigrid_fixed_rehearsal import build_minigrid_old_a_rehearsal
from src.adapters import minigrid_weighted_feature_route as weighted_feature
from src.adapters import minigrid_joint_catalogue_route as joint
from src.adapters import minigrid_cue_relative_route as cue_relative
from src.adapters.cue_relative_torch_feature_learner import (
    CueRelativeTorchFeatureLearner,
    CueRelativeTorchFeatureSnapshot,
)
from src.adapters import minigrid_weighted_catalogue_route as weighted_catalogue_route
from src.adapters import minigrid_relaxation_control as relaxation
from src.adapters import minigrid_memory_control as memory
from src.adapters import minigrid_probe_rollout as probe_rollout
from src.adapters import minigrid_deferred_probe_rollout as deferred_rollout
from src.adapters.invariant_action_memory_learner import (
    InvariantMemorySnapshot,
    copy_invariant_memory_snapshot,
)
from src.adapters import minigrid_gradient_readouts as gradient_readouts
from src.adapters import minigrid_action_margin_readout as action_margins
from src.adapters import minigrid_return_fit as return_fit
from src.adapters import minigrid_train_rollout as train_rollout
from src.adapters.minigrid_train_rollout import run_train_rollout
from src.adapters.minigrid_return_fit import return_training_sources as _return_training_sources
from src.adapters import torch_recurrent_episode_batch as episode_batch
from src.adapters.torch_paired_probe import TorchPairedProbeLearner
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
    InvariantTorchFeatureSnapshot,
)
from src.adapters.torch_activation_diagnostics import read_native_activations
from src.adapters.minigrid_probe_support import (
    summarize_probe_support,
    _probe_pair,
    summarize_train_fit_bounds,
    describe_train_fit_bounds,
    validate_train_fit_bounds_option,
    fit_gap_outcomes,
)
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.coupled_attribution import AttributionBatch, _validate_batch, _same_value
from src.app.fresh_task_probe import run_fresh_task_probe, _validate_batches
from src.app.stream_rehearsal import StreamRehearsal
from src.app.probe_endpoint_observer import EndpointObserver, _BudgetWitness
from src.app.probe_return_phase import budget_binding as _endpoint_budget_binding
from src.app.probe_return_phase import paid_report_guard
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_probability_scores import probability_scores, validate_class_probabilities
from src.core.categorical_probability_margins import probability_margins
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    copy_episode,
    episode_arrays,
)
from src.core.minigrid_key_door import CueKeyDoorTeacher, predefined_probe_layouts
from src.core.minigrid_public_history import encode_public_history_episode
from src.core.action_memory import _feature_keys
from src.core.minigrid_invariant_encoding import (
    INVARIANT_HISTORY_ENCODING_ID,
    canonicalize_public_history,
)
from src.core.learnability_diagnostics import diagnose_activations
from src.infra.minigrid_key_door import MiniGridKeyDoor
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch

PROBE_POINTS = (0, 4, 8, 16)
PAIRED_PROBE_POINTS = (0, 2, 4, 8)
BODY_SECONDS = 180.0
RSS_LIMIT = 768 * 1024**2
INVARIANT_METHODS = tuple(
    f"public_invariant_{method}" for method in ("bp", "pc", "neutral_cpc", "gated_cpc")
)
INVARIANT_LEARNABILITY_METHODS = (*INVARIANT_METHODS, "recurrent_bc")
BOUND_METHODS = (*INVARIANT_METHODS, *relaxation.BOUND_METHODS, *cue_relative.METHODS)
WIDE_METHODS = (*INVARIANT_METHODS, *relaxation.BOUND_METHODS)
PC_QUALIFICATION_METHODS = ("public_invariant_pc",)
QUALIFICATION_METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc", *relaxation.METHODS)


def _validate_cue_relative_option(flag, joint_catalogue, method, unsupported) -> None:
    if type(flag) is not bool:
        raise ValueError("cue_relative_history must be an exact bool")
    if flag and (
        joint_catalogue is not True
        or type(method) is not str
        or method not in ("bp", "pc", "neutral_cpc", "gated_cpc")
        or any(value is not False and value is not None for value in unsupported)
    ):
        # Why: old diagnostic/weighting readers bind another representation.
        raise ValueError(
            "cue-relative history requires the explicit native joint F/old-A/return route "
            "without projection-specific diagnostics, weighting or rollouts"
        )


def _validate_wide_neural_option(flag: bool, **options: Any) -> None:
    if flag is True and options.get("probe_inference") is not None:
        # Why: reuse the existing weighted objective only on disposable F/A;
        # its fixed control must retain the same original H32 benchmark recipe.
        bounded_probe.validate_option(
            options["probe_inference"], **{**options, "wide_neural": True}
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
        "activation_health",
        "gradient_geometry",
        "pc_qualification",
        "acquisition_only",
        "train_a_rollout",
        "memory_control",
        "weighted_bp",
        "weighted_pc",
        "weighted_catalogue",
        "train_action_margins",
        "train_fit_bounds",
        "cue_relative_history",
        "bp_update_diagnostics",
    )
    if type(flag) is not bool or (
        flag
        and (
            type(options.get("qualification_method")) is not str
            or options["qualification_method"] not in QUALIFICATION_METHODS
            or any(options.get(name) is not True for name in required)
            or any(options.get(name, False) is not False for name in forbidden)
            or type(options.get("prior_gradient_geometry", False)) is not bool
            or (
                options.get("prior_gradient_geometry", False) is True
                and (
                    options["qualification_method"] not in ("bp", "pc", "neutral_cpc", "gated_cpc")
                    or any(
                        options.get(name, False) is not False
                        for name in (
                            "pc_update_diagnostics",
                            "old_a_rehearsal",
                            "extra_f_exposure",
                            "probe_rollouts",
                            "deferred_probe_rollouts",
                            "extended_wide_probe_budget",
                        )
                    )
                )
            )
            or (
                options["qualification_method"] in relaxation.METHODS
                and any(
                    options.get(name, False) is not False
                    for name in (
                        "old_a_rehearsal",
                        "extra_f_exposure",
                        "probe_rollouts",
                        "deferred_probe_rollouts",
                        "extended_wide_probe_budget",
                        "pc_update_diagnostics",
                    )
                )
            )
            or type(options.get("pc_update_diagnostics", False)) is not bool
            or (
                options.get("pc_update_diagnostics", False) is True
                and (
                    options["qualification_method"] not in ("pc", "neutral_cpc")
                    or any(
                        options.get(name, False) is not False
                        for name in (
                            "old_a_rehearsal",
                            "extra_f_exposure",
                            "probe_rollouts",
                            "deferred_probe_rollouts",
                            "extended_wide_probe_budget",
                        )
                    )
                )
            )
            or type(options.get("old_a_rehearsal", False)) is not bool
            or (
                options.get("old_a_rehearsal", False) is True
                and (
                    options.get("probe_rollouts", False) is not False
                    or options.get("deferred_probe_rollouts", False) is not False
                )
            )
            or type(options.get("probe_rollouts", False)) is not bool
            or type(options.get("deferred_probe_rollouts", False)) is not bool
            or (
                options.get("probe_rollouts", False) is True
                and options.get("deferred_probe_rollouts", False) is True
            )
        )
    ):
        raise ValueError(
            "wide_neural requires explicit native joint H32 F16/old-A/A16 without extra options"
        )


def _validate_extended_wide_probe_budget(
    flag, wide_neural, probe_rollouts, *, deferred_probe_rollouts=False
) -> None:
    games = (probe_rollouts is True and deferred_probe_rollouts is False) or (
        probe_rollouts is False and deferred_probe_rollouts is True
    )
    if type(flag) is not bool or (flag and (wide_neural is not True or not games)):
        raise ValueError(
            "extended_wide_probe_budget requires an exact bool and explicit H32 probe rollouts"
        )


def _describe_wide_neural(
    protocol: dict[str, Any], method: str, *, probe_rollouts: bool = False
) -> None:
    fixed64 = method in relaxation.METHODS
    inference_steps = 64 if fixed64 else 2
    protocol["protocol_id"] = protocol["protocol_id"].removesuffix("_v1") + "_h32_v1"
    protocol["hidden_dim"] = 32
    protocol["trainable_parameters"] = {f"public_invariant_{method}": 9735}
    protocol["public_history"]["parameters_by_method"] = {f"public_invariant_{method}": 9735}
    # Replay work is declared before width promotion; retain its actual costs.
    native_calls = protocol["planned_work"]["native_train_calls"]
    training_rows = protocol["planned_work"]["training_rows"]
    protocol["wide_neural"] = dict(
        method=method,
        feature_dim=296,
        hidden_dim=32,
        num_classes=7,
        seed=17,
        parameter_elements=9735,
        parameter_array_equivalent_bytes=38940,
        benchmark_native_calls=128,
        benchmark_training_rows=6912,
        probe_native_calls=protocol["planned_work"]["probe_native_train_calls"],
        probe_training_rows=protocol["planned_work"]["probe_training_rows"],
        return_native_calls=32,
        return_training_rows=288,
        bp_optimizer_steps_derived=native_calls if method == "bp" else 0,
        pc_parameter_updates_derived=0 if method == "bp" else native_calls,
        nominal_latent_row_steps=0 if method == "bp" else inference_steps * training_rows,
        scope="fixed H32 native recipe; existing disposable F16 then A16, both A cues, original source/actor/budgets isolated; copies/optimizer/adaptive/RNG/projection/validation/Python/RSS/time additional; no live/final/selection or benefit claim",
    )
    if fixed64:
        protocol["wide_neural"].update(
            native_method=relaxation.base_method(method), inference_steps=64
        )
        protocol["wide_neural"]["scope"] += (
            "; separately named fixed64 sensitivity control, higher latent computation"
        )
    if "old_a_rehearsal" in protocol:
        replay = protocol["planned_work"]["old_a_rehearsal"]
        protocol["wide_neural"].update(
            replay_native_calls=replay["native_train_calls"],
            replay_training_rows=replay["training_rows"],
        )
        protocol["wide_neural"]["scope"] += (
            "; fixed old-A TRAIN replay on the disposable candidate adds separately "
            "paid native calls and exposures; no matched-dose or benefit claim"
        )
    if "extra_f_exposure" in protocol:
        protocol["wide_neural"]["scope"] = protocol["wide_neural"]["scope"].replace("F16", "F48")
    if "probe_inference" in protocol:
        protocol["wide_neural"]["scope"] += (
            "; benchmark objective unchanged; isolated F/A uses explicit class-weighted "
            "PC objective; nominal latent row steps are configured maxima, actual "
            "attempted/accepted solver work is reported separately"
        )
    if "pc_update_diagnostics" in protocol:
        # Why: the TRAIN-only reader is shape independent; declare the actual
        # owned parameter recipe without changing its work or learner updates.
        protocol["pc_update_diagnostics"]["native_recipe"] = dict(
            method=method,
            feature_dim=296,
            hidden_dim=32,
            num_classes=7,
            seed=17,
            learning_rate=0.03,
            momentum=0.9,
            inference_steps=2,
            inference_learning_rate=0.04,
            parameter_elements=9735,
            parameter_array_equivalent_bytes=38940,
        )
    if probe_rollouts:
        protocol["wide_neural"]["scope"] = protocol["wide_neural"]["scope"].replace(
            "no live/final/selection", "fixed known A/F games; no final/selection"
        )
    protocol["planned_work"]["wide_neural"] = deepcopy(protocol["wide_neural"])
    protocol["limitations"] += (
        "; H32 benchmark native objective unchanged; disposable F16/A16 objective "
        "explicitly class-weighted; no useful retention/recovery or solver benefit established"
        if "probe_inference" in protocol
        else "; H32 capacity differs from original H8, same native objectives with explicit F48 exposure control and A16 return; not matched total computation or useful acquired retention/recovery evidence"
        if "extra_f_exposure" in protocol
        else "; H32 capacity differs from original H8, same native objectives and probe/return doses; not matched computation or useful acquired retention/recovery evidence"
    )


def describe_protocol(
    *,
    public_history: bool = False,
    activation_health: bool = False,
    gradient_geometry: bool = False,
    prior_gradient_geometry: bool = False,
    invariant_history: bool = False,
    pc_qualification: bool = False,
    qualification_method: str | None = None,
    acquisition_only: bool = False,
    distinct_probe: bool = False,
    old_task_readouts: bool = False,
    return_to_a: bool = False,
    train_a_rollout: bool = False,
    memory_control: bool = False,
    probe_rollouts: bool = False,
    deferred_probe_rollouts: bool = False,
    weighted_bp: bool = False,
    weighted_pc: bool = False,
    weighted_catalogue: bool = False,
    train_action_margins: bool = False,
    joint_catalogue: bool = False,
    pc_update_diagnostics: bool = False,
    probe_inference: str | None = None,
    train_fit_bounds: bool = False,
    cue_relative_history: bool = False,
    old_a_rehearsal: bool = False,
    bp_update_diagnostics: bool = False,
    wide_neural: bool = False,
    extended_wide_probe_budget: bool = False,
    extra_f_exposure: bool = False,
) -> dict[str, Any]:
    """Freeze the diagnostic without constructing native models or environments."""
    _validate_extended_wide_probe_budget(
        extended_wide_probe_budget,
        wide_neural,
        probe_rollouts,
        deferred_probe_rollouts=deferred_probe_rollouts,
    )
    f_exposure.validate_option(extra_f_exposure, **locals())
    _validate_wide_neural_option(wide_neural, **locals())
    bp_update_route.validate_option(bp_update_diagnostics, **locals())
    probe_rehearsal.validate_option(old_a_rehearsal, **locals())
    _validate_cue_relative_option(
        cue_relative_history,
        joint_catalogue,
        qualification_method,
        (
            activation_health,
            gradient_geometry,
            prior_gradient_geometry,
            weighted_bp,
            weighted_pc,
            weighted_catalogue,
            train_action_margins,
            pc_update_diagnostics,
            train_a_rollout,
            probe_rollouts,
            deferred_probe_rollouts,
            memory_control,
            pc_qualification,
            acquisition_only,
            probe_inference,
        ),
    )
    validate_train_fit_bounds_option(train_fit_bounds, joint_catalogue)
    bounded_probe.validate_option(
        probe_inference,
        public_history=public_history,
        invariant_history=invariant_history,
        joint_catalogue=joint_catalogue,
        qualification_method=qualification_method,
        distinct_probe=distinct_probe,
        old_task_readouts=old_task_readouts,
        return_to_a=return_to_a,
        pc_qualification=pc_qualification,
        acquisition_only=acquisition_only,
        memory_control=memory_control,
        weighted_bp=weighted_bp,
        weighted_pc=weighted_pc,
        activation_health=activation_health,
        gradient_geometry=gradient_geometry,
        prior_gradient_geometry=prior_gradient_geometry,
        train_a_rollout=train_a_rollout,
        probe_rollouts=probe_rollouts,
        deferred_probe_rollouts=deferred_probe_rollouts,
        train_action_margins=train_action_margins,
        pc_update_diagnostics=pc_update_diagnostics,
    )
    relaxation.validate_option(
        qualification_method,
        joint_catalogue=joint_catalogue,
        train_action_margins=train_action_margins,
        pc_update_diagnostics=pc_update_diagnostics,
    )
    gradient_readouts.validate_pc_option(
        pc_update_diagnostics,
        joint_catalogue=joint_catalogue,
        qualification_method=qualification_method,
        train_action_margins=train_action_margins,
    )
    weighted_catalogue_route.validate_option(
        weighted_catalogue,
        public_history=public_history,
        invariant_history=invariant_history,
        joint_catalogue=joint_catalogue,
        qualification_method=qualification_method,
        distinct_probe=distinct_probe,
        old_task_readouts=old_task_readouts,
        return_to_a=return_to_a,
        weighted_bp=weighted_bp,
        weighted_pc=weighted_pc,
        pc_qualification=pc_qualification,
        acquisition_only=acquisition_only,
        memory_control=memory_control,
        pc_update_diagnostics=pc_update_diagnostics,
    )
    joint.validate_option(
        joint_catalogue,
        public_history=public_history,
        invariant_history=invariant_history,
        qualification_method=qualification_method,
        distinct_probe=distinct_probe,
        old_task_readouts=old_task_readouts,
        return_to_a=return_to_a,
        pc_qualification=pc_qualification,
        acquisition_only=acquisition_only,
        weighted_bp=weighted_bp,
        weighted_pc=weighted_pc,
        memory_control=memory_control,
        activation_health=activation_health,
        gradient_geometry=gradient_geometry,
        prior_gradient_geometry=prior_gradient_geometry,
        train_a_rollout=train_a_rollout,
        # Why: exact wide admission above owns optional endpoint games; this
        # legacy validator binds the unchanged joint-training recipe only.
        probe_rollouts=False if wide_neural else probe_rollouts,
        deferred_probe_rollouts=False if wide_neural else deferred_probe_rollouts,
        train_action_margins=train_action_margins,
        pc_update_diagnostics=pc_update_diagnostics,
        **({"wide_neural": True} if wide_neural else {}),
    )
    action_margins.validate_option(train_action_margins, public_history, invariant_history)
    weighted_feature.validate_option(
        weighted_bp,
        public_history=public_history,
        invariant_history=invariant_history,
        qualification_method=qualification_method,
        pc_qualification=pc_qualification,
        memory_control=memory_control,
    )
    weighted_feature.validate_pc_option(
        weighted_pc,
        public_history=public_history,
        invariant_history=invariant_history,
        qualification_method=qualification_method,
        pc_qualification=pc_qualification,
        memory_control=memory_control,
        weighted_bp=weighted_bp,
        joint_catalogue=joint_catalogue,
    )
    probe_rollout.validate_option(
        probe_rollouts, memory_control, **({"wide_neural": True} if wide_neural else {})
    )
    deferred_rollout.validate_option(
        deferred_probe_rollouts,
        memory_control,
        probe_rollouts,
        **(
            {"wide_method": qualification_method}
            if wide_neural and qualification_method not in relaxation.METHODS
            else {}
        ),
    )
    memory.validate_option(
        memory_control,
        public_history=public_history,
        invariant_history=invariant_history,
        acquisition_only=acquisition_only,
        distinct_probe=distinct_probe,
        old_task_readouts=old_task_readouts,
        return_to_a=return_to_a,
        qualification_method=qualification_method,
        pc_qualification=pc_qualification,
        activation_health=activation_health,
        gradient_geometry=gradient_geometry,
        train_a_rollout=train_a_rollout,
    )
    if type(public_history) is not bool:
        raise ValueError("public_history must be a native bool")
    _validate_distinct_probe(distinct_probe, public_history, invariant_history)
    _validate_old_task_readouts(
        old_task_readouts, public_history, invariant_history, distinct_probe
    )
    _validate_return_to_a(return_to_a, old_task_readouts)
    train_rollout.validate_option(
        train_a_rollout, acquisition_only, joint_catalogue=joint_catalogue
    )
    if type(invariant_history) is not bool or (invariant_history and not public_history):
        raise ValueError("invariant_history requires an exact bool and public_history=True")
    if invariant_history and gradient_geometry is not False:
        raise ValueError("invariant history does not support gradient diagnostics")
    _validate_prior_gradient_option(
        prior_gradient_geometry,
        public_history,
        invariant_history,
        gradient_geometry,
        memory_control,
        acquisition_only,
    )
    if type(pc_qualification) is not bool or (
        pc_qualification and (public_history is not True or invariant_history is not True)
    ):
        raise ValueError("pc_qualification requires exact bools and both public/invariant history")
    _validate_qualification_method(
        qualification_method, public_history, invariant_history, pc_qualification
    )
    _validate_acquisition_only(
        acquisition_only,
        public_history,
        invariant_history,
        "bp" if memory_control else qualification_method,
        pc_qualification,
        activation_health,
        gradient_geometry,
    )
    if type(gradient_geometry) is not bool:
        raise ValueError("gradient_geometry must be an exact bool")
    if type(activation_health) is not bool or (activation_health and not public_history):
        raise ValueError("activation_health requires an exact bool and public_history=True")
    protocol = shared.describe_protocol()
    protocol.update(
        protocol_id="minigrid_cue_key_door_checkpoint_learnability_v1",
        classification="exploratory_single_seed_checkpoint_fit_and_fresh_task_diagnostic",
        layouts=protocol["layouts"][:3],
        probe_layouts=[asdict(layout) for layout in predefined_probe_layouts()],
        evaluation="all five original streams finish before any readout or reserved E collection; teacher-forced TRAIN fit at0/32/64/96/128; independent E probes from0/128 only; no final D",
        fit_scope="teacher_forced_train_fit_not_live_success",
        metric_direction="CE lower; accuracy/exact_sequence_accuracy higher; positive gains desirable, positive loss/gap worse; no live success/return measured",
        acquisition_premise="raw nine-action exact_sequence_accuracy=1 for both cues at A32/B64/C96; otherwise forgetting/recovery remains inconclusive",
        probe=dict(
            source_updates=[0, 128],
            updates=16,
            checkpoint_updates=list(PROBE_POINTS),
            training="two physical probe_train episodes copied into16 unique alternating-cue packets",
            evaluation="two separately collected probe_test episodes; same deterministic E domain, not generalization",
            input_array_bytes=193104,
            decision_diagnostics=True,
            isolation="owned disposable forks, original native optimizer state; probe work never enters benchmark/source/actor/checkpoints",
            comparison="CE gain=mean CE0-CE16, positive improvement; post-return minus initial probe gain; gated_cpc-minus-pc/neutral_cpc at each source checkpoint",
            storage="ten fixed probe reports and budgets;193104 detached input-array bytes per probe excludes model/report allocations measured by time/RSS",
        ),
        primary_comparisons=["gated_cpc-minus-pc", "gated_cpc-minus-neutral_cpc"],
        limits=dict(
            body_seconds=180,
            parent_seconds=210,
            process_rss_bytes=RSS_LIMIT,
            output_bytes=32 * 1024**2,
            cpu_threads=1,
            interop_threads=1,
            gpu=0,
        ),
        planned_work=dict(
            benchmark_native_train_calls=640,
            benchmark_training_rows=5760,
            probe_native_train_calls=160,
            probe_training_rows=1440,
            native_train_calls=800,
            training_rows=7200,
            known_raw_score_forwards=150,
            known_raw_score_rows=1350,
            probe_score_forwards=80,
            probe_score_rows=720,
            score_forwards=230,
            score_rows=2070,
            teacher_episodes=10,
            teacher_decisions=90,
            source_environment_steps=90,
            maximum_source_environment_steps=160,
            learner_live_episodes=0,
            final_episodes=0,
            replay_rows=0,
            recurrent_unroll_steps=1854,
        ),
        limitations="raw teacher-forced fit is not useful live skill; probe gain alone does not isolate plasticity; same E domain is not generalization; equal update/row dose is not equal computation; RNN architecture differs; all fixed signed/negative results retained, no ranking or circadian advantage claim",
    )
    if public_history:
        protocol.update(
            protocol_id="minigrid_public_initial_current_learnability_v1",
            feature_dim=296,
            trainable_parameters=dict(feature_methods=2439, recurrent_bc=1319),
            feature_dim_by_method={
                method: 148 if method == "recurrent_bc" else 296 for method in shared.METHODS
            },
            public_history=dict(
                encoding="own episode first public148 observation followed by current public148 observation",
                reset="each complete episode independently; no previous episode, labels, context, reward or future frame in features",
                raw_sources="original148 teacher source evidence retained; only derived learner packets are296",
                parameters_by_method={
                    method: 1319 if method == "recurrent_bc" else 2439 for method in shared.METHODS
                },
                original_feedforward_parameters=1255,
                initial_frame_values=148,
                initial_frame_numeric_bytes=1184,
                initial_frame_scope="transient encoder; Python object overhead included only in process RSS",
                retained_benchmark_array_bytes=6 * 9 * 149 * 8
                + 128 * 9 * 149 * 8
                + 6 * 9 * 297 * 8
                + 128 * 9 * 297 * 8,
                retained_array_scope="pilot input cohorts only; stream/fork copies, DTOs and native state add time/RSS",
                retained_probe_array_bytes=193104 + 384912,
                computation="same update and row dose; larger input/parameter/memory cost, not matched compute or architecture",
            ),
        )
        protocol["probe"].update(
            input_array_bytes=384912,
            storage="384912 derived input-array bytes for each feedforward probe;193104 original for RNN; both cohorts retained,578016 bytes total; original65536entry cap",
        )
        protocol["limitations"] += (
            "; TRAIN alias motivated representation change is exploratory, not an independent confirmation; no comparison with original representation is matched capacity/compute"
        )
    if activation_health:
        protocol["protocol_id"] = "minigrid_public_history_native_activation_health_v1"
        protocol["activation_health"] = dict(
            scope="six known TRAIN packets at each owned0/32/64/96/128 checkpoint; no probe/final inputs",
            definition="native float32 tanh feedforward priors, not label-relaxed PC states; RNN ordered hidden state from zero each episode",
            metrics="existing diagnose_activations: mean absolute activity; dead unit mean<=1e-3; active complement; saturated entry abs>=.99; centered singular-value entropy effective rank with existing absolute cutoff",
            isolation="detached validated snapshots/features; no labels, gradients, optimizer/update/RNG changes",
            computation="one diagnostic call per raw TRAIN cell; FF hidden affine only; RNN reuses native forward per row, including discarded output projections",
            storage="all150x9x8 hidden values stored,86400 float64 array-equivalent bytes total,576 bytes per returned array; copies/records/native tensors/time/RSS additional",
            interpretation="diagnostic observations, no causal plasticity, importance/age, benefit or live-skill claim",
        )
        protocol["planned_work"].update(
            activation_calls=150,
            activation_rows=1350,
            activation_array_bytes=86400,
            activation_svd_calls=150,
            activation_feedforward_hidden_affines=120,
            activation_recurrent_hidden_steps=270,
            activation_recurrent_output_projections=270,
        )
    if gradient_geometry:
        protocol["protocol_id"] = (
            protocol["protocol_id"].removesuffix("_v1") + "_cue_gradient_geometry_v1"
        )
        protocol["cue_gradient_geometry"] = dict(
            definition="torch_recurrent_train_mean_ce_gradient_geometry_v1",
            scope="recurrent_bc only; original148/K7 nine-row TRAIN A/B/C blue-red pairs after six paid raw scores at0/32/64/96/128",
            isolation="detached owned checkpoint gradients; original meanCE; no SGD, source/actor/checkpoint/optimizer/gradient/RNG changes; no final/probe inputs",
            interpretation="signed local whole-episode gradient geometry, not causal interference or adaptation; shared labels can hide cue conflict; zero-norm cosine undefined",
            accounting="diagnostic counts separate from raw scoring, training and probes; logical input bytes exclude native/report copies measured by time/RSS",
        )
        protocol["planned_work"]["cue_gradient_geometry"] = dict(
            pair_calls=15,
            snapshot_validations=15,
            validation_temporary_optimizers=15,
            forward_calls=30,
            forward_rows=270,
            backward_calls=30,
            backward_rows=270,
            geometry_reductions=90,
            input_array_equivalent_bytes=5 * 64368,
            optimizer_steps=0,
            published_updates=0,
        )
        protocol["planned_work"]["recurrent_unroll_steps"] += 270 + (
            270 if activation_health else 0
        )
        protocol["cue_gradient_geometry"]["recurrent_work_scope"] = (
            "recurrent_unroll_steps includes completed training, scoring, probes, "
            "gradient forwards and optional recurrent activation reads; backwards separate"
        )
    if invariant_history:
        protocol.update(
            protocol_id="minigrid_invariant_public_history_learnability_v1",
            methods=list(INVARIANT_LEARNABILITY_METHODS),
            feature_dim_by_method={
                method: 148 if method == "recurrent_bc" else 296
                for method in INVARIANT_LEARNABILITY_METHODS
            },
            primary_comparisons=[
                "public_invariant_gated_cpc-minus-public_invariant_pc",
                "public_invariant_gated_cpc-minus-public_invariant_neutral_cpc",
            ],
            invariant_history=dict(
                encoding_id=INVARIANT_HISTORY_ENCODING_ID,
                feature_inputs="raw296 own first/current public frames; projection inside bound learner only",
                recurrent_bc="original raw148 recurrent learner unchanged",
                projection_accounting="returned committed training rows and returned score rows times2368; validation/copies/Python/native/failed-attempt allocations additional or unknown",
            ),
        )
        protocol["public_history"]["parameters_by_method"] = {
            method: 1319 if method == "recurrent_bc" else 2439
            for method in INVARIANT_LEARNABILITY_METHODS
        }
        protocol["probe"]["comparison"] = (
            "CE gain=mean CE0-CE16; post-return minus initial probe gain; bound gated_cpc-minus-bound pc/neutral_cpc at each source checkpoint"
        )
        protocol["limitations"] += (
            "; declared encoding contract does not authenticate arbitrary external neural weights; no matched raw representation experiment in this recipe"
        )
    if pc_qualification or qualification_method is not None or memory_control:
        _describe_pc_qualification(
            protocol, qualification_method or ("bp" if memory_control else "pc")
        )
    if invariant_history and activation_health:
        _describe_invariant_health(protocol)
    if acquisition_only:
        _describe_a_only_acquisition(protocol, qualification_method or "bp")
    if distinct_probe:
        _describe_distinct_probe(protocol)
    if old_task_readouts:
        _describe_old_task_readouts(protocol)
    if return_to_a:
        return_fit.describe_return(protocol)
    if train_a_rollout and not joint_catalogue:
        train_rollout.describe_rollout(protocol)
    if memory_control:
        memory.describe(protocol)
    if probe_rollouts and not wide_neural:
        probe_rollout.describe(protocol)
    if deferred_probe_rollouts and not wide_neural:
        deferred_rollout.describe(protocol)
    if prior_gradient_geometry:
        _describe_prior_gradients(protocol)
    if weighted_bp:
        weighted_feature.describe(protocol)
    if weighted_pc:
        weighted_feature.describe(protocol, method=cast(str, qualification_method))
    if train_action_margins:
        action_margins.describe(protocol)
    if joint_catalogue:
        joint.describe(protocol)
        if train_a_rollout:
            train_rollout.describe_rollout(protocol, joint_catalogue=True)
    if weighted_catalogue:
        weighted_catalogue_route.describe(protocol, method=cast(str, qualification_method))
    if pc_update_diagnostics:
        gradient_readouts.describe_pc(
            protocol, **({"weighted": True} if weighted_catalogue else {})
        )
    relaxation.describe(
        protocol, qualification_method, **({"wide_neural": True} if wide_neural else {})
    )
    if probe_inference is not None:
        bounded_probe.describe(
            protocol,
            probe_inference,
            cast(str, qualification_method),
            **({"wide_neural": True} if wide_neural else {}),
        )
    if train_fit_bounds:
        protocol["protocol_id"] = (
            protocol["protocol_id"].removesuffix("_v1") + "_train_fit_bounds_v1"
        )
        bound_options = {"cue_relative": True} if cue_relative_history else {}
        protocol["train_fit_bounds"] = describe_train_fit_bounds(**bound_options)
        protocol["planned_work"]["train_fit_bounds"] = describe_train_fit_bounds(**bound_options)
    if cue_relative_history:
        cue_relative.describe(protocol, cast(str, qualification_method))
    if old_a_rehearsal:
        probe_rehearsal.describe(protocol, **({"wide_neural": True} if wide_neural else {}))
    if extra_f_exposure:
        f_exposure.describe(protocol)
    if bp_update_diagnostics:
        bp_update_route.describe(protocol)
    if wide_neural:
        _describe_wide_neural(
            protocol,
            cast(str, qualification_method),
            **({"probe_rollouts": True} if probe_rollouts or deferred_probe_rollouts else {}),
        )
        if probe_rollouts:
            # Why: joint.describe owns score totals and would overwrite game metadata.
            probe_rollout.describe(
                protocol,
                wide_neural=True,
                **({"extended_wide_probe_budget": True} if extended_wide_probe_budget else {}),
            )
        elif deferred_probe_rollouts:
            deferred_rollout.describe(
                protocol,
                wide_neural=True,
                **({"extended_wide_probe_budget": True} if extended_wide_probe_budget else {}),
            )
    if extended_wide_probe_budget:
        protocol["limits"].update(body_seconds=240, parent_seconds=270, shared_cleanup_seconds=280)
        protocol["extended_wide_probe_budget"] = dict(
            endpoint_wall_seconds=probe_rollout.WIDE_EXTENDED_LIVE_SECONDS,
            parent_wall_seconds=probe_rollout.WIDE_EXTENDED_BODY_SECONDS,
        )
    return protocol


def _validate_prior_gradient_option(
    flag, public_history, invariant_history, recurrent_gradient, memory_control, acquisition_only
):
    if type(flag) is not bool or (
        flag
        and (
            public_history is not True
            or invariant_history is not True
            or recurrent_gradient is not False
            or memory_control is not False
            or acquisition_only is not False
        )
    ):
        raise ValueError(
            "prior_gradient_geometry requires bound neural history and six TRAIN sources, without recurrent gradients, memory control or acquisition_only"
        )


def _describe_prior_gradients(protocol: dict[str, Any]) -> None:
    methods = [method for method in protocol["methods"] if method in INVARIANT_METHODS]
    pairs = len(methods) * 5 * 3
    protocol["protocol_id"] = (
        protocol["protocol_id"].removesuffix("_v1") + "_prior_gradient_geometry_v1"
    )
    protocol["prior_gradient_geometry"] = dict(
        definition="torch_invariant_feature_prior_mean_ce_gradient_geometry_v1",
        methods=methods,
        scope="original six TRAIN packets only; no probe/final access or updates",
        objective="prediction mean CE at the feedforward tanh prior; not native PC latent-energy update direction",
        interpretation="gradient cosine is descriptive, not causal interference or circadian benefit",
    )
    protocol["planned_work"]["prior_gradient_geometry"] = dict(
        pair_calls=pairs,
        snapshot_validations=pairs,
        validation_temporary_optimizers=0,
        forward_calls=2 * pairs,
        forward_rows=18 * pairs,
        backward_calls=2 * pairs,
        backward_rows=18 * pairs,
        projection_calls=2 * pairs,
        projection_rows=18 * pairs,
        projection_array_bytes=2368 * 18 * pairs,
        input_array_equivalent_bytes=pairs * 18 * (2368 + 8),
        geometry_reductions=5 * pairs,
        optimizer_steps=0,
        published_updates=0,
    )


def _validate_distinct_probe(flag, public_history, invariant_history) -> None:
    if type(flag) is not bool or (
        flag and (public_history is not True or invariant_history is not True)
    ):
        raise ValueError(
            "distinct_probe requires exact bools and both public/invariant history=True"
        )


class _CatalogueOptions(TypedDict, total=False):
    catalogue_rows: int
    wide_neural: bool


class _ProbeRolloutOptions(TypedDict, total=False):
    wide_method: str
    extended_wide_probe_budget: bool


class _OldTaskReadoutOptions(TypedDict, total=False):
    extra_f_exposure: bool
    wide_neural: bool
    extended_wide_probe_budget: bool
    old_a_rehearsal: bool
    probe_inference: str
    catalogue_rows: int
    probe_rollouts: bool
    deferred: deferred_rollout.DeferredProbeRollouts
    training_readouts: tuple[AttributionBatch, ...]
    return_to_a: bool


class _ReturnAppOptions(TypedDict, total=False):
    extra_f_exposure: bool
    rehearsal_factory: Callable[[Any, ToyBudgetSession, int], StreamRehearsal]
    endpoint_observer: EndpointObserver
    training_readouts: tuple[AttributionBatch, ...]
    return_training: tuple[AttributionBatch, ...]
    return_budget: ToyBudgetSession


def _validate_return_to_a(flag, old_task_readouts) -> None:
    if type(flag) is not bool or (flag and old_task_readouts is not True):
        raise ValueError("return_to_a requires an exact bool and old_task_readouts=True")


def _validate_old_task_readouts(flag, public_history, invariant_history, distinct_probe) -> None:
    if type(flag) is not bool or (
        flag
        and (
            public_history is not True
            or invariant_history is not True
            or distinct_probe is not True
        )
    ):
        raise ValueError(
            "old_task_readouts requires exact bools, both history modes and distinct_probe=True"
        )


def _describe_old_task_readouts(protocol: dict[str, Any]) -> None:
    methods = protocol["methods"]
    calls = 16 * len(methods)
    protocol["protocol_id"] = protocol["protocol_id"].removesuffix("_v1") + "_old_task_readouts_v1"
    protocol["old_task_readouts"] = dict(
        role="train",
        contexts=["A"],
        cues=["blue", "red"],
        checkpoint_updates=list(PROBE_POINTS),
        scope="original two A TRAIN episodes, same disposable F learner after each F score; old TRAIN fit/per-cue drift, not acquired-task retention or generalization",
        updates="F probe_train only; old A inputs/labels used for readouts only",
        input_array_bytes_by_method={
            method: 21456 if method == "recurrent_bc" else 42768 for method in methods
        },
        computation="eight additional forwards/72 rows per probe, separately counted; metrics/decision diagnostics reuse those predictions; native projection/validation/copies/RSS additional",
        direction="CE gain=CE0-CE16; accuracy/exact gain=16-0; negative means worse old fit; A17/18 does not satisfy joint acquisition",
    )
    protocol["planned_work"].update(
        training_readout_forwards=calls, training_readout_rows=9 * calls
    )
    protocol["planned_work"]["score_forwards"] += calls
    protocol["planned_work"]["score_rows"] += 9 * calls
    if "recurrent_bc" in methods:
        protocol["planned_work"]["recurrent_unroll_steps"] += 144


def _describe_distinct_probe(protocol: dict[str, Any]) -> None:
    from src.core.minigrid_key_door import predefined_distinct_probe_layouts

    protocol["protocol_id"] = protocol["protocol_id"].removesuffix("_v1") + "_distinct_probe_v1"
    protocol["probe_layouts"] = [asdict(layout) for layout in predefined_distinct_probe_layouts()]
    protocol["evaluation"] = (
        protocol["evaluation"]
        .replace("reserved E", "reserved F")
        .replace("independent E probes", "independent F probes")
        .replace("isolated E probes", "isolated F probes")
    )
    protocol["probe"].update(
        training="two fixed F-blue/F-red branch-row2 probe_train episodes copied into16 unique alternating-cue packets",
        evaluation="two separately collected F-blue/F-red probe_test episodes; same deterministic F domain, not generalization",
        support_gate="before probe updates compare actual128 raw TRAIN packets against all18 raw probe episodes via causal raw296 and canonical296 keys; every episode must have9of9 canonical inputs unseen in TRAIN or refuse",
    )
    protocol["probe_support"] = dict(
        definition="finite_observed_training_input_and_label_support_v1",
        training_packets=128,
        training_rows=1152,
        probe_packets=18,
        probe_rows=162,
        history_rows=1314,
        canonical_projection_rows=1314,
        canonical_projected_array_equivalent_bytes=1314 * 2368,
        native_updates=0,
        native_score_forwards=0,
        cost_scope="transient causal history/key/label maps plus elapsed time; logical key payload excludes Python/copies/next-frame/failed allocations; actual RSS additional",
    )
    protocol["limitations"] += (
        "; legacy canonical E aliases TRAIN A; F support is measured before updates, not a novelty/generalization proof; no final-derived task or probe selection"
    )


def _describe_invariant_health(protocol: dict[str, Any]) -> None:
    """Declare projected TRAIN-only work after selecting the actual method set."""
    methods = protocol["methods"]
    feedforward_calls = 30 * sum(method in INVARIANT_METHODS for method in methods)
    recurrent_calls = 30 if "recurrent_bc" in methods else 0
    calls = feedforward_calls + recurrent_calls
    protocol["protocol_id"] = (
        protocol["protocol_id"].removesuffix("_v1") + "_native_activation_health_v1"
    )
    protocol["activation_health"].update(
        computation="one diagnostic per TRAIN cell; bound raw296 inputs project before native float32 hidden affine; original raw RNN recurrence unchanged; SVD/copies/validation/time/RSS additional",
        storage=f"{calls}x9x8 hidden values stored;{calls * 576} float64 array-equivalent bytes; projection/copies/records/native tensors additional",
        projection_accounting="returned complete bound-reader rows times2368 bytes; failed reader work may be unobserved; separate from training/scoring projections",
    )
    protocol["planned_work"]["recurrent_unroll_steps"] += 9 * recurrent_calls
    protocol["planned_work"].update(
        activation_calls=calls,
        activation_rows=9 * calls,
        activation_array_bytes=576 * calls,
        activation_svd_calls=calls,
        activation_feedforward_hidden_affines=feedforward_calls,
        activation_recurrent_hidden_steps=9 * recurrent_calls,
        activation_recurrent_output_projections=9 * recurrent_calls,
        activation_projection_rows=9 * feedforward_calls,
        activation_projection_array_bytes=9 * feedforward_calls * 2368,
    )


def _validate_qualification_method(
    method, public_history, invariant_history, pc_qualification
) -> None:
    if method is None:
        return
    if type(method) is not str or method not in QUALIFICATION_METHODS:
        raise ValueError("qualification_method requires an exact declared native method/control")
    if public_history is not True or invariant_history is not True:
        raise ValueError("qualification_method requires both public/invariant history=True")
    if pc_qualification:
        raise ValueError("qualification_method and pc_qualification are mutually exclusive")


def _validate_acquisition_only(
    flag,
    public_history,
    invariant_history,
    method,
    pc_qualification,
    activation_health,
    gradient_geometry,
) -> None:
    if type(flag) is not bool or (
        flag
        and (
            public_history is not True
            or invariant_history is not True
            or type(method) is not str
            or method not in QUALIFICATION_METHODS
            or pc_qualification is not False
            or activation_health is not False
            or gradient_geometry is not False
        )
    ):
        raise ValueError(
            "acquisition_only requires exact bools, an explicit bound qualification method and no PC alias/health/gradient"
        )


def _describe_a_only_acquisition(protocol: dict[str, Any], native_method: str = "bp") -> None:
    # Keep curriculum/metrics fixed while naming the existing selected objective.
    label = {"bp": "BP", "pc": "PC", "neutral_cpc": "neutral CPC", "gated_cpc": "active CPC"}[
        native_method
    ]
    protocol.update(
        protocol_id=f"minigrid_invariant_{native_method}_A_only_acquisition_v1",
        task_sequence=["A"] * 4,
        training=f"six original TRAIN teacher traces collected once; only A blue/red copied into128 unique alternating-cue delayed1 nine-row packets; original continuous {label} optimizer",
        evaluation=f"one A128 {label} stream completes before reserved E collection/readouts; all six original TRAIN traces at0/32/64/96/128: A trained, B/C_train untrained-context fit; no live/final evaluation",
        acquisition_premise="A_at32/A_at128 require both original A cue nine-action sequences exact; single-task fit only, no retention/recovery or generalization claim",
        metric_direction="CE lower; accuracy/exact higher; A0-to-A128 gain CE0-CE128 and accuracy/exact128-0; all B/C_train untrained-context fit reported descriptively",
        limitations=f"single fixed A-only {label} curriculum diagnostic; A128 versus old A64/B32/C32 is not matched context exposure; all TRAIN/E previously inspected, no independent generalization, retention/recovery, useful control, isolated plasticity or advantage claim; negative/inconclusive/partial results retained",
    )
    protocol.pop("differences")
    protocol["probe"].update(
        source_stages=["initial", "post_A128"],
        comparison="within-method CE0-CE16, accuracy16-accuracy0 and exact16-exact0; post-A128 minus initial probe gain; starting fit/optimizer confounded, not post-return recovery",
    )
    protocol["public_history"]["raw_sources"] = (
        "original six148 TRAIN teacher records retained; only A-derived128 learner packets are raw296 at bound learner boundary; B/C used for untrained-context readouts only"
    )


def _describe_pc_qualification(protocol: dict[str, Any], native_method: str = "pc") -> None:
    """Narrow execution scope while retaining original objective and dose laws."""
    base = (
        relaxation.base_method(native_method)
        if native_method in relaxation.METHODS
        else native_method
    )
    label = {"bp": "BP", "pc": "PC", "neutral_cpc": "neutral CPC", "gated_cpc": "active CPC"}[base]
    if native_method in relaxation.METHODS:
        label += " fixed64-step control"
    method = f"public_invariant_{native_method}"
    # Preserve exact native neutral/active configs; selecting an arm never tunes
    # its objective or relabels another method as a PC comparator.
    configs = (
        {native_method: deepcopy(protocol["cpc_configs"][base])}
        if base in ("neutral_cpc", "gated_cpc")
        else {}
    )
    protocol.update(
        protocol_id=f"minigrid_invariant_{native_method}_learnability_qualification_v1",
        methods=[method],
        feature_dim_by_method={method: 296},
        trainable_parameters={method: 2439},
        cpc_configs=configs,
        primary_comparisons=[],
        evaluation=f"one canonical {label} stream completes128 A/B/C_train/A updates before reserved E collection/readouts; six original TRAIN traces at0/32/64/96/128; isolated E probes from0/128 only; no final D",
        limitations=f"single fixed canonical {label} qualification; known TRAIN and previously inspected deterministic E are exploratory fit diagnostics, not independent generalization, useful live control or isolated plasticity; no crossmethod comparison, tuning or circadian advantage claim; all negative/inconclusive/partial results retained",
    )
    protocol["invariant_history"].pop("recurrent_bc")
    history = protocol["public_history"]
    history["parameters_by_method"] = {method: 2439}
    history.pop("original_feedforward_parameters")
    history["computation"] = (
        f"one fixed {label} objective/dose; projection, native validation and complete-state guards add computation and memory"
    )
    protocol["probe"].update(
        comparison="within-method CE gain=mean CE0-CE16 and accuracy gain=accuracy16-accuracy0; post-return minus initial probe gain; no crossmethod comparison",
        storage="two fixed probe reports/budgets;384912 derived input-array bytes per probe; original193104 plus derived384912 cohorts retained (578016 bytes), excluding model/report/copy allocations; original65536entry cap",
    )
    protocol["planned_work"].update(
        benchmark_native_train_calls=128,
        benchmark_training_rows=1152,
        probe_native_train_calls=32,
        probe_training_rows=288,
        native_train_calls=160,
        training_rows=1440,
        known_raw_score_forwards=30,
        known_raw_score_rows=270,
        probe_score_forwards=16,
        probe_score_rows=144,
        score_forwards=46,
        score_rows=414,
        recurrent_unroll_steps=0,
    )
    protocol["limits"]["shared_cleanup_seconds"] = 220


def _packet(episode: Any) -> AttributionBatch:
    features, targets = episode_arrays(episode)
    return AttributionBatch(
        episode.role,
        episode.context,
        tuple(f"{episode.episode_id}:step:{i}" for i in range(len(features))),
        features,
        targets,
        episode,
        7,
    )


def _history_packets(packets: tuple[AttributionBatch, ...]) -> tuple[AttributionBatch, ...]:
    """Derive owned per-episode features without exposing metadata to learners."""
    return tuple(
        _packet(encode_public_history_episode(shared._episode(packet))) for packet in packets
    )


def _history_learner(method: str, *, wide_neural: bool = False) -> Any:
    if type(wide_neural) is not bool or (
        wide_neural and (type(method) is not str or method not in WIDE_METHODS)
    ):
        raise ValueError("wide_neural requires a registered native H32 method or fixed64 control")
    if wide_neural:
        if method in relaxation.BOUND_METHODS:
            return InvariantTorchFeatureLearner(relaxation.policy(method, wide_neural=True))
        return InvariantTorchFeatureLearner(
            InvariantTorchFeaturePolicy(
                TorchFeaturePolicy(
                    method.removeprefix("public_invariant_"), 296, 32, 17, num_classes=7
                )
            )
        )
    if type(method) is str and method in cue_relative.METHODS:
        return CueRelativeTorchFeatureLearner(cue_relative.policy(method))
    if type(method) is str and method in relaxation.BOUND_METHODS:
        return InvariantTorchFeatureLearner(relaxation.policy(method))
    if type(method) is str and method == memory.METHOD:
        return memory.learner()
    if method == "recurrent_bc":
        return shared._learner(method)
    if type(method) is str and method in INVARIANT_METHODS:
        return InvariantTorchFeatureLearner(
            InvariantTorchFeaturePolicy(
                TorchFeaturePolicy(
                    method.removeprefix("public_invariant_"), 296, 8, 17, num_classes=7
                )
            )
        )
    return TorchFeatureLearner(TorchFeaturePolicy(method, 296, 8, 17, num_classes=7))


def _invariant_snapshot(
    state: Any,
    point: int,
    method: str,
    *,
    catalogue_rows: int | None = None,
    wide_neural: bool = False,
) -> Any:
    if type(wide_neural) is not bool or (
        wide_neural
        and (
            type(catalogue_rows) is not int
            or catalogue_rows != 54
            or type(method) is not str
            or method not in WIDE_METHODS
        )
    ):
        raise ValueError("wide_neural snapshot requires the original joint54-row invariant method")
    if type(method) is str and method in cue_relative.METHODS:
        return cue_relative.snapshot(state, point, method, catalogue_rows=catalogue_rows)
    if catalogue_rows is not None and (
        type(catalogue_rows) is not int
        or catalogue_rows != 54
        or type(method) is not str
        or method not in BOUND_METHODS
    ):
        raise ValueError("joint54-row checkpoint requires its exact bound neural method")
    if type(method) is str and method == memory.METHOD:
        if type(point) is not int or point not in shared.POINTS:
            raise ValueError("memory checkpoint requires an original declared native point")
        shared._invariant_memory_counter(state, point, method)
        return copy_invariant_memory_snapshot(state)
    if type(method) is not str or method not in BOUND_METHODS:
        raise ValueError("bound checkpoint requires its declared invariant method")
    if type(point) is not int or (
        point not in shared.POINTS if catalogue_rows is None else not 0 <= point <= 128
    ):
        raise ValueError(
            "bound checkpoint requires an original declared native point"
            if catalogue_rows is None
            else "joint54-row checkpoint requires a native point in0..128"
        )
    if catalogue_rows is None:
        return shared._invariant_feature_snapshot(state, point, method)
    return shared._invariant_feature_snapshot(
        state,
        point,
        method,
        catalogue_rows=catalogue_rows,
        **({"wide_neural": True} if wide_neural else {}),
    )


def _invariant_fit_sources(
    sources: tuple[AttributionBatch, ...], *, cue_relative_history: bool = False
) -> tuple[Any, ...]:
    if type(cue_relative_history) is not bool:
        raise ValueError("cue_relative_history must be an exact bool")
    if type(sources) is not tuple or len(sources) != 6:
        raise ValueError("bound TRAIN fit requires all six original public-history cue packets")
    seen: set[str] = set()
    for index, packet in enumerate(sources):
        _validate_batch(packet, "train", seen)
        episode = shared._episode(packet)
        if (
            packet.features.shape != (9, 296)
            or packet.num_classes != 7
            or episode.context != ("A", "B", "C_train")[index // 2]
            or episode.episode_id.rsplit("-", 1)[-1] != ("blue", "red")[index % 2]
        ):
            raise ValueError("bound TRAIN fit requires fixed296/K7 nine-row TRAIN cue order")
        for row in _feature_keys(packet.features, 296):
            canonicalize_public_history(row)
    if cue_relative_history:
        cue_relative.validate_histories(tuple(shared._episode(packet) for packet in sources))
    return shared._signature(sources)


def _invariant_probe_inputs(data, *, cue_relative_history: bool = False) -> None:
    if type(cue_relative_history) is not bool:
        raise ValueError("cue_relative_history must be an exact bool")
    if type(data) is not tuple or len(data) != 2:
        raise ValueError("bound probe requires an exact training/evaluation tuple")
    _validate_batches(*data)
    if len(data[0]) != 16 or len(data[1]) != 2:
        raise ValueError("bound probe requires original sixteen updates and two E score packets")
    pair = _probe_pair(data)
    for packets in data:
        for index, packet in enumerate(packets):
            if (
                packet.features.shape != (9, 296)
                or packet.num_classes != 7
                or packet.context != pair[index % 2]
            ):
                raise ValueError("bound probe requires fixed alternating nine-row296/K7 E packets")
            for row in _feature_keys(packet.features, 296):
                canonicalize_public_history(row)

    if cue_relative_history:
        cue_relative.validate_histories(
            tuple(shared._episode(packet) for packets in data for packet in packets)
        )


def _collect_probe_sources(
    check_boundary: Callable[[], None],
    reports: list[dict[str, Any]],
    *,
    distinct_probe: bool = False,
) -> tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]]:
    if type(distinct_probe) is not bool:
        raise ValueError("distinct_probe must be an exact bool")
    originals: list[AttributionBatch] = []
    for role in ("probe_train", "probe_test"):
        if distinct_probe:
            from src.core.minigrid_key_door import predefined_distinct_probe_layouts

            layouts = predefined_distinct_probe_layouts()
        else:
            layouts = predefined_probe_layouts()
        for layout, cue in zip(layouts, ("blue", "red"), strict=True):
            check_boundary()
            episode_id = f"minigrid-probe-source-{role}-{cue}"
            try:
                environment = MiniGridKeyDoor(
                    layout, cue, seed=17, max_steps=16, role=role, episode_id=episode_id
                )
            except Exception as error:
                reports.append(
                    dict(
                        status="construction_failed",
                        role=role,
                        context=layout.context,
                        cue=cue,
                        episode_id=episode_id,
                        reason=f"{type(error).__name__}: {error}",
                        environment_steps_attempted=None,
                        environment_steps=None,
                        failure_work_may_be_unobserved=True,
                    )
                )
                raise
            teacher, problem = CueKeyDoorTeacher(), None
            packet = None
            try:
                for _ in range(16):
                    check_boundary()
                    action = teacher.choose_action(environment.observation)
                    check_boundary()
                    environment.step(action)
                    check_boundary()
                    if environment.terminal:
                        break
                episode = environment.to_demonstration()
                if len(episode.transitions) != 9 or environment.report["total_return"] <= 0:
                    raise ValueError("reserved teacher must complete the fixed nine-step task")
                packet = _packet(episode)
                originals.append(packet)
            except Exception as error:
                problem = error
                raise
            finally:
                cleanup: Exception | None = None
                secondary: dict[str, Any] = {}
                try:
                    environment.close()
                except Exception as error:
                    cleanup = error
                    secondary["close_error"] = f"{type(error).__name__}: {error}"
                try:
                    observed = environment.report
                except Exception as error:
                    observed = dict(failure_work_may_be_unobserved=True)
                    secondary["readback_error"] = f"{type(error).__name__}: {error}"
                    cleanup = cleanup or error
                observed.update(secondary)
                observed["source_array_bytes"] = (
                    None if packet is None else packet.features.nbytes + packet.targets.nbytes
                )
                if problem is not None:
                    observed["primary_error"] = f"{type(problem).__name__}: {problem}"
                reports.append(observed)
                if cleanup is not None and problem is None:
                    raise cleanup
    training = tuple(
        _packet(
            replace(
                shared._episode(originals[i % 2]),
                episode_id=f"{shared._episode(originals[i % 2]).episode_id}:copy:{i}",
            )
        )
        for i in range(16)
    )
    evaluation = tuple(originals[2:])
    _validate_batches(training, evaluation)
    return training, evaluation


def _score_checkpoint(
    source: Any,
    state: Any,
    *,
    method: str,
    updates: int,
    sources: tuple[AttributionBatch, ...],
    budget: ToyBudgetSession,
    check_boundary: Callable[[], None],
    activation_health: bool = False,
    gradient_geometry: bool = False,
    prior_gradient_geometry: bool = False,
    train_action_margins: bool = False,
    catalogue_rows: int | None = None,
    pc_update_diagnostics: bool = False,
    weighted_catalogue: bool = False,
    wide_neural: bool = False,
) -> dict[str, Any]:
    if type(wide_neural) is not bool or (
        wide_neural
        and (
            catalogue_rows != 54
            or method not in WIDE_METHODS
            or any(
                (
                    activation_health,
                    gradient_geometry,
                    train_action_margins,
                    weighted_catalogue,
                )
            )
            or type(prior_gradient_geometry) is not bool
            or (
                prior_gradient_geometry
                and (method not in INVARIANT_METHODS or pc_update_diagnostics is not False)
            )
        )
    ):
        raise ValueError(
            "wide_neural readout requires the fixed joint recipe without extra diagnostics"
        )
    if (
        wide_neural
        and prior_gradient_geometry
        and (type(updates) is not int or updates not in shared.POINTS)
    ):
        raise ValueError("H32 prior gradients require the five original TRAIN checkpoints")
    if type(weighted_catalogue) is not bool or (
        weighted_catalogue and pc_update_diagnostics is not True
    ):
        raise ValueError("weighted checkpoint diagnostics require pc_update_diagnostics")
    if type(pc_update_diagnostics) is not bool or (
        pc_update_diagnostics
        and (
            catalogue_rows != 54
            or type(catalogue_rows) is not int
            or method not in ("public_invariant_pc", "public_invariant_neutral_cpc")
            or type(updates) is not int
            or updates not in (0, 128)
        )
    ):
        raise ValueError(
            "pc_update_diagnostics requires exact joint54-row PC/neutral checkpoints0/128"
        )
    if catalogue_rows is not None and (
        type(catalogue_rows) is not int
        or catalogue_rows != 54
        or type(method) is not str
        or method not in BOUND_METHODS
    ):
        raise ValueError("joint54-row TRAIN readout requires its exact bound neural method")
    if type(train_action_margins) is not bool:
        raise ValueError("train_action_margins must be an exact bool")
    if type(prior_gradient_geometry) is not bool or (
        prior_gradient_geometry and method not in INVARIANT_METHODS
    ):
        raise ValueError("prior_gradient_geometry requires an exact bool and bound neural method")
    memory_method = type(method) is str and method == memory.METHOD
    same_state = _same_value if memory_method else _same_state
    bound_method = type(method) is str and (method in BOUND_METHODS or memory_method)
    if (
        type(state)
        in (InvariantTorchFeatureSnapshot, InvariantMemorySnapshot, CueRelativeTorchFeatureSnapshot)
        and not bound_method
    ):
        raise ValueError("encoding-bound checkpoint requires its own invariant method")
    if type(activation_health) is not bool:
        raise ValueError("activation_health must be an exact bool")
    if method in cue_relative.METHODS and (
        activation_health
        or gradient_geometry
        or prior_gradient_geometry
        or train_action_margins
        or pc_update_diagnostics
    ):
        raise ValueError("cue-relative readouts require projection-aware diagnostic readers")
    if memory_method and activation_health:
        raise ValueError("memory control has no native activation-health port")
    if bound_method and gradient_geometry is not False:
        raise ValueError("bound checkpoints do not support gradient diagnostics")
    if type(gradient_geometry) is not bool or (gradient_geometry and method != "recurrent_bc"):
        raise ValueError("gradient_geometry must be an exact bool and requires recurrent_bc")
    paired_method = method == "recurrent_paired_bc"
    if paired_method and (activation_health is not False or gradient_geometry is not False):
        raise ValueError("paired checkpoints support original raw scores only")
    strict_scores = gradient_geometry or paired_method or bound_method or train_action_margins
    signature = (
        _invariant_fit_sources(sources, cue_relative_history=True)
        if method in cue_relative.METHODS
        else _invariant_fit_sources(sources)
        if bound_method
        else gradient_readouts.validate_sources(sources)
        if strict_scores
        else None
    )
    if paired_method:
        shared._paired_counter(state, updates)
    if bound_method:
        if catalogue_rows is None:
            _invariant_snapshot(state, updates, method)
        else:
            _invariant_snapshot(
                state,
                updates,
                method,
                catalogue_rows=catalogue_rows,
                **({"wide_neural": True} if wide_neural else {}),
            )
    work: dict[str, Any] = dict(
        score_forward_calls_attempted=0,
        score_forward_calls_completed=0,
        score_rows_attempted=0,
        score_rows_completed=0,
        failure_work_may_be_unobserved=False,
    )
    if bound_method:
        work.update(internal_projection_rows_returned=0, internal_projection_array_bytes_returned=0)
    if activation_health:
        work.update(
            activation_calls_attempted=0,
            activation_calls_completed=0,
            activation_rows_attempted=0,
            activation_rows_completed=0,
            activation_array_bytes=0,
            activation_svd_calls_attempted=0,
            activation_svd_calls_completed=0,
            activation_seconds=0.0,
            activation_feedforward_hidden_affines_completed=0,
            activation_recurrent_hidden_steps_completed=0,
            activation_recurrent_output_projections_completed=0,
        )
    if bound_method and activation_health:
        work.update(
            activation_projection_rows_returned=0, activation_projection_array_bytes_returned=0
        )
    report: dict[str, Any] = dict(
        method=method,
        updates=updates,
        role="train",
        scope="teacher_forced_train_fit_not_live_success",
        status="running",
        reason=None,
        scores=[],
        work=work,
        source_unchanged=None,
        state_unchanged=None,
    )
    if catalogue_rows is not None:
        # Why: a joint wake sees all54 original rows. Keep that exposure law
        # explicit and guarded, without changing the existing nine-row baseline.
        report.update(
            catalogue_rows_per_update=catalogue_rows, expected_training_examples=54 * updates
        )
    if memory_method:
        report["table_observation"] = memory.table_observation(state)
    table_owner = report.get("table_observation")
    table_expected = deepcopy(table_owner)
    if gradient_geometry:
        report["cue_gradient_geometry"] = []
        work["cue_gradient_geometry"] = gradient_readouts.empty_work()
    if prior_gradient_geometry:
        report["prior_gradient_geometry"] = []
        work["prior_gradient_geometry"] = gradient_readouts.empty_prior_work()
    if pc_update_diagnostics:
        report["pc_update_diagnostics"] = []
        work["pc_update_diagnostics"] = gradient_readouts.empty_pc_work(
            **({"weighted": True} if weighted_catalogue else {})
        )
    if train_action_margins:
        work.update(action_margins.empty_work())
    if paired_method:
        report.update(training_episodes=2 * updates, training_rows=18 * updates)
    paired_work = deepcopy(work) if paired_method or bound_method or train_action_margins else None
    paired_metadata = (
        {
            key: deepcopy(report[key])
            for key in (
                "method",
                "updates",
                "role",
                "scope",
                "status",
                "reason",
                "training_episodes",
                "training_rows",
                "source_unchanged",
                "state_unchanged",
                "catalogue_rows_per_update",
                "expected_training_examples",
            )
            if key in report
        }
        if paired_method or bound_method or train_action_margins
        else {}
    )
    score_records: list[dict[str, Any]] = []
    score_list = report["scores"]
    score_payload = pickle.dumps([], protocol=5)
    prior_running = False
    prior_records = report.get("prior_gradient_geometry")
    prior_work_owner = work.get("prior_gradient_geometry")
    prior_payload = pickle.dumps(prior_records, protocol=5)
    prior_proof: Callable[[], None] | None = None

    def capture_prior() -> None:
        nonlocal prior_records, prior_work_owner, prior_payload, prior_proof
        prior_records = report["prior_gradient_geometry"]
        prior_work_owner = work["prior_gradient_geometry"]
        prior_payload = pickle.dumps(prior_records, protocol=5)
        # Why: the helper owns its evidence while running; publication still has
        # callbacks afterward. Reuse the existing paid-report seal for that tail.
        prior_proof, _ = paid_report_guard(dict(records=prior_records, work=prior_work_owner))

    if prior_gradient_geometry:
        capture_prior()
    pc_running = False
    pc_records = report.get("pc_update_diagnostics")
    pc_payload = pickle.dumps(pc_records, protocol=5)
    pc_owners: tuple[Any, ...] = ()
    pc_work_owner = work.get("pc_update_diagnostics")
    pc_work_payload = pickle.dumps(pc_work_owner, protocol=5)

    def capture_pc():
        nonlocal pc_records, pc_payload, pc_owners, pc_work_owner, pc_work_payload
        pc_records = report["pc_update_diagnostics"]
        pc_payload = pickle.dumps(pc_records, protocol=5)
        pc_owners = tuple(
            owner for record in pc_records for owner in gradient_readouts.pc_record_owners(record)
        )
        pc_work_owner = work["pc_update_diagnostics"]
        pc_work_payload = pickle.dumps(pc_work_owner, protocol=5)

    def charge_health(**increments: Any) -> None:
        # Why: strict bound guards compare public costs with a private paid ledger.
        # Only observed operation increments enter both; never copy mutable public work.
        for key, value in increments.items():
            work[key] += value
            if paired_work is not None:
                paired_work[key] += value

    def capture_scores() -> None:
        nonlocal score_payload, score_records
        if strict_scores:
            score_records = list(report["scores"])
            score_payload = pickle.dumps(report["scores"], protocol=5)

    def input_evidence() -> None:
        if prior_gradient_geometry and not prior_running:
            if (
                report.get("prior_gradient_geometry") is not prior_records
                or work.get("prior_gradient_geometry") is not prior_work_owner
            ):
                raise ValueError("preceding prior-gradient evidence replaced")
            assert prior_proof is not None
            prior_proof()
        if (
            pc_update_diagnostics
            and not pc_running
            and (
                report.get("pc_update_diagnostics") is not pc_records
                or work.get("pc_update_diagnostics") is not pc_work_owner
                or pickle.dumps(work.get("pc_update_diagnostics"), protocol=5) != pc_work_payload
                or pickle.dumps(report.get("pc_update_diagnostics"), protocol=5) != pc_payload
                or any(
                    a is not b
                    for a, b in zip(
                        (
                            owner
                            for record in report["pc_update_diagnostics"]
                            for owner in gradient_readouts.pc_record_owners(record)
                        ),
                        pc_owners,
                        strict=True,
                    )
                )
            )
        ):
            raise ValueError("preceding pc-update evidence changed")
        if memory_method and (
            report.get("table_observation") is not table_owner
            or not same_state(report.get("table_observation"), table_expected)
        ):
            raise ValueError("memory TRAIN table observation changed")
        if strict_scores and (
            report.get("scores") is not score_list
            or report.get("work") is not work
            or signature is None
            or not shared._check_signature(sources, signature)
            or (
                paired_work is not None
                and (
                    {
                        key: value
                        for key, value in work.items()
                        if not (
                            (prior_running and key == "prior_gradient_geometry")
                            or (pc_running and key == "pc_update_diagnostics")
                        )
                    }
                    != {
                        key: value
                        for key, value in paired_work.items()
                        if not (
                            (prior_running and key == "prior_gradient_geometry")
                            or (pc_running and key == "pc_update_diagnostics")
                        )
                    }
                    or any(type(work[key]) is not type(value) for key, value in paired_work.items())
                )
            )
            or any(
                type(report.get(key)) is not type(value) or report.get(key) != value
                for key, value in paired_metadata.items()
            )
            or pickle.dumps(report["scores"], protocol=5) != score_payload
            or len(report["scores"]) != len(score_records)
            or any(
                score is not owner
                for score, owner in zip(report["scores"], score_records, strict=True)
            )
        ):
            raise ValueError("raw-fit TRAIN packet or preceding score evidence changed")

    def bind_snapshot(value, point):
        if not bound_method:
            return value
        if catalogue_rows is None:
            return _invariant_snapshot(value, point, method)
        return _invariant_snapshot(
            value,
            point,
            method,
            catalogue_rows=catalogue_rows,
            **({"wide_neural": True} if wide_neural else {}),
        )

    expected = bind_snapshot(state, updates) if bound_method else deepcopy(state)
    source_before = scorer = None

    def read_snapshot(owner, point):
        value = owner.snapshot_state()
        return bind_snapshot(value, point)

    def guard() -> None:
        input_evidence()
        check_boundary()
        budget.before_final()
        if not same_state(expected, state):
            raise ValueError("owned raw-fit checkpoint changed")
        if source_before is not None and not same_state(source_before, read_snapshot(source, 0)):
            raise ValueError("raw-fit source changed")
        if scorer is not None and not same_state(expected, read_snapshot(scorer, updates)):
            raise ValueError("raw-fit scorer learned state changed")
        input_evidence()
        if strict_scores and not same_state(expected, state):
            raise ValueError("owned raw-fit checkpoint changed after callback")
        if bound_method:
            bind_snapshot(state, updates)
            check_boundary()
            input_evidence()
            if not same_state(expected, state):
                raise ValueError("bound raw-fit checkpoint changed after final readback callback")

    try:
        guard()
        if not bound_method:
            (shared._paired_counter if paired_method else shared._counter)(expected, updates)
        source_before = deepcopy(read_snapshot(source, 0))
        if bound_method and not same_state(expected.policy, source_before.policy):
            raise ValueError("bound TRAIN source policy differs from its checkpoint")
        guard()
        fork = source.fork()
        guard()
        if fork is source:
            raise ValueError("raw-fit scorer must be an independent fork")
        fork.restore_state(deepcopy(expected))
        scorer = fork
        guard()
        for packet in sources:
            guard()
            if packet.role != "train" or shared._episode(packet).role != "train":
                raise ValueError("checkpoint diagnostics require TRAIN episodes")
            work["score_forward_calls_attempted"] += 1
            work["score_rows_attempted"] += len(packet.features)
            if paired_work is not None:
                paired_work["score_forward_calls_attempted"] += 1
                paired_work["score_rows_attempted"] += len(packet.features)
            prediction = scorer.predict(packet.features.copy())
            work["score_forward_calls_completed"] += 1
            work["score_rows_completed"] += len(packet.features)
            if paired_work is not None:
                paired_work["score_forward_calls_completed"] += 1
                paired_work["score_rows_completed"] += len(packet.features)
            if bound_method:
                if paired_work is None:
                    raise ValueError("bound score work capture is unavailable")
                for counters in (work, paired_work):
                    counters["internal_projection_rows_returned"] += len(packet.features)
                    counters["internal_projection_array_bytes_returned"] += packet.features.nbytes
            guard()
            validate_class_probabilities(prediction, 7)
            if prediction.shape != (9, 7):
                raise ValueError("raw fit requires exactly nine K7 probability rows")
            metrics = probability_scores(prediction, packet.targets, 7)
            actions = np.argmax(prediction, axis=1)
            score: dict[str, Any] = dict(
                context=packet.context,
                cue=shared._episode(packet).episode_id.rsplit("-", 1)[-1],
                episode_id=shared._episode(packet).episode_id,
                row_ids=list(packet.row_ids),
                **metrics,
                exact_sequence_accuracy=float(np.array_equal(actions, packet.targets[:, 0])),
                predicted_actions=actions.tolist(),
                target_actions=packet.targets[:, 0].astype(int).tolist(),
                probabilities=prediction.tolist(),
            )
            # A failed optional diagnostic cannot erase an already paid score.
            report["scores"].append(score)
            capture_scores()
            if train_action_margins:
                guard()
                charge_health(
                    action_margin_calls_attempted=1,
                    action_margin_rows_attempted=len(prediction),
                    action_margin_input_array_bytes=prediction.nbytes + packet.targets.nbytes,
                )
                before = monotonic()
                try:
                    margins = probability_margins(prediction.copy(), packet.targets.copy(), 7)
                    charge_health(
                        action_margin_calls_completed=1,
                        action_margin_rows_completed=len(prediction),
                    )
                finally:
                    charge_health(action_margin_seconds=monotonic() - before)
                # Preserve the paid raw score if reduction or a callback fails;
                # never reseal preceding scores before their guard has passed.
                guard()
                score["action_margins"] = margins
                capture_scores()
                guard()
            if activation_health:
                before = monotonic()
                try:
                    charge_health(
                        activation_calls_attempted=1, activation_rows_attempted=len(packet.features)
                    )
                    hidden = read_native_activations(deepcopy(expected), packet.features.copy())
                    charge_health(
                        activation_calls_completed=1, activation_rows_completed=len(packet.features)
                    )
                    if bound_method:
                        charge_health(
                            activation_projection_rows_returned=len(packet.features),
                            activation_projection_array_bytes_returned=packet.features.nbytes,
                            activation_feedforward_hidden_affines_completed=1,
                        )
                        # A read may execute callbacks: validate preceding paid evidence
                        # before publishing health can capture the public score list.
                        guard()
                    if (
                        type(hidden) is not np.ndarray
                        or hidden.dtype != np.float64
                        or hidden.shape != (9, 8)
                    ):
                        raise ValueError(
                            "native health requires owned nine-row/eight-unit float64 activations"
                        )
                    if not np.isfinite(hidden).all() or np.any(np.abs(hidden) > 1):
                        raise ValueError(
                            "native health activations must be finite tanh values in [-1, 1]"
                        )
                    charge_health(activation_array_bytes=hidden.nbytes)
                    if method == "recurrent_bc":
                        charge_health(
                            activation_recurrent_hidden_steps_completed=9,
                            activation_recurrent_output_projections_completed=9,
                        )
                    elif not bound_method:
                        charge_health(activation_feedforward_hidden_affines_completed=1)
                    score["activation_health"] = dict(
                        definition="label_free_native_tanh_prior_or_ordered_recurrent_hidden_v1",
                        activations=hidden.tolist(),
                        metrics=None,
                        status="read",
                    )
                    capture_scores()
                    guard()
                    charge_health(activation_svd_calls_attempted=1)
                    metrics_health = diagnose_activations((hidden,))
                    charge_health(activation_svd_calls_completed=1)
                    guard()
                    score["activation_health"].update(
                        metrics=[asdict(value) for value in metrics_health], status="complete"
                    )
                    capture_scores()
                finally:
                    charge_health(activation_seconds=monotonic() - before)
        guard()
        if gradient_geometry:
            gradient_readouts.run_gradient_readouts(report, state, sources, check_boundary=guard)
        if prior_gradient_geometry:
            # Why: the composition owns an independent paid diagnostic ledger;
            # this outer guard continues checking every original raw-fit charge.
            prior_running = True
            try:
                gradient_readouts.run_prior_gradient_readouts(
                    report, state, sources, check_boundary=guard
                )
            finally:
                prior_running = False
                if paired_work is not None:
                    paired_work["prior_gradient_geometry"] = deepcopy(
                        report["work"]["prior_gradient_geometry"]
                    )
                capture_prior()
            guard()
        if pc_update_diagnostics:
            pc_running = True
            try:
                gradient_readouts.run_pc_update_readouts(
                    report,
                    state,
                    sources,
                    check_boundary=guard,
                    **({"weighted": True} if weighted_catalogue else {}),
                )
            finally:
                pc_running = False
                if paired_work is not None:
                    paired_work["pc_update_diagnostics"] = deepcopy(
                        report["work"]["pc_update_diagnostics"]
                    )
                capture_pc()
            guard()
        report.update(status="complete", source_unchanged=True, state_unchanged=True)
    except Exception as error:
        if paired_method or bound_method or train_action_margins:
            report.update(paired_metadata)
        if strict_scores:
            report["scores"] = pickle.loads(score_payload)
        if prior_gradient_geometry:
            report["prior_gradient_geometry"] = pickle.loads(prior_payload)
        if pc_update_diagnostics:
            report["pc_update_diagnostics"] = pickle.loads(pc_payload)
        if paired_work is not None:
            work.clear()
            work.update(paired_work)
            report["work"] = work
        work["failure_work_may_be_unobserved"] = (
            work["score_forward_calls_attempted"] > work["score_forward_calls_completed"]
            or work.get("activation_calls_attempted", 0) > work.get("activation_calls_completed", 0)
            or work.get("activation_svd_calls_attempted", 0)
            > work.get("activation_svd_calls_completed", 0)
            or work.get("cue_gradient_geometry", {}).get("failure_work_may_be_unobserved", False)
            or work.get("prior_gradient_geometry", {}).get("failure_work_may_be_unobserved", False)
            or work.get("pc_update_diagnostics", {}).get("failure_work_may_be_unobserved", False)
            or work.get("action_margin_calls_attempted", 0)
            > work.get("action_margin_calls_completed", 0)
        )
        report.update(status="failed", reason=f"{type(error).__name__}: {error}")
        if memory_method:
            report["table_observation"] = deepcopy(table_expected)
    return report


def _budget_binding(budget: ToyBudgetSession) -> tuple[Any, ...]:
    return (
        id(budget),
        id(budget.budget),
        pickle.dumps(budget.budget, protocol=5),
        id(budget.clock),
        budget.started_at,
        id(budget.process_rss_sampler),
        id(budget.progress),
    )


def _probe_outcomes(report: dict[str, Any]) -> dict[str, Any]:
    curve = []
    for point in report["checkpoints"]:
        curve.append(
            dict(
                updates=point["updates"],
                ce=sum(score["ce"] for score in point["scores"].values()) / 2,
                accuracy=sum(score["accuracy"] for score in point["scores"].values()) / 2,
                exact_sequence_accuracy=sum(
                    all(decision["correct"] for decision in detail["decisions"])
                    for detail in point["decision_diagnostics"].values()
                )
                / 2,
            )
        )
    return dict(
        curve=curve,
        ce_gain=curve[0]["ce"] - curve[-1]["ce"],
        accuracy_gain=curve[-1]["accuracy"] - curve[0]["accuracy"],
        exact_sequence_gain=curve[-1]["exact_sequence_accuracy"]
        - curve[0]["exact_sequence_accuracy"],
    )


def _paired_probe_data(
    data: tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]],
) -> tuple[
    tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]],
    tuple[DemonstrationEpisode, ...],
]:
    """Own the original fixed E recipe, joining only its array transport."""
    if (
        type(data) is not tuple
        or len(data) != 2
        or type(data[0]) is not tuple
        or len(data[0]) != 16
        or type(data[1]) is not tuple
        or len(data[1]) != 2
    ):
        raise ValueError("paired E probe requires original16training and2evaluation packets")
    _validate_batches(*data)
    pair = _probe_pair(data)
    owned: list[tuple[DemonstrationEpisode, ...]] = []
    for packets, role in zip(data, ("probe_train", "probe_test"), strict=True):
        episodes: list[DemonstrationEpisode] = []
        for index, packet in enumerate(packets):
            episode = shared._episode(packet)
            episode_batch._exact(episode, DemonstrationEpisode)
            if (
                packet.features.shape != (9, 148)
                or packet.num_classes != 7
                or packet.context != pair[index % 2]
                or type(episode.transitions) is not tuple
                or len(episode.transitions) != 9
                or episode.role != role
            ):
                raise ValueError("paired E probe requires original alternating9row148/K7 episodes")
            for step in episode.transitions:
                episode_batch._exact(step, DemonstrationTransition)
            if index >= 2 and episode.transitions != episodes[index % 2].transitions:
                raise ValueError("paired E training must repeat its two original complete episodes")
            episodes.append(copy_episode(episode))
        owned.append(tuple(episodes))
    schedule, evaluation_episodes = owned
    if len({episode.episode_id for group in owned for episode in group}) != 18:
        raise ValueError("paired E probe requires distinct original episode IDs")
    training = tuple(
        AttributionBatch(
            "probe_train",
            data[0][index].context,
            tuple(row for packet in data[0][index : index + 2] for row in packet.row_ids),
            np.concatenate([packet.features for packet in data[0][index : index + 2]]),
            np.concatenate([packet.targets for packet in data[0][index : index + 2]]),
            None,
            7,
        )
        for index in range(0, 16, 2)
    )
    evaluation = tuple(
        AttributionBatch(
            "probe_test",
            episode.context,
            packet.row_ids,
            packet.features.copy(),
            packet.targets.copy(),
            None,
            7,
        )
        for episode, packet in zip(evaluation_episodes, data[1], strict=True)
    )
    _validate_batches(training, evaluation)
    return (training, evaluation), schedule


def _validate_a_training_readouts(packets, width: int) -> None:
    if type(packets) is not tuple or len(packets) != 2:
        raise ValueError("old A readouts require exactly two original TRAIN cue episodes")
    seen: set[str] = set()
    for packet, cue in zip(packets, ("blue", "red"), strict=True):
        _validate_batch(packet, "train", seen)
        episode = shared._episode(packet)
        if (
            packet.features.shape != (9, width)
            or packet.num_classes != 7
            or packet.context != "A"
            or not episode.episode_id.endswith(cue)
        ):
            raise ValueError("old A readouts require original A blue/red nine-row TRAIN geometry")
        if width == 296:
            for row in _feature_keys(packet.features, 296):
                canonicalize_public_history(row)


def _run_probe(
    slot: dict[str, Any],
    arm: dict[str, Any],
    state: Any,
    data: tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]],
    *,
    sampler: Any,
    deadline: float,
    check_boundary: Callable[[], None],
    paired: bool = False,
    training_readouts: tuple[AttributionBatch, ...] = (),
    return_to_a: bool = False,
    probe_rollouts: bool = False,
    deferred: deferred_rollout.DeferredProbeRollouts | None = None,
    catalogue_rows: int | None = None,
    probe_inference: str | None = None,
    old_a_rehearsal: bool = False,
    wide_neural: bool = False,
    extended_wide_probe_budget: bool = False,
    extra_f_exposure: bool = False,
) -> Callable[[], None]:
    if type(extra_f_exposure) is not bool or (
        extra_f_exposure
        and (
            wide_neural is not True
            or old_a_rehearsal is not False
            or probe_rollouts is not False
            or deferred is not None
            or extended_wide_probe_budget is not False
        )
    ):
        raise ValueError(
            "extra_f_exposure requires the explicit H32 F48/A16 route without replay or games"
        )
    _validate_extended_wide_probe_budget(
        extended_wide_probe_budget,
        wide_neural,
        probe_rollouts,
        deferred_probe_rollouts=deferred is not None,
    )
    if type(wide_neural) is not bool or (
        wide_neural
        and (
            type(catalogue_rows) is not int
            or catalogue_rows != 54
            or paired
            or not training_readouts
            or return_to_a is not True
            or type(slot.get("checkpoint_updates")) is not int
            or slot["checkpoint_updates"] not in (0, 128)
        )
    ):
        raise ValueError("wide_neural probe requires original0/128 joint F16/old-A/A16")
    if type(old_a_rehearsal) is not bool:
        raise ValueError("old_a_rehearsal must be an exact bool")
    _validate_return_to_a(return_to_a, bool(training_readouts))
    if type(paired) is not bool:
        raise ValueError("paired probe selection must be an exact bool")
    if type(training_readouts) is not tuple:
        raise ValueError("training_readouts must be an exact tuple")
    if training_readouts:
        if paired:
            raise ValueError("paired transport does not admit old A readouts")
        _validate_a_training_readouts(training_readouts, data[0][0].features.shape[1])
    row_method = arm.get("row", {}).get("method")
    method: Any = slot.get(
        "method",
        row_method
        if row_method is not None
        else getattr(getattr(state, "policy", None), "method", None),
    )
    if wide_neural and (
        type(method) is not str
        or method not in WIDE_METHODS
        or _probe_pair(data) != ("F-blue", "F-red")
    ):
        raise ValueError(
            "wide_neural probe requires its declared native method and original F pair"
        )
    if (
        wide_neural
        and method in relaxation.BOUND_METHODS
        and (
            old_a_rehearsal is not False
            or extra_f_exposure is not False
            or probe_rollouts is not False
            or deferred is not None
            or extended_wide_probe_budget is not False
        )
    ):
        # Why: S64 is a distinct inference-cost control of the original F16/A16
        # experiment; replay, games and exposure controls need separate recipes.
        raise ValueError("H32 fixed64 probe requires original F16/A16 without extra options")
    if probe_inference is not None:
        if (
            type(probe_inference) is not str
            or probe_inference not in bounded_probe.MODES
            or type(method) is not str
            or method.removeprefix("public_invariant_") not in bounded_probe.METHODS
            or catalogue_rows != 54
            or paired
            or not training_readouts
            or return_to_a is not True
            or probe_rollouts
            or deferred is not None
            or (
                wide_neural
                and (
                    method not in ("public_invariant_pc", "public_invariant_neutral_cpc")
                    or old_a_rehearsal is not False
                    or extra_f_exposure is not False
                    or extended_wide_probe_budget is not False
                )
            )
        ):
            raise ValueError(
                "weighted probe recipe requires its unpaired invariant joint F16/A16 route"
            )
    if old_a_rehearsal:
        probe_rehearsal.validate_option(
            True,
            public_history=True,
            invariant_history=True,
            joint_catalogue=catalogue_rows == 54,
            qualification_method=method.removeprefix("public_invariant_")
            if type(method) is str and method.startswith("public_invariant_")
            else None,
            distinct_probe=_probe_pair(data) == ("F-blue", "F-red"),
            old_task_readouts=bool(training_readouts),
            return_to_a=return_to_a,
            probe_inference=probe_inference,
            probe_rollouts=probe_rollouts,
            deferred_probe_rollouts=deferred is not None,
            **({"wide_neural": True} if wide_neural else {}),
        )
        if paired:
            raise ValueError("old-A rehearsal requires unpaired original F episodes")
    if catalogue_rows is not None and (
        type(catalogue_rows) is not int
        or catalogue_rows != 54
        or type(method) is not str
        or method not in BOUND_METHODS
        or paired
    ):
        raise ValueError("joint54-row probe requires its exact unpaired bound neural method")
    memory_method = type(method) is str and method == memory.METHOD
    probe_rollout.validate_option(
        probe_rollouts,
        memory_method and return_to_a,
        **({"wide_neural": True} if wide_neural else {}),
    )
    if deferred is not None:
        if (not memory_method and not wide_neural) or not return_to_a or probe_rollouts:
            raise ValueError(
                "deferred readouts require isolated count/H32 return without inline games"
            )
        if wide_neural and (
            type(deferred) is not deferred_rollout.DeferredProbeRollouts
            or deferred.wide_method != method.removeprefix("public_invariant_")
            or deferred.extended_wide_probe_budget is not extended_wide_probe_budget
        ):
            raise ValueError("deferred H32 manager differs from its declared method or budget")
        deferred.check()
    rollout_options: _ProbeRolloutOptions = {}
    if wide_neural:
        rollout_options["wide_method"] = method.removeprefix("public_invariant_")
    if extended_wide_probe_budget:
        rollout_options["extended_wide_probe_budget"] = True
    live = (
        probe_rollout.ProbeRollouts(slot["checkpoint_updates"], **rollout_options)
        if probe_rollouts
        else None
    )
    if live is not None:
        slot["probe_rollouts"] = live.records
    same_state = _same_value if memory_method else _same_state
    bound_method = type(method) is str and (method in BOUND_METHODS or memory_method)
    if (
        type(state)
        in (InvariantTorchFeatureSnapshot, InvariantMemorySnapshot, CueRelativeTorchFeatureSnapshot)
        and not bound_method
    ):
        raise ValueError("bound probe checkpoint requires its declared invariant method")

    def bind_probe_snapshot(value, point):
        if catalogue_rows is None:
            return _invariant_snapshot(value, point, method)
        return _invariant_snapshot(
            value,
            point,
            method,
            catalogue_rows=catalogue_rows,
            **({"wide_neural": True} if wide_neural else {}),
        )

    if bound_method:
        if paired or (row_method is not None and row_method != method):
            raise ValueError("bound probe method/arm/transport identity differs")
        if type(slot.get("checkpoint_updates")) is not int or (
            slot["checkpoint_updates"] not in (0, 128)
            if catalogue_rows is None
            else not 0 <= slot["checkpoint_updates"] <= 128
        ):
            raise ValueError(
                "bound probes require original source checkpoints0/128"
                if catalogue_rows is None
                else "joint54-row probes require a native checkpoint in0..128"
            )
        bind_probe_snapshot(state, slot["checkpoint_updates"])
        if method in cue_relative.METHODS:
            _invariant_probe_inputs(data, cue_relative_history=True)
            if training_readouts:
                cue_relative.validate_histories(
                    tuple(shared._episode(packet) for packet in training_readouts)
                )
        else:
            _invariant_probe_inputs(data)
    catalogue_metadata: tuple[tuple[str, Any], ...] = ()
    if catalogue_rows is not None:
        # This binds only the initial benchmark state. Disposable F and return
        # updates keep their original additional nine-row exposure and budgets.
        slot.update(
            catalogue_rows_per_update=54,
            expected_training_examples=54 * slot["checkpoint_updates"],
        )
        catalogue_metadata = tuple(
            (key, slot[key]) for key in ("catalogue_rows_per_update", "expected_training_examples")
        )
    strict_probe = paired or bound_method or bool(training_readouts)
    source_owner = arm["source"]
    slot_method_present = "method" in slot
    started, budget, binding, problem = monotonic(), None, None, None
    probe_data, schedule = _paired_probe_data(data) if paired else (data, ())
    signatures = shared._signature((*data[0], *data[1])) if strict_probe else None
    readout_signature = shared._signature(training_readouts) if training_readouts else None
    return_training = _return_training_sources(training_readouts) if return_to_a else ()
    return_signature = shared._signature(return_training) if return_to_a else None
    return_budget = return_binding = None
    replay_budget = replay_binding = None
    replay_plan = None
    replay_last_clock = None
    replay_work: dict[str, Any] = dict(
        charged_replay_examples=0,
        attempted_updates=0,
        returned_updates=0,
        committed_updates=0,
        returned_training_rows=0,
        bank_encoded_bytes=0,
        validation_projection_rows=0,
        validation_projected_array_equivalent_bytes=0,
        budget_binding_preserved=False,
        work_unknown=False,
    )
    return_last_updates, return_last_clock = 0, None
    transport_signature = shared._signature((*probe_data[0], *probe_data[1])) if paired else None
    if paired:
        # Validate native payloads before copying a caller-owned tensor graph.
        shared._paired_counter(state, slot["checkpoint_updates"])
    frozen_state = (
        bind_probe_snapshot(state, slot["checkpoint_updates"]) if bound_method else deepcopy(state)
    )
    source_table = memory.table_observation(frozen_state) if memory_method else None
    if memory_method:
        slot["source_table_observation"] = deepcopy(source_table)
    table_owner = slot.get("source_table_observation")
    source_before = None
    updates = 48 if extra_f_exposure else 8 if paired else 16
    points = (
        f_exposure.POINTS if extra_f_exposure else PAIRED_PROBE_POINTS if paired else PROBE_POINTS
    )
    paired_point: Any = slot["checkpoint_updates"] if strict_probe else None
    paired_labels = None
    probe_source: TorchPairedProbeLearner | bounded_probe.TorchTrainWeightedLearner | None = None
    paired_scope = None
    last_updates, last_clock = 0, None
    outcomes = frozen_result = old_readout_outcomes = None
    return_outcomes = None
    slot.update(status="running", reason=None)

    def own_budget_guard() -> None:
        nonlocal last_updates, last_clock, return_last_updates, return_last_clock, replay_last_clock
        if budget is None or binding is None:
            return
        if (
            _budget_binding(budget) != binding
            or type(budget.updates_completed) is not int
            or not last_updates <= budget.updates_completed <= updates
            or type(budget.replay_examples_completed) is not int
            or budget.replay_examples_completed != 0
            or type(budget.last_clock) is not float
            or not np.isfinite(budget.last_clock)
            or (last_clock is not None and budget.last_clock < last_clock)
        ):
            raise ValueError("original independent probe budget or counters changed")
        last_updates, last_clock = budget.updates_completed, budget.last_clock
        if return_budget is not None:
            if (
                return_budget is budget
                or _budget_binding(return_budget) != return_binding
                or type(return_budget.updates_completed) is not int
                or not return_last_updates <= return_budget.updates_completed <= 16
                or type(return_budget.replay_examples_completed) is not int
                or return_budget.replay_examples_completed != 0
                or type(return_budget.last_clock) is not float
                or not np.isfinite(return_budget.last_clock)
                or (return_last_clock is not None and return_budget.last_clock < return_last_clock)
            ):
                raise ValueError("separate A-return budget or counters changed")
            return_last_updates, return_last_clock = (
                return_budget.updates_completed,
                return_budget.last_clock,
            )
        if replay_budget is not None:
            if (
                _budget_binding(replay_budget) != replay_binding
                or type(replay_budget.updates_completed) is not int
                or replay_budget.updates_completed != 0
                or type(replay_budget.replay_examples_completed) is not int
                or not 0 <= replay_budget.replay_examples_completed <= 288
                or type(replay_budget.last_clock) is not float
                or not np.isfinite(replay_budget.last_clock)
                or (replay_last_clock is not None and replay_budget.last_clock < replay_last_clock)
                or replay_budget.replay_examples_completed
                != (
                    0
                    if replay_plan is None
                    else sum(event["charged_exposures"] for event in replay_plan.rehearsal._events)
                )
            ):
                raise ValueError("original old-A replay budget or permanent charges changed")
            replay_last_clock = replay_budget.last_clock

    def pure_guard() -> None:
        own_budget_guard()
        if any(
            type(slot.get(key)) is not type(value) or slot.get(key) != value
            for key, value in catalogue_metadata
        ):
            raise ValueError("original joint-catalogue initial row binding changed")
        if deferred is not None:
            deferred.check()
        if live is not None:
            if slot.get("probe_rollouts") is not live.records:
                raise ValueError("original live endpoint record list replaced")
            live.check()
        if memory_method and (
            slot.get("source_table_observation") is not table_owner
            or not same_state(slot.get("source_table_observation"), source_table)
        ):
            raise ValueError("memory probe source-table observation changed")
        if return_to_a and (
            return_signature is None
            or not shared._check_signature(return_training, return_signature)
        ):
            raise ValueError("original A-return update templates changed")
        if (bound_method or training_readouts) and arm["source"] is not source_owner:
            raise ValueError("bound probe original source owner changed")
        if training_readouts and (
            readout_signature is None
            or not shared._check_signature(training_readouts, readout_signature)
        ):
            raise ValueError("original old A readout packets changed")
        if strict_probe and (
            signatures is None
            or not shared._check_signature((*data[0], *data[1]), signatures)
            or (
                paired
                and (
                    transport_signature is None
                    or not shared._check_signature(
                        (*probe_data[0], *probe_data[1]), transport_signature
                    )
                )
            )
            or (
                paired_scope is not None
                and not same_state(slot.get("paired_probe_scope"), paired_scope)
            )
            or type(slot.get("checkpoint_updates")) is not int
            or slot["checkpoint_updates"] != paired_point
            or (
                paired_labels is not None
                and not same_state(
                    (
                        slot.get("native_updates_before"),
                        slot.get("native_training_examples_before"),
                    ),
                    paired_labels,
                )
            )
        ):
            raise ValueError("paired original/transport packets or declared scope changed")
        if bound_method:
            bind_probe_snapshot(state, paired_point)
            if (
                ("method" in slot) != slot_method_present
                or slot.get("method", method) != method
                or arm.get("row", {}).get("method", method) != method
            ):
                raise ValueError("bound probe method identity changed")
        if not same_state(frozen_state, state):
            raise ValueError("probe changed its original checkpoint")

    def source_snapshot():
        value = arm["source"].snapshot_state()
        return bind_probe_snapshot(value, 0) if bound_method else value

    def guard(*, include_boundary: bool = True) -> None:
        if include_boundary:
            check_boundary()
        pure_guard()
        if source_before is not None:
            observed_source = source_snapshot()
            if not same_state(source_before, observed_source):
                raise ValueError("probe changed its source learner")
        if (
            (paired or probe_inference is not None)
            and probe_source is not None
            and not same_state(frozen_state, probe_source.snapshot_state())
        ):
            raise ValueError("paired probe changed its disposable-fork source template")
        if strict_probe:
            # The final source/template snapshot is also an external callback.
            # Reprove the original packets, scope, state and charges afterwards.
            if bound_method and include_boundary:
                check_boundary()
            pure_guard()

    try:
        guard()
        if not paired and not bound_method:
            shared._counter(state, slot["checkpoint_updates"])
        _validate_batches(*probe_data)
        expected_data_bytes = sum(
            batch.features.nbytes + batch.targets.nbytes
            for batches in probe_data
            for batch in batches
        )
        if expected_data_bytes not in (193104, 384912):
            raise ValueError("probe requires the fixed original or public-history input envelope")
        slot.update(
            native_updates_before=state.completed_updates,
            native_training_examples_before=state.training_examples,
        )
        if strict_probe:
            paired_labels = (state.completed_updates, state.training_examples)
        source_before = deepcopy(source_snapshot())
        if bound_method:
            source_before = bind_probe_snapshot(source_before, 0)
            if not same_state(source_before.policy, frozen_state.policy):
                raise ValueError("bound probe source and checkpoint policies differ")
        guard()
        if probe_inference is not None:
            probe_source = bounded_probe.build_source(
                frozen_state, probe_data, return_training, probe_inference
            )
            guard()
        if paired:
            # Why: the stream's original source is frozen and its trained
            # candidate checkpoints intentionally have different native weights.
            shared._paired_counter(source_before, source_before.completed_updates)
            probe_source = TorchPairedProbeLearner(state, schedule)
            paired_scope = dict(
                protocol_id=(
                    "minigrid_original_f_two_episode_probe_v1"
                    if _probe_pair(data) == ("F-blue", "F-red")
                    else "minigrid_original_e_two_episode_probe_v1"
                ),
                checkpoint_updates=list(points),
                probe_updates=8,
                probe_episode_exposures=16,
                probe_training_rows=144,
                training_episode_ids=[episode.episode_id for episode in schedule],
                evaluation_episode_ids=[shared._episode(packet).episode_id for packet in data[1]],
                native_training_episodes_before=state.training_episodes,
                transport_context="first episode context labels transport only; original schedule contains both E contexts",
                storage="193104 detached input-array bytes; original+transport+app copies, complete schedule, model/optimizer/reports add time/RSS",
            )
            slot["paired_probe_scope"] = deepcopy(paired_scope)
            guard()
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=updates,
                max_wall_seconds=max(0.0, deadline - monotonic()),
                max_process_rss_bytes=RSS_LIMIT,
            ),
            monotonic,
        )
        budget.attach_memory(sampler)
        binding = _budget_binding(budget)
        last_clock = budget.last_clock
        return_options: _ReturnAppOptions = {}
        if extra_f_exposure:
            return_options["extra_f_exposure"] = True
        if old_a_rehearsal:
            replay_budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_replay_examples=288,
                    max_wall_seconds=max(0.0, deadline - monotonic()),
                    max_process_rss_bytes=RSS_LIMIT,
                ),
                monotonic,
            )
            replay_binding = _budget_binding(replay_budget)
            replay_budget.attach_memory(sampler)
            # Memory enrollment adds the declared original sampler. Pin that
            # reference while keeping every other original limit/clock binding.
            if _budget_binding(replay_budget)[:5] != replay_binding[:5]:
                raise ValueError("old-A replay memory enrollment changed original budget")
            replay_binding = _budget_binding(replay_budget)
            replay_last_clock = replay_budget.last_clock

            def replay_factory(learner, wake_budget, count):
                nonlocal replay_plan
                own_budget_guard()
                assert replay_budget is not None
                replay_plan = build_minigrid_old_a_rehearsal(
                    learner,
                    wake_budget,
                    tuple(shared._episode(packet) for packet in training_readouts),
                    replay_budget=replay_budget,
                    total_updates=count,
                    wake_role="probe_train",
                    **({"wide_neural": True} if wide_neural else {}),
                )
                own_budget_guard()
                return replay_plan.rehearsal

            return_options["rehearsal_factory"] = replay_factory
        if training_readouts:
            return_options["training_readouts"] = training_readouts
        if return_to_a:
            return_budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=16,
                    max_wall_seconds=max(0.0, deadline - monotonic()),
                    max_process_rss_bytes=RSS_LIMIT,
                ),
                monotonic,
            )
            return_budget.attach_memory(sampler)
            return_binding = _budget_binding(return_budget)
            return_last_clock = return_budget.last_clock
            return_options["return_training"] = return_training
            return_options["return_budget"] = return_budget
        if live is not None:
            return_options["endpoint_observer"] = live.observe_qualified
        if deferred is not None:
            return_options["endpoint_observer"] = deferred.observer(slot["checkpoint_updates"])
        guard()
        slot["report"] = run_fresh_task_probe(
            probe_source if probe_source is not None else arm["source"],
            state,
            *probe_data,
            budget=budget,
            checkpoint_updates=points,
            timer=monotonic,
            decision_diagnostics=True,
            check_boundary=guard,
            **return_options,
        )
        frozen_result = pickle.dumps(slot["report"], protocol=5)
        guard()
        result = slot["report"]
        if old_a_rehearsal:
            rehearsal_report = result.get("rehearsal")
            if (
                replay_budget is None
                or replay_plan is None
                or type(rehearsal_report) is not dict
                or rehearsal_report != replay_plan.rehearsal.report
                or rehearsal_report.get("wake_role") != "probe_train"
                or any(
                    type(rehearsal_report.get(key)) is not int or rehearsal_report[key] != value
                    for key, value in dict(
                        charged_exposures=288,
                        attempted_updates=32,
                        returned_updates=32,
                        committed_updates=32,
                    ).items()
                )
                or replay_budget.replay_examples_completed != 288
            ):
                raise ValueError("old-A replay differs from the frozen complete F16 recipe")
        if (
            result["status"] != "complete"
            or result["completed_updates"] != updates
            or result["completed_training_examples"] != (432 if extra_f_exposure else 144)
            or result["score_forward_calls_completed"] != 8
            or result["reserved_data_bytes"] != expected_data_bytes
            or result["num_classes"] != 7
            or result.get("decision_diagnostics_enabled") is not True
            or [point["updates"] for point in result["checkpoints"]] != list(points)
            or budget.updates_completed != result["completed_updates"]
        ):
            raise ValueError(
                "probe failed or differs from the frozen complete eight-pair recipe"
                if paired
                else "probe failed or differs from the frozen complete forty-eight-update recipe"
                if extra_f_exposure
                else "probe failed or differs from the frozen complete sixteen-update recipe"
            )
        outcomes = _probe_outcomes(result)
        if training_readouts:
            monitored = result.get("training_readouts")
            expected_keys = {shared._episode(packet).episode_id for packet in training_readouts}
            if (
                type(monitored) is not dict
                or monitored.get("status") != "complete"
                or any(
                    type(monitored.get(key)) is not int or monitored[key] != expected
                    for key, expected in (
                        ("score_forward_calls_completed", 8),
                        ("score_forward_calls_attempted", 8),
                        ("attempted_rows", 72),
                        ("completed_rows", 72),
                        (
                            "reserved_data_bytes",
                            sum(p.features.nbytes + p.targets.nbytes for p in training_readouts),
                        ),
                    )
                )
                or any(type(point["updates"]) is not int for point in monitored["checkpoints"])
                or [point["updates"] for point in monitored["checkpoints"]] != list(points)
                or any(
                    point.get("status") != "complete"
                    or set(point["scores"]) != expected_keys
                    or set(point.get("decision_diagnostics", {})) != expected_keys
                    for point in monitored["checkpoints"]
                )
            ):
                raise ValueError("old A readouts differ from the fixed complete four-point recipe")
            old_readout_outcomes = _probe_outcomes(monitored)
        if return_to_a:
            returned_phase = result.get("return_phase")
            return_fit.validate_return_report(
                returned_phase,
                input_bytes=sum(p.features.nbytes + p.targets.nbytes for p in return_training),
                monitor_keys={shared._episode(p).episode_id for p in training_readouts},
                prior_report=result,
            )
            if (
                return_budget is None
                or return_budget.updates_completed != returned_phase["completed_updates"]
            ):
                raise ValueError("A-return report and permanent charge differ")
            return_outcomes = dict(
                F=_probe_outcomes(returned_phase),
                A=_probe_outcomes(returned_phase["training_readouts"]),
            )
        guard()
    except Exception as error:
        problem = error
        slot.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        return_binding_validated = False
        readback: dict[str, Any] = dict(
            total_seconds=None,
            budget_binding_preserved=None,
            source_unchanged=None,
            state_unchanged=None,
        )
        try:
            own_budget_guard()
            result = slot["report"]
            readback["budget_binding_preserved"] = budget is not None and (
                result is None or budget.updates_completed == result["completed_updates"]
            )
            readback["source_unchanged"] = source_before is not None and same_state(
                source_before, source_snapshot()
            )
            readback["state_unchanged"] = same_state(frozen_state, state)
            if not all(
                readback[key] is True
                for key in ("budget_binding_preserved", "source_unchanged", "state_unchanged")
            ):
                raise ValueError("probe isolation or original charge binding failed")
            readback["total_seconds"] = monotonic() - started
            guard()
            if (
                frozen_result is not None
                and pickle.dumps(slot["report"], protocol=5) != frozen_result
            ):
                raise ValueError("returned probe report changed before publication")
            own_budget_guard()
            return_binding_validated = True
        except Exception as error:
            slot["readback_error"] = f"{type(error).__name__}: {error}"
            slot.update(status="failed")
            if problem is None:
                problem = error
                slot["reason"] = slot["readback_error"]
        # Canonical charges and flags are published after the final callback,
        # alongside outcomes below; a callback cannot forge these readbacks.
        readback.update(
            charged_updates=None if budget is None else budget.updates_completed,
            work_unknown=slot["report"] is None and budget is not None,
        )
        if return_to_a:
            returned_phase = (slot["report"] or {}).get("return_phase")
            readback.update(
                return_charged_updates=None
                if return_budget is None
                else return_budget.updates_completed,
                return_budget_binding_preserved=return_binding_validated
                and return_budget is not None
                and _budget_binding(return_budget) == return_binding
                and (
                    returned_phase is None
                    or return_budget.updates_completed == returned_phase["completed_updates"]
                ),
                return_work_unknown=return_budget is not None and returned_phase is None,
            )
        if frozen_result is not None:
            # Preserve the paid prefix after refusal; never seal a callback's
            # changed public result as the original completed probe evidence.
            slot["report"] = pickle.loads(frozen_result)
        if paired_scope is not None:
            slot["paired_probe_scope"] = deepcopy(paired_scope)
        slot.update(readback)
        if old_a_rehearsal:
            replay_work["charged_replay_examples"] = (
                0 if replay_budget is None else replay_budget.replay_examples_completed
            )
            replay_work["budget_binding_preserved"] = replay_budget is not None and (
                _budget_binding(replay_budget) == replay_binding
            )
            if replay_plan is not None:
                replay_work.update(
                    bank_encoded_bytes=87280,
                    validation_projection_rows=replay_plan.validation_projection_rows,
                    validation_projected_array_equivalent_bytes=replay_plan.validation_projected_array_equivalent_bytes,
                )
                try:
                    replay_report = replay_plan.rehearsal.report
                    replay_work.update(
                        {
                            key: replay_report[key]
                            for key in (
                                "attempted_updates",
                                "returned_updates",
                                "committed_updates",
                            )
                        }
                    )
                    replay_work["returned_training_rows"] = replay_report["returned_updates"] * 9
                except Exception:
                    for key in (
                        "attempted_updates",
                        "returned_updates",
                        "committed_updates",
                        "returned_training_rows",
                    ):
                        replay_work[key] = None
                    replay_work["work_unknown"] = True
            replay_work["work_unknown"] |= replay_plan is None and replay_budget is not None
            replay_work["work_unknown"] |= problem is not None
            slot["rehearsal_work"] = deepcopy(replay_work)
        # Publish the original initial exposure law with canonical paid charges;
        # retain any refusal/reason instead of publishing callback-written fields.
        slot.update(catalogue_metadata)
        if bound_method:
            returned = slot["report"]
            training_rows: Any = (
                None if returned is None else returned["completed_training_examples"]
            )
            score_rows: Any = (
                None if returned is None else 9 * returned["score_forward_calls_completed"]
            )
            monitor_rows = (
                0
                if returned is None
                else returned.get("training_readouts", {}).get("completed_rows", 0)
            )
            if score_rows is not None:
                score_rows += monitor_rows
            return_rows: int | None = 0
            if return_to_a:
                phase = (returned or {}).get("return_phase")
                return_rows = (
                    None
                    if phase is None
                    else phase["completed_training_examples"]
                    + 9 * phase["score_forward_calls_completed"]
                    + phase.get("training_readouts", {}).get("completed_rows", 0)
                )
            slot["projection_accounting"] = dict(
                training_rows_returned=training_rows,
                score_rows_returned=score_rows,
                array_bytes_returned=None
                if returned is None or return_rows is None
                else 2368 * (training_rows + score_rows + return_rows),
                scope="additional296-float64 projected rows for returned committed training and scores; validation/copies/Python/native/failed-attempt allocations unknown",
            )
            if old_a_rehearsal:
                accounting = slot["projection_accounting"]
                accounting["replay_rows_returned"] = replay_work["returned_training_rows"]
                if replay_work["returned_training_rows"] is None:
                    accounting["array_bytes_returned"] = None
                elif accounting["array_bytes_returned"] is not None:
                    accounting["array_bytes_returned"] += (
                        2368 * replay_work["returned_training_rows"]
                    )
                accounting["replay_validation_projection_rows"] = replay_work[
                    "validation_projection_rows"
                ]
            if training_readouts:
                slot["projection_accounting"]["training_readout_score_rows_returned"] = monitor_rows
            if return_to_a:
                slot["projection_accounting"]["return_rows_returned"] = return_rows
        if memory_method:
            slot["source_table_observation"] = deepcopy(source_table)
        if live is not None and problem is not None:
            live.restore_paid()
            slot["probe_rollouts"] = live.records
    if problem is not None:
        raise problem
    if budget is None:
        raise ValueError("complete probe requires its original independent budget")
    slot.update(status="complete", outcomes=deepcopy(outcomes) if strict_probe else outcomes)
    if training_readouts:
        slot["old_task_readout_outcomes"] = deepcopy(old_readout_outcomes)
    if return_to_a:
        slot["return_fit_outcomes"] = deepcopy(return_outcomes)
    completed_budget = budget
    committed = completed_budget.updates_completed
    committed_return = None if return_budget is None else return_budget.updates_completed
    completed_fields: tuple[str, ...] = (
        (
            "checkpoint_updates",
            "native_updates_before",
            "native_training_examples_before",
            "status",
            "reason",
            "charged_updates",
            "budget_binding_preserved",
            "source_unchanged",
            "state_unchanged",
            "work_unknown",
            "total_seconds",
        )
        if strict_probe
        else ()
    )
    completed_scalars = tuple(slot[key] for key in completed_fields)
    if catalogue_rows is not None:
        completed_fields += ("catalogue_rows_per_update", "expected_training_examples")
        completed_scalars = tuple(slot[key] for key in completed_fields)
    if return_to_a:
        completed_fields += (
            "return_charged_updates",
            "return_budget_binding_preserved",
            "return_work_unknown",
        )
        completed_scalars = tuple(slot[key] for key in completed_fields)
    completed_replay_work = deepcopy(slot.get("rehearsal_work")) if old_a_rehearsal else None
    committed_replay = None if replay_budget is None else replay_budget.replay_examples_completed
    completed_projection = deepcopy(slot.get("projection_accounting")) if bound_method else None
    bound_owners = (
        {key: slot[key] for key in ("report", "outcomes")}
        if bound_method or training_readouts
        else {}
    )
    if bound_method:
        bound_owners["projection_accounting"] = slot["projection_accounting"]
    if memory_method:
        table_owner = slot["source_table_observation"]
        bound_owners["source_table_observation"] = table_owner
    if training_readouts:
        bound_owners["old_task_readout_outcomes"] = slot["old_task_readout_outcomes"]
    if return_to_a:
        bound_owners["return_fit_outcomes"] = slot["return_fit_outcomes"]
    if old_a_rehearsal:
        bound_owners["rehearsal_work"] = slot["rehearsal_work"]
    return_owners = (
        ()
        if not return_to_a
        else (
            slot["report"]["return_phase"],
            slot["report"]["return_phase"]["training_readouts"],
            slot["return_fit_outcomes"]["F"],
            slot["return_fit_outcomes"]["A"],
        )
    )

    def completed_guard() -> None:
        own_budget_guard()
        if completed_budget.updates_completed != committed:
            raise ValueError("preceding probe permanent charges changed")
        if return_budget is not None and return_budget.updates_completed != committed_return:
            raise ValueError("preceding A-return permanent charges changed")
        if old_a_rehearsal and (
            replay_budget is None
            or replay_budget.replay_examples_completed != committed_replay
            or not same_state(slot.get("rehearsal_work"), completed_replay_work)
        ):
            raise ValueError("preceding old-A replay charges or result changed")
        if strict_probe:
            # The parent already checks every completed arm/probe in a flat loop.
            # Calling it again here would recurse through this published guard.
            guard(include_boundary=not (bound_method or training_readouts))
            if (
                any(key not in slot for key in completed_fields)
                or any(slot.get(key) is not owner for key, owner in bound_owners.items())
                or not same_state(
                    tuple(slot.get(key) for key in completed_fields), completed_scalars
                )
                or not same_state(slot.get("outcomes"), outcomes)
                or pickle.dumps(slot.get("report"), protocol=5) != frozen_result
                or (
                    return_to_a and not same_state(slot.get("return_fit_outcomes"), return_outcomes)
                )
                or (
                    return_to_a
                    and any(
                        actual is not original
                        for actual, original in zip(
                            (
                                slot.get("report", {}).get("return_phase"),
                                slot.get("report", {})
                                .get("return_phase", {})
                                .get("training_readouts"),
                                slot.get("return_fit_outcomes", {}).get("F"),
                                slot.get("return_fit_outcomes", {}).get("A"),
                            ),
                            return_owners,
                            strict=True,
                        )
                    )
                )
                or (
                    training_readouts
                    and not same_state(slot.get("old_task_readout_outcomes"), old_readout_outcomes)
                )
                or (
                    bound_method
                    and not same_state(slot.get("projection_accounting"), completed_projection)
                )
            ):
                raise ValueError("preceding paired probe result changed")

    if deferred is not None:
        completed_guard()
        if return_budget is None:
            raise ValueError("deferred readout lost its paid return budget")
        deferred.bind_completed(slot["checkpoint_updates"], budget, return_budget)
    return completed_guard


def _run_deferred_phase(deferred, arms, *, check_phase, resources) -> dict[str, Any]:
    """Enclose readonly games after training; retain work before closing refusal."""
    record: dict[str, Any] = dict(
        status="running", reason=None, entry_validated=False, exit_validated=False, report=None
    )
    witnesses: tuple[_BudgetWitness, ...] = ()

    def proof() -> None:
        deferred.check()
        for witness in witnesses:
            witness.check()

    def close_port(port: Callable[[], None]) -> None:
        try:
            port()
        except Exception:
            try:
                proof()
            except Exception:
                # Keep the first port error; the closing proof still ran and
                # no later read or game can be admitted after either failure.
                pass
            raise
        else:
            proof()

    def boundary(
        read_ports: Callable[[], None] | None = None,
        *,
        after_clocks: Callable[[], None] | None = None,
    ) -> None:
        if after_clocks is not None and (not callable(after_clocks) or read_ports is None):
            raise ValueError("deferred final reads require their original read group")
        proof()
        close_port(resources)
        if read_ports is not None:
            # No external call intervenes: the resource closing proof also
            # opens the supplied read group. Never cache it across a port.
            first_error = None
            try:
                result = read_ports()
                if after_clocks is not None and result is not None:
                    raise ValueError("deferred read ports must return None")
            except Exception as error:
                first_error = error
            try:
                # Preserve the original closing wall/RSS observation. Refuse
                # changed evidence before admitting that external resource port.
                proof()
                close_port(resources)
            except Exception:
                if first_error is None:
                    raise
            if first_error is not None:
                raise first_error
            if after_clocks is not None:
                # Why: the last resource clock may mutate the captured scorer.
                # Reprove native state after it, then close without another clock.
                def final_reads() -> None:
                    if after_clocks() is not None:
                        raise ValueError("deferred final read ports must return None")

                close_port(final_reads)

    try:
        if len(arms) != 1:
            raise ValueError("deferred games require a singleton admitted control")
        originals = tuple(
            (arm["budget"], _endpoint_budget_binding(arm["budget"]), deepcopy(arm["budget"].budget))
            for arm in arms
        )
        # Why: the trusted completed-arm proof observes its original clock/RSS.
        # Keep authority from before that port; freeze only its proved readings.
        check_phase()
        if any(
            arm["budget"] is not original[0] for arm, original in zip(arms, originals, strict=True)
        ):
            raise ValueError("phase entry replaced the original benchmark budget")
        witnesses = tuple(
            _BudgetWitness(budget, binding, limits, 128) for budget, binding, limits in originals
        )
        boundary()
        record["entry_validated"] = True
        returned = deferred.run_qualified(check_boundary=boundary)
        if returned is not None:
            record["report"] = pickle.loads(pickle.dumps(returned, protocol=5))
    except Exception as error:
        record.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        # Why: completed game work stays paid even when phase-exit proof refuses.
        try:
            if record["report"] is None:
                record["report"] = pickle.loads(pickle.dumps(deferred.report(), protocol=5))
        except Exception as error:
            record["status"] = "failed"
            record["reason"] = record["reason"] or f"capture {type(error).__name__}: {error}"
        finally:
            try:
                try:
                    boundary()
                finally:
                    check_phase()
                record["exit_validated"] = True
            except Exception as error:
                record["status"] = "failed"
                record["reason"] = record["reason"] or f"phase exit {type(error).__name__}: {error}"
    if record["status"] == "running":
        completed = record["report"]
        if type(completed) is not dict or completed.get("status") not in ("complete", "failed"):
            record.update(
                status="failed", reason="deferred runner returned an invalid phase report"
            )
        else:
            record.update(
                status="complete" if completed["status"] == "complete" else "failed",
                reason=completed.get("reason"),
            )
    return record


def _new_row(method: str) -> dict[str, Any]:
    row = shared._row(method)
    row["fit_scope"] = "teacher_forced_train_fit_not_live_success"
    row["probes"] = [
        dict(
            checkpoint_updates=point,
            status="not_run",
            reason="not reached",
            report=None,
            charged_updates=None,
            source_unchanged=None,
            state_unchanged=None,
            budget_binding_preserved=None,
            outcomes=None,
            work_unknown=None,
            total_seconds=None,
        )
        for point in (0, 128)
    ]
    return row


def _fit_points(readouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for readout in readouts:
        point: dict[str, Any] = dict(updates=readout["updates"])
        for context in ("A", "B", "C_train"):
            scores = [score for score in readout["scores"] if score["context"] == context]
            if len(scores) != 2:
                raise ValueError("both fixed cue scores are required")
            point[context] = {
                metric: sum(score[metric] for score in scores) / 2
                for metric in ("ce", "accuracy", "exact_sequence_accuracy")
            }
        points.append(point)
    return points


def _a_only_fit_outcomes(readouts: list[dict[str, Any]]) -> dict[str, Any]:
    points = _fit_points(readouts)
    if (
        any(type(point["updates"]) is not int for point in points)
        or tuple(point["updates"] for point in points) != shared.POINTS
    ):
        raise ValueError("A-only acquisition requires all five original checkpoint readouts")
    gains = {
        metric: (-1 if metric == "ce" else 1) * (points[-1]["A"][metric] - points[0]["A"][metric])
        for metric in ("ce", "accuracy", "exact_sequence_accuracy")
    }
    return dict(
        outcomes=points,
        A_fit_gains=gains,
        A_acquisition={
            "A_at32": points[1]["A"]["exact_sequence_accuracy"] == 1,
            "A_at128": points[-1]["A"]["exact_sequence_accuracy"] == 1,
        },
        interpretation="single_task_A_fit_only_no_retention_or_generalization_claim",
        status="complete",
    )


def _fit_outcomes(
    readouts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, bool]]:
    points = _fit_points(readouts)
    differences = dict(
        initial_A_gain=("A", 0, 1),
        new_B_gain=("B", 1, 2),
        new_C_gain=("C_train", 2, 3),
        old_A_loss=("A", 3, 1),
        return_A_gain=("A", 3, 4),
        return_gap=("A", 4, 1),
    )
    metrics = {
        metric: {
            name: (-1 if metric == "ce" else 1)
            * (points[right][context][metric] - points[left][context][metric])
            for name, (context, left, right) in differences.items()
        }
        for metric in ("ce", "accuracy", "exact_sequence_accuracy")
    }
    acquired = {
        context: points[index][context]["exact_sequence_accuracy"] == 1
        for context, index in (("A", 1), ("B", 2), ("C_train", 3))
    }
    return points, metrics, acquired


def _work(
    rows: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    *,
    activation_health: bool = False,
    gradient_geometry: bool = False,
    prior_gradient_geometry: bool = False,
    pc_update_diagnostics: bool = False,
    weighted_catalogue: bool = False,
    train_action_margins: bool = False,
    old_task_readouts: bool = False,
    return_to_a: bool = False,
    old_a_rehearsal: bool = False,
) -> dict[str, Any]:
    raw = [slot["work"] for row in rows for slot in row["readouts"] if "work" in slot]
    probes = [
        slot["report"] for row in rows for slot in row["probes"] if slot["report"] is not None
    ]
    benchmark_calls = sum(row["charged_updates"] or 0 for row in rows)
    benchmark_rows = sum(row["native_training_examples"] or 0 for row in rows)
    probe_calls = sum(slot["charged_updates"] or 0 for row in rows for slot in row["probes"])
    probe_rows = sum(probe["completed_training_examples"] for probe in probes)
    recurrent = [row for row in rows if row["method"] == "recurrent_bc"]
    recurrent_rows = sum(row["native_training_examples"] or 0 for row in recurrent)
    recurrent_raw = sum(
        slot["work"]["score_rows_completed"]
        for row in recurrent
        for slot in row["readouts"]
        if "work" in slot
    )
    recurrent_probes = [
        slot["report"] for row in recurrent for slot in row["probes"] if slot["report"] is not None
    ]
    result = dict(
        benchmark_native_train_calls=benchmark_calls,
        benchmark_training_rows=benchmark_rows,
        probe_native_train_calls=probe_calls,
        probe_training_rows=probe_rows,
        native_train_calls=benchmark_calls + probe_calls,
        training_rows=benchmark_rows + probe_rows,
        known_raw_score_forwards_attempted=sum(v["score_forward_calls_attempted"] for v in raw),
        known_raw_score_forwards_completed=sum(v["score_forward_calls_completed"] for v in raw),
        known_raw_score_rows_completed=sum(v["score_rows_completed"] for v in raw),
        probe_score_forwards_attempted=sum(v["score_forward_calls_attempted"] for v in probes),
        probe_score_forwards_completed=sum(v["score_forward_calls_completed"] for v in probes),
        probe_score_rows_completed=9 * sum(v["score_forward_calls_completed"] for v in probes),
        teacher_episodes_observed=len(reports),
        source_environment_steps_attempted=sum(
            v.get("environment_steps_attempted") or 0 for v in reports
        ),
        source_environment_steps_returned=sum(v.get("environment_steps") or 0 for v in reports),
        source_environment_counts_are_lower_bounds=any(
            v.get("environment_steps") is None or v.get("environment_steps_attempted") is None
            for v in reports
        ),
        recurrent_unroll_steps=recurrent_rows
        + recurrent_raw
        + sum(
            probe["completed_training_examples"] + 9 * probe["score_forward_calls_completed"]
            for probe in recurrent_probes
        ),
        learner_live_episodes=0,
        final_episodes=0,
        replay_rows=0,
        failure_work_may_be_unobserved=any(
            v.get("failure_work_may_be_unobserved", False) for v in reports
        )
        or any(v["failure_work_may_be_unobserved"] for v in raw)
        or any(row["wake_calls_attempted"] > row["wake_calls_returned"] for row in rows)
        or any(
            probe["attempted_updates"] > probe["completed_updates"]
            or probe["score_forward_calls_attempted"] > probe["score_forward_calls_completed"]
            for probe in probes
        )
        or any(slot["work_unknown"] is True for row in rows for slot in row["probes"]),
    )
    if activation_health:
        result["activation_health"] = {
            key: sum(value.get(key, 0) for value in raw)
            for key in (
                "activation_calls_attempted",
                "activation_calls_completed",
                "activation_rows_attempted",
                "activation_rows_completed",
                "activation_array_bytes",
                "activation_svd_calls_attempted",
                "activation_svd_calls_completed",
                "activation_seconds",
                "activation_feedforward_hidden_affines_completed",
                "activation_recurrent_hidden_steps_completed",
                "activation_recurrent_output_projections_completed",
            )
        }
        if any("activation_projection_rows_returned" in value for value in raw):
            result["activation_health"].update(
                {
                    key: sum(value.get(key, 0) for value in raw)
                    for key in (
                        "activation_projection_rows_returned",
                        "activation_projection_array_bytes_returned",
                    )
                }
            )
        result["activation_health"]["scope"] = (
            "extra diagnostic work, excluded from score forward counts above"
        )
    if gradient_geometry:
        result["cue_gradient_geometry"] = gradient_readouts.sum_work(
            [value["cue_gradient_geometry"] for value in raw if "cue_gradient_geometry" in value]
        )
        result["failure_work_may_be_unobserved"] |= result["cue_gradient_geometry"][
            "failure_work_may_be_unobserved"
        ]
        result["recurrent_unroll_steps"] += result["cue_gradient_geometry"][
            "forward_rows_completed"
        ] + result.get("activation_health", {}).get(
            "activation_recurrent_hidden_steps_completed", 0
        )
    if prior_gradient_geometry:
        result["prior_gradient_geometry"] = gradient_readouts.sum_prior_work(
            [
                value["prior_gradient_geometry"]
                for value in raw
                if "prior_gradient_geometry" in value
            ]
        )
        result["failure_work_may_be_unobserved"] |= result["prior_gradient_geometry"][
            "failure_work_may_be_unobserved"
        ]
    if train_action_margins:
        result["train_action_margins"] = action_margins.total_work(raw)
    if pc_update_diagnostics:
        result["pc_update_diagnostics"] = gradient_readouts.sum_pc_work(
            [value["pc_update_diagnostics"] for value in raw if "pc_update_diagnostics" in value],
            **({"weighted": True} if weighted_catalogue else {}),
        )
        result["failure_work_may_be_unobserved"] |= result["pc_update_diagnostics"][
            "failure_work_may_be_unobserved"
        ]
    bound_rows = [
        row for row in rows if row["method"] in BOUND_METHODS or row["method"] == memory.METHOD
    ]
    if bound_rows:
        if activation_health:
            result["recurrent_unroll_steps"] += result["activation_health"][
                "activation_recurrent_hidden_steps_completed"
            ]
        training_rows = [row["native_training_examples"] for row in bound_rows]
        bound_raw = [
            slot["work"] for row in bound_rows for slot in row["readouts"] if "work" in slot
        ]
        bound_probes = [
            slot["projection_accounting"]
            for row in bound_rows
            for slot in row["probes"]
            if "projection_accounting" in slot
        ]
        result["invariant_projection"] = dict(
            benchmark_training_rows_returned=None
            if any(value is None for value in training_rows)
            else sum(training_rows),
            benchmark_training_array_bytes_returned=None
            if any(value is None for value in training_rows)
            else 2368 * sum(training_rows),
            fit_rows_returned=sum(
                value.get("internal_projection_rows_returned", 0) for value in bound_raw
            ),
            fit_array_bytes_returned=sum(
                value.get("internal_projection_array_bytes_returned", 0) for value in bound_raw
            ),
            probe_array_bytes_returned=None
            if any(value["array_bytes_returned"] is None for value in bound_probes)
            else sum(value["array_bytes_returned"] for value in bound_probes),
            scope="recorded returned projected rows only; unreached work excluded, missing delegated work unknown; validation/copies/Python/native/failed-attempt allocations additional or unknown",
        )
    monitored = [probe["training_readouts"] for probe in probes if "training_readouts" in probe]
    if old_task_readouts or monitored:
        result["training_readouts"] = {
            key: sum(value.get(key) or 0 for value in monitored)
            for key in (
                "score_forward_calls_attempted",
                "score_forward_calls_completed",
                "attempted_rows",
                "completed_rows",
                "reserved_data_bytes",
                "elapsed_seconds",
            )
        }
        result["training_readouts"].update(
            failure_work_may_be_unobserved=any(
                value["failure_work_may_be_unobserved"] for value in monitored
            ),
            storage_may_be_unobserved=any(
                value["reserved_data_bytes"] is None for value in monitored
            ),
            scope="additional readonly TRAIN predictions on disposable F learner; counts separate from F probe_test; elapsed covers completed measured intervals, not full monitor wall time; copies/Python/state/RSS additional",
        )
        result["total_score_forwards_completed"] = (
            result["known_raw_score_forwards_completed"]
            + result["probe_score_forwards_completed"]
            + result["training_readouts"]["score_forward_calls_completed"]
        )
        result["total_score_rows_completed"] = (
            result["known_raw_score_rows_completed"]
            + result["probe_score_rows_completed"]
            + result["training_readouts"]["completed_rows"]
        )
        result["recurrent_unroll_steps"] += sum(
            probe.get("training_readouts", {}).get("completed_rows", 0)
            for probe in recurrent_probes
        )
        result["failure_work_may_be_unobserved"] |= result["training_readouts"][
            "failure_work_may_be_unobserved"
        ]
    phases = [p["return_phase"] for p in probes if "return_phase" in p]
    if return_to_a or phases:
        work = return_fit.sum_return_work(phases)
        work["native_train_calls"] = sum(
            slot.get("return_charged_updates") or 0 for row in rows for slot in row["probes"]
        )
        work["failure_work_may_be_unobserved"] |= any(
            slot.get("return_work_unknown") is True for row in rows for slot in row["probes"]
        )
        result["return_phase"] = work
        result["native_train_calls"] += work["native_train_calls"]
        result["training_rows"] += work["training_rows"]
        result["total_score_forwards_completed"] += work["score_forwards_completed"]
        result["total_score_rows_completed"] += work["score_rows_completed"]
        result["failure_work_may_be_unobserved"] |= work["failure_work_may_be_unobserved"]
        result["recurrent_unroll_steps"] += sum(
            p["return_phase"]["completed_training_examples"]
            + 9 * p["return_phase"]["score_forward_calls_completed"]
            + p["return_phase"].get("training_readouts", {}).get("completed_rows", 0)
            for p in recurrent_probes
            if "return_phase" in p
        )
    memory_rows = [row for row in rows if row["method"] == memory.METHOD]
    if memory_rows:
        memory_probes = [slot for row in memory_rows for slot in row["probes"]]
        missing_count_work = any(
            row["charged_updates"] is None and row["wake_calls_attempted"] > 0
            for row in memory_rows
        ) or any(
            slot.get("work_unknown") is True or slot.get("return_work_unknown") is True
            for slot in memory_probes
        )
        missing_count_rows = missing_count_work or any(
            row["native_training_examples"] is None and (row["charged_updates"] or 0) > 0
            for row in memory_rows
        )
        result.update(
            count_update_calls=None
            if missing_count_work
            else sum(row["charged_updates"] or 0 for row in memory_rows)
            + sum(
                (slot.get("charged_updates") or 0) + (slot.get("return_charged_updates") or 0)
                for slot in memory_probes
            ),
            count_training_rows=None
            if missing_count_rows
            else sum(row["native_training_examples"] or 0 for row in memory_rows)
            + sum(
                (slot.get("report") or {}).get("completed_training_examples", 0)
                + ((slot.get("report") or {}).get("return_phase") or {}).get(
                    "completed_training_examples", 0
                )
                for slot in memory_probes
            ),
            neural_sgd_updates=0,
            count_work_may_be_unobserved=missing_count_work or missing_count_rows,
            count_work_scope="native_train_calls count backend calls, not SGD; count rows are observed committed rows, absent delegated work remains unknown; fixed table observations are checkpoint/source-only, disposable endpoints unavailable",
        )
    relaxation.add_work(result, rows)
    if old_a_rehearsal:
        extra = probe_rehearsal.aggregate(rows)
        result["old_a_rehearsal"] = extra
        for key in ("native_train_calls", "training_rows"):
            result[key] = None if extra[key] is None else result[key] + extra[key]
        result["replay_rows"] = extra["replay_rows"]
        if extra["failure_work_may_be_unobserved"]:
            result["failure_work_may_be_unobserved"] = True
    return result


def _add_wide_neural_work(result: dict[str, Any], wide_method: str) -> None:
    if type(wide_method) is not str or wide_method not in QUALIFICATION_METHODS:
        raise ValueError("wide work requires a registered H32 method or fixed64 control")
    fixed64 = wide_method in relaxation.METHODS
    inference_steps = 64 if fixed64 else 2
    # Why: charged failed calls cannot certify optimizer/latent work completed.
    unknown = result["failure_work_may_be_unobserved"]
    calls = None if unknown else result["native_train_calls"]
    count_rows = None if unknown else result["training_rows"]
    result["wide_neural"] = dict(
        method=wide_method,
        hidden_dim=32,
        parameter_elements=9735,
        parameter_array_equivalent_bytes=38940,
        native_calls_reported=result["native_train_calls"],
        training_rows_reported=result["training_rows"],
        bp_optimizer_steps_derived=calls if wide_method == "bp" else 0,
        bp_backward_calls_derived=calls if wide_method == "bp" else 0,
        pc_parameter_updates_derived=0 if wide_method == "bp" else calls,
        nominal_latent_row_steps=0
        if wide_method == "bp"
        else (None if count_rows is None else inference_steps * count_rows),
        failure_work_may_be_unobserved=unknown,
        scope="reported paid prefix including benchmark, F and A return; primitive costs derive only when work is observed, not an external operation meter; full state/copies/projection/validation/Python/RSS/time additional",
    )
    if fixed64:
        result["wide_neural"].update(
            native_method=relaxation.base_method(wide_method), inference_steps=64
        )

    if "old_a_rehearsal" in result:
        result["wide_neural"]["scope"] += (
            "; fixed old-A replay native calls and rows are included in this paid prefix"
        )


def _run_bp_checkpoint_readout(
    arm: dict[str, Any],
    sources: tuple[AttributionBatch, ...],
    *,
    sampler: ProcessRssSampler,
    deadline: float,
    check_boundary: Callable[[], None],
) -> tuple[dict[str, Any], Callable[[], None]]:
    """Observe an owned TRAIN copy; keep its allowance separate from all probes."""
    started = monotonic()
    budget: ToyBudgetSession | None = None
    binding: tuple[Any, ...] | None = None
    completed_clock: float | None = None
    value: dict[str, Any] = dict(status="incomplete", reason=None, work=None)
    try:
        check_boundary()
        point, state = arm["states"][-1]
        if type(point) is not int or point != 128 or arm["row"]["method"] != "public_invariant_bp":
            raise ValueError("BP observer requires the original joint checkpoint128")
        shared._invariant_feature_counter(state, point, "public_invariant_bp", catalogue_rows=54)
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=3,
                max_replay_examples=18,
                max_wall_seconds=max(0.0, deadline - monotonic()),
                max_process_rss_bytes=RSS_LIMIT,
            ),
            monotonic,
        )
        binding = _budget_binding(budget)
        budget.attach_memory(sampler)
        if _budget_binding(budget)[:5] != binding[:5] or budget.progress is not None:
            raise ValueError("BP observer original allowance changed during sampler enrollment")
        binding = _budget_binding(budget)
        check_boundary()
        value = diagnose_bp_updates(
            state,
            shared._episode(sources[2]),
            tuple(shared._episode(packet) for packet in sources[:2]),
            budget=budget,
            check_boundary=check_boundary,
        )
        check_boundary()
        work = value["work"]
        expected = dict(
            native_train_calls_attempted=3,
            native_train_calls_returned=3,
            native_training_rows_attempted=27,
            native_training_rows_returned=27,
            charged_updates=3,
            charged_replay_rows=18,
            published_updates=0,
            budget_binding_preserved=True,
            failure_work_may_be_unobserved=False,
        )
        if (
            value["status"] != "complete"
            or value["reason"] is not None
            or any(
                value.get(key) is not True
                for key in ("snapshot_unchanged", "inputs_unchanged", "rng_unchanged")
            )
            or any(
                type(work.get(key)) is not type(want) or work[key] != want
                for key, want in expected.items()
            )
            or [step["native_before_updates"] for step in value["steps"]] != [128, 129, 130]
            or [step["native_after_updates"] for step in value["steps"]] != [129, 130, 131]
        ):
            raise ValueError("BP observer did not complete the separate original three-call cycle")
        completed_clock = budget.last_clock
    except GradientDiagnosticFailed as error:
        value = deepcopy(error.report)
    except Exception as error:
        value = deepcopy(value)
        value.update(status="incomplete", reason=f"{type(error).__name__}: {error}")
        if value.get("work") is not None:
            value["work"]["failure_work_may_be_unobserved"] = True

    def completed_guard() -> None:
        if (
            budget is None
            or _budget_binding(budget) != binding
            or type(budget.updates_completed) is not int
            or budget.updates_completed != 3
            or type(budget.replay_examples_completed) is not int
            or budget.replay_examples_completed != 18
            or type(budget.last_clock) not in (int, float)
            or completed_clock is None
            or not isfinite(budget.last_clock)
            or budget.last_clock != completed_clock
        ):
            raise ValueError("completed BP observer allowance or charged prefix changed")

    value.update(
        checkpoint_updates=128,
        benchmark_training_examples=6912,
        elapsed_seconds=monotonic() - started,
        observed_budget_updates=None if budget is None else budget.updates_completed,
        observed_replay_rows=None if budget is None else budget.replay_examples_completed,
        budget_binding_preserved=budget is not None and _budget_binding(budget) == binding,
    )
    if value["status"] == "complete":
        try:
            completed_guard()
        except ValueError as error:
            value.update(status="incomplete", reason=str(error))
            value["work"]["failure_work_may_be_unobserved"] = True
    return value, completed_guard


def run_pilot(
    *,
    public_history: bool = False,
    activation_health: bool = False,
    gradient_geometry: bool = False,
    prior_gradient_geometry: bool = False,
    invariant_history: bool = False,
    pc_qualification: bool = False,
    qualification_method: str | None = None,
    acquisition_only: bool = False,
    distinct_probe: bool = False,
    old_task_readouts: bool = False,
    return_to_a: bool = False,
    train_a_rollout: bool = False,
    memory_control: bool = False,
    probe_rollouts: bool = False,
    deferred_probe_rollouts: bool = False,
    weighted_bp: bool = False,
    weighted_pc: bool = False,
    weighted_catalogue: bool = False,
    train_action_margins: bool = False,
    joint_catalogue: bool = False,
    pc_update_diagnostics: bool = False,
    probe_inference: str | None = None,
    train_fit_bounds: bool = False,
    cue_relative_history: bool = False,
    old_a_rehearsal: bool = False,
    bp_update_diagnostics: bool = False,
    wide_neural: bool = False,
    extended_wide_probe_budget: bool = False,
    extra_f_exposure: bool = False,
) -> dict[str, Any]:
    """Run one bounded fixed case; operational faults stop and retain the prefix."""
    _validate_extended_wide_probe_budget(
        extended_wide_probe_budget,
        wide_neural,
        probe_rollouts,
        deferred_probe_rollouts=deferred_probe_rollouts,
    )
    f_exposure.validate_option(extra_f_exposure, **locals())
    _validate_wide_neural_option(wide_neural, **locals())
    bp_update_route.validate_option(bp_update_diagnostics, **locals())
    probe_rehearsal.validate_option(old_a_rehearsal, **locals())
    _validate_cue_relative_option(
        cue_relative_history,
        joint_catalogue,
        qualification_method,
        (
            activation_health,
            gradient_geometry,
            prior_gradient_geometry,
            weighted_bp,
            weighted_pc,
            weighted_catalogue,
            train_action_margins,
            pc_update_diagnostics,
            train_a_rollout,
            probe_rollouts,
            deferred_probe_rollouts,
            memory_control,
            pc_qualification,
            acquisition_only,
            probe_inference,
        ),
    )
    validate_train_fit_bounds_option(train_fit_bounds, joint_catalogue)
    bounded_probe.validate_option(
        probe_inference,
        public_history=public_history,
        invariant_history=invariant_history,
        joint_catalogue=joint_catalogue,
        qualification_method=qualification_method,
        distinct_probe=distinct_probe,
        old_task_readouts=old_task_readouts,
        return_to_a=return_to_a,
        pc_qualification=pc_qualification,
        acquisition_only=acquisition_only,
        memory_control=memory_control,
        weighted_bp=weighted_bp,
        weighted_pc=weighted_pc,
        activation_health=activation_health,
        gradient_geometry=gradient_geometry,
        prior_gradient_geometry=prior_gradient_geometry,
        train_a_rollout=train_a_rollout,
        probe_rollouts=probe_rollouts,
        deferred_probe_rollouts=deferred_probe_rollouts,
        train_action_margins=train_action_margins,
        pc_update_diagnostics=pc_update_diagnostics,
    )
    relaxation.validate_option(
        qualification_method,
        joint_catalogue=joint_catalogue,
        train_action_margins=train_action_margins,
        pc_update_diagnostics=pc_update_diagnostics,
    )
    gradient_readouts.validate_pc_option(
        pc_update_diagnostics,
        joint_catalogue=joint_catalogue,
        qualification_method=qualification_method,
        train_action_margins=train_action_margins,
    )
    weighted_catalogue_route.validate_option(
        weighted_catalogue,
        public_history=public_history,
        invariant_history=invariant_history,
        joint_catalogue=joint_catalogue,
        qualification_method=qualification_method,
        distinct_probe=distinct_probe,
        old_task_readouts=old_task_readouts,
        return_to_a=return_to_a,
        weighted_bp=weighted_bp,
        weighted_pc=weighted_pc,
        pc_qualification=pc_qualification,
        acquisition_only=acquisition_only,
        memory_control=memory_control,
        pc_update_diagnostics=pc_update_diagnostics,
    )
    joint.validate_option(
        joint_catalogue,
        public_history=public_history,
        invariant_history=invariant_history,
        qualification_method=qualification_method,
        distinct_probe=distinct_probe,
        old_task_readouts=old_task_readouts,
        return_to_a=return_to_a,
        pc_qualification=pc_qualification,
        acquisition_only=acquisition_only,
        weighted_bp=weighted_bp,
        weighted_pc=weighted_pc,
        memory_control=memory_control,
        activation_health=activation_health,
        gradient_geometry=gradient_geometry,
        prior_gradient_geometry=prior_gradient_geometry,
        train_a_rollout=train_a_rollout,
        # Why: exact wide admission above owns optional endpoint games; this
        # legacy validator binds the unchanged joint-training recipe only.
        probe_rollouts=False if wide_neural else probe_rollouts,
        deferred_probe_rollouts=False if wide_neural else deferred_probe_rollouts,
        train_action_margins=train_action_margins,
        pc_update_diagnostics=pc_update_diagnostics,
        **({"wide_neural": True} if wide_neural else {}),
    )
    action_margins.validate_option(train_action_margins, public_history, invariant_history)
    weighted_feature.validate_option(
        weighted_bp,
        public_history=public_history,
        invariant_history=invariant_history,
        qualification_method=qualification_method,
        pc_qualification=pc_qualification,
        memory_control=memory_control,
    )
    weighted_feature.validate_pc_option(
        weighted_pc,
        public_history=public_history,
        invariant_history=invariant_history,
        qualification_method=qualification_method,
        pc_qualification=pc_qualification,
        memory_control=memory_control,
        weighted_bp=weighted_bp,
        joint_catalogue=joint_catalogue,
    )
    probe_rollout.validate_option(
        probe_rollouts, memory_control, **({"wide_neural": True} if wide_neural else {})
    )
    deferred_rollout.validate_option(
        deferred_probe_rollouts,
        memory_control,
        probe_rollouts,
        **(
            {"wide_method": qualification_method}
            if wide_neural and qualification_method not in relaxation.METHODS
            else {}
        ),
    )
    memory.validate_option(
        memory_control,
        public_history=public_history,
        invariant_history=invariant_history,
        acquisition_only=acquisition_only,
        distinct_probe=distinct_probe,
        old_task_readouts=old_task_readouts,
        return_to_a=return_to_a,
        qualification_method=qualification_method,
        pc_qualification=pc_qualification,
        activation_health=activation_health,
        gradient_geometry=gradient_geometry,
        train_a_rollout=train_a_rollout,
    )
    if type(gradient_geometry) is not bool:
        raise ValueError("gradient_geometry must be an exact bool")
    _validate_distinct_probe(distinct_probe, public_history, invariant_history)
    _validate_old_task_readouts(
        old_task_readouts, public_history, invariant_history, distinct_probe
    )
    _validate_return_to_a(return_to_a, old_task_readouts)
    train_rollout.validate_option(
        train_a_rollout, acquisition_only, joint_catalogue=joint_catalogue
    )
    if type(invariant_history) is not bool or (invariant_history and public_history is not True):
        raise ValueError("invariant_history requires an exact bool and public_history=True")
    if invariant_history and gradient_geometry is not False:
        raise ValueError("invariant history does not support gradient diagnostics")
    _validate_prior_gradient_option(
        prior_gradient_geometry,
        public_history,
        invariant_history,
        gradient_geometry,
        memory_control,
        acquisition_only,
    )
    if type(pc_qualification) is not bool or (
        pc_qualification and (public_history is not True or invariant_history is not True)
    ):
        raise ValueError("pc_qualification requires exact bools and both public/invariant history")
    _validate_qualification_method(
        qualification_method, public_history, invariant_history, pc_qualification
    )
    _validate_acquisition_only(
        acquisition_only,
        public_history,
        invariant_history,
        "bp" if memory_control else qualification_method,
        pc_qualification,
        activation_health,
        gradient_geometry,
    )
    # Why: select one fixed envelope before resources; never extend a live deadline.
    body_seconds = (
        probe_rollout.WIDE_EXTENDED_BODY_SECONDS if extended_wide_probe_budget else BODY_SECONDS
    )
    started, sampler = monotonic(), None
    methods = (
        (memory.METHOD,)
        if memory_control
        else (f"public_invariant_{qualification_method}",)
        if qualification_method is not None
        else PC_QUALIFICATION_METHODS
        if pc_qualification
        else INVARIANT_LEARNABILITY_METHODS
        if invariant_history
        else shared.METHODS
    )
    if cue_relative_history:
        methods = (f"public_cue_relative_{qualification_method}",)
    rows = [_new_row(method) for method in methods]
    catalogue_options: _CatalogueOptions = {"catalogue_rows": 54} if joint_catalogue else {}
    if wide_neural:
        catalogue_options["wide_neural"] = True
    if joint_catalogue:
        for row in rows:
            row.pop("signed_metrics")
            row.pop("acquisition_conditions")
            row["joint_catalogue_acquired"] = None
    if acquisition_only:
        for row in rows:
            row.pop("signed_metrics")
            row.pop("acquisition_conditions")
            row.update(
                A_fit_gains=None, A_acquisition=None, probe_source_stages=["initial", "post_A128"]
            )
    protocol_options: dict[str, Any] = dict(
        public_history=public_history, activation_health=activation_health
    )
    if wide_neural:
        protocol_options["wide_neural"] = True
    if old_a_rehearsal:
        protocol_options["old_a_rehearsal"] = True
    if extra_f_exposure:
        protocol_options["extra_f_exposure"] = True
    if gradient_geometry:
        protocol_options["gradient_geometry"] = True
    if prior_gradient_geometry:
        protocol_options["prior_gradient_geometry"] = True
    if pc_update_diagnostics:
        protocol_options["pc_update_diagnostics"] = True
    if invariant_history:
        protocol_options["invariant_history"] = True
    if cue_relative_history:
        protocol_options["cue_relative_history"] = True
    if pc_qualification:
        protocol_options["pc_qualification"] = True
    if qualification_method is not None:
        protocol_options["qualification_method"] = qualification_method
    if acquisition_only:
        protocol_options["acquisition_only"] = True
    if distinct_probe:
        protocol_options["distinct_probe"] = True
    if old_task_readouts:
        protocol_options["old_task_readouts"] = True
    if return_to_a:
        protocol_options["return_to_a"] = True
    if train_a_rollout:
        protocol_options["train_a_rollout"] = True
    if memory_control:
        protocol_options["memory_control"] = True
    if probe_rollouts:
        protocol_options["probe_rollouts"] = True
    if extended_wide_probe_budget:
        protocol_options["extended_wide_probe_budget"] = True
    if deferred_probe_rollouts:
        protocol_options["deferred_probe_rollouts"] = True
    if train_fit_bounds:
        protocol_options["train_fit_bounds"] = True
    if bp_update_diagnostics:
        protocol_options["bp_update_diagnostics"] = True
    if probe_inference is not None:
        protocol_options["probe_inference"] = probe_inference
    if weighted_bp:
        protocol_options["weighted_bp"] = True
    if weighted_pc:
        protocol_options["weighted_pc"] = True
    if weighted_catalogue:
        protocol_options["weighted_catalogue"] = True
    if train_action_margins:
        protocol_options["train_action_margins"] = True
    if joint_catalogue:
        protocol_options["joint_catalogue"] = True
    report: dict[str, Any] = dict(
        protocol=describe_protocol(**protocol_options),
        status="running",
        reason=None,
        methods=rows,
        training_sources=[],
        source_reports=[],
        probe_source_reports=[],
        probe_data=None,
        comparisons=None,
        work=None,
        observed_process_rss=None,
        total_seconds=None,
        scientific_advantage_verified=False,
    )
    deferred_options: _ProbeRolloutOptions = {}
    if wide_neural:
        deferred_options["wide_method"] = cast(str, qualification_method)
        if extended_wide_probe_budget:
            deferred_options["extended_wide_probe_budget"] = True
    deferred = (
        deferred_rollout.DeferredProbeRollouts(**deferred_options)
        if deferred_probe_rollouts
        else None
    )
    deferred_payload: bytes | None = None
    arms: list[dict[str, Any]] = []
    paid_benchmark_costs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    probe_guards: list[Callable[[], None]] = []
    diagnostic_guards: list[Callable[[], None]] = []
    rollout_guards: list[Callable[[], None]] = []
    rollout_payload = (
        pickle.dumps(train_rollout.empty_record(), protocol=5) if train_a_rollout else None
    )
    support_payload: bytes | None = None
    data: tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]] | None = None
    sources: tuple[AttributionBatch, ...] = ()
    packets: tuple[AttributionBatch, ...] = ()
    history_sources: tuple[AttributionBatch, ...] = ()
    history_packets: tuple[AttributionBatch, ...] = ()
    history_data: tuple[tuple[AttributionBatch, ...], tuple[AttributionBatch, ...]] | None = None
    signatures: list[tuple[tuple[AttributionBatch, ...], tuple[Any, ...]]] = []
    expected_rows = tuple(rows)
    original_protocol = report["protocol"]
    protocol_bytes = pickle.dumps(original_protocol, protocol=5)
    evidence_keys: tuple[str, ...] = (
        "training_sources",
        "source_reports",
        "probe_source_reports",
        "probe_data",
    )
    if joint_catalogue:
        report["catalogue_inputs"] = None
        evidence_keys += ("catalogue_inputs",)
    if train_fit_bounds:
        report["train_fit_bounds"] = None
        evidence_keys += ("train_fit_bounds",)
    if bp_update_diagnostics:
        report["bp_update_diagnostics"] = None
        evidence_keys += ("bp_update_diagnostics",)
    evidence_owners = {key: report[key] for key in evidence_keys}
    evidence_payloads = {key: pickle.dumps(report[key], protocol=5) for key in evidence_keys}
    collected_source_reports: list[dict[str, Any]] = []
    collected_probe_reports: list[dict[str, Any]] = []
    lists = [(row["readouts"], row["probes"]) for row in rows]
    expected_slots = [[list(readouts), list(probes)] for readouts, probes in lists]
    pending = {
        id(slot): pickle.dumps(slot, protocol=5)
        for row in rows
        for slot in (*row["readouts"], *row["probes"])
    }
    active_probe = None
    collection_phase = None

    def resources() -> None:
        if monotonic() - started > body_seconds:
            raise ValueError("fixed learnability whole-case wall-time budget exhausted")
        if sampler is not None:
            sampler.sample()
            if sampler.snapshot().peak_bytes > RSS_LIMIT:
                raise ValueError("fixed learnability sampled RSS budget exhausted")

    def common_guard() -> None:
        resources()
        if deferred is not None:
            deferred.check()
        if (
            report["methods"] is not rows
            or len(rows) != len(methods)
            or any(row is not original for row, original in zip(rows, expected_rows, strict=True))
            or tuple(row["method"] for row in rows) != methods
            or report["protocol"] is not original_protocol
            or pickle.dumps(original_protocol, protocol=5) != protocol_bytes
        ):
            raise ValueError("fixed method/protocol bindings changed")
        for batches, signature in signatures:
            if not shared._check_signature(batches, signature):
                raise ValueError("original training or reserved probe data changed")
        if any(
            report[key] is not evidence_owners[key]
            or pickle.dumps(report[key], protocol=5) != evidence_payloads[key]
            for key in evidence_keys
        ):
            raise ValueError("original source evidence container or values changed")
        for index, row in enumerate(rows):
            for column, key in enumerate(("readouts", "probes")):
                slots = row[key]
                expected = expected_slots[index][column]
                if slots is not lists[index][column] or len(slots) != len(expected):
                    raise ValueError("fixed result slot container changed")
                for slot_index, (slot, original) in enumerate(zip(slots, expected, strict=True)):
                    if slot is not original:
                        raise ValueError("prior result or planned slot replaced")
                    if type(slot) is dict:
                        if slot is active_probe:
                            if (
                                type(slot["checkpoint_updates"]) is not int
                                or slot["checkpoint_updates"] != (0, 128)[slot_index]
                            ):
                                raise ValueError("active probe source checkpoint changed")
                        elif pickle.dumps(slot, protocol=5) != pending[id(slot)]:
                            raise ValueError("unreached result slot changed")

    def all_completed() -> None:
        common_guard()
        for arm in arms:
            arm["guard"]()
        for guard in probe_guards:
            guard()
        for guard in diagnostic_guards:
            guard()
        for guard in rollout_guards:
            guard()
        common_guard()

    def freeze(row_index: int, key: str, index: int, value: dict[str, Any]) -> None:
        frozen = pickle.dumps(value, protocol=5)
        rows[row_index][key][index] = frozen
        expected_slots[row_index][0 if key == "readouts" else 1][index] = frozen

    def publish_evidence(key: str, value: Any) -> None:
        # Original collection buffers stay private while native work is pending.
        # Fixed owned bytes also preserve paid evidence if a public copy changes.
        evidence_payloads[key] = pickle.dumps(value, protocol=5)
        report[key] = pickle.loads(evidence_payloads[key])
        evidence_owners[key] = report[key]

    publication: list[dict[str, Any]] | None = None
    try:
        if not memory_control:
            torch = require_torch()
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
        with ProcessRssSampler() as active_sampler:
            sampler = active_sampler
            collection_phase = "benchmark_train_sources"
            sources = shared._collect_sources(common_guard, collected_source_reports)
            common_guard()
            collection_phase = None
            publish_evidence("source_reports", collected_source_reports)
            packets = (
                ()
                if joint_catalogue
                else shared._packets(sources, context_sequence=("A",) * 4)
                if acquisition_only
                else shared._packets(sources)
            )
            publish_evidence(
                "training_sources", [asdict(shared._episode(packet)) for packet in sources]
            )
            signatures.extend(
                (batches, shared._signature(batches)) for batches in (sources, packets)
            )
            if public_history:
                history_sources = _history_packets(sources)
                if joint_catalogue:
                    catalogue_plan = (
                        joint.compose(history_sources, cue_relative=True)
                        if cue_relative_history
                        else joint.compose(history_sources)
                    )
                    history_packets = catalogue_plan.training
                    publish_evidence("catalogue_inputs", joint.accounting(catalogue_plan))
                else:
                    history_packets = _history_packets(packets)
                signatures.extend(
                    (batches, shared._signature(batches))
                    for batches in (history_sources, history_packets)
                )
            for row in rows:
                all_completed()
                preceding = tuple(arm["guard"] for arm in arms)

                def before_arm(preceding=preceding) -> None:
                    common_guard()
                    for guard in preceding:
                        guard()
                    if invariant_history:
                        common_guard()

                if public_history:
                    arm_packets = packets if row["method"] == "recurrent_bc" else history_packets
                    arm = shared._train_arm(
                        row,
                        arm_packets,
                        sampler,
                        started + body_seconds,
                        before_arm,
                        learner_factory=(
                            weighted_catalogue_route.factory(
                                history_sources, arm_packets, method=cast(str, qualification_method)
                            )
                            if weighted_catalogue
                            else weighted_feature.factory(arm_packets)
                            if weighted_bp
                            else weighted_feature.factory(
                                arm_packets, method=cast(str, qualification_method)
                            )
                            if weighted_pc
                            else (lambda method: _history_learner(method, wide_neural=True))
                            if wide_neural
                            else _history_learner
                        ),
                        **catalogue_options,
                    )
                else:
                    arm = shared._train_arm(
                        row, packets, sampler, started + body_seconds, before_arm
                    )
                arms.append(arm)
                paid_benchmark_costs.append(
                    (
                        row,
                        {
                            key: row[key]
                            for key in (
                                "method",
                                "charged_updates",
                                "native_training_examples",
                                "wake_calls_attempted",
                                "wake_calls_returned",
                            )
                        },
                    )
                )
                if weighted_catalogue:
                    weighted_catalogue_route.check_receipts(
                        arm, method=cast(str, qualification_method)
                    )
                if weighted_bp:
                    weighted_feature.check_receipts(arm)
                if weighted_pc:
                    weighted_feature.check_receipts(arm, method=cast(str, qualification_method))
            all_completed()
            if bp_update_diagnostics:
                diagnostic, diagnostic_guard = _run_bp_checkpoint_readout(
                    arms[0],
                    history_sources,
                    sampler=sampler,
                    deadline=started + body_seconds,
                    check_boundary=all_completed,
                )
                publish_evidence("bp_update_diagnostics", diagnostic)
                if diagnostic["status"] != "complete":
                    raise ValueError(diagnostic["reason"])
                diagnostic_guards.append(diagnostic_guard)
                all_completed()
            if train_fit_bounds:
                # The learner has finished; bounds are report-only TRAIN evidence.
                bound_report = summarize_train_fit_bounds(
                    tuple(shared._episode(packet) for packet in history_sources),
                    **({"cue_relative": True} if cue_relative_history else {}),
                )
                all_completed()
                publish_evidence("train_fit_bounds", bound_report)
            collection_phase = "reserved_probe_sources"
            data = (
                _collect_probe_sources(all_completed, collected_probe_reports, distinct_probe=True)
                if distinct_probe
                else _collect_probe_sources(all_completed, collected_probe_reports)
            )
            all_completed()
            collection_phase = None
            publish_evidence("probe_source_reports", collected_probe_reports)
            _validate_batches(*data)
            if (
                len(data[0]) != 16
                or len(data[1]) != 2
                or any(
                    batch.features.shape != (9, 148) or batch.num_classes != 7
                    for batch in (*data[0], *data[1])
                )
            ):
                raise ValueError(
                    "reserved data must be sixteen training/two evaluation nine-row K7 episodes"
                )
            pair = _probe_pair(data)
            if pair != (("F-blue", "F-red") if distinct_probe else ("E-blue", "E-red")):
                raise ValueError(
                    "reserved packets must bind the fixed E cue contexts and alternating order"
                )
            probe_metadata = dict(
                input_array_bytes=sum(
                    batch.features.nbytes + batch.targets.nbytes for batch in (*data[0], *data[1])
                ),
                physical_episodes=4,
                physical_source_array_bytes=sum(
                    record["source_array_bytes"] for record in collected_probe_reports
                )
                if len(collected_probe_reports) == 4
                and all(
                    type(record.get("source_array_bytes")) is int
                    for record in collected_probe_reports
                )
                else None,
                source_array_retention="two original TRAIN arrays released after sixteen detached copies; two original TEST packet arrays included in input_array_bytes",
                scope=(
                    "same deterministic F domain, separate probe_train/probe_test collection; observed TRAIN support checked before updates, no generalization"
                    if distinct_probe
                    else "same deterministic E domain, separate probe_train/probe_test collection; no generalization"
                ),
                packets=[
                    dict(
                        episode_id=shared._episode(batch).episode_id,
                        role=batch.role,
                        context=batch.context,
                        source_episode_id=shared._episode(batch).episode_id.split(":copy:")[0],
                    )
                    for batch in (*data[0], *data[1])
                ],
            )
            if probe_metadata["input_array_bytes"] != 193104:
                raise ValueError("reserved input storage differs from the fixed18episode envelope")
            publish_evidence("probe_data", probe_metadata)
            signatures.extend((batches, shared._signature(batches)) for batches in data)
            if public_history:
                history_data = (_history_packets(data[0]), _history_packets(data[1]))
                _validate_batches(*history_data)
                history_bytes = sum(
                    batch.features.nbytes + batch.targets.nbytes
                    for batches in history_data
                    for batch in batches
                )
                if history_bytes != 384912:
                    raise ValueError("history probe storage differs from fixed18episode envelope")
                probe_metadata.update(
                    history_input_array_bytes=history_bytes,
                    retained_input_array_bytes=193104 + history_bytes,
                )
                publish_evidence("probe_data", probe_metadata)
                signatures.extend((batches, shared._signature(batches)) for batches in history_data)
            if distinct_probe:
                all_completed()
                if cue_relative_history:
                    # Admit the bijective domain before reusing canonical support counts.
                    if history_data is None:
                        raise ValueError("cue-relative support requires its raw history cohort")
                    cue_relative.validate_histories(
                        tuple(shared._episode(packet) for packet in history_sources)
                        + tuple(
                            shared._episode(packet)
                            for packets in history_data
                            for packet in packets
                        )
                    )
                prior_probe_packets = report["probe_data"]["packets"]
                prior_probe_records = tuple(prior_probe_packets)
                support = (
                    summarize_probe_support(sources, data, catalogue_sources=True)
                    if joint_catalogue
                    else summarize_probe_support(packets, data)
                )
                support_payload = pickle.dumps(support, protocol=5)
                # Why: an additive publication must not replace or reseal
                # preceding evidence changed during the diagnostic callback.
                all_completed()
                current_probe_packets = report["probe_data"]["packets"]
                if current_probe_packets is not prior_probe_packets or any(
                    current is not prior
                    for current, prior in zip(
                        current_probe_packets, prior_probe_records, strict=True
                    )
                ):
                    raise ValueError("preceding probe metadata packet records replaced")
                # Bind admission to the same captured record that is published,
                # rather than a helper-retained mutable return value.
                support = pickle.loads(support_payload)
                probe_metadata["support"] = support
                # Retain the completed diagnostic before refusing overlap or
                # admitting more work; benchmark receipts remain permanently paid.
                publish_evidence("probe_data", probe_metadata)
                all_completed()
                if (
                    support["all_canonical_inputs_unseen"] is not True
                    or len(support["probe"]["packets"]) != 18
                    or any(
                        record["canonical"]["unmatched_input_rows"] != 9
                        for record in support["probe"]["packets"]
                    )
                ):
                    raise ValueError(
                        "distinct F probe requires9of9 canonical TRAIN-unseen inputs in every episode"
                    )
            for row_index, arm in enumerate(arms):
                for index, (point, state) in enumerate(arm["states"]):
                    all_completed()
                    raw = _score_checkpoint(
                        arm["source"],
                        state,
                        method=rows[row_index]["method"],
                        updates=point,
                        sources=history_sources
                        if public_history and rows[row_index]["method"] != "recurrent_bc"
                        else sources,
                        budget=arm["budget"],
                        check_boundary=all_completed,
                        **catalogue_options,
                        **({"activation_health": True} if activation_health else {}),
                        **({"train_action_margins": True} if train_action_margins else {}),
                        **(
                            {
                                "pc_update_diagnostics": True,
                                **({"weighted_catalogue": True} if weighted_catalogue else {}),
                            }
                            if pc_update_diagnostics and point in (0, 128)
                            else {}
                        ),
                        **(
                            {"gradient_geometry": True}
                            if gradient_geometry and rows[row_index]["method"] == "recurrent_bc"
                            else {}
                        ),
                        **(
                            {"prior_gradient_geometry": True}
                            if prior_gradient_geometry
                            and rows[row_index]["method"] in INVARIANT_METHODS
                            else {}
                        ),
                    )
                    freeze(row_index, "readouts", index, raw)
                    if raw["status"] != "complete":
                        raise RuntimeError(str(raw["reason"]))
                    all_completed()
            for row_index, arm in enumerate(arms):
                for index, state_index in enumerate((0, 4)):
                    all_completed()
                    slot = rows[row_index]["probes"][index]
                    active_probe = slot
                    readout_options: _OldTaskReadoutOptions = {}
                    if wide_neural:
                        readout_options["wide_neural"] = True
                    if old_a_rehearsal:
                        readout_options["old_a_rehearsal"] = True
                    if extra_f_exposure:
                        readout_options["extra_f_exposure"] = True
                    if probe_inference is not None:
                        readout_options["probe_inference"] = probe_inference
                    if joint_catalogue:
                        readout_options["catalogue_rows"] = 54
                    if old_task_readouts:
                        readout_options["training_readouts"] = (
                            history_sources
                            if rows[row_index]["method"] != "recurrent_bc"
                            else sources
                        )[:2]
                    if return_to_a:
                        readout_options["return_to_a"] = True
                    if probe_rollouts:
                        readout_options["probe_rollouts"] = True
                    if extended_wide_probe_budget:
                        readout_options["extended_wide_probe_budget"] = True
                    if deferred is not None:
                        readout_options["deferred"] = deferred
                    try:
                        guard = _run_probe(
                            slot,
                            weighted_catalogue_route.probe_arm(
                                arm, method=cast(str, qualification_method)
                            )
                            if weighted_catalogue
                            else weighted_feature.probe_arm(arm)
                            if weighted_bp
                            else weighted_feature.probe_arm(
                                arm, method=cast(str, qualification_method)
                            )
                            if weighted_pc
                            else arm,
                            arm["states"][state_index][1],
                            history_data
                            if history_data is not None
                            and rows[row_index]["method"] != "recurrent_bc"
                            else data,
                            sampler=sampler,
                            deadline=started + body_seconds,
                            check_boundary=all_completed,
                            **readout_options,
                        )
                    finally:
                        active_probe = None
                        freeze(row_index, "probes", index, slot)
                    probe_guards.append(guard)
                    all_completed()
            all_completed()
            if deferred is not None:
                phase_record = _run_deferred_phase(
                    deferred, arms, check_phase=all_completed, resources=resources
                )
                deferred_payload = pickle.dumps(phase_record, protocol=5)
                if phase_record["status"] != "complete":
                    raise RuntimeError(str(phase_record["reason"]))
            if train_a_rollout:
                if joint_catalogue:
                    live, live_guard = run_train_rollout(
                        arms[0], check_boundary=all_completed, catalogue_method=rows[0]["method"]
                    )
                else:
                    live, live_guard = run_train_rollout(arms[0], check_boundary=all_completed)
                rollout_payload = pickle.dumps(live, protocol=5)
                if live["status"] != "complete":
                    raise RuntimeError(str(live["reason"]))
                rollout_guards.append(live_guard)
                all_completed()
            publication = []
            for row_index, row in enumerate(rows):
                raw_points = [pickle.loads(slot) for slot in expected_slots[row_index][0]]
                if joint_catalogue:
                    points = _fit_points(raw_points)
                    gaps = (
                        {}
                        if not train_fit_bounds
                        else dict(
                            fit_gap_outcomes=fit_gap_outcomes(
                                pickle.loads(evidence_payloads["train_fit_bounds"]), points
                            )
                        )
                    )
                    joint_acquired = all(
                        points[-1][context]["exact_sequence_accuracy"] == 1
                        for context in ("A", "B", "C_train")
                    )
                    publication.append(
                        dict(
                            outcomes=points,
                            joint_catalogue_acquired=joint_acquired,
                            **gaps,
                            interpretation="observed_joint_TRAIN_fit_only_no_live_claim"
                            if joint_acquired
                            else "inconclusive_joint_TRAIN_acquisition_incomplete",
                            status="complete",
                        )
                    )
                    continue
                if acquisition_only:
                    publication.append(_a_only_fit_outcomes(raw_points))
                    continue
                points, metrics, acquired = _fit_outcomes(raw_points)
                publication.append(
                    dict(
                        outcomes=points,
                        signed_metrics=metrics,
                        acquisition_conditions=acquired,
                        initial_A_acquired=acquired["A"],
                        interpretation="teacher_forced_acquired_fit_only_no_live_claim"
                        if acquired["A"]
                        else "inconclusive_initial_A_fit_incomplete",
                        status="complete",
                    )
                )
    except Exception as error:
        report.update(status="incomplete", reason=f"{type(error).__name__}: {error}")
        if collection_phase is not None:
            report["collection_failure"] = dict(
                phase=collection_phase, failure_work_may_be_unobserved=True
            )
            evidence_payloads["source_reports"] = pickle.dumps(collected_source_reports, protocol=5)
            evidence_payloads["probe_source_reports"] = pickle.dumps(
                collected_probe_reports, protocol=5
            )
    finally:
        for arm in arms:
            try:
                if memory_control:
                    shared._readback(
                        arm["row"],
                        arm["source"],
                        arm["states"][0][1],
                        arm["stream"],
                        arm["budget"],
                        state_equal=_same_value,
                    )
                elif joint_catalogue:
                    shared._readback(
                        arm["row"],
                        arm["source"],
                        arm["states"][0][1],
                        arm["stream"],
                        arm["budget"],
                        catalogue_rows=54,
                    )
                else:
                    shared._readback(
                        arm["row"], arm["source"], arm["states"][0][1], arm["stream"], arm["budget"]
                    )
                arm["guard"]()
            except Exception as error:
                report.setdefault("readback_errors", []).append(f"{type(error).__name__}: {error}")
                report["status"] = "incomplete"
        if sampler is not None:
            report["observed_process_rss"] = asdict(sampler.snapshot())
        report["total_seconds"] = monotonic() - started
    if report["status"] == "running":
        try:
            all_completed()
        except Exception as error:
            report.update(
                status="incomplete", reason=f"publication {type(error).__name__}: {error}"
            )
    if deferred is not None and deferred_payload is None:
        fallback: dict[str, Any] = dict(
            status="not_run",
            reason="training or capture did not complete",
            entry_validated=False,
            exit_validated=False,
            report=None,
        )
        try:
            fallback["report"] = pickle.loads(pickle.dumps(deferred.report(), protocol=5))
        except Exception as error:
            fallback.update(status="failed", reason=f"capture {type(error).__name__}: {error}")
        finally:
            try:
                all_completed()
                fallback["exit_validated"] = True
            except Exception as error:
                fallback["status"] = "failed"
                report["status"] = "incomplete"
                report["reason"] = (
                    report["reason"] or f"deferred closure {type(error).__name__}: {error}"
                )
        deferred_payload = pickle.dumps(fallback, protocol=5)
    # Completed records are private immutable bytes until the final publication;
    # no growing mutable record chain is rescanned at every native callback.
    report["methods"] = list(expected_rows)
    # A rejected late readback cannot erase work validated before diagnostics.
    # Retain only five fixed scalar records, not another native state history.
    for row, original_costs in paid_benchmark_costs:
        row.update(original_costs)
    for key in evidence_keys:
        report[key] = pickle.loads(evidence_payloads[key])
    for row_index, row in enumerate(expected_rows):
        for column, key in enumerate(("readouts", "probes")):
            row[key] = [
                pickle.loads(slot if type(slot) is bytes else pending[id(slot)])
                for slot in expected_slots[row_index][column]
            ]
    report["work"] = _work(
        list(expected_rows),
        [*report["source_reports"], *report["probe_source_reports"]],
        activation_health=activation_health,
        **({"gradient_geometry": True} if gradient_geometry else {}),
        **({"prior_gradient_geometry": True} if prior_gradient_geometry else {}),
        **({"pc_update_diagnostics": True} if pc_update_diagnostics else {}),
        **({"weighted_catalogue": True} if pc_update_diagnostics and weighted_catalogue else {}),
        **({"train_action_margins": True} if train_action_margins else {}),
        **({"old_task_readouts": True} if old_task_readouts else {}),
        **({"return_to_a": True} if return_to_a else {}),
        **({"old_a_rehearsal": True} if old_a_rehearsal else {}),
    )
    if wide_neural:
        _add_wide_neural_work(report["work"], cast(str, qualification_method))
    if bp_update_diagnostics:
        bp_update_route.add_work(report["work"], report["bp_update_diagnostics"])
    if probe_inference is not None:
        report["work"]["probe_inference"] = bounded_probe.sum_work(list(expected_rows))
    if distinct_probe:
        completed_support = None if support_payload is None else pickle.loads(support_payload)
        if completed_support is not None and (
            report["probe_data"] is None or "support" not in report["probe_data"]
        ):
            report["unpublished_probe_support"] = completed_support
        report["work"]["probe_support"] = (
            deepcopy(completed_support["work"])
            if completed_support is not None
            else dict(
                status="unavailable",
                native_updates=0,
                native_score_forwards=0,
                elapsed_seconds=None,
                failure_work_may_be_unobserved=True,
            )
        )
    if probe_rollouts:
        probe_rollout.add_work(report["work"], expected_rows)
    if deferred is not None:
        assert deferred_payload is not None
        phase_record = pickle.loads(deferred_payload)
        report["deferred_probe_rollouts"] = phase_record
        deferred_rollout.add_work(report["work"], phase_record)
    if train_a_rollout:
        assert rollout_payload is not None
        live_record = pickle.loads(rollout_payload)
        expected_rows[0]["train_a_rollout"] = live_record
        train_rollout.add_work(report["work"], live_record)
    if "collection_failure" in report:
        report["work"].update(
            failure_work_may_be_unobserved=True, source_environment_counts_are_lower_bounds=True
        )
    for row in expected_rows:
        row["work"] = _work(
            [row],
            [],
            activation_health=activation_health,
            **({"gradient_geometry": True} if gradient_geometry else {}),
            **({"prior_gradient_geometry": True} if prior_gradient_geometry else {}),
            **({"pc_update_diagnostics": True} if pc_update_diagnostics else {}),
            **(
                {"weighted_catalogue": True} if pc_update_diagnostics and weighted_catalogue else {}
            ),
            **({"train_action_margins": True} if train_action_margins else {}),
            **({"old_task_readouts": True} if old_task_readouts else {}),
            **({"return_to_a": True} if return_to_a else {}),
            **({"old_a_rehearsal": True} if old_a_rehearsal else {}),
        )
        if wide_neural:
            _add_wide_neural_work(row["work"], cast(str, qualification_method))
        if bp_update_diagnostics:
            bp_update_route.add_work(row["work"], report["bp_update_diagnostics"])
        if probe_rollouts:
            probe_rollout.add_work(row["work"], [row])
        if deferred is not None:
            deferred_rollout.add_work(row["work"], report["deferred_probe_rollouts"])
        if train_a_rollout:
            train_rollout.add_work(row["work"], row["train_a_rollout"])
        if probe_inference is not None:
            row["work"]["probe_inference"] = bounded_probe.sum_work([row])
        row["work"]["shared_teacher_work_excluded"] = True
        if distinct_probe:
            row["work"]["shared_probe_support_work_excluded"] = True
    if publication is not None and report["status"] == "running":
        probes = {row["method"]: row["probes"] for row in expected_rows}
        prefix = (
            "public_cue_relative_"
            if cue_relative_history
            else "public_invariant_"
            if invariant_history
            else ""
        )
        active = prefix + "gated_cpc"
        report["comparisons"] = (
            None
            if pc_qualification or qualification_method is not None or memory_control
            else {
                f"{active}-minus-{control}": {
                    "initial_probe_ce_gain": probes[active][0]["outcomes"]["ce_gain"]
                    - probes[control][0]["outcomes"]["ce_gain"],
                    "post_return_probe_ce_gain": probes[active][1]["outcomes"]["ce_gain"]
                    - probes[control][1]["outcomes"]["ce_gain"],
                }
                for control in (prefix + "pc", prefix + "neutral_cpc")
            }
        )
        for row, derived in zip(expected_rows, publication, strict=True):
            row.update(derived)
            row["post_A128_probe_gain_change" if acquisition_only else "probe_gain_change"] = (
                row["probes"][1]["outcomes"]["ce_gain"] - row["probes"][0]["outcomes"]["ce_gain"]
            )
        report["status"] = "complete"
    return report
