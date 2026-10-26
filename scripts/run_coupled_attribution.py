"""Run the frozen local pilot or describe its protocol without training."""

import argparse
import json
import os
from typing import Any

# Limit only this invocation, before NumPy/BLAS is imported by the adapter.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.adapters.train_acquisition_pilot import (
    describe_protocol as describe_train_acquisition_protocol,
    run_pilot as run_train_acquisition_pilot,
    describe_live_protocol as describe_train_live_protocol,
    run_live_pilot as run_train_live_pilot,
    describe_weighted_live_protocol as describe_weighted_train_live_protocol,
    run_weighted_live_pilot as run_weighted_train_live_pilot,
    describe_fixed_exposure_protocol as describe_fixed_a_exposure_protocol,
    run_fixed_exposure_pilot as run_fixed_a_exposure_pilot,
    describe_acquired_return_protocol as describe_acquired_task_return_protocol,
    run_acquired_return_pilot as run_acquired_task_return_pilot,
    describe_acquired_replay_protocol,
    run_acquired_replay_pilot,
    describe_return_protocol as describe_train_context_return_protocol,
    run_return_pilot as run_train_context_return_pilot,
    describe_four_phase_protocol,
    run_four_phase_pilot,
    describe_four_phase_evaluation_protocol,
    run_four_phase_evaluation_pilot,
    describe_four_phase_replay_protocol,
    run_four_phase_replay_pilot,
    describe_four_phase_probe_protocol,
    run_four_phase_probe_pilot,
    describe_return_evaluation_protocol as describe_return_policy_protocol,
    run_return_evaluation_pilot as run_return_policy_pilot,
    describe_return_replay_protocol,
    run_return_replay_pilot,
    describe_return_probe_protocol,
    run_return_probe_pilot,
)
from src.adapters.coupled_attribution_pilot import describe_protocol, run_pilot
from src.adapters.minigrid_learning_pilot import (
    describe_protocol as describe_minigrid_protocol,
    run_pilot as run_minigrid_pilot,
)
from src.adapters.minigrid_action_memory_pilot import (
    describe_protocol as describe_minigrid_action_memory_protocol,
    run_pilot as run_minigrid_action_memory_pilot,
)
from src.adapters.minigrid_learnability import (
    describe_protocol as describe_minigrid_learnability_protocol,
    run_pilot as run_minigrid_learnability_pilot,
)
from src.adapters.unit_renewal_pilot import (
    describe_protocol as describe_renewal_protocol,
    run_pilot as run_renewal_pilot,
)
from src.adapters.torch_feature_rehearsal_pilot import (
    describe_protocol as describe_torch_protocol,
    run_pilot as run_torch_pilot,
    describe_continuation_protocol,
    run_continuation_pilot,
    describe_trajectory_protocol,
    run_trajectory_pilot,
)
from src.adapters.line_rollout_pilot import (
    describe_protocol as describe_rollout_protocol,
    run_pilot as run_rollout_pilot,
)
from src.adapters.categorical_trajectory_pilot import (
    describe_protocol as describe_categorical_protocol,
    run_pilot as run_categorical_pilot,
)
from src.adapters.key_door_pilot import (
    describe_protocol as describe_key_door_protocol,
    run_pilot as run_key_door_pilot,
    describe_recurrent_protocol,
    run_recurrent_pilot,
    describe_learnability_protocol,
    run_learnability_pilot,
)

from src.adapters.key_door_reward_pilot import (
    describe_protocol as describe_reward_protocol,
    run_pilot as run_reward_pilot,
    describe_protocol as describe_legal_reward_protocol,
    run_pilot as run_legal_reward_pilot,
    describe_warm_start_protocol,
    run_warm_start_pilot,
    describe_warm_start_comparison_protocol,
    run_warm_start_comparison_pilot,
    describe_warm_start_comparison_protocol as describe_legal_warm_start_comparison_protocol,
    run_warm_start_comparison_pilot as run_legal_warm_start_comparison_pilot,
    describe_demonstration_protocol,
    run_demonstration_pilot,
)
from src.adapters.key_door_quota_comparison import (
    describe_protocol as describe_quota_comparison_protocol,
    run_pilot as run_quota_comparison_pilot,
)
from src.adapters.key_door_comparison_pilot import (
    describe_protocol as describe_comparison_protocol,
    run_pilot as run_comparison_pilot,
)
from src.adapters.key_door_acquired_quota import (
    describe_protocol as describe_acquired_prefix_protocol,
    run_pilot as run_acquired_prefix_pilot,
)
from src.adapters.key_door_quota_probes import (
    describe_protocol as describe_quota_probe_protocol,
    run_pilot as run_quota_probe_pilot,
)
from src.adapters.key_door_memory_pilot import (
    describe_protocol as describe_action_memory_protocol,
    run_pilot as run_action_memory_pilot,
    describe_probe_protocol as describe_action_memory_probe_protocol,
    run_probe_pilot as run_action_memory_probe_pilot,
)
from src.adapters.key_door_public_history_pilot import (
    describe_protocol as describe_public_history_protocol,
    run_pilot as run_public_history_pilot,
    describe_probe_protocol as describe_public_history_probe_protocol,
    run_probe_pilot as run_public_history_probe_pilot,
    describe_live_protocol as describe_public_history_live_protocol,
    run_live_pilot as run_public_history_live_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed local coupled attribution pilot")
    parser.add_argument("--describe", action="store_true", help="Print protocol without training")
    parser.add_argument(
        "--scenario",
        choices=(
            "original",
            "renewal-learnability",
            "torch-feature-rehearsal",
            "torch-feature-continuation",
            "torch-trajectory-rehearsal",
            "torch-trajectory-rollouts",
            "torch-categorical-rehearsal",
            "torch-key-door",
            "torch-minigrid-cue-key-door",
            "torch-minigrid-learnability",
            "torch-minigrid-history-learnability",
            "torch-key-door-recurrent",
            "torch-key-door-learnability",
            "torch-key-door-rl",
            "torch-key-door-legal-rl",
            "torch-key-door-legal-action-remap",
            "torch-key-door-legal-interaction-quota",
            "torch-key-door-frozen-interaction-quota",
            "torch-key-door-paired-interaction-quota",
            "torch-key-door-repeated-reward-control",
            "torch-key-door-acquired-interaction-quota",
            "torch-key-door-quota-probe",
            "torch-key-door-public-history-train",
            "torch-key-door-public-history-probe",
            "torch-key-door-public-history-live",
            "key-door-action-memory",
            "key-door-action-memory-probe",
            "minigrid-action-memory",
            "torch-key-door-warm-start-rl",
            "torch-key-door-warm-start-control",
            "torch-key-door-legal-warm-start-control",
            "torch-key-door-bc",
            "torch-key-door-weighted-bc",
            "torch-key-door-train-fit",
            "torch-key-door-train-live",
            "torch-key-door-weighted-train-live",
            "torch-key-door-fixed-a-exposure",
            "torch-key-door-acquired-return",
            "torch-key-door-acquired-recent48-replay",
            "torch-key-door-acquired-coverage-replay",
            "torch-key-door-train-return",
            "torch-key-door-train-four-phase",
            "torch-key-door-four-phase-policy",
            "torch-key-door-four-phase-recent",
            "torch-key-door-four-phase-partitioned",
            "torch-key-door-four-phase-recent48",
            "torch-key-door-four-phase-partitioned48",
            "torch-key-door-four-phase-coverage",
            "torch-key-door-four-phase-recent48-probe",
            "torch-key-door-four-phase-partitioned48-probe",
            "torch-key-door-four-phase-coverage-probe",
            "torch-key-door-return-policy",
            "torch-key-door-return-replay",
            "torch-key-door-return-probe",
            "torch-key-door-return-replay-probe",
        ),
        default="original",
    )
    parser.add_argument(
        "--source-artifact", help="recorded trajectory JSON for live rollout evaluation"
    )
    parser.add_argument(
        "--probe-decision-diagnostics",
        action="store_true",
        help="include reserved decision/class reports for torch-key-door-learnability",
    )
    parser.add_argument(
        "--native-activation-health",
        action="store_true",
        help="TRAIN-only native feature health for torch-minigrid-history-learnability",
    )
    parser.add_argument(
        "--prior-gradient-geometry",
        action="store_true",
        help="TRAIN-only bound feedforward prior mean-CE gradients; requires invariant history",
    )
    parser.add_argument(
        "--pc-update-diagnostics",
        action="store_true",
        help="TRAIN-only realized PC update at joint0/128; requires explicit PC/neutral joint catalogue",
    )
    parser.add_argument(
        "--bp-update-diagnostics",
        action="store_true",
        help="TRAIN-only disposable BP SGD/momentum cycle at joint128; requires explicit original BP F/old-A/return",
    )
    parser.add_argument(
        "--train-action-margins",
        action="store_true",
        help="TRAIN-only action margins from existing paid probabilities; requires invariant history",
    )
    qualification = parser.add_mutually_exclusive_group()
    qualification.add_argument(
        "--pc-qualification",
        action="store_true",
        help="single canonical-PC TRAIN-fit/probe qualification; requires invariant history",
    )
    qualification.add_argument(
        "--qualification-method",
        choices=("bp", "pc", "neutral_cpc", "gated_cpc", "pc_relax64", "neutral_cpc_relax64"),
        help="one existing bound TRAIN-fit/probe control; requires invariant history",
    )
    parser.add_argument(
        "--weighted-bp",
        action="store_true",
        help="distinct weighted TRAIN BP objective; requires invariant history and --qualification-method bp; probes stay unweighted",
    )
    parser.add_argument(
        "--weighted-pc",
        action="store_true",
        help="explicit weighted TRAIN PC/neutral objective; requires invariant history and explicit pc/neutral_cpc; probes stay unweighted",
    )
    parser.add_argument(
        "--weighted-catalogue",
        action="store_true",
        help="Explicit fixed aggregate TRAIN weighting for joint BP/PC/neutral catalogue",
    )
    parser.add_argument(
        "--joint-catalogue",
        action="store_true",
        help="Joint54-row TRAIN acquisition; requires explicit invariant method, distinct F, old-A readouts and A return",
    )
    parser.add_argument(
        "--wide-neural",
        action="store_true",
        help="Fixed H32 native joint F16/old-A/A16 diagnostic for one explicit method",
    )
    parser.add_argument(
        "--cue-relative-history",
        action="store_true",
        help="Public cue-relative key encoding for the explicit joint F/old-A/return native route",
    )
    parser.add_argument(
        "--old-a-rehearsal",
        action="store_true",
        help="Fixed old-A TRAIN replay on isolated F learners before their same-candidate A return",
    )
    parser.add_argument(
        "--extra-f-exposure",
        action="store_true",
        help="Explicit H32 F48/A16 control with the same native calls/rows as old-A replay",
    )
    parser.add_argument(
        "--acquisition-only",
        action="store_true",
        help="A-only128-update diagnostic for one explicit bound native method",
    )
    parser.add_argument(
        "--distinct-probe",
        action="store_true",
        help="Use the fixed row-two F task with measured canonical TRAIN support overlap",
    )
    parser.add_argument(
        "--old-task-readouts",
        action="store_true",
        help="Read old A TRAIN fit on each disposable F learner with separately counted forwards",
    )
    parser.add_argument(
        "--return-to-a",
        action="store_true",
        help="Fixed separately budgeted A16 return on each F-adapted disposable learner",
    )
    parser.add_argument(
        "--probe-rollouts",
        action="store_true",
        help="Target-free A/F games after each isolated count/H32 F16 and A16 return",
    )
    parser.add_argument(
        "--extended-wide-probe-budget",
        action="store_true",
        help="Explicit H32 probe games with fixed60s endpoints/240s parent; default40s/180s unchanged",
    )
    parser.add_argument(
        "--deferred-probe-rollouts",
        action="store_true",
        help="Freeze paid count/H32 endpoints and run target-free A/F games after all fixed training",
    )
    parser.add_argument(
        "--memory-control",
        action="store_true",
        help="Existing bounded invariant exact-memory A128/F16/old-A/A16 control",
    )
    parser.add_argument(
        "--train-a-rollout",
        action="store_true",
        help="Two target-free known TRAIN A games from the frozen A-only or joint128 checkpoint",
    )
    parser.add_argument(
        "--invariant-memory",
        action="store_true",
        help="Existing invariant count control on the fixed recurring MiniGrid TRAIN stream",
    )
    parser.add_argument(
        "--invariant-fifo-memory",
        action="store_true",
        help="Fixed36-entry invariant FIFO control on the recurring MiniGrid TRAIN stream",
    )
    parser.add_argument(
        "--invariant-history",
        action="store_true",
        help="encoding-bound feedforward TRAIN fit/probes for torch-minigrid-history-learnability",
    )
    parser.add_argument(
        "--cue-gradient-geometry",
        action="store_true",
        help="TRAIN-only recurrent mean-CE gradient pairs for MiniGrid learnability",
    )
    parser.add_argument(
        "--environment-seed",
        type=int,
        choices=range(24),
        help="training catalogue instance0..23 for mapped quota route",
    )
    parser.add_argument(
        "--heldout-environment-seed",
        type=int,
        choices=range(24),
        help="distinct final-only catalogue instance0..23 for mapped quota route",
    )
    parser.add_argument(
        "--paired-wall-seconds",
        type=float,
        help="prospective shared wall cap in (0, 180] seconds for paired interaction quota only (default90)",
    )
    parser.add_argument(
        "--probe-inference",
        choices=("fixed-weighted", "bounded-weighted"),
        help="Explicit matched weighted F16/A16 probe recipe on a PC/neutral joint history arm",
    )
    parser.add_argument(
        "--train-fit-bounds",
        action="store_true",
        help="Report TRAIN-only identical-input row accuracy/CE bounds and signed fit gaps after the joint benchmark",
    )
    args = parser.parse_args()
    if args.extra_f_exposure and (
        not args.wide_neural
        or args.old_a_rehearsal
        or args.probe_rollouts
        or args.deferred_probe_rollouts
        or args.extended_wide_probe_budget
    ):
        parser.error(
            "--extra-f-exposure requires the explicit H32 F48/A16 route without replay or games"
        )
    if args.wide_neural and (
        args.scenario != "torch-minigrid-history-learnability"
        or not all(
            (
                args.invariant_history,
                args.joint_catalogue,
                args.distinct_probe,
                args.old_task_readouts,
                args.return_to_a,
            )
        )
        or args.qualification_method
        not in ("bp", "pc", "neutral_cpc", "gated_cpc", "pc_relax64", "neutral_cpc_relax64")
        or any(
            (
                args.native_activation_health,
                args.cue_gradient_geometry,
                args.pc_qualification,
                args.acquisition_only,
                args.train_a_rollout,
                args.memory_control,
                args.weighted_bp,
                args.weighted_pc,
                args.weighted_catalogue,
                args.train_action_margins,
                args.train_fit_bounds,
                args.cue_relative_history,
                args.bp_update_diagnostics,
            )
        )
        or (args.probe_rollouts and args.deferred_probe_rollouts)
    ):
        parser.error(
            "--wide-neural requires explicit native history joint H32 F16/old-A/A16 without extra options"
        )
    if (
        args.wide_neural
        and args.prior_gradient_geometry
        and (
            args.qualification_method not in ("bp", "pc", "neutral_cpc", "gated_cpc")
            or args.pc_update_diagnostics
            or args.old_a_rehearsal
            or args.extra_f_exposure
            or args.probe_rollouts
            or args.deferred_probe_rollouts
            or args.extended_wide_probe_budget
        )
    ):
        parser.error("H32 prior gradients require the original joint TRAIN recipe alone")
    if (
        args.wide_neural
        and args.pc_update_diagnostics
        and (
            args.qualification_method not in ("pc", "neutral_cpc")
            or args.old_a_rehearsal
            or args.extra_f_exposure
            or args.probe_rollouts
            or args.deferred_probe_rollouts
            or args.extended_wide_probe_budget
        )
    ):
        parser.error(
            "H32 PC update diagnostics require original PC/neutral F16/A16 without replay or games"
        )
    if args.extended_wide_probe_budget and not (
        args.wide_neural and (args.probe_rollouts != args.deferred_probe_rollouts)
    ):
        parser.error("--extended-wide-probe-budget requires one explicit H32 game route")
    if args.bp_update_diagnostics and (
        args.scenario != "torch-minigrid-history-learnability"
        or not args.invariant_history
        or not args.joint_catalogue
        or args.qualification_method != "bp"
        or not args.distinct_probe
        or not args.old_task_readouts
        or not args.return_to_a
        or args.pc_qualification
        or args.cue_relative_history
        or args.probe_inference is not None
        or args.weighted_bp
        or args.weighted_pc
        or args.weighted_catalogue
        or args.acquisition_only
        or args.memory_control
        or args.native_activation_health
        or args.cue_gradient_geometry
        or args.prior_gradient_geometry
        or args.train_action_margins
        or args.pc_update_diagnostics
        or args.old_a_rehearsal
        or args.train_a_rollout
        or args.probe_rollouts
        or args.deferred_probe_rollouts
        or args.probe_decision_diagnostics
        or args.invariant_memory
        or args.invariant_fifo_memory
        or args.environment_seed is not None
        or args.heldout_environment_seed is not None
        or args.source_artifact is not None
    ):
        parser.error(
            "--bp-update-diagnostics requires explicit original BP joint invariant F/old-A/return without other diagnostics, rehearsal, weighting or games"
        )
    if args.old_a_rehearsal and (
        args.scenario != "torch-minigrid-history-learnability"
        or not args.invariant_history
        or not args.joint_catalogue
        or args.qualification_method not in ("bp", "pc", "neutral_cpc", "gated_cpc")
        or not args.distinct_probe
        or not args.old_task_readouts
        or not args.return_to_a
        or args.pc_qualification
        or args.cue_relative_history
        or args.probe_inference is not None
        or args.weighted_bp
        or args.weighted_pc
        or args.weighted_catalogue
        or args.acquisition_only
        or args.memory_control
        or args.native_activation_health
        or args.cue_gradient_geometry
        or args.prior_gradient_geometry
        or args.train_action_margins
        or args.pc_update_diagnostics
        or args.train_a_rollout
        or args.probe_rollouts
        or args.deferred_probe_rollouts
        or args.probe_decision_diagnostics
        or args.invariant_memory
        or args.invariant_fifo_memory
        or args.environment_seed is not None
        or args.heldout_environment_seed is not None
        or args.source_artifact is not None
    ):
        parser.error(
            "--old-a-rehearsal requires explicit native joint invariant F/old-A/return without extra diagnostics, weighting or games"
        )
    if args.train_fit_bounds and (
        args.scenario != "torch-minigrid-history-learnability" or not args.joint_catalogue
    ):
        parser.error("--train-fit-bounds requires the history-learnability joint catalogue")
    if args.probe_inference is not None and (
        args.scenario != "torch-minigrid-history-learnability"
        or not args.joint_catalogue
        or not args.invariant_history
        or not args.distinct_probe
        or not args.old_task_readouts
        or not args.return_to_a
        or args.qualification_method
        not in ("pc", "neutral_cpc", "pc_relax64", "neutral_cpc_relax64")
        or args.pc_qualification
        or args.acquisition_only
        or args.memory_control
        or args.weighted_bp
        or args.weighted_pc
        or args.native_activation_health
        or args.cue_gradient_geometry
        or args.prior_gradient_geometry
        or args.train_a_rollout
        or args.probe_rollouts
        or args.deferred_probe_rollouts
        or args.train_action_margins
        or args.pc_update_diagnostics
        or (
            args.wide_neural
            and (
                args.qualification_method not in ("pc", "neutral_cpc")
                or args.old_a_rehearsal
                or args.extra_f_exposure
                or args.extended_wide_probe_budget
            )
        )
    ):
        parser.error(
            "--probe-inference requires explicit PC/neutral joint invariant F16/readouts/A16 without extra diagnostics"
        )

    if args.qualification_method in ("pc_relax64", "neutral_cpc_relax64") and (
        args.scenario != "torch-minigrid-history-learnability"
        or not args.joint_catalogue
        or args.train_action_margins
        or args.pc_update_diagnostics
        or (
            args.wide_neural
            and (
                args.old_a_rehearsal
                or args.extra_f_exposure
                or args.probe_rollouts
                or args.deferred_probe_rollouts
                or args.extended_wide_probe_budget
            )
        )
    ):
        parser.error("fixed64 control requires history joint catalogue without extra diagnostics")
    if args.pc_update_diagnostics and (
        args.scenario != "torch-minigrid-history-learnability"
        or not args.joint_catalogue
        or args.qualification_method not in ("pc", "neutral_cpc")
        or args.train_action_margins
    ):
        parser.error(
            "--pc-update-diagnostics requires joint catalogue with explicit pc/neutral_cpc, without action margins"
        )
    if args.weighted_catalogue and (
        args.scenario != "torch-minigrid-history-learnability"
        or not args.joint_catalogue
        or args.qualification_method
        not in ("bp", "pc", "neutral_cpc", "pc_relax64", "neutral_cpc_relax64")
        or args.weighted_bp
        or args.weighted_pc
    ):
        parser.error(
            "--weighted-catalogue requires joint history catalogue and explicit bp/pc/neutral_cpc or fixed64 control, without episode weighting"
        )
    if args.cue_relative_history and (
        args.scenario != "torch-minigrid-history-learnability"
        or not args.joint_catalogue
        or args.qualification_method not in ("bp", "pc", "neutral_cpc", "gated_cpc")
        or args.weighted_catalogue
        or args.weighted_bp
        or args.weighted_pc
        or args.train_action_margins
        or args.pc_update_diagnostics
        or args.native_activation_health
        or args.cue_gradient_geometry
        or args.prior_gradient_geometry
        or args.train_a_rollout
        or args.probe_rollouts
        or args.deferred_probe_rollouts
        or args.probe_inference is not None
    ):
        parser.error(
            "--cue-relative-history requires the explicit native joint F/old-A/return route without projection-specific diagnostics or weighting"
        )
    if args.joint_catalogue and (
        args.scenario != "torch-minigrid-history-learnability"
        or not args.invariant_history
        or args.qualification_method is None
        or not args.distinct_probe
        or not args.old_task_readouts
        or not args.return_to_a
        or args.pc_qualification
        or args.acquisition_only
        or args.weighted_bp
        or args.weighted_pc
        or args.memory_control
        or args.native_activation_health
        or args.cue_gradient_geometry
        or (args.prior_gradient_geometry and not args.wide_neural)
        or (args.probe_rollouts and not args.wide_neural)
        or (args.deferred_probe_rollouts and not args.wide_neural)
    ):
        parser.error(
            "--joint-catalogue requires invariant history, one explicit neural method, distinct probe/old-task readouts/return-to-a without other training diagnostics"
        )
    if args.invariant_memory and args.scenario != "minigrid-action-memory":
        parser.error("--invariant-memory requires --scenario minigrid-action-memory")
    if args.invariant_fifo_memory and args.scenario != "minigrid-action-memory":
        parser.error("--invariant-fifo-memory requires --scenario minigrid-action-memory")
    if args.invariant_fifo_memory and args.invariant_memory:
        parser.error("--invariant-memory and --invariant-fifo-memory are mutually exclusive")
    seeded = args.environment_seed is not None or args.heldout_environment_seed is not None
    if args.paired_wall_seconds is not None and (
        args.scenario != "torch-key-door-paired-interaction-quota"
        or not 0 < args.paired_wall_seconds <= 180
    ):
        parser.error(
            "--paired-wall-seconds requires paired interaction quota and finite seconds in (0, 180]"
        )
    if seeded and (
        args.scenario
        not in (
            "torch-key-door-legal-interaction-quota",
            "torch-key-door-frozen-interaction-quota",
            "torch-key-door-paired-interaction-quota",
        )
        or args.environment_seed is None
        or args.heldout_environment_seed is None
        or args.environment_seed == args.heldout_environment_seed
    ):
        parser.error(
            "environment seeds require a distinct pair on a legal, frozen or paired interaction-quota scenario"
        )
    if args.weighted_bp and (
        args.scenario != "torch-minigrid-history-learnability"
        or not args.invariant_history
        or args.qualification_method != "bp"
        or args.pc_qualification
        or args.memory_control
    ):
        parser.error(
            "--weighted-bp requires invariant history with --qualification-method bp and no PC alias/memory control"
        )
    if args.weighted_pc and (
        args.scenario != "torch-minigrid-history-learnability"
        or not args.invariant_history
        or args.qualification_method not in ("pc", "neutral_cpc")
        or args.pc_qualification
        or args.memory_control
        or args.weighted_bp
        or args.joint_catalogue
    ):
        parser.error(
            "--weighted-pc requires invariant history and explicit pc/neutral_cpc without BP weighting, PC alias, memory or joint catalogue"
        )
    if args.train_action_margins and (
        args.scenario != "torch-minigrid-history-learnability" or not args.invariant_history
    ):
        parser.error(
            "--train-action-margins requires history-learnability with --invariant-history"
        )
    if args.prior_gradient_geometry and (
        args.scenario != "torch-minigrid-history-learnability"
        or not args.invariant_history
        or args.cue_gradient_geometry
        or args.memory_control
        or args.acquisition_only
    ):
        parser.error(
            "--prior-gradient-geometry requires invariant neural history without cue gradients, memory control or acquisition-only"
        )
    if args.pc_qualification and (
        args.scenario != "torch-minigrid-history-learnability" or not args.invariant_history
    ):
        parser.error("--pc-qualification requires history-learnability with --invariant-history")
    if args.qualification_method is not None and (
        args.scenario != "torch-minigrid-history-learnability" or not args.invariant_history
    ):
        parser.error(
            "--qualification-method requires history-learnability with --invariant-history"
        )
    if args.old_task_readouts and (
        args.scenario != "torch-minigrid-history-learnability"
        or not args.invariant_history
        or not args.distinct_probe
    ):
        parser.error("--old-task-readouts requires invariant history with --distinct-probe")
    if args.probe_rollouts and not (args.memory_control or args.wide_neural):
        parser.error(
            "--probe-rollouts requires the complete --memory-control or explicit --wide-neural recipe"
        )
    if args.deferred_probe_rollouts and (
        not (args.memory_control or args.wide_neural) or args.probe_rollouts
    ):
        parser.error("--deferred-probe-rollouts requires count/H32 without --probe-rollouts")
    if args.memory_control and not (
        args.scenario == "torch-minigrid-history-learnability"
        and args.invariant_history
        and args.acquisition_only
        and args.distinct_probe
        and args.old_task_readouts
        and args.return_to_a
        and args.qualification_method is None
        and not args.pc_qualification
        and not args.native_activation_health
        and not args.cue_gradient_geometry
        and not args.train_a_rollout
    ):
        parser.error(
            "--memory-control requires the invariant A-only/distinct-probe/old-task/return recipe without neural or live flags"
        )
    if args.train_a_rollout and not (
        args.scenario == "torch-minigrid-history-learnability"
        and args.invariant_history
        and (args.acquisition_only or args.joint_catalogue)
        and args.qualification_method
        in ("bp", "pc", "neutral_cpc", "gated_cpc", "pc_relax64", "neutral_cpc_relax64")
        and (
            not args.joint_catalogue
            or not (args.train_action_margins or args.pc_update_diagnostics)
        )
    ):
        parser.error(
            "--train-a-rollout requires an explicit bound A-only or joint method without extra joint diagnostics"
        )
    if args.return_to_a and not args.old_task_readouts:
        parser.error(
            "--return-to-a requires --old-task-readouts and its invariant distinct-task scope"
        )
    if args.distinct_probe and (
        args.scenario != "torch-minigrid-history-learnability" or not args.invariant_history
    ):
        parser.error("--distinct-probe requires history-learnability with --invariant-history")
    if args.acquisition_only and (
        args.scenario != "torch-minigrid-history-learnability"
        or not args.invariant_history
        or (
            args.qualification_method not in ("bp", "pc", "neutral_cpc", "gated_cpc")
            and not args.memory_control
        )
        or args.native_activation_health
        or args.cue_gradient_geometry
    ):
        parser.error(
            "--acquisition-only requires invariant history with one explicit bound method and no health/gradient flags"
        )
    if args.invariant_history and (
        args.scenario != "torch-minigrid-history-learnability" or args.cue_gradient_geometry
    ):
        parser.error(
            "--invariant-history requires the history learnability scenario without gradient flags"
        )
    if args.cue_gradient_geometry and args.scenario not in (
        "torch-minigrid-learnability",
        "torch-minigrid-history-learnability",
    ):
        parser.error("--cue-gradient-geometry requires a MiniGrid learnability scenario")
    if args.native_activation_health and args.scenario != "torch-minigrid-history-learnability":
        parser.error("--native-activation-health requires torch-minigrid-history-learnability")
    if args.probe_decision_diagnostics and args.scenario != "torch-key-door-learnability":
        parser.error("--probe-decision-diagnostics is only valid for torch-key-door-learnability")
    if args.scenario == "torch-trajectory-rollouts":
        if args.describe:
            if args.source_artifact is not None:
                parser.error("omit --source-artifact when describing the fixed rollout protocol")
            report = describe_rollout_protocol()
        else:
            if args.source_artifact is None:
                parser.error("torch-trajectory-rollouts requires --source-artifact")
            report = run_rollout_pilot(args.source_artifact)
    elif args.source_artifact is not None:
        parser.error("--source-artifact is only valid for torch-trajectory-rollouts")
    elif args.scenario == "torch-minigrid-cue-key-door":
        report = describe_minigrid_protocol() if args.describe else run_minigrid_pilot()
    elif args.scenario == "torch-minigrid-learnability":
        gradient_options: dict[str, Any] = (
            {"gradient_geometry": True} if args.cue_gradient_geometry else {}
        )
        report = (
            describe_minigrid_learnability_protocol(**gradient_options)
            if args.describe
            else run_minigrid_learnability_pilot(**gradient_options)
        )
    elif args.scenario == "torch-minigrid-history-learnability":
        history_options: dict[str, Any] = {"public_history": True}
        if args.wide_neural:
            history_options["wide_neural"] = True
        if args.invariant_history:
            history_options["invariant_history"] = True
        if args.pc_qualification:
            history_options["pc_qualification"] = True
        if args.qualification_method is not None:
            history_options["qualification_method"] = args.qualification_method
        if args.acquisition_only:
            history_options["acquisition_only"] = True
        if args.distinct_probe:
            history_options["distinct_probe"] = True
        if args.old_task_readouts:
            history_options["old_task_readouts"] = True
        if args.return_to_a:
            history_options["return_to_a"] = True
        if args.train_a_rollout:
            history_options["train_a_rollout"] = True
        if args.memory_control:
            history_options["memory_control"] = True
        if args.probe_rollouts:
            history_options["probe_rollouts"] = True
        if args.extended_wide_probe_budget:
            history_options["extended_wide_probe_budget"] = True
        if args.deferred_probe_rollouts:
            history_options["deferred_probe_rollouts"] = True
        if args.native_activation_health:
            history_options["activation_health"] = True
        if args.cue_gradient_geometry:
            history_options["gradient_geometry"] = True
        if args.prior_gradient_geometry:
            history_options["prior_gradient_geometry"] = True
        if args.train_fit_bounds:
            history_options["train_fit_bounds"] = True
        if args.cue_relative_history:
            history_options["cue_relative_history"] = True
        if args.old_a_rehearsal:
            history_options["old_a_rehearsal"] = True
        if args.extra_f_exposure:
            history_options["extra_f_exposure"] = True
        if args.probe_inference is not None:
            history_options["probe_inference"] = args.probe_inference
        if args.weighted_bp:
            history_options["weighted_bp"] = True
        if args.weighted_pc:
            history_options["weighted_pc"] = True
        if args.weighted_catalogue:
            history_options["weighted_catalogue"] = True
        if args.train_action_margins:
            history_options["train_action_margins"] = True
        if args.joint_catalogue:
            history_options["joint_catalogue"] = True
        if args.pc_update_diagnostics:
            history_options["pc_update_diagnostics"] = True
        if args.bp_update_diagnostics:
            history_options["bp_update_diagnostics"] = True
        report = (
            describe_minigrid_learnability_protocol(**history_options)
            if args.describe
            else run_minigrid_learnability_pilot(**history_options)
        )
    elif args.scenario in ("torch-key-door-return-probe", "torch-key-door-return-replay-probe"):
        replay_options = (
            {"episode_replay": True}
            if args.scenario == "torch-key-door-return-replay-probe"
            else {}
        )
        report = (
            describe_return_probe_protocol(**replay_options)
            if args.describe
            else run_return_probe_pilot(**replay_options)
        )
    elif args.scenario == "torch-key-door-return-replay":
        report = describe_return_replay_protocol() if args.describe else run_return_replay_pilot()
    elif args.scenario == "torch-key-door-return-policy":
        report = describe_return_policy_protocol() if args.describe else run_return_policy_pilot()
    elif args.scenario == "torch-key-door-repeated-reward-control":
        report = describe_comparison_protocol() if args.describe else run_comparison_pilot()
    elif args.scenario == "torch-key-door-acquired-interaction-quota":
        report = (
            describe_acquired_prefix_protocol() if args.describe else run_acquired_prefix_pilot()
        )
    elif args.scenario == "torch-key-door-quota-probe":
        report = describe_quota_probe_protocol() if args.describe else run_quota_probe_pilot()
    elif args.scenario == "key-door-action-memory-probe":
        report = (
            describe_action_memory_probe_protocol()
            if args.describe
            else run_action_memory_probe_pilot()
        )
    elif args.scenario == "minigrid-action-memory":
        options = (
            dict(invariant_fifo_memory=True)
            if args.invariant_fifo_memory
            else dict(invariant_memory=True)
            if args.invariant_memory
            else {}
        )
        report = (
            describe_minigrid_action_memory_protocol(**options)
            if args.describe
            else run_minigrid_action_memory_pilot(**options)
        )
    elif args.scenario == "key-door-action-memory":
        report = describe_action_memory_protocol() if args.describe else run_action_memory_pilot()
    elif args.scenario == "torch-key-door-public-history-train":
        report = describe_public_history_protocol() if args.describe else run_public_history_pilot()
    elif args.scenario == "torch-key-door-public-history-probe":
        report = (
            describe_public_history_probe_protocol()
            if args.describe
            else run_public_history_probe_pilot()
        )
    elif args.scenario == "torch-key-door-public-history-live":
        report = (
            describe_public_history_live_protocol()
            if args.describe
            else run_public_history_live_pilot()
        )
    elif args.scenario in (
        "torch-key-door-four-phase-recent48-probe",
        "torch-key-door-four-phase-partitioned48-probe",
        "torch-key-door-four-phase-coverage-probe",
    ):
        control = args.scenario.removesuffix("-probe").rsplit("-", 1)[1]
        report = (
            describe_four_phase_probe_protocol(control=control)
            if args.describe
            else run_four_phase_probe_pilot(control=control)
        )
    elif args.scenario in (
        "torch-key-door-four-phase-recent",
        "torch-key-door-four-phase-partitioned",
        "torch-key-door-four-phase-recent48",
        "torch-key-door-four-phase-partitioned48",
        "torch-key-door-four-phase-coverage",
    ):
        control = args.scenario.rsplit("-", 1)[1]
        report = (
            describe_four_phase_replay_protocol(control=control)
            if args.describe
            else run_four_phase_replay_pilot(control=control)
        )
    elif args.scenario == "torch-key-door-four-phase-policy":
        report = (
            describe_four_phase_evaluation_protocol()
            if args.describe
            else run_four_phase_evaluation_pilot()
        )
    elif args.scenario == "torch-key-door-train-four-phase":
        report = describe_four_phase_protocol() if args.describe else run_four_phase_pilot()
    elif args.scenario == "torch-key-door-train-return":
        report = (
            describe_train_context_return_protocol()
            if args.describe
            else run_train_context_return_pilot()
        )
    elif args.scenario == "torch-key-door-train-live":
        report = describe_train_live_protocol() if args.describe else run_train_live_pilot()
    elif args.scenario == "torch-key-door-weighted-train-live":
        report = (
            describe_weighted_train_live_protocol()
            if args.describe
            else run_weighted_train_live_pilot()
        )
    elif args.scenario == "torch-key-door-fixed-a-exposure":
        report = (
            describe_fixed_a_exposure_protocol() if args.describe else run_fixed_a_exposure_pilot()
        )
    elif args.scenario == "torch-key-door-train-fit":
        report = (
            describe_train_acquisition_protocol()
            if args.describe
            else run_train_acquisition_pilot()
        )
    elif args.scenario in (
        "torch-key-door-acquired-recent48-replay",
        "torch-key-door-acquired-coverage-replay",
    ):
        control = args.scenario.removesuffix("-replay").rsplit("-", 1)[1]
        report = (
            describe_acquired_replay_protocol(control=control)
            if args.describe
            else run_acquired_replay_pilot(control=control)
        )
    elif args.scenario == "torch-key-door-acquired-return":
        report = (
            describe_acquired_task_return_protocol()
            if args.describe
            else run_acquired_task_return_pilot()
        )
    elif args.scenario in ("torch-key-door-bc", "torch-key-door-weighted-bc"):
        options = {"class_weighted": True} if args.scenario == "torch-key-door-weighted-bc" else {}
        report = (
            describe_demonstration_protocol(**options)
            if args.describe
            else run_demonstration_pilot(**options)
        )
    elif args.scenario == "torch-key-door-legal-warm-start-control":
        selected = (
            describe_legal_warm_start_comparison_protocol
            if args.describe
            else run_legal_warm_start_comparison_pilot
        )
        report = selected(condition_on_legal_actions=True)
    elif args.scenario == "torch-key-door-warm-start-control":
        report = (
            describe_warm_start_comparison_protocol()
            if args.describe
            else run_warm_start_comparison_pilot()
        )
    elif args.scenario == "torch-key-door-warm-start-rl":
        report = describe_warm_start_protocol() if args.describe else run_warm_start_pilot()
    elif args.scenario == "torch-key-door-paired-interaction-quota":
        selected_quota = (
            describe_quota_comparison_protocol if args.describe else run_quota_comparison_pilot
        )
        quota_options: dict[str, Any] = (
            dict(
                environment_seed=args.environment_seed,
                heldout_environment_seed=args.heldout_environment_seed,
            )
            if seeded
            else {}
        )
        if args.paired_wall_seconds is not None:
            quota_options["wall_seconds"] = args.paired_wall_seconds
        report = selected_quota(**quota_options)
    elif args.scenario in (
        "torch-key-door-legal-rl",
        "torch-key-door-legal-action-remap",
        "torch-key-door-legal-interaction-quota",
        "torch-key-door-frozen-interaction-quota",
    ):
        selected = describe_legal_reward_protocol if args.describe else run_legal_reward_pilot
        options = {"action_remapping": True} if args.scenario.endswith("action-remap") else {}
        if args.scenario.endswith("interaction-quota"):
            options = dict(action_remapping=True, fill_interaction_quota=True)
            if args.scenario == "torch-key-door-frozen-interaction-quota":
                options["frozen_recurrent_control"] = True
            if seeded:
                options.update(
                    environment_seed=args.environment_seed,
                    heldout_environment_seed=args.heldout_environment_seed,
                )
        report = selected(condition_on_legal_actions=True, **options)
    elif args.scenario == "torch-key-door-rl":
        report = describe_reward_protocol() if args.describe else run_reward_pilot()
    elif args.scenario == "torch-key-door-learnability":
        options = {"decision_diagnostics": True} if args.probe_decision_diagnostics else {}
        report = (
            describe_learnability_protocol(**options)
            if args.describe
            else run_learnability_pilot(**options)
        )
    elif args.scenario == "torch-key-door-recurrent":
        report = describe_recurrent_protocol() if args.describe else run_recurrent_pilot()
    elif args.scenario == "torch-key-door":
        report = describe_key_door_protocol() if args.describe else run_key_door_pilot()
    elif args.scenario == "torch-categorical-rehearsal":
        report = describe_categorical_protocol() if args.describe else run_categorical_pilot()
    elif args.scenario == "torch-trajectory-rehearsal":
        report = describe_trajectory_protocol() if args.describe else run_trajectory_pilot()
    elif args.scenario == "torch-feature-continuation":
        report = describe_continuation_protocol() if args.describe else run_continuation_pilot()
    elif args.scenario == "torch-feature-rehearsal":
        report = describe_torch_protocol() if args.describe else run_torch_pilot()
    elif args.scenario == "renewal-learnability":
        report = describe_renewal_protocol() if args.describe else run_renewal_pilot()
    else:
        report = describe_protocol() if args.describe else run_pilot()
    encoded = json.dumps(report, allow_nan=False, indent=2)
    limit = (
        8 * 1024**2
        if args.scenario
        in (
            "torch-key-door",
            "torch-key-door-recurrent",
            "torch-key-door-learnability",
            "torch-key-door-rl",
            "torch-key-door-legal-rl",
            "torch-key-door-warm-start-rl",
            "torch-key-door-legal-action-remap",
            "torch-key-door-legal-interaction-quota",
            "torch-key-door-frozen-interaction-quota",
            "torch-key-door-paired-interaction-quota",
            "torch-key-door-repeated-reward-control",
            "torch-key-door-acquired-interaction-quota",
            "torch-key-door-quota-probe",
            "torch-key-door-warm-start-control",
            "torch-key-door-public-history-train",
            "torch-key-door-public-history-probe",
            "torch-key-door-public-history-live",
            "key-door-action-memory",
            "key-door-action-memory-probe",
            "torch-key-door-legal-warm-start-control",
            "torch-key-door-bc",
            "torch-key-door-weighted-bc",
            "torch-key-door-train-fit",
            "torch-key-door-train-live",
            "torch-key-door-weighted-train-live",
            "torch-key-door-fixed-a-exposure",
            "torch-key-door-acquired-return",
            "torch-key-door-acquired-recent48-replay",
            "torch-key-door-acquired-coverage-replay",
            "torch-key-door-train-return",
            "torch-key-door-train-four-phase",
            "torch-key-door-four-phase-policy",
            "torch-key-door-four-phase-recent",
            "torch-key-door-four-phase-partitioned",
            "torch-key-door-four-phase-recent48",
            "torch-key-door-four-phase-partitioned48",
            "torch-key-door-four-phase-coverage",
            "torch-key-door-four-phase-recent48-probe",
            "torch-key-door-four-phase-partitioned48-probe",
            "torch-key-door-four-phase-coverage-probe",
            "torch-key-door-return-policy",
            "torch-key-door-return-replay",
            "torch-key-door-return-probe",
            "torch-key-door-return-replay-probe",
        )
        else (
            2 * 1024**2
            if args.scenario
            in (
                "torch-feature-rehearsal",
                "torch-feature-continuation",
                "torch-trajectory-rehearsal",
                "torch-trajectory-rollouts",
                "torch-categorical-rehearsal",
            )
            else 32 * 1024**2
        )
    )
    if args.scenario == "torch-minigrid-cue-key-door":
        # Full148-coordinate live records are larger than the legacy D8 task.
        limit = 64 * 1024**2
    if len(encoded.encode("utf-8")) > limit:
        raise RuntimeError("pilot report exceeds its fixed output allowance")
    print(encoded)
    if not args.describe and report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
