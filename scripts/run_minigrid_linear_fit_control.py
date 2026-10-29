"""Read saved TRAIN evidence for bounded fit diagnostics or frozen affine games."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from typing import Any

from src.adapters.minigrid_linear_fit_control import (
    LinearFitAdmissionFailed,
    describe_protocol,
    diagnose_train_linear_fit,
    describe_context_linear_protocol,
    diagnose_context_train_linear_fit,
)
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.reference_controller import ReferencePolicy
from src.adapters.minigrid_probe_support import (
    describe_train_fit_bounds,
    summarize_train_fit_bounds,
    describe_context_train_bounds,
    summarize_context_train_bounds,
)
from src.adapters.minigrid_constructive_fit import (
    describe_protocol as describe_constructive_protocol,
    diagnose_train_fit,
)
from src.core.minigrid_public_history import encode_public_history_episode
from src.adapters import minigrid_context_control as context_control
from src.adapters.minigrid_context_linear_policy import (
    ContextLinearReader,
    snapshot_from_context_fit_report,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.process_memory import ProcessRssSampler
import numpy as np
import sys

QUALIFIED_CONTEXT_FIT_SHA = "7f9b01ecf732722d8f19dea4c3662c797846839ebe458c2761244ba14227dc94"
QUALIFIED_CONTEXT_STREAM_SHA = "ff534a539fa39d9f2d70c87c59141e29cfacbcaabd61b0cd646f80eabce7fe56"
QUALIFIED_CONTEXT_CATALOGUE_SHA = "4a5d4e67b295efc31b13f7a0cae31bb4aae9296c566eb105c120a3726d18130b"
QUALIFIED_CAPACITY_TRAIN_SHA = "4b29e1de792765fead9259ef1c80445b3ba083492a489f200a9fb2ac582dda28"
QUALIFIED_AFTER_C_SHA = "05ee6741180dcebe52598c5e90e1566e1c3a17c83951db25f1e257e4d4c9a1b6"


def _qualified_capacity_train_bytes(path):
    with Path(path).open("rb") as source:
        raw = source.read(3 * 1024**2 + 1)
    if len(raw) > 3 * 1024**2 or sha256(raw).hexdigest() != QUALIFIED_CAPACITY_TRAIN_SHA:
        raise ValueError("exact original TRAIN source required for the saved after-B model")
    return raw


def run_saved_context_linear_new_c(path, retained_path):
    from src.adapters.minigrid_catalogue_capacity import compare_new_c_capacity

    raw = _qualified_context_catalogue_bytes(retained_path)
    saved = _causal_affine_states(json.loads(raw), retained_catalogue=True)[1]
    original = _qualified_capacity_train_bytes(path)
    episodes = read_training_sources(Path(path), expected_sha256=QUALIFIED_CAPACITY_TRAIN_SHA)

    def guard():
        if _qualified_context_catalogue_bytes(retained_path) != raw:
            raise ValueError("new-C original retained evidence changed")
        if _qualified_capacity_train_bytes(path) != original:
            raise ValueError("new-C original TRAIN evidence changed")

    guard()
    result = compare_new_c_capacity(episodes, saved, check_boundary=guard)
    result["source_bindings"] = dict(
        retained_sha256=QUALIFIED_CONTEXT_CATALOGUE_SHA,
        retained_bytes=len(raw),
        training_sha256=QUALIFIED_CAPACITY_TRAIN_SHA,
        training_bytes=len(original),
    )
    return result


def run_saved_context_linear_probe(path, retained_path):
    from src.adapters.minigrid_affine_probe import run_affine_fresh_probe

    raw = _qualified_after_c_bytes(retained_path)
    saved = _after_c_affine_state(json.loads(raw))
    original = _qualified_capacity_train_bytes(path)
    episodes = read_training_sources(Path(path), expected_sha256=QUALIFIED_CAPACITY_TRAIN_SHA)

    def guard():
        if _qualified_after_c_bytes(retained_path) != raw:
            raise ValueError("affine probe original after-C evidence changed")
        if _qualified_capacity_train_bytes(path) != original:
            raise ValueError("affine probe readonly TRAIN source changed")

    guard()
    result = run_affine_fresh_probe(episodes, saved, check_boundary=guard)
    result["source_bindings"] = dict(
        after_c_sha256=QUALIFIED_AFTER_C_SHA,
        after_c_bytes=len(raw),
        training_sha256=QUALIFIED_CAPACITY_TRAIN_SHA,
        training_bytes=len(original),
    )
    return result


def describe_context_linear_causal_games():
    return dict(
        definition="minigrid_saved_causal_affine_games_v1",
        source_sha256=QUALIFIED_CONTEXT_STREAM_SHA,
        source_fit_calls=6,
        source_fit_rows=54,
        checkpoint_updates=(2, 4, 6),
        live_plan="A blue/red afterA; B blue/red afterB; A blue/red afterreturnA; seed17,max16steps each",
        new_fit_calls=0,
        new_optimizer_updates=0,
        max_environment_steps=96,
        max_readout_work_seconds=140,
        wall_seconds=150,
        peak_rss_bytes=512 * 1024**2,
        qualification="known current TRAIN phase games from causal fits; no native old-task retention/generalization/CPC proof",
    )


def _qualified_context_stream_bytes(path):
    with Path(path).open("rb") as source:
        raw = source.read(1024**2 + 1)
    if len(raw) > 1024**2 or sha256(raw).hexdigest() != QUALIFIED_CONTEXT_STREAM_SHA:
        raise ValueError("exact qualified107 causal TRAIN stream report required")
    return raw


def describe_context_linear_retained_games():
    return dict(
        describe_context_linear_causal_games(),
        definition="minigrid_saved_retained_affine_games_v1",
        source_sha256=QUALIFIED_CONTEXT_CATALOGUE_SHA,
        source_fit_rows=90,
        checkpoint_updates=(4,),
        checkpoint_fit_rows=54,
        checkpoint_arrival_row_exposures=36,
        checkpoint_replay_row_exposures=18,
        live_plan="one after-B retained state4; OLD A blue/red then NEW B blue/red; seed17,max16steps each",
        max_environment_steps=64,
        qualification="previously trained A and new B performance afterB; no before-B native comparison/return/generalization/CPC proof",
    )


def _qualified_context_catalogue_bytes(path):
    with Path(path).open("rb") as source:
        raw = source.read(1024**2 + 1)
    if len(raw) > 1024**2 or sha256(raw).hexdigest() != QUALIFIED_CONTEXT_CATALOGUE_SHA:
        raise ValueError("exact qualified109 retained TRAIN catalogue report required")
    return raw


def describe_context_linear_after_c_games():
    return dict(
        describe_context_linear_retained_games(),
        definition="minigrid_saved_after_c_affine_games_v1",
        source_sha256=QUALIFIED_AFTER_C_SHA,
        source_fit_calls=10,
        source_fit_rows=180,
        original_source_paid_fit_calls=6,
        original_source_paid_fit_rows=90,
        capacity_study_paid_fit_calls=4,
        capacity_study_paid_fit_rows=90,
        checkpoint_updates=(6,),
        checkpoint_fit_rows=108,
        checkpoint_arrival_row_exposures=54,
        checkpoint_replay_row_exposures=54,
        live_plan="one after-C capacity2 state6; B blue/red then C_train blue/red; seed17,max16steps each",
        qualification="known TRAIN B/C afterC usefulness; no before-C native comparison/generalization/reliability/CPC proof",
    )


def _qualified_after_c_bytes(path):
    with Path(path).open("rb") as source:
        raw = source.read(1024**2 + 1)
    if len(raw) > 1024**2 or sha256(raw).hexdigest() != QUALIFIED_AFTER_C_SHA:
        raise ValueError("exact qualified111 after-C TRAIN capacity report required")
    return raw


def _context_linear_training_snapshot(saved):
    """Ordinary dataclass conversion shared by the qualified saved-state modes."""
    from src.adapters.minigrid_context_linear_policy import ContextLinearPolicy
    from src.adapters.minigrid_context_linear_learner import (
        ContextLinearTrainingSnapshot,
        copy_context_linear_training_snapshot,
    )
    from src.core.categorical_linear_fit import CategoricalLinearFit

    fields = dict(saved)
    fields["policy"] = ContextLinearPolicy(**fields["policy"])
    heads: list[CategoricalLinearFit | None] = []
    for head in fields["heads"]:
        if head is None:
            heads.append(None)
            continue
        model = dict(head)
        model["coefficients"] = tuple(tuple(row) for row in model["coefficients"])
        model["singular_values"] = tuple(model["singular_values"])
        heads.append(CategoricalLinearFit(**model))
    fields["heads"] = tuple(heads)
    return copy_context_linear_training_snapshot(ContextLinearTrainingSnapshot(**fields))


def _after_c_affine_state(report):
    if (
        report["definition"] != "minigrid_after_b_new_c_capacity_v1"
        or report["status"] != "complete"
        or report["training_complete"] is not True
        or report["source_unchanged"] is not True
        or report["budget"]["updates_completed"] != 4
        or report["budget"]["replay_examples_completed"] != 0
        or report["initial_study_limits"] != report["budget"]["limits"]
        or report["initial_study_limits"]
        != dict(
            max_training_updates=4,
            max_wall_seconds=90,
            max_replay_examples=None,
            max_hidden_width=None,
            max_process_rss_bytes=512 * 1024**2,
        )
        or report["budget"]["ended_at"] < report["budget"]["started_at"]
        or not 0 <= report["budget"]["elapsed_seconds"] <= 90
        or report["source_bindings"]
        != dict(
            retained_sha256=QUALIFIED_CONTEXT_CATALOGUE_SHA,
            retained_bytes=522500,
            training_sha256=QUALIFIED_CAPACITY_TRAIN_SHA,
            training_bytes=2346812,
        )
        or [a["retained_per_cue"] for a in report["arms"]] != [1, 2]
    ):
        raise ValueError("complete qualified after-C TRAIN capacity study required")
    for index, arm in enumerate(report["arms"]):
        rows = (36, 54)[index]
        if (
            arm["status"] != "complete"
            or arm["source_unchanged"] is not True
            or arm["actor_unchanged"] is not True
            or arm["failure_work"] is not None
            or arm["receipt_capture_error"] is not None
            or arm["stream"]["evaluation_scope"] != "training_only"
            or arm["stream"]["completed_updates"] != 2
            or arm["stream"]["training_examples"] != rows
            or arm["catalogue"]["failure_work_may_be_unobserved"] is not False
            or arm["catalogue"]["committed_rows"]
            != dict(arrivals=18, replay=rows - 18, native=rows)
            or len(arm["committed_receipts"]) != 2
            or arm["stream"]["applied_updates"] != arm["committed_receipts"]
            or any(
                (r["observed_at"], r["arrived_at"], r["applied_at"], r["update_number"])
                != (n, n + 1, n + 1, index * 2 + n + 1)
                for n, r in enumerate(arm["committed_receipts"])
            )
        ):
            raise ValueError("complete original after-C TRAIN arm/rows/receipts required")
    selected = report["arms"][1]["last_checkpoint"]
    state = selected["state"]
    if (
        selected["suffix_updates"] != 2
        or state["completed_updates"] != 6
        or state["training_examples"] != 108
        or state["failed_work"] is not None
        or [head["fit_rows"] for head in state["heads"]] != [27, 27]
    ):
        raise ValueError("after-C capacity2 native6/108/two27-row heads required")
    return _context_linear_training_snapshot(state)


def _causal_affine_states(report, *, retained_catalogue=False):
    if type(retained_catalogue) is not bool:
        raise ValueError("retained_catalogue must be an exact bool")
    if retained_catalogue and (
        report["definition"] != "minigrid_arrived_cue_catalogue_train_stream_v1"
        or report["catalogue"]["committed_rows"] != dict(arrivals=54, replay=36, native=90)
        or report["catalogue"]["failure_work_may_be_unobserved"] is not False
    ):
        raise ValueError("complete retained TRAIN catalogue ledger required before games")
    if (
        report["status"] != "complete"
        or report["source_unchanged"] is not True
        or report["actor_unchanged"] is not True
        or report["failure_work"] is not None
        or report["receipt_capture_error"] is not None
        or report["stream"]["evaluation_scope"] != "training_only"
        or report["stream"]["completed_updates"] != 6
        or report["stream"]["training_examples"] != (90 if retained_catalogue else 54)
        or [e["updates"] for e in report["endpoints"]] != [0, 2, 4, 6]
    ):
        raise ValueError("complete qualified causal TRAIN stream required before games")
    states = []
    for entry in report["endpoints"][1:]:
        state = _context_linear_training_snapshot(entry["state"])
        if state.completed_updates != entry["updates"] or state.training_examples != (
            {2: 18, 4: 54, 6: 90}[entry["updates"]] if retained_catalogue else entry["updates"] * 9
        ):
            raise ValueError("causal snapshot fit/row counters differ from original endpoint")
        states.append(state)
    return tuple(states)


def run_saved_context_linear_rehearsal(path, retained_path):
    from scripts.minigrid_affine_probe_games import _qualified_probe_bytes, _after_probe_state
    from src.adapters.minigrid_affine_rehearsal import compare_private_rehearsal
    from src.infra.minigrid_affine_example import example_affine_training

    bundled_sha = "5a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508"
    try:
        before_raw = _qualified_after_c_bytes(retained_path)
        probe_raw = _qualified_probe_bytes(path)
        before, probe = json.loads(before_raw), json.loads(probe_raw)
        state = _after_c_affine_state(before)
        _after_probe_state(probe)  # Original complete release/charges, not another fit.
        records = probe["probe_data"]
        if type(records) is not list or len(records) != 4:
            raise ValueError("qualified113 selected two-train/two-test sources required")
        selected = []
        for record in records:
            if (
                type(record) is not dict
                or set(record) != {"episode_id", "context", "role", "transitions", "num_classes"}
                or type(record["transitions"]) is not list
                or len(record["transitions"]) != 9
            ):
                raise ValueError("selected113 source requires exact complete episode fields")
            selected.append(
                DemonstrationEpisode(
                    **(
                        record
                        | {"transitions": tuple(_transition(t) for t in record["transitions"])}
                    )
                )
            )
        with example_affine_training() as data_path:
            episodes = read_training_sources(data_path, expected_sha256=bundled_sha)

        def guard():
            if (
                _qualified_after_c_bytes(retained_path) != before_raw
                or _qualified_probe_bytes(path) != probe_raw
            ):
                raise ValueError("original after-C/selected probe evidence changed")

        result = compare_private_rehearsal(episodes, state, tuple(selected), check_boundary=guard)
        result["source_bindings"] = dict(
            after_c_sha256=QUALIFIED_AFTER_C_SHA,
            after_c_bytes=len(before_raw),
            probe_sha256=sha256(probe_raw).hexdigest(),
            probe_bytes=len(probe_raw),
            TRAIN_asset_sha256=bundled_sha,
            TRAIN_asset_bytes=94705,
        )
        result["original_probe_budget"] = deepcopy(probe["budget"])
        result["original_probe_source_fork_calls"] = probe["source_fork_calls"]
        return result
    except Exception as error:
        return dict(
            definition="minigrid_private_affine_rehearsal_comparison_v1",
            status="failed",
            reason=f"{type(error).__name__}: {error}"[:512],
            arms=[],
        )


def run_saved_context_linear_causal_games(
    path, *, retained_catalogue=False, after_c_capacity=False
):
    from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
    from src.adapters.minigrid_policy_evaluation import (
        evaluate_context_linear_stream_checkpoint,
        MiniGridEvaluationFailed,
    )
    from src.app.coupled_attribution import _same_value
    from src.app.probe_return_phase import budget_binding

    if (
        type(retained_catalogue) is not bool
        or type(after_c_capacity) is not bool
        or (retained_catalogue and after_c_capacity)
    ):
        raise ValueError("exact mutually exclusive retained/after-C flags required")
    protocol = (
        describe_context_linear_after_c_games()
        if after_c_capacity
        else describe_context_linear_retained_games()
        if retained_catalogue
        else describe_context_linear_causal_games()
    )
    read_source = (
        _qualified_after_c_bytes
        if after_c_capacity
        else _qualified_context_catalogue_bytes
        if retained_catalogue
        else _qualified_context_stream_bytes
    )
    limits = ToyExecutionBudget(
        max_training_updates=0,
        max_replay_examples=0,
        max_wall_seconds=150,
        max_process_rss_bytes=512 * 1024**2,
    )
    report: dict[str, Any] = dict(
        definition=protocol["definition"],
        protocol=protocol,
        status="running",
        reason=None,
        phases=[],
        source_unchanged=None,
        initial_readout_limits=asdict(limits),
    )
    phases = report["phases"]
    preceding: list[dict[str, Any]] = []
    pending = None
    started = monotonic()
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(limits, monotonic)
        budget.attach_memory(memory)
        binding, expected_limits = budget_binding(budget), replace(limits)
        try:
            raw = read_source(path)
            if after_c_capacity:
                states = (_after_c_affine_state(json.loads(raw)),)
            elif retained_catalogue:
                # Why: state4 predates the return packets; loading the source
                # report grants no future-fit data to the selected scorer.
                states = (_causal_affine_states(json.loads(raw), retained_catalogue=True)[1],)
            else:
                states = _causal_affine_states(json.loads(raw))
            source = ContextLinearLearner()
            initial = source.snapshot_state()

            def guard():
                # Why: the outer150s cap includes cleanup and serialization.
                # Stop work early enough to preserve an honest failed prefix.
                if monotonic() - started >= 140:
                    raise TimeoutError("causal game work deadline reached; cleanup time reserved")
                if (
                    budget_binding(budget) != binding
                    or not _same_value(vars(budget.budget), vars(expected_limits))
                    or not _same_value(
                        (budget.updates_completed, budget.replay_examples_completed), (0, 0)
                    )
                ):
                    raise ValueError("original causal game lifetime budget changed")
                if read_source(path) != raw:
                    raise ValueError("preceding qualified causal TRAIN source changed")
                if report["phases"] is not phases or not _same_value(phases, preceding):
                    raise ValueError("preceding causal game result changed or was replaced")
                if source.snapshot_state() != initial:
                    raise ValueError("causal game untrained source changed")

            for state in states:
                guard()
                try:
                    if after_c_capacity:
                        phase = evaluate_context_linear_stream_checkpoint(
                            source,
                            state,
                            budget=budget,
                            check_boundary=guard,
                            after_c_capacity=True,
                        )
                    elif retained_catalogue:
                        phase = evaluate_context_linear_stream_checkpoint(
                            source,
                            state,
                            budget=budget,
                            check_boundary=guard,
                            retained_catalogue=True,
                        )
                    else:
                        phase = evaluate_context_linear_stream_checkpoint(
                            source, state, budget=budget, check_boundary=guard
                        )
                except MiniGridEvaluationFailed as error:
                    pending = error.report
                    raise
                pending = phase
                if (
                    phase.get("status") != "complete"
                    or phase.get("state_unchanged") is not True
                    or len(phase.get("episodes", []))
                    != (4 if retained_catalogue or after_c_capacity else 2)
                    or any(
                        e.get("status") != "complete"
                        or e.get("environment", {}).get("closed") is not True
                        for e in phase["episodes"]
                    )
                ):
                    raise ValueError("preceding causal game phase is incomplete")
                guard()
                phases.append(phase)
                preceding.append(deepcopy(phase))
                pending = None
            guard()
            budget.before_final()
            report.update(status="complete", source_unchanged=True)
        except Exception as error:
            report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
            # Why: a later callback cannot rewrite already observed successes.
            # Keep the detached known prefix and any paid current-phase report.
            report["phases"] = deepcopy(preceding) + (
                [] if pending is None else [deepcopy(pending)]
            )
        finally:
            report["budget"] = dict(
                limits=asdict(budget.budget),
                updates_completed=budget.updates_completed,
                replay_examples_completed=budget.replay_examples_completed,
                started_at=budget.started_at,
                ended_at=budget.clock(),
                elapsed_seconds=monotonic() - started,
                sampled_peak_rss_bytes=memory.peak_bytes,
            )
    return report


def describe_context_linear_stream():
    return dict(
        definition="minigrid_arrived_context_linear_train_stream_v1",
        schedule="A blue, A red, B blue, B red, A blue, A red; one complete9-row episode/update",
        label_delay=1,
        checkpoint_updates=(0, 2, 4, 6),
        fit_calls=6,
        fit_rows=54,
        score_calls=16,
        score_rows=144,
        classes=7,
        seed=None,
        wall_seconds=90,
        peak_rss_bytes=512 * 1024**2,
        qualification="TRAIN-only causal fit/forgetting/return; no gameplay, final or probe evaluation",
    )


def describe_context_linear_reference(
    policy: ReferencePolicy,
    *,
    prequential_feedback=False,
    ready_feedback=False,
    retain_arrived=False,
    a4_training=False,
    acknowledge_correct=False,
):
    if type(acknowledge_correct) is not bool or (
        acknowledge_correct and (not ready_feedback or prequential_feedback or a4_training)
    ):
        raise ValueError("correct acknowledgement requires ordinary ready-feedback references")
    if (
        type(policy) is not ReferencePolicy
        or type(prequential_feedback) is not bool
        or type(ready_feedback) is not bool
        or type(retain_arrived) is not bool
        or type(a4_training) is not bool
        or (a4_training and not retain_arrived)
        or (prequential_feedback and ready_feedback)
        or (policy.mode == "drift" and not (prequential_feedback or ready_feedback))
    ):
        raise ValueError("affine reference requires an exact supported fixed ReferencePolicy")
    policy.__post_init__()
    protocol = describe_context_linear_stream()
    del protocol["fit_calls"], protocol["fit_rows"]
    result = dict(
        protocol,
        definition="minigrid_context_linear_reference_window_v1",
        reference_policy=asdict(policy),
        checkpoint_updates=tuple(range(7)),
        window_opportunities=7,
        fit_calls_upper_bound=6,
        fit_rows_upper_bound=54,
        arrival_rows=54,
        score_calls=0,
        score_rows=0,
        game_episodes=0,
        qualification="finite ordinary TRAIN window; pending is not completed training; no automatic scores/games/final/probe/retention claim",
    )
    if prequential_feedback:
        result.update(
            prequential_feedback=True,
            diagnostic_forks_upper_bound=6,
            diagnostic_prediction_calls_upper_bound=6,
            diagnostic_prediction_rows_upper_bound=54,
            feedback_version_policy="retire changed candidate version/update errors without a wake vote",
        )
    if ready_feedback:
        result.update(
            ready_feedback=True,
            diagnostic_forks_upper_bound=6,
            diagnostic_prediction_calls_upper_bound=6,
            diagnostic_prediction_rows_upper_bound=54,
            feedback_version_policy="current candidate and oldest eligible FIFO TRAIN packet after label arrival",
        )
    if retain_arrived:
        result.update(
            definition="minigrid_arrived_catalogue_reference_window_v1",
            retain_arrived=True,
            fit_rows_upper_bound=90,
            replay_rows_upper_bound=36,
            bank_policy="one prior arrived TRAIN episode per public cue, jointly fitted only on selected wakes",
            prepared_input_array_bytes=90 * 297 * 8,
            planned_peak_bank_rows=18,
        )
        if prequential_feedback or ready_feedback:
            # Why: a waiting ready control can repeatedly diagnose an18-row
            # oldest packet. This is a bound, not an executed work count.
            result["diagnostic_prediction_rows_upper_bound"] = 108 if ready_feedback else 90
    if a4_training:
        result.update(
            definition="minigrid_a3_a4_arrived_reference_window_v1",
            schedule="original A3 blue/red, fresh A4 TRAIN blue/red, returned A3 blue/red",
            checkpoint_export="actual completed prefixes only, unscored; logical observations2/4/6",
        )
    if acknowledge_correct:
        proof_rows = result["diagnostic_prediction_rows_upper_bound"]
        result.update(
            acknowledge_correct=True,
            acknowledgement_proof_forks_upper_bound=6,
            acknowledgement_proof_prediction_calls_upper_bound=6,
            acknowledgement_proof_prediction_rows_upper_bound=proof_rows,
            diagnostic_forks_upper_bound=12,
            diagnostic_prediction_calls_upper_bound=12,
            diagnostic_prediction_rows_upper_bound=2 * proof_rows,
            consumption="one fit OR fresh proof-bound correct acknowledgement per opportunity; consumed is not all fitted",
        )
    return result


def describe_context_linear_catalogue():
    return dict(
        describe_context_linear_stream(),
        definition="minigrid_arrived_cue_catalogue_train_stream_v1",
        fit_rows=90,
        arrival_row_exposures=54,
        catalogue_replay_row_exposures=36,
        distinct_source_rows=36,
        bank_policy="one previous arrived TRAIN episode per initial public cue; joint old/current fit",
        qualification="TRAIN-only acquisition/retention/return with extra replay exposures; no matched-compute/game/CPC claim",
    )


def describe_context_linear_abc():
    return dict(
        describe_context_linear_catalogue(),
        definition="minigrid_arrived_abc_capacity2_train_stream_v1",
        schedule="A blue/red, B blue/red, C_train blue/red; fixed six arrivals",
        fit_rows=108,
        catalogue_replay_row_exposures=54,
        distinct_source_rows=54,
        score_calls=24,
        score_rows=216,
        bank_policy="two previous arrived TRAIN episodes per initial public cue; joint old/current fit",
        qualification="known TRAIN ABC only; extra replay exposures; no generalization/CPC claim",
    )


def describe_context_linear_recurring():
    return dict(
        describe_context_linear_catalogue(),
        definition="minigrid_arrived_recurring_capacity3_train_stream_v1",
        schedule="A blue/red, B blue/red, C_train blue/red, G_train blue/red, A blue/red; ten arrivals",
        checkpoint_updates=(0, 2, 4, 6, 8, 10),
        fit_calls=10,
        fit_rows=252,
        arrival_row_exposures=90,
        catalogue_replay_row_exposures=162,
        distinct_source_rows=72,
        score_calls=48,
        score_rows=432,
        bank_policy="three previous arrived TRAIN episodes per initial public cue; joint old/current fit",
        qualification="known TRAIN acquisition/retention/return; no native/generalization/matched-compute/CPC claim",
    )


def describe_context_linear_a4():
    return dict(
        describe_context_linear_catalogue(),
        definition="minigrid_affine_a3_a4_a3_train_v1",
        schedule="original A3 blue/red, fresh A4 TRAIN blue/red, returned A3 blue/red",
        games="A3 then A4 blue/red at each2/4/6 checkpoint; seed17/max16/K7firstargmax",
        qualification="known TRAIN acquisition/retention/return; recovery only after loss/regain; no generalization/matchedcompute/CPC claim",
    )


def describe_context_linear_a4_current_only():
    return dict(
        describe_context_linear_stream(),
        definition="minigrid_affine_a3_a4_a3_current_only_v1",
        schedule=describe_context_linear_a4()["schedule"],
        games=describe_context_linear_a4()["games"],
        arrival_row_exposures=54,
        catalogue_replay_row_exposures=0,
        distinct_source_rows=36,
        bank_policy="current arrived TRAIN episode only; no joint replay",
        qualification="same-data current-only control; recovery requires loss/regain; unequal54vs90exposures is not matchedcompute/CPC benefit",
    )


def describe_context_linear_a4_current_repeated():
    return dict(
        describe_context_linear_a4_current_only(),
        definition="minigrid_affine_a3_a4_a3_current_repeated_v1",
        fit_rows=90,
        repeated_current_row_exposures=36,
        current_packet_multiplicities=(1, 1, 2, 2, 2, 2),
        bank_policy="current arrived TRAIN packet only, repeated within wake; no prior-packet bank",
        qualification="six fits/90 row exposures match joint batch shapes; not matched wall/FLOPs, native usefulness or CPC benefit",
    )


def run_context_linear_stream(
    episodes,
    *,
    retain_arrived=False,
    abc_catalogue=False,
    endpoint_observer=None,
    check_boundary=None,
    recurring_catalogue=False,
    a4_training=False,
    a4_current_only=False,
    a4_current_repeated=False,
    reference_policy: ReferencePolicy | None = None,
    prequential_feedback=False,
    ready_feedback=False,
    shift_diagnostics=False,
    acknowledge_correct=False,
):
    from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
    from src.adapters.minigrid_train_catalogue import (
        _validate_sources,
        compose_arrived_cue_catalogue,
        compose_repeated_current_training,
    )
    from src.app.actor_shadow import StableActor
    from src.app.coupled_attribution import AttributionBatch, AttributionStream, _same_value
    from src.core.categorical_probability_scores import probability_scores
    from src.core.demonstration_episode import episode_arrays
    from src.core.learner_ports import TrainingUpdateFailed

    reference_protocol = None
    policy_pin = None
    if type(acknowledge_correct) is not bool or (
        acknowledge_correct
        and (reference_policy is None or not ready_feedback or prequential_feedback or a4_training)
    ):
        raise ValueError("correct acknowledgement requires ordinary ready-feedback references")
    if type(shift_diagnostics) is not bool or (shift_diagnostics and reference_policy is None):
        raise ValueError("shift_diagnostics requires an exact bool and the native reference lane")
    if (
        type(prequential_feedback) is not bool
        or type(ready_feedback) is not bool
        or (prequential_feedback and ready_feedback)
        or ((prequential_feedback or ready_feedback) and reference_policy is None)
    ):
        raise ValueError("prequential feedback requires an explicit reference and exact bool flag")
    if reference_policy is not None:
        reference_protocol = describe_context_linear_reference(
            reference_policy,
            prequential_feedback=prequential_feedback,
            ready_feedback=ready_feedback,
            retain_arrived=retain_arrived,
            a4_training=a4_training,
            **(dict(acknowledge_correct=True) if acknowledge_correct else {}),
        )
        policy_pin = deepcopy(reference_policy)
        if (
            any(
                (
                    abc_catalogue,
                    recurring_catalogue,
                    a4_current_only,
                    a4_current_repeated,
                )
            )
            or endpoint_observer is not None
        ):
            raise ValueError(
                "reference window requires its distinct A/B or joint A4 route without endpoint handoff"
            )
    if any(
        type(flag) is not bool
        for flag in (
            retain_arrived,
            abc_catalogue,
            recurring_catalogue,
            a4_training,
            a4_current_only,
            a4_current_repeated,
        )
    ):
        raise ValueError("catalogue flags must be exact bools")
    if a4_training and (not retain_arrived or abc_catalogue or recurring_catalogue):
        raise ValueError("A4 TRAIN requires retention and a distinct mode")
    if a4_current_only and (a4_training or retain_arrived or abc_catalogue or recurring_catalogue):
        raise ValueError("current-only A4 requires a distinct nonretained mode")
    if a4_current_repeated and any(
        (a4_current_only, a4_training, retain_arrived, abc_catalogue, recurring_catalogue)
    ):
        raise ValueError("repeated-current A4 requires a distinct nonretained mode")
    a4_mode = a4_training or a4_current_only or a4_current_repeated
    if abc_catalogue and not retain_arrived:
        raise ValueError("ABC requires arrived catalogue retention")
    if recurring_catalogue and (not retain_arrived or abc_catalogue):
        raise ValueError("recurring catalogue requires retention and a distinct mode from ABC")
    if endpoint_observer is not None and not callable(endpoint_observer):
        raise ValueError("endpoint observer must be callable")
    if check_boundary is not None:
        if not callable(check_boundary):
            raise ValueError("outer TRAIN boundary must be callable")
        check_boundary()
        if reference_policy is not None and reference_policy != policy_pin:
            raise ValueError("original affine reference policy changed")
    contexts = (
        ("A", "A4-train")
        if a4_mode
        else ("A", "B", "C_train", "G_train")
        if recurring_catalogue
        else ("A", "B", "C_train")
        if abc_catalogue
        else ("A", "B")
    )
    expected = tuple(
        f"minigrid-source-{context}-{cue}" for context in contexts for cue in ("blue", "red")
    )
    if type(episodes) is not tuple or len(episodes) not in (
        (4,) if a4_mode else (8,) if recurring_catalogue else (4, 6)
    ):
        raise ValueError("fixed affine stream requires original A/B TRAIN source pairs")
    selected = episodes if abc_catalogue or recurring_catalogue else episodes[:4]
    if tuple(e.episode_id for e in selected) != expected or any(
        len(e.transitions) != 9 for e in selected
    ):
        raise ValueError("fixed affine stream requires ordered original nine-step A/B source IDs")
    if a4_mode and any(e.context != "A" for e in selected):
        raise ValueError("A3/A4 TRAIN sources retain original A environment context")
    # Why: this explicit finite extension shares the existing owner and stream;
    # it never converts private probe records or widens the six-wake defaults.
    update_count = 10 if recurring_catalogue else 6
    witness, _ = _validate_sources(selected, update_count)
    plan = (
        (*range(8), 0, 1)
        if recurring_catalogue
        else tuple(range(6))
        if abc_catalogue
        else (0, 1, 2, 3, 0, 1)
    )
    training = []
    scheduled = []
    for i, index in enumerate(plan):
        episode = replace(selected[index], episode_id=f"{expected[index]}:wake:{i}")
        if retain_arrived or a4_current_repeated:
            scheduled.append(episode)
            continue
        x, y = episode_arrays(episode)
        training.append(
            AttributionBatch(
                "train",
                episode.context,
                tuple(f"{episode.episode_id}:step:{t.step}" for t in episode.transitions),
                x,
                y,
                episode,
                7,
            )
        )
    catalogue = None
    repetition = None
    if a4_current_repeated:
        repetition = compose_repeated_current_training(tuple(scheduled))
        training = list(repetition.training)
    if retain_arrived:
        catalogue = (
            compose_arrived_cue_catalogue(tuple(scheduled), retained_per_cue=3, max_arrivals=10)
            if recurring_catalogue
            else compose_arrived_cue_catalogue(
                tuple(scheduled), retained_per_cue=2 if abc_catalogue else 1
            )
        )
        if tuple(wake.cue for wake in catalogue.wakes) != ("blue", "red") * (update_count // 2):
            raise ValueError("fixed catalogue protocol requires alternating public blue/red cues")
        training = list(catalogue.training)
    protocol = (
        reference_protocol
        if reference_protocol is not None
        else describe_context_linear_a4_current_repeated()
        if a4_current_repeated
        else describe_context_linear_a4_current_only()
        if a4_current_only
        else describe_context_linear_a4()
        if a4_training
        else describe_context_linear_recurring()
        if recurring_catalogue
        else describe_context_linear_abc()
        if abc_catalogue
        else describe_context_linear_catalogue()
        if retain_arrived
        else describe_context_linear_stream()
    )
    report: dict[str, Any] = dict(
        definition=protocol["definition"],
        protocol=protocol,
        status="running",
        reason=None,
        endpoints=[],
        stream=None,
        failure_work=None,
        source_unchanged=None,
        actor_unchanged=None,
        score_calls_attempted=0,
        score_rows_attempted=0,
        score_calls_completed=0,
        score_rows_completed=0,
    )
    shift = None
    if shift_diagnostics:
        from src.adapters.minigrid_shift_diagnostics import describe as describe_shift

        protocol["shift_diagnostics"] = describe_shift()
    if repetition is not None:
        report["current_repetition"] = dict(
            wakes=[
                dict(asdict(wake), original_episode_id=expected[plan[i]])
                for i, wake in enumerate(repetition.wakes)
            ],
            arrival_row_exposures=repetition.arrival_row_exposures,
            repeated_current_row_exposures=repetition.repeated_current_row_exposures,
            prior_packet_replay_rows=0,
            fit_row_exposures=repetition.fit_row_exposures,
            input_array_bytes=repetition.input_array_bytes,
            retained_bank_rows=0,
            accounting_scope="repeated current rows inside one native wake; no additional arrival or consolidation event",
        )
    if catalogue is not None:
        report["catalogue"] = dict(
            wakes=[asdict(wake) for wake in catalogue.wakes],
            arrival_row_exposures=catalogue.arrival_row_exposures,
            replay_row_exposures=catalogue.replay_row_exposures,
            fit_row_exposures=catalogue.fit_row_exposures,
            input_array_bytes=catalogue.input_array_bytes,
            peak_joint_template_array_bytes=catalogue.peak_joint_template_array_bytes,
            peak_bank_rows=catalogue.peak_bank_rows,
            peak_bank_observation_target_float64_equivalent_bytes=catalogue.peak_bank_observation_target_float64_equivalent_bytes,
            bank_storage_scope="shared source episode references; current+next observations and targets only; reward/metadata/Python overhead excluded; full source/stream copies count in RSS",
            accounting_scope="old rows fitted jointly inside native wake; explicit replay exposures, not separate consolidation-budget events",
        )
        if reference_policy is not None:
            # Why: a finite reference may consume none of this prepared plan.
            # Storage/preparation is paid; planned exposures are not actual fits.
            report["catalogue"] = dict(planned=report["catalogue"])
    limits = ToyExecutionBudget(
        max_training_updates=update_count, max_wall_seconds=90, max_process_rss_bytes=512 * 1024**2
    )
    states: list[tuple[int, Any]] = []
    stream = None
    verified_repeated_receipts: tuple[Any, ...] = ()
    verified_repeated_state = None
    started = monotonic()
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(limits, monotonic)
        budget.attach_memory(memory)
        source = ContextLinearLearner()
        initial = source.snapshot_state()
        actor = StableActor(source.fork(), version="affine-untrained")
        original_actor = actor.snapshot_state()

        def boundary():
            if check_boundary is not None or a4_current_repeated:
                if check_boundary is not None:
                    check_boundary()
                witness.check(selected)
                if source.snapshot_state() != initial or actor.snapshot_state() != original_actor:
                    raise ValueError("outer TRAIN boundary changed source or actor")
            if reference_policy is not None and reference_policy != policy_pin:
                raise ValueError("original affine reference policy changed")

        observed_fit_rows = 0

        def observe(n, state, batch):
            nonlocal observed_fit_rows
            boundary()
            witness.check(selected)
            expected_rows = sum(len(b.features) for b in training[:n])
            if acknowledge_correct:
                # ACKs consume packets without fits: the actual observed batch
                # can follow skipped indices in the original prepared plan.
                if n != len(states) or (n == 0) != (batch is None):
                    raise ValueError("acknowledged window checkpoint order changed")
                if batch is not None and not any(_same_value(batch, item) for item in training):
                    raise ValueError("acknowledged window fitted an unknown TRAIN batch")
                expected_rows = observed_fit_rows + (0 if batch is None else len(batch.features))
            if (catalogue is not None or a4_current_only or a4_current_repeated) and (
                state.completed_updates != n or state.training_examples != expected_rows
            ):
                raise ValueError(
                    "arrived catalogue native fit/row counters disagree with receipt prefix"
                )
            states.append((n, state))
            observed_fit_rows = expected_rows

        def observe_arrived(tick, index, batch, candidate):
            boundary()
            if not _same_value(batch, training[index]):
                raise ValueError("arrived shift template differs from the original TRAIN plan")
            # Joint fitting may contain replay. Diagnose only this newly labelled
            # packet, never older replay rows or a future source's labels.
            episode = replace(
                selected[plan[index]], episode_id=f"{expected[plan[index]]}:wake:{index}"
            )
            x, y = episode_arrays(episode)
            current = AttributionBatch(
                "train",
                episode.context,
                tuple(f"{episode.episode_id}:step:{t.step}" for t in episode.transitions),
                x,
                y,
                episode,
                7,
            )
            assert shift is not None
            shift.observe(tick, index, current, candidate)
            boundary()

        try:
            if shift_diagnostics:
                from src.adapters.minigrid_shift_diagnostics import MiniGridShiftDiagnostics

                shift = MiniGridShiftDiagnostics()
            stream = AttributionStream(
                source,
                tuple(training),
                (),
                training_only=True,
                checkpoint_updates=protocol["checkpoint_updates"],
                label_delay=1,
                budget=budget,
                observe_checkpoint=observe,
                actor=actor,
            )
            if reference_policy is not None:
                # A terminal window can leave pending packets. Do not finish,
                # score or treat partial states as six completed fits. Only the
                # explicit A4 route returns its bounded unscored actual prefixes.
                boundary()
                arrived_options: dict[str, Any] = (
                    dict(observe_arrived=observe_arrived) if shift_diagnostics else {}
                )
                if acknowledge_correct:
                    arrived_options["acknowledge_correct"] = True
                try:
                    report["stream"] = stream.run_reference_window(
                        reference_policy,
                        prequential_feedback=prequential_feedback,
                        ready_feedback=ready_feedback,
                        **arrived_options,
                    )
                finally:
                    if report["stream"] is None:
                        report["stream"] = stream.reference_report
                if report["stream"]["status"] != "complete":
                    raise ValueError("affine reference window capture is incomplete")
                witness.check(selected)
                if source.snapshot_state() != initial or actor.snapshot_state() != original_actor:
                    raise ValueError("affine reference changed source or actor")
                budget.before_final()
                boundary()
                report.update(status="complete", source_unchanged=True, actor_unchanged=True)
                return report
            if a4_current_repeated:
                verified_repeated_state = stream.candidate_snapshot()
                for stop in range(1, update_count + 1):
                    boundary()
                    if not _same_value(
                        stream.applied_updates, verified_repeated_receipts
                    ) or not _same_value(stream.candidate_snapshot(), verified_repeated_state):
                        raise ValueError("preceding repeated-current receipt or candidate changed")
                    stream.run_until(stop)
                    observed = stream.applied_updates
                    current = stream.candidate_snapshot()
                    if (
                        len(observed) != stop
                        or not _same_value(observed[:-1], verified_repeated_receipts)
                        or observed[-1].sample_id != f"batch-{stop - 1}"
                        or (
                            observed[-1].observed_at,
                            observed[-1].arrived_at,
                            observed[-1].applied_at,
                            observed[-1].update_number,
                        )
                        != (stop - 1, stop, stop, stop)
                        or current.completed_updates != stop
                        or current.training_examples
                        != sum(len(b.features) for b in training[:stop])
                        or current.failed_work is not None
                    ):
                        raise ValueError("repeated-current update lacks its original paid prefix")
                    verified_repeated_receipts = deepcopy(observed)
                    verified_repeated_state = current
                boundary()
                if not _same_value(
                    stream.applied_updates, verified_repeated_receipts
                ) or not _same_value(stream.candidate_snapshot(), verified_repeated_state):
                    raise ValueError("final repeated-current evidence changed before finish")
            elif check_boundary is None:
                stream.run_until(update_count)
            else:
                # Why: incremental advancement preserves delay-one receipts and
                # checkpoints while checking the whole-workflow deadline.
                for stop in range(1, update_count + 1):
                    boundary()
                    stream.run_until(stop)
                    boundary()
            report["stream"] = stream.finish()
            # Why: all fits finish before any endpoint score. TRAIN scores cannot
            # change the schedule, trigger more updates or select checkpoints.
            reader = ContextLinearLearner()
            for n, state in states:
                boundary()
                budget.before_final()
                witness.check(selected)
                reader.restore_state(state)
                row: dict[str, Any] = dict(updates=n, state=asdict(state), scores={})
                report["endpoints"].append(row)
                for episode in selected:
                    boundary()
                    budget.before_final()
                    x, y = episode_arrays(episode)
                    report["score_calls_attempted"] += 1
                    report["score_rows_attempted"] += len(x)
                    p = reader.predict(x)
                    metrics = probability_scores(p, y, 7)
                    row["scores"][episode.episode_id] = dict(
                        metrics,
                        actions=p.argmax(axis=1).tolist(),
                        expected=y[:, 0].astype(int).tolist(),
                    )
                    report["score_calls_completed"] += 1
                    report["score_rows_completed"] += len(x)
                    if reader.snapshot_state() != state:
                        raise ValueError("affine TRAIN scoring changed checkpoint")
            witness.check(selected)
            if source.snapshot_state() != initial or actor.snapshot_state() != original_actor:
                raise ValueError("affine TRAIN stream changed source or actor")
            budget.before_final()
            boundary()
            report.update(status="complete", source_unchanged=True, actor_unchanged=True)
        except Exception as error:
            report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
            if isinstance(error, TrainingUpdateFailed):
                report["failure_work"] = dict(definition=error.definition, work=error.work)
        finally:
            if shift is not None:
                report["shift_diagnostics"] = shift.report()
            report["observed_checkpoints"] = [n for n, state in states]
            scored = {row["updates"] for row in report["endpoints"]}
            report["unscored_checkpoints"] = (
                [dict(updates=n, state=asdict(state)) for n, state in states if n not in scored]
                if reference_policy is None or a4_training
                else []
            )
            report["committed_receipts"] = None
            report["receipt_capture_error"] = None
            if stream is not None:
                try:
                    captured = stream.applied_updates
                    captured_indices = tuple(range(len(captured)))
                    if acknowledge_correct and catalogue is not None:
                        captured_indices = tuple(r.observed_at for r in captured)
                        if any(
                            type(i) is not int or not 0 <= i < len(catalogue.wakes)
                            for i in captured_indices
                        ) or captured_indices != tuple(sorted(set(captured_indices))):
                            raise ValueError(
                                "acknowledged reference receipts lost actual FIFO indices"
                            )
                    if (
                        catalogue is not None
                        and reference_policy is not None
                        and (
                            len(captured) > len(catalogue.wakes)
                            or report["stream"] is None
                            or not _same_value(
                                [asdict(r) for r in captured],
                                report["stream"].get("applied_updates"),
                            )
                            or any(
                                r.sample_id != f"batch-{packet_index}"
                                or r.update_number != i + 1
                                or r.observed_at != catalogue.wakes[packet_index].observed_at
                                or r.arrived_at != catalogue.wakes[packet_index].arrived_at
                                or r.applied_at < r.arrived_at
                                for i, (r, packet_index) in enumerate(
                                    zip(captured, captured_indices, strict=True)
                                )
                            )
                        )
                    ):
                        raise ValueError(
                            "reference catalogue receipts differ from original FIFO prefix"
                        )
                    if a4_current_repeated and not _same_value(
                        captured, verified_repeated_receipts
                    ):
                        raise ValueError(
                            "repeated-current receipt capture changed or is unobserved"
                        )
                    report["committed_receipts"] = [asdict(r) for r in captured]
                except Exception as error:
                    report["receipt_capture_error"] = f"{type(error).__name__}: {error}"[:512]
                    if report["status"] == "complete":
                        report.update(status="failed", reason="committed receipt capture failed")
            if repetition is not None:
                report["current_repetition"]["verified_receipts"] = [
                    asdict(r) for r in verified_repeated_receipts
                ]
                verified_prefix = repetition.wakes[: len(verified_repeated_receipts)]
                report["current_repetition"]["verified_rows"] = dict(
                    arrivals=sum(w.current_rows for w in verified_prefix),
                    repeated_current=sum(w.repeated_current_rows for w in verified_prefix),
                    native=sum(w.current_rows + w.repeated_current_rows for w in verified_prefix),
                )
                receipts = report["committed_receipts"]
                committed = None if receipts is None else len(receipts)
                prefix = () if committed is None else repetition.wakes[:committed]
                report["current_repetition"]["committed_rows"] = (
                    None
                    if committed is None
                    else dict(
                        arrivals=sum(w.current_rows for w in prefix),
                        repeated_current=sum(w.repeated_current_rows for w in prefix),
                        native=sum(w.current_rows + w.repeated_current_rows for w in prefix),
                    )
                )
                report["current_repetition"]["failed_attempt_repeated_rows_upper_bound"] = (
                    repetition.wakes[committed].repeated_current_rows
                    if report["status"] != "complete"
                    and committed is not None
                    and committed < len(repetition.wakes)
                    else None
                )
                report["current_repetition"]["failure_work_may_be_unobserved"] = (
                    report["status"] != "complete"
                )
            if catalogue is not None:
                receipts = report["committed_receipts"]
                committed = None if receipts is None else len(receipts)
                prefix = () if committed is None else catalogue.wakes[:committed]
                if acknowledge_correct and receipts is not None:
                    indices = [receipt["observed_at"] for receipt in receipts]
                    if any(
                        type(i) is not int or not 0 <= i < len(catalogue.wakes) for i in indices
                    ) or indices != sorted(set(indices)):
                        raise ValueError(
                            "acknowledged catalogue receipts require actual FIFO indices"
                        )
                    prefix = tuple(catalogue.wakes[i] for i in indices)
                report["catalogue"]["committed_rows"] = (
                    None
                    if committed is None
                    else dict(
                        arrivals=sum(w.current_rows for w in prefix),
                        replay=sum(w.replay_rows for w in prefix),
                        native=sum(w.current_rows + w.replay_rows for w in prefix),
                    )
                )
                report["catalogue"]["failed_attempt_replay_rows_upper_bound"] = (
                    (
                        max(w.replay_rows for w in catalogue.wakes)
                        if acknowledge_correct
                        else catalogue.wakes[committed].replay_rows
                    )
                    if report["status"] != "complete"
                    and committed is not None
                    and committed < len(catalogue.wakes)
                    else None
                )
                report["catalogue"]["failure_work_may_be_unobserved"] = (
                    report["status"] != "complete"
                )
                if reference_policy is not None:
                    window = report["stream"]
                    attempts = None if window is None else window.get("attempted_wakes")
                    attempted_prefix = (
                        catalogue.wakes[:attempts]
                        if type(attempts) is int and 0 <= attempts <= len(catalogue.wakes)
                        else None
                    )
                    report["catalogue"]["attempted_rows_upper_bound"] = (
                        None
                        if attempted_prefix is None
                        else dict(
                            arrivals=sum(w.current_rows for w in attempted_prefix),
                            replay=sum(w.replay_rows for w in attempted_prefix),
                            native=sum(w.current_rows + w.replay_rows for w in attempted_prefix),
                        )
                    )
                    if (
                        acknowledge_correct
                        and attempted_prefix is not None
                        and type(attempts) is int
                    ):
                        # A failed call need not reveal its skipped packet index.
                        # Keep actual committed charges exact and the attempted
                        # ceiling conservative, including all possible prior rows.
                        current_max = max(w.current_rows for w in catalogue.wakes)
                        replay_max = max(w.replay_rows for w in catalogue.wakes)
                        report["catalogue"]["attempted_rows_upper_bound"] = dict(
                            arrivals=attempts * current_max,
                            replay=attempts * replay_max,
                            native=attempts * (current_max + replay_max),
                        )
            report["budget"] = dict(
                initial_limits=asdict(limits),
                limits=asdict(budget.budget),
                updates_completed=budget.updates_completed,
                replay_examples_completed=budget.replay_examples_completed,
                started_at=budget.started_at,
                ended_at=budget.clock(),
                elapsed_seconds=monotonic() - started,
                sampled_peak_rss_bytes=memory.peak_bytes,
            )
    # Why: native endpoint handoff follows final receipt/failure capture, never a
    # reconstructed saved-report checkpoint or an incompletely observed fit.
    if report["status"] == "complete" and endpoint_observer is not None:
        if len(report["committed_receipts"]) != update_count or states[-1][0] != update_count:
            required = "10" if recurring_catalogue else "six"
            raise ValueError(f"complete native endpoint requires {required} committed receipts")
        endpoint_observer(states[-1][1])
    return report


def describe_context_linear_games():
    return dict(
        definition="minigrid_saved_context_linear_games_v1",
        source_fit_sha256=QUALIFIED_CONTEXT_FIT_SHA,
        source_fit_calls=2,
        source_fit_rows=54,
        new_fit_calls=0,
        new_optimizer_updates=0,
        live_plan="known TRAIN A blue/red only; seed17; max16steps each; all7actions raw firstargmax",
        wall_seconds=90,
        peak_rss_bytes=512 * 1024**2,
        qualification="frozen offline joint TRAIN fit; no chronological adaptation or generalization proof",
    )


def _qualified_context_fit_bytes(path):
    with Path(path).open("rb") as source:
        raw = source.read(512 * 1024 + 1)
    if len(raw) > 512 * 1024 or sha256(raw).hexdigest() != QUALIFIED_CONTEXT_FIT_SHA:
        raise ValueError("exact qualified full-action-fit105 report required")
    return raw


def run_saved_context_linear_games(path):
    from src.adapters.minigrid_policy_evaluation import (
        evaluate_context_linear_checkpoint,
        MiniGridEvaluationFailed,
    )

    started = monotonic()
    limits = ToyExecutionBudget(
        max_training_updates=0,
        max_replay_examples=0,
        max_wall_seconds=90,
        max_process_rss_bytes=512 * 1024**2,
    )
    report: dict[str, Any] = dict(
        definition=describe_context_linear_games()["definition"],
        status="running",
        reason=None,
        protocol=describe_context_linear_games(),
        source_unchanged=None,
        games=None,
        initial_readout_limits=asdict(limits),
    )
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(limits, monotonic)
        budget.attach_memory(memory)
        try:
            raw = _qualified_context_fit_bytes(path)
            state = snapshot_from_context_fit_report(json.loads(raw))
            source = ContextLinearReader(state)

            def guard():
                if _qualified_context_fit_bytes(path) != raw:
                    raise ValueError("preceding qualified affine fit report changed")

            report["games"] = evaluate_context_linear_checkpoint(
                source,
                state,
                budget=budget,
                check_boundary=guard,
            )
            guard()
            report.update(status="complete", source_unchanged=True)
        except Exception as error:
            if isinstance(error, MiniGridEvaluationFailed):
                report["games"] = error.report
            report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
        finally:
            report["budget"] = dict(
                limits=asdict(budget.budget),
                updates_completed=budget.updates_completed,
                replay_examples_completed=budget.replay_examples_completed,
                started_at=budget.started_at,
                last_clock=budget.last_clock,
            )
            report["elapsed_seconds"] = monotonic() - started
            report["memory"] = asdict(memory.snapshot())
    return report


def _transition(record: Any) -> DemonstrationTransition:
    if (
        type(record) is not dict
        or set(record)
        != {
            "step",
            "observation",
            "action",
            "reward",
            "next_observation",
            "terminal",
            "num_classes",
        }
        or any(
            type(record[name]) is not list or len(record[name]) != 148
            for name in ("observation", "next_observation")
        )
    ):
        raise ValueError("linear source requires complete original148-frame transition records")
    owned: dict[str, Any] = {
        name: tuple(value) if name in ("observation", "next_observation") else value
        for name, value in record.items()
    }
    return DemonstrationTransition(**owned)


def read_training_sources(
    path: Path, *, expected_sha256=None, raw_frames: bool = False
) -> tuple[DemonstrationEpisode, ...]:
    """Decode only original148-frame TRAIN sources, never evaluation sections."""
    if type(raw_frames) is not bool:
        raise ValueError("raw_frames must be an exact boolean")
    with path.open("rb") as handle:
        raw = handle.read(16 * 1024**2 + 1)
    if len(raw) > 16 * 1024**2:
        raise ValueError("linear control source report exceeds16MiB")
    if expected_sha256 is not None and sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("TRAIN source bytes differ from the declared saved-model source")
    source = json.loads(raw.decode("utf8"))
    if type(source) is not dict or type(source.get("training_sources")) is not list:
        raise ValueError("linear control requires saved training_sources records")
    records = source["training_sources"]
    if not 1 <= len(records) <= 16:
        raise ValueError("linear control requires1..16 original TRAIN sources")
    episodes, rows = [], 0
    for record in records:
        if (
            type(record) is not dict
            or set(record) != {"episode_id", "context", "role", "transitions", "num_classes"}
            or record["role"] != "train"
            or type(record["num_classes"]) is not int
            or record["num_classes"] != 7
            or type(record["transitions"]) is not list
            or not 1 <= len(record["transitions"]) <= 16
        ):
            raise ValueError("linear control requires exact complete TRAIN-source records")
        rows += len(record["transitions"])
        if rows > 128 or any(type(item) is not dict for item in record["transitions"]):
            raise ValueError("linear control requires at most128 complete TRAIN transition rows")
        transitions = tuple(_transition(transition) for transition in record["transitions"])
        episode = DemonstrationEpisode(**(record | {"transitions": transitions}))
        # Native recurrence owns temporal activation; it needs original frames.
        # Keep the established affine/history default unchanged.
        episodes.append(episode if raw_frames else encode_public_history_episode(episode))
    return tuple(episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument(
        "--source",
        type=Path,
        help="saved TRAIN148 report, or the matching qualified saved-state report for game modes",
    )
    parser.add_argument("--output", type=Path, help="create a new result JSON; never overwrite")
    parser.add_argument(
        "--retained-source",
        type=Path,
        help="qualified109report for new-C, or111report for probe/probe-game/rehearsal modes",
    )
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument(
        "--context-linear-retained-rehearsal",
        choices=("diagnostics", "games"),
        help="Fresh A3/A4 source then private equal-row F-only/joint rehearsal; optional fixed native games after both arms",
    )
    diagnostic.add_argument(
        "--context-linear-controller",
        choices=("current_only", "retain_arrived"),
        help="Fit four TRAIN chooser trials and apply to fresh A4; warm/discard/application costs and pending packets reported",
    )
    diagnostic.add_argument(
        "--context-linear-reference",
        choices=("no_learning", "periodic", "drift", "budget_aware"),
        help="Finite delayed TRAIN A/B/A window with original fixed policy defaults; pending reported, no scores/games",
    )
    diagnostic.add_argument(
        "--example-affine-published",
        action="store_true",
        help="Train known A, guard and publish its actor, then play two known TRAIN games",
    )
    diagnostic.add_argument(
        "--example-affine-live-a4",
        action="store_true",
        help="Publish A3, then learn fresh A4 privately while the shared actor keeps serving",
    )
    diagnostic.add_argument(
        "--example-affine-deploy-a4",
        action="store_true",
        help="Publish A3, start from its actor, learn fresh A4 and guard its deployment; original lifetime budgets",
    )
    parser.add_argument(
        "--example-affine-a4",
        action="store_true",
        help="Collect fresh A4 TRAIN, learn A3/A4/A3 and play both layouts; no saved source",
    )
    diagnostic.add_argument(
        "--context-linear-a4-current-only",
        action="store_true",
        help="Same-data A3/A4/A3 control without joint replay; six fits/54 total rows and both-layout games",
    )
    diagnostic.add_argument(
        "--context-linear-a4-current-repeated",
        action="store_true",
        help="Repeat current arrived A3/A4/A3 rows: six fits/90 exposures, no prior-packet bank, both-layout games",
    )
    diagnostic.add_argument(
        "--context-linear-a4-adaptation",
        action="store_true",
        help="A3/A4/A3 TRAIN learning and both-layout games from fresh qualified A4 TRAIN data",
    )
    diagnostic.add_argument(
        "--context-linear-a4-development",
        action="store_true",
        help="Read-only A branch4 development games from qualified ordinary recurring main state",
    )
    diagnostic.add_argument(
        "--example-affine-recurring",
        action="store_true",
        help="From-birth ordinary ABC/G/A TRAIN adaptation and known G/A games; no saved source",
    )
    diagnostic.add_argument(
        "--example-affine-adaptation",
        action="store_true",
        help="From-birth bundled ABC, fresh private F adaptation and returned-A games; no saved source",
    )
    diagnostic.add_argument(
        "--context-linear-return-game",
        action="store_true",
        help="Matched private F games followed by returning A from saved118states; no fitting",
    )
    diagnostic.add_argument(
        "--context-linear-rehearsal",
        action="store_true",
        help="Private two-fit ABC/F rehearsal versus equal-row F-only control; no games",
    )
    diagnostic.add_argument(
        "--context-linear-probe-game",
        action="store_true",
        help="Fixed F probe_test games before/after the saved isolated probe; no fitting",
    )
    diagnostic.add_argument(
        "--example-affine-play",
        action="store_true",
        help="Train untrained affine ABC from bundled TRAIN examples, then play known B/C games",
    )
    diagnostic.add_argument(
        "--context-linear-probe",
        action="store_true",
        help="Disposable fixed Ftwo-cue affine probe with readonly A/B/C monitors",
    )
    diagnostic.add_argument(
        "--context-linear-after-c-game",
        action="store_true",
        help="Four known B/C_train games from qualified after-C capacity2state; no refit",
    )
    diagnostic.add_argument(
        "--context-linear-new-c",
        action="store_true",
        help="Fixed1vs2 catalogue capacity, C_train-only fits from saved after-B model",
    )
    diagnostic.add_argument(
        "--bounds-only",
        action="store_true",
        help="Compute TRAIN-only input-key row bounds without a fit or prediction",
    )
    diagnostic.add_argument(
        "--context-bounds",
        action="store_true",
        help="Readonly original TRAIN alignment and initial-public-cue input bounds",
    )
    diagnostic.add_argument(
        "--context-linear-fit",
        action="store_true",
        help="One fixed least-squares TRAIN fit per initial public cue, without games",
    )
    diagnostic.add_argument(
        "--context-linear-retained-game",
        action="store_true",
        help="Four known OLD A/NEW B games from the qualified retained after-B model; no refit",
    )
    diagnostic.add_argument(
        "--context-linear-causal-game",
        action="store_true",
        help="Six known current-phase A/B/A games from the qualified causal stream; no refit",
    )
    diagnostic.add_argument(
        "--context-linear-catalogue",
        action="store_true",
        help="Fixed delayed A/B/A TRAIN fits with one arrived same-cue episode retained; no games",
    )
    diagnostic.add_argument(
        "--context-linear-stream",
        action="store_true",
        help="Untrained affine learner on fixed delayed TRAIN A/B/A episodes; no games",
    )
    diagnostic.add_argument(
        "--context-linear-game",
        action="store_true",
        help="Two frozen known-A games from the qualified saved affine fit; no refitting",
    )
    diagnostic.add_argument(
        "--constructive-bp",
        action="store_true",
        help="Disposable original H8 BP TRAIN fit with a larger fixed work cap",
    )
    diagnostic.add_argument(
        "--perceptron",
        action="store_true",
        help="Separate deterministic unit-mistake multiclass TRAIN classifier",
    )
    diagnostic.add_argument(
        "--context-perceptron",
        action="store_true",
        help="Fixed32-wake public-cue contextual CPU control and known-A games",
    )
    diagnostic.add_argument(
        "--kernel-perceptron",
        action="store_true",
        help="Separate bounded degree2 polynomial-kernel TRAIN classifier",
    )
    diagnostic.add_argument(
        "--wide-neural",
        choices=("bp", "pc", "neutral_cpc", "gated_cpc"),
        help="Separate fixed H32 invariant TRAIN fit for one existing neural method",
    )
    parser.add_argument(
        "--fit-updates", type=int, help="TRAIN learner passes in1..512 (default512)"
    )
    parser.add_argument("--fit-seconds", type=float, help="TRAIN fit wall cap in(0,90] (default90)")
    parser.add_argument(
        "--train-a-rollout",
        action="store_true",
        help="Two target-free known TRAIN A games from the explicit H32/128 endpoint",
    )
    parser.add_argument(
        "--cue-turn-gradients",
        action="store_true",
        help="Original A-step2 prior gradients after constructive TRAIN fit readouts",
    )
    parser.add_argument(
        "--cue-relative",
        action="store_true",
        help="Explicit public cue-relative key encoding for the disposable constructive BP fit",
    )
    parser.add_argument(
        "--prequential-feedback",
        action="store_true",
        help="Explicit candidate predictions before labels for a reference window",
    )
    parser.add_argument(
        "--ready-feedback",
        action="store_true",
        help="Diagnose the oldest eligible FIFO TRAIN packet on the current candidate after its label arrives",
    )
    parser.add_argument(
        "--acknowledge-correct",
        action="store_true",
        help="Ordinary ready-feedback reference only: consume fresh proof-bound correct TRAIN without an optimizer update",
    )
    parser.add_argument(
        "--reference-retain-arrived",
        action="store_true",
        help="jointly fit one prior arrived TRAIN episode per cue on selected reference wakes",
    )
    args = parser.parse_args()
    if args.acknowledge_correct and (
        args.context_linear_reference is None
        or not args.ready_feedback
        or args.prequential_feedback
        or args.example_affine_a4
    ):
        parser.error(
            "--acknowledge-correct requires an ordinary --context-linear-reference with --ready-feedback"
        )
    if args.example_affine_a4 and any(
        v is not None and v is not False
        for k, v in vars(args).items()
        if k
        not in (
            {
                "example_affine_a4",
                "describe",
                "output",
                "context_linear_reference",
                "ready_feedback",
            }
            if args.context_linear_reference is not None
            else {"example_affine_a4", "context_linear_controller", "describe", "output"}
            if args.context_linear_controller is not None
            else {"example_affine_a4", "context_linear_retained_rehearsal", "describe", "output"}
            if args.context_linear_retained_rehearsal is not None
            else {"example_affine_a4", "describe", "output"}
        )
    ):
        parser.error(
            "fresh A4 accepts describe/output, an explicit reference with ready feedback, a controller, or retained rehearsal"
        )
    if args.context_linear_retained_rehearsal is not None:
        if not args.example_affine_a4:
            parser.error("--context-linear-retained-rehearsal requires --example-affine-a4")
        from scripts.minigrid_a4_adaptation import handle_rehearsal_cli

        handle_rehearsal_cli(args, parser)
        return
    if args.context_linear_controller is not None:
        if not args.example_affine_a4:
            parser.error("--context-linear-controller requires --example-affine-a4")
        from scripts.minigrid_a4_adaptation import handle_controller_cli

        handle_controller_cli(args, parser)
        return
    if args.reference_retain_arrived and args.context_linear_reference is None:
        parser.error("--reference-retain-arrived requires --context-linear-reference")
    if args.prequential_feedback and args.ready_feedback:
        parser.error("choose only one feedback mode")
    if (args.prequential_feedback or args.ready_feedback) and args.context_linear_reference is None:
        parser.error("feedback requires --context-linear-reference")
    if args.context_linear_reference == "drift" and not (
        args.prequential_feedback or args.ready_feedback
    ):
        parser.error("drift requires explicit --prequential-feedback or --ready-feedback")
    if (
        args.example_affine_published
        or args.example_affine_live_a4
        or args.example_affine_deploy_a4
    ):
        from scripts.minigrid_affine_serving import handle_cli

        handle_cli(args, parser)
        return
    if (
        args.context_linear_a4_adaptation
        or args.context_linear_a4_current_only
        or args.context_linear_a4_current_repeated
    ):
        from scripts.minigrid_a4_adaptation import handle_cli

        handle_cli(args, parser)
        return
    if args.context_linear_a4_development:
        from scripts.minigrid_affine_development import handle_cli

        handle_cli(args, parser)
        return
    if args.example_affine_a4 and args.context_linear_reference is not None:
        from scripts.minigrid_a4_adaptation import handle_reference_cli

        handle_reference_cli(args, parser)
        return
    first_party = (
        args.example_affine_play
        or args.example_affine_adaptation
        or args.example_affine_recurring
        or args.example_affine_a4
    )
    if first_party and (args.source is not None or args.retained_source is not None):
        parser.error("first-party affine examples use bundled TRAIN data, without saved sources")
    if args.retained_source is not None and not (
        args.context_linear_new_c
        or args.context_linear_probe
        or args.context_linear_probe_game
        or args.context_linear_rehearsal
    ):
        parser.error("--retained-source requires a saved new-C/probe/probe-game/rehearsal mode")
    if args.context_perceptron and (
        args.fit_updates is not None
        or args.fit_seconds is not None
        or args.cue_turn_gradients
        or args.cue_relative
        or args.train_a_rollout
    ):
        parser.error("--context-perceptron has a fixed schedule/readout without fit overrides")
    constructive = (
        args.constructive_bp
        or args.perceptron
        or args.kernel_perceptron
        or args.wide_neural is not None
    )
    if not constructive and (
        args.fit_updates is not None
        or args.fit_seconds is not None
        or args.cue_turn_gradients
        or args.cue_relative
        or args.train_a_rollout
    ):
        parser.error(
            "fit options require --constructive-bp, --perceptron, --kernel-perceptron "
            "or --wide-neural"
        )
    if args.wide_neural is not None and (args.cue_relative or args.cue_turn_gradients):
        parser.error("--wide-neural requires the original encoding without gradient diagnostics")
    if args.train_a_rollout and (args.wide_neural is None or args.fit_updates != 128):
        parser.error("--train-a-rollout requires --wide-neural and explicit --fit-updates 128")
    if args.kernel_perceptron and (args.cue_relative or args.cue_turn_gradients):
        parser.error("--kernel-perceptron requires the original encoding without gradients")
    if args.perceptron and (args.cue_relative or args.cue_turn_gradients):
        parser.error("--perceptron requires the original encoding without gradient diagnostics")
    if args.cue_relative and args.cue_turn_gradients:
        parser.error(
            "--cue-relative requires a projection-aware gradient reader; omit --cue-turn-gradients"
        )
    fit_options: dict[str, Any] = dict(
        updates=512 if args.fit_updates is None else args.fit_updates,
        max_wall_seconds=90 if args.fit_seconds is None else args.fit_seconds,
    )
    if args.cue_turn_gradients:
        fit_options["cue_turn_gradients"] = True
    if args.cue_relative:
        fit_options["cue_relative"] = True
    if args.perceptron:
        fit_options["perceptron"] = True
    if args.kernel_perceptron:
        fit_options["kernel_perceptron"] = True
    if args.wide_neural is not None:
        fit_options["wide_method"] = args.wide_neural
    if args.train_a_rollout:
        fit_options["train_a_rollout"] = True
    if constructive:
        describe_constructive_protocol(**fit_options)
    if args.describe:
        if args.source is not None or args.output is not None or args.retained_source is not None:
            parser.error("--describe accepts no source or output")
        if args.example_affine_a4:
            from scripts.minigrid_affine_workflow import describe_workflow

            print(json.dumps(describe_workflow(a4_adaptation=True), indent=2, allow_nan=False))
            return
        if args.example_affine_recurring:
            from scripts.minigrid_affine_workflow import describe_workflow

            print(json.dumps(describe_workflow(recurring_training=True), indent=2, allow_nan=False))
            return
        if args.example_affine_play:
            from scripts.minigrid_affine_workflow import describe_workflow

            print(json.dumps(describe_workflow(), indent=2, allow_nan=False))
            return
        if args.example_affine_adaptation:
            from scripts.minigrid_affine_workflow import describe_workflow

            print(json.dumps(describe_workflow(private_adaptation=True), indent=2, allow_nan=False))
            return
        if args.context_linear_probe_game:
            from scripts.minigrid_affine_probe_games import describe_protocol as game_protocol

            print(json.dumps(game_protocol(), indent=2, allow_nan=False))
            return
        if args.context_linear_return_game:
            from scripts.minigrid_affine_probe_games import describe_rehearsal_protocol

            print(json.dumps(describe_rehearsal_protocol(), indent=2, allow_nan=False))
            return
        if args.context_linear_rehearsal:
            from src.adapters.minigrid_affine_rehearsal import (
                describe_protocol as rehearsal_protocol,
            )

            print(json.dumps(rehearsal_protocol(), indent=2, allow_nan=False))
            return
        if args.context_linear_new_c:
            from src.adapters.minigrid_catalogue_capacity import (
                describe_protocol as capacity_protocol,
            )

            print(json.dumps(capacity_protocol(), indent=2, allow_nan=False))
            return
        if args.context_linear_probe:
            from src.adapters.minigrid_affine_probe import describe_protocol as probe_protocol

            print(json.dumps(probe_protocol(), indent=2, allow_nan=False))
            return
        if args.context_linear_after_c_game:
            print(json.dumps(describe_context_linear_after_c_games(), indent=2, allow_nan=False))
            return
        print(
            json.dumps(
                context_control.describe_protocol()
                if args.context_perceptron
                else describe_context_train_bounds()
                if args.context_bounds
                else describe_context_linear_protocol()
                if args.context_linear_fit
                else describe_context_linear_games()
                if args.context_linear_game
                else describe_context_linear_stream()
                if args.context_linear_stream
                else describe_context_linear_reference(
                    ReferencePolicy(args.context_linear_reference),
                    prequential_feedback=args.prequential_feedback,
                    ready_feedback=args.ready_feedback,
                    retain_arrived=args.reference_retain_arrived,
                    **(dict(acknowledge_correct=True) if args.acknowledge_correct else {}),
                )
                if args.context_linear_reference is not None
                else describe_context_linear_catalogue()
                if args.context_linear_catalogue
                else describe_context_linear_causal_games()
                if args.context_linear_causal_game
                else describe_context_linear_retained_games()
                if args.context_linear_retained_game
                else describe_constructive_protocol(**fit_options)
                if constructive
                else describe_train_fit_bounds()
                if args.bounds_only
                else describe_protocol(),
                indent=2,
                allow_nan=False,
            )
        )
        return
    if args.source is None and not first_party:
        parser.error("--source is required unless --describe")
    if (
        args.context_linear_new_c
        or args.context_linear_probe
        or args.context_linear_probe_game
        or args.context_linear_rehearsal
    ) and args.retained_source is None:
        parser.error("saved after-B/new-C/probe modes require --retained-source")
    if args.output is not None and args.output.exists():
        raise FileExistsError("linear control refuses to replace an existing output")
    result: dict[str, Any]
    try:
        episodes = (
            ()
            if first_party
            or args.context_linear_game
            or args.context_linear_causal_game
            or args.context_linear_retained_game
            or args.context_linear_new_c
            or args.context_linear_after_c_game
            or args.context_linear_probe
            or args.context_linear_probe_game
            or args.context_linear_rehearsal
            or args.context_linear_return_game
            else read_training_sources(args.source)
        )
        from scripts.minigrid_affine_workflow import run_workflow
        from scripts.minigrid_affine_probe_games import run_saved_affine_probe_games

        result = (
            run_workflow(a4_adaptation=True)
            if args.example_affine_a4
            else run_workflow(recurring_training=True)
            if args.example_affine_recurring
            else run_workflow(private_adaptation=True)
            if args.example_affine_adaptation
            else run_workflow()
            if args.example_affine_play
            else run_saved_affine_probe_games(args.source, rehearsal=True)
            if args.context_linear_return_game
            else run_saved_affine_probe_games(args.source, args.retained_source)
            if args.context_linear_probe_game
            else run_saved_context_linear_rehearsal(args.source, args.retained_source)
            if args.context_linear_rehearsal
            else run_saved_context_linear_probe(args.source, args.retained_source)
            if args.context_linear_probe
            else run_saved_context_linear_causal_games(args.source, after_c_capacity=True)
            if args.context_linear_after_c_game
            else run_saved_context_linear_new_c(args.source, args.retained_source)
            if args.context_linear_new_c
            else run_saved_context_linear_games(args.source)
            if args.context_linear_game
            else run_saved_context_linear_causal_games(args.source)
            if args.context_linear_causal_game
            else run_saved_context_linear_causal_games(args.source, retained_catalogue=True)
            if args.context_linear_retained_game
            else run_context_linear_stream(episodes)
            if args.context_linear_stream
            else run_context_linear_stream(
                episodes,
                reference_policy=ReferencePolicy(args.context_linear_reference),
                prequential_feedback=args.prequential_feedback,
                ready_feedback=args.ready_feedback,
                retain_arrived=args.reference_retain_arrived,
                **(dict(acknowledge_correct=True) if args.acknowledge_correct else {}),
            )
            if args.context_linear_reference is not None
            else run_context_linear_stream(episodes, retain_arrived=True)
            if args.context_linear_catalogue
            else context_control.run_control(episodes)
            if args.context_perceptron
            else summarize_context_train_bounds(episodes)
            if args.context_bounds
            else diagnose_context_train_linear_fit(episodes)
            if args.context_linear_fit
            else diagnose_train_fit(episodes, **fit_options)
            if constructive
            else summarize_train_fit_bounds(episodes)
            if args.bounds_only
            else diagnose_train_linear_fit(episodes)
        )
    except LinearFitAdmissionFailed as error:
        result = error.report
    result["runtime"] = dict(numpy=np.__version__, python=sys.version)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.context_linear_rehearsal and len(payload.encode("utf8")) > 4 * 1024**2:
        raise ValueError("private rehearsal report exceeds4MiB; numerical execution must not retry")
    if first_party and not args.example_affine_a4 and len(payload.encode("utf8")) > 4 * 1024**2:
        raise ValueError("affine workflow report exceeds4MiB; execution must not retry")
    if args.context_linear_probe and len(payload.encode("utf8")) > 4 * 1024**2:
        raise ValueError("affine probe report exceeds4MiB; numerical execution must not retry")
    if args.context_linear_new_c and len(payload.encode("utf8")) > 4 * 1024**2:
        raise ValueError("new-C report exceeds4MiB; numerical execution must not retry")
    if args.context_linear_catalogue and len(payload.encode("utf8")) > 2 * 1024**2:
        raise ValueError("arrived catalogue report exceeds2MiB; numerical execution must not retry")
    if args.example_affine_a4 and len(payload.encode("utf8")) > 8 * 1024**2:
        raise ValueError("fresh A4 report exceeds8MiB; numerical execution must not retry")
    if (
        args.context_linear_causal_game
        or args.context_linear_retained_game
        or args.context_linear_after_c_game
        or args.context_linear_probe_game
        or args.context_linear_return_game
    ) and len(payload.encode("utf8")) > 4 * 1024**2:
        raise ValueError(
            "causal affine game report exceeds4MiB; numerical execution must not retry"
        )
    if args.context_linear_stream and len(payload.encode("utf8")) > 2 * 1024**2:
        raise ValueError("affine stream report exceeds2MiB; numerical execution must not retry")
    if args.context_linear_reference is not None and len(payload.encode("utf8")) > 256 * 1024:
        raise ValueError("affine reference report exceeds256KiB; execution must not retry")
    if args.context_linear_game and len(payload.encode("utf8")) > 2 * 1024**2:
        raise ValueError("affine game report exceeds2MiB; numerical execution must not retry")
    if args.context_linear_fit and len(payload.encode("utf8")) > 512 * 1024:
        raise ValueError("context linear report exceeds512KiB; numerical execution must not retry")
    if args.context_perceptron and len(payload.encode("utf8")) > 4 * 1024**2:
        raise ValueError("context control report exceeds4MiB; numerical execution must not retry")
    if args.output is not None:
        with args.output.open("x", encoding="utf8") as handle:
            handle.write(payload + "\n")
    print(payload)
    if result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
