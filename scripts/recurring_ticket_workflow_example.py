"""Fixed invoiceâ†’parcelâ†’invoice TRAIN with complete task readouts between phases.

Composes the existing learner, delayed stream and dispatcher; no new learning
rule, callback controller, probe/final data, replay, restoration or tool handler.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic
from typing import Any
import random

import numpy as np

from scripts.run_ticket_workflow import (
    CONTEXT_FEATURES,
    CONTEXT_TASKS,
    FEATURES,
    training_cases,
    _encode_report,
    _same_value,
)
from scripts.ticket_controller_streams import (
    CONTEXT_RETURN_PHASES,
    POLICY_CHANGE_PHASES,
    build_context_return_stream,
    build_policy_change_stream,
)
from src.adapters.perceptron_decision import (
    PerceptronDecisionPredictor,
    build_decision_training_batch,
)
from src.adapters.perceptron_learner import PerceptronLearner
from src.adapters.context_perceptron import (
    ContextPerceptronLearner,
    ContextPerceptronDecisionPredictor,
)
from src.app.coupled_attribution import AttributionStream
from src.app.ticket_workflow import run_workflow_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_perceptron import PerceptronPolicy
from src.core.context_perceptron import ContextPerceptronPolicy
from src.core.decision_features import DecisionFeatureSpec
from src.core.temporal_decision_sequence import VOCABULARY
from src.core.ticket_workflow import ACTION_IDS, CATALOG, WorkflowCapabilities, WorkflowState
from src.shared.process_memory import ProcessRssSampler

PHASES = (("A", "invoice"), ("B", "parcel"), ("A_return", "invoice"))
CHECKPOINTS = (0, 4, 8, 12)


def prepare_training(*, policy_change=False, context_return=False):
    """Keep existing TRAIN teacher traces; never convert episode outcomes to labels."""
    if type(policy_change) is not bool or type(context_return) is not bool:
        raise ValueError("policy_change must be an exact boolean")
    if context_return:
        if policy_change:
            raise ValueError("context return and policy change are distinct recipes")
        spec = DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)
        return spec, build_context_return_stream(spec)
    if policy_change:
        spec = DecisionFeatureSpec(FEATURES, CATALOG)
        return spec, build_policy_change_stream(spec)
    cases = training_cases()
    if len(cases) != 12 or any(
        c.role != "train" or c.source_group != "train-workflow-traces" for c in cases
    ):
        raise ValueError("only original workflow TRAIN demonstrations are admitted")
    spec = DecisionFeatureSpec(FEATURES, CATALOG)
    batches = []
    for phase, concept in PHASES:
        index = 3 * VOCABULARY.index(concept)
        selected = cases[index : index + 3]
        for repeat in range(4):
            rows = tuple(
                replace(c, request=replace(c.request, request_id=f"{phase}-{repeat}-{i}"))
                for i, c in enumerate(selected)
            )
            batches.append(build_decision_training_batch(rows, spec, phase, allow_subsets=True))
    return spec, tuple(batches)


def episode_slots(point):
    slots = [("ordinary", "invoice"), ("ordinary", "parcel")]
    if point == 8:
        slots.append(("stale_once", "parcel"))
    if point == 12:
        slots.extend((("stale_once", "invoice"), ("revoke_commit", "invoice")))
    return tuple(slots)


def summarize(episodes):
    ordinary = {
        point: {
            row["concept"]: row["episode"]["success"]
            for row in episodes
            if row["source_updates"] == point and row["episode"]["scenario"] == "ordinary"
        }
        for point in CHECKPOINTS
    }
    return dict(
        ordinary_success_by_update=ordinary,
        new_task_success_gain=int(ordinary[8]["parcel"]) - int(ordinary[4]["parcel"]),
        old_task_success_forgetting=int(ordinary[4]["invoice"]) - int(ordinary[8]["invoice"]),
        old_task_return_success_change=int(ordinary[12]["invoice"]) - int(ordinary[8]["invoice"]),
        new_task_success_change_during_return=int(ordinary[12]["parcel"])
        - int(ordinary[8]["parcel"]),
        interpretation="signed descriptive differences on fixed repeated development tasks, not statistical or CPC benefit",
    )


def summarize_policy_changes(episodes, *, opportunity_axis=False):
    """Report current truth and unchanged-task retention without obsolete rewards."""
    if type(opportunity_axis) is not bool:
        raise ValueError("summary axis must be an exact boolean")
    point_key = "opportunity" if opportunity_axis else "source_updates"
    ordinary = {
        point: {
            r["concept"]: r["episode"]["success"]
            for r in episodes
            if r[point_key] == point and r["episode"]["scenario"] == "ordinary"
        }
        for point in CHECKPOINTS
    }
    obsolete = {}
    for point in CHECKPOINTS:
        rows = [
            r
            for r in episodes
            if r[point_key] == point
            and r["concept"] == "invoice"
            and r["episode"]["scenario"] == "ordinary"
        ]
        episode = rows[0]["episode"]
        obsolete[point] = (
            None
            if point <= 4
            else any(
                step["transition"]["state"]["phase"] == "committed"
                and step["transition"]["state"]["committed_tool"] == "route"
                for step in episode["steps"]
            )
        )
    result = dict(
        ordinary_success_by_update=ordinary,
        policy_version_by_update={0: "v1", 4: "v1", 8: "v2", 12: "v2"},
        unchanged_password_success={p: ordinary[p]["password"] for p in CHECKPOINTS},
        new_parcel_success_gain=int(ordinary[8]["parcel"]) - int(ordinary[4]["parcel"]),
        current_invoice_return_change=int(ordinary[12]["invoice"]) - int(ordinary[8]["invoice"]),
        obsolete_invoice_route_committed=obsolete,
        interpretation="v2 invoice route is obsolete even though permitted; adaptation to changed truth is not forgetting/recovery; known-template descriptive outcomes only",
    )
    if opportunity_axis:
        result["ordinary_success_by_opportunity"] = result.pop("ordinary_success_by_update")
        result["policy_version_by_opportunity"] = result.pop("policy_version_by_update")
    return result


def summarize_context_return(episodes, *, seeded_initial=False):
    """Signed fixed-policy changes, with conditional loss/recovery definitions."""
    if type(seeded_initial) is not bool:
        raise ValueError("initial-state description requires an exact boolean")
    ordinary = {
        p: {
            f"{r['policy_version']}:{r['concept']}": r["episode"]["success"]
            for r in episodes
            if r["source_updates"] == p and r["episode"]["scenario"] == "ordinary"
        }
        for p in CHECKPOINTS
    }
    old = ("v1:invoice", "v1:password")
    new = ("v2:invoice", "v2:parcel")
    lost = {t: ordinary[4][t] and not ordinary[8][t] for t in old}
    return dict(
        ordinary_success_by_update=ordinary,
        no_learning_success_reused_from_update0=ordinary[0],
        a_acquisition_change={t: int(ordinary[4][t]) - int(ordinary[0][t]) for t in old},
        b_ability_before_training={t: ordinary[4][t] for t in new},
        b_acquisition_change={t: int(ordinary[8][t]) - int(ordinary[4][t]) for t in new},
        acquired_a_lost_during_b=lost,
        a_success_change_during_b={t: int(ordinary[8][t]) - int(ordinary[4][t]) for t in old},
        a_return_change={t: int(ordinary[12][t]) - int(ordinary[8][t]) for t in old},
        acquired_a_recovered_on_return={t: ordinary[12][t] if lost[t] else None for t in old},
        b_success_change_during_return={t: int(ordinary[12][t]) - int(ordinary[8][t]) for t in new},
        interpretation="same-policy known-template descriptive outcomes; null recovery means no acquired skill was lost; "
        + ("seeded initial-state" if seeded_initial else "zero-state")
        + " control reused, not compute matched or CPC/generalization evidence",
    )


def _require_train_readout_mode(options):
    """Keep this opt-in ordinary TRAIN control separate from every probe mode."""
    value = options.get("train_readout")
    cohort = options.get("train_readout_cohort")
    if cohort is not None and (
        type(cohort) is not str or cohort != "joint" or type(value) is not str or value != "ce"
    ):
        raise ValueError("joint TRAIN CE readout cohort requires --train-readout ce")
    if value is None:
        return
    if (
        type(value) is not str
        or value != "ce"
        or options.get("method") != "bp"
        or options.get("context_return") is not True
        or options.get("replay") != "none"
        or any(
            options.get(key) is not False
            for key in (
                "policy_change",
                "separate_context_heads",
                "private_adaptation",
                "adaptive_serving",
                "calibrated_serving",
                "catalog_study",
                "reference_validation",
                "record_train_fit",
                "class_balanced_bp",
                "reserved_probe",
            )
        )
        or any(
            options.get(key) is not None
            for key in (
                "paired_serving",
                "probe_rehearsal",
                "probe_update",
                "train_update",
                "incident_stream",
                "catalog_action",
                "catalog_context",
                "probe_initialization",
            )
        )
    ):
        raise ValueError("TRAIN CE readout requires ordinary BP context-return without other modes")


def _require_ce_probe_update_mode(options):
    value = options.get("probe_update")
    if type(value) is not str or value != "ce_rehearsal":
        return False
    if (
        type(options.get("method")) is not str
        or options["method"] != "bp"
        or options.get("reserved_probe") is not True
        or options.get("context_return") is not True
        or type(options.get("probe_initialization")) is not str
        or options["probe_initialization"] != "ce"
        or options.get("separate_context_heads") is not False
        or options.get("probe_output_only") is not False
        or options.get("probe_update_diagnostics") is not False
    ):
        raise ValueError(
            "CE rehearsal requires initialized ordinary BP reserved probe without SGD modes"
        )
    return True


def _require_shared_replay_mode(options):
    control = options.get("replay_joint")
    if control is None:
        return
    if (
        type(control) is not str
        or control not in ("joint", "current_only")
        or options.get("method") != "linear_fit"
        or options.get("context_return") is not True
        or options.get("replay") != "coverage"
        or any(
            options.get(key) not in (None, False)
            for key in (
                "separate_context_heads",
                "policy_change",
                "private_adaptation",
                "reserved_probe",
                "adaptive_serving",
                "calibrated_serving",
                "catalog_study",
                "record_train_fit",
                "class_balanced_bp",
                "probe_return_once",
                "probe_update_diagnostics",
                "probe_output_only",
                "reference_validation",
                "reference_validation_protocol",
            )
        )
        or any(
            options.get(key) is not None
            for key in (
                "paired_serving",
                "probe_rehearsal",
                "probe_update",
                "probe_initialization",
                "train_readout",
                "train_readout_cohort",
                "catalog_action",
                "catalog_context",
                "incident_stream",
            )
        )
    ):
        raise ValueError(
            "shared joint TRAIN replay requires ordinary shared linear context-return coverage alone"
        )


def run_example(
    *,
    replay="none",
    replay_joint=None,
    shared_serving=False,
    method="perceptron",
    policy_change=False,
    context_return=False,
    separate_context_heads=False,
    private_adaptation=False,
    paired_serving=None,
    probe_rehearsal=None,
    probe_return_once=False,
    probe_update=None,
    train_update=None,
    adaptive_serving=False,
    incident_stream=None,
    calibrated_serving=False,
    catalog_action=None,
    catalog_context=None,
    catalog_study=False,
    reference_validation=False,
    record_train_fit=False,
    class_balanced_bp=False,
    reserved_probe=False,
    probe_initialization=None,
    train_readout=None,
    train_readout_cohort=None,
    probe_update_diagnostics=False,
    probe_output_only=False,
):
    if type(replay) is not str or replay not in ("none", "recent", "coverage"):
        raise ValueError("replay must be none, recent or coverage")
    if shared_serving is not False:
        from scripts.shared_linear_publication import require_mode

        require_mode(locals())
    _require_shared_replay_mode(locals())
    linear_replay = (
        method == "linear_fit"
        and context_return is True
        and separate_context_heads is False
        and replay in ("recent", "coverage")
    )
    if (
        type(probe_output_only) is not bool
        or probe_output_only
        and (reserved_probe is not True or context_return is not True or method != "bp")
    ):
        raise ValueError("output-only probe requires ordinary BP context-return reserved probe")
    if (
        type(probe_update_diagnostics) is not bool
        or probe_update_diagnostics
        and (reserved_probe is not True or context_return is not True or method != "bp")
    ):
        raise ValueError("probe update diagnostics require an ordinary BP reserved probe")
    ce_reserved_probe = _require_ce_probe_update_mode(locals())
    _require_train_readout_mode(locals())
    linear_reserved_probe = (
        reserved_probe is True
        and method == "linear_fit"
        and type(separate_context_heads) is bool
        and probe_update == "minimum_change"
        and type(probe_update) is str
    )
    linear_joint_probe = (
        linear_reserved_probe
        and separate_context_heads is False
        and type(probe_rehearsal) is str
        and probe_rehearsal in ("joint", "current_only")
    )
    if type(probe_return_once) is not bool or probe_return_once and not linear_joint_probe:
        raise ValueError("once-return requires explicit ordinary shared-head rehearsal")
    if probe_initialization is not None and (
        type(probe_initialization) is not str
        or probe_initialization != "ce"
        or reserved_probe is not True
        or method != "bp"
    ):
        raise ValueError("CE probe initialization requires an ordinary BP reserved probe")
    if (
        type(reserved_probe) is not bool
        or reserved_probe
        and (
            (not linear_reserved_probe and method not in ("bp", "pc", "neutral_cpc", "gated_cpc"))
            or context_return is not True
            or (not linear_reserved_probe and separate_context_heads is not False)
            or policy_change is not False
            or replay != "none"
            or private_adaptation is not False
            or paired_serving is not None
            or adaptive_serving is not False
            or catalog_action is not None
            or catalog_context is not None
            or catalog_study is not False
            or reference_validation is not False
            or incident_stream is not None
            or calibrated_serving is not False
            or record_train_fit is not False
            or class_balanced_bp is not False
            or (not linear_reserved_probe and not ce_reserved_probe and probe_update is not None)
            or (probe_rehearsal is not None and not linear_joint_probe)
            or (not linear_reserved_probe and train_update is not None)
        )
    ):
        raise ValueError(
            "reserved probe requires ordinary native context-return without other modes"
        )
    if train_update is not None and (
        type(train_update) is not str
        or train_update != "minimum_change"
        or method != "linear_fit"
        or context_return is not True
        or paired_serving not in (None, "retained_joint", "joint")
        or (replay != "none" and not linear_replay)
        or private_adaptation
        or (probe_update is not None and not linear_reserved_probe)
        or (probe_rehearsal is not None and not linear_joint_probe)
        or reference_validation
        or catalog_study
        or catalog_action is not None
        or catalog_context is not None
        or adaptive_serving
        or incident_stream is not None
        or calibrated_serving
        or record_train_fit
        or class_balanced_bp
    ):
        raise ValueError("minimum-change TRAIN requires ordinary context-return LS")
    if (
        probe_update is not None
        and not ce_reserved_probe
        and (
            type(probe_update) is not str
            or probe_update != "minimum_change"
            or (paired_serving != "retained_joint_probe" and not linear_reserved_probe)
            or method != "linear_fit"
            or context_return is not True
            or (separate_context_heads is not True and not linear_reserved_probe)
            or (probe_rehearsal is not None and not linear_joint_probe)
        )
    ):
        raise ValueError("minimum-change probe requires sequential owned LS retained probe")
    if probe_rehearsal is not None and (
        type(probe_rehearsal) is not str
        or probe_rehearsal not in ("joint", "current_only")
        or (paired_serving != "retained_joint_probe" and not linear_joint_probe)
        or method != "linear_fit"
        or context_return is not True
        or (separate_context_heads is not True and not linear_joint_probe)
    ):
        raise ValueError("probe rehearsal requires explicit owned LS retained probe")
    if (
        type(reference_validation) is not bool
        or reference_validation
        and not (
            method == "linear_fit"
            and context_return is True
            and separate_context_heads is True
            and private_adaptation is False
            and paired_serving is None
            and adaptive_serving is False
            and catalog_action is None
            and catalog_context is None
            and catalog_study is False
            and incident_stream is None
            and calibrated_serving is False
            and replay == "none"
            and policy_change is False
            and record_train_fit is False
            and class_balanced_bp is False
        )
    ):
        raise ValueError("reference validation requires its explicit ordinary owned LS mode")
    if (
        type(catalog_study) is not bool
        or catalog_study
        and not (
            method == "linear_fit"
            and context_return is True
            and separate_context_heads is True
            and private_adaptation is False
            and paired_serving is None
            and adaptive_serving is False
            and catalog_action is None
            and catalog_context is None
            and incident_stream is None
            and calibrated_serving is False
            and replay == "none"
            and policy_change is False
            and record_train_fit is False
            and class_balanced_bp is False
        )
    ):
        raise ValueError("catalog study requires its explicit ordinary owned LS mode")
    owned_serving = (
        paired_serving is not None
        or catalog_action is not None
        or catalog_study
        or reference_validation
    )
    source_handoff = (
        private_adaptation
        or owned_serving
        or reserved_probe
        or train_readout is not None
        or shared_serving
    )
    if (catalog_action is not None or catalog_context is not None) and not (
        type(catalog_action) is str
        and catalog_action in ("wait", "wake1", "replay6-wake1")
        and type(catalog_context) is str
        and catalog_context in ("clean", "mistakes")
        and method == "linear_fit"
        and context_return is True
        and separate_context_heads is True
        and private_adaptation is False
        and paired_serving is None
        and adaptive_serving is False
        and incident_stream is None
        and calibrated_serving is False
        and replay == "none"
        and policy_change is False
        and record_train_fit is False
        and class_balanced_bp is False
    ):
        raise ValueError("catalog serving requires explicit ordinary owned LS action/context")
    if (
        type(calibrated_serving) is not bool
        or calibrated_serving
        and incident_stream is None
        or incident_stream is not None
        and (
            type(incident_stream) is not str
            or incident_stream not in ("stationary", "shift")
            or adaptive_serving is not True
            or paired_serving != "joint"
        )
    ):
        raise ValueError("incident serving requires explicit adaptive joint mode and stream")
    if (
        type(adaptive_serving) is not bool
        or adaptive_serving
        and not (
            method == "linear_fit"
            and context_return is True
            and separate_context_heads is True
            and paired_serving in ("current", "joint", "same_context_joint")
            and private_adaptation is False
            and replay == "none"
            and policy_change is False
        )
    ):
        raise ValueError("adaptive serving requires explicit ordinary owned LS paired mode")
    if method == "linear_fit" and (
        context_return is not True
        or (replay != "none" and not linear_replay)
        or policy_change is not False
        or type(separate_context_heads) is not bool
        or private_adaptation is not False
        or record_train_fit is not False
        or class_balanced_bp is not False
    ):
        raise ValueError(
            "linear_fit requires ordinary context-return without TRAIN diagnostics; replay uses one head"
        )
    if (
        type(class_balanced_bp) is not bool
        or class_balanced_bp
        and (
            context_return is not True
            or method != "bp"
            or replay != "none"
            or policy_change is not False
            or separate_context_heads is not False
            or private_adaptation is not False
            or paired_serving is not None
        )
    ):
        raise ValueError("class-balanced BP requires the ordinary native BP context-return recipe")
    if (
        type(record_train_fit) is not bool
        or record_train_fit
        and (
            context_return is not True
            or method not in ("bp", "pc", "neutral_cpc", "gated_cpc")
            or replay != "none"
            or separate_context_heads is not False
            or private_adaptation is not False
            or paired_serving is not None
        )
    ):
        raise ValueError("TRAIN fit diagnostics require the native context-return recipe")
    if (
        type(private_adaptation) is not bool
        or private_adaptation
        and (
            separate_context_heads is not True
            or context_return is not True
            or policy_change is not False
            or method != "perceptron"
            or replay != "none"
        )
    ):
        raise ValueError("private adaptation requires the separate-head context-return recipe")
    if paired_serving is not None and (
        type(paired_serving) is not str
        or paired_serving
        not in (
            "current",
            "joint",
            "same_context_joint",
            "retained_joint",
            "retained_joint_publish",
            "retained_joint_probe",
        )
        or paired_serving
        in (
            "same_context_joint",
            "retained_joint",
            "retained_joint_publish",
            "retained_joint_probe",
        )
        and method != "linear_fit"
        or private_adaptation
        or separate_context_heads is not True
        or context_return is not True
        or policy_change is not False
        or method not in ("perceptron", "linear_fit")
        or replay != "none"
    ):
        raise ValueError(
            "paired serving requires separate-head context-return without private adaptation"
        )
    if type(separate_context_heads) is not bool or (
        separate_context_heads
        and (
            context_return is not True
            or method not in ("perceptron", "linear_fit")
            or replay != "none"
        )
    ):
        raise ValueError(
            "separate context heads require explicit context-return, perceptron or linear_fit and replay=none"
        )
    if type(context_return) is not bool or (context_return and policy_change is not False):
        raise ValueError("context-return requires policy_change=False")
    if type(policy_change) is not bool or (policy_change and replay != "none"):
        raise ValueError("policy-change recipe requires explicit boolean and replay=none")
    if type(replay) is not str or replay not in ("none", "recent", "coverage"):
        raise ValueError("replay must be explicit none, recent or coverage")
    if type(method) is not str or method not in (
        "perceptron",
        "bp",
        "pc",
        "neutral_cpc",
        "gated_cpc",
        "linear_fit",
    ):
        raise ValueError("method must be an explicit supported learner")
    if method not in ("perceptron", "linear_fit") and replay != "none":
        raise ValueError("native chronological recipe requires replay=none")
    native: Any = None
    linear: Any = None
    if method == "linear_fit":
        from scripts import ticket_workflow_linear

        linear = ticket_workflow_linear
    elif method != "perceptron":
        from scripts import ticket_workflow_native

        native = ticket_workflow_native
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="chronological_ticket_workflow_v1",
        complete=False,
        episodes=[],
        checkpoints=[],
        phases=[dict(name=p, concept=c, passes=4, rows_per_pass=3) for p, c in PHASES],
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        external_actions=0,
        scope="one conventional repeated-task acquisition/retention/return case; no native/CPC/generalization/release claim",
    )
    learner = budget = stream = None
    helper = replay_budget = None
    publication: Any = None
    wake_row_counts: tuple[int, ...] = ()
    wake_commit_count: int | None = None
    previous_receipts = ()
    attempted_updates = attempted_rows = returned_updates = 0
    snapshot_states: dict[int, Any] = {}
    same_state = (lambda a, b: a == b) if native is None else native.same_state
    torch_rng = None
    train_readout_guard = None
    train_readout_finished_snapshot = None
    python_rng = random.getstate() if linear is not None else None
    readout_key = "linear_readout_work" if linear is not None else "native_readout_work"
    if linear is not None:
        report.update(method=method, linear_readout_work=[])
    if train_update is not None:
        report["linear_train_update"] = train_update
    if replay_joint is not None:
        report["linear_replay_joint"] = replay_joint
    if reserved_probe:
        report["reserved_probe_requested"] = True
        if linear_reserved_probe or ce_reserved_probe:
            report["reserved_probe_update"] = probe_update
        if linear_joint_probe:
            report["reserved_probe_rehearsal"] = probe_rehearsal
        if probe_return_once:
            report["reserved_probe_return_once"] = True
    if probe_initialization is not None:
        report["probe_initialization_requested"] = probe_initialization
    if train_readout is not None:
        report["train_readout_requested"] = train_readout
        if train_readout_cohort is not None:
            report["train_readout_cohort_requested"] = train_readout_cohort
    if native is not None:
        torch_rng = native.require_torch().get_rng_state().clone()
        report.update(
            method=method,
            native_readout_work=[],
            scope="fixed native-family repeated development-task observation; no matched-compute, generalization, CPC benefit or release claim",
        )
    if policy_change:
        report.update(
            definition="versioned_policy_ticket_workflow_v1",
            policy_change=True,
            phases=[
                dict(name=p, policy_version=v, concepts=list(c), passes=4, rows_per_pass=6)
                for p, v, c in POLICY_CHANGE_PHASES
            ],
            scope="fixed development-only policy change; current-rule adaptation is not retention of obsolete instructions; no meta-generalization/controller/CPC benefit",
            planned_teacher_generation_work=dict(teacher_traces=8, dispatches=24, requests=24),
        )
    if context_return:
        report.update(
            definition="context_return_ticket_workflow_v1",
            context_return=True,
            phases=[
                dict(name=p, policy_version=v, concepts=list(c), passes=4, rows_per_pass=6)
                for p, v, c in CONTEXT_RETURN_PHASES
            ],
            scope="fresh-zero known-template true context return; no final/probe/replay/native or comparative benefit claim",
            no_learning_control="same deterministic tasks and unchanged zero snapshot; reuse update0 outcomes without retraining",
        )
        if native is not None:
            report.update(
                definition="native_context_return_ticket_workflow_v1",
                scope="fixed native D9 known-context return; no heldout, matched-total-compute or CPC benefit claim",
                no_learning_control="same deterministic tasks and unchanged seeded initial snapshot; reuse update0 outcomes without retraining",
            )
        if linear is not None:
            report.update(
                definition="linear_context_return_ticket_workflow_v1",
                scope="explicit minimum-change LS known-context return; no heldout, matched compute, CPC or release claim"
                if train_update is not None
                else "current-packet fixed LS known-context return; no heldout, matched compute, CPC or release claim",
                no_learning_control="same deterministic tasks and unchanged unfitted uniform initial state; reuse update0 outcomes without fitting",
            )
    if separate_context_heads:
        report.update(
            definition="separate_context_head_ticket_workflow_v1",
            separate_context_heads=True,
            head_routing=dict(
                public_tokens=["policy_v1", "policy_v2"],
                head_feature_vocabulary=list(FEATURES),
                homogeneous_batches=True,
                coefficients=96,
                coefficient_bytes=768,
            ),
            scope="fixed known public-context conventional control; increased capacity96vs60, not matched capacity/compute or CPC/generalization/release benefit",
        )
        if linear is not None:
            report.update(
                definition="separate_context_linear_ticket_workflow_v1",
                scope="fixed public-context LS heads, increased capacity96vs60; no context discovery, matched compute, CPC/generalization/release claim",
            )
    if record_train_fit:
        report.update(
            train_fit_readouts=[],
            train_fit_scope="fixed original TRAIN truth at four checkpoints; descriptive diagnostics, not arrived online/controller feedback or selection",
        )
    balanced = native.ClassBalancedUpdates(report) if class_balanced_bp else None
    context_guard = lambda: None
    rehearsal_options = {"context_return": True} if context_return else {}
    progress_options = dict(
        rehearsal_options, **({} if replay_joint is None else dict(shared_joint=replay_joint))
    )

    def require_train_update():
        changed = (
            ("linear_replay_joint" in report)
            if replay_joint is None
            else (
                type(report.get("linear_replay_joint")) is not str
                or report["linear_replay_joint"] != replay_joint
            )
        )
        if changed:
            raise ValueError("original shared joint TRAIN replay intent changed")
        if linear is not None:
            linear.require_train_update(report, train_update)
            if helper is not None and (
                type(report.get("replay_mode")) is not str
                or report["replay_mode"] != replay
                or report.get("replay_consent")
                != "explicit synthetic original TRAIN packet reuse; not inbox permission renewal"
            ):
                raise ValueError("original TRAIN replay intent or consent changed")

    def before_update(features):
        nonlocal attempted_updates, attempted_rows
        require_train_update()
        if (native is not None or linear is not None) and context_return:
            context_guard()
        if balanced is not None:
            assert stream is not None
            inbox = stream._runtime._inbox
            ready = inbox._ready(inbox._last_time)
            if len(ready) != 1:
                raise ValueError("class-balanced update requires one ready TRAIN packet")
            key = ready[0]
            balanced.admit(
                inbox._experiences[key],
                inbox._labels[key],
                inbox._last_time,
                inbox._learner_version,
            )
        attempted_updates += 1
        attempted_rows += len(features)
        if publication is not None:
            publication.before_fit()

    def after_update():
        nonlocal returned_updates
        returned_updates += 1
        if publication is not None:
            publication.after_fit()

    # Why this: counts precede delegation, so a throwing update is never free.
    class CountedLearner(PerceptronLearner):
        def fork(self):
            child = CountedLearner(self.policy)
            child.restore_state(self.snapshot_state())
            return child

        def train_batch(self, features, targets):
            nonlocal attempted_updates, attempted_rows, returned_updates
            context_guard()
            attempted_updates += 1
            attempted_rows += len(features)
            result = super().train_batch(features, targets)
            returned_updates += 1
            return result

    class CountedContextLearner(ContextPerceptronLearner):
        def fork(self):
            child = CountedContextLearner(self.policy)
            child.restore_state(self.snapshot_state())
            return child

        def train_batch(self, features, targets):
            context_guard()
            before_update(features)
            result = super().train_batch(features, targets)
            after_update()
            return result

    rng = np.random.get_state()
    assert isinstance(rng, tuple)
    with ProcessRssSampler(interval_seconds=0.02) as sampler:
        try:
            spec, batches = (
                prepare_training(context_return=True)
                if context_return
                else prepare_training(policy_change=True)
                if policy_change
                else prepare_training()
            )
            fit_source = native.prepare_train_fit(spec) if record_train_fit else None
            if replay != "none":
                from scripts.ticket_workflow_rehearsal import trajectory_batches

                batches = trajectory_batches(spec, batches, **rehearsal_options)
                report["replay_mode"] = replay
                report["replay_consent"] = (
                    "explicit synthetic original TRAIN packet reuse; not inbox permission renewal"
                )
                if context_return:
                    report.update(
                        definition="context_return_queue_rehearsal_v1",
                        scope="fixed contextual TRAIN queue rehearsal; extra replay exposure is explicit; no final/probe/native/CPC/generalization/release claim",
                        queue_semantics=dict(
                            tickets_per_queue=2,
                            steps_per_ticket=3,
                            advance_after_successful_commit_step=2,
                            terminal_step=5,
                            observation="next public policy decision after explicit queue advancement",
                            feature_dim=9,
                            num_classes=6,
                        ),
                    )
            report["training_data"] = [
                dict(role=b.role, context=b.context, row_ids=list(b.row_ids), rows=len(b.features))
                for b in batches
            ]
            # Why: refused report edits must not rewrite the cost of paid fits.
            wake_row_counts = tuple(len(batch.features) for batch in batches)
            if policy_change or context_return:
                phase_definitions = (
                    CONTEXT_RETURN_PHASES if context_return else POLICY_CHANGE_PHASES
                )
                for index, row in enumerate(report["training_data"]):
                    row["policy_version"] = phase_definitions[index // 4][1]
            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=12,
                    max_wall_seconds=240
                    if catalog_study or reference_validation
                    else 120
                    if paired_serving == "retained_joint_probe"
                    else 30,
                    max_process_rss_bytes=512 * 1024**2,
                ),
                monotonic,
            )
            budget.attach_memory(sampler)
            learner = (
                linear.counted_learner(
                    before_update,
                    after_update,
                    separate=separate_context_heads,
                    spec=spec,
                    **(dict(train_update=train_update) if train_update is not None else {}),
                )
                if linear is not None
                else CountedContextLearner(
                    ContextPerceptronPolicy(
                        len(spec.vocabulary),
                        len(CATALOG),
                        (spec.vocabulary.index("policy_v1"), spec.vocabulary.index("policy_v2")),
                    )
                )
                if separate_context_heads
                else CountedLearner(PerceptronPolicy(len(spec.vocabulary), len(CATALOG)))
                if native is None
                else native.counted_learner(
                    method,
                    len(spec.vocabulary),
                    len(CATALOG),
                    before_update,
                    after_update,
                    **({"balanced": balanced} if balanced is not None else {}),
                )
            )
            original_seed = learner.snapshot_state()
            if replay != "none":
                from scripts.ticket_workflow_rehearsal import make_rehearsal, require_progress

                helper, replay_budget = make_rehearsal(
                    replay, learner, budget, sampler, **progress_options
                )

            def capture(point, state, batch):
                snapshot_states[point] = state

            if shared_serving:
                from scripts.shared_linear_publication import SharedLinearPublication

                publication = SharedLinearPublication(learner, spec, budget, batches)
                report["shared_serving"] = publication.row
            stream = AttributionStream(
                learner,
                batches,
                (),
                checkpoint_updates=CHECKPOINTS,
                label_delay=1,
                budget=budget,
                observe_checkpoint=capture,
                training_only=True,
                **(dict(actor=publication.actor) if publication is not None else {}),
                **(
                    dict(replay_consent=True)
                    if linear_joint_probe or replay_joint is not None
                    else {}
                ),
                **({} if helper is None else {"rehearsal": helper}),
            )
            actor_pin = (
                stream._runtime.actor.snapshot_state() if policy_change or context_return else None
            )
            if context_return:
                input_pin = deepcopy((spec, batches, report["training_data"], fit_source))
                budget_pin = (
                    id(budget.budget),
                    budget.budget,
                    budget.started_at,
                    id(budget.clock),
                    id(budget.progress),
                    id(budget.process_rss_sampler),
                )
                paid_episodes = report["episodes"]
                paid_checkpoints = report["checkpoints"]
                episode_pins: list[Any] = []
                checkpoint_pins: list[Any] = []
                paid_fits = report.get("train_fit_readouts", [])
                fit_pins: list[Any] = []
                paid_linear_reads = report.get("linear_readout_work", [])
                linear_read_pins: list[Any] = []

                def context_guard():
                    if publication is not None:
                        if report.get("shared_serving") is not publication.row:
                            raise ValueError("original shared serving record replaced")
                        publication.require()
                    if balanced is not None:
                        # Already inside the runtime write lease during training.
                        balanced.require_receipts(stream._runtime._inbox.applied_updates)
                    if not _same_value(
                        (spec, batches, report["training_data"], fit_source), input_pin
                    ):
                        raise ValueError("context TRAIN input changed")
                    if (
                        not _same_value(np.random.get_state(), rng)
                        or linear is not None
                        and random.getstate() != python_rng
                        or native is not None
                        and not native.require_torch().equal(
                            torch_rng, native.require_torch().get_rng_state()
                        )
                        or not same_state(learner.snapshot_state(), original_seed)
                        or not same_state(
                            stream._runtime.actor.snapshot_state(),
                            actor_pin if publication is None else publication.expected_actor,
                        )
                        or budget_pin
                        != (
                            id(budget.budget),
                            budget.budget,
                            budget.started_at,
                            id(budget.clock),
                            id(budget.progress),
                            id(budget.process_rss_sampler),
                        )
                    ):
                        raise ValueError("context source/RNG/budget origin changed")
                    for key, owner, pins in (
                        (
                            ("episodes", paid_episodes, episode_pins),
                            ("checkpoints", paid_checkpoints, checkpoint_pins),
                        )
                        + (
                            (("train_fit_readouts", paid_fits, fit_pins),)
                            if record_train_fit
                            else ()
                        )
                        + (
                            (("linear_readout_work", paid_linear_reads, linear_read_pins),)
                            if linear is not None
                            else ()
                        )
                    ):
                        if (
                            report[key] is not owner
                            or len(owner) != len(pins)
                            or any(
                                row is not original or not _same_value(row, value)
                                for row, (original, value) in zip(owner, pins, strict=True)
                            )
                        ):
                            owner[:] = [original for original, _ in pins]
                            for original, value in pins:
                                original.clear()
                                original.update(deepcopy(value))
                            report[key] = owner
                            raise ValueError("context paid evidence changed")

            for point in CHECKPOINTS:
                context_guard()
                if point:
                    if helper is not None:
                        require_progress(
                            stream,
                            helper,
                            replay_budget,
                            point - 4,
                            stream.candidate_snapshot(),
                            previous_receipts,
                            **progress_options,
                        )
                    stream.run_until(point)
                state = stream.candidate_snapshot()
                if (
                    not same_state(state, snapshot_states[point])
                    or (helper is None and state.completed_updates != point)
                    or budget.updates_completed != point
                ):
                    raise ValueError("chronological source/checkpoint/budget mismatch")
                replay_pin = None
                if helper is not None:
                    previous_receipts = require_progress(
                        stream,
                        helper,
                        replay_budget,
                        point,
                        state,
                        previous_receipts,
                        **progress_options,
                    )
                    replay_pin = helper.report
                report["checkpoints"].append(
                    linear.checkpoint(point, state)
                    if linear is not None
                    else dict(
                        source_updates=point,
                        source_rows=state.training_examples,
                        source_mistakes=state.mistake_updates,
                        state=asdict(state),
                    )
                    if native is None
                    else native.checkpoint(point, state)
                )
                if context_return:
                    checkpoint_pins.append(
                        (report["checkpoints"][-1], deepcopy(report["checkpoints"][-1]))
                    )
                predict = (
                    linear.readout(spec, state)
                    if linear is not None
                    else ContextPerceptronDecisionPredictor(spec, state)
                    if separate_context_heads
                    else PerceptronDecisionPredictor(spec, state)
                    if native is None
                    else native.readout(spec, state)
                )
                if native is not None or linear is not None:
                    report[readout_key].append(
                        dict(source_updates=point, work=asdict(predict.work))
                    )
                if linear is not None:
                    linear_read_pins.append(
                        (report[readout_key][-1], deepcopy(report[readout_key][-1]))
                    )
                readout_pin = (
                    deepcopy(predict)
                    if context_return and native is None and linear is None
                    else None
                )

                def require_source():
                    require_train_update()
                    if (
                        "reserved_probe_return_once" in report
                    ) is not probe_return_once or report.get(
                        "reserved_probe_return_once", False
                    ) is not probe_return_once:
                        raise ValueError("original once-return intent changed")
                    if (
                        linear_joint_probe
                        and report.get("reserved_probe_rehearsal") != probe_rehearsal
                    ):
                        raise ValueError("original shared rehearsal intent changed")
                    if train_readout is not None and (
                        report.get("train_readout_requested") != train_readout
                        or report.get("train_readout_cohort_requested") != train_readout_cohort
                        or type(report.get("train_readout_cohort_requested"))
                        is not type(train_readout_cohort)
                        or ("train_readout_cohort_requested" in report)
                        != (train_readout_cohort is not None)
                    ):
                        raise ValueError("TRAIN CE readout intent changed")
                    if (
                        probe_initialization is not None
                        and report.get("probe_initialization_requested") != probe_initialization
                    ):
                        raise ValueError("CE probe initialization intent changed")
                    if ce_reserved_probe and (
                        type(report.get("reserved_probe_update")) is not str
                        or report.get("reserved_probe_update") != "ce_rehearsal"
                    ):
                        raise ValueError("CE probe update intent changed")
                    if reserved_probe and report.get("reserved_probe_requested") is not True:
                        raise ValueError("reserved probe intent changed")
                    context_guard()
                    if source_handoff and (
                        stream.completed_updates != point or len(stream.applied_updates) != point
                    ):
                        raise ValueError("private source native/receipt progress changed")
                    if context_return:
                        if linear is not None:
                            linear.require_readout(predict, spec, state)
                        elif native is not None:
                            native.require_readout(predict, spec, state)
                        elif not _same_value(predict, readout_pin):
                            raise ValueError("context frozen predictor changed")
                    source_state = (
                        stream.candidate_snapshot()
                        if train_readout_finished_snapshot is None
                        else train_readout_finished_snapshot()
                    )
                    if not same_state(source_state, state) or budget.updates_completed != point:
                        raise ValueError(
                            "frozen task readout changed original TRAIN source or budget"
                        )
                    if (
                        native is not None or linear is not None or policy_change
                    ) and not same_state(learner.snapshot_state(), original_seed):
                        raise ValueError("frozen task readout changed original seed learner")
                    if policy_change and not same_state(
                        stream._runtime.actor.snapshot_state(), actor_pin
                    ):
                        raise ValueError("frozen task readout changed original actor")
                    if helper is not None:
                        require_progress(
                            stream,
                            helper,
                            replay_budget,
                            point,
                            state,
                            previous_receipts,
                            **progress_options,
                        )
                        if helper.report != replay_pin:
                            raise ValueError("original replay work changed during frozen inference")

                def guarded_predict(request):
                    budget.before_final()
                    require_source()
                    try:
                        result = predict(request)
                    finally:
                        if native is not None:
                            report["native_readout_work"][-1]["work"] = asdict(predict.work)
                        if linear is not None:
                            linear.record_readout_work(
                                report, paid_linear_reads, linear_read_pins, predict
                            )
                        if context_return:
                            require_source()
                    require_source()
                    return result

                if record_train_fit:
                    row = dict(source_updates=point, role="train", complete=False)
                    report["pending_train_fit"] = row

                    def require_fit_source():
                        if (
                            report["pending_train_fit"] is not row
                            or type(row.get("source_updates")) is not int
                            or row["source_updates"] != point
                            or row.get("role") != "train"
                        ):
                            raise ValueError("pending TRAIN fit metadata changed")
                        budget.before_final()
                        require_source()

                    try:
                        native.record_train_fit(predict, fit_source, row, require_fit_source)
                        if (
                            report["pending_train_fit"] is not row
                            or type(row.get("source_updates")) is not int
                            or row["source_updates"] != point
                            or row.get("role") != "train"
                            or row.get("complete") is not True
                            or not native.same_state(
                                row.get("work"),
                                dict(
                                    native_forward_attempts=1,
                                    rows_attempted=12,
                                    native_forward_returns=1,
                                    rows_returned=12,
                                    summary_returns=1,
                                    unreturned_work_unknown=False,
                                ),
                            )
                        ):
                            raise ValueError("returned TRAIN fit evidence changed or incomplete")
                    except Exception as error:
                        row.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])
                        raise
                    finally:
                        report["train_fit_readouts"].append(row)
                        fit_pins.append((row, deepcopy(row)))
                        report["pending_train_fit"] = None
                    require_source()

                slots = episode_slots(point)
                if policy_change:
                    slots = tuple(
                        ("ordinary", c) for c in ("invoice", "parcel", "password")
                    ) + tuple((s, c) for s, c in slots if s != "ordinary")
                version = "v1" if policy_change and point <= 4 else "v2"
                versioned_slots = tuple((s, version, c) for s, c in slots)
                if context_return:
                    versioned_slots = tuple(("ordinary", v, c) for v, c in CONTEXT_TASKS)
                    if point == 12:
                        versioned_slots += tuple(
                            (s, v, c)
                            for s in ("stale_once", "revoke_commit")
                            for v, c in CONTEXT_TASKS
                        )
                for scenario, version, concept in versioned_slots:
                    require_source()
                    report["pending_episode"] = dict(
                        source_updates=point,
                        scenario=scenario,
                        concept=concept,
                        effects_unknown=True,
                    )
                    episode = run_workflow_episode(
                        WorkflowState(f"task-{point}-{concept}", concept, policy_version=version),
                        guarded_predict,
                        capabilities=WorkflowCapabilities(ACTION_IDS),
                        max_steps=6,
                        scenario=scenario,
                        **({"include_policy_context": True} if context_return else {}),
                    )
                    report["episodes"].append(
                        dict(
                            source_updates=point,
                            concept=concept,
                            role="outer_selection",
                            source_group="development-context-return-workflow-traces"
                            if context_return
                            else "development-policy-workflow-traces"
                            if policy_change
                            else "development-workflow-traces",
                            episode=episode,
                        )
                    )
                    if context_return:
                        report["episodes"][-1].update(policy_version=version)
                        episode_pins.append(
                            (report["episodes"][-1], deepcopy(report["episodes"][-1]))
                        )
                    report["pending_episode"] = None
                    require_source()
                    if not episode["complete"]:
                        raise RuntimeError("task episode has uncertain paid work")
            if policy_change or context_return:
                require_source()
            if not source_handoff:
                report["stream"] = stream.finish()
            if not same_state(learner.snapshot_state(), original_seed):
                raise ValueError("original seed learner changed")
            if policy_change or context_return:
                if not same_state(stream._runtime.actor.snapshot_state(), actor_pin):
                    raise ValueError("original actor changed at finish")
                report.update(source_unchanged=True, actor_unchanged=True)
            report["summary"] = (
                summarize_context_return(report["episodes"], seeded_initial=True)
                if context_return and native is not None
                else summarize_context_return(report["episodes"])
                if context_return
                else summarize_policy_changes(report["episodes"])
                if policy_change
                else summarize(report["episodes"])
            )
            if train_readout is not None:
                from scripts.ticket_ce_tasks import run_train_readout

                report["train_readout_source_training"] = dict(
                    completed_updates=stream.completed_updates,
                    applied_update_count=len(stream.applied_updates),
                )
                task_row = report["train_readout"] = {}
                _, train_readout_guard = run_train_readout(
                    stream,
                    state,
                    spec,
                    report,
                    budget,
                    check_boundary=require_source,
                    memory=sampler,
                    row=task_row,
                    **(
                        dict(cohort=train_readout_cohort)
                        if train_readout_cohort is not None
                        else {}
                    ),
                )
                require_source()
                if task_row.get("complete") is not True:
                    raise ValueError("TRAIN CE task execution incomplete")
                report["stream"] = stream.finish()
                from scripts.ticket_ce_tasks import _finished_snapshot

                train_readout_finished_snapshot = _finished_snapshot(
                    stream, report["stream"], report
                )
                train_readout_guard()
                require_source()
            if private_adaptation:
                from scripts.context_head_rehearsal import run_owned_pair

                report["private_source_training"] = dict(
                    completed_updates=stream.completed_updates,
                    applied_update_count=len(stream.applied_updates),
                )
                # Why: hand off only the live qualified native checkpoint. Its
                # original budget/clock and source/actor guards stay active;
                # private fitting never advances the source runtime or budget.
                report["private_adaptation"] = run_owned_pair(
                    state, spec, report, budget, check_boundary=require_source, memory=sampler
                )
                private = report["private_adaptation"]
                report["probe_updates"] = private["training_work"]["returned_updates"]
                report["replay_examples"] = sum(
                    s["replay"] for arm in private["cases"] for s in arm.get("sessions", [])
                )
                report.update(source_unchanged=None, actor_unchanged=None)
                require_source()
                report.update(source_unchanged=True, actor_unchanged=True)
                if private["complete"] is not True:
                    raise ValueError(f"private adaptation incomplete: {private.get('error')}")
                report["stream"] = stream.finish()
            if reserved_probe:
                from scripts.ticket_native_probe import run_owned_probe

                report["reserved_probe_source_training"] = dict(
                    completed_updates=stream.completed_updates,
                    applied_update_count=len(stream.applied_updates),
                )
                private = report["reserved_probe"] = {}
                report.update(source_unchanged=None, actor_unchanged=None)
                if probe_initialization is not None:
                    from scripts.ticket_ce_probe import run_initialized_probe

                    initialization_row = report["probe_initialization"] = {}
                    run_initialized_probe(
                        stream,
                        state,
                        spec,
                        report,
                        budget,
                        check_boundary=require_source,
                        memory=sampler,
                        row=private,
                        initialization_row=initialization_row,
                        **(dict(ce_rehearsal=True) if ce_reserved_probe else {}),
                        **(dict(probe_output_only=True) if probe_output_only else {}),
                        **(dict(probe_update_diagnostics=True) if probe_update_diagnostics else {}),
                    )
                else:
                    run_owned_probe(
                        state,
                        spec,
                        report,
                        budget,
                        check_boundary=require_source,
                        memory=sampler,
                        row=private,
                        **(dict(probe_output_only=True) if probe_output_only else {}),
                        **(dict(probe_update_diagnostics=True) if probe_update_diagnostics else {}),
                        **(dict(linear_probe_update=probe_update) if linear_reserved_probe else {}),
                        **(
                            dict(linear_rehearsal=probe_rehearsal, source_stream=stream)
                            if linear_joint_probe
                            else {}
                        ),
                        **(dict(linear_return_once=True) if probe_return_once else {}),
                    )
                work = private["preserved_training_work"]
                if ce_reserved_probe:
                    report["probe_auxiliary_updates"] = private.get("preserved_ce_rehearsal_work")
                if linear_joint_probe:
                    report["probe_auxiliary_updates"] = private.get(
                        "preserved_linear_rehearsal_work"
                    )
                    report["probe_replay_examples"] = (
                        report["probe_auxiliary_updates"].get("additional_rows_charged")
                        if type(report["probe_auxiliary_updates"]) is dict
                        else None
                    )
                report["probe_updates"] = (
                    work["returned_updates"]
                    if work["attempted_updates"] == work["returned_updates"]
                    else None
                )
                if linear_joint_probe:
                    report["probe_updates"] = (
                        report["probe_auxiliary_updates"].get("applied_updates")
                        if type(report["probe_auxiliary_updates"]) is dict
                        and report["probe_auxiliary_updates"].get("application_unknown_updates")
                        == 0
                        else None
                    )
                require_source()
                report.update(source_unchanged=True, actor_unchanged=True)
                if private["complete"] is not True:
                    raise ValueError(f"reserved probe incomplete: {private.get('error')}")
                report["stream"] = stream.finish()
            if owned_serving:
                from scripts.context_head_live import run_owned_case

                report["serving_source_training"] = dict(
                    completed_updates=stream.completed_updates,
                    applied_update_count=len(stream.applied_updates),
                )
                # Keep the source open and guarded, exactly as for private
                # adaptation, but train/publish only the ordinary TRAIN child.
                child_key = (
                    "reference_validation"
                    if reference_validation
                    else "catalog_study"
                    if catalog_study
                    else "paired_serving"
                    if paired_serving is not None
                    else "catalog_serving"
                )
                report[child_key] = run_owned_case(
                    state,
                    spec,
                    report,
                    budget,
                    control=(
                        "reference_validation"
                        if reference_validation
                        else "catalog_study"
                        if catalog_study
                        else paired_serving
                        if paired_serving is not None
                        else "catalog"
                    ),
                    check_boundary=require_source,
                    memory=sampler,
                    **(
                        dict(probe_rehearsal=probe_rehearsal) if probe_rehearsal is not None else {}
                    ),
                    **(dict(probe_update=probe_update) if probe_update is not None else {}),
                    **(dict(train_update=train_update) if train_update is not None else {}),
                    **(dict(linear_fit=True) if linear is not None else {}),
                    **(dict(adaptive=True) if adaptive_serving else {}),
                    **(
                        dict(catalog_action=catalog_action, catalog_context=catalog_context)
                        if catalog_action is not None
                        else {}
                    ),
                    **(
                        dict(incident_stream=incident_stream, calibrated=calibrated_serving)
                        if incident_stream is not None
                        else {}
                    ),
                )
                report.update(source_unchanged=None, actor_unchanged=None)
                require_source()
                report.update(source_unchanged=True, actor_unchanged=True)
                serving = report[child_key]
                if serving["complete"] is not True:
                    raise ValueError(f"paired serving incomplete: {serving.get('error')}")
                report["stream"] = stream.finish()
            if publication is not None:
                require_source()
                report.update(actor_unchanged=None)
                publication.run(stream, require_source, context_guard)
                require_source()
                report["stream"] = stream.finish()
                publication.observe_clock()
                report.update(source_unchanged=True, actor_unchanged=True)
            report["complete"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
            if publication is not None:
                report.update(source_unchanged=None, actor_unchanged=None)
            if train_readout is not None:
                report.update(source_unchanged=None, actor_unchanged=None)
            if linear is not None:
                from src.core.learner_ports import TrainingUpdateFailed

                if isinstance(error, TrainingUpdateFailed):
                    report["native_failure_definition"] = error.definition
                    report["native_failure_work"] = error.work
            if helper is not None:
                report["preserved_replay_receipts"] = previous_receipts
        finally:
            if publication is not None:
                try:
                    publication.close()
                except Exception as close_error:
                    report.update(complete=False, source_unchanged=None, actor_unchanged=None)
                    report.setdefault("error", f"{type(close_error).__name__}: {close_error}"[:512])
                preserved = publication.preserved_work()
                # Keep complete paid episodes once so the fixed-size CLI report
                # fits its original bound. Failures still need a trusted copy.
                if (
                    report.get("complete") is not True
                    or preserved.get("complete") is not True
                    or not _same_value(report.get("shared_serving"), preserved)
                ):
                    report["shared_serving_preserved_work"] = preserved
            if balanced is not None:
                balanced.finish()
            report["training_work"] = dict(
                attempted_updates=attempted_updates,
                attempted_rows=attempted_rows,
                returned_updates=returned_updates,
                unreturned_work_unknown=attempted_updates != returned_updates,
            )
            if budget is not None:
                report["training_work"]["budget_committed_updates"] = budget.updates_completed
                report["budget"] = dict(
                    limits=asdict(budget.budget),
                    started_at=budget.started_at,
                    last_clock=budget.last_clock,
                    updates_completed=budget.updates_completed,
                    replay_examples_completed=budget.replay_examples_completed,
                )
            if stream is not None:
                try:
                    wake_receipts = stream.applied_updates
                    report["applied_updates"] = [asdict(row) for row in wake_receipts]
                    wake_commit_count = len(wake_receipts)
                except Exception as receipt_error:
                    report.update(
                        complete=False,
                        receipt_prefix_unknown=True,
                        receipt_error=f"{type(receipt_error).__name__}: {receipt_error}"[:512],
                    )
            if helper is not None:
                try:
                    report["replay"] = helper.report
                    report["replay_retention"] = asdict(helper.retention)
                    report["published_replay_receipts"] = [
                        asdict(r) for r in stream.applied_consolidations
                    ]
                except Exception as replay_error:
                    report.update(
                        complete=False,
                        replay_prefix_unknown=True,
                        replay_error=f"{type(replay_error).__name__}: {replay_error}"[:512],
                    )
                report["replay_examples"] = replay_budget.replay_examples_completed
                report["replay_budget"] = dict(
                    limits=asdict(replay_budget.budget),
                    started_at=replay_budget.started_at,
                    updates_completed=replay_budget.updates_completed,
                    replay_examples_completed=replay_budget.replay_examples_completed,
                )
            works = [row["episode"]["work"] for row in report["episodes"]]
            report["episode_work"] = (
                {key: sum(w[key] for w in works) for key in works[0]} if works else {}
            )
            current_rng = np.random.get_state()
            assert isinstance(current_rng, tuple)
            report["numpy_rng_unchanged"] = (
                rng[0] == current_rng[0]
                and np.array_equal(rng[1], current_rng[1])
                and rng[2:] == current_rng[2:]
            )
            if not report["numpy_rng_unchanged"]:
                report.update(complete=False, error="caller NumPy RNG changed")
            report["costs"] = dict(
                elapsed_seconds=monotonic() - started,
                model_random_draws=0,
                attempted_training_affine_terms=attempted_rows
                * (
                    len(
                        FEATURES
                        if separate_context_heads
                        else CONTEXT_FEATURES
                        if context_return
                        else FEATURES
                    )
                    + 1
                )
                * len(CATALOG),
                validated_readout_affine_terms=report["episode_work"].get(
                    "prediction_calls_validated", 0
                )
                * (
                    len(
                        FEATURES
                        if separate_context_heads
                        else CONTEXT_FEATURES
                        if context_return
                        else FEATURES
                    )
                    + 1
                )
                * 5,
                retained_checkpoint_coefficient_bytes=len(report["checkpoints"])
                * (
                    len(
                        FEATURES
                        if separate_context_heads
                        else CONTEXT_FEATURES
                        if context_return
                        else FEATURES
                    )
                    + 1
                )
                * len(CATALOG)
                * 8
                * (2 if separate_context_heads else 1),
                qualification="logical coefficient/affine lower bounds; Python, copies, softmax and validation additional",
                sampled_process_rss=asdict(sampler.snapshot()),
            )
            if separate_context_heads:
                report["costs"].update(
                    model_coefficient_bytes=768,
                    context_routing="each homogeneous TRAIN/readout routes by public onehot; typed requests add full-D9 encode before original D7 readout; route rows/feature values and token visits are API quantities, not all copy/validation or total compute",
                )
            if native is not None:
                report["torch_rng_unchanged"] = bool(
                    native.require_torch().equal(torch_rng, native.require_torch().get_rng_state())
                )
                if not report["torch_rng_unchanged"]:
                    report.update(complete=False, error="caller Torch RNG changed")
                # Do not label native work with perceptron-only counters or zeros.
                for key in (
                    "attempted_training_affine_terms",
                    "validated_readout_affine_terms",
                    "retained_checkpoint_coefficient_bytes",
                ):
                    report["costs"].pop(key)
                report["costs"].update(
                    native.native_costs(
                        report,
                        attempted_rows,
                        method,
                        feature_dim=len(CONTEXT_FEATURES) if context_return else len(FEATURES),
                    )
                )
                if reserved_probe:
                    report["costs"]["scope"] = (
                        "original TRAIN source only; reserved_probe records separate "
                        "probe/return training, diagnostics and typed readouts"
                    )
                    if probe_initialization is not None:
                        report["costs"]["scope"] += (
                            "; probe_initialization records separate auxiliary TRAIN CE fitting and state derivation"
                        )
                if train_readout is not None:
                    report["costs"]["scope"] = (
                        "original TRAIN source only; train_readout records separate auxiliary "
                        "CE fitting and known-TRAIN task execution; elapsed times overlap"
                    )
            if linear is not None:
                report["python_rng_unchanged"] = random.getstate() == python_rng
                if not report["python_rng_unchanged"]:
                    report.update(complete=False, error="caller Python RNG changed")
                for key in (
                    "attempted_training_affine_terms",
                    "validated_readout_affine_terms",
                    "retained_checkpoint_coefficient_bytes",
                ):
                    report["costs"].pop(key)
                report["costs"].update(
                    linear.costs(report, wake_rows=wake_row_counts, wake_commits=wake_commit_count)
                )
                if reserved_probe:
                    report["costs"]["scope"] = (
                        "original TRAIN source only; reserved_probe records separate minimum-change "
                        "probe/return updates, diagnostics and typed LS readouts; not matched total computation"
                    )
    if train_readout_guard is not None:
        try:
            train_readout_guard()
        except Exception as error:
            report.update(
                complete=False,
                source_unchanged=None,
                actor_unchanged=None,
            )
            # Keep the initiating failure when this final check also refuses.
            if not report.get("error"):
                report["error"] = f"{type(error).__name__}: {error}"[:512]
    if reserved_probe:
        from src.app.reserved_probe_summary import add_reserved_probe_summary

        add_reserved_probe_summary(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replay", choices=("none", "recent", "coverage"), default="none")
    parser.add_argument(
        "--replay-joint",
        choices=("joint", "current_only"),
        help="explicit shared LS coverage TRAIN packet control",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--policy-change", action="store_true")
    modes.add_argument("--context-return", action="store_true")
    parser.add_argument("--separate-context-heads", action="store_true")
    parser.add_argument("--record-train-fit", action="store_true")
    parser.add_argument("--class-balanced-bp", action="store_true")
    adaptation = parser.add_mutually_exclusive_group()
    adaptation.add_argument("--private-adaptation", action="store_true")
    adaptation.add_argument("--reserved-probe", action="store_true")
    adaptation.add_argument(
        "--shared-serving",
        action="store_true",
        help="guarded shared LS coverage publication, typed serving and exact rollback",
    )
    adaptation.add_argument(
        "--paired-serving",
        choices=(
            "current",
            "joint",
            "same_context_joint",
            "retained_joint",
            "retained_joint_publish",
            "retained_joint_probe",
        ),
    )
    adaptation.add_argument("--catalog-action", choices=("wait", "wake1", "replay6-wake1"))
    adaptation.add_argument("--catalog-study", action="store_true")
    adaptation.add_argument("--reference-validation", action="store_true")
    parser.add_argument("--reference-validation-protocol", action="store_true")
    parser.add_argument("--catalog-context", choices=("clean", "mistakes"))
    parser.add_argument("--probe-rehearsal", choices=("joint", "current_only"))
    parser.add_argument(
        "--probe-return-once",
        action="store_true",
        help="fit each certified old TRAIN packet once on private return",
    )
    parser.add_argument("--probe-update", choices=("minimum_change", "ce_rehearsal"))
    parser.add_argument("--probe-initialization", choices=("ce",))
    parser.add_argument("--probe-update-diagnostics", action="store_true")
    parser.add_argument("--probe-output-only", action="store_true")
    parser.add_argument(
        "--train-readout",
        choices=("ce",),
        help="separate TRAIN CE fit and four known-task episodes; no actor promotion",
    )
    parser.add_argument(
        "--train-readout-cohort",
        choices=("joint",),
        help="joint old+fresh TRAIN18 and six tasks with --train-readout ce",
    )
    parser.add_argument("--train-update", choices=("minimum_change",))
    parser.add_argument("--adaptive-serving", action="store_true")
    parser.add_argument("--incident-stream", choices=("stationary", "shift"))
    parser.add_argument("--calibrated-serving", action="store_true")
    parser.add_argument(
        "--method",
        choices=("perceptron", "bp", "pc", "neutral_cpc", "gated_cpc", "linear_fit"),
        default="perceptron",
    )
    args = parser.parse_args(argv)
    if args.shared_serving:
        from scripts.shared_linear_publication import require_mode

        try:
            require_mode(vars(args))
        except ValueError as error:
            parser.error(str(error))
    try:
        _require_shared_replay_mode(vars(args))
    except ValueError as error:
        parser.error(str(error))
    if args.replay_joint is not None and args.output.exists():
        raise FileExistsError("shared joint replay output must be new")
    try:
        ce_reserved_probe = _require_ce_probe_update_mode(vars(args))
        _require_train_readout_mode(vars(args))
    except ValueError as error:
        parser.error(str(error))
    if args.probe_output_only and (
        not args.reserved_probe or not args.context_return or args.method != "bp"
    ):
        parser.error("--probe-output-only requires ordinary BP context-return reserved probe")
    if args.probe_update_diagnostics and (not args.reserved_probe or args.method != "bp"):
        parser.error("probe update diagnostics require an ordinary BP reserved probe")
    if args.probe_initialization is not None and (not args.reserved_probe or args.method != "bp"):
        parser.error("--probe-initialization ce requires --reserved-probe --method bp")
    linear_reserved_probe = (
        args.reserved_probe
        and args.method == "linear_fit"
        and args.probe_update == "minimum_change"
    )
    linear_joint_probe = (
        linear_reserved_probe
        and not args.separate_context_heads
        and args.probe_rehearsal in ("joint", "current_only")
    )
    if args.probe_return_once and not linear_joint_probe:
        parser.error("--probe-return-once requires explicit ordinary shared-head rehearsal")
    if args.reserved_probe and (
        not args.context_return
        or args.policy_change
        or args.private_adaptation
        or args.paired_serving is not None
        or args.reference_validation
        or args.catalog_study
        or args.catalog_action is not None
        or (
            not linear_reserved_probe
            and args.method not in ("bp", "pc", "neutral_cpc", "gated_cpc")
        )
        or (not linear_reserved_probe and args.separate_context_heads)
        or args.replay != "none"
        or args.record_train_fit
        or args.class_balanced_bp
        or args.catalog_context is not None
        or (not linear_reserved_probe and not ce_reserved_probe and args.probe_update is not None)
        or (args.probe_rehearsal is not None and not linear_joint_probe)
        or (not linear_reserved_probe and args.train_update is not None)
        or args.adaptive_serving
        or args.incident_stream is not None
        or args.calibrated_serving
        or args.reference_validation_protocol
    ):
        parser.error(
            "--reserved-probe requires ordinary native --context-return without other modes"
        )
    if (args.reserved_probe or args.train_readout is not None) and args.output.exists():
        raise FileExistsError("reserved probe output must be new")
    linear_replay = (
        args.method == "linear_fit"
        and args.context_return
        and not args.separate_context_heads
        and args.replay in ("recent", "coverage")
    )
    if args.train_update is not None and (
        args.method != "linear_fit"
        or not args.context_return
        or args.paired_serving not in (None, "retained_joint", "joint")
        or (args.replay != "none" and not linear_replay)
        or args.private_adaptation
        or (args.probe_update is not None and not linear_reserved_probe)
        or (args.probe_rehearsal is not None and not linear_joint_probe)
        or args.reference_validation
        or args.catalog_study
        or args.catalog_action is not None
        or args.catalog_context is not None
        or args.adaptive_serving
        or args.incident_stream is not None
        or args.calibrated_serving
        or args.record_train_fit
        or args.class_balanced_bp
    ):
        parser.error("--train-update requires ordinary context-return LS")
    if (
        args.probe_update is not None
        and not ce_reserved_probe
        and (
            (args.paired_serving != "retained_joint_probe" and not linear_reserved_probe)
            or args.method != "linear_fit"
            or not args.context_return
            or (not args.separate_context_heads and not linear_reserved_probe)
            or (args.probe_rehearsal is not None and not linear_joint_probe)
        )
    ):
        parser.error(
            "--probe-update requires sequential owned LS --paired-serving retained_joint_probe"
        )
    if args.probe_rehearsal is not None and (
        (args.paired_serving != "retained_joint_probe" and not linear_joint_probe)
        or args.method != "linear_fit"
        or not args.context_return
        or (not args.separate_context_heads and not linear_joint_probe)
    ):
        parser.error(
            "--probe-rehearsal requires ordinary owned LS --paired-serving retained_joint_probe"
        )
    if args.reference_validation_protocol and not args.reference_validation:
        parser.error("--reference-validation-protocol requires --reference-validation")
    if args.reference_validation and not (
        args.method == "linear_fit"
        and args.context_return
        and args.separate_context_heads
        and not args.adaptive_serving
        and args.incident_stream is None
        and not args.calibrated_serving
        and args.catalog_context is None
        and args.replay == "none"
        and not args.policy_change
        and not args.record_train_fit
        and not args.class_balanced_bp
    ):
        parser.error("--reference-validation requires ordinary owned LS context heads")
    if args.reference_validation_protocol:
        from scripts.context_stream_comparison import protocol_snapshot
        from scripts.context_catalog_learning import write_catalog_report

        if args.output.exists():
            raise FileExistsError("reference validation protocol output must be new")
        return (
            0
            if write_catalog_report(
                args.output, protocol_snapshot(withheld=True, owned_linear=True), max_bytes=1024**2
            )
            else 1
        )
    if args.catalog_study and not (
        args.method == "linear_fit"
        and args.context_return
        and args.separate_context_heads
        and not args.private_adaptation
        and args.paired_serving is None
        and args.catalog_action is None
        and args.catalog_context is None
        and not args.adaptive_serving
        and args.incident_stream is None
        and not args.calibrated_serving
    ):
        parser.error("--catalog-study requires its separate ordinary owned LS context heads")
    if (args.catalog_action is not None or args.catalog_context is not None) and not (
        args.catalog_action is not None
        and args.catalog_context is not None
        and args.method == "linear_fit"
        and args.context_return
        and args.separate_context_heads
        and not args.private_adaptation
        and args.paired_serving is None
        and not args.adaptive_serving
        and args.incident_stream is None
        and not args.calibrated_serving
    ):
        parser.error("--catalog-action/--catalog-context require ordinary owned LS context heads")
    if (
        args.calibrated_serving
        and args.incident_stream is None
        or args.incident_stream is not None
        and (not args.adaptive_serving or args.paired_serving != "joint")
    ):
        parser.error(
            "--incident-stream requires --adaptive-serving --paired-serving joint; calibration requires a stream"
        )
    if args.adaptive_serving and not (
        args.method == "linear_fit"
        and args.context_return
        and args.separate_context_heads
        and args.paired_serving in ("current", "joint", "same_context_joint")
        and not args.private_adaptation
        and args.replay == "none"
    ):
        parser.error("--adaptive-serving requires ordinary owned LS --paired-serving mode")
    if args.method == "linear_fit" and not (
        args.context_return
        and (args.replay == "none" or linear_replay)
        and not args.private_adaptation
        and not args.record_train_fit
        and not args.class_balanced_bp
    ):
        parser.error(
            "linear_fit requires ordinary --context-return without TRAIN diagnostics; replay uses one head"
        )
    if args.class_balanced_bp and not (
        args.context_return
        and args.method == "bp"
        and args.replay == "none"
        and not args.separate_context_heads
        and not args.private_adaptation
        and args.paired_serving is None
    ):
        parser.error(
            "--class-balanced-bp requires ordinary --context-return --method bp --replay none"
        )
    if args.record_train_fit and not (args.context_return and args.method != "perceptron"):
        parser.error("--record-train-fit requires --context-return and a native method")
    if args.policy_change and args.replay != "none":
        parser.error("--policy-change requires --replay none")
    if args.method not in ("perceptron", "linear_fit") and args.replay != "none":
        parser.error("native workflows require --replay none")
    if args.separate_context_heads and (
        not args.context_return
        or args.method not in ("perceptron", "linear_fit")
        or args.replay != "none"
    ):
        parser.error(
            "--separate-context-heads requires --context-return --method perceptron or linear_fit --replay none"
        )
    if args.private_adaptation and not (args.separate_context_heads and args.context_return):
        parser.error("--private-adaptation requires --context-return --separate-context-heads")
    if args.paired_serving is not None and not (
        args.separate_context_heads and args.context_return
    ):
        parser.error("--paired-serving requires --context-return --separate-context-heads")
    if args.paired_serving == "same_context_joint" and args.method != "linear_fit":
        parser.error("--paired-serving same_context_joint requires --method linear_fit")
    if (
        args.paired_serving is not None
        or args.catalog_action is not None
        or args.catalog_study
        or args.reference_validation
    ):
        if args.output.exists():
            raise FileExistsError("paired serving workflow output must be new")
        report = run_example(
            context_return=True,
            separate_context_heads=True,
            **(
                dict(reference_validation=True)
                if args.reference_validation
                else dict(catalog_study=True)
                if args.catalog_study
                else (
                    dict(paired_serving=args.paired_serving)
                    if args.paired_serving is not None
                    else dict(
                        catalog_action=args.catalog_action, catalog_context=args.catalog_context
                    )
                )
            ),
            **(dict(method="linear_fit") if args.method == "linear_fit" else {}),
            **(
                dict(probe_rehearsal=args.probe_rehearsal)
                if args.probe_rehearsal is not None
                else {}
            ),
            **(dict(probe_update=args.probe_update) if args.probe_update is not None else {}),
            **(dict(train_update=args.train_update) if args.train_update is not None else {}),
            **(dict(adaptive_serving=True) if args.adaptive_serving else {}),
            **(
                dict(
                    incident_stream=args.incident_stream, calibrated_serving=args.calibrated_serving
                )
                if args.incident_stream is not None
                else {}
            ),
        )
        from scripts.context_catalog_learning import write_catalog_report

        source_work = report.get("training_work", {})
        source_work = source_work if type(source_work) is dict else {}
        serving = report.get(
            "reference_validation",
            report.get(
                "paired_serving", report.get("catalog_serving", report.get("catalog_study"))
            ),
        )
        serving_work = serving.get("training_work") if type(serving) is dict else {}
        serving_work = serving_work if type(serving_work) is dict else {}
        totals: dict[str, Any] = {}
        private_work = (
            serving.get("isolated_transfer_probe", {}).get("private_work", {})
            if type(serving) is dict and args.paired_serving == "retained_joint_probe"
            else None
        )
        for field, key in (
            ("train_passes", "attempted_updates"),
            ("training_rows", "attempted_rows"),
        ):
            values = (
                source_work.get(key),
                serving_work.get(key) if serving is not None else 0,
                private_work.get("attempted_training_examples" if key == "attempted_rows" else key)
                if private_work is not None
                else 0,
            )
            totals[field] = sum(values) if all(type(v) is int and v >= 0 for v in values) else None
        return (
            0
            if write_catalog_report(
                args.output,
                report,
                max_bytes=(24 if args.catalog_study or args.reference_validation else 4) * 1024**2,
                observed_work=dict(
                    seconds=report.get("elapsed_seconds"),
                    peak_rss_bytes=report.get("memory", {}).get("peak_bytes"),
                    counts=totals,
                ),
            )
            else 1
        )
    if args.train_readout is not None:
        report = run_example(
            context_return=True,
            method="bp",
            train_readout=args.train_readout,
            **(
                dict(train_readout_cohort=args.train_readout_cohort)
                if args.train_readout_cohort is not None
                else {}
            ),
        )
        from scripts.context_catalog_learning import write_catalog_report

        return (
            0
            if write_catalog_report(
                args.output,
                report,
                observed_work=dict(
                    source=report.get("training_work"),
                    train_readout=report.get(
                        "preserved_train_readout", report.get("train_readout")
                    ),
                    train_readout_budget=report.get(
                        "preserved_train_readout_budget", report.get("train_readout_budget")
                    ),
                ),
                max_bytes=4 * 1024**2,
            )
            else 1
        )
    if args.reserved_probe:
        report = run_example(
            context_return=True,
            method=args.method,
            reserved_probe=True,
            **(dict(probe_rehearsal=args.probe_rehearsal) if linear_joint_probe else {}),
            **(dict(probe_return_once=True) if args.probe_return_once else {}),
            **(dict(probe_update="ce_rehearsal") if ce_reserved_probe else {}),
            **(dict(probe_output_only=True) if args.probe_output_only else {}),
            **(dict(probe_update_diagnostics=True) if args.probe_update_diagnostics else {}),
            **(
                dict(
                    separate_context_heads=args.separate_context_heads,
                    probe_update=args.probe_update,
                )
                if linear_reserved_probe
                else {}
            ),
            **(dict(train_update=args.train_update) if args.train_update is not None else {}),
            **(
                dict(probe_initialization=args.probe_initialization)
                if args.probe_initialization is not None
                else {}
            ),
        )
        from scripts.context_catalog_learning import write_catalog_report

        observed_work = dict(
            source=report.get("training_work"),
            probe=report.get("reserved_probe", {}).get("preserved_training_work"),
        )
        if linear_joint_probe:
            observed_work["linear_rehearsal"] = report.get("reserved_probe", {}).get(
                "preserved_linear_rehearsal_work"
            )
        if args.probe_initialization is not None:
            observed_work["probe_initialization"] = report.get("probe_initialization")
        if args.probe_update_diagnostics:
            observed_work["probe_update_diagnostics"] = (
                report.get("reserved_probe", {}).get("preserved_update_diagnostics", {}) or {}
            ).get("work")
        if ce_reserved_probe:
            observed_work["probe_ce_rehearsal"] = report.get("reserved_probe", {}).get(
                "preserved_ce_rehearsal_work"
            )
        if args.probe_output_only:
            private_report = report.get("reserved_probe", {})
            observed_work["probe_parameter_update_scope"] = private_report.get(
                "preserved_metadata", private_report
            ).get("parameter_update_scope")
        return (
            0
            if write_catalog_report(
                args.output,
                report,
                max_bytes=4 * 1024**2,
                observed_work=observed_work,
            )
            else 1
        )
    if args.private_adaptation:
        if args.output.exists():
            raise FileExistsError("private workflow output must be new")
        report = run_example(
            context_return=True, separate_context_heads=True, private_adaptation=True
        )
        from scripts.context_catalog_learning import write_catalog_report

        return (
            0
            if write_catalog_report(
                args.output,
                report,
                observed_work=dict(
                    source=report.get("training_work"),
                    private=report.get("private_adaptation", {}).get("training_work"),
                ),
                max_bytes=4 * 1024**2,
            )
            else 1
        )
    # Reserve before fitting; a failed case cannot overwrite an earlier result.
    with args.output.open("xb") as target:
        target.write(b"{}\n")
    if args.separate_context_heads:
        report = run_example(
            context_return=True,
            separate_context_heads=True,
            **({"method": "linear_fit"} if args.method == "linear_fit" else {}),
            **(dict(train_update=args.train_update) if args.train_update is not None else {}),
        )
    elif args.context_return:
        report = (
            run_example(
                context_return=True,
                method="bp",
                class_balanced_bp=True,
                **({"record_train_fit": True} if args.record_train_fit else {}),
            )
            if args.class_balanced_bp
            else run_example(context_return=True, method=args.method, record_train_fit=True)
            if args.record_train_fit
            else run_example(
                context_return=True,
                method=args.method,
                **(dict(train_update=args.train_update) if args.train_update is not None else {}),
                **(dict(replay=args.replay) if args.replay != "none" else {}),
                **(dict(replay_joint=args.replay_joint) if args.replay_joint is not None else {}),
                **(dict(shared_serving=True) if args.shared_serving else {}),
            )
            if args.method != "perceptron"
            else run_example(context_return=True)
            if args.replay == "none"
            else run_example(context_return=True, replay=args.replay)
        )
    elif args.policy_change:
        report = run_example(replay=args.replay, method=args.method, policy_change=True)
    elif args.method != "perceptron":
        report = run_example(replay=args.replay, method=args.method)
    else:
        report = run_example() if args.replay == "none" else run_example(replay=args.replay)
    args.output.write_bytes(_encode_report(report))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
