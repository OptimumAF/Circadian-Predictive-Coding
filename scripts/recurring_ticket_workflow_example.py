"""Fixed invoice→parcel→invoice TRAIN with complete task readouts between phases.

Composes the existing learner, delayed stream and dispatcher; no new learning
rule, callback controller, probe/final data, replay, restoration or tool handler.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic
from typing import Any

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


def summarize_context_return(episodes):
    """Signed fixed-policy changes, with conditional loss/recovery definitions."""
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
        interpretation="same-policy known-template descriptive outcomes; null recovery means no acquired skill was lost; zero-state control reused, not compute matched or CPC/generalization evidence",
    )


def run_example(
    *,
    replay="none",
    method="perceptron",
    policy_change=False,
    context_return=False,
    separate_context_heads=False,
):
    if type(separate_context_heads) is not bool or (
        separate_context_heads
        and (context_return is not True or method != "perceptron" or replay != "none")
    ):
        raise ValueError(
            "separate context heads require explicit context-return, perceptron and replay=none"
        )
    if type(context_return) is not bool or (
        context_return and (policy_change is not False or method != "perceptron")
    ):
        raise ValueError("context-return requires perceptron and policy_change=False")
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
    ):
        raise ValueError("method must be an explicit supported learner")
    if method != "perceptron" and replay != "none":
        raise ValueError("native chronological recipe requires replay=none")
    native: Any = None
    if method != "perceptron":
        from scripts import ticket_workflow_native as native
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
    previous_receipts = ()
    attempted_updates = attempted_rows = returned_updates = 0
    snapshot_states: dict[int, Any] = {}
    same_state = (lambda a, b: a == b) if native is None else native.same_state
    torch_rng = None
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
    context_guard = lambda: None
    rehearsal_options = {"context_return": True} if context_return else {}

    def before_update(features):
        nonlocal attempted_updates, attempted_rows
        attempted_updates += 1
        attempted_rows += len(features)

    def after_update():
        nonlocal returned_updates
        returned_updates += 1

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
            if policy_change or context_return:
                phase_definitions = (
                    CONTEXT_RETURN_PHASES if context_return else POLICY_CHANGE_PHASES
                )
                for index, row in enumerate(report["training_data"]):
                    row["policy_version"] = phase_definitions[index // 4][1]
            budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=12,
                    max_wall_seconds=30,
                    max_process_rss_bytes=512 * 1024**2,
                ),
                monotonic,
            )
            budget.attach_memory(sampler)
            learner = (
                CountedContextLearner(
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
                    method, len(FEATURES), len(CATALOG), before_update, after_update
                )
            )
            original_seed = learner.snapshot_state()
            if replay != "none":
                from scripts.ticket_workflow_rehearsal import make_rehearsal, require_progress

                helper, replay_budget = make_rehearsal(
                    replay, learner, budget, sampler, **rehearsal_options
                )

            def capture(point, state, batch):
                snapshot_states[point] = state

            stream = AttributionStream(
                learner,
                batches,
                (),
                checkpoint_updates=CHECKPOINTS,
                label_delay=1,
                budget=budget,
                observe_checkpoint=capture,
                training_only=True,
                **({} if helper is None else {"rehearsal": helper}),
            )
            actor_pin = (
                stream._runtime.actor.snapshot_state() if policy_change or context_return else None
            )
            if context_return:
                input_pin = deepcopy((spec, batches, report["training_data"]))
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

                def context_guard():
                    if not _same_value((spec, batches, report["training_data"]), input_pin):
                        raise ValueError("context TRAIN input changed")
                    if (
                        not _same_value(np.random.get_state(), rng)
                        or not same_state(learner.snapshot_state(), original_seed)
                        or not same_state(stream._runtime.actor.snapshot_state(), actor_pin)
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
                        ("episodes", paid_episodes, episode_pins),
                        ("checkpoints", paid_checkpoints, checkpoint_pins),
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
                            **rehearsal_options,
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
                        **rehearsal_options,
                    )
                    replay_pin = helper.report
                report["checkpoints"].append(
                    dict(
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
                    ContextPerceptronDecisionPredictor(spec, state)
                    if separate_context_heads
                    else PerceptronDecisionPredictor(spec, state)
                    if native is None
                    else native.readout(spec, state)
                )
                if native is not None:
                    report["native_readout_work"].append(
                        dict(source_updates=point, work=asdict(predict.work))
                    )
                readout_pin = deepcopy(predict) if context_return else None

                def require_source():
                    context_guard()
                    if context_return and not _same_value(predict, readout_pin):
                        raise ValueError("context frozen predictor changed")
                    if (
                        not same_state(stream.candidate_snapshot(), state)
                        or budget.updates_completed != point
                    ):
                        raise ValueError(
                            "frozen task readout changed original TRAIN source or budget"
                        )
                    if (native is not None or policy_change) and not same_state(
                        learner.snapshot_state(), original_seed
                    ):
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
                            **rehearsal_options,
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
                        if context_return:
                            require_source()
                    require_source()
                    return result

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
            report["stream"] = stream.finish()
            if not same_state(learner.snapshot_state(), original_seed):
                raise ValueError("original seed learner changed")
            if policy_change or context_return:
                if not same_state(stream._runtime.actor.snapshot_state(), actor_pin):
                    raise ValueError("original actor changed at finish")
                report.update(source_unchanged=True, actor_unchanged=True)
            report["summary"] = (
                summarize_context_return(report["episodes"])
                if context_return
                else summarize_policy_changes(report["episodes"])
                if policy_change
                else summarize(report["episodes"])
            )
            report["complete"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
            if helper is not None:
                report["preserved_replay_receipts"] = previous_receipts
        finally:
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
                    report["applied_updates"] = [asdict(row) for row in stream.applied_updates]
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
                    context_routing="each homogeneous TRAIN/readout routes by public onehot; typed requests add full-D9 encode before original D7 readout; copies/projection/validation counted separately in meter",
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
                report["costs"].update(native.native_costs(report, attempted_rows, method))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replay", choices=("none", "recent", "coverage"), default="none")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--policy-change", action="store_true")
    modes.add_argument("--context-return", action="store_true")
    parser.add_argument("--separate-context-heads", action="store_true")
    parser.add_argument(
        "--method",
        choices=("perceptron", "bp", "pc", "neutral_cpc", "gated_cpc"),
        default="perceptron",
    )
    args = parser.parse_args(argv)
    if args.policy_change and args.replay != "none":
        parser.error("--policy-change requires --replay none")
    if args.context_return and args.method != "perceptron":
        parser.error("--context-return requires --method perceptron")
    if args.separate_context_heads and (
        not args.context_return or args.method != "perceptron" or args.replay != "none"
    ):
        parser.error(
            "--separate-context-heads requires --context-return --method perceptron --replay none"
        )
    # Reserve before fitting; a failed case cannot overwrite an earlier result.
    with args.output.open("xb") as target:
        target.write(b"{}\n")
    if args.separate_context_heads:
        report = run_example(context_return=True, separate_context_heads=True)
    elif args.context_return:
        report = (
            run_example(context_return=True)
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
