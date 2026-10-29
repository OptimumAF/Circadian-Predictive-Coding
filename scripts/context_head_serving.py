"""Finite contextual serving slice through existing stream and promotion ports.

Known TRAIN templates only; qualified source weights are data, never restored
runtime authority. No probe invocation, final access, new scheduler or thread.
"""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from time import monotonic
from typing import Any

import numpy as np

from scripts.context_head_probe import HEAD_SOURCE_SHA, _source_bytes, head_states
from scripts.context_training_streams import PAIRED_NEW_TASKS
from scripts.run_ticket_workflow import CONTEXT_FEATURES, build_joint_context_batch, training_cases
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.adapters.context_serving import (
    ContextServingPredictor,
    build_context,
    categorical_guard,
    snapshot_digest,
    snapshot_record_bytes,
)
from src.adapters.perceptron_decision import build_decision_training_batch
from src.app.coupled_attribution import AttributionBatch, AttributionStream, _same_value
from src.app.arrived_shift_observer import ArrivedShiftObserver
from src.app.serving_promotion import PromotableActor, PromotionRejected, ServingPromotionController
from src.app.ticket_workflow import run_workflow_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.decision_features import DecisionFeatureSpec
from src.core.context_perceptron import predict_context_perceptron
from src.core.promotion_guard import GuardBatch, PromotionPolicy
from src.core.serving_ports import ServingConfiguration
from src.core.temporal_decision_sequence import VOCABULARY
from src.core.ticket_workflow import ACTION_IDS, CATALOG, WorkflowCapabilities, WorkflowState
from src.shared.process_memory import ProcessRssSampler

TASKS = (
    ("v1", "invoice"),
    ("v1", "password"),
    ("v2", "invoice"),
    ("v2", "parcel"),
    ("v1", "legacy"),
    ("v2", "legacy"),
)
PAIRED_TASKS = (*TASKS[:4], *PAIRED_NEW_TASKS)


def policy():
    return PromotionPolicy(
        "global_categorical_accuracy",
        "complete_context_record_bytes",
        1.0,
        0.1,
        0.0,
        0.25,
        4096,
        ACTION_IDS,
    )


def prepare_data(spec, *, head_task_transfer=False):
    if type(head_task_transfer) is not bool:
        raise ValueError("paired ordinary TRAIN requires an exact boolean opt-in")
    if head_task_transfer and (
        type(spec) is not DecisionFeatureSpec
        or spec != DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)
    ):
        raise ValueError("paired ordinary TRAIN requires original D9/catalog schema")
    teachers = {
        v: training_cases(policy_version=v, include_policy_context=True) for v in ("v1", "v2")
    }
    _, old = build_joint_context_batch(spec, teachers)
    tasks = PAIRED_NEW_TASKS if head_task_transfer else (("v1", "legacy"), ("v2", "legacy"))
    fresh = tuple(
        build_decision_training_batch(
            teachers[v][3 * VOCABULARY.index(c) : 3 * VOCABULARY.index(c) + 3],
            spec,
            f"{c}-{v}",
            allow_subsets=True,
        )
        for v, c in tasks
    )
    if head_task_transfer:
        train_packets = []
        for i, (version, concept) in enumerate(tasks):
            for context, base, start, concepts in (
                (f"new-{version}", fresh[i], 0, (concept,)),
                (
                    f"old-{version}",
                    old,
                    i * 6,
                    ("invoice", "password") if i == 0 else ("invoice", "parcel"),
                ),
            ):
                n = 3 * len(concepts)
                train_packets.append(
                    AttributionBatch(
                        "train",
                        context,
                        tuple(
                            f"train:{version}:{c}:{stage}"
                            for c in concepts
                            for stage in ("sealed", "ready", "staged")
                        ),
                        base.features[start : start + n].copy(),
                        base.targets[start : start + n].copy(),
                        num_classes=6,
                    )
                )
        training = tuple(train_packets)
    else:
        training = tuple(
            AttributionBatch(
                "train",
                name,
                tuple(f"update-{i}-{r}" for r in range(n)),
                base.features[:n].copy(),
                base.targets[:n].copy(),
                num_classes=6,
            )
            for i, (name, base, n) in enumerate((("new-v1", fresh[0], 3), ("old-v1", old, 6)))
        )
    new_features = np.concatenate(tuple(b.features for b in fresh))
    new_targets = np.concatenate(tuple(b.targets for b in fresh))
    guards = tuple(
        GuardBatch(
            name,
            tuple((f"guard-{name}", str(i)) for i in range(len(x))),
            0,
            x.copy(),
            y.copy(),
            "inner_guard",
        )
        for name, x, y in (("new", new_features, new_targets), ("old", old.features, old.targets))
    )
    return training, guards


def _episodes(predictor, phase, require, rows, *, role=None, tasks=None):
    if tasks is not None and (type(tasks) is not tuple or tasks != PAIRED_TASKS):
        raise ValueError("selected serving tasks require the fixed paired TRAIN family")
    if role is not None and (type(role) is not str or role != "controller_train"):
        raise ValueError("explicit development episodes require controller_train role")
    for version, concept in TASKS if tasks is None else tasks:
        require()
        reads: list[dict[str, Any]] = []
        row = dict(
            phase=phase, policy_version=version, concept=concept, reads=reads, complete=False
        )
        if role is not None:
            row["role"] = role
        rows.append(row)

        def predict(request):
            require()
            result = predictor.serve(request)
            reads.append(
                dict(
                    generation=result.generation,
                    actor_version=result.actor_version,
                    completed_updates=result.completed_updates,
                )
            )
            require()
            return result.result

        episode = run_workflow_episode(
            WorkflowState(f"{phase}-{version}-{concept}", concept, policy_version=version),
            predict,
            capabilities=WorkflowCapabilities(ACTION_IDS),
            max_steps=6,
            include_policy_context=True,
        )
        row.update(episode=episode, complete=episode.get("complete") is True)
        if not row["complete"]:
            raise ValueError("context serving episode incomplete; paid prefix retained")
        require()


def run_arm(mode, initial, spec, data, sampler, require_source, row, *, shift_diagnostics=False):
    if type(shift_diagnostics) is not bool:
        raise ValueError("shift_diagnostics must be an exact bool")
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=2 if mode == "guarded" else 0,
            max_wall_seconds=45,
            max_process_rss_bytes=512 * 1024**2,
        ),
        monotonic,
    )
    budget.attach_memory(sampler)
    row["training_work"] = dict(attempted_updates=0, attempted_rows=0, returned_updates=0)

    class CountedLearner(ContextPerceptronLearner):
        def fork(self):
            child = CountedLearner(self.policy)
            child.restore_state(self.snapshot_state())
            return child

        def train_batch(self, features, targets):
            require_source()
            row["training_work"]["attempted_updates"] += 1
            row["training_work"]["attempted_rows"] += len(features)
            result = super().train_batch(features, targets)
            row["training_work"]["returned_updates"] += 1
            return result

    source = CountedLearner(initial.policy)
    source.restore_state(initial)
    actor = PromotableActor(
        source,
        version="qualified-12",
        configuration=ServingConfiguration(3, 2),
        feature_digest=lambda x: sha256(x.tobytes()).hexdigest(),
        metadata={"source": HEAD_SOURCE_SHA, "history_updates": 12},
    )
    predictor = ContextServingPredictor(actor, spec)
    training, (new, old) = data
    stream = None
    observer = None
    receipt = None
    budget_pin = (id(budget.budget), budget.budget, budget.started_at, id(budget.clock))

    def require():
        require_source()
        budget.before_final()
        if source.snapshot_state() != initial or budget_pin != (
            id(budget.budget),
            budget.budget,
            budget.started_at,
            id(budget.clock),
        ):
            raise ValueError("original source owner or budget changed")

    try:
        if mode == "guarded":
            if shift_diagnostics:
                # Informational only: the original three-batch reference cannot
                # monitor this two-update slice. Do not tune it to force alarms.
                observer = ArrivedShiftObserver(
                    predict_context_perceptron, num_classes=6, max_updates=2
                )
            observer_options: dict[str, Any] = (
                {"observe_checkpoint": observer} if observer is not None else {}
            )
            stream = AttributionStream(
                source,
                training,
                (),
                checkpoint_updates=(0, 1, 2),
                label_delay=1,
                budget=budget,
                training_only=True,
                actor=actor,
                **observer_options,
            )
        controller = ServingPromotionController(
            actor,
            policy=policy(),
            evaluator=categorical_guard(spec, monotonic),
            build_learner=build_context,
            state_digest=snapshot_digest,
            max_pending=1,
        )
        for point, phase in enumerate(("initial", "after-new", "after-old")):
            require()
            if point and stream is not None:
                stream.run_until(point)
                state = stream.candidate_snapshot()
                row["candidate_states"].append(asdict(state))
                if point == 2:
                    row["stream"] = stream.finish()
                try:
                    ticket = controller.prepare(
                        stream._runtime, new, old, now=point, training_ids=frozenset()
                    )
                except PromotionRejected as error:
                    row["guards"].append(dict(point=point, report=asdict(error.report)))
                else:
                    row["guards"].append(dict(point=point, report=asdict(ticket.report)))
                    # Fixed publication boundary: first accepted ticket is only
                    # measured/discarded; final publication never feeds training.
                    if point == 2:
                        receipt = controller.commit(stream._runtime, ticket)
                        row["receipt"] = asdict(receipt)
                    else:
                        controller.discard(ticket)
            _episodes(predictor, phase, require, row["episodes"])
            row["publications"].append(asdict(actor.serving_snapshot()))
        if receipt is not None:
            before = actor.serving_snapshot()
            row["rollback_generation"] = controller.rollback(receipt)
            after = actor.serving_snapshot()
            row["rollback_state_equal_source"] = after.model_state == initial
            _episodes(predictor, "rollback", require, row["episodes"])
            row["rollback"] = dict(before=asdict(before), after=asdict(after))
        require()
        row["complete"] = True
    finally:
        row.update(
            typed_work=dict(predictor.work),
            source_unchanged=source.snapshot_state() == initial,
            budget=dict(
                limits=asdict(budget.budget),
                updates=budget.updates_completed,
                started_at=budget.started_at,
                last_clock=budget.last_clock,
            ),
            final_actor=asdict(actor.serving_snapshot()),
        )
        if stream is not None:
            row["applied_updates"] = [asdict(u) for u in stream._runtime.applied_updates]
        if observer is not None:
            row["shift_observations"] = dict(
                native_update_origin=initial.completed_updates,
                prediction_clock="native_update_origin + relative prediction_state_updates",
                report=observer.report(),
            )


def run_case(input_path, *, shift_diagnostics=False):
    if type(shift_diagnostics) is not bool:
        raise ValueError("shift_diagnostics must be an exact bool")
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="guarded_context_serving_v1",
        complete=False,
        cases=[],
        source_sha256=HEAD_SOURCE_SHA,
        prior_training=dict(updates=12, rows=72, mistakes=16),
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
        scope="known TRAIN template guard/ordinary task demonstration; no concurrent worker, heldout, matched-compute or CPC claim",
    )
    if shift_diagnostics:
        report["shift_diagnostics"] = dict(
            enabled=True,
            mode="guarded_arrived_train_only",
            num_classes=6,
            monitoring_possible=False,
            reason="two updates; original reference requires three",
            feedback=False,
        )
    with ProcessRssSampler(interval_seconds=0.02) as sampler:
        try:
            raw = _source_bytes(input_path)
            if sha256(raw).hexdigest() != HEAD_SOURCE_SHA:
                raise ValueError("qualified contextual source replaced")
            initial = head_states(json.loads(raw))[0]
            spec = DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG)
            data = prepare_data(spec)
            pin = deepcopy(data)
            rng = deepcopy(np.random.get_state())

            def require_source():
                if monotonic() - started > 120:
                    raise ValueError("original serving case deadline exhausted")
                if _source_bytes(input_path) != raw or not _same_value(data, pin):
                    raise ValueError("original source or TRAIN/guard data changed")
                if not _same_value(np.random.get_state(), rng):
                    raise ValueError("caller NumPy RNG changed")

            report.update(
                input_bytes=sum(b.features.nbytes + b.targets.nbytes for b in data[0])
                + sum(b.features.nbytes + b.targets.nbytes for b in data[1]),
                coefficient_bytes=768,
                source_record_bytes=snapshot_record_bytes(initial),
                promotion_policy=asdict(policy()),
            )
            for mode in ("frozen", "guarded"):
                require_source()
                row: dict[str, Any] = dict(
                    mode=mode,
                    complete=False,
                    episodes=[],
                    guards=[],
                    candidate_states=[],
                    publications=[],
                )
                report["cases"].append(row)
                run_arm(
                    mode,
                    initial,
                    spec,
                    data,
                    sampler,
                    require_source,
                    row,
                    **({"shift_diagnostics": True} if shift_diagnostics else {}),
                )
                require_source()
            report["complete"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            report["elapsed_seconds"] = monotonic() - started
            report["memory"] = asdict(sampler.snapshot())
    return report
