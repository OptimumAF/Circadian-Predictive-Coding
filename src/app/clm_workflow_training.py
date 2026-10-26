"""Fixed catalog TRAIN batches through the existing delayed inbox.

Consumes already-bound frozen features and a fresh native learner. Returns actual
paid receipts and optionally summarizes fixed TRAIN probabilities; no backend,
encoder/loading, outer evaluation, promotion, episode effects or replay.
"""

from copy import deepcopy
from dataclasses import asdict
import json
from math import isclose, isfinite
from typing import Any

from src.app.clm_reserved_probe import (
    _feature_pin,
    _require_features,
    _tensor_traits,
    _require_work,
)
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.clm_catalog import CATALOG_OBJECTIVE, CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_head import _exact
from src.core.clm_training_observation import CLMObservedUpdate, GRADIENT_DEFINITION
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.temporal_decision_sequence import VOCABULARY, policy_for_phase
from src.core.ticket_workflow import CATALOG

ACTIVE_IDS = tuple(c.candidate_id for c in CATALOG if c.candidate_id != "retire")
PASSES = 8
MAX_PASSES = 32


def expected_targets():
    rules = policy_for_phase("after_return")["rules"]
    return tuple(
        ACTIVE_IDS.index(action)
        for concept in VOCABULARY
        for action in ("open", rules[concept], "commit")
    )


def summarize_training_fit(probabilities):
    """Reduce the fixed twelve TRAIN rows; margins are probabilities, not logits.

    No model, feature extraction, fitting, thresholds or checkpoint selection.
    The phase/concept order is the original teacher order, including duplicates.
    """
    if type(probabilities) is not list or len(probabilities) != 12:
        raise ValueError("TRAIN fit requires twelve full-catalog probability rows")
    rows: list[dict[str, Any]] = []
    phases = ("sealed", "ready", "staged")
    for index, (values, target) in enumerate(zip(probabilities, expected_targets(), strict=True)):
        if (
            type(values) is not list
            or len(values) != 5
            or any(type(p) is not float or not isfinite(p) or not 0 <= p <= 1 for p in values)
            or not isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise ValueError("TRAIN fit requires complete finite probability distributions")
        winner = min(range(5), key=lambda candidate: (-values[candidate], ACTIVE_IDS[candidate]))
        competitor = max(p for candidate, p in enumerate(values) if candidate != target)
        rows.append(
            dict(
                row=index,
                phase=phases[index % 3],
                concept=VOCABULARY[index // 3],
                target_id=ACTIVE_IDS[target],
                selected_id=ACTIVE_IDS[winner],
                correct=winner == target,
                target_probability=values[target],
                best_other_probability=competitor,
                probability_margin=values[target] - competitor,
            )
        )
    by_phase = {}
    for phase in phases:
        group = [row for row in rows if row["phase"] == phase]
        by_phase[phase] = dict(
            rows=len(group),
            correct=sum(row["correct"] for row in group),
            mean_probability_margin=sum(row["probability_margin"] for row in group) / len(group),
        )
    return dict(rows=rows, by_phase=by_phase)


class _ObservedWorkflowTrainer:
    """Unwrap the original loss for the same inbox; retain one scalar return only."""

    def __init__(self, learner):
        self.learner = learner
        self.last_update: CLMObservedUpdate | None = None

    def train_batch(self, batch, targets):
        self.last_update = None
        result = self.learner.train_batch_observed(batch, targets)
        _exact(result, CLMObservedUpdate)
        if (
            result.observation.committed_update != self.learner.completed_updates
            or result.observation.training_rows != len(batch.states)
        ):
            raise ValueError("gradient observation differs from the returned native update")
        self.last_update = result
        return result.diagnostic


def fit_catalog_workflow(
    learner,
    batch,
    targets,
    report,
    *,
    passes=PASSES,
    before_step=lambda: None,
    save=lambda: None,
    record_gradient_norm=False,
):
    """One fresh 8..32-pass TRAIN allowance; failed work is never retried.

    The learner's policy must be born for this count. A longer OneCycle schedule
    is a distinct run, not a continuation of a spent eight-pass learner.
    """
    if type(record_gradient_norm) is not bool:
        raise ValueError("record_gradient_norm must be an exact boolean")
    if record_gradient_norm and not callable(getattr(learner, "train_batch_observed", None)):
        raise ValueError("workflow learner does not support returned gradient observations")
    if type(passes) is not int or not PASSES <= passes <= MAX_PASSES:
        raise ValueError("workflow passes must be an integer from 8 through 32")
    _exact(batch, FrozenCatalogTrainingBatch)
    _exact(targets, CLMCatalogTargets)
    batch.__post_init__()
    targets.__post_init__()
    if type(report) is not dict or "training" in report:
        raise ValueError("workflow training needs an unused report key")
    if (
        batch.candidate_ids != ACTIVE_IDS
        or targets.candidate_ids != ACTIVE_IDS
        or targets.candidate_indices != expected_targets()
        or batch.identity != learner.identity
    ):
        raise ValueError("fixed twelve current-policy workflow TRAIN rows required")
    if (
        learner.policy.max_updates != passes
        or learner.policy.max_rows != 12
        or learner.policy.max_restores != 0
        or learner.completed_updates != 0
        or learner.training_examples != 0
        or learner.generation != 0
        or learner.work["train_attempts"] != 0
        or learner.work["unreturned_work_unknown"]
    ):
        raise ValueError(f"workflow requires fresh exact {passes}-update TRAIN authority")
    if not callable(before_step) or not callable(save):
        raise ValueError("workflow training callbacks must be callable")
    _tensor_traits(batch.states, (12, learner.identity.embedding_dim))
    _tensor_traits(batch.actions, (5, learner.identity.embedding_dim))
    features = _feature_pin(batch)
    identity, policy = deepcopy(learner.identity), deepcopy(learner.policy)
    tick = 0
    clock = LogicalClock()
    limits = ToyExecutionBudget(max_training_updates=passes)
    # Why: comparing against the live limits object cannot detect in-place edits.
    expected_limits = deepcopy(limits)
    budget = ToyBudgetSession(limits, lambda: 0.0)
    birth = (budget.budget, budget.clock, budget.started_at)
    observed_trainer = _ObservedWorkflowTrainer(learner) if record_gradient_norm else None
    # Why: this inbox uses only TRAIN; the scalar wrapper grants no snapshot or
    # restore capability from the broader NativeLearner annotation.
    inbox_learner: Any = learner if observed_trainer is None else observed_trainer
    inbox = ExperienceInbox(
        inbox_learner,
        clock=clock,
        budget=budget,
        learner_version="clm-workflow",
        max_experiences=passes,
    )
    record = report["training"] = dict(
        complete=False,
        objective=CATALOG_OBJECTIVE,
        rows_per_update=12,
        candidate_ids=list(ACTIVE_IDS),
        receipts=[],
        committed_updates=0,
        logical_clock=0,
        learner_work=learner.work,
        permissions=asdict(ExperiencePermissions(training=True)),
    )
    if passes != PASSES:
        record["requested_passes"] = passes
    gradients: list[dict[str, Any]] = []
    gradient_owners: list[dict[str, Any]] = []
    returned_owner: CLMObservedUpdate | None = None
    gradient_proof = "[]"
    if record_gradient_norm:
        record["gradient_observation_definition"] = GRADIENT_DEFINITION
        record["gradient_observations"] = gradients
    proof = ""
    work = learner.work

    def encoded():
        return json.dumps(record, sort_keys=True, allow_nan=False)

    def guard():
        require_gradients()
        if observed_trainer is not None:
            if observed_trainer.last_update is not returned_owner:
                raise ValueError("original returned gradient observation owner changed")
            if returned_owner is not None:
                _exact(returned_owner, CLMObservedUpdate)
                if not gradients or asdict(returned_owner.observation) != gradients[-1]:
                    raise ValueError("original returned gradient observation changed")
        _require_work(learner.work, work)
        _require_features(batch, features)
        _exact(batch, FrozenCatalogTrainingBatch)
        _exact(targets, CLMCatalogTargets)
        if (
            report.get("training") is not record
            or (proof and encoded() != proof)
            or batch.identity != identity
            or learner.identity != identity
            or learner.policy != policy
            or batch.role != "train"
            or batch.candidate_ids != ACTIVE_IDS
            or targets.candidate_ids != ACTIVE_IDS
            or targets.candidate_indices != expected_targets()
        ):
            raise ValueError("original workflow TRAIN data, policy or paid prefix changed")
        if (
            budget.budget is not birth[0]
            or budget.clock is not birth[1]
            or budget.started_at != birth[2]
            or budget.budget != expected_limits
            or budget.replay_examples_completed != 0
            or clock.now() != tick
            or budget.updates_completed != len(inbox.applied_updates)
            or learner.completed_updates != budget.updates_completed
            or learner.generation != budget.updates_completed
            or learner.training_examples != 12 * budget.updates_completed
        ):
            raise ValueError("original workflow clock, budget or native work changed")

    def require_gradients():
        if not record_gradient_norm:
            return
        if (
            record.get("gradient_observations") is not gradients
            or len(gradients) != len(gradient_owners)
            or any(a is not b for a, b in zip(gradients, gradient_owners, strict=True))
            or json.dumps(gradients, sort_keys=True, allow_nan=False) != gradient_proof
            or record.get("gradient_observation_definition") != GRADIENT_DEFINITION
        ):
            raise ValueError("original gradient observation prefix or owner changed")

    def collect_gradient():
        nonlocal gradient_proof, returned_owner
        require_gradients()
        if observed_trainer is None or observed_trainer.last_update is None:
            return
        result = observed_trainer.last_update
        _exact(result, CLMObservedUpdate)
        observation = result.observation
        item = asdict(observation)
        if observation.committed_update == len(gradients):
            if not gradients or item != gradients[-1]:
                raise ValueError("original returned gradient observation changed")
            return
        if (
            observation.committed_update != len(gradients) + 1
            or observation.committed_update != budget.updates_completed
            or observation.committed_update != len(inbox.applied_updates)
            or result.diagnostic != inbox.applied_updates[-1].diagnostic
        ):
            raise ValueError("gradient observation lacks its original committed receipt")
        gradients.append(item)
        gradient_owners.append(item)
        returned_owner = result
        gradient_proof = json.dumps(gradients, sort_keys=True, allow_nan=False)

    def persist():
        nonlocal proof, work
        collect_gradient()
        work = learner.work
        record.update(
            receipts=[asdict(r) for r in inbox.applied_updates],
            committed_updates=budget.updates_completed,
            logical_clock=clock.now(),
            learner_work=work,
        )
        proof = encoded()
        save()
        guard()

    try:
        persist()
        for index in range(passes):
            guard()
            before_step()
            guard()
            tick = 2 * index
            clock.advance_to(tick)
            sample = f"workflow-pass{index}"
            inbox.record_experience(
                Experience(
                    sample,
                    "clm-workflow",
                    tick,
                    "workflow-v2",
                    batch,
                    "train",
                    ExperiencePermissions(training=True),
                )
            )
            if inbox.drain(max_updates=1):
                raise ValueError("workflow updated before its label arrived")
            tick += 1
            clock.advance_to(tick)
            inbox.record_label(
                LabelArrival(
                    "label-" + sample, sample, "clm-workflow", tick, "workflow-v2", targets, "train"
                )
            )
            try:
                applied = inbox.drain(max_updates=1)
                if len(applied) != 1 or applied[0].diagnostic.definition != CATALOG_OBJECTIVE:
                    raise ValueError("one actual catalog workflow receipt required")
            finally:
                persist()
        guard()
        if record_gradient_norm and len(gradients) != passes:
            raise ValueError("workflow did not return every committed gradient observation")
        record["complete"] = True
        persist()
    except Exception as error:
        if proof:
            report["preserved_training_prefix"] = json.loads(proof)
        record.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])
        raise
    finally:
        record.update(
            receipts=[asdict(r) for r in inbox.applied_updates],
            committed_updates=budget.updates_completed,
            logical_clock=clock.now(),
            learner_work=learner.work,
        )
        save()
    return record
