"""Three declared context exposures through the existing inbox and tool runner.

Inputs: one loaded frozen encoder/head source and ordinary reporting callbacks.
Outputs: scalar fit/episode costs and outcomes, including incomplete prefixes.
No final/probe selection, model export, new scheduler or scientific allowance.
"""

from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any

import torch

from scripts.clm_linear_workflow import _head_traits, _json
from scripts.run_ticket_workflow import CONTEXT_TASKS, training_cases
from src.adapters.clm_frozen_encoder import FrozenCLMEncoder
from src.adapters.clm_decision import _action_feature_pin, _require_action_feature_pin
from src.adapters.clm_head_learner import _equal
from src.adapters.clm_linear_catalog_learner import CLMLinearCatalogLearner
from src.adapters.clm_linear_decision import (
    CLMLinearDecisionPredictor,
    _encoder_return_work,
    _same_work,
)
from src.adapters.clm_linear_fit_control import _capture, _require_original
from src.app.experience_inbox import ExperienceInbox
from src.app.ticket_workflow import run_workflow_episode
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.clm_action_features import FrozenActionFeatures
from src.core.clm_catalog import CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_linear_catalog import CLMLinearCatalogPolicy, OBJECTIVE, linear_catalog_work
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.temporal_decision_sequence import VOCABULARY, policy_for_phase
from src.core.ticket_workflow import ACTION_IDS, CATALOG, WorkflowCapabilities, WorkflowState


def _teacher(version, instructions):
    """Validate the original TRAIN declarations before any encoding."""
    cases = training_cases(
        policy_version=version, include_policy_context=True, public_observation_train=True
    )
    policy = policy_for_phase("after_A" if version == "v1" else "after_return")
    active = tuple(
        c for c in CATALOG if c.candidate_id in (*policy["allowed_ids"], "open", "commit")
    )
    expected = tuple(
        tool for concept in VOCABULARY for tool in ("open", policy["rules"][concept], "commit")
    )
    group = (
        "train-workflow-traces"
        + ("-v1" if version == "v1" else "")
        + "-context-public-observation-v1"
    )
    if (
        type(cases) is not tuple
        or len(cases) != 12
        or any(c.role != "train" or c.source_group != group for c in cases)
        or tuple(c.target_id for c in cases) != expected
        or any(c.request.allowed_candidates != active for c in cases)
    ):
        raise ValueError("original public context TRAIN teacher declarations differ")
    texts = tuple(
        dict.fromkeys(c.request.query.strip() + "\n\n" + instructions.strip() for c in cases)
    )
    if len(texts) != 6 or any("policy_" + version not in text.split() for text in texts):
        raise ValueError("original public context TRAIN text geometry differs")
    return cases, texts


def _batch(encoder, cases, texts, values, actions, instructions):
    return (
        FrozenCatalogTrainingBatch(
            encoder.identity,
            values[
                [
                    texts.index(c.request.query.strip() + "\n\n" + instructions.strip())
                    for c in cases
                ]
            ].clone(),
            actions.clone(),
            ACTION_IDS,
        ),
        CLMCatalogTargets(ACTION_IDS, tuple(ACTION_IDS.index(c.target_id) for c in cases)),
    )


def _summary(episodes):
    stages = {
        phase: [
            r["episode"]["success"]
            for r in episodes
            if r["phase"] == phase and r["scenario"] == "ordinary"
        ]
        for phase in ("after_A", "after_B", "after_return")
    }
    # Why: returning after explicit rehearsal is reacquisition evidence only if
    # the same old task actually failed before that rehearsal.
    recovered = [a and not b and r for a, b, r in zip(*stages.values(), strict=True)]
    return dict(
        task_order=[list(task) for task in CONTEXT_TASKS],
        ordinary_successes=stages,
        old_context_after_B=stages["after_B"][:2],
        learning_recovery_demonstrated=any(recovered[:2]),
        stale_episode_recoveries=sum(r["episode"]["error_recovery_demonstrated"] for r in episodes),
        permission_denials=sum(
            r["episode"]["stopped_reason"] == "permission_denied" for r in episodes
        ),
    )


def _container_owners(value):
    """Retain bounded report container identities as well as their JSON values."""
    if type(value) is dict:
        return (id(value), *(_container_owners(v) for v in value.values()))
    if type(value) in (list, tuple):
        return (id(value), *(_container_owners(v) for v in value))
    return None


def run_context_loaded(
    loaded, report, *, instructions, before_step=lambda: None, save=lambda: None
):
    if (
        type(report) is not dict
        or any(k in report for k in ("identity", "training", "episodes"))
        or not callable(before_step)
        or not callable(save)
    ):
        raise ValueError("context workflow requires unused report keys and callable boundaries")
    source = (
        loaded.model,
        loaded.tokenizer,
        loaded.state_head,
        loaded.action_head,
        loaded.logit_scale,
        loaded.identity,
    )
    identity = deepcopy(loaded.identity)
    heads = deepcopy(
        (loaded.state_head.state_dict(), loaded.action_head.state_dict(), loaded.logit_scale)
    )
    head_traits, rng = _head_traits(loaded), torch.get_rng_state().clone()
    encoder = FrozenCLMEncoder(
        loaded.model,
        loaded.tokenizer,
        identity=loaded.identity,
        max_sequences=16,
        max_forward_calls=86,
    )
    clock, limits = LogicalClock(), ToyExecutionBudget(max_training_updates=3)
    budget = ToyBudgetSession(limits, lambda: 0.0)
    budget_birth = (budget.budget, budget.clock, budget.started_at)
    permissions = ExperiencePermissions(training=True, replay=True)
    provider: Any = None
    learner: Any = None
    inbox: Any = None
    pins: list[Any] = []
    action_features, action_pin, action_values = None, None, None
    proofs: dict[str, Any] = {}
    record_owners: dict[str, Any] = {}
    expected_encoder = encoder.work
    expected_learner: dict[str, Any] = {}
    expected_provider: dict[str, Any] = {}
    expected_updates, tick, attempts = 0, 0, 0
    expected_receipts: tuple = ()
    succeeded = False
    report.update(
        complete=False,
        objective=OBJECTIVE,
        identity=asdict(identity),
        training=[],
        episodes=[],
        pending_update=None,
        pending_episode=None,
        logical_prediction_attempts=0,
        exposure=dict(
            original_teacher_rows=24,
            teacher_dispatch_transitions=24,
            repeated_return_rows=12,
            total_train_rows=36,
            preparation_sequences=18,
            replay_consent_from_birth=True,
            return_is_novel_data=False,
        ),
        scope="TRAIN v1/v2/repeated v1; development episodes only; no final/probe/CPC or native usefulness claim",
    )

    def protect(key):
        value = report[key]
        proofs[key] = value, _json(value)
        record_owners[key] = _container_owners(value)

    def refresh():
        report["encoder_work"] = encoder.work
        report["learner_work"] = {} if learner is None else learner.work
        report["predictor_work"] = {} if provider is None else provider.work
        report["linear_last_operation"] = {} if learner is None else learner.last_operation
        report["budget"] = dict(
            updates_completed=budget.updates_completed,
            max_training_updates=3,
            logical_tick=clock.now(),
        )
        for key in (
            "encoder_work",
            "learner_work",
            "predictor_work",
            "linear_last_operation",
            "budget",
        ):
            protect(key)

    def guard(enc=None, work=None, pred=None, updates=None, receipts=None):
        if not _same_work(encoder.work, expected_encoder if enc is None else enc):
            raise ValueError("context encoder work differs from its observed return")
        if learner is not None:
            learner._ready()
            if not _same_work(learner.work, expected_learner if work is None else work):
                raise ValueError("context learner work differs from its observed solve")
        if provider is not None:
            provider._check()
            if not _same_work(provider.work, expected_provider if pred is None else pred):
                raise ValueError("context provider work differs from its observed request")
        for key, (owner, proof) in proofs.items():
            value = report.get(key)
            if value is not owner or _json(value) != proof:
                raise ValueError("original context workflow prefix changed: " + key)
            if _container_owners(value) != record_owners[key]:
                raise ValueError("original context workflow record replaced: " + key)
        if (
            report.get("logical_prediction_attempts") != attempts
            or type(report.get("logical_prediction_attempts")) is not int
        ):
            raise ValueError("context request clock changed")
        if (
            any(
                a is not b
                for a, b in zip(
                    source,
                    (
                        loaded.model,
                        loaded.tokenizer,
                        loaded.state_head,
                        loaded.action_head,
                        loaded.logit_scale,
                        loaded.identity,
                    ),
                    strict=True,
                )
            )
            or asdict(loaded.identity) != asdict(identity)
            or _head_traits(loaded) != head_traits
            or not _equal(
                heads,
                (
                    loaded.state_head.state_dict(),
                    loaded.action_head.state_dict(),
                    loaded.logit_scale,
                ),
            )
            or not torch.equal(torch.get_rng_state(), rng)
        ):
            raise ValueError("original context source or caller RNG changed")
        if (
            budget.budget is not budget_birth[0]
            or budget.clock is not budget_birth[1]
            or budget.started_at != budget_birth[2]
            or asdict(limits) != asdict(ToyExecutionBudget(max_training_updates=3))
            or budget.updates_completed != (expected_updates if updates is None else updates)
            or budget.replay_examples_completed != 0
            or clock.now() != tick
            or asdict(permissions) != dict(training=True, replay=True, evaluation=False)
        ):
            raise ValueError("original context consent, clocks or lifetime budget changed")
        if inbox is not None:
            if (
                inbox._learner is not learner
                or inbox._clock is not clock
                or inbox._budget is not budget
            ):
                raise ValueError("original context inbox owners changed")
            inbox._require_training_scope()
            desired = expected_receipts if receipts is None else receipts
            actual = inbox.applied_updates
            if len(actual) != len(desired) or any(
                a is not b for a, b in zip(actual, desired, strict=True)
            ):
                raise ValueError("original context receipts replaced")
            if [asdict(r) for r in actual] != [
                r["receipt"] for r in report["training"]
            ] and receipts is None:
                raise ValueError("context receipt values changed")
        encoder._check()
        if action_features is not None:
            _require_action_feature_pin(encoder.identity, action_features, action_pin)
            if not torch.equal(action_features.features, action_values):
                raise ValueError("original union action feature values changed")
        for batch, targets, pin in pins:
            _require_original(batch, targets, pin)

    def checkpoint():
        guard()
        save()
        guard()

    def boundary():
        guard()
        before_step()
        guard()

    def fail(error):
        report.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])
        for key in ("training", "episodes", "pending_update", "pending_episode", "summary"):
            if key in proofs:
                report["preserved_" + key] = json.loads(proofs[key][1])

    def run_episode(phase, version, concept, scenario):
        nonlocal provider, expected_provider, expected_encoder, attempts
        boundary()
        active = tuple(
            c
            for c in CATALOG
            if c.candidate_id
            in (
                *policy_for_phase("after_A" if version == "v1" else "after_return")["allowed_ids"],
                "open",
                "commit",
            )
        )
        provider = CLMLinearDecisionPredictor(
            encoder,
            learner.current_control,
            action_features,
            instructions=instructions,
            max_requests=6,
            active_candidates=active,
        )
        expected_provider = provider.work
        refresh()
        report["pending_episode"] = dict(
            phase=phase,
            policy_version=version,
            concept=concept,
            scenario=scenario,
            returned=False,
            effects_unknown=True,
        )
        protect("pending_episode")
        checkpoint()

        def predict(request):
            nonlocal expected_encoder, expected_provider, attempts
            boundary()
            attempts += 1
            report["logical_prediction_attempts"] = attempts
            try:
                result = provider(request)
                enc, _ = _encoder_return_work(encoder, expected_encoder, ("tail",))
                pred = dict(expected_provider)
                for key in (
                    "request_attempts",
                    "request_returns",
                    "encoder_attempts",
                    "encoder_returns",
                    "linear_readout_attempts",
                    "linear_readout_returns",
                    "linear_rows",
                    "active_catalog_readout_attempts",
                    "active_catalog_readout_returns",
                ):
                    pred[key] += 1
                guard(enc=enc, pred=pred)
                expected_encoder, expected_provider = enc, pred
                refresh()
                checkpoint()
                return result
            except Exception:
                # Terminal accounting is observed, never admitted as a new success.
                refresh()
                raise

        episode = run_workflow_episode(
            WorkflowState(f"context-{len(report['episodes'])}", concept, policy_version=version),
            predict,
            capabilities=WorkflowCapabilities(ACTION_IDS),
            max_steps=6,
            scenario=scenario,
            include_policy_context=True,
        )
        if not episode["complete"]:
            report["pending_episode"] = dict(
                report["pending_episode"], returned=True, episode=episode
            )
            protect("pending_episode")
            raise RuntimeError("context episode contains uncertain paid work")
        guard()
        report["episodes"].append(
            dict(
                phase=phase,
                policy_version=version,
                concept=concept,
                scenario=scenario,
                role="outer_selection",
                episode=episode,
                predictor_work=provider.work,
            )
        )
        protect("episodes")
        report["pending_episode"] = None
        protect("pending_episode")
        checkpoint()
        provider, expected_provider = None, {}
        refresh()

    for key in (
        "objective",
        "identity",
        "exposure",
        "scope",
        "training",
        "episodes",
        "pending_update",
        "pending_episode",
    ):
        protect(key)
    refresh()
    try:
        boundary()
        # Validate both teachers before either paid encoder call.
        a_cases, a_texts = _teacher("v1", instructions)
        b_cases, b_texts = _teacher("v2", instructions)
        values = encoder.embed_texts(
            a_texts + tuple(c.description for c in CATALOG), ("tail",) * 6 + ("head",) * 6
        )
        enc, _ = _encoder_return_work(encoder, expected_encoder, ("tail",) * 6 + ("head",) * 6)
        guard(enc=enc)
        expected_encoder = enc
        refresh()
        action_features = FrozenActionFeatures(encoder.identity, CATALOG, values[6:].clone())
        action_pin = _action_feature_pin(encoder.identity, action_features)
        action_values = action_features.features.clone()
        a = _batch(encoder, a_cases, a_texts, values[:6], action_features.features, instructions)
        pins.append((*a, _capture(*a)))
        boundary()
        b_values = encoder.embed_texts(b_texts, ("tail",) * 6)
        enc, _ = _encoder_return_work(encoder, expected_encoder, ("tail",) * 6)
        guard(enc=enc)
        expected_encoder = enc
        refresh()
        b = _batch(encoder, b_cases, b_texts, b_values, action_features.features, instructions)
        pins.append((*b, _capture(*b)))
        learner = CLMLinearCatalogLearner(
            encoder.identity,
            action_features.features,
            (),
            candidate_ids=ACTION_IDS,
            policy=CLMLinearCatalogPolicy(max_updates=3, max_rows=12),
        )
        expected_learner = learner.work
        inbox = ExperienceInbox(
            learner,
            clock=clock,
            budget=budget,
            learner_version="clm-context-linear",
            max_experiences=3,
        )
        refresh()
        for index, (phase, version, pair) in enumerate(
            (("after_A", "v1", a), ("after_B", "v2", b), ("after_return", "v1", a))
        ):
            boundary()
            tick = 2 * index
            clock.advance_to(tick)
            source_event = Experience(
                phase,
                "context-training",
                tick,
                "clm-context-source",
                pair[0],
                "train",
                permissions,
                ACTION_IDS,
            )
            inbox.record_experience(source_event)
            if inbox.drain(max_updates=1):
                raise ValueError("context training occurred before label arrival")
            report["pending_update"] = dict(
                phase=phase,
                policy_version=version,
                repeated_rows=12 if index == 2 else 0,
                observed_at=tick,
                returned=False,
            )
            protect("pending_update")
            refresh()
            checkpoint()
            boundary()
            tick += 1
            clock.advance_to(tick)
            inbox.record_label(
                LabelArrival(
                    "label-" + phase, phase, "context-training", tick, "clm-context-source", pair[1]
                )
            )
            receipts = inbox.drain(max_updates=1)
            if (
                len(receipts) != 1
                or receipts[0].observed_at != tick - 1
                or receipts[0].arrived_at != tick
                or receipts[0].applied_at != tick
            ):
                raise ValueError("context update lacks original delayed-label receipt")
            work = linear_catalog_work(index + 1, 12 * (index + 1))
            all_receipts = (*expected_receipts, receipts[0])
            guard(work=work, updates=index + 1, receipts=all_receipts)
            expected_learner, expected_updates, expected_receipts = work, index + 1, all_receipts
            report["training"].append(
                dict(
                    phase=phase,
                    policy_version=version,
                    repeated_rows=12 if index == 2 else 0,
                    receipt=asdict(receipts[0]),
                    operation=learner.last_operation,
                )
            )
            protect("training")
            report["pending_update"] = None
            protect("pending_update")
            refresh()
            checkpoint()
            for task_version, concept in CONTEXT_TASKS:
                run_episode(phase, task_version, concept, "ordinary")
        run_episode("after_return", "v1", "password", "stale_once")
        run_episode("after_return", "v2", "parcel", "revoke_commit")
        report["summary"] = _summary(report["episodes"])
        protect("summary")
        guard()
        report["complete"] = True
        succeeded = True
    except Exception as error:
        fail(error)
        raise
    finally:
        refresh()
        try:
            save()
            if succeeded:
                guard()
                if report.get("complete") is not True:
                    raise ValueError("context completion changed during final save")
        except Exception as error:
            fail(error)
            raise
    return report
