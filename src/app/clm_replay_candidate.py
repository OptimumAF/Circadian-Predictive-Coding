"""Finite paired-CLM TRAIN replay through existing inbox and budget ports.

Consumes original consented/applied history and an injected candidate factory.
Returns unscored before/after readouts and paid provenance. No backend import,
probe conversion, final data, publication, scoring or permission grant occurs.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import json
from math import isclose, isfinite
from typing import Any

from src.app.clm_temporal_sequence import (
    _feature_pin,
    _require_features,
    _tensor_traits,
)
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.clm_head import (
    CLMHeadPolicy,
    CLMTargets,
    FrozenEmbeddingBatch,
    FrozenChoiceBatch,
    _exact,
)
from src.core.experience import (
    Experience,
    LabelArrival,
    AppliedExperience,
    LogicalClock,
    ExperiencePermissions,
)
from src.core.learner_ports import TrainingDiagnostic

OBJECTIVE = "clm_paired_group_masked_symmetric_cross_entropy_v1"


@dataclass(frozen=True)
class CLMReplayRecord:
    experience: Experience
    label: LabelArrival
    receipt: AppliedExperience
    policy_version: str

    def __post_init__(self):
        _exact(self.experience, Experience)
        _exact(self.label, LabelArrival)
        if type(self.receipt) is not AppliedExperience or set(vars(self.receipt)) != {
            field.name for field in fields(AppliedExperience)
        }:
            raise ValueError("exact original applied replay receipt required")
        _exact(self.receipt.diagnostic, TrainingDiagnostic)
        for name in ("observed_at", "arrived_at", "applied_at", "update_number"):
            if type(getattr(self.receipt, name)) is not int:
                raise ValueError("replay receipt clocks/counts must be native integers")
        self.experience.__post_init__()
        self.label.__post_init__()
        permissions = self.experience.permissions
        _exact(permissions, ExperiencePermissions)
        permissions.__post_init__()
        if (
            self.experience.role != "train"
            or self.label.role != "train"
            or not permissions.training
            or not permissions.replay
        ):
            raise ValueError("replay requires original TRAIN and reuse consent")
        _exact(self.experience.features, FrozenEmbeddingBatch)
        _exact(self.label.targets, CLMTargets)
        self.experience.features.__post_init__()
        self.label.targets.__post_init__()
        if (
            type(self.policy_version) is not str
            or not self.policy_version
            or len(self.policy_version.encode("utf8")) > 128
        ):
            raise ValueError("replay requires a bounded declared policy version")
        for value in (
            self.experience.sample_id,
            self.experience.episode_id,
            self.experience.model_version,
            self.label.event_id,
            self.receipt.learner_version,
        ):
            if len(value.encode("utf8")) > 128:
                raise ValueError("replay identifiers exceed their bounded metadata allowance")


def _metadata(record):
    source, label = record.experience, record.label
    return dict(
        policy_version=record.policy_version,
        experience={key: value for key, value in vars(source).items() if key != "features"},
        label={key: value for key, value in vars(label).items() if key != "targets"},
        target_codes=label.targets.group_codes,
        identity=asdict(source.features.identity),
        receipt=asdict(record.receipt),
    )


def _history(history, source, policy, source_version, current_policy, at):
    if type(history) is not tuple or not 1 <= len(history) <= 16:
        raise ValueError("bounded original replay history tuple required")
    pins, keys = [], set()
    for record in history:
        _exact(record, CLMReplayRecord)
        record.__post_init__()
        e, label, receipt = record.experience, record.label, record.receipt
        if e.key in keys or e.key != label.key or e.model_version != label.model_version:
            raise ValueError("replay source keys/model versions must be original and unique")
        keys.add(e.key)
        if (
            record.policy_version != current_policy
            or receipt.learner_version != source_version
            or receipt.sample_id != e.sample_id
            or receipt.episode_id != e.episode_id
            or receipt.event_id != label.event_id
            or receipt.actor_version != e.model_version
            or receipt.observed_at != e.observed_at
            or receipt.arrived_at != label.arrived_at
            or not e.observed_at <= label.arrived_at <= receipt.applied_at <= at
            or type(receipt.update_number) is not int
            or not 1 <= receipt.update_number <= source.completed_updates
            or receipt.diagnostic.definition != OBJECTIVE
        ):
            raise ValueError(
                "replay requires current-policy, causally applied original TRAIN receipts"
            )
        batch = e.features
        rows = len(label.targets.group_codes)
        if batch.identity != source.identity or not 2 <= rows <= policy.max_rows:
            raise ValueError("replay rows or frozen encoder identity differ")
        for value in (batch.states, batch.actions):
            _tensor_traits(value, (rows, source.identity.embedding_dim))
        pins.append(
            (
                record,
                e,
                label,
                receipt,
                e.permissions,
                batch,
                label.targets,
                deepcopy(_metadata(record)),
                _feature_pin(batch),
            )
        )
    return tuple(pins)


def _require_history(history, pins):
    for record, pin in zip(history, pins, strict=True):
        owner, e, label, receipt, permissions, batch, targets, metadata, features = pin
        record.__post_init__()
        if (
            record is not owner
            or record.experience is not e
            or record.label is not label
            or record.receipt is not receipt
            or e.permissions is not permissions
            or e.features is not batch
            or label.targets is not targets
            or _metadata(record) != metadata
        ):
            raise ValueError("original replay history, owner, consent or receipt changed")
        _require_features(batch, features)


def _budget_birth(budget, replay_budget, updates, rows):
    if (
        type(budget) is not ToyBudgetSession
        or type(replay_budget) is not ToyBudgetSession
        or budget is replay_budget
    ):
        raise ValueError("separate original update and replay budget sessions required")
    for session, limits in (
        (budget, ToyExecutionBudget(max_training_updates=updates)),
        (replay_budget, ToyExecutionBudget(max_replay_examples=rows)),
    ):
        _exact(session.budget, ToyExecutionBudget)
        if (
            session.budget != limits
            or session.updates_completed != 0
            or session.replay_examples_completed != 0
            or session.progress is not None
            or session.process_rss_sampler is not None
            or session.checkpoint_position is not None
        ):
            raise ValueError("fresh exact update-only/replay-only authority required")
    return tuple(
        (session, session.budget, deepcopy(session.budget), session.clock, session.started_at)
        for session in (budget, replay_budget)
    )


def _readout_probabilities(value, batch):
    _tensor_traits(value, (batch.states.shape[0], batch.actions.shape[0]))
    result = value.detach().tolist()
    if any(
        any(not 0 <= p <= 1 for p in row) or not isclose(sum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        for row in result
    ):
        raise ValueError("candidate readout must return complete normalized probabilities")
    return result


def run_clm_replay_candidate(
    source,
    history,
    plan,
    readout,
    report,
    *,
    policy,
    source_version,
    current_policy,
    at,
    candidate_factory,
    budget,
    replay_budget,
    before_step=lambda: None,
    save=lambda: None,
):
    """One candidate and one finite replay plan; every failure stops this run."""
    _exact(policy, CLMHeadPolicy)
    if type(report) is not dict or report:
        raise ValueError("candidate requires a fresh report")
    if (
        type(plan) is not tuple
        or len(plan) != policy.max_updates
        or not 1 <= len(plan) <= 16
        or len(plan) == 10
        or policy.max_restores != 0
        or any(type(index) is not int or not 0 <= index < len(history) for index in plan)
    ):
        raise ValueError("finite original candidate plan/policy required")
    if (
        type(at) is not int
        or at < 0
        or not all(callable(f) for f in (candidate_factory, before_step, save))
    ):
        raise ValueError("candidate requires a causal cutoff and callable boundaries")
    pins = _history(history, source, policy, source_version, current_policy, at)
    policy_birth = deepcopy(policy)
    source_birth = (
        source.identity,
        source.policy,
        source.completed_updates,
        source.training_examples,
        source.generation,
        source.work,
    )
    _exact(readout, FrozenChoiceBatch)
    readout.__post_init__()
    if readout.identity != source.identity:
        raise ValueError("candidate readout encoder identity differs")
    if (
        not 1 <= readout.states.shape[0] <= policy.max_rows
        or not 2 <= readout.actions.shape[0] <= policy.max_rows
    ):
        raise ValueError("candidate readout rows exceed declared bounds")
    _tensor_traits(readout.states, (readout.states.shape[0], source.identity.embedding_dim))
    _tensor_traits(readout.actions, (readout.actions.shape[0], source.identity.embedding_dim))
    readout_pin = _feature_pin(readout)
    readout_birth = (readout.identity, deepcopy(readout.identity), readout.role)
    rows = sum(len(history[index].label.targets.group_codes) for index in plan)
    if rows > 1024:
        raise ValueError("candidate replay exposure exceeds1024rows")
    birth = _budget_birth(budget, replay_budget, len(plan), rows)
    report.update(
        definition="clm_paired_train_replay_candidate_v1",
        complete=False,
        construction_attempts=0,
        construction_returns=0,
        unreturned_construction_work_unknown=False,
        policy=asdict(policy),
        current_policy=current_policy,
        cutoff=at,
        origins=[deepcopy(_metadata(record)) for record in history],
        plan=list(plan),
        attempts=[],
        receipts=[],
        readouts=[],
        replay_rows_attempted=0,
    )
    proof = ""
    native_pin = None
    candidate: Any = None
    inbox: Any = None
    committed = attempted_rows = 0
    clock = LogicalClock(at)

    def snapshot():
        return json.dumps(
            {
                k: report[k]
                for k in (
                    "definition",
                    "policy",
                    "current_policy",
                    "cutoff",
                    "origins",
                    "plan",
                    "attempts",
                    "receipts",
                    "readouts",
                    "replay_rows_attempted",
                    "complete",
                    "construction_attempts",
                    "construction_returns",
                    "unreturned_construction_work_unknown",
                    "candidate_work",
                    "copy_work",
                )
            },
            sort_keys=True,
            default=lambda value: asdict(value),
            allow_nan=False,
        )

    def guard():
        _require_history(history, pins)
        _exact(readout, FrozenChoiceBatch)
        _require_features(readout, readout_pin)
        if (
            policy != policy_birth
            or readout.identity is not readout_birth[0]
            or readout.identity != readout_birth[1]
            or readout.role != readout_birth[2]
            or (
                source.identity,
                source.policy,
                source.completed_updates,
                source.training_examples,
                source.generation,
                source.work,
            )
            != source_birth
        ):
            raise ValueError("original source, candidate policy or readout metadata changed")
        if proof and snapshot() != proof:
            raise ValueError("completed replay evidence prefix changed")
        for session, limits, limit_value, timer, started in birth:
            if (
                session.budget is not limits
                or session.budget != limit_value
                or session.clock is not timer
                or session.started_at != started
                or type(session.started_at) is not float
                or not isfinite(session.started_at)
                or session.progress is not None
                or session.process_rss_sampler is not None
                or session.checkpoint_position is not None
                or session.hidden_width_observed is not None
                or session.peak_hidden_width_observed is not None
                or session.rejected_proposed_hidden_width is not None
            ):
                raise ValueError("original candidate clock or budget owner changed")
        if (
            budget.updates_completed != committed
            or budget.replay_examples_completed != 0
            or replay_budget.updates_completed != 0
            or replay_budget.replay_examples_completed != attempted_rows
        ):
            raise ValueError("original paid candidate/replay counters changed")
        if candidate is not None:
            if candidate.completed_updates != committed or candidate.generation != committed:
                raise ValueError("candidate progress differs from original committed prefix")
            if native_pin is not None and candidate.work != native_pin:
                raise ValueError("original paid native candidate work changed")

    def persist():
        nonlocal proof, native_pin
        native_pin = None if candidate is None else candidate.work
        report["candidate_work"] = None if candidate is None else candidate.work
        report["copy_work"] = None if candidate is None else candidate.copy_work
        proof = snapshot()
        save()
        guard()

    def boundary():
        guard()
        before_step()
        guard()

    def observe(phase):
        boundary()
        value = candidate.predict_choices(readout)
        report["readouts"].append(
            dict(
                phase=phase,
                completed_updates=committed,
                probabilities=_readout_probabilities(value, readout),
            )
        )
        persist()

    try:
        persist()
        boundary()
        report["construction_attempts"] = 1
        report["unreturned_construction_work_unknown"] = True
        candidate = candidate_factory(source, policy=policy, max_readouts=2)
        if (
            candidate.definition != OBJECTIVE
            or candidate.identity != source.identity
            or candidate.policy != policy
            or candidate.completed_updates != 0
            or candidate.training_examples != 0
            or candidate.generation != 0
        ):
            raise ValueError("fresh paired candidate differs from declared source/policy")
        report["construction_returns"] = 1
        report["unreturned_construction_work_unknown"] = False
        inbox = ExperienceInbox(
            candidate,
            clock=clock,
            budget=budget,
            learner_version="clm-replay-candidate",
            max_experiences=len(plan),
        )
        persist()
        observe("before")
        for index, origin in enumerate(plan):
            boundary()
            budget.before_update()
            replay_budget.before_final()
            if replay_budget.remaining_replay_examples < len(
                history[origin].label.targets.group_codes
            ):
                replay_budget.stop_replay()
            record = history[origin]
            sample = f"clm-replay-{index}"
            inbox.record_experience(
                Experience(
                    sample,
                    "clm-replay",
                    at,
                    current_policy,
                    record.experience.features,
                    "train",
                    record.experience.permissions,
                )
            )
            inbox.record_label(
                LabelArrival(
                    "label-" + sample,
                    sample,
                    "clm-replay",
                    at,
                    current_policy,
                    record.label.targets,
                    "train",
                )
            )
            size = len(record.label.targets.group_codes)
            # An admitted replay attempt stays paid even if the native step fails.
            replay_budget.record_replay(size)
            attempted_rows += size
            report["replay_rows_attempted"] = attempted_rows
            report["attempts"].append(
                dict(
                    index=index,
                    original_history_index=origin,
                    original_key=list(record.experience.key),
                    original_event_id=record.label.event_id,
                    original_update=record.receipt.update_number,
                    replay_rows=size,
                    consumed_at=at,
                    child_sample_id=sample,
                )
            )
            proof = snapshot()
            try:
                applied = inbox.drain(max_updates=1)
                if len(applied) != 1 or applied[0].diagnostic.definition != OBJECTIVE:
                    raise ValueError("one actual paired candidate receipt required")
            finally:
                committed = budget.updates_completed
                report["receipts"] = [asdict(item) for item in inbox.applied_updates]
                proof = snapshot()
            persist()
        observe("after")
        boundary()
        if committed != len(plan) or candidate.training_examples != rows:
            raise ValueError("candidate did not finish its declared replay plan")
        report["complete"] = True
        persist()
    except Exception as error:
        report["complete"] = False
        report["error"] = f"{type(error).__name__}: {error}"[:512]
        raise
    finally:
        report["candidate_work"] = None if candidate is None else candidate.work
        report["copy_work"] = None if candidate is None else candidate.copy_work
        report["update_budget_commits"] = budget.updates_completed
        report["replay_budget_rows"] = replay_budget.replay_examples_completed
        save()
    return report
