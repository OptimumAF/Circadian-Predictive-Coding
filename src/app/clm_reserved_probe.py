"""Fixed six-update reserved CLM probe over existing inbox/clock/budget ports.

Consumes explicitly role-bound frozen features and an owned injected probe
factory. Produces only unscored scalar reports; no backend imports, scoring,
model export, source mutation, budget renewal or filesystem operations.
"""

from copy import deepcopy
from dataclasses import asdict, fields
import json
from math import isfinite
from typing import Any

from src.app.clm_temporal_sequence import (
    _feature_pin,
    _probabilities,
    _require_features,
    _tensor_traits,
)
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionProgress
from src.core.clm_catalog import CATALOG_OBJECTIVE, CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_catalog_probe import (
    CatalogProbeMarkerControl,
    CatalogProbeRehearsal,
    FrozenProbeCatalogBatch,
)
from src.core.clm_head import CLMHeadPolicy, CLMTargets, FrozenEncoderIdentity, _exact
from src.core.clm_probe import FrozenProbeChoiceBatch, FrozenProbeEmbeddingBatch
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.temporal_clm_probe import PROBE_GROUP_CODES, PROBE_UPDATES, reserved_probe_examples
from src.core.temporal_decision_sequence import (
    CHECKPOINTS,
    VOCABULARY,
    policy_for_phase,
    temporal_examples,
)
from src.core.ticket_workflow import CATALOG as WORKFLOW_CATALOG

_GROUPS = ("probe_train", "probe_test", "old_task")
_QUERY_ROWS = (2, 2, 4)


def _expected_work(
    updates,
    reads,
    queries,
    *,
    catalog_training=False,
    rehearsal_rows=0,
    extra_marker_rows=0,
    readout_candidate_rows=None,
):
    if type(catalog_training) is not bool:
        raise ValueError("catalog_training must be an exact boolean")
    if type(rehearsal_rows) is not int or rehearsal_rows not in (0, 2):
        raise ValueError("rehearsal rows must be zero or two per update")
    if rehearsal_rows and not catalog_training:
        raise ValueError("rehearsal requires catalog training")
    if type(extra_marker_rows) is not int or extra_marker_rows not in (0, 2):
        raise ValueError("extra marker rows must be zero or two per update")
    if extra_marker_rows and (not catalog_training or rehearsal_rows):
        raise ValueError("extra marker control requires catalog training without rehearsal")
    training_rows = 2 + rehearsal_rows + extra_marker_rows
    work = dict.fromkeys(
        (
            "train_attempts",
            "commits",
            "loss_forward_attempts",
            "loss_forward_returns",
            "backward_attempts",
            "backward_returns",
            "optimizer_attempts",
            "optimizer_returns",
            "scheduler_attempts",
            "scheduler_returns",
        ),
        updates,
    )
    work.update(
        dict.fromkeys(
            (
                "prediction_attempts",
                "prediction_returns",
                "choice_prediction_attempts",
                "choice_prediction_returns",
            ),
            reads,
        )
    )
    work.update(
        training_rows_attempted=training_rows * updates,
        prediction_rows_attempted=queries,
        choice_candidate_rows_attempted=(
            3 * reads if readout_candidate_rows is None else readout_candidate_rows
        ),
        snapshot_attempts=0,
        snapshot_returns=0,
        restore_attempts=0,
        restore_returns=0,
        unreturned_work_unknown=False,
    )
    if catalog_training:
        work.update(
            catalog_candidate_rows_attempted=3 * updates,
            catalog_logit_values_attempted=3 * training_rows * updates,
        )
    return work


def _admit_marker_control(control, training, phase, identity):
    """Bind the same two marker rows twice, with no history or replay grant."""
    _exact(control, CatalogProbeMarkerControl)
    if control.phase != phase:
        raise ValueError("marker control phase differs from its source checkpoint")
    marker, targets = training
    pins = []
    storages = {}
    for index, pair in enumerate((training, *control.packets)):
        batch, labels = pair
        _exact(batch, FrozenProbeCatalogBatch)
        _exact(labels, CLMCatalogTargets)
        _tensor_traits(batch.states, (2 if index == 0 else 4, identity.embedding_dim))
        _tensor_traits(batch.actions, (3, identity.embedding_dim))
        if (
            batch.identity != identity
            or batch.candidate_ids != marker.candidate_ids
            or labels.candidate_ids != targets.candidate_ids
            or labels.candidate_indices != targets.candidate_indices * (1 if index == 0 else 2)
            or not batch.actions.equal(marker.actions)
            or (index and not batch.states[:2].equal(marker.states))
            or (index and not batch.states[2:].equal(marker.states))
        ):
            raise ValueError("marker control differs from original repeated markers or catalog")
        pins.append(
            (
                pair,
                batch,
                labels,
                batch.identity,
                deepcopy(batch.identity),
                batch.candidate_ids,
                labels.candidate_ids,
                labels.candidate_indices,
                _feature_pin(batch),
            )
        )
        for tensor in (batch.states, batch.actions):
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
    return (control.phase, control.packets, training, tuple(pins)), dict(
        source_phase=phase,
        source_role="probe_train",
        source_row_indices=[[0, 1] for _ in range(PROBE_UPDATES)],
        retained_tensor_storage_bytes=sum(storages.values()),
        extra_marker_rows_attempted=0,
    )


def _require_marker_control(control, pin):
    _exact(control, CatalogProbeMarkerControl)
    phase, packets, training, pins = pin
    if control.phase != phase or control.packets is not packets:
        raise ValueError("original marker control checkpoint or packet owner changed")
    for pair, original in zip((training, *packets), pins, strict=True):
        owner, batch, targets, identity, identity_value, ids, target_ids, indices, features = (
            original
        )
        _exact(batch, FrozenProbeCatalogBatch)
        _exact(targets, CLMCatalogTargets)
        if (
            pair is not owner
            or pair[0] is not batch
            or pair[1] is not targets
            or batch.identity is not identity
            or batch.identity != identity_value
            or batch.candidate_ids is not ids
            or targets.candidate_ids is not target_ids
            or targets.candidate_indices is not indices
        ):
            raise ValueError("original marker control tuple, feature or target owner changed")
        _require_features(batch, features)


def _admit_rehearsal(rehearsal, training, phase, identity):
    """Bind only already-arrived original TRAIN and six fixed mixed packets."""
    _exact(rehearsal, CatalogProbeRehearsal)
    if rehearsal.phase != phase:
        raise ValueError("rehearsal phase differs from its original source checkpoint")
    batch, targets = rehearsal.training
    _exact(batch, FrozenCatalogTrainingBatch)
    _exact(targets, CLMCatalogTargets)
    examples = temporal_examples(phase, "train")
    ids = tuple(candidate.candidate_id for candidate in examples[0].request.allowed_candidates)
    indices = tuple(ids.index(example.target_id) for example in examples)
    dim = identity.embedding_dim
    if (
        batch.identity != identity
        or batch.candidate_ids != ids
        or targets.candidate_ids != ids
        or targets.candidate_indices != indices
    ):
        raise ValueError("rehearsal original TRAIN identity or canonical labels differ")
    _tensor_traits(batch.states, (3, dim))
    _tensor_traits(batch.actions, (3, dim))
    if not batch.actions.equal(training[0].actions):
        raise ValueError("rehearsal TRAIN must share the original full active catalog")
    row_indices = tuple((2 * index % 3, (2 * index + 1) % 3) for index in range(PROBE_UPDATES))
    pairs = (rehearsal.training, *rehearsal.packets)
    pins = []
    storages = {}
    for index, pair in enumerate(pairs):
        value, labels = pair
        _exact(value, FrozenCatalogTrainingBatch if index == 0 else FrozenProbeCatalogBatch)
        _exact(labels, CLMCatalogTargets)
        if index:
            rows = row_indices[index - 1]
            _tensor_traits(value.states, (4, dim))
            _tensor_traits(value.actions, (3, dim))
            if (
                value.identity != identity
                or value.candidate_ids != ids
                or labels.candidate_ids != ids
                or labels.candidate_indices
                != training[1].candidate_indices + tuple(indices[row] for row in rows)
                or not value.states[:2].equal(training[0].states)
                or any(
                    not value.states[2 + offset].equal(batch.states[row])
                    for offset, row in enumerate(rows)
                )
                or not value.actions.equal(batch.actions)
            ):
                raise ValueError("rehearsal packet differs from fixed marker/TRAIN rows or catalog")
        pins.append(
            (
                pair,
                value,
                labels,
                value.identity,
                value.candidate_ids,
                labels.candidate_ids,
                labels.candidate_indices,
                _feature_pin(value),
                deepcopy(value.identity),
            )
        )
        for tensor in (value.states, value.actions):
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
    return (
        rehearsal.phase,
        rehearsal.training,
        rehearsal.packets,
        rehearsal.permissions,
        tuple(pins),
    ), dict(
        source_phase=phase,
        source_role="train",
        source_row_indices=[list(rows) for rows in row_indices],
        retained_tensor_storage_bytes=sum(storages.values()),
        replay_rows_attempted=0,
    )


def _require_rehearsal(rehearsal, pin):
    _exact(rehearsal, CatalogProbeRehearsal)
    phase, training, packets, permissions, pins = pin
    if (
        rehearsal.phase != phase
        or rehearsal.training is not training
        or rehearsal.packets is not packets
        or rehearsal.permissions is not permissions
    ):
        raise ValueError("original rehearsal record or consent owner changed")
    _exact(permissions, ExperiencePermissions)
    if permissions != ExperiencePermissions(training=True, replay=True):
        raise ValueError("original TRAIN rehearsal consent changed")
    for pair, original in zip((training, *packets), pins, strict=True):
        owner, batch, targets, identity, ids, target_ids, indices, features, identity_value = (
            original
        )
        _exact(batch, FrozenCatalogTrainingBatch if pair is training else FrozenProbeCatalogBatch)
        _exact(targets, CLMCatalogTargets)
        if (
            pair is not owner
            or pair[0] is not batch
            or pair[1] is not targets
            or batch.identity is not identity
            or batch.identity != identity_value
            or batch.candidate_ids is not ids
            or targets.candidate_ids is not target_ids
            or targets.candidate_indices is not indices
        ):
            raise ValueError("original rehearsal tuple, feature or target owner changed")
        _require_features(batch, features)


def _admit_training_readout(training_readout, training, phase, identity, *, workflow=False):
    """Retain already-arrived TRAIN for observation, without a training conversion."""
    if type(training_readout) is not tuple or len(training_readout) != 2:
        raise ValueError("TRAIN readout requires its original feature/target tuple")
    batch, targets = training_readout
    _exact(batch, FrozenCatalogTrainingBatch)
    _exact(targets, CLMCatalogTargets)
    if workflow:
        # Why: these are the workflow's known rows, not unrelated temporal queries.
        ids = tuple(c.candidate_id for c in WORKFLOW_CATALOG if c.candidate_id != "retire")
        rules = policy_for_phase("after_return")["rules"]
        indices = tuple(
            ids.index(action)
            for concept in VOCABULARY
            for action in ("open", rules[concept], "commit")
        )
    else:
        examples = temporal_examples(phase, "train")
        ids = tuple(candidate.candidate_id for candidate in examples[0].request.allowed_candidates)
        indices = tuple(ids.index(example.target_id) for example in examples)
    if (
        batch.identity != identity
        or batch.candidate_ids != ids
        or targets.candidate_ids != ids
        or targets.candidate_indices != indices
    ):
        raise ValueError("TRAIN readout identity or canonical phase labels differ")
    _tensor_traits(batch.states, (len(indices), identity.embedding_dim))
    _tensor_traits(batch.actions, (len(ids), identity.embedding_dim))
    actions = (
        batch.actions[[ids.index(c) for c in training[0].candidate_ids]]
        if workflow
        else batch.actions
    )
    if not actions.equal(training[0].actions):
        raise ValueError("TRAIN readout must share the original full active catalog")
    # Why: the original features suffice; observation adds no clone or update grant.
    choice = FrozenProbeChoiceBatch(batch.identity, batch.states, batch.actions, "train")
    features = _feature_pin(batch)
    pin = (
        training_readout,
        batch,
        targets,
        batch.identity,
        deepcopy(batch.identity),
        batch.candidate_ids,
        targets.candidate_ids,
        targets.candidate_indices,
        features,
        choice,
    )
    storages = {}
    for tensor in (batch.states, batch.actions):
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
    return pin, dict(
        source_phase=phase,
        source_role="train",
        source_row_indices=list(range(len(indices))),
        candidate_ids=list(ids),
        candidate_indices=list(indices),
        retained_tensor_storage_bytes=sum(storages.values()),
        readouts=[],
    )


def _require_training_readout(training_readout, pin):
    owner, batch, targets, identity, identity_value, ids, target_ids, indices, features, choice = (
        pin
    )
    _exact(batch, FrozenCatalogTrainingBatch)
    _exact(targets, CLMCatalogTargets)
    _exact(choice, FrozenProbeChoiceBatch)
    if (
        training_readout is not owner
        or training_readout[0] is not batch
        or training_readout[1] is not targets
        or batch.identity is not identity
        or batch.identity != identity_value
        or batch.candidate_ids is not ids
        or targets.candidate_ids is not target_ids
        or targets.candidate_indices is not indices
        or choice.identity is not identity
        or choice.role != "train"
    ):
        raise ValueError("original TRAIN readout tuple, metadata or feature owner changed")
    _require_features(batch, features)
    _require_features(choice, features)


def _require_work(actual, expected):
    if (
        type(actual) is not dict
        or set(actual) != set(expected)
        or any(
            type(actual[key]) is not type(value) or actual[key] != value
            for key, value in expected.items()
        )
    ):
        raise ValueError("probe physical work differs from the observed paid prefix")


def _require_replay_budget(replay_budget, update_budget, *, birth=None, attempted=0):
    """Keep replay admission independent of the native update-only budget."""
    if type(replay_budget) is not ToyBudgetSession or replay_budget is update_budget:
        raise ValueError("rehearsal requires a distinct original replay budget session")
    _exact(replay_budget.budget, ToyExecutionBudget)
    if set(vars(replay_budget)) != {
        "budget",
        "clock",
        "progress",
        "started_at",
        "last_clock",
        "updates_completed",
        "replay_examples_completed",
        "hidden_width_observed",
        "peak_hidden_width_observed",
        "rejected_proposed_hidden_width",
        "checkpoint_position",
        "process_rss_sampler",
        "process_rss_segment",
    }:
        raise ValueError("original rehearsal replay budget attributes changed")
    if birth is None:
        birth = (
            replay_budget,
            replay_budget.budget,
            replay_budget.clock,
            replay_budget.started_at,
            replay_budget.last_clock,
            replay_budget.progress,
        )
    if (
        replay_budget is not birth[0]
        or not callable(replay_budget.clock)
        or type(replay_budget.started_at) is not float
        or not isfinite(replay_budget.started_at)
        or type(replay_budget.last_clock) is not float
        or not isfinite(replay_budget.last_clock)
        or replay_budget.budget is not birth[1]
        or replay_budget.clock is not birth[2]
        or replay_budget.started_at != birth[3]
        or replay_budget.last_clock != birth[4]
        or replay_budget.last_clock != replay_budget.started_at
        or replay_budget.progress is not birth[5]
        or replay_budget.budget != ToyExecutionBudget(max_replay_examples=12)
        or type(replay_budget.updates_completed) is not int
        or replay_budget.updates_completed != 0
        or type(replay_budget.replay_examples_completed) is not int
        or replay_budget.replay_examples_completed != attempted
        or replay_budget.checkpoint_position is not None
        or replay_budget.process_rss_sampler is not None
        or replay_budget.process_rss_segment is not None
        or replay_budget.hidden_width_observed is not None
        or replay_budget.peak_hidden_width_observed is not None
        or replay_budget.rejected_proposed_hidden_width is not None
    ):
        raise ValueError("original rehearsal replay budget differs from its paid prefix")
    progress = replay_budget.progress
    if progress is not None and (
        progress is update_budget.progress
        or type(progress) is not ToyExecutionProgress
        or set(vars(progress)) != {field.name for field in fields(ToyExecutionProgress)}
        or type(progress.updates_completed) is not int
        or type(progress.replay_examples_completed) is not int
        or progress != ToyExecutionProgress(replay_examples_completed=attempted)
    ):
        raise ValueError("original rehearsal replay progress differs from its paid prefix")
    return birth


def _source_metadata(source):
    identity, policy = source.identity, source.policy
    _exact(identity, FrozenEncoderIdentity)
    _exact(policy, CLMHeadPolicy)
    counts = (source.completed_updates, source.training_examples, source.generation)
    if any(type(value) is not int or not 0 <= value <= 2**53 - 1 for value in counts):
        raise ValueError("source checkpoint counters must be bounded native integers")
    return (identity, policy, *counts, source.work)


def _require_training(batch, targets, identity, examples, *, catalog_training=False):
    if catalog_training:
        _exact(batch, FrozenProbeCatalogBatch)
        _exact(targets, CLMCatalogTargets)
        allowed = tuple(
            candidate.candidate_id for candidate in examples[0].request.allowed_candidates
        )
        indices = tuple(allowed.index(example.target_id) for example in examples)
        if (
            batch.identity != identity
            or batch.candidate_ids != allowed
            or targets.candidate_ids != allowed
            or targets.candidate_indices != indices
        ):
            raise ValueError("reserved catalog probe identity, IDs or indices differ")
    else:
        _exact(batch, FrozenProbeEmbeddingBatch)
        _exact(targets, CLMTargets)
        if batch.identity != identity or targets.group_codes != PROBE_GROUP_CODES:
            raise ValueError("reserved probe TRAIN identity or group codes differ")


def _admit(
    source,
    training,
    readouts,
    report,
    phase,
    probe_factory,
    permissions,
    budget,
    before_step,
    save,
    *,
    catalog_training=False,
    workflow=False,
):
    examples = reserved_probe_examples(phase, "probe_train")
    source_meta = _source_metadata(source)
    expected_update = (
        32 if workflow else dict((name, update) for update, name in CHECKPOINTS)[phase]
    )
    if source_meta[2] != expected_update:
        raise ValueError("source checkpoint differs from its declared probe phase")
    if workflow:
        policy, counts, work = source_meta[1], source_meta[2:5], source_meta[5]
        expected = dict.fromkeys(
            (
                "train_attempts",
                "loss_forward_attempts",
                "loss_forward_returns",
                "backward_attempts",
                "backward_returns",
                "optimizer_attempts",
                "optimizer_returns",
                "scheduler_attempts",
                "scheduler_returns",
                "commits",
            ),
            32,
        )
        expected.update(
            training_rows_attempted=384,
            catalog_candidate_rows_attempted=160,
            catalog_logit_values_attempted=1920,
        )
        if (
            counts != (32, 384, 32)
            or policy.max_updates != 32
            or policy.max_rows != 12
            or policy.max_restores != 0
            or work.get("unreturned_work_unknown") is not False
            or any(
                type(v) is not int or v < 0
                for k, v in work.items()
                if k != "unreturned_work_unknown"
            )
            or any(type(work.get(k)) is not int or work[k] != v for k, v in expected.items())
        ):
            raise ValueError("workflow probe requires the exact spent32-update catalog source")
    if type(report) is not dict or report:
        raise ValueError("reserved probe requires a fresh empty child report")
    if any(not callable(value) for value in (probe_factory, before_step, save)):
        raise ValueError("probe factory and boundaries must be callable")
    _exact(permissions, ExperiencePermissions)
    if permissions != ExperiencePermissions(training=True):
        raise ValueError("probe requires explicit training-only consent without replay/evaluation")
    if type(budget) is not ToyBudgetSession:
        raise ValueError("probe requires the original fresh ToyBudgetSession")
    _exact(budget.budget, ToyExecutionBudget)
    if (
        budget.budget != ToyExecutionBudget(max_training_updates=PROBE_UPDATES)
        or budget.updates_completed != 0
        or budget.replay_examples_completed != 0
        or budget.checkpoint_position is not None
        or budget.process_rss_sampler is not None
        or budget.process_rss_segment is not None
        or budget.hidden_width_observed is not None
        or budget.peak_hidden_width_observed is not None
        or budget.rejected_proposed_hidden_width is not None
    ):
        raise ValueError("probe requires a fresh exact six-update independent budget")
    if budget.progress is not None:
        if (
            type(budget.progress) is not ToyExecutionProgress
            or set(vars(budget.progress)) != {field.name for field in fields(ToyExecutionProgress)}
            or budget.progress != ToyExecutionProgress()
        ):
            raise ValueError("probe progress must be independently fresh")
    if type(training) is not tuple or len(training) != 2:
        raise ValueError("expected one probe feature/target tuple")
    batch, targets = training
    _require_training(batch, targets, source_meta[0], examples, catalog_training=catalog_training)
    dim = source_meta[0].embedding_dim
    _tensor_traits(batch.states, (2, dim))
    _tensor_traits(batch.actions, (3 if catalog_training else 2, dim))
    if type(readouts) is not dict or tuple(readouts) != _GROUPS:
        raise ValueError("reserved readouts must preserve probe_train/probe_test/old_task order")
    pins = []
    for group, rows, role in zip(
        _GROUPS,
        (2, 2, 12) if workflow else _QUERY_ROWS,
        ("probe_train", "probe_test", "train" if workflow else "outer_selection"),
        strict=True,
    ):
        value = readouts[group]
        _exact(value, FrozenProbeChoiceBatch)
        if value.identity != source_meta[0] or value.role != role:
            raise ValueError("reserved readout identity or role differs")
        _tensor_traits(value.states, (rows, dim))
        _tensor_traits(value.actions, (5 if workflow and group == "old_task" else 3, dim))
        if not (workflow and group == "old_task") and not value.actions.equal(
            readouts["probe_train"].actions
        ):
            raise ValueError("probe readouts must share the original active catalog features")
        pins.append((group, value, role, _feature_pin(value)))
    if not readouts["probe_train"].states.equal(batch.states):
        raise ValueError("TRAIN diagnostic states differ from paired probe features")
    allowed = tuple(candidate.candidate_id for candidate in examples[0].request.allowed_candidates)
    if catalog_training:
        if not batch.actions.equal(readouts["probe_train"].actions):
            raise ValueError("catalog probe actions differ from the complete readout catalog")
    else:
        for row, example in enumerate(examples):
            if not batch.actions[row].equal(
                readouts["probe_train"].actions[allowed.index(example.target_id)]
            ):
                raise ValueError(
                    "paired probe actions differ from fixed inspect/route catalog features"
                )
    return source_meta, (batch, targets, _feature_pin(batch)), tuple(pins)


def _admit_owned_probe(probe, source, source_meta, *, catalog_training=False):
    if probe is source or any(
        not callable(getattr(probe, name, None)) for name in ("train_batch", "predict_choices")
    ):
        raise ValueError("factory must return a distinct owned probe port")
    if any(
        callable(getattr(probe, name, None))
        for name in ("snapshot_state", "restore_state", "fork", "export", "install")
    ):
        raise ValueError("disposable probe must not expose checkpoint, export or install ports")
    policy = probe.policy
    _exact(policy, CLMHeadPolicy)
    if policy.max_updates != PROBE_UPDATES or policy.max_restores != 0:
        raise ValueError("factory probe must own exactly six updates and zero restores")
    for field in fields(CLMHeadPolicy):
        if field.name not in ("max_updates", "max_restores"):
            new, old = getattr(policy, field.name), getattr(source_meta[1], field.name)
            if type(new) is not type(old) or new != old:
                raise ValueError("factory changed another native source policy field")
    if (
        probe.identity != source_meta[0]
        or probe.completed_updates != 0
        or probe.training_examples != 0
        or probe.generation != 0
        or (probe.source_completed_updates, probe.source_training_examples, probe.source_generation)
        != source_meta[2:5]
    ):
        raise ValueError("factory probe identity, freshness or source progress differs")
    _require_work(probe.work, _expected_work(0, 0, 0, catalog_training=catalog_training))
    return deepcopy(policy)


def _parameter_update_metadata(probe):
    counts = probe.parameter_counts
    if (
        probe.parameter_scope != "state_only"
        or type(counts) is not dict
        or set(counts) != {"optimized", "fixed", "total"}
        or any(type(value) is not int for value in counts.values())
        or not 0 < counts["optimized"] < counts["total"] <= probe.policy.max_parameters
        or counts["fixed"] != counts["total"] - counts["optimized"]
        or probe.fixed_parameters_unchanged is not True
    ):
        raise ValueError("state-only probe parameter scope, counts or fixed values changed")
    return dict(
        name="state_only",
        optimized_parameter_elements=counts["optimized"],
        fixed_parameter_elements=counts["fixed"],
        total_parameter_elements=counts["total"],
        fixed_parameters_unchanged=True,
    )


def run_reserved_probe(
    source,
    training,
    readouts,
    report,
    *,
    phase,
    probe_factory,
    permissions,
    budget,
    before_step=lambda: None,
    save=lambda: None,
    catalog_training=False,
    rehearsal=None,
    replay_budget=None,
    marker_control=None,
    training_readout=None,
    parameter_scope="all",
    workflow_training_readout=None,
):
    """Run one fixed disposable probe; propagate failures with paid facts intact."""
    if (
        type(parameter_scope) is not str
        or parameter_scope not in ("all", "state_only")
        or (parameter_scope == "state_only" and not catalog_training)
    ):
        raise ValueError("state-only scope requires an explicit catalog probe")
    if type(catalog_training) is not bool:
        raise ValueError("catalog_training must be an exact boolean")
    workflow = workflow_training_readout is not None
    if workflow and (
        not catalog_training
        or phase != "after_return"
        or rehearsal is not None
        or replay_budget is not None
        or marker_control is not None
        or training_readout is not None
    ):
        raise ValueError(
            "workflow TRAIN readout requires the standalone after_return catalog probe"
        )
    if training_readout is not None and (not catalog_training or phase == "baseline"):
        raise ValueError("TRAIN readout requires a nonbaseline catalog checkpoint")
    if rehearsal is not None and not catalog_training:
        raise ValueError("rehearsal requires the catalog objective")
    if marker_control is not None and (
        not catalog_training or rehearsal is not None or replay_budget is not None
    ):
        raise ValueError("marker control requires catalog training without rehearsal or replay")
    if rehearsal is None and replay_budget is not None:
        raise ValueError("a replay budget requires explicit catalog rehearsal")
    rehearsal_rows = 2 if rehearsal is not None else 0
    extra_marker_rows = 2 if marker_control is not None else 0
    training_rows = 2 + rehearsal_rows + extra_marker_rows
    source_meta, training_pin, readout_pins = _admit(
        source,
        training,
        readouts,
        report,
        phase,
        probe_factory,
        permissions,
        budget,
        before_step,
        save,
        catalog_training=catalog_training,
        workflow=workflow,
    )
    workflow_pin = workflow_report = None
    if workflow:
        workflow_pin, workflow_report = _admit_training_readout(
            workflow_training_readout, training, phase, source_meta[0], workflow=True
        )
        old = readouts["old_task"]
        if old.states is not workflow_pin[1].states or old.actions is not workflow_pin[1].actions:
            raise ValueError(
                "workflow old-task readout must retain the original TRAIN tensor owners"
            )
        workflow_report.pop("readouts")
    rehearsal_pin = None
    rehearsal_report = None
    replay_birth = None
    marker_control_pin = None
    marker_control_report = None
    if rehearsal is not None:
        replay_birth = _require_replay_budget(replay_budget, budget)
        rehearsal_pin, rehearsal_report = _admit_rehearsal(
            rehearsal, training, phase, source_meta[0]
        )
    if marker_control is not None:
        marker_control_pin, marker_control_report = _admit_marker_control(
            marker_control, training, phase, source_meta[0]
        )
    training_readout_pin = None
    training_readout_report = None
    if training_readout is not None:
        if rehearsal is not None and training_readout is not rehearsal.training:
            raise ValueError("TRAIN readout must retain the original rehearsal TRAIN tuple")
        training_readout_pin, training_readout_report = _admit_training_readout(
            training_readout, training, phase, source_meta[0]
        )
    examples = reserved_probe_examples(phase, "probe_train")
    objective = (
        CATALOG_OBJECTIVE
        if catalog_training
        else "clm_paired_group_masked_symmetric_cross_entropy_v1"
    )
    report.update(
        phase=phase,
        source_update=source_meta[2],
        complete=False,
        readouts=[],
        stream=dict(completed_updates=0, training_pairs=0, receipts=[]),
        probe_work=None,
    )
    if catalog_training:
        report.update(
            training_objective=CATALOG_OBJECTIVE,
            candidate_ids=list(training[0].candidate_ids),
            candidate_indices=list(training[1].candidate_indices),
        )
    if rehearsal_report is not None:
        report["rehearsal"] = rehearsal_report
    if marker_control_report is not None:
        report["marker_control"] = marker_control_report
    if training_readout_report is not None:
        report["arrived_train_readouts"] = training_readout_report
    if workflow_report is not None:
        report["workflow_training_readout"] = workflow_report
    limits = budget.budget
    birth = (budget, limits, budget.clock, budget.started_at, budget.progress)
    clock = LogicalClock()
    episode = "clm-probe-" + phase
    probe: Any = None
    inbox: Any = None
    probe_policy: CLMHeadPolicy | None = None
    parameter_report = None
    receipts: list[dict[str, Any]] = []
    reads = queries = readout_candidates = tick = replay_rows_attempted = 0
    extra_marker_rows_attempted = 0
    proof = ""

    scalar_keys: tuple[str, ...] = ("phase", "source_update", "complete", "readouts", "stream")
    if parameter_scope == "state_only":
        report["parameter_update_scope"] = None
        scalar_keys += ("parameter_update_scope",)
    if catalog_training:
        scalar_keys += ("training_objective", "candidate_ids", "candidate_indices")
    if rehearsal_report is not None:
        scalar_keys += ("rehearsal",)
    if marker_control_report is not None:
        scalar_keys += ("marker_control",)
    if training_readout_report is not None:
        scalar_keys += ("arrived_train_readouts",)
    if workflow_report is not None:
        scalar_keys += ("workflow_training_readout",)
    marker_metadata = (
        training[0].identity,
        getattr(training[0], "candidate_ids", None),
        getattr(training[1], "candidate_ids", None),
        getattr(training[1], "candidate_indices", None),
    )
    source_snapshot = (
        deepcopy(source_meta)
        if (
            rehearsal_pin is not None
            or marker_control_pin is not None
            or training_readout_pin is not None
            or workflow_pin is not None
            or parameter_scope == "state_only"
        )
        else None
    )

    def scalar_state():
        return json.dumps(
            {key: report[key] for key in scalar_keys},
            sort_keys=True,
            allow_nan=False,
        )

    def guard():
        if _source_metadata(source) != source_meta:
            raise ValueError("reserved probe changed source metadata or physical work")
        if workflow:
            _require_work(source.work, source_meta[5])
        if source_snapshot is not None and (_source_metadata(source) != source_snapshot):
            if marker_control_pin is not None:
                raise ValueError("original marker control source checkpoint changed")
            if training_readout_pin is not None:
                raise ValueError("original TRAIN readout source checkpoint changed")
            raise ValueError("original rehearsal source checkpoint changed")
        _exact(permissions, ExperiencePermissions)
        if permissions != ExperiencePermissions(training=True):
            raise ValueError("original probe consent changed")
        if (
            budget is not birth[0]
            or budget.budget is not birth[1]
            or budget.clock is not birth[2]
            or budget.started_at != birth[3]
            or budget.progress is not birth[4]
            or budget.budget != ToyExecutionBudget(max_training_updates=PROBE_UPDATES)
            or budget.updates_completed != len(receipts)
            or budget.replay_examples_completed != 0
            or clock.now() != tick
        ):
            raise ValueError("reserved probe original clock or budget changed")
        if budget.progress is not None and budget.progress != ToyExecutionProgress(
            updates_completed=len(receipts)
        ):
            raise ValueError("original probe progress differs from its paid prefix")
        if training[0] is not training_pin[0] or training[1] is not training_pin[1]:
            raise ValueError("original paired probe record changed")
        _require_training(
            training[0], training[1], source_meta[0], examples, catalog_training=catalog_training
        )
        _require_features(training[0], training_pin[2])
        if rehearsal_pin is not None:
            _require_rehearsal(rehearsal, rehearsal_pin)
            _require_replay_budget(
                replay_budget, budget, birth=replay_birth, attempted=replay_rows_attempted
            )
            if (
                report["rehearsal"] is not rehearsal_report
                or rehearsal_report["replay_rows_attempted"] != replay_rows_attempted
                or type(budget.updates_completed) is not int
                or type(budget.replay_examples_completed) is not int
                or (
                    budget.progress is not None
                    and (
                        type(budget.progress.updates_completed) is not int
                        or type(budget.progress.replay_examples_completed) is not int
                    )
                )
            ):
                raise ValueError("original rehearsal report or paid replay prefix changed")
            if any(
                value is not original
                for value, original in zip(
                    (
                        training[0].identity,
                        training[0].candidate_ids,
                        training[1].candidate_ids,
                        training[1].candidate_indices,
                    ),
                    marker_metadata,
                    strict=True,
                )
            ):
                raise ValueError("original marker target or catalog owner changed")
        if marker_control_pin is not None:
            _require_marker_control(marker_control, marker_control_pin)
            if (
                report["marker_control"] is not marker_control_report
                or type(marker_control_report["extra_marker_rows_attempted"]) is not int
                or marker_control_report["extra_marker_rows_attempted"]
                != extra_marker_rows_attempted
                or type(extra_marker_rows_attempted) is not int
                or not 0 <= extra_marker_rows_attempted <= 2 * PROBE_UPDATES
                or type(budget.updates_completed) is not int
                or type(budget.replay_examples_completed) is not int
                or (
                    budget.progress is not None
                    and (
                        type(budget.progress.updates_completed) is not int
                        or type(budget.progress.replay_examples_completed) is not int
                    )
                )
            ):
                raise ValueError("original marker control report or paid prefix changed")
        if training_readout_pin is not None or workflow_pin is not None:
            if (
                type(budget.updates_completed) is not int
                or type(budget.replay_examples_completed) is not int
                or (
                    budget.progress is not None
                    and (
                        type(budget.progress.updates_completed) is not int
                        or type(budget.progress.replay_examples_completed) is not int
                    )
                )
            ):
                raise ValueError("TRAIN readout budget progress requires native integer counts")
            if training_readout_pin is not None:
                _require_training_readout(training_readout, training_readout_pin)
                if report["arrived_train_readouts"] is not training_readout_report:
                    raise ValueError("original TRAIN readout report owner changed")
            if any(
                value is not original
                for value, original in zip(
                    (
                        training[0].identity,
                        training[0].candidate_ids,
                        training[1].candidate_ids,
                        training[1].candidate_indices,
                    ),
                    marker_metadata,
                    strict=True,
                )
            ):
                raise ValueError("original marker target or catalog owner changed")
        if workflow_pin is not None:
            _require_training_readout(workflow_training_readout, workflow_pin)
            if report["workflow_training_readout"] is not workflow_report:
                raise ValueError("original workflow TRAIN readout report owner changed")
        if type(readouts) is not dict or tuple(readouts) != _GROUPS:
            raise ValueError("original probe readout order changed")
        for group, value, role, pin in readout_pins:
            if readouts[group] is not value:
                raise ValueError("original probe readout owner changed")
            _exact(value, FrozenProbeChoiceBatch)
            if value.identity != source_meta[0] or value.role != role:
                raise ValueError("original probe readout metadata changed")
            _require_features(value, pin)
        if probe is not None and probe_policy is not None:
            if getattr(probe, "parameter_scope", "all") != parameter_scope:
                raise ValueError("original probe parameter scope changed")
            if parameter_scope == "state_only" and (
                report["parameter_update_scope"] is not parameter_report
                or _parameter_update_metadata(probe) != parameter_report
            ):
                raise ValueError("original state-only parameter evidence changed")
            if (
                marker_control_pin is not None
                or training_readout_pin is not None
                or workflow_pin is not None
            ) and any(
                type(value) is not int
                for value in (probe.completed_updates, probe.training_examples, probe.generation)
            ):
                raise ValueError("optional probe diagnostics require native integer counts")
            if (
                probe.identity != source_meta[0]
                or probe.policy != probe_policy
                or probe.completed_updates != len(receipts)
                or probe.training_examples != training_rows * len(receipts)
                or probe.generation != len(receipts)
            ):
                raise ValueError("owned probe policy or progress changed")
            _require_work(
                probe.work,
                _expected_work(
                    len(receipts),
                    reads,
                    queries,
                    catalog_training=catalog_training,
                    rehearsal_rows=rehearsal_rows,
                    extra_marker_rows=extra_marker_rows,
                    readout_candidate_rows=readout_candidates,
                ),
            )
            if report["probe_work"] != probe.work:
                raise ValueError("recorded probe work changed")
        if inbox is not None and [asdict(item) for item in inbox.applied_updates] != receipts:
            raise ValueError("actual probe receipt prefix changed")
        if proof and scalar_state() != proof:
            raise ValueError("recorded reserved probe prefix changed")

    def boundary():
        guard()
        before_step()
        guard()

    def persist():
        nonlocal proof
        report["probe_work"] = probe.work if probe is not None else None
        proof = scalar_state()
        save()
        guard()

    def refresh():
        nonlocal proof
        if proof and scalar_state() != proof:
            raise ValueError("recorded probe prefix changed during native work")
        actual = [asdict(item) for item in inbox.applied_updates]
        if actual[: len(receipts)] != receipts or len(actual) > len(receipts) + 1:
            raise ValueError("original probe receipts changed")
        for item in actual[len(receipts) :]:
            index = len(receipts)
            sample = f"clm-probe-{phase}-{index}"
            if (
                item["sample_id"] != sample
                or item["episode_id"] != episode
                or item["event_id"] != "label-" + sample
                or item["actor_version"] != phase
                or item["learner_version"] != episode
                or item["observed_at"] != 2 * index
                or item["arrived_at"] != 2 * index + 1
                or item["applied_at"] != 2 * index + 1
                or item["update_number"] != index + 1
                or item["diagnostic"]["definition"] != objective
            ):
                raise ValueError("actual reserved probe receipt binding differs")
            receipts.append(deepcopy(item))
        report["stream"] = dict(
            completed_updates=budget.updates_completed,
            training_pairs=training_rows * len(receipts),
            receipts=deepcopy(actual),
        )
        report["probe_work"] = probe.work
        proof = scalar_state()

    def observe(point):
        nonlocal reads, queries, readout_candidates, proof
        for group, rows in zip(_GROUPS, (2, 2, 12) if workflow else _QUERY_ROWS, strict=True):
            boundary()
            item = dict(point=point, group=group, probabilities=None)
            report["readouts"].append(item)
            proof = scalar_state()
            value = probe.predict_choices(readouts[group])
            reads += 1
            queries += rows
            candidates = 5 if workflow and group == "old_task" else 3
            readout_candidates += candidates
            probabilities = _probabilities(value, rows=rows, candidates=candidates)
            if scalar_state() != proof:
                raise ValueError("recorded probe prefix changed during native readout")
            item["probabilities"] = probabilities
            report["probe_work"] = probe.work
            proof = scalar_state()
            guard()
            persist()
        if training_readout_pin is not None:
            if training_readout_report is None:
                raise ValueError("original TRAIN readout report missing")
            boundary()
            item = dict(point=point, probabilities=None)
            training_readout_report["readouts"].append(item)
            proof = scalar_state()
            value = probe.predict_choices(training_readout_pin[-1])
            reads += 1
            queries += 3
            readout_candidates += 3
            probabilities = _probabilities(value, rows=3)
            if scalar_state() != proof:
                raise ValueError("recorded TRAIN readout prefix changed during native readout")
            item["probabilities"] = probabilities
            report["probe_work"] = probe.work
            proof = scalar_state()
            guard()
            persist()

    try:
        proof = scalar_state()
        boundary()
        probe = probe_factory(source)
        probe_policy = _admit_owned_probe(
            probe, source, source_meta, catalog_training=catalog_training
        )
        if getattr(probe, "parameter_scope", "all") != parameter_scope:
            raise ValueError("factory returned a different probe parameter scope")
        if parameter_scope == "state_only":
            parameter_report = _parameter_update_metadata(probe)
            report["parameter_update_scope"] = parameter_report
            proof = scalar_state()
        report["probe_work"] = probe.work
        inbox = ExperienceInbox(
            probe,
            clock=clock,
            budget=budget,
            learner_version=episode,
            max_experiences=PROBE_UPDATES,
            training_role="probe_train",
        )
        guard()
        observe("before")
        for index in range(PROBE_UPDATES):
            boundary()
            batch, targets = training if rehearsal is None else rehearsal.packets[index]
            if marker_control is not None:
                batch, targets = marker_control.packets[index]
            tick = 2 * index
            clock.advance_to(tick)
            sample = f"clm-probe-{phase}-{index}"
            inbox.record_experience(
                Experience(sample, episode, tick, phase, batch, "probe_train", permissions)
            )
            if inbox.drain(max_updates=1):
                raise ValueError("probe updated before its label arrived")
            guard()
            tick += 1
            clock.advance_to(tick)
            inbox.record_label(
                LabelArrival(
                    "label-" + sample, sample, episode, tick, phase, targets, "probe_train"
                )
            )
            if rehearsal_report is not None:
                guard()
                # Why: an uncertain native failure still consumed the admitted
                # replay exposure. This attempted charge is never refunded.
                replay_budget.record_replay(rehearsal_rows)
                replay_rows_attempted += rehearsal_rows
                rehearsal_report["replay_rows_attempted"] = replay_rows_attempted
                proof = scalar_state()
            if marker_control_report is not None:
                guard()
                # Why: failed native attempts still consume the declared extra
                # marker exposure; this is TRAIN work, never replay credit.
                extra_marker_rows_attempted += extra_marker_rows
                marker_control_report["extra_marker_rows_attempted"] = extra_marker_rows_attempted
                proof = scalar_state()
            try:
                applied = inbox.drain(max_updates=1)
            finally:
                refresh()
            if len(applied) != 1:
                raise ValueError("expected exactly one actual reserved probe update")
            guard()
            persist()
        observe("after")
        boundary()
        expected_reads, expected_queries = (
            (6, 32) if workflow else (8, 22) if training_readout_pin is not None else (6, 16)
        )
        if len(receipts) != PROBE_UPDATES or reads != expected_reads or queries != expected_queries:
            raise ValueError("reserved probe did not finish its fixed work")
        report["complete"] = True
        persist()
    except Exception as error:
        if workflow and proof:
            # Keep one last trusted prefix when a callback corrupts public evidence.
            report["preserved_workflow_probe_prefix"] = json.loads(proof)
        report["complete"] = False
        report["error"] = (type(error).__name__ + ": " + str(error))[:512]
        raise
    finally:
        report["probe_work"] = probe.work if probe is not None else None
        report["stream"] = dict(
            completed_updates=budget.updates_completed,
            training_pairs=training_rows * len(inbox.applied_updates) if inbox is not None else 0,
            receipts=[asdict(item) for item in inbox.applied_updates] if inbox is not None else [],
        )
    return report
