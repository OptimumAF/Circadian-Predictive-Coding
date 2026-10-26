"""Bind and observe fixed reserved probes around the existing temporal stream.

Owns frozen-feature mapping and disposable checkpoint composition. Cached scoring
reuses existing metric arithmetic after all TRAIN finishes. No encoder loading,
benchmark update, IO, final data, model installation or new stream runner.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from typing import Any

from src.app.clm_temporal_sequence import _feature_pin, _require_features, _tensor_traits
from src.app.decision_reference_evaluation import _metrics
from src.app.clm_reserved_probe import _expected_work, _require_work
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.clm_head import CLMTargets, FrozenEncoderIdentity, _exact
from src.core.clm_catalog import CATALOG_OBJECTIVE, CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_catalog_probe import (
    CatalogProbeMarkerControl,
    CatalogProbeRehearsal,
    FrozenProbeCatalogBatch,
)
from src.core.clm_probe import FrozenProbeChoiceBatch, FrozenProbeEmbeddingBatch
from src.core.experience import ExperiencePermissions
from src.core.temporal_clm_probe import PROBE_GROUP_CODES, PROBE_UPDATES, reserved_probe_examples
from src.core.temporal_decision_sequence import CATALOG, CHECKPOINTS, temporal_examples
from src.core.temporal_catalog_support import catalog_training_examples

GROUPS = ("probe_train", "probe_test", "old_task")


def build_reserved_probe_batches(
    identity, features, *, catalog_support="none", catalog_training=False
):
    import torch

    _exact(identity, FrozenEncoderIdentity)
    if type(catalog_training) is not bool:
        raise ValueError("catalog probe selection must be an exact boolean")
    if type(catalog_support) is not str or catalog_support != "none":
        raise ValueError(
            "reserved probes require the original none recipe within16 encoder sequences"
        )
    if type(features) is not torch.Tensor:
        raise ValueError("reserved probe features require an exact native tensor")
    _tensor_traits(features, (16, identity.embedding_dim))
    bank = features.detach().clone()
    base = torch.cat((bank[:8], bank[12:16]))
    paired_training = None
    if not catalog_training:
        paired_training = (
            FrozenProbeEmbeddingBatch(identity, bank[8:10].clone(), bank[[13, 15]].clone()),
            CLMTargets(PROBE_GROUP_CODES),
        )
    all_ids = tuple(row.candidate_id for row in CATALOG)
    readouts = {}
    catalog_pairs = {}
    for _, phase in CHECKPOINTS:
        active = reserved_probe_examples(phase, "probe_train")[0].request.allowed_candidates
        actions = bank[[12 + all_ids.index(row.candidate_id) for row in active]]
        ids = tuple(row.candidate_id for row in active)
        if catalog_training:
            examples = reserved_probe_examples(phase, "probe_train")
            catalog_pairs[phase] = (
                FrozenProbeCatalogBatch(identity, bank[8:10].clone(), actions.clone(), ids),
                CLMCatalogTargets(ids, tuple(ids.index(row.target_id) for row in examples)),
            )
        readouts[phase] = {
            "probe_train": FrozenProbeChoiceBatch(
                identity, bank[8:10].clone(), actions.clone(), "probe_train"
            ),
            "probe_test": FrozenProbeChoiceBatch(
                identity, bank[10:12].clone(), actions.clone(), "probe_test"
            ),
            "old_task": FrozenProbeChoiceBatch(
                identity, bank[4:8].clone(), actions.clone(), "outer_selection"
            ),
        }
    return base, catalog_pairs if catalog_training else paired_training, readouts


def build_probe_rehearsal(training, arrived_training, *, phase, permissions):
    """Materialize six fixed mixed packets from a declared arrived TRAIN triple.

    Checkpoint availability and lifetime guards belong to the existing hook/app;
    this adapter performs bounded tensor concatenation, never label selection.
    """
    import torch

    if type(training) is not tuple or len(training) != 2:
        raise ValueError("original probe feature/target tuple required")
    marker, targets = training
    _exact(marker, FrozenProbeCatalogBatch)
    _exact(targets, CLMCatalogTargets)
    if type(arrived_training) is not tuple or len(arrived_training) != 2:
        raise ValueError("arrived TRAIN feature/target tuple required")
    old, old_targets = arrived_training
    _exact(old, FrozenCatalogTrainingBatch)
    _exact(old_targets, CLMCatalogTargets)
    examples = reserved_probe_examples(phase, "probe_train")
    ids = tuple(row.candidate_id for row in examples[0].request.allowed_candidates)
    if (
        marker.candidate_ids != ids
        or targets.candidate_ids != ids
        or targets.candidate_indices != tuple(ids.index(row.target_id) for row in examples)
        or old.identity != marker.identity
        or old.candidate_ids != ids
        or not old.actions.equal(marker.actions)
    ):
        raise ValueError("rehearsal source and original marker catalog differ")
    dim = marker.identity.embedding_dim
    for value, shape in (
        (marker.states, (2, dim)),
        (marker.actions, (3, dim)),
        (old.states, (3, dim)),
        (old.actions, (3, dim)),
    ):
        _tensor_traits(value, shape)
    packets = []
    for index in range(PROBE_UPDATES):
        rows = (2 * index % 3, (2 * index + 1) % 3)
        packets.append(
            (
                FrozenProbeCatalogBatch(
                    marker.identity,
                    torch.cat((marker.states, old.states[list(rows)])),
                    marker.actions.detach().clone(),
                    ids,
                ),
                CLMCatalogTargets(
                    ids,
                    targets.candidate_indices
                    + tuple(old_targets.candidate_indices[row] for row in rows),
                ),
            )
        )
    return CatalogProbeRehearsal(phase, arrived_training, tuple(packets), permissions)


def build_probe_marker_control(training, *, phase):
    """Repeat the original marker pair to match rehearsal's four-query dose."""
    if type(phase) is not str or phase not in tuple(phase for _, phase in CHECKPOINTS[1:]):
        raise ValueError("extra-marker control requires a non-baseline checkpoint")
    if type(training) is not tuple or len(training) != 2:
        raise ValueError("original marker feature/target tuple required")
    marker, targets = training
    _exact(marker, FrozenProbeCatalogBatch)
    _exact(targets, CLMCatalogTargets)
    examples = reserved_probe_examples(phase, "probe_train")
    ids = tuple(row.candidate_id for row in examples[0].request.allowed_candidates)
    if (
        marker.candidate_ids != ids
        or targets.candidate_ids != ids
        or targets.candidate_indices != tuple(ids.index(row.target_id) for row in examples)
    ):
        raise ValueError("original marker control labels/catalog differ")
    _tensor_traits(marker.states, (2, marker.identity.embedding_dim))
    _tensor_traits(marker.actions, (3, marker.identity.embedding_dim))
    import torch

    packets = tuple(
        (
            FrozenProbeCatalogBatch(
                marker.identity,
                torch.cat((marker.states, marker.states)),
                marker.actions.detach().clone(),
                ids,
            ),
            CLMCatalogTargets(ids, targets.candidate_indices * 2),
        )
        for _ in range(PROBE_UPDATES)
    )
    return CatalogProbeMarkerControl(phase, packets)


def _rehearsal_training_pins(training, readouts):
    if type(training) is not dict or tuple(training) != tuple(
        phase for _, phase in CHECKPOINTS[1:]
    ):
        raise ValueError("ordered arrived TRAIN triples required for rehearsal")
    pins = []
    for phase, pair in training.items():
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("rehearsal TRAIN tuples required")
        batch, targets = pair
        _exact(batch, FrozenCatalogTrainingBatch)
        _exact(targets, CLMCatalogTargets)
        examples = temporal_examples(phase, "train")
        ids = tuple(row.candidate_id for row in examples[0].request.allowed_candidates)
        if (
            batch.candidate_ids != ids
            or targets.candidate_ids != ids
            or targets.candidate_indices != tuple(ids.index(row.target_id) for row in examples)
            or batch.identity != readouts[phase]["probe_train"].identity
            or not batch.actions.equal(readouts[phase]["probe_train"].actions)
        ):
            raise ValueError("rehearsal requires original phase-current TRAIN labels/catalog")
        _tensor_traits(batch.states, (3, batch.identity.embedding_dim))
        _tensor_traits(batch.actions, (3, batch.identity.embedding_dim))
        pins.append((phase, pair, batch, targets, deepcopy(targets), _feature_pin(batch)))
    return tuple(pins)


def _training_plan(training, *, catalog_training):
    phases = tuple(phase for _, phase in CHECKPOINTS)
    if catalog_training:
        if type(training) is not dict or tuple(training) != phases:
            raise ValueError("four ordered catalog probe training phases required")
        return tuple((phase, training[phase]) for phase in phases)
    return tuple((phase, training) for phase in phases)


def _plan_pins(training, readouts, *, catalog_training=False):
    plan = _training_plan(training, catalog_training=catalog_training)
    identity = None
    pins = []
    for index, (_, pair) in enumerate(plan):
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("reserved training must contain exact feature/target tuples")
        batch, targets = pair
        _exact(batch, FrozenProbeCatalogBatch if catalog_training else FrozenProbeEmbeddingBatch)
        _exact(targets, CLMCatalogTargets if catalog_training else CLMTargets)
        if identity is None:
            identity = batch.identity
        if batch.identity != identity:
            raise ValueError("reserved training identities differ")
        if not catalog_training and targets.group_codes != PROBE_GROUP_CODES:
            raise ValueError("reserved paired targets changed")
        _tensor_traits(batch.states, (2, identity.embedding_dim))
        _tensor_traits(batch.actions, (3 if catalog_training else 2, identity.embedding_dim))
        if catalog_training or index == 0:
            pins.append((batch, _feature_pin(batch)))
    if type(readouts) is not dict or tuple(readouts) != tuple(phase for _, phase in CHECKPOINTS):
        raise ValueError("all four reserved checkpoint feature groups required")
    for phase, groups in readouts.items():
        batch, targets = training[phase] if catalog_training else training
        if type(groups) is not dict or tuple(groups) != GROUPS:
            raise ValueError("three ordered reserved readout groups required")
        for group, rows in zip(GROUPS, (2, 2, 4), strict=True):
            selected = groups[group]
            _exact(selected, FrozenProbeChoiceBatch)
            role = "outer_selection" if group == "old_task" else group
            if selected.identity != identity or selected.role != role:
                raise ValueError("reserved readout identity or role changed")
            _tensor_traits(selected.states, (rows, identity.embedding_dim))
            _tensor_traits(selected.actions, (3, identity.embedding_dim))
            pins.append((selected, _feature_pin(selected)))
        active = reserved_probe_examples(phase, "probe_train")[0].request.allowed_candidates
        ids = tuple(row.candidate_id for row in active)
        if not batch.states.equal(groups["probe_train"].states):
            raise ValueError("reserved diagnostic states differ from probe features")
        if catalog_training:
            expected = tuple(
                ids.index(row.target_id) for row in reserved_probe_examples(phase, "probe_train")
            )
            if (
                batch.candidate_ids != ids
                or targets.candidate_ids != ids
                or targets.candidate_indices != expected
                or not batch.actions.equal(groups["probe_train"].actions)
            ):
                raise ValueError("reserved catalog IDs, indexed labels or features differ")
        elif not batch.actions.equal(
            groups["probe_train"].actions[[ids.index("inspect"), ids.index("route")]]
        ):
            raise ValueError("reserved diagnostic and paired features disagree")
    return pins


def _new_probe(source):
    from src.adapters.clm_head_probe import CLMHeadProbe

    return CLMHeadProbe(
        source,
        policy=replace(source.policy, max_updates=PROBE_UPDATES, max_restores=0),
        max_readouts=6,
    )


def _new_catalog_probe(source, *, max_readouts=6, parameter_scope="all"):
    from src.adapters.clm_catalog_probe import CLMCatalogProbe

    return CLMCatalogProbe(
        source,
        policy=replace(source.policy, max_updates=PROBE_UPDATES, max_restores=0),
        max_readouts=max_readouts,
        **({"parameter_scope": parameter_scope} if parameter_scope != "all" else {}),
    )


def run_workflow_reserved_probe(
    source,
    training,
    readouts,
    workflow_training_readout,
    report,
    *,
    permissions,
    budget,
    parameter_scope="all",
    before_step=lambda: None,
    save=lambda: None,
):
    """One isolated F probe from a spent workflow32 source, no scoring or source updates.

    Callers supply the original TRAIN tuple and independent six-update authority.
    The app admits its exact protocol; this outer boundary protects native state.
    """
    import torch
    from src.app.clm_reserved_probe import run_reserved_probe

    if type(parameter_scope) is not str or parameter_scope not in ("all", "state_only"):
        raise ValueError("workflow probe parameter_scope must be all or state_only")
    if type(report) is not dict or report:
        raise ValueError("workflow probe requires a fresh empty report")
    owner, original = source._bundle, source._capture()
    work_owner, rng_owner = source._work, source._rng
    work = source.work
    caller_rng = torch.get_rng_state().clone()

    def source_guard():
        source._require_original(owner, original)
        _require_work(source.work, work)
        if (
            source._work is not work_owner
            or source._rng is not rng_owner
            or not torch.equal(torch.get_rng_state(), caller_rng)
        ):
            raise ValueError("workflow probe changed source work, owners or caller RNG")

    def boundary():
        source_guard()
        before_step()
        source_guard()

    def persist():
        source_guard()
        save()
        source_guard()

    # Why: reuse the fixed-action control without changing default factory calls.
    factory = (
        _new_catalog_probe
        if parameter_scope == "all"
        else lambda learner: _new_catalog_probe(learner, parameter_scope=parameter_scope)
    )
    try:
        return run_reserved_probe(
            source,
            training,
            readouts,
            report,
            phase="after_return",
            probe_factory=factory,
            permissions=permissions,
            budget=budget,
            before_step=boundary,
            save=persist,
            catalog_training=True,
            parameter_scope=parameter_scope,
            workflow_training_readout=workflow_training_readout,
        )
    finally:
        try:
            source_guard()
        except Exception as error:
            report.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])
            raise


def make_reserved_probe_hook(
    training,
    readouts_by_phase,
    report,
    *,
    before_step=lambda: None,
    save=lambda: None,
    catalog_training=False,
    rehearsal_training=None,
    marker_control=False,
    readout_training=None,
    parameter_scope="all",
    linear_fit_control=False,
):
    from src.app.clm_reserved_probe import run_reserved_probe

    if type(linear_fit_control) is not bool:
        raise ValueError("linear probe selection must be an exact bool")
    if linear_fit_control:
        if (
            catalog_training is not True
            or rehearsal_training is not None
            or marker_control is not False
            or readout_training is not None
            or parameter_scope != "all"
        ):
            raise ValueError(
                "linear probes require the original catalog plan without native probe options"
            )
        _plan_pins(training, readouts_by_phase, catalog_training=True)
        from src.adapters.clm_linear_checkpoint_probes import make_linear_hook

        return make_linear_hook(
            training, readouts_by_phase, report, before_step=before_step, save=save
        )

    if (
        type(parameter_scope) is not str
        or parameter_scope not in ("all", "state_only")
        or (parameter_scope == "state_only" and not catalog_training)
    ):
        raise ValueError("state-only scope requires catalog reserved probes")
    if type(catalog_training) is not bool:
        raise ValueError("catalog probe selection must be an exact boolean")
    if type(report) is not dict or "reserved_probes" in report:
        raise ValueError("reserved checkpoint composition requires a fresh report field")
    if parameter_scope == "state_only":
        if report.get("probe_parameter_scope", "state_only") != "state_only":
            raise ValueError("original state-only probe declaration changed")
        report["probe_parameter_scope"] = "state_only"
    if not callable(before_step) or not callable(save):
        raise ValueError("reserved checkpoint boundaries must be callable")
    if type(marker_control) is not bool or (
        marker_control and (not catalog_training or rehearsal_training is not None)
    ):
        raise ValueError("extra-marker control requires catalog probes without rehearsal")
    if marker_control:
        if "probe_marker_control" in report and report["probe_marker_control"] is not True:
            raise ValueError("original marker control declaration changed")
        report["probe_marker_control"] = True
    if rehearsal_training is not None and not catalog_training:
        raise ValueError("rehearsal requires the catalog reserved probe")
    rehearsal_pins = (
        ()
        if rehearsal_training is None
        else _rehearsal_training_pins(rehearsal_training, readouts_by_phase)
    )
    if rehearsal_training is not None:
        if "probe_rehearsal" in report and report["probe_rehearsal"] is not True:
            raise ValueError("original rehearsal selection changed")
        report["probe_rehearsal"] = True
    if readout_training is not None and not catalog_training:
        raise ValueError("TRAIN readouts require catalog reserved probes")
    if (
        readout_training is not None
        and rehearsal_training is not None
        and readout_training is not rehearsal_training
    ):
        raise ValueError("TRAIN readouts and rehearsal require the same original history owner")
    readout_training_pins = (
        ()
        if readout_training is None
        else _rehearsal_training_pins(readout_training, readouts_by_phase)
    )
    if readout_training is not None:
        if "probe_train_readouts" in report and report["probe_train_readouts"] is not True:
            raise ValueError("original TRAIN readout declaration changed")
        report["probe_train_readouts"] = True
    pins = _plan_pins(training, readouts_by_phase, catalog_training=catalog_training)
    training_plan = _training_plan(training, catalog_training=catalog_training)
    training_owners = tuple(
        (phase, pair, pair[0], pair[1], deepcopy(pair[1])) for phase, pair in training_plan
    )
    original_groups = tuple(
        (phase, groups, tuple(groups.values())) for phase, groups in readouts_by_phase.items()
    )
    metadata = tuple(
        (type(batch), deepcopy(batch.identity), batch.role, getattr(batch, "candidate_ids", None))
        for batch, _ in pins
    )
    records = report["reserved_probes"] = []
    owners: list[dict[str, Any]] = []
    proof = "[]"

    def require_prefix():
        if parameter_scope == "state_only" and report.get("probe_parameter_scope") != "state_only":
            raise ValueError("original probe parameter scope changed")
        if readout_training is not None:
            if report.get("probe_train_readouts") is not True or tuple(readout_training) != tuple(
                row[0] for row in readout_training_pins
            ):
                raise ValueError("original TRAIN readout declaration or phase order changed")
            for phase, pair, batch, targets, original_targets, pin in readout_training_pins:
                _exact(batch, FrozenCatalogTrainingBatch)
                _exact(targets, CLMCatalogTargets)
                if (
                    readout_training[phase] is not pair
                    or pair[0] is not batch
                    or pair[1] is not targets
                    or targets != original_targets
                    or batch.role != "train"
                ):
                    raise ValueError("original TRAIN readout tuple or targets changed")
                _require_features(batch, pin)
                if (
                    batch.identity != training[phase][0].identity
                    or batch.candidate_ids != targets.candidate_ids
                ):
                    raise ValueError("original TRAIN readout feature metadata changed")
        if marker_control and report.get("probe_marker_control") is not True:
            raise ValueError("original extra-marker control declaration changed")
        if rehearsal_training is not None:
            if (
                report.get("training_replay_consent") is not True
                or report.get("probe_rehearsal") is not True
                or tuple(rehearsal_training) != tuple(row[0] for row in rehearsal_pins)
            ):
                raise ValueError("original rehearsal mode or phase order changed")
            for phase, pair, batch, targets, original_targets, pin in rehearsal_pins:
                if (
                    rehearsal_training[phase] is not pair
                    or pair[0] is not batch
                    or pair[1] is not targets
                    or targets != original_targets
                    or batch.role != "train"
                ):
                    raise ValueError("original rehearsal TRAIN owner or targets changed")
                _exact(batch, FrozenCatalogTrainingBatch)
                _exact(targets, CLMCatalogTargets)
                _require_features(batch, pin)
                if (
                    batch.identity != training[phase][0].identity
                    or batch.candidate_ids != targets.candidate_ids
                ):
                    raise ValueError("original rehearsal TRAIN metadata changed")
        if report.get("reserved_probes") is not records or len(records) not in (
            len(owners),
            len(owners) + 1,
        ):
            raise ValueError("original reserved checkpoint record owner changed")
        if any(records[index] is not original for index, original in enumerate(owners)):
            raise ValueError("completed reserved checkpoint record was replaced")
        if json.dumps(records[: len(owners)], sort_keys=True, allow_nan=False) != proof:
            raise ValueError("completed reserved checkpoint evidence changed")
        for batch, pin in pins:
            _require_features(batch, pin)
        current_plan = _training_plan(training, catalog_training=catalog_training)
        for (phase, pair), (original_phase, original_pair, batch, targets, original_targets) in zip(
            current_plan, training_owners, strict=True
        ):
            if (
                phase != original_phase
                or pair is not original_pair
                or pair[0] is not batch
                or pair[1] is not targets
            ):
                raise ValueError("original reserved training record replaced")
            _exact(targets, CLMCatalogTargets if catalog_training else CLMTargets)
            if targets != original_targets:
                raise ValueError("original reserved probe targets changed")
        if tuple(readouts_by_phase) != tuple(phase for phase, _, _ in original_groups):
            raise ValueError("original reserved checkpoint feature order changed")
        for phase, groups, batches in original_groups:
            if (
                readouts_by_phase[phase] is not groups
                or tuple(groups) != GROUPS
                or any(groups[key] is not batch for key, batch in zip(GROUPS, batches, strict=True))
            ):
                raise ValueError("original reserved readout records replaced")
        for (batch, _), (kind, identity, role, ids) in zip(pins, metadata, strict=True):
            _exact(batch, kind)
            if (
                batch.identity != identity
                or batch.role != role
                or getattr(batch, "candidate_ids", None) != ids
            ):
                raise ValueError("original reserved feature metadata changed")

    def checkpoint(phase, source):
        nonlocal proof
        require_prefix()
        index = len(owners)
        if index >= len(CHECKPOINTS) or (source.completed_updates, phase) != CHECKPOINTS[index]:
            raise ValueError("reserved checkpoint source or order changed")
        owner, original = source._bundle, source._capture()
        work_owner, rng_owner = source._work, source._rng
        source_work = source.work
        source_progress = (
            source.identity,
            source.policy,
            source.completed_updates,
            source.training_examples,
            source.generation,
        )
        child: dict[str, Any] = {}
        records.append(child)

        def source_guard():
            source._require_original(owner, original)
            if (
                source._work is not work_owner
                or source._rng is not rng_owner
                or source.work != source_work
                or (
                    source.identity,
                    source.policy,
                    source.completed_updates,
                    source.training_examples,
                    source.generation,
                )
                != source_progress
            ):
                raise ValueError("reserved probe changed benchmark source state or permanent work")
            require_prefix()

        def boundary():
            source_guard()
            before_step()
            source_guard()

        try:
            extra: dict[str, Any] = {"catalog_training": True} if catalog_training else {}
            if parameter_scope == "state_only":
                extra["parameter_scope"] = parameter_scope
            selected_rehearsal = rehearsal_training is not None and phase != "baseline"
            selected_marker_control = marker_control and phase != "baseline"
            selected_train_readouts = readout_training is not None and phase != "baseline"
            if selected_marker_control:
                extra["marker_control"] = build_probe_marker_control(training[phase], phase=phase)
            if selected_rehearsal or selected_train_readouts:
                # Why: only the observed checkpoint admits its phase's TRAIN rows.
                stream = report.get("stream", {})
                if (
                    stream.get("completed_updates") != source.completed_updates
                    or len(stream.get("receipts", [])) != source.completed_updates
                    or len(report.get("rows", [])) != source.completed_updates
                ):
                    raise ValueError(
                        "rehearsal TRAIN history has not reached its source checkpoint"
                    )
            if selected_train_readouts:
                extra["training_readout"] = readout_training[phase]
            if selected_rehearsal:
                extra["replay_budget"] = ToyBudgetSession(
                    ToyExecutionBudget(max_replay_examples=12), lambda: 0.0
                )
                extra["rehearsal"] = build_probe_rehearsal(
                    training[phase],
                    rehearsal_training[phase],
                    phase=phase,
                    permissions=ExperiencePermissions(training=True, replay=True),
                )
            run_reserved_probe(
                source,
                training[phase] if catalog_training else training,
                readouts_by_phase[phase],
                child,
                phase=phase,
                probe_factory=(
                    lambda source: _new_catalog_probe(
                        source,
                        max_readouts=8 if selected_train_readouts else 6,
                        parameter_scope=parameter_scope,
                    )
                )
                if selected_train_readouts or parameter_scope == "state_only"
                else (_new_catalog_probe if catalog_training else _new_probe),
                permissions=ExperiencePermissions(training=True),
                budget=ToyBudgetSession(
                    ToyExecutionBudget(max_training_updates=PROBE_UPDATES),
                    lambda: 0.0,
                ),
                before_step=boundary,
                save=save,
                **extra,
            )
            _require_cached_probe(
                child,
                phase,
                catalog_training=catalog_training,
                rehearsal=selected_rehearsal,
                marker_control=selected_marker_control,
                train_readouts=selected_train_readouts,
                parameter_scope=parameter_scope,
            )
        finally:
            boundary()
        owners.append(child)
        proof = json.dumps(records, sort_keys=True, allow_nan=False)

    return checkpoint


def _catalog_markers(report, work_key):
    work = report.get(work_key)
    return "training_objective" in report or (
        type(work) is dict and any(type(key) is str and key.startswith("catalog_") for key in work)
    )


def _require_cached_probe(
    child,
    phase,
    *,
    catalog_training=False,
    rehearsal=False,
    marker_control=False,
    train_readouts=False,
    parameter_scope="all",
):
    if (
        type(parameter_scope) is not str
        or parameter_scope not in ("all", "state_only")
        or (parameter_scope == "state_only" and not catalog_training)
    ):
        raise ValueError("explicit catalog probe parameter scope required")
    if type(child) is not dict or child.get("complete") is not True or child.get("phase") != phase:
        raise ValueError("complete phase-bound reserved probe required")
    if ("parameter_update_scope" in child) != (parameter_scope == "state_only"):
        raise ValueError("reserved parameter scope evidence differs from mode")
    if parameter_scope == "state_only":
        record = child["parameter_update_scope"]
        if (
            type(record) is not dict
            or set(record)
            != {
                "name",
                "optimized_parameter_elements",
                "fixed_parameter_elements",
                "total_parameter_elements",
                "fixed_parameters_unchanged",
            }
            or record.get("name") != "state_only"
            or any(
                type(record.get(key)) is not int
                for key in (
                    "optimized_parameter_elements",
                    "fixed_parameter_elements",
                    "total_parameter_elements",
                )
            )
            or not 0
            < record["optimized_parameter_elements"]
            < record["total_parameter_elements"]
            <= 32000000
            or record["fixed_parameter_elements"]
            != record["total_parameter_elements"] - record["optimized_parameter_elements"]
            or record.get("fixed_parameters_unchanged") is not True
        ):
            raise ValueError("complete fixed/optimized parameter evidence required")
    if _catalog_markers(child, "probe_work") != catalog_training:
        raise ValueError("reserved probe objective and source mode differ")
    if catalog_training:
        examples = reserved_probe_examples(phase, "probe_train")
        ids = [row.candidate_id for row in examples[0].request.allowed_candidates]
        if (
            child.get("training_objective") != CATALOG_OBJECTIVE
            or type(child.get("candidate_ids")) is not list
            or child["candidate_ids"] != ids
            or type(child.get("candidate_indices")) is not list
            or any(type(value) is not int for value in child["candidate_indices"])
            or child["candidate_indices"] != [ids.index(row.target_id) for row in examples]
        ):
            raise ValueError("reserved catalog objective or indexed labels changed")
    if type(marker_control) is not bool or (
        marker_control and (rehearsal or not catalog_training or phase == "baseline")
    ):
        raise ValueError("extra-marker control requires one non-baseline catalog probe")
    if ("marker_control" in child) != marker_control:
        raise ValueError("reserved extra-marker metadata differs from selected mode")
    if marker_control:
        record = child["marker_control"]
        if (
            type(record) is not dict
            or set(record)
            != {
                "source_phase",
                "source_role",
                "source_row_indices",
                "retained_tensor_storage_bytes",
                "extra_marker_rows_attempted",
            }
            or record.get("source_phase") != phase
            or record.get("source_role") != "probe_train"
            or record.get("source_row_indices") != [[0, 1]] * PROBE_UPDATES
            or any(type(value) is not int for row in record["source_row_indices"] for value in row)
            or type(record.get("extra_marker_rows_attempted")) is not int
            or record["extra_marker_rows_attempted"] != 12
            or type(record.get("retained_tensor_storage_bytes")) is not int
            or not 0 < record["retained_tensor_storage_bytes"] <= 1024**2
        ):
            raise ValueError("complete extra-marker exposure/storage record required")
    if type(rehearsal) is not bool or (rehearsal and (not catalog_training or phase == "baseline")):
        raise ValueError("catalog rehearsal requires an arrived TRAIN checkpoint")
    if ("rehearsal" in child) != rehearsal:
        raise ValueError("reserved rehearsal metadata differs from selected mode")
    if rehearsal:
        record = child["rehearsal"]
        if (
            type(record) is not dict
            or set(record)
            != {
                "source_phase",
                "source_role",
                "source_row_indices",
                "retained_tensor_storage_bytes",
                "replay_rows_attempted",
            }
            or record.get("source_phase") != phase
            or record.get("source_role") != "train"
            or record.get("source_row_indices")
            != [[2 * i % 3, (2 * i + 1) % 3] for i in range(PROBE_UPDATES)]
            or any(type(value) is not int for row in record["source_row_indices"] for value in row)
            or type(record.get("replay_rows_attempted")) is not int
            or record["replay_rows_attempted"] != 12
            or type(record.get("retained_tensor_storage_bytes")) is not int
            or not 0 < record["retained_tensor_storage_bytes"] <= 1024**2
        ):
            raise ValueError("complete rehearsal source/exposure/storage record required")
    if type(train_readouts) is not bool or (
        train_readouts and (not catalog_training or phase == "baseline")
    ):
        raise ValueError("TRAIN readouts require a non-baseline catalog probe")
    if ("arrived_train_readouts" in child) != train_readouts:
        raise ValueError("arrived TRAIN readout record differs from selected mode")
    if train_readouts:
        record = child["arrived_train_readouts"]
        train_examples = temporal_examples(phase, "train")
        ids = [row.candidate_id for row in train_examples[0].request.allowed_candidates]
        if (
            type(record) is not dict
            or set(record)
            != {
                "source_phase",
                "source_role",
                "source_row_indices",
                "candidate_ids",
                "candidate_indices",
                "retained_tensor_storage_bytes",
                "readouts",
            }
            or record.get("source_phase") != phase
            or record.get("source_role") != "train"
            or type(record.get("source_row_indices")) is not list
            or record["source_row_indices"] != [0, 1, 2]
            or any(type(value) is not int for value in record["source_row_indices"])
            or type(record.get("candidate_ids")) is not list
            or record["candidate_ids"] != ids
            or type(record.get("candidate_indices")) is not list
            or record["candidate_indices"] != [ids.index(row.target_id) for row in train_examples]
            or any(type(value) is not int for value in record["candidate_indices"])
            or type(record.get("retained_tensor_storage_bytes")) is not int
            or not 0 < record["retained_tensor_storage_bytes"] <= 1024**2
        ):
            raise ValueError("original arrived TRAIN source/label/storage record required")
        extra_reads = record["readouts"]
        if type(extra_reads) is not list or len(extra_reads) != 2:
            raise ValueError("both arrived TRAIN endpoint readouts required")
        for row, point in zip(extra_reads, ("before", "after"), strict=True):
            if (
                type(row) is not dict
                or set(row) != {"point", "probabilities"}
                or row.get("point") != point
            ):
                raise ValueError("arrived TRAIN endpoint identity/order changed")
            values = row["probabilities"]
            if type(values) is not list or len(values) != 3:
                raise ValueError("three arrived TRAIN probability rows required")
            for value in values:
                if (
                    type(value) is not list
                    or len(value) != 3
                    or any(
                        type(p) is not float or not math.isfinite(p) or not 0 <= p <= 1
                        for p in value
                    )
                    or not math.isclose(sum(value), 1.0, rel_tol=0, abs_tol=1e-6)
                ):
                    raise ValueError("finite normalized arrived TRAIN probabilities required")
    source_update = dict((phase, update) for update, phase in CHECKPOINTS)[phase]
    if type(child.get("source_update")) is not int or child["source_update"] != source_update:
        raise ValueError("reserved probe source update differs")
    stream = child.get("stream")
    if (
        type(stream) is not dict
        or type(stream.get("completed_updates")) is not int
        or stream.get("completed_updates") != PROBE_UPDATES
        or type(stream.get("training_pairs")) is not int
        or stream.get("training_pairs") != (24 if rehearsal or marker_control else 12)
        or type(stream.get("receipts")) is not list
        or len(stream["receipts"]) != PROBE_UPDATES
    ):
        raise ValueError("six actual reserved probe receipts required")
    for index, receipt in enumerate(stream["receipts"]):
        sample = f"clm-probe-{phase}-{index}"
        expected = dict(
            sample_id=sample,
            episode_id=f"clm-probe-{phase}",
            event_id="label-" + sample,
            actor_version=phase,
            learner_version=f"clm-probe-{phase}",
            observed_at=2 * index,
            arrived_at=2 * index + 1,
            applied_at=2 * index + 1,
            update_number=index + 1,
        )
        if type(receipt) is not dict or any(
            type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in expected.items()
        ):
            raise ValueError("reserved receipt identity, clocks or update changed")
        diagnostic = receipt.get("diagnostic")
        if (
            type(diagnostic) is not dict
            or diagnostic.get("definition")
            != (
                CATALOG_OBJECTIVE
                if catalog_training
                else "clm_paired_group_masked_symmetric_cross_entropy_v1"
            )
            or type(diagnostic.get("value")) is not float
            or not math.isfinite(diagnostic["value"])
            or diagnostic["value"] < 0
        ):
            raise ValueError("reserved receipt native diagnostic changed")
    work = child.get("probe_work")
    expected_work = dict.fromkeys(
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
            "prediction_attempts",
            "prediction_returns",
            "choice_prediction_attempts",
            "choice_prediction_returns",
        ),
        6,
    )
    expected_work.update(
        training_rows_attempted=24 if rehearsal or marker_control else 12,
        prediction_rows_attempted=16,
        choice_candidate_rows_attempted=18,
        snapshot_attempts=0,
        snapshot_returns=0,
        restore_attempts=0,
        restore_returns=0,
        unreturned_work_unknown=False,
    )
    if train_readouts:
        expected_work.update(
            dict.fromkeys(
                (
                    "prediction_attempts",
                    "prediction_returns",
                    "choice_prediction_attempts",
                    "choice_prediction_returns",
                ),
                8,
            )
        )
        expected_work.update(prediction_rows_attempted=22, choice_candidate_rows_attempted=24)
    if catalog_training:
        expected_work.update(
            catalog_candidate_rows_attempted=18,
            catalog_logit_values_attempted=72 if rehearsal or marker_control else 36,
        )
    if (
        type(work) is not dict
        or set(work) != set(expected_work)
        or any(type(work[k]) is not type(v) or work[k] != v for k, v in expected_work.items())
    ):
        raise ValueError("reserved probe physical work differs from fixed dose")
    rows = child.get("readouts")
    expected_reads = tuple(
        (point, group, size)
        for point in ("before", "after")
        for group, size in zip(GROUPS, (2, 2, 4), strict=True)
    )
    if type(rows) is not list or len(rows) != 6:
        raise ValueError("all six reserved endpoint readouts required")
    for row, (point, group, size) in zip(rows, expected_reads, strict=True):
        if type(row) is not dict or row.get("point") != point or row.get("group") != group:
            raise ValueError("reserved endpoint group/order changed")
        values = row.get("probabilities")
        if type(values) is not list or len(values) != size:
            raise ValueError("reserved cached query rows differ")
        for value in values:
            if (
                type(value) is not list
                or len(value) != 3
                or any(
                    type(p) is not float or not math.isfinite(p) or not 0 <= p <= 1 for p in value
                )
                or not math.isclose(sum(value), 1.0, rel_tol=0, abs_tol=1e-6)
            ):
                raise ValueError("finite normalized reserved probabilities required")


def _reserved_probe_metrics(probabilities, examples):
    rows = []
    for example, values in zip(examples, probabilities, strict=True):
        ids = tuple(candidate.candidate_id for candidate in example.request.allowed_candidates)
        target = ids.index(example.target_id)
        winner = max(range(len(ids)), key=values.__getitem__)
        obsolete = getattr(example, "obsolete_target_id", None)
        rows.append(
            dict(
                request=asdict(example.request),
                role=example.role,
                source_group=example.source_group,
                phase=example.phase,
                concept=example.concept,
                target_id=example.target_id,
                selected_id=ids[winner],
                correct=winner == target,
                target_probability=values[target],
                probabilities=list(zip(ids, values, strict=True)),
                confidence=values[winner],
                candidate_recalled=True,
                brier=sum(
                    (value - float(index == target)) ** 2 for index, value in enumerate(values)
                ),
                obsolete_target_id=obsolete,
                obsolete_rule_error=obsolete is not None and ids[winner] == obsolete,
            )
        )
    return dict(
        cases=rows,
        metrics=_metrics(rows),
        obsolete_rule_errors=sum(row["obsolete_rule_error"] for row in rows),
    )


def _require_benchmark_completion(report):
    if type(report) is not dict or report.get("sequence_complete") is not True:
        raise ValueError("finish all benchmark and probe training before cached scoring")
    stream = report.get("stream")
    if (
        type(stream) is not dict
        or stream.get("completed_updates") != 12
        or stream.get("training_pairs") != 36
    ):
        raise ValueError("original fixed benchmark dose required")
    if report.get("catalog_support", "none") != "none":
        raise ValueError("reserved probes require the original benchmark recipe")
    catalog_training = _catalog_markers(report, "learner_work")
    if catalog_training and report.get("training_objective") != CATALOG_OBJECTIVE:
        raise ValueError("explicit catalog benchmark objective required")
    diagnostic = "train_raw_readouts" in report
    expected_work = _expected_work(12, 8 if diagnostic else 5, 29 if diagnostic else 20)
    expected_work["training_rows_attempted"] = 36
    if catalog_training:
        expected_work.update(
            catalog_candidate_rows_attempted=36, catalog_logit_values_attempted=108
        )
    _require_work(report.get("learner_work"), expected_work)
    receipts, rows = stream.get("receipts"), report.get("rows")
    if (
        type(receipts) is not list
        or len(receipts) != 12
        or type(rows) is not list
        or len(rows) != 12
    ):
        raise ValueError("twelve actual benchmark receipts and rows required")
    for index, (receipt, row) in enumerate(zip(receipts, rows, strict=True)):
        phase = CHECKPOINTS[1 + index // 4][1]
        sample = f"clm-{phase}-{index % 4}"
        expected = dict(
            sample_id=sample,
            episode_id="clm-temporal",
            event_id="label-" + sample,
            actor_version=phase,
            learner_version="clm-temporal",
            observed_at=2 * index,
            arrived_at=2 * index + 1,
            applied_at=2 * index + 1,
            update_number=index + 1,
        )
        if type(receipt) is not dict or any(
            type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in expected.items()
        ):
            raise ValueError("benchmark receipt identity, clocks or update changed")
        value = receipt.get("diagnostic")
        if (
            type(value) is not dict
            or value.get("definition")
            != (
                CATALOG_OBJECTIVE
                if catalog_training
                else "clm_paired_group_masked_symmetric_cross_entropy_v1"
            )
            or type(value.get("value")) is not float
            or not math.isfinite(value["value"])
            or value["value"] < 0
        ):
            raise ValueError("benchmark receipt diagnostic changed")
        if (
            type(row) is not dict
            or row.get("update") != index + 1
            or row.get("phase") != phase
            or row.get("exposure") != index % 4
            or row.get("loss") != value["value"]
        ):
            raise ValueError("benchmark rows differ from actual receipts")
        if catalog_training:
            examples = catalog_training_examples(phase, "none", exposure=index % 4)
            ids = [candidate.candidate_id for candidate in examples[0].request.allowed_candidates]
            if (
                type(row.get("update")) is not int
                or type(row.get("exposure")) is not int
                or row.get("candidate_ids") != ids
                or type(row.get("candidate_indices")) is not list
                or any(type(value) is not int for value in row["candidate_indices"])
                or row["candidate_indices"] != [ids.index(item.target_id) for item in examples]
            ):
                raise ValueError("benchmark catalog rows differ from observed labels")
    return catalog_training


def score_reserved_probes(report, *, score_group, metric_difference):
    catalog_training = _require_benchmark_completion(report)
    if "probe_linear_fit_control" in report:
        if report["probe_linear_fit_control"] is not True or not catalog_training:
            raise ValueError("explicit catalog linear probe mode required")
        from src.adapters.clm_linear_checkpoint_probes import score_linear_records

        return score_linear_records(report, score_group=score_group)
    rehearsal = report.get("probe_rehearsal", False)
    if ("probe_rehearsal" in report and rehearsal is not True) or (
        rehearsal and (not catalog_training or report.get("training_replay_consent") is not True)
    ):
        raise ValueError("explicit catalog rehearsal mode required")
    marker_control = report.get("probe_marker_control", False)
    if ("probe_marker_control" in report and marker_control is not True) or (
        marker_control
        and (not catalog_training or rehearsal or "training_replay_consent" in report)
    ):
        raise ValueError(
            "explicit extra-marker control requires catalog mode without replay consent"
        )
    train_readouts = report.get("probe_train_readouts", False)
    if ("probe_train_readouts" in report and train_readouts is not True) or (
        train_readouts and not catalog_training
    ):
        raise ValueError("explicit catalog TRAIN readout mode required")
    parameter_scope = report.get("probe_parameter_scope", "all")
    if "probe_parameter_scope" in report and (
        type(parameter_scope) is not str or parameter_scope != "state_only" or not catalog_training
    ):
        raise ValueError("explicit state-only catalog probe mode required")
    children = report.get("reserved_probes")
    if type(children) is not list or len(children) != 4:
        raise ValueError("all four reserved checkpoint probes required")
    # Why: no partial earlier result is scored while later evidence remains unadmitted.
    for child, (_, phase) in zip(children, CHECKPOINTS, strict=True):
        _require_cached_probe(
            child,
            phase,
            catalog_training=catalog_training,
            rehearsal=rehearsal and phase != "baseline",
            marker_control=marker_control and phase != "baseline",
            train_readouts=train_readouts and phase != "baseline",
            parameter_scope=parameter_scope,
        )
    if parameter_scope == "state_only" and any(
        child["parameter_update_scope"] != children[0]["parameter_update_scope"]
        for child in children[1:]
    ):
        raise ValueError("original probe parameter counts differ across checkpoints")
    checkpoints = []
    for child in children:
        groups = {}
        for index, group in enumerate(GROUPS):
            examples = (
                temporal_examples(child["phase"], "outer_selection")
                if group == "old_task"
                else reserved_probe_examples(child["phase"], group)
            )
            before = score_group(child["readouts"][index]["probabilities"], examples)
            after = score_group(child["readouts"][index + 3]["probabilities"], examples)
            difference = metric_difference(after["metrics"], before["metrics"])
            difference["obsolete_rule_error_count"] = (
                after["obsolete_rule_errors"] - before["obsolete_rule_errors"]
            )
            groups[group] = dict(before=before, after=after, after_minus_before=difference)
        if train_readouts and child["phase"] != "baseline":
            raw = child["arrived_train_readouts"]["readouts"]
            examples = temporal_examples(child["phase"], "train")
            before = score_group(raw[0]["probabilities"], examples)
            after = score_group(raw[1]["probabilities"], examples)
            difference = metric_difference(after["metrics"], before["metrics"])
            difference["obsolete_rule_error_count"] = (
                after["obsolete_rule_errors"] - before["obsolete_rule_errors"]
            )
            groups["arrived_train"] = dict(
                before=before, after=after, after_minus_before=difference
            )
        checkpoints.append(
            dict(phase=child["phase"], source_update=child["source_update"], groups=groups)
        )
    return dict(
        checkpoints=checkpoints,
        cached_probability_rows=82 if train_readouts else 64,
        direction="accuracy higher; NLL/Brier/ECE and obsolete errors lower; all endpoints retained",
        scope="fixed reserved CLM probe, no final/CPC/population generalization claim",
    )
