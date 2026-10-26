"""Cold raw-feature control at the existing reserved checkpoint boundary.

Consumes original role-bound plans and an immutable source checkpoint. Reports
one two-row fit plus three read-only groups, with no warm weights, source update,
probe export, native optimizer, final access or new stream/experiment authority.
"""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from typing import Any

import torch

from src.adapters.clm_linear_fit_control import (
    CLMLinearFitControl,
    fit_clm_linear_control,
    predict_clm_linear_control,
    _capture,
    _require_original,
    _features,
)
from src.app.clm_reserved_probe import _require_work
from src.core.clm_head import _exact
from src.core.clm_probe import FrozenProbeChoiceBatch
from src.core.temporal_clm_probe import reserved_probe_examples
from src.core.temporal_decision_sequence import CHECKPOINTS, temporal_examples

DEFINITION = "cold_raw_feature_reserved_linear_control_v1"
GROUPS = ("probe_train", "probe_test", "old_task")
NATIVE_OPTIONS = (
    "reserved_probes",
    "probe_rehearsal",
    "probe_marker_control",
    "probe_train_readouts",
    "probe_parameter_scope",
    "training_replay_consent",
)


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _require_mode(report):
    if any(key in report for key in NATIVE_OPTIONS):
        raise ValueError("linear control cannot acquire native probe/replay options")


def _probabilities(value, rows):
    if (
        type(value) is not list
        or len(value) != rows
        or any(
            type(row) is not list
            or len(row) != 3
            or any(type(p) is not float or not math.isfinite(p) or not 0 <= p <= 1 for p in row)
            or not math.isclose(sum(row), 1.0, rel_tol=0, abs_tol=1e-12)
            for row in value
        )
    ):
        raise ValueError("complete finite linear-control probability rows required")


def _require_operation(row, rows, width):
    expected = dict(
        complete=True,
        attempts=1,
        returns=1,
        rows_attempted=rows,
        rows_returned=rows,
        feature_dim=width,
        classes=3,
        unreturned_work_unknown=False,
        native_head_forwards=0,
        encoder_forwards=0,
        optimizer_steps=0,
    )
    if type(row) is not dict or any(
        type(row.get(k)) is not type(v) or row[k] != v for k, v in expected.items()
    ):
        raise ValueError("linear-control paid operation differs from its fixed dose")


def require_linear_record(child, phase, update):
    if (
        type(child) is not dict
        or child.get("definition") != DEFINITION
        or child.get("phase") != phase
        or type(child.get("source_update")) is not int
        or child["source_update"] != update
        or child.get("complete") is not True
        or child.get("initialization") != "cold_raw_features_no_source_weights"
        or type(child.get("optimizer_steps")) is not int
        or child["optimizer_steps"] != 0
        or child.get("pending_readout") is not None
    ):
        raise ValueError("complete phase-bound cold linear control required")
    work = child.get("linear_fit", {})
    width = work.get("feature_dim")
    if type(width) is not int or not 1 <= width <= 4096:
        raise ValueError("bounded fitted feature dimension required")
    _require_operation(work, 2, width)
    summary = child.get("fit_summary", {})
    if (
        type(summary.get("rank")) is not int
        or not 0 <= summary["rank"] <= 2
        or type(summary.get("squared_error")) is not float
        or not math.isfinite(summary["squared_error"])
        or summary["squared_error"] < 0
        or type(summary.get("singular_values")) is not list
        or len(summary["singular_values"]) != 2
        or any(
            type(v) is not float or not math.isfinite(v) or v < 0
            for v in summary["singular_values"]
        )
    ):
        raise ValueError("complete finite fit diagnostics required")
    ids = [
        c.candidate_id
        for c in reserved_probe_examples(phase, "probe_train")[0].request.allowed_candidates
    ]
    reads = child.get("readouts")
    if type(reads) is not list or len(reads) != 3:
        raise ValueError("three completed linear readouts required")
    for read, group, rows in zip(reads, GROUPS, (2, 2, 4), strict=True):
        role = "outer_selection" if group == "old_task" else group
        if (
            read.get("group") != group
            or read.get("role") != role
            or read.get("candidate_ids") != ids
        ):
            raise ValueError("original linear readout role/catalog/order differs")
        _require_operation(read.get("work"), rows, width)
        _probabilities(read.get("probabilities"), rows)


def make_linear_hook(training, readouts, report, *, before_step, save):
    """Called after the original plan admission; keep its values/owners at birth."""
    _require_mode(report)
    if (
        not callable(before_step)
        or not callable(save)
        or any(
            k in report
            for k in (
                "linear_reserved_probes",
                "probe_linear_fit_control",
                "reserved_probes",
                "pending_linear_probe",
                "linear_probe_proof",
            )
        )
    ):
        raise ValueError("linear probes require fresh records and callable boundaries")
    phases = tuple(phase for _, phase in CHECKPOINTS)
    plan = tuple(
        (phase, training[phase], readouts[phase], tuple(readouts[phase].values()))
        for phase in phases
    )
    pins = []
    for phase, pair, groups, batches in plan:
        batch, targets = pair
        _features(batch, batch.candidate_ids)
        pins.append((batch, targets, _capture(batch, targets)))
        for selected in batches:
            _exact(selected, FrozenProbeChoiceBatch)
            _features(selected, batch.candidate_ids, min_rows=1)
            if not torch.equal(selected.actions, batch.actions):
                raise ValueError("linear readout must share the original raw action catalog")
            pins.append((selected, None, _capture(selected)))
    records: list[dict[str, Any]] = []
    owners: list[dict[str, Any]] = []
    proof = "[]"
    failed = False
    report.update(
        probe_linear_fit_control=True,
        linear_reserved_probes=records,
        pending_linear_probe=None,
        linear_probe_proof=proof,
    )

    def require_prefix():
        _require_mode(report)
        if failed:
            raise ValueError("linear probe hook stopped after uncertain paid work")
        if (
            report.get("probe_linear_fit_control") is not True
            or report.get("linear_reserved_probes") is not records
            or len(records) != len(owners)
            or any(a is not b for a, b in zip(records, owners, strict=True))
            or _json(records) != proof
            or report.get("linear_probe_proof") != proof
        ):
            raise ValueError("original completed linear probe evidence changed")
        if tuple(training) != phases or tuple(readouts) != phases:
            raise ValueError("original linear checkpoint order changed")
        for phase, pair, groups, batches in plan:
            if (
                training[phase] is not pair
                or readouts[phase] is not groups
                or tuple(groups) != GROUPS
                or any(groups[k] is not b for k, b in zip(GROUPS, batches, strict=True))
            ):
                raise ValueError("original linear feature plan owner changed")
        for batch, targets, pin in pins:
            _require_original(batch, targets, pin)

    def checkpoint(phase, source):
        nonlocal failed, proof
        require_prefix()
        index = len(owners)
        if (
            index >= 4
            or type(source.completed_updates) is not int
            or (source.completed_updates, phase) != CHECKPOINTS[index]
        ):
            raise ValueError("linear checkpoint source/order differs")
        if source.identity != training[phase][0].identity:
            raise ValueError("linear probe and source encoder differ")
        owner, original = source._bundle, source._capture()
        work_owner, rng_owner, work = source._work, source._rng, source.work
        progress = deepcopy(
            (
                source.identity,
                source.policy,
                source.completed_updates,
                source.training_examples,
                source.generation,
            )
        )
        caller = torch.get_rng_state().clone()
        child: dict[str, Any] = dict(
            definition=DEFINITION,
            phase=phase,
            source_update=source.completed_updates,
            initialization="cold_raw_features_no_source_weights",
            optimizer_steps=0,
            complete=False,
            readouts=[],
            pending_readout=None,
        )
        trusted = {"readouts": (child["readouts"], "[]")}
        published = False
        pending_owner = None
        fitted = None
        fitted_fields = None
        fitted_proof = None
        report["pending_linear_probe"] = child
        birth = tuple(
            child[k]
            for k in ("definition", "phase", "source_update", "initialization", "optimizer_steps")
        )

        def guard():
            require_prefix()
            if (
                report.get("pending_linear_probe") is not (None if published else child)
                or tuple(
                    child[k]
                    for k in (
                        "definition",
                        "phase",
                        "source_update",
                        "initialization",
                        "optimizer_steps",
                    )
                )
                != birth
                or type(child["optimizer_steps"]) is not int
                or child["complete"] is not published
                or child.get("pending_readout") is not pending_owner
            ):
                raise ValueError("original pending linear probe record changed")
            source._require_original(owner, original)
            _require_work(source.work, work)
            if (
                source._work is not work_owner
                or source._rng is not rng_owner
                or (
                    source.identity,
                    source.policy,
                    source.completed_updates,
                    source.training_examples,
                    source.generation,
                )
                != progress
                or not torch.equal(torch.get_rng_state(), caller)
                or any(
                    type(v) is not int
                    for v in (source.completed_updates, source.training_examples, source.generation)
                )
            ):
                raise ValueError("linear control changed source state/work/owners/RNG")
            for key, (value, text) in trusted.items():
                if child.get(key) is not value or _json(child[key]) != text:
                    raise ValueError("original linear probe paid operation changed")
            if fitted_fields is not None:
                _exact(fitted, CLMLinearFitControl)
                if (
                    any(getattr(fitted, k) is not v for k, v in fitted_fields.items())
                    or _json(asdict(fitted)) != fitted_proof
                ):
                    raise ValueError("original fitted linear probe model changed")

        def boundary():
            guard()
            before_step()
            guard()

        def persist():
            guard()
            save()
            guard()

        try:
            boundary()
            persist()
            batch, targets = training[phase]
            fitted = fit_clm_linear_control(batch, targets, child)
            guard()
            _exact(fitted, CLMLinearFitControl)
            if (
                fitted.role != "probe_train"
                or fitted.identity != source.identity
                or fitted.candidate_ids != batch.candidate_ids
            ):
                raise ValueError("linear probe returned a different role/encoder/catalog")
            _require_operation(child["linear_fit"], 2, batch.identity.embedding_dim)
            fitted_fields = dict(vars(fitted))
            fitted_proof = _json(asdict(fitted))
            child["fit_summary"] = dict(
                rank=fitted.model.rank,
                singular_values=list(fitted.model.singular_values),
                squared_error=fitted.model.squared_error,
            )
            for key in ("linear_fit", "fit_summary"):
                trusted[key] = child[key], _json(child[key])
            persist()
            for group in GROUPS:
                boundary()
                selected = readouts[phase][group]
                row = dict(
                    group=group,
                    role=selected.role,
                    candidate_ids=list(fitted.candidate_ids),
                    work={},
                )
                pending_owner = row
                child["pending_readout"] = row
                values = predict_clm_linear_control(
                    fitted, selected, fitted.candidate_ids, row["work"]
                )
                guard()
                row["probabilities"] = values.tolist()
                _require_operation(
                    row["work"]["linear_prediction"],
                    len(selected.states),
                    batch.identity.embedding_dim,
                )
                _probabilities(row["probabilities"], len(selected.states))
                row["work"] = row["work"]["linear_prediction"]
                child["readouts"].append(row)
                pending_owner = None
                child["pending_readout"] = None
                trusted["readouts"] = child["readouts"], _json(child["readouts"])
                persist()
            child["complete"] = True
            require_linear_record(child, phase, source.completed_updates)
            child["complete"] = False
            guard()
            child["complete"] = True
            records.append(child)
            owners.append(child)
            proof = _json(records)
            published = True
            report["pending_linear_probe"] = None
            report["linear_probe_proof"] = proof
            persist()
        except Exception as error:
            failed = True
            child.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])
            child["preserved_paid_prefix"] = {k: json.loads(v[1]) for k, v in trusted.items()}
            report["failed_linear_probe"] = child
            report["preserved_linear_probe_prefix"] = json.loads(proof)
            raise

    return checkpoint


def score_linear_records(report, *, score_group):
    _require_mode(report)
    children = report.get("linear_reserved_probes")
    if (
        report.get("probe_linear_fit_control") is not True
        or type(children) is not list
        or len(children) != 4
    ):
        raise ValueError("four completed linear checkpoint controls required")
    if report.get("pending_linear_probe") is not None or report.get("linear_probe_proof") != _json(
        children
    ):
        raise ValueError("original linear probe completion evidence changed")
    for child, (update, phase) in zip(children, CHECKPOINTS, strict=True):
        require_linear_record(child, phase, update)
    checkpoints = []
    for child in children:
        phase = child["phase"]
        groups = {}
        for row, group in zip(child["readouts"], GROUPS, strict=True):
            examples = (
                temporal_examples(phase, "outer_selection")
                if group == "old_task"
                else reserved_probe_examples(phase, group)
            )
            groups[group] = score_group(row["probabilities"], examples)
        checkpoints.append(dict(phase=phase, source_update=child["source_update"], groups=groups))
    return dict(
        definition=DEFINITION,
        checkpoints=checkpoints,
        cached_probability_rows=32,
        direction="accuracy higher; NLL/Brier/ECE and obsolete errors lower; all endpoints retained",
        scope="cold raw-feature separability; old-task compatibility, not warm plasticity/retention; no final/CPC claim",
    )
