"""Compose three fixed TRAIN cue-pair diagnostics after six raw fit scores.

Inputs are original MiniGrid packets and one owned checkpoint; output extends
the raw report with scalar gradient geometry and separate paid work. No learner
update, environment, final/probe access or new orchestration framework lives here.
"""

from collections.abc import Callable
from copy import deepcopy
import pickle
from time import monotonic
from typing import Any

import numpy as np

from src.adapters import minigrid_learning_pilot as shared
from src.adapters.torch_catalogue_weighted import (
    catalogue_class_weights,
    prepare_train_catalogue,
)
from src.adapters.torch_feature_gradient_diagnostics import diagnose_prior_gradients
from src.adapters.torch_pc_update_diagnostics import (
    diagnose_pc_update,
    DEFINITION as PC_DEFINITION,
    WEIGHTED_DEFINITION as PC_WEIGHTED_DEFINITION,
)
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_gradient_diagnostics import (
    GradientDiagnosticFailed,
    diagnose_episode_gradients,
)
from src.app.coupled_attribution import AttributionBatch, _validate_batch
from src.core.action_memory import _feature_keys
from src.core.categorical_pc_weighted import OBJECTIVE as PC_WEIGHTED_OBJECTIVE
from src.core.minigrid_invariant_encoding import canonicalize_public_history

NATIVE_WORK_KEYS = (
    "snapshot_validations_attempted",
    "snapshot_validations_completed",
    "validation_temporary_optimizers",
    "forward_calls_attempted",
    "forward_calls_completed",
    "forward_rows_attempted",
    "forward_rows_completed",
    "backward_calls_attempted",
    "backward_calls_completed",
    "backward_rows_attempted",
    "backward_rows_completed",
    "optimizer_steps",
    "published_updates",
)
PRIOR_NATIVE_WORK_KEYS = NATIVE_WORK_KEYS + (
    "projection_calls_attempted",
    "projection_calls_completed",
    "projection_rows_attempted",
    "projection_rows_completed",
    "projection_array_bytes_returned",
)
PC_NATIVE_WORK_KEYS = (
    "snapshot_validations_attempted",
    "snapshot_validations_completed",
    "validation_temporary_optimizers",
    "projection_calls_attempted",
    "projection_calls_completed",
    "projection_rows_attempted",
    "projection_rows_completed",
    "projection_array_bytes_returned",
    "prior_forward_calls_attempted",
    "prior_forward_calls_completed",
    "prior_forward_rows_attempted",
    "prior_forward_rows_completed",
    "prior_backward_calls_attempted",
    "prior_backward_calls_completed",
    "prior_backward_rows_attempted",
    "prior_backward_rows_completed",
    "relaxation_forward_calls_attempted",
    "relaxation_forward_calls_completed",
    "relaxation_forward_rows_attempted",
    "relaxation_forward_rows_completed",
    "candidate_parameter_sets_completed",
    "native_train_calls",
    "optimizer_steps",
    "published_updates",
)
PC_WEIGHTED_NATIVE_WORK_KEYS = PC_NATIVE_WORK_KEYS + tuple(
    f"{group}_{event}"
    for group in (
        "proposal_calls",
        "proposal_rows",
        "proposal_nominal_latent_row_iterations",
        "fixed_state_energy_calls",
        "fixed_state_energy_rows",
        "weight_derivations",
        "weight_rows",
    )
    for event in ("attempted", "completed")
)


def _validate_pc_weighted(weighted):
    if type(weighted) is not bool:
        raise ValueError("weighted PC diagnostics require an exact boolean flag")


def validate_pc_option(flag, *, joint_catalogue, qualification_method, train_action_margins):
    if type(flag) is not bool or (
        flag
        and (
            joint_catalogue is not True
            or type(qualification_method) is not str
            or qualification_method not in ("pc", "neutral_cpc")
            or train_action_margins is not False
        )
    ):
        raise ValueError(
            "pc_update_diagnostics requires joint_catalogue, explicit pc/neutral_cpc, without action margins"
        )


def describe_pc(protocol, *, weighted=False):
    _validate_pc_weighted(weighted)
    protocol["protocol_id"] = protocol["protocol_id"].removesuffix("_v1") + (
        "_weighted_pc_update_diagnostics_v1" if weighted else "_pc_update_diagnostics_v1"
    )
    protocol["pc_update_diagnostics"] = dict(
        definition=PC_DEFINITION,
        source_updates=[0, 128],
        training_rows_per_readout=54,
        scope="owned TRAIN-only ungated PC/neutral snapshot; unchanged benchmark/probe/return learner and original budget; no final data",
        interpretation="gradient dot actual float32 delta negative denotes local descent; counterfactual TRAIN CE is not deployed fit, cause or benefit",
        costs="native readout counters separate from scores/updates; validation, copies and guards additionally consume original pilot wall time/RSS",
    )
    protocol["planned_work"]["pc_update_diagnostics"] = dict(
        catalogue_calls=2,
        projection_rows=108,
        projection_array_bytes=108 * 296 * 8,
        prior_forward_calls=4,
        prior_forward_rows=216,
        prior_backward_calls=2,
        prior_backward_rows=108,
        relaxation_forward_calls=6,
        relaxation_forward_rows=324,
        candidate_parameter_sets=2,
        native_train_calls=0,
        optimizer_steps=0,
        published_updates=0,
    )
    if weighted:
        protocol["pc_update_diagnostics"].update(
            definition=PC_WEIGHTED_DEFINITION,
            objective=PC_WEIGHTED_OBJECTIVE,
            class_weights="N/(present_classes*class_count) over the original six TRAIN sources; absent classes1; fixed per complete catalogue",
            scope="owned TRAIN-only weighted PC/neutral snapshot; exact detached native proposal; unchanged benchmark/probe/return learner and original budget; no final data",
        )
        protocol["planned_work"]["pc_update_diagnostics"].update(
            relaxation_forward_calls=0,
            relaxation_forward_rows=0,
            proposal_calls=2,
            proposal_rows=108,
            proposal_nominal_latent_row_iterations=216,
            fixed_state_energy_calls=2,
            fixed_state_energy_rows=108,
            weight_derivations=2,
            weight_rows=108,
            geometry_reductions=10,
        )


def empty_pc_work(*, weighted=False):
    _validate_pc_weighted(weighted)
    keys = PC_WEIGHTED_NATIVE_WORK_KEYS if weighted else PC_NATIVE_WORK_KEYS
    return dict.fromkeys(keys, 0) | dict(
        catalogue_calls_attempted=0,
        catalogue_calls_completed=0,
        seconds=0.0,
        input_array_equivalent_bytes=0,
        train_rows_admitted=0,
        geometry_reductions_completed=0,
        failure_work_may_be_unobserved=False,
    )


def sum_pc_work(records, *, weighted=False):
    result = empty_pc_work(weighted=weighted)
    for record in records:
        for key in result:
            if key == "failure_work_may_be_unobserved":
                result[key] |= record.get(key, False)
            else:
                result[key] += record.get(key, 0)
    return result


def pc_record_owners(record):
    """Bind the fixed scalar DTO containers, without recursive archive traversal."""
    diagnostic = record.get("diagnostic")
    owners = (record, record.get("episode_ids"), diagnostic)
    if diagnostic is None:
        return owners
    groups = diagnostic.get("per_parameter_geometry")
    result = owners + (
        diagnostic.get("source_episode_ids"),
        diagnostic.get("work"),
        diagnostic.get("relaxation"),
        *diagnostic.get("relaxation", ()),
        diagnostic.get("legacy_diagnostic"),
        diagnostic.get("gradient_delta_geometry"),
        groups,
        *(groups.values() if groups is not None else ()),
    )
    if "class_weights" in diagnostic or "proposal" in diagnostic:
        result += (diagnostic.get("class_weights"), diagnostic.get("proposal"))
    return result


def _weighted_pc_readout(state, episodes):
    # Why: preparation itself is paid even if source validation or the numerical
    # API refuses before it can return its own partial native-work report.
    rows = sum(len(episode.transitions) for episode in episodes)
    preparation = dict(
        weight_derivations_attempted=1,
        weight_derivations_completed=0,
        weight_rows_attempted=rows,
        weight_rows_completed=0,
    )
    weights = None
    try:
        weights = catalogue_class_weights(prepare_train_catalogue(episodes))
        preparation.update(weight_derivations_completed=1, weight_rows_completed=rows)
        payload = diagnose_pc_update(state, episodes, class_weights=weights)
    except Exception as error:
        if isinstance(error, GradientDiagnosticFailed):
            payload = deepcopy(error.report)
        else:
            payload = dict(
                status="incomplete",
                reason=f"{type(error).__name__}: {error}",
                definition=PC_WEIGHTED_DEFINITION,
                objective=PC_WEIGHTED_OBJECTIVE,
                rows=rows,
                source_episode_ids=[episode.episode_id for episode in episodes],
                class_weights=None if weights is None else list(weights),
                proposal=None,
                work=dict.fromkeys(PC_WEIGHTED_NATIVE_WORK_KEYS, 0),
            )
            payload["work"]["failure_work_may_be_unobserved"] = True
        payload["work"].update(preparation)
        raise GradientDiagnosticFailed(payload) from error
    payload = deepcopy(payload)
    payload["work"].update(preparation)
    return payload


def run_pc_update_readouts(report, state, sources, *, check_boundary, weighted=False):
    _validate_pc_weighted(weighted)
    _run_readouts(
        report,
        state,
        sources,
        check_boundary=check_boundary,
        validate=validate_prior_sources,
        read=_weighted_pc_readout if weighted else diagnose_pc_update,
        key="pc_update_diagnostics",
        native_keys=PC_WEIGHTED_NATIVE_WORK_KEYS if weighted else PC_NATIVE_WORK_KEYS,
        reductions=5,
        label="pc-update",
        groups=(("joint_train_catalogue", sources),),
        call_kind="catalogue",
        exact_evidence=True,
    )


def empty_work() -> dict[str, Any]:
    """Keep diagnostic exposure separate from benchmark/probe update counts."""
    return dict.fromkeys(NATIVE_WORK_KEYS, 0) | dict(
        pair_calls_attempted=0,
        pair_calls_completed=0,
        seconds=0.0,
        input_array_equivalent_bytes=0,
        train_rows_admitted=0,
        geometry_reductions_completed=0,
        failure_work_may_be_unobserved=False,
    )


def sum_work(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = empty_work()
    for record in records:
        for key in result:
            if key == "failure_work_may_be_unobserved":
                result[key] = result[key] or record.get(key, False)
            else:
                result[key] += record.get(key, 0)
    return result


def empty_prior_work() -> dict[str, Any]:
    """Account for canonical projection separately from the prior CE gradient."""
    return empty_work() | dict.fromkeys(PRIOR_NATIVE_WORK_KEYS, 0)


def sum_prior_work(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = empty_prior_work()
    for record in records:
        for key in result:
            if key == "failure_work_may_be_unobserved":
                result[key] = result[key] or record.get(key, False)
            else:
                result[key] += record.get(key, 0)
    return result


def validate_sources(sources: tuple[AttributionBatch, ...]) -> tuple[Any, ...]:
    """Admit exactly original148/K7 nine-row TRAIN A/B/C blue/red packets."""
    if type(sources) is not tuple or len(sources) != 6:
        raise ValueError("cue gradients require the six original TRAIN source packets")
    seen: set[str] = set()
    for index, packet in enumerate(sources):
        _validate_batch(packet, "train", seen)
        episode = shared._episode(packet)
        if (
            packet.features.shape != (9, 148)
            or packet.targets.shape != (9, 1)
            or packet.num_classes != 7
            or episode.role != "train"
            or episode.context != ("A", "B", "C_train")[index // 2]
            or episode.episode_id.rsplit("-", 1)[-1] != ("blue", "red")[index % 2]
        ):
            raise ValueError("cue gradients require fixed original148/K7 TRAIN cue order")
    return shared._signature(sources)


def validate_prior_sources(sources: tuple[AttributionBatch, ...]) -> tuple[Any, ...]:
    """Admit the six fixed TRAIN296 packets before any native diagnostic work."""
    if type(sources) is not tuple or len(sources) != 6:
        raise ValueError("prior gradients require the six original TRAIN source packets")
    seen: set[str] = set()
    for index, packet in enumerate(sources):
        # Why: bound dimensions before the full batch validator scans values.
        if (
            type(packet) is not AttributionBatch
            or type(packet.features) is not np.ndarray
            or packet.features.shape != (9, 296)
            or type(packet.targets) is not np.ndarray
            or packet.targets.shape != (9, 1)
        ):
            raise ValueError("prior gradients require fixed296/K7 nine-row TRAIN cue order")
        _validate_batch(packet, "train", seen)
        episode = shared._episode(packet)
        if (
            packet.num_classes != 7
            or episode.role != "train"
            or episode.context != ("A", "B", "C_train")[index // 2]
            or episode.episode_id.rsplit("-", 1)[-1] != ("blue", "red")[index % 2]
        ):
            raise ValueError("prior gradients require fixed296/K7 nine-row TRAIN cue order")
        for row in _feature_keys(packet.features, 296):
            canonicalize_public_history(row)
    return shared._signature(sources)


def run_gradient_readouts(
    report: dict[str, Any],
    state: Any,
    sources: tuple[AttributionBatch, ...],
    *,
    check_boundary: Callable[[], None],
) -> None:
    """Read original recurrent TRAIN pairs, preserving the established report schema."""
    _run_readouts(
        report,
        state,
        sources,
        check_boundary=check_boundary,
        validate=validate_sources,
        read=diagnose_episode_gradients,
        key="cue_gradient_geometry",
        native_keys=NATIVE_WORK_KEYS,
        reductions=6,
        label="cue-gradient",
    )


def run_prior_gradient_readouts(
    report: dict[str, Any],
    state: Any,
    sources: tuple[AttributionBatch, ...],
    *,
    check_boundary: Callable[[], None],
) -> None:
    """Read fixed TRAIN pairs on owned bound heads without updating the learner."""
    _run_readouts(
        report,
        state,
        sources,
        check_boundary=check_boundary,
        validate=validate_prior_sources,
        read=diagnose_prior_gradients,
        key="prior_gradient_geometry",
        native_keys=PRIOR_NATIVE_WORK_KEYS,
        reductions=5,
        label="prior-gradient",
    )


def _run_readouts(
    report: dict[str, Any],
    state: Any,
    sources: tuple[AttributionBatch, ...],
    *,
    check_boundary: Callable[[], None],
    validate: Callable[[tuple[AttributionBatch, ...]], tuple[Any, ...]],
    read: Callable[..., dict[str, Any]],
    key: str,
    native_keys: tuple[str, ...],
    reductions: int,
    label: str,
    groups: tuple[tuple[str, tuple[AttributionBatch, ...]], ...] | None = None,
    call_kind: str = "pair",
    exact_evidence: bool = False,
) -> None:
    """Retain every returned paid payload before a subsequent boundary proof.

    Why: immutable private copies preserve evidence after refusal without
    accepting a callback's replacement as a new successful diagnostic record.
    """
    signature = validate(sources)
    if not callable(check_boundary) or len(report["scores"]) != 6:
        raise ValueError("cue gradients require a guard and all six paid raw scores")
    expected_state = deepcopy(state)
    records: list[dict[str, Any]] = report[key]
    work_owner: dict[str, Any] = report["work"]
    raw_work = {name: deepcopy(value) for name, value in work_owner.items() if name != key}
    work = work_owner[key]
    owned_records: list[dict[str, Any]] = []
    record_owners: list[dict[str, Any]] = []
    payload_owners: list[tuple[Any, ...]] = []
    score_owner = report["scores"]
    score_payload = pickle.dumps(score_owner, protocol=5)
    expected_work = deepcopy(work)
    work_payload = deepcopy(work)
    diagnostic_state = None

    def evidence() -> None:
        if (
            report.get(key) is not records
            or report.get("scores") is not score_owner
            or pickle.dumps(score_owner, protocol=5) != score_payload
            or report.get("work") is not work_owner
            or {name: value for name, value in work_owner.items() if name != key} != raw_work
            or work_owner.get(key) is not work
            or (
                pickle.dumps(work, protocol=5) != pickle.dumps(work_payload, protocol=5)
                if exact_evidence
                else work != work_payload
            )
            or len(records) != len(owned_records)
            or any(
                record is not owner
                or (
                    pickle.dumps(record, protocol=5) != pickle.dumps(original, protocol=5)
                    if exact_evidence
                    else record != original
                )
                or (
                    exact_evidence
                    and any(
                        a is not b for a, b in zip(pc_record_owners(record), bound, strict=True)
                    )
                )
                for record, owner, original, bound in zip(
                    records, record_owners, owned_records, payload_owners, strict=True
                )
            )
            or not shared._check_signature(sources, signature)
            or not _same_state(expected_state, state)
            or (diagnostic_state is not None and not _same_state(expected_state, diagnostic_state))
        ):
            raise ValueError(
                f"{label} checkpoint, TRAIN packets or preceding paid evidence changed"
            )

    def guard() -> None:
        evidence()
        check_boundary()
        evidence()

    def merge(payload: dict[str, Any]) -> None:
        for name in native_keys:
            expected_work[name] += payload["work"][name]
        expected_work["failure_work_may_be_unobserved"] |= payload["work"][
            "failure_work_may_be_unobserved"
        ]
        if payload["status"] == "complete":
            expected_work["geometry_reductions_completed"] += reductions

    def publish_work() -> None:
        nonlocal work_payload
        work.clear()
        work.update(deepcopy(expected_work))
        work_payload = deepcopy(expected_work)

    try:
        guard()
        selected = (
            groups
            if groups is not None
            else tuple(
                (context, sources[2 * index : 2 * index + 2])
                for index, context in enumerate(("A", "B", "C_train"))
            )
        )
        for context, packets in selected:
            guard()
            pair = tuple(shared._episode(packet) for packet in packets)
            record = dict(
                context=context,
                episode_ids=[episode.episode_id for episode in pair],
                diagnostic=None,
            )
            records.append(record)
            record_owners.append(record)
            owned_records.append(deepcopy(record))
            payload_owners.append(pc_record_owners(record) if exact_evidence else ())
            expected_work[call_kind + "_calls_attempted"] += 1
            expected_work["train_rows_admitted"] += sum(len(packet.features) for packet in packets)
            expected_work["input_array_equivalent_bytes"] += sum(
                packet.features.nbytes + packet.targets.nbytes for packet in packets
            )
            publish_work()
            diagnostic_state = deepcopy(expected_state)
            before = monotonic()
            problem = None
            try:
                payload = read(diagnostic_state, pair)
                expected_work[call_kind + "_calls_completed"] += 1
            except GradientDiagnosticFailed as error:
                payload = error.report
                problem = error
            except Exception:
                expected_work["failure_work_may_be_unobserved"] = True
                raise
            finally:
                expected_work["seconds"] += monotonic() - before
            # Keep a detached result before any external caller guard can refuse.
            corrupt = None
            try:
                evidence()
            except Exception as error:
                corrupt = error
            original = deepcopy(owned_records[-1])
            original["diagnostic"] = deepcopy(payload)
            owned_records[-1] = original
            record.clear()
            record.update(deepcopy(original))
            payload_owners[-1] = pc_record_owners(record) if exact_evidence else ()
            merge(payload)
            publish_work()
            if corrupt is not None:
                raise corrupt
            guard()
            if problem is not None:
                raise problem
            if payload["status"] != "complete":
                raise ValueError(f"{label} diagnostic did not complete")
        guard()
    except Exception:
        expected_work["failure_work_may_be_unobserved"] = True
        report["scores"] = pickle.loads(score_payload)
        report[key] = deepcopy(owned_records)
        report["work"] = work_owner
        work_owner.clear()
        work_owner.update(deepcopy(raw_work))
        report["work"][key] = deepcopy(expected_work)
        raise
