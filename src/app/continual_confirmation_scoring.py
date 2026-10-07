"""Compose globally checked final views and the complete fixed endpoint matrix.

Inputs are reproduced training, a fixed scored manifest and injected release/
evaluation/checkpoint ports. Outputs preserve every endpoint/cell/failure and
locally accounted costs. No source construction, IO, file provenance, external
resource observation, analysis selection or training belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from src.app.continual_confirmation_analysis import FinalRoleIdentity, OutcomeCell
from src.app.continual_confirmation_manifest import ConfirmationManifest
from src.app.continual_confirmation_scoring_manifest import (
    ConfirmationScoringManifest,
    scoring_manifest_digest,
    validate_scoring_manifest,
)
from src.app.continual_confirmation_scoring_state import (
    TrainingStateProof,
    verify_scoring_training_state,
)
from src.app.continual_confirmation_state import HeldSeed, RoleFacts
from src.app.continual_confirmation_training import TrainedConfirmation
from src.app.continual_replay_factor_pilot import Model
from src.core.confirmation_final_roles import (
    EndpointResult,
    FinalRole,
    FinalRoleFacts,
    capture_final_role,
    validate_endpoint_result,
)
from src.core.continual_metrics import TwoTaskAccuracy


FinalReleaser = Callable[[HeldSeed, str], FinalRole]
FinalEvaluator = Callable[[Model, FinalRole], EndpointResult]
BoundaryCheckpoint = Callable[[str], None]


@dataclass(frozen=True)
class ReleasedSeedFacts:
    family: str
    seed: int
    a: FinalRoleFacts
    b: FinalRoleFacts


@dataclass(frozen=True)
class EndpointEvaluation:
    family: str
    seed: int
    arm: str
    endpoint: str
    checkpoint: str
    phase: str
    role_sha256: str
    example_count: int
    result: EndpointResult


@dataclass(frozen=True)
class EvaluationTotals:
    release_calls: int
    final_role_views: int
    endpoint_calls: int
    example_count: int
    successful_endpoints: int
    failed_endpoints: int
    successful_cells: int
    failed_cells: int


@dataclass(frozen=True)
class ScoredConfirmation:
    training_before_release: TrainingStateProof
    training_after_release: TrainingStateProof
    training_after_evaluation: TrainingStateProof
    final_roles: tuple[ReleasedSeedFacts, ...]
    evaluations: tuple[EndpointEvaluation, ...]
    cells: tuple[OutcomeCell, ...]
    totals: EvaluationTotals
    scoring_manifest_sha256: str | None = None
    validation_scope: str = "app_training_state_final_role_content_and_endpoint_records_only"
    external_execution_verified: bool = False
    outer_selection_scored: bool = False


@dataclass(frozen=True)
class _ReleasedSeed:
    held: HeldSeed
    a: FinalRole
    b: FinalRole
    facts: ReleasedSeedFacts


def _capture_arrived_final(role: FinalRole, expected: RoleFacts) -> FinalRoleFacts:
    facts = capture_final_role(role, expected.expected_final_count)
    if (facts.phase, facts.seed, facts.sample_ids) != (
        expected.phase,
        expected.seed,
        expected.sample_ids["final_test"],
    ):
        raise ValueError("confirmation released final role differs from arrived identity")
    return facts


def _release_all(
    trained: TrainedConfirmation, release: FinalReleaser, checkpoint: BoundaryCheckpoint
) -> tuple[_ReleasedSeed, ...]:
    released = []
    for item in trained.held:
        views = []
        facts = []
        for phase, expected in zip(("a", "b"), item.facts.roles, strict=True):
            checkpoint("before_role_release")
            role = release(item, phase)
            views.append(role)
            facts.append(_capture_arrived_final(role, expected))
            checkpoint("after_role_release")
        row = ReleasedSeedFacts(item.facts.family, item.facts.seed, facts[0], facts[1])
        released.append(_ReleasedSeed(item, views[0], views[1], row))
    return tuple(released)


def _require_final_views(rows: tuple[_ReleasedSeed, ...]) -> None:
    shared: dict[tuple[str, int], FinalRoleFacts] = {}
    for row in rows:
        for role, before, expected in zip(
            (row.a, row.b), (row.facts.a, row.facts.b), row.held.facts.roles, strict=True
        ):
            after = _capture_arrived_final(role, expected)
            if after != before:
                raise ValueError("confirmation released final role content changed after capture")
            key = (after.phase, after.seed)
            if key in shared and shared[key] != after:
                raise ValueError("confirmation shared source final role signatures differ")
            shared[key] = after


def _evaluate_cell(
    row: _ReleasedSeed, arm: str, evaluate: FinalEvaluator, checkpoint: BoundaryCheckpoint
) -> tuple[OutcomeCell, tuple[EndpointEvaluation, ...]]:
    requests = (
        ("a_after_a", "a", row.held.models_after_a[arm], row.a, row.facts.a),
        ("a_after_b", "b", row.held.models_after_b[arm], row.a, row.facts.a),
        ("b_after_b", "b", row.held.models_after_b[arm], row.b, row.facts.b),
    )
    records = []
    for endpoint, held_phase, model, role, facts in requests:
        checkpoint("before_endpoint")
        result = evaluate(model, role)
        validate_endpoint_result(result, facts.count)
        records.append(
            EndpointEvaluation(
                row.facts.family,
                row.facts.seed,
                arm,
                endpoint,
                held_phase,
                facts.phase,
                facts.sha256,
                facts.count,
                result,
            )
        )
        checkpoint("after_endpoint")
    endpoints = tuple(records)
    return _cell_from_records(row, arm, endpoints), endpoints


def _cell_from_records(
    row: _ReleasedSeed, arm: str, records: tuple[EndpointEvaluation, ...]
) -> OutcomeCell:
    expected = (
        ("a_after_a", "a", row.facts.a),
        ("a_after_b", "b", row.facts.a),
        ("b_after_b", "b", row.facts.b),
    )
    if len(records) != 3:
        raise ValueError("confirmation endpoint record count differs")
    values = []
    failures = []
    for record, (endpoint, held_phase, facts) in zip(records, expected, strict=True):
        _require_endpoint_identity(record, row, arm, endpoint, held_phase, facts)
        values.append(validate_endpoint_result(record.result, facts.count))
        if record.result.failure is not None:
            failures.append(
                f"{endpoint}:{record.result.failure.code}:{record.result.failure.error_type or 'none'}"
            )
    identity = FinalRoleIdentity(
        row.facts.a.sha256, row.facts.b.sha256, row.facts.a.count, row.facts.b.count
    )
    accuracy = None
    if not failures:
        a_after_a, a_after_b, b_after_b = values
        assert a_after_a is not None and a_after_b is not None and b_after_b is not None
        accuracy = TwoTaskAccuracy(a_after_a, a_after_b, b_after_b)
    return OutcomeCell(
        row.facts.family,
        row.facts.seed,
        arm,
        identity,
        accuracy,
        ";".join(failures) if failures else None,
    )


def _require_endpoint_identity(
    record: EndpointEvaluation,
    row: _ReleasedSeed,
    arm: str,
    endpoint: str,
    held_phase: str,
    facts: FinalRoleFacts,
) -> None:
    if (
        type(record) is not EndpointEvaluation
        or type(record.seed) is not int
        or type(record.example_count) is not int
    ):
        raise ValueError("confirmation endpoint record type differs")
    actual = (
        record.family,
        record.seed,
        record.arm,
        record.endpoint,
        record.checkpoint,
        record.phase,
        record.role_sha256,
        record.example_count,
    )
    expected = (
        row.facts.family,
        row.facts.seed,
        arm,
        endpoint,
        held_phase,
        facts.phase,
        facts.sha256,
        facts.count,
    )
    if actual != expected:
        raise ValueError("confirmation endpoint record identity differs")


def _require_outcome_links(
    rows: tuple[_ReleasedSeed, ...],
    manifest: ConfirmationManifest,
    records: tuple[EndpointEvaluation, ...],
    cells: tuple[OutcomeCell, ...],
) -> None:
    families = {family.name: family for family in manifest.families}
    expected_cells = sum(len(family.seeds) * len(family.arms) for family in manifest.families)
    if len(cells) != expected_cells or len(records) != 3 * expected_cells:
        raise ValueError("confirmation complete endpoint/cell scope differs")
    index = 0
    for row in rows:
        for arm in families[row.facts.family].arms:
            expected = _cell_from_records(row, arm, records[3 * index : 3 * index + 3])
            if cells[index] != expected:
                raise ValueError("confirmation cell changed from retained endpoint records")
            index += 1


def _totals(
    rows: tuple[_ReleasedSeed, ...],
    evaluations: tuple[EndpointEvaluation, ...],
    cells: tuple[OutcomeCell, ...],
) -> EvaluationTotals:
    failed_endpoints = sum(item.result.failure is not None for item in evaluations)
    failed_cells = sum(item.failure is not None for item in cells)
    return EvaluationTotals(
        2 * len(rows),
        2 * len(rows),
        len(evaluations),
        sum(item.example_count for item in evaluations),
        len(evaluations) - failed_endpoints,
        failed_endpoints,
        len(cells) - failed_cells,
        failed_cells,
    )


def _evaluate_trained(
    trained: TrainedConfirmation,
    manifest: ConfirmationManifest,
    verify_state: Callable[[], TrainingStateProof],
    release: FinalReleaser,
    evaluate: FinalEvaluator,
    checkpoint: BoundaryCheckpoint,
) -> ScoredConfirmation:
    """Private development seam; public API uses the fixed whole production gate."""
    checkpoint("before_final_release")
    before = verify_state()
    rows = _release_all(trained, release, checkpoint)
    _require_final_views(rows)
    checkpoint("before_final_evaluation")
    after_release = verify_state()
    _require_final_views(rows)
    cells: list[OutcomeCell] = []
    evaluations: list[EndpointEvaluation] = []
    families = {family.name: family for family in manifest.families}
    for row in rows:
        for arm in families[row.facts.family].arms:
            cell, records = _evaluate_cell(row, arm, evaluate, checkpoint)
            cells.append(cell)
            evaluations.extend(records)
    checkpoint("after_final_evaluation")
    after_evaluation = verify_state()
    _require_final_views(rows)
    endpoints, outcomes = tuple(evaluations), tuple(cells)
    _require_outcome_links(rows, manifest, endpoints, outcomes)
    return ScoredConfirmation(
        before,
        after_release,
        after_evaluation,
        tuple(row.facts for row in rows),
        endpoints,
        outcomes,
        _totals(rows, endpoints, outcomes),
    )


def evaluate_confirmation(
    trained: TrainedConfirmation,
    manifest: ConfirmationScoringManifest,
    release: FinalReleaser,
    evaluate: FinalEvaluator,
    checkpoint: BoundaryCheckpoint,
) -> ScoredConfirmation:
    """Compose only the complete bound scope; outer provenance remains required."""
    validate_scoring_manifest(manifest)
    result = _evaluate_trained(
        trained,
        manifest.train_manifest,
        lambda: verify_scoring_training_state(trained, manifest),
        release,
        evaluate,
        checkpoint,
    )
    return replace(result, scoring_manifest_sha256=scoring_manifest_digest(manifest))
