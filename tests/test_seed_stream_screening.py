from dataclasses import asdict

import pytest

from src.core.seed_stream_screening import (
    EvidenceIdentity,
    confirmation_seed_streams,
    screen_seed_roles,
    validate_evidence_identity,
)


def test_should_include_every_fixed_source_split_exposure_model_and_local_rng_stream() -> None:
    assert [(row.name, row.value) for row in confirmation_seed_streams(7)] == [
        ("phase_a_source", 7),
        ("phase_b_source", 108),
        ("phase_a_roles", 24),
        ("phase_b_roles", 145),
        ("phase_b_exposure", 125),
        ("model_initialization", 1008),
        ("parent_selection", 5008),
        ("circadian_local_rng", 11009),
    ]


def test_should_screen_cross_stream_overlap_even_when_new_base_is_absent() -> None:
    report = screen_seed_roles((100,), {201: 3, 1102: 2}, {"symbolic": 1})
    assert [
        (row.stream.name, row.stream.value, row.prior_occurrences) for row in report.collisions
    ] == [
        ("phase_b_source", 201, 3),
    ]
    assert report.uncertainty_counts == (("symbolic", 1),)
    assert not report.fresh_roles_authorized


def test_should_never_admit_uncollided_seeds_from_declarations_or_empty_history() -> None:
    report = screen_seed_roles((100, 900), {}, {})
    assert report.proposed_base_seeds == (100, 900)
    assert [row.base_seed for row in report.streams] == [100] * 8 + [900] * 8
    assert not report.collisions and not report.fresh_roles_authorized
    assert "actual_execution_and_release_chronology_unverified" in report.blockers
    assert "complete_prior_usage_resource_gate_unaccepted" in report.blockers
    assert "prospective_role_and_full_execution_contract_unfinished" in report.blockers
    assert asdict(report)["fresh_roles_authorized"] is False


def test_should_report_cross_stream_reuse_between_distinct_proposed_bases() -> None:
    report = screen_seed_roles((100, 201), {}, {})
    collisions = report.proposal_collisions
    assert len(collisions) == 2
    assert (collisions[0].first.base_seed, collisions[0].first.name) == (100, "phase_b_source")
    assert (collisions[0].second.base_seed, collisions[0].second.name) == (201, "phase_a_source")
    assert (collisions[1].first.base_seed, collisions[1].first.name) == (100, "phase_b_exposure")
    assert (collisions[1].second.base_seed, collisions[1].second.name) == (201, "phase_a_roles")
    assert not report.fresh_roles_authorized


@pytest.mark.parametrize("seed", [True, -1, 7.0, "7"])
def test_should_reject_nonexact_or_negative_seed_identities(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        confirmation_seed_streams(seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("seeds", [[7], (7, 7), (True,)])
def test_should_reject_mutable_duplicate_or_boolean_proposals(seeds: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        screen_seed_roles(seeds, {}, {})  # type: ignore[arg-type]


@pytest.mark.parametrize("prior", [{True: 1}, {7: 0}, {7: True}, {-1: 3}])
def test_should_reject_invalid_prior_occurrence_counts(prior: dict[int, int]) -> None:
    with pytest.raises(ValueError, match="occurrence"):
        screen_seed_roles((), prior, {})


@pytest.mark.parametrize(
    "identity",
    [
        EvidenceIdentity(True, "a" * 64),
        EvidenceIdentity(-1, "a" * 64),
        EvidenceIdentity(1, "A" * 64),
        EvidenceIdentity(1, "a" * 63),
    ],
)
def test_should_reject_malformed_whole_evidence_identities(identity: EvidenceIdentity) -> None:
    with pytest.raises(ValueError, match="identity"):
        validate_evidence_identity(identity)
