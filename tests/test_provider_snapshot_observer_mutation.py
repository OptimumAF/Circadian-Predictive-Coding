"""Check freshness after an actual snapshot-return observer changes a value.

Inputs are owned two-entry native dictionaries and saved ordered object pairs.
Results are actual tuple identity checks, freshness outcomes, and a small JSON
work record. Why this: a snapshot retains old references after a cooperative
observer changes the live dictionary, so the later freshness check must refuse.
These tests do not prove atomicity, concurrency safety, universal hook behavior,
or provider/source birth; they do not modify or optimize the loader.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from types import FrameType

import provider_source_loader as loader
from test_provider_snapshot_profile_observer import (
    ProfileRestoration,
    make_namespace,
    namespace_has_identity,
    observe_profile,
    snapshot_has_identity,
)

Pairs = tuple[tuple[str, object], ...]


@dataclass
class Observations:
    snapshot_returns: list[object] = field(default_factory=list)
    dictionary_result: object = None
    freshness_result: object = field(default_factory=object)
    freshness_error: Exception | None = None
    mutations: int = 0
    dict_calls: int = 0
    unchanged_calls: int = 0
    restoration: ProfileRestoration = field(default_factory=ProfileRestoration)


def run_checks(
    namespace: dict[str, object], pairs: Pairs, replacement: object | None
) -> Observations:
    observations = Observations()
    original_snapshot_code = loader._snapshot.__code__

    def observer(frame: FrameType, event: str, argument: object) -> None:
        if frame.f_code is original_snapshot_code and event == "return":
            observations.snapshot_returns.append(argument)
            if replacement is not None and len(observations.snapshot_returns) == 1:
                namespace[pairs[0][0]] = replacement
                observations.mutations += 1

    with observe_profile(observer, observations.restoration):
        observations.dict_calls += 1
        observations.dictionary_result = loader._dict(namespace)
        observations.unchanged_calls += 1
        try:
            observations.freshness_result = loader._unchanged(namespace, pairs)
        except Exception as error:
            observations.freshness_error = error

    return observations


def check_results(
    namespace: dict[str, object],
    original_pairs: Pairs,
    current_pairs: Pairs,
    observations: Observations,
    expect_refusal: bool,
) -> tuple[bool, bool, bool, bool]:
    assert observations.dict_calls == observations.unchanged_calls == 1
    assert observations.mutations == int(expect_refusal)
    assert len(observations.snapshot_returns) == 2
    first_valid = snapshot_has_identity(observations.snapshot_returns[0], original_pairs)
    second_valid = snapshot_has_identity(observations.snapshot_returns[1], current_pairs)
    dictionary_original = observations.dictionary_result is namespace
    namespace_valid = namespace_has_identity(namespace, current_pairs)
    assert first_valid and second_valid and dictionary_original and namespace_valid
    if expect_refusal:
        assert type(observations.freshness_error) is ValueError
        assert str(observations.freshness_error) == "invalid source witness"
    else:
        assert observations.freshness_error is None
        assert observations.freshness_result is None
    return first_valid, second_valid, dictionary_original, namespace_valid


def save_work(
    tmp_path: Path,
    test_name: str,
    observations: Observations,
    identity_checks: tuple[bool, bool, bool, bool],
) -> None:
    restoration = observations.restoration
    assert restoration.prior_profile_restored and restoration.initial_profile_restored
    assert restoration.profile_set_calls == 4
    assert sys.getprofile() is None and sys.gettrace() is None
    first_valid, second_valid, dictionary_original, namespace_valid = identity_checks
    error = observations.freshness_error
    record = dict(
        test_name=test_name,
        fixture_entries=2,
        dict_calls=observations.dict_calls,
        unchanged_calls=observations.unchanged_calls,
        mutations=observations.mutations,
        snapshot_return_events=len(observations.snapshot_returns),
        first_snapshot_identity_valid=first_valid,
        second_snapshot_identity_valid=second_valid,
        dictionary_return_is_original=dictionary_original,
        namespace_identity_valid=namespace_valid,
        freshness_refused=error is not None,
        freshness_error=None if error is None else str(error),
        prior_profile_restored=restoration.prior_profile_restored,
        initial_profile_restored=restoration.initial_profile_restored,
        profile_set_calls=restoration.profile_set_calls,
        runtime_implementation=sys.implementation.name,
        runtime_version=".".join(str(value) for value in sys.version_info[:3]),
    )
    (tmp_path / "snapshot-profile-work.json").write_text(json.dumps(record), encoding="utf8")


def test_snapshot_return_observer_mutation_is_refused_by_freshness_check(tmp_path: Path) -> None:
    namespace, original_pairs = make_namespace()
    replacement = object()
    assert replacement is not original_pairs[0][1]
    current_pairs = ((original_pairs[0][0], replacement), original_pairs[1])

    observations = run_checks(namespace, original_pairs, replacement)

    identity_checks = check_results(
        namespace, original_pairs, current_pairs, observations, expect_refusal=True
    )
    save_work(
        tmp_path,
        "test_snapshot_return_observer_mutation_is_refused_by_freshness_check",
        observations,
        identity_checks,
    )


def test_snapshot_return_observer_preserves_unchanged_namespace(tmp_path: Path) -> None:
    namespace, original_pairs = make_namespace()

    observations = run_checks(namespace, original_pairs, replacement=None)

    identity_checks = check_results(
        namespace, original_pairs, original_pairs, observations, expect_refusal=False
    )
    save_work(
        tmp_path,
        "test_snapshot_return_observer_preserves_unchanged_namespace",
        observations,
        identity_checks,
    )
