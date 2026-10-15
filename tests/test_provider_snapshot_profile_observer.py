"""Observe actual snapshot returns around one owned profiling callback.

Inputs are two-entry native dictionaries; results are identity assertions and one
small work record per case. These tests do not alter the loader or run provider,
source-loading, model, or timing work. Why this: an ignored return is still
observable by an external hook, while automatic profiling events are suppressed
during that hook. Neither fact proves a universal event transcript.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import FrameType
from typing import Callable, Iterator

import provider_source_loader as loader

ProfileCallback = Callable[[FrameType, str, object], None]
Pairs = tuple[tuple[str, object], ...]


@dataclass
class ProfileRestoration:
    profile_set_calls: int = 0
    prior_profile_restored: bool = False
    initial_profile_restored: bool = False


@contextmanager
def observe_profile(observer: ProfileCallback, restoration: ProfileRestoration) -> Iterator[None]:
    initial_profile = sys.getprofile()
    assert initial_profile is None
    assert sys.gettrace() is None

    def prior_profile(frame: FrameType, event: str, argument: object) -> None:
        pass

    def set_profile(callback: ProfileCallback | None) -> None:
        restoration.profile_set_calls += 1
        sys.setprofile(callback)

    try:
        set_profile(prior_profile)
        try:
            set_profile(observer)
            yield
        finally:
            set_profile(prior_profile)
            restoration.prior_profile_restored = sys.getprofile() is prior_profile
            assert restoration.prior_profile_restored
    finally:
        set_profile(initial_profile)
        restoration.initial_profile_restored = sys.getprofile() is initial_profile
        assert restoration.initial_profile_restored
    assert restoration.profile_set_calls == 4


def make_namespace() -> tuple[dict[str, object], Pairs]:
    pairs = (("first", object()), ("second", object()))
    return dict(pairs), pairs


def snapshot_has_identity(snapshot: object, pairs: Pairs) -> bool:
    if type(snapshot) is not tuple or len(snapshot) != len(pairs):
        return False
    for item, (key, value) in zip(snapshot, pairs):
        if type(item) is not tuple or len(item) != 2:
            return False
        if item[0] is not key or item[1] is not value:
            return False
    return True


def namespace_has_identity(namespace: dict[str, object], pairs: Pairs) -> bool:
    return len(namespace) == len(pairs) and all(
        key is saved_key and value is saved_value
        for (key, value), (saved_key, saved_value) in zip(namespace.items(), pairs)
    )


def save_work(
    tmp_path: Path,
    test_name: str,
    context: str,
    dict_calls: int,
    sentinel_calls: int,
    sentinel_body_entries: int,
    snapshot_return_events: int,
    snapshot_identity_valid: bool | None,
    dictionary_return_is_original: bool,
    namespace_identity_valid: bool,
    restoration: ProfileRestoration,
) -> None:
    assert restoration.prior_profile_restored
    assert restoration.initial_profile_restored
    assert sys.getprofile() is None and sys.gettrace() is None
    record = dict(
        test_name=test_name,
        context=context,
        fixture_entries=2,
        dict_calls=dict_calls,
        sentinel_calls=sentinel_calls,
        sentinel_body_entries=sentinel_body_entries,
        snapshot_return_events=snapshot_return_events,
        snapshot_identity_valid=snapshot_identity_valid,
        dictionary_return_is_original=dictionary_return_is_original,
        namespace_identity_valid=namespace_identity_valid,
        prior_profile_restored=restoration.prior_profile_restored,
        initial_profile_restored=restoration.initial_profile_restored,
        profile_set_calls=restoration.profile_set_calls,
        runtime_implementation=sys.implementation.name,
        runtime_version=".".join(str(value) for value in sys.version_info[:3]),
    )
    (tmp_path / "snapshot-profile-work.json").write_text(json.dumps(record), encoding="utf8")


def test_discarded_snapshot_is_observable_outside_profile_callback(tmp_path: Path) -> None:
    namespace, pairs = make_namespace()
    original_snapshot_code = loader._snapshot.__code__
    observed_returns: list[object] = []
    restoration = ProfileRestoration()
    dict_calls = 0

    def observer(frame: FrameType, event: str, argument: object) -> None:
        if frame.f_code is original_snapshot_code and event == "return":
            observed_returns.append(argument)

    with observe_profile(observer, restoration):
        dict_calls += 1
        result = loader._dict(namespace)

    assert dict_calls == 1
    assert len(observed_returns) == 1
    snapshot_identity_valid = snapshot_has_identity(observed_returns[0], pairs)
    dictionary_return_is_original = result is namespace
    namespace_identity_valid = namespace_has_identity(namespace, pairs)
    assert snapshot_identity_valid
    assert dictionary_return_is_original
    assert namespace_identity_valid
    save_work(
        tmp_path,
        "test_discarded_snapshot_is_observable_outside_profile_callback",
        "outside-callback",
        dict_calls,
        0,
        0,
        len(observed_returns),
        snapshot_identity_valid,
        dictionary_return_is_original,
        namespace_identity_valid,
        restoration,
    )


def test_snapshot_events_are_suppressed_inside_profile_callback(tmp_path: Path) -> None:
    namespace, pairs = make_namespace()
    original_snapshot_code = loader._snapshot.__code__
    observed_returns: list[object] = []
    helper_results: list[object] = []
    restoration = ProfileRestoration()
    dict_calls = 0
    sentinel_calls = 0
    sentinel_body_entries = 0

    def sentinel() -> None:
        nonlocal sentinel_body_entries
        sentinel_body_entries += 1

    def observer(frame: FrameType, event: str, argument: object) -> None:
        nonlocal dict_calls
        if frame.f_code is original_snapshot_code and event == "return":
            observed_returns.append(argument)
        if frame.f_code is sentinel.__code__ and event == "call":
            dict_calls += 1
            helper_results.append(loader._dict(namespace))

    with observe_profile(observer, restoration):
        sentinel_calls += 1
        sentinel()

    assert dict_calls == sentinel_calls == sentinel_body_entries == 1
    assert len(helper_results) == 1 and observed_returns == []
    dictionary_return_is_original = helper_results[0] is namespace
    namespace_identity_valid = namespace_has_identity(namespace, pairs)
    assert dictionary_return_is_original
    assert namespace_identity_valid
    save_work(
        tmp_path,
        "test_snapshot_events_are_suppressed_inside_profile_callback",
        "inside-callback",
        dict_calls,
        sentinel_calls,
        sentinel_body_entries,
        len(observed_returns),
        None,
        dictionary_return_is_original,
        namespace_identity_valid,
        restoration,
    )
