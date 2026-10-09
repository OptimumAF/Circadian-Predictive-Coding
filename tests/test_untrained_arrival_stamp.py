"""S1: twelve bounded arrival-integrity cases; no native work or admission."""

from dataclasses import replace

import numpy as np
import pytest

from src.app.untrained_inbox_erasure import _arrival_stamp
from src.core.checkpoint_content import CheckpointContentLimits
from src.core.data_lifecycle import DataConsent, DataProvenance, LifecycleDeclaration
from src.core.experience import Experience, ExperiencePermissions, LabelArrival


def _arrivals(form="pair"):
    source = (
        None
        if form == "label"
        else Experience(
            "sample",
            "episode",
            1,
            "actor",
            np.array([1.0, 2.0], dtype=np.float64),
            "train",
            ExperiencePermissions(True, True),
        )
    )
    label = (
        None
        if form == "source"
        else LabelArrival(
            "event",
            "sample",
            "episode",
            2,
            "actor",
            np.array([3.0, 4.0], dtype=np.float64),
        )
    )
    declaration = LifecycleDeclaration(
        ("episode", "sample"),
        DataProvenance("local", "person", True, False),
        DataConsent(True, True),
        "replay",
    )
    return source, label, declaration, CheckpointContentLimits(256, 8192, 64, 16)


@pytest.mark.parametrize("form", ["source", "label", "pair"])
def test_should_keep_same_original_arrival_stamp_under_retained_scalar_wrapper_churn(form):
    source, label, declaration, limits = _arrivals(form)
    original = _arrival_stamp(source, label, declaration, limits)
    retained: list[tuple[int, ...]] = []

    for tick in range(32):
        # Why: retain fresh scalar tuples between calls so unchanged arrivals
        # cannot depend on reuse of temporary proof-wrapper addresses.
        retained.extend(
            (
                tuple([tick] * 8),
                tuple([tick]),
                tuple([tick] * 2),
                tuple([tick] * 7),
            )
        )
        assert _arrival_stamp(source, label, declaration, limits) == original

    assert len(retained) == 128
    assert len({id(wrapper) for wrapper in retained}) == 128


@pytest.mark.parametrize("replacement", ["source", "label", "declaration"])
def test_should_change_stamp_for_same_content_original_object_replacement(replacement):
    source, label, declaration, limits = _arrivals()
    assert source is not None and label is not None
    original = _arrival_stamp(source, label, declaration, limits)

    if replacement == "source":
        clone = replace(source)
        assert clone is not source and clone.features is source.features
        current = _arrival_stamp(clone, label, declaration, limits)
    elif replacement == "label":
        clone_label = replace(label)
        assert clone_label is not label and clone_label.targets is label.targets
        current = _arrival_stamp(source, clone_label, declaration, limits)
    else:
        clone_declaration = replace(declaration)
        assert clone_declaration is not declaration
        assert clone_declaration.provenance is declaration.provenance
        assert clone_declaration.consent is declaration.consent
        current = _arrival_stamp(source, label, clone_declaration, limits)

    assert current != original


@pytest.mark.parametrize("mutation", ["features", "targets", "observed_at", "arrived_at"])
def test_should_change_stamp_for_valid_original_content_or_time_mutation(mutation):
    source, label, declaration, limits = _arrivals()
    assert source is not None and label is not None
    original = _arrival_stamp(source, label, declaration, limits)

    if mutation == "features":
        source.features[0] = 1.25
    elif mutation == "targets":
        label.targets[0] = 3.25
    elif mutation == "observed_at":
        object.__setattr__(source, "observed_at", 0)
    else:
        object.__setattr__(label, "arrived_at", 3)
    Experience.__post_init__(source)
    LabelArrival.__post_init__(label)

    assert _arrival_stamp(source, label, declaration, limits) != original


def test_should_refuse_original_source_replay_permission_revocation():
    source, label, declaration, limits = _arrivals()
    assert source is not None
    object.__setattr__(source.permissions, "replay", False)

    with pytest.raises(ValueError, match="original train/replay permission"):
        _arrival_stamp(source, label, declaration, limits)


def test_should_refuse_original_declaration_replay_consent_revocation():
    source, label, declaration, limits = _arrivals()
    object.__setattr__(declaration.consent, "replay", False)

    with pytest.raises(ValueError, match="original training/replay declaration"):
        _arrival_stamp(source, label, declaration, limits)
