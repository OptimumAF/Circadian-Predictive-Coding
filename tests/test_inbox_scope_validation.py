"""Scope validation must preserve the original metadata and callback contract."""

from dataclasses import replace

import pytest

from src.app.experience_inbox import ExperienceInbox
from src.core.experience import (
    Experience,
    ExperiencePermissions,
    LabelArrival,
    require_training_role,
)


def original_scope(inbox):
    """Frozen pre-optimization body; constructor failures are checked separately."""
    if (
        require_training_role(inbox._training_role) != inbox._original_training_role
        or inbox._training_role is not inbox._original_training_role
    ):
        raise ValueError("inbox requires its original training role")
    for source in inbox._experiences.values():
        if type(source) is not Experience:
            raise ValueError("inbox source must remain an exact Experience")
        replace(source)
        replace(source.permissions)
        if source.role != inbox._training_role or not source.permissions.training:
            raise ValueError("inbox source differs from its original training role")
    for label in inbox._labels.values():
        if type(label) is not LabelArrival:
            raise ValueError("inbox label must remain an exact LabelArrival")
        replace(label)
        if label.role != inbox._training_role:
            raise ValueError("inbox label differs from its original training role")


class OpaquePayload:
    def __deepcopy__(self, memo):
        raise AssertionError("scope validation must not copy payloads")

    def __eq__(self, other):
        raise AssertionError("scope validation must not compare payloads")


def make_inbox(role="train"):
    source = Experience(
        "sample",
        "episode",
        0,
        "actor",
        OpaquePayload(),
        role,
        ExperiencePermissions(training=True),
        ("left", "right"),
        "left",
        1.0,
    )
    label = LabelArrival("event", "sample", "episode", 1, "actor", OpaquePayload(), role)
    inbox = object.__new__(ExperienceInbox)
    inbox._training_role = inbox._original_training_role = role
    inbox._experiences = {source.key: source}
    inbox._labels = {label.key: label}
    return inbox, source, label


def outcome(check, inbox):
    try:
        check(inbox)
    except Exception as error:
        return type(error), str(error)
    return None


@pytest.mark.parametrize("role", ["train", "probe_train"])
def test_should_validate_again_without_copying_or_mutating_authority(role):
    inbox, source, label = make_inbox(role)
    source_fields, label_fields = dict(vars(source)), dict(vars(label))
    permissions = source.permissions
    for _ in range(3):
        inbox._require_training_scope()
        assert inbox._experiences[source.key] is source
        assert inbox._labels[label.key] is label
        assert source.permissions is permissions
        assert all(vars(source)[key] is value for key, value in source_fields.items())
        assert all(vars(label)[key] is value for key, value in label_fields.items())


@pytest.mark.parametrize(
    ("owner", "field", "value", "message"),
    [
        ("source", "sample_id", " ", "sample_id must be a nonempty identifier"),
        ("source", "episode_id", 2, "episode_id must be a nonempty identifier"),
        ("source", "model_version", " actor", "model_version must be a nonempty identifier"),
        ("source", "observed_at", -1, "observed_at must be a nonnegative integer"),
        ("source", "observed_at", True, "observed_at must be a nonnegative integer"),
        ("source", "role", "unknown", "experience role must name"),
        ("source", "role", "probe_train", "inbox source differs from its original training role"),
        ("source", "permissions", None, "experience requires typed permissions"),
        ("source", "candidate_ids", ["left"], "candidate_ids must be an immutable tuple"),
        ("source", "candidate_ids", ("left", "left"), "candidate_ids must be unique"),
        ("source", "candidate_ids", ("left", " bad"), "candidate_id must be a nonempty identifier"),
        ("source", "action_id", "absent", "action_id must identify a declared candidate"),
        ("source", "reward", float("nan"), "reward must be finite and numeric"),
        ("source", "reward", float("inf"), "reward must be finite and numeric"),
        ("source", "reward", True, "reward must be finite and numeric"),
        ("label", "event_id", "", "event_id must be a nonempty identifier"),
        ("label", "sample_id", "sample ", "sample_id must be a nonempty identifier"),
        ("label", "arrived_at", -1, "arrived_at must be a nonnegative integer"),
        ("label", "role", "final_test", "inbox label differs from its original training role"),
    ],
)
def test_should_refuse_later_metadata_corruption_like_original(owner, field, value, message):
    outcomes = []
    for check in (original_scope, ExperienceInbox._require_training_scope):
        inbox, source, label = make_inbox()
        assert outcome(check, inbox) is None
        record = source if owner == "source" else label
        object.__setattr__(record, field, value)
        observed = outcome(check, inbox)
        assert observed[0] is ValueError
        assert observed[1].startswith(message)
        assert outcome(check, inbox) == observed
        assert getattr(record, field) is value
        outcomes.append(observed)
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("role", ["inner_guard", "outer_selection", "final_test"])
def test_should_refuse_heldout_sources_even_when_permissions_are_valid(role):
    for check in (original_scope, ExperienceInbox._require_training_scope):
        inbox, source, _ = make_inbox()
        object.__setattr__(source, "role", role)
        object.__setattr__(source, "permissions", ExperiencePermissions(evaluation=True))
        assert outcome(check, inbox) == (
            ValueError,
            "inbox source differs from its original training role",
        )


@pytest.mark.parametrize(
    ("role", "training", "replay", "evaluation", "message"),
    [
        ("train", False, False, False, "inbox source differs from its original training role"),
        (
            "probe_train",
            True,
            True,
            False,
            "probe training cannot grant replay or evaluation permission",
        ),
        (
            "probe_train",
            True,
            False,
            True,
            "probe training cannot grant replay or evaluation permission",
        ),
        (
            "final_test",
            True,
            False,
            False,
            "held-out roles cannot grant training or replay permission",
        ),
    ],
)
def test_should_refuse_current_permission_changes(role, training, replay, evaluation, message):
    for check in (original_scope, ExperienceInbox._require_training_scope):
        inbox, source, _ = make_inbox("probe_train" if role == "probe_train" else "train")
        object.__setattr__(source, "role", role)
        object.__setattr__(
            source, "permissions", ExperiencePermissions(training, replay, evaluation)
        )
        assert outcome(check, inbox) == (ValueError, message)


class PermissionCallback:
    def __init__(self, source, trace, effect):
        self.source, self.trace, self.effect = source, trace, effect

    def __bool__(self):
        self.trace.append("permission bool")
        if self.effect == "raise":
            raise RuntimeError("permission callback stopped validation")
        object.__setattr__(self.source, "sample_id", "")
        if self.effect == "replace permissions":
            object.__setattr__(self.source, "permissions", ExperiencePermissions(training=True))
        return False


@pytest.mark.parametrize("effect", ["mutate source", "replace permissions", "raise"])
def test_should_keep_original_callback_order_and_snapshot_semantics(effect):
    observed = []
    for check in (original_scope, ExperienceInbox._require_training_scope):
        inbox, source, _ = make_inbox("probe_train")
        trace = []
        permissions = source.permissions
        callback = PermissionCallback(source, trace, effect)
        object.__setattr__(permissions, "replay", callback)
        first, second = outcome(check, inbox), outcome(check, inbox)
        observed.append(
            (first, second, tuple(trace), source.sample_id, source.permissions is permissions)
        )
    assert observed[0] == observed[1]
    first, second, trace, sample_id, same_permissions = observed[1]
    if effect == "raise":
        assert first == second == (RuntimeError, "permission callback stopped validation")
        assert trace == ("permission bool", "permission bool")
        assert sample_id == "sample" and same_permissions
    else:
        assert trace == ("permission bool",)
        assert second == (
            ValueError,
            "sample_id must be a nonempty identifier without surrounding whitespace",
        )
        assert sample_id == ""
        if effect == "replace permissions":
            assert first is None and not same_permissions
        else:
            assert first == (ValueError, "experience permissions must be exact boolean flags")
            assert same_permissions


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"sample_id": "", "observed_at": -1},
            "sample_id must be a nonempty identifier without surrounding whitespace",
        ),
        (
            {"observed_at": -1, "role": "bad"},
            "observed_at must be a nonnegative integer logical tick",
        ),
        (
            {"role": "bad", "permissions": None},
            "experience role must name train, probe_train, inner_guard, outer_selection or final_test",
        ),
        ({"permissions": None, "candidate_ids": []}, "experience requires typed permissions"),
        (
            {"candidate_ids": ("left", "left"), "action_id": "missing"},
            "candidate_ids must be unique",
        ),
        (
            {"action_id": "missing", "reward": float("inf")},
            "action_id must identify a declared candidate",
        ),
    ],
)
def test_should_preserve_constructor_validation_order(changes, message):
    _, source, _ = make_inbox()
    with pytest.raises(ValueError, match="^" + message + "$"):
        Experience(**(vars(source) | changes))


def test_should_preserve_label_and_permission_constructor_errors():
    _, _, label = make_inbox()
    with pytest.raises(
        ValueError, match="^event_id must be a nonempty identifier without surrounding whitespace$"
    ):
        LabelArrival(**(vars(label) | {"event_id": "", "arrived_at": -1, "role": "bad"}))
    with pytest.raises(ValueError, match="^experience permissions must be exact boolean flags$"):
        ExperiencePermissions(training=1)


class MetadataString(str):
    def strip(self, *args, **kwargs):
        raise AssertionError("exact metadata type refusal must precede opaque methods")


@pytest.mark.parametrize(
    "owner,field", [("source", "sample_id"), ("source", "role"), ("label", "role")]
)
def test_should_refuse_string_subclasses_without_opaque_callbacks(owner, field):
    observations = []
    for check in (original_scope, ExperienceInbox._require_training_scope):
        inbox, source, label = make_inbox()
        record = source if owner == "source" else label
        object.__setattr__(record, field, MetadataString(getattr(record, field)))
        observations.append(outcome(check, inbox))
    assert observations[0] == observations[1]
    assert observations[1][0] is ValueError


@pytest.mark.parametrize("owner", ["source", "label"])
def test_should_refuse_replaced_record_subclasses(owner):
    inbox, source, label = make_inbox()
    original = source if owner == "source" else label
    record_type = type("RecordSubclass", (type(original),), {})
    replacement = record_type(**vars(original))
    collection = inbox._experiences if owner == "source" else inbox._labels
    collection[original.key] = replacement
    message = (
        "inbox source must remain an exact Experience"
        if owner == "source"
        else "inbox label must remain an exact LabelArrival"
    )
    assert (
        outcome(original_scope, inbox)
        == outcome(ExperienceInbox._require_training_scope, inbox)
        == (ValueError, message)
    )


def test_should_refuse_source_before_label_and_bad_role_before_records():
    for check in (original_scope, ExperienceInbox._require_training_scope):
        inbox, source, label = make_inbox()
        object.__setattr__(source, "sample_id", "")
        object.__setattr__(label, "event_id", "")
        assert outcome(check, inbox) == (
            ValueError,
            "sample_id must be a nonempty identifier without surrounding whitespace",
        )
        inbox._training_role = "final_test"
        assert outcome(check, inbox) == (
            ValueError,
            "training role must be exactly train or probe_train",
        )


@pytest.mark.parametrize(
    "owner,removed",
    [
        ("source", ("features",)),
        ("label", ("targets",)),
        ("source", ("observed_at", "model_version")),
        ("label", ("arrived_at", "model_version")),
        ("source", ("permissions", "candidate_ids")),
        ("permissions", ("training", "replay")),
        ("source", ("candidate_ids", "action_id", "reward")),
        ("label", ("role",)),
    ],
)
def test_should_preserve_missing_payload_metadata_and_default_field_capture(owner, removed):
    results = []
    for check in (original_scope, ExperienceInbox._require_training_scope):
        inbox, source, label = make_inbox()
        record = {"source": source, "label": label, "permissions": source.permissions}[owner]
        for name in removed:
            object.__delattr__(record, name)
        first = outcome(check, inbox)
        assert outcome(check, inbox) == first
        results.append(first)
    assert results[0] == results[1]
    if owner in ("source", "label") and removed[0] not in ("candidate_ids", "role"):
        assert results[1][0] is AttributeError


@pytest.mark.parametrize("owner", ["source", "permissions", "label"])
def test_should_add_no_callbacks_for_opaque_extra_attribute_keys(owner):
    results = []
    for check in (original_scope, ExperienceInbox._require_training_scope):
        inbox, source, label = make_inbox()
        trace = []

        class Key:
            def __hash__(self):
                trace.append("hash")
                return 17

            def __eq__(self, other):
                trace.append("equal")
                return False

        record = {"source": source, "permissions": source.permissions, "label": label}[owner]
        vars(record)[Key()] = object()
        trace.clear()
        result = outcome(check, inbox)
        results.append((result, tuple(trace)))
    assert results[0] == results[1] == (None, ())
