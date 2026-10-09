"""Consent admission before opaque payload access and after owned handoff."""

from dataclasses import replace
from typing import Any

import pytest

from src.app.managed_experience import ManagedExperienceOwner
from src.core.data_lifecycle import (
    DataConsent,
    DataProvenance,
    LifecycleDeclaration,
    LifecycleLimits,
)
from src.core.experience import ExperiencePermissions
from test_experience_inbox import experience, label
from test_resource_sharing import shared
from test_candidate_checkpoint import controller


def declaration(sample="s1", subject="person", **kwargs):
    return LifecycleDeclaration(
        ("e1", sample),
        DataProvenance("local", subject, True, False),
        DataConsent(True, True),
        "replay",
        **kwargs,
    )


def setup(limits=None):
    wrapper, runtime, clock, gate, source, budget = shared()
    owner = ManagedExperienceOwner(wrapper, limits=limits or LifecycleLimits(4, 4))
    return owner, wrapper, runtime, clock, gate, source, budget


def record(owner, sample="s1", at=3):
    owner.record_label(replace(label(sample, arrived_at=at), targets=[8, 8]))
    owner.record_experience(
        replace(experience(sample), permissions=ExperiencePermissions(True, True))
    )


class CannotCopy:
    def __deepcopy__(self, memo):
        raise AssertionError("forbidden payload was accessed")


@pytest.mark.parametrize(
    "field,value", [("verified", 1), ("synthetic", None), ("source_id", " "), ("subject_id", "")]
)
def test_should_validate_provenance_exact_metadata(field, value):
    with pytest.raises(ValueError):
        DataProvenance(
            **{
                **dict(source_id="local", subject_id="person", verified=True, synthetic=False),
                field: value,
            }
        )


@pytest.mark.parametrize("limits", [(True, 1), (-1, 1), (1, -1), (1, 1.0)])
def test_should_reject_invalid_lifetime_quotas(limits):
    with pytest.raises(ValueError):
        LifecycleLimits(*limits)


@pytest.mark.parametrize(
    "change", ["unverified", "synthetic", "transient", "audit_only", "training", "replay"]
)
def test_should_refuse_unsupported_declarations_without_consuming_catalog(change):
    owner, *_ = setup()
    item = declaration()
    if change in ("unverified", "synthetic"):
        item = replace(
            item,
            provenance=replace(
                item.provenance,
                **({"verified": False} if change == "unverified" else {"synthetic": True}),
            ),
        )
    elif change in ("transient", "audit_only"):
        item = replace(item, retention=change)
    else:
        item = replace(item, consent=replace(item.consent, **{change: False}))
    with pytest.raises(ValueError):
        owner.declare(item)
    assert owner.declarations == ()


@pytest.mark.parametrize("kind", ["source", "label", "heldout", "permissions"])
def test_should_deny_unknown_or_unpermitted_events_before_payload_copy(kind):
    owner, _, runtime, *_ = setup()
    owner.declare(declaration())
    if kind == "label":
        event = replace(label("unknown"), targets=CannotCopy())
        operation = owner.record_label
    else:
        event = replace(experience("unknown" if kind == "source" else "s1"), features=CannotCopy())
        if kind == "heldout":
            event = replace(event, role="final_test", permissions=ExperiencePermissions())
        operation = owner.record_experience
    with pytest.raises(ValueError):
        operation(event)
    assert runtime._inbox._experiences == runtime._inbox._labels == {}


def test_should_handle_label_first_future_delivery_and_reject_duplicates():
    owner, wrapper, runtime, clock, gate, _, budget = setup()
    owner.declare(declaration())
    record(owner)
    assert wrapper.train_ready().updates == ()
    clock.advance_to(3)
    assert len(wrapper.train_ready().updates) == 1
    with pytest.raises(ValueError, match="duplicate"):
        owner.record_label(label())
    assert (
        runtime.train_ready() == ()
        and budget.updates_completed == gate.snapshot().admitted_updates == 1
    )


@pytest.mark.parametrize("kind", ["source", "label"])
def test_should_deny_underlying_public_registration_even_for_declared_identity(kind):
    owner, _, runtime, *_ = setup()
    owner.declare(declaration())
    with pytest.raises(ValueError, match="managed"):
        if kind == "source":
            runtime.record_experience(
                replace(
                    experience(),
                    features=CannotCopy(),
                    permissions=ExperiencePermissions(True, True),
                )
            )
        else:
            runtime.record_label(replace(label(), targets=CannotCopy()))


def test_should_revoke_pending_and_future_training_without_refunding_lifetime_quota():
    owner, wrapper, runtime, clock, gate, _, budget = setup(LifecycleLimits(1, 1))
    owner.declare(declaration())
    record(owner)
    owner.opt_out("person")
    owner.opt_out("person")
    clock.advance_to(3)
    assert runtime.train_ready() == () and wrapper.train_ready().updates == ()
    assert budget.updates_completed == gate.snapshot().admitted_updates == 0
    with pytest.raises(ValueError, match="opted out"):
        owner.declare(declaration("s2"))
    with pytest.raises(ValueError, match="quota"):
        owner.declare(declaration("s2", "another"))
    assert len(owner.declarations) == 1
    assert runtime._inbox._experiences  # opt-out does not promise deletion


def test_should_preserve_original_authority_through_checkpoint_handoff():
    owner, wrapper, old, clock, gate, _, budget = setup()
    owner.declare(declaration())
    record(owner)
    gate.pause()
    control = controller(wrapper)
    new = control.restore(control.capture())
    assert new._inbox._registration_guard is old._inbox._registration_guard
    assert new._inbox._training_guard is old._inbox._training_guard
    owner.opt_out("person")
    gate.resume()
    clock.advance_to(3)
    assert new.train_ready() == () and budget.updates_completed == 0
    owner.declare(declaration("s2", "another"))
    record(owner, "s2")
    assert len(wrapper.train_ready().updates) == 1
    with pytest.raises(ValueError, match="retired"):
        old.train_ready()
    with pytest.raises(ValueError, match="managed"):
        new.record_label(label("extra"))


@pytest.mark.parametrize("mutation", ["declare", "opt_out"])
def test_should_invalidate_checkpoint_before_consent_catalog_mutation_can_be_undone(mutation):
    owner, wrapper, *_ = setup()
    owner.declare(declaration())
    wrapper._sharing.pause()
    control = controller(wrapper)
    token = control.capture()
    if mutation == "declare":
        owner.declare(declaration("s2"))
    else:
        owner.opt_out("person")
    with pytest.raises(ValueError, match="stale|changed"):
        control.restore(token)


def test_should_refuse_installation_after_registration_or_second_install():
    owner, wrapper, *_ = setup()
    with pytest.raises(ValueError, match="fresh"):
        ManagedExperienceOwner(wrapper, limits=LifecycleLimits(4, 4))
    wrapper, runtime, *_ = shared()
    runtime.record_label(label())
    with pytest.raises(ValueError, match="fresh"):
        ManagedExperienceOwner(wrapper, limits=LifecycleLimits(4, 4))


def test_should_refuse_busy_mutation_without_partial_catalog_change():
    owner, _, runtime, *_ = setup()
    with runtime._exclusive(), pytest.raises(ValueError, match="busy"):
        owner.declare(declaration())
    assert owner.declarations == ()
    with pytest.raises(ValueError, match="unknown"):
        owner.opt_out("unknown")


@pytest.mark.parametrize("kind", ["consent", "key", "category", "flags"])
def test_should_reject_malformed_declaration_and_policy(kind):
    bad_flag: Any = 1
    with pytest.raises(ValueError):
        if kind == "consent":
            DataConsent(bad_flag, True)
        elif kind == "key":
            replace(declaration(), key=("e1", ""))
        elif kind == "category":
            replace(declaration(), retention="unknown")
        else:
            LifecycleLimits(1, 1, allow_synthetic=bad_flag)


def test_should_allow_explicit_synthetic_unverified_policy_and_enforce_zero_quota():
    owner, *_ = setup(LifecycleLimits(1, 1, True, True))
    item = replace(declaration(), provenance=DataProvenance("fixture", "person", False, True))
    owner.declare(item)
    with pytest.raises(ValueError, match="duplicate"):
        owner.declare(item)
    owner, *_ = setup(LifecycleLimits(1, 0))
    with pytest.raises(ValueError, match="quota"):
        owner.declare(declaration())


@pytest.mark.parametrize("kind", ["source", "label"])
def test_should_deny_wrong_actor_version_before_copy(kind):
    owner, *_ = setup()
    owner.declare(declaration())
    with pytest.raises(ValueError, match="actor version"):
        if kind == "source":
            owner.record_experience(
                replace(
                    experience(),
                    model_version="other",
                    features=CannotCopy(),
                    permissions=ExperiencePermissions(True, True),
                )
            )
        else:
            owner.record_label(replace(label(), model_version="other", targets=CannotCopy()))


def test_should_refuse_reentrant_catalog_mutation_during_payload_copy_and_clear_ticket():
    owner, _, runtime, *_ = setup()
    owner.declare(declaration())

    class Reentrant:
        def __deepcopy__(self, memo):
            owner.opt_out("person")

    with pytest.raises(ValueError, match="busy"):
        owner.record_label(replace(label(), targets=Reentrant()))
    assert owner._issuing is None and runtime._inbox._labels == {}
    record(owner)


@pytest.mark.parametrize("kind", ["backprop", "circadian"])
def test_should_apply_one_native_wake_then_revoke_second_pair_without_actor_mutation(kind):
    import pickle
    import numpy as np
    from src.app.actor_shadow import ActorShadowRuntime
    from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.experience import Experience, LabelArrival, LogicalClock
    from src.core.resource_sharing import SharingLimits
    from test_experience_inbox import make_native_pair

    _, source = make_native_pair(kind)
    source_before = pickle.dumps(source.snapshot_state())
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    runtime = ActorShadowRuntime(
        source, actor_version="actor-0", candidate_version="candidate-0", clock=clock, budget=budget
    )
    gate = ServingPriorityGate(SharingLimits(2, 2, 1), resource_available=lambda: True)
    wrapper = ResourceSharedRuntime(runtime, gate)
    owner = ManagedExperienceOwner(wrapper, limits=LifecycleLimits(2, 2, allow_synthetic=True))
    actor_before = pickle.dumps(runtime.actor.snapshot_state())
    for sample, at in (("first", 3), ("second", 5)):
        owner.declare(
            replace(declaration(sample), provenance=DataProvenance("fixture", "person", True, True))
        )
        owner.record_label(
            LabelArrival("label-" + sample, sample, "e1", at, "actor-0", np.array([[1.0], [0.0]]))
        )
        owner.record_experience(
            Experience(
                sample,
                "e1",
                1,
                "actor-0",
                np.array([[0.3, -0.2], [-0.5, 0.4]]),
                "train",
                ExperiencePermissions(True, True),
            )
        )
    clock.advance_to(3)
    assert [r.sample_id for r in wrapper.train_ready().updates] == ["first"]
    candidate_after = pickle.dumps(runtime.candidate_snapshot().state)
    owner.opt_out("person")
    clock.advance_to(5)
    assert runtime.train_ready() == () and wrapper.train_ready().updates == ()
    assert pickle.dumps(runtime.candidate_snapshot().state) == candidate_after
    assert pickle.dumps(runtime.actor.snapshot_state()) == actor_before
    assert pickle.dumps(source.snapshot_state()) == source_before
    assert budget.updates_completed == gate.snapshot().admitted_updates == 1
