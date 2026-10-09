"""Pure handoff framework contracts; fake leases confer no native authority.

Why this: allocating the exact coordinator without its constructor isolates scope
and publication dispatch from ledger/native behavior. Only the trusted class
method is replaced, with a clearly fake context manager. Actual coordinator
integration belongs to the separate root-owned tests.
"""

from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import dataclass, field
from threading import Thread
from types import SimpleNamespace
from typing import Any

import pytest

from src.app.checkpoint_replay_handoff import (
    _ReplayTransition,
    checkpoint_publication_lease,
    commit_replay_transition,
    lease_replay_handoff,
)
from src.app.managed_replay_checkpoints import ManagedReplayCheckpoints
from src.app.managed_replay_origins import ManagedReplayOrigins


def _make_fake_exact_ledger():
    """Constructor-free exact ledger shell; no origin admission or native owner."""
    ledger = object.__new__(ManagedReplayOrigins)
    ledger._anchors = {}
    ledger._rows = {}
    return ledger


@dataclass
class FakePublicationLease:
    coordinator: Any = field(default_factory=lambda: object.__new__(ManagedReplayCheckpoints))
    controller: Any = field(default_factory=object)
    original: Any = field(default_factory=object)
    prepared: Any = field(default_factory=object)
    pending: Any = field(default_factory=object)
    ledger: Any = field(default_factory=_make_fake_exact_ledger)
    calls: list[tuple[Any, ...]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    active: bool = False
    failure: Exception | None = None
    transition: Any = None

    def publication(self, controller=None):
        return checkpoint_publication_lease(
            self.controller if controller is None else controller,
            self.original,
            self.prepared,
            self.pending,
        )


@pytest.fixture
def fake_publication(monkeypatch):
    fake = FakePublicationLease()
    fake.transition = _ReplayTransition(fake.ledger, {"actor": object()}, {1: object()})

    @contextmanager
    def fake_trusted_class_publication_lease(coordinator, controller, original, prepared, pending):
        """Fake lifecycle witness only: no ledger admission or native construction."""
        fake.calls.append((coordinator, controller, original, prepared, pending))
        fake.active = True
        fake.events.append("acquired")
        try:
            if fake.failure is not None:
                raise fake.failure
            yield fake.transition
        finally:
            fake.active = False
            fake.events.append("released")

    monkeypatch.setattr(
        ManagedReplayCheckpoints,
        "_publication_lease",
        fake_trusted_class_publication_lease,
    )
    return fake


def test_should_yield_none_without_scope_or_trusted_callback(fake_publication):
    with fake_publication.publication() as transition:
        assert transition is None
    assert fake_publication.calls == []
    assert fake_publication.events == []


def test_should_release_default_publication_when_body_raises(fake_publication):
    with pytest.raises(RuntimeError, match="default body failed"):
        with fake_publication.publication() as transition:
            assert transition is None
            raise RuntimeError("default body failed")
    assert fake_publication.calls == []


@pytest.mark.parametrize("kind", ["object", "subclass"])
def test_should_reject_nonexact_coordinator_before_entering_scope(fake_publication, kind):
    class CoordinatorSubclass(ManagedReplayCheckpoints):
        pass

    coordinator = object() if kind == "object" else object.__new__(CoordinatorSubclass)
    with pytest.raises(ValueError, match="exact trusted replay coordinator"):
        with lease_replay_handoff(fake_publication.controller, coordinator):
            pytest.fail("nonexact coordinator entered handoff body")
    assert fake_publication.calls == []
    with fake_publication.publication() as transition:
        assert transition is None


def test_should_hold_trusted_lease_through_body_then_release(fake_publication):
    fake = fake_publication
    with lease_replay_handoff(fake.controller, fake.coordinator):
        with fake.publication() as transition:
            assert transition is fake.transition
            assert fake.active
            assert fake.events == ["acquired"]
            fake.events.append("body")
        assert not fake.active
        assert fake.events == ["acquired", "body", "released"]
    assert fake.calls == [
        (fake.coordinator, fake.controller, fake.original, fake.prepared, fake.pending)
    ]


def test_should_dispatch_unbound_trusted_method_over_instance_shadow(fake_publication, monkeypatch):
    fake = fake_publication

    def forbidden_instance_callback(*args, **kwargs):
        pytest.fail("instance-supplied publication callback ran")

    monkeypatch.setattr(fake.coordinator, "_publication_lease", forbidden_instance_callback)
    with lease_replay_handoff(fake.controller, fake.coordinator):
        with fake.publication() as transition:
            assert transition is fake.transition
    assert len(fake.calls) == 1


def test_should_reject_other_controller_without_spending_original_attempt(fake_publication):
    fake = fake_publication
    with lease_replay_handoff(fake.controller, fake.coordinator):
        with pytest.raises(ValueError, match="original single handoff attempt"):
            with fake.publication(object()):
                pytest.fail("foreign controller entered publication body")
        assert fake.calls == []
        with fake.publication() as transition:
            assert transition is fake.transition
    assert len(fake.calls) == 1


def test_should_allow_only_one_publication_in_scope(fake_publication):
    fake = fake_publication
    with lease_replay_handoff(fake.controller, fake.coordinator):
        with fake.publication():
            with pytest.raises(ValueError, match="original single handoff attempt"):
                with fake.publication():
                    pytest.fail("nested publication entered body")
        with pytest.raises(ValueError, match="original single handoff attempt"):
            with fake.publication():
                pytest.fail("second publication entered body")
    assert len(fake.calls) == 1
    assert fake.events == ["acquired", "released"]


def test_should_spend_attempt_when_trusted_lease_callback_fails(fake_publication):
    fake = fake_publication
    failure = RuntimeError("fake trusted lease failed")
    fake.failure = failure
    with lease_replay_handoff(fake.controller, fake.coordinator):
        with pytest.raises(RuntimeError, match="fake trusted lease failed") as caught:
            with fake.publication():
                pytest.fail("failed callback entered publication body")
        assert caught.value is failure
        fake.failure = None
        with pytest.raises(ValueError, match="original single handoff attempt"):
            with fake.publication():
                pytest.fail("failed attempt was reused")
    assert len(fake.calls) == 1
    assert not fake.active
    assert fake.events == ["acquired", "released"]


@pytest.mark.parametrize("kind", ["none", "lookalike", "subclass"])
def test_should_reject_nonexact_transition_before_body_and_spend_attempt(fake_publication, kind):
    fake = fake_publication

    class TransitionSubclass(_ReplayTransition):
        pass

    values = fake.transition
    if kind == "none":
        fake.transition = None
    elif kind == "lookalike":
        fake.transition = SimpleNamespace(
            ledger=values.ledger, anchors=values.anchors, rows=values.rows
        )
    else:
        fake.transition = TransitionSubclass(values.ledger, values.anchors, values.rows)
    with lease_replay_handoff(fake.controller, fake.coordinator):
        with pytest.raises(ValueError, match="prepared exact replay transition"):
            with fake.publication():
                pytest.fail("nonexact transition reached publication body")
        assert not fake.active
        with pytest.raises(ValueError, match="original single handoff attempt"):
            with fake.publication():
                pytest.fail("invalid transition attempt was reused")
    assert len(fake.calls) == 1
    assert fake.events == ["acquired", "released"]


def test_should_release_lease_and_spend_attempt_after_body_failure(fake_publication):
    fake = fake_publication
    with lease_replay_handoff(fake.controller, fake.coordinator):
        with pytest.raises(RuntimeError, match="publication body failed"):
            with fake.publication():
                assert fake.active
                raise RuntimeError("publication body failed")
        assert not fake.active
        with pytest.raises(ValueError, match="original single handoff attempt"):
            with fake.publication():
                pytest.fail("body failure allowed another publication")
    assert fake.events == ["acquired", "released"]


def test_should_refuse_scope_close_while_publication_is_active(fake_publication):
    fake = fake_publication
    with lease_replay_handoff(fake.controller, fake.coordinator) as scope:
        with fake.publication() as transition:
            with pytest.raises(ValueError, match="cannot close during active publication"):
                scope.__exit__(None, None, None)
            assert not scope.closed
            assert fake.active
            assert fake.events == ["acquired"]
            scope.require()
            assert transition is fake.transition
        assert not fake.active
        assert fake.events == ["acquired", "released"]
    assert scope.closed


def test_should_refuse_copied_context_publication_close_before_coordinator_cleanup(
    fake_publication,
):
    fake = fake_publication
    publication = fake.publication()
    with lease_replay_handoff(fake.controller, fake.coordinator) as scope:
        with publication as transition:
            inherited = copy_context()
            with pytest.raises(ValueError, match="original context"):
                inherited.run(publication.__exit__, None, None, None)
            assert fake.active
            assert fake.events == ["acquired"]
            assert not scope.closed
            scope.require()
            assert transition is fake.transition
        assert not fake.active
        assert fake.events == ["acquired", "released"]
        with pytest.raises(ValueError, match="original live thread/lease"):
            publication.__exit__(None, None, None)
    assert scope.closed
    assert len(fake.calls) == 1


def test_should_reject_foreign_ledger_before_body_or_setter(fake_publication):
    fake = fake_publication
    setters: list[str] = []

    class ForeignLedger:
        def __setattr__(self, name, value):
            setters.append(name)
            raise AssertionError("foreign ledger setter ran")

    valid = fake.transition
    fake.transition = _ReplayTransition(ForeignLedger(), valid.anchors, valid.rows)
    with lease_replay_handoff(fake.controller, fake.coordinator):
        with pytest.raises(ValueError, match="prepared exact replay transition"):
            with fake.publication() as transition:
                commit_replay_transition(transition)
                pytest.fail("foreign ledger reached publication body")
        assert setters == []
        assert not fake.active
        with pytest.raises(ValueError, match="original single handoff attempt"):
            with fake.publication():
                pytest.fail("foreign ledger attempt was reused")
    assert fake.events == ["acquired", "released"]


@pytest.mark.parametrize("field_name", ["anchors", "rows"])
def test_should_reject_dictionary_subclass_before_body(fake_publication, field_name):
    fake = fake_publication

    class DictionarySubclass(dict):
        pass

    valid = fake.transition
    anchors = DictionarySubclass(valid.anchors) if field_name == "anchors" else valid.anchors
    rows = DictionarySubclass(valid.rows) if field_name == "rows" else valid.rows
    fake.transition = _ReplayTransition(valid.ledger, anchors, rows)
    with lease_replay_handoff(fake.controller, fake.coordinator):
        with pytest.raises(ValueError, match="prepared exact replay transition"):
            with fake.publication():
                pytest.fail("dictionary subclass reached publication body")
        assert not fake.active
    assert fake.events == ["acquired", "released"]


def test_should_refuse_inherited_context_publication_and_close_preserving_original(
    fake_publication,
):
    fake = fake_publication
    with lease_replay_handoff(fake.controller, fake.coordinator) as scope:
        inherited = copy_context()

        def foreign_publication():
            with fake.publication():
                pytest.fail("inherited context entered publication body")

        with pytest.raises(ValueError, match="original context"):
            inherited.run(foreign_publication)
        with pytest.raises(ValueError, match="original context"):
            inherited.run(scope.__exit__, None, None, None)
        assert not scope.closed
        assert fake.calls == []
        scope.require()
        with fake.publication() as transition:
            assert transition is fake.transition
    assert scope.closed
    assert len(fake.calls) == 1


def _join_single_contender(action):
    """One bounded contender per caller; suite has exactly two callers."""
    errors: list[BaseException] = []

    def run_action():
        try:
            action()
        except BaseException as error:
            errors.append(error)

    contender = Thread(target=run_action, daemon=True)
    contender.start()
    contender.join(timeout=2)
    assert not contender.is_alive(), "bounded handoff contender did not finish"
    assert errors == [], errors


def test_should_refuse_inherited_scope_on_foreign_thread_without_spending_attempt(fake_publication):
    fake = fake_publication
    with lease_replay_handoff(fake.controller, fake.coordinator) as scope:
        inherited = copy_context()

        def foreign_attempts():
            with pytest.raises(ValueError, match="original live thread/scope"):
                with fake.publication():
                    pytest.fail("foreign thread reached publication body")
            with pytest.raises(ValueError, match="original live thread/scope"):
                scope.__exit__(None, None, None)

        _join_single_contender(lambda: inherited.run(foreign_attempts))
        assert not scope.closed
        assert fake.calls == []
        with fake.publication() as transition:
            assert transition is fake.transition
    assert len(fake.calls) == 1


def test_should_refuse_foreign_thread_entry_before_setting_scope(fake_publication):
    fake = fake_publication
    scope = lease_replay_handoff(fake.controller, fake.coordinator)

    def foreign_entry():
        with pytest.raises(ValueError, match="original thread"):
            scope.__enter__()
        with fake.publication() as transition:
            assert transition is None

    _join_single_contender(foreign_entry)
    with scope:
        with fake.publication() as transition:
            assert transition is fake.transition
    assert len(fake.calls) == 1


def test_should_reject_nested_scope_and_reentry_preserving_original(fake_publication):
    fake = fake_publication
    with lease_replay_handoff(fake.controller, fake.coordinator) as scope:
        nested = lease_replay_handoff(fake.controller, fake.coordinator)
        with pytest.raises(ValueError, match="cannot nest or reopen"):
            nested.__enter__()
        with pytest.raises(ValueError, match="cannot nest or reopen"):
            scope.__enter__()
        with fake.publication() as transition:
            assert transition is fake.transition
    assert len(fake.calls) == 1


def test_should_close_original_scope_and_refuse_reopen_or_repeated_close(fake_publication):
    fake = fake_publication
    scope = lease_replay_handoff(fake.controller, fake.coordinator)
    with scope:
        with fake.publication() as transition:
            assert transition is fake.transition
    with pytest.raises(ValueError, match="original live thread/scope"):
        scope.require()
    with pytest.raises(ValueError, match="cannot nest or reopen"):
        scope.__enter__()
    with pytest.raises(ValueError, match="original live thread/scope"):
        scope.__exit__(None, None, None)
    with fake.publication() as transition:
        assert transition is None
    assert len(fake.calls) == 1


def test_should_release_scope_when_outer_body_raises(fake_publication):
    fake = fake_publication
    with pytest.raises(RuntimeError, match="handoff body failed"):
        with lease_replay_handoff(fake.controller, fake.coordinator) as scope:
            raise RuntimeError("handoff body failed")
    assert scope.closed
    with fake.publication() as transition:
        assert transition is None
    assert fake.calls == []


def test_should_assign_only_prepared_anchor_and_row_references_on_commit(fake_publication):
    fake = fake_publication
    old_anchors, old_rows = fake.ledger._anchors, fake.ledger._rows
    fake.ledger.unrelated = object()
    unrelated = fake.ledger.unrelated
    with lease_replay_handoff(fake.controller, fake.coordinator):
        with fake.publication() as transition:
            assert fake.ledger._anchors is old_anchors
            assert fake.ledger._rows is old_rows
            commit_replay_transition(transition)
            assert fake.ledger._anchors is transition.anchors
            assert fake.ledger._rows is transition.rows
            assert fake.ledger.unrelated is unrelated
            assert fake.active
    assert fake.events == ["acquired", "released"]
