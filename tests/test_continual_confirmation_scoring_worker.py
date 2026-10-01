"""Private worker composition on genuine development models/fake final fields.

Partial JSON validation is a declared private spy; no fixture policy can pass
the public fixed worker or publication gate. No reserved data/model/score.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import pytest

from test_continual_confirmation_scoring_state import development_training as development_training
from test_continual_confirmation_scoring import _fixture_training, _run_fixture, _seal_scoring
from test_continual_confirmation_final_runtime import _methods
from src.app.continual_confirmation_execution import json_value
from src.app.continual_confirmation_scoring_manifest import fixed_scoring_manifest
from src.app.continual_confirmation_scoring_state import _verify_reproduced_state
from src.app.continual_confirmation_scoring_validation import verify_scored_payload
from src.core.confirmation_final_roles import final_content_digest
from src.infra import continual_confirmation_scoring_worker as worker
from src.infra.continual_confirmation_io import parse_json
from src.infra.continual_confirmation_runtime import ConfirmationStopped, ExecutionObserver
from src.shared.process_memory import ProcessRssSampler


def _ports(
    trained: Any, identity: Any, after_json: Any = lambda observed, payload, observer: None
) -> Any:
    held_observer: Any = None

    def state(held: Any, manifest: Any) -> Any:
        return _verify_reproduced_state(held, held.facts.manifest, identity)

    def evaluate(held: Any, manifest: Any, observer: Any, checkpoint: Any) -> Any:
        nonlocal held_observer
        held_observer = observer
        return _run_fixture(
            held,
            identity,
            release=observer.release,
            evaluate=observer.evaluate,
            checkpoint=checkpoint,
        )

    def json_spy(observed: Any, payload: Any, manifest: Any) -> None:
        # The private seam validates actual development state/observations in
        # production code. Only whole-scientific JSON validation is delegated;
        # the public validator MUST reject this incomplete/unbound body.
        assert len(payload["cells"]) == 56 and observed["prediction_attempts"] == 168
        assert payload["external_execution_verified"] is False
        after_json(observed, payload, held_observer)

    return worker._ScoringPorts(evaluate, state, json_spy)


def test_should_serialize_and_recheck_genuine_development_state_views_counts_and_restored_guards(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    trained, identity, sources = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    before = _methods()
    checks = 0

    def check() -> None:
        nonlocal checks
        checks += 1

    with ProcessRssSampler(read_rss_bytes=lambda: 100) as sampler:
        budget = ExecutionObserver(
            fixed_scoring_manifest().train_manifest, sampler, clock=lambda: 0.0
        )
        ports = _ports(trained, identity)
        parts = worker._score_held(trained, fixed_scoring_manifest(), budget, check, ports)
        worker._require_complete_state(trained, fixed_scoring_manifest(), parts, ports)
        assert parse_json(parts.result_json) == json_value(asdict(parts.scored))
        assert parts.observations["prediction_examples"] == 6720
        assert parts.observations["by_model_kind"] == {"backprop": 42, "pc": 45, "circadian": 81}
        assert parts.observer.active is False
    assert checks == 4 and _methods() == before
    assert budget.executed_updates == budget.attempted_updates == 0
    assert all(source.reads_input == source.reads_target == 1 for source in sources)
    with pytest.raises(ValueError):
        verify_scored_payload(parse_json(parts.result_json), fixed_scoring_manifest())


@pytest.mark.parametrize(
    "kind",
    [
        "parameter",
        "selector_rng",
        "fact",
        "input",
        "label",
        "resealed_label",
        "observed",
        "json_error",
    ],
)
def test_should_reject_late_corruption_from_serialized_validation_and_restore_all_hooks(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    trained, identity, sources = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    before = _methods()

    def after(observed: Any, payload: Any, observer: Any) -> None:
        item = trained.held[-1]
        model = item.models_after_b[trained.facts.manifest.families[-1].arms[-1]]
        if kind == "parameter":
            model.bias_output[0, 0] += 1.0
        elif kind == "selector_rng":
            parent = item.models_after_b["random_growth"]
            parent._parent_selection_rng.random()
        elif kind == "fact":
            object.__setattr__(trained.facts, "all_a_completed_before_first_b", False)
        elif kind == "input":
            sources[-1].inputs[0, 0] += 1.0
        elif kind in {"label", "resealed_label"}:
            sources[-1].targets[0, 0] = 1.0 - sources[-1].targets[0, 0]
            if kind == "resealed_label":
                role = observer.views[observer.bindings[-1].key][0]
                object.__setattr__(role, "sha256", final_content_digest(role))
        elif kind == "observed":
            observed["prediction_examples"] -= 40
        else:
            raise RuntimeError("fabricated serialized validator failure")

    callbacks = 0

    def check() -> None:
        nonlocal callbacks
        callbacks += 1

    with ProcessRssSampler(read_rss_bytes=lambda: 100) as sampler:
        budget = ExecutionObserver(
            fixed_scoring_manifest().train_manifest, sampler, clock=lambda: 0.0
        )
        with pytest.raises((ValueError, RuntimeError)):
            worker._score_held(
                trained, fixed_scoring_manifest(), budget, check, _ports(trained, identity, after)
            )
    assert _methods() == before and all(
        source.reads_input == source.reads_target == 1 for source in sources
    )


@pytest.mark.parametrize("kind", ["parameter", "role", "binding", "io_error"])
def test_should_recheck_every_live_state_and_retained_view_after_the_last_binding_callback(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    trained, identity, sources = _fixture_training(development_training)
    if kind == "role":
        item = trained.held[-1]
        # The module fixture shares sealed training arrays across examples;
        # corrupt an owned copy so later cases still start from valid facts.
        role = item.roles_b
        item.roles_b = replace(
            role,
            train=replace(
                role.train, input=role.train.input.copy(), target=role.train.target.copy()
            ),
        )
    _seal_scoring(monkeypatch)
    checks = 0
    before = _methods()

    def check() -> None:
        nonlocal checks
        checks += 1
        if checks != 4:
            return
        item = trained.held[-1]
        if kind == "parameter":
            item.models_after_b[trained.facts.manifest.families[-1].arms[-1]].bias_output[0, 0] += (
                1.0
            )
        elif kind == "role":
            item.roles_b.train.target[0, 0] = 1.0 - item.roles_b.train.target[0, 0]
        elif kind == "binding":
            item.models_after_b[trained.facts.manifest.families[-1].arms[-1]] = trained.held[
                0
            ].models_after_a[trained.facts.manifest.families[0].arms[0]]
        else:
            raise OSError("fabricated current reference/source/request failure")

    with ProcessRssSampler(read_rss_bytes=lambda: 100) as sampler:
        budget = ExecutionObserver(
            fixed_scoring_manifest().train_manifest, sampler, clock=lambda: 0.0
        )
        with pytest.raises((ValueError, OSError)):
            worker._score_held(
                trained, fixed_scoring_manifest(), budget, check, _ports(trained, identity)
            )
    assert _methods() == before and all(
        source.reads_input == source.reads_target == 1 for source in sources
    )


@pytest.mark.parametrize("kind", ["rss", "wall"])
def test_should_abort_on_original_caps_after_scientific_serialization_without_returning_parts(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    trained, identity, sources = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    status: dict[str, Any] = {"rss": 100, "clock": 0.0}
    before = _methods()

    def after(observed: Any, payload: Any, observer: Any) -> None:
        status["rss" if kind == "rss" else "clock"] = 536870913 if kind == "rss" else 600.0

    with ProcessRssSampler(read_rss_bytes=lambda: status["rss"]) as sampler:
        budget = ExecutionObserver(
            fixed_scoring_manifest().train_manifest, sampler, clock=lambda: status["clock"]
        )
        with pytest.raises(ConfirmationStopped, match="rss_limit|wall_limit"):
            worker._score_held(
                trained,
                fixed_scoring_manifest(),
                budget,
                lambda: None,
                _ports(trained, identity, after),
            )
    assert _methods() == before and all(
        source.reads_input == source.reads_target == 1 for source in sources
    )


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_should_restore_every_observer_hook_when_serialized_validation_is_canceled(
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    error: type[BaseException],
) -> None:
    trained, identity, sources = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    before = _methods()

    def after(observed: Any, payload: Any, observer: Any) -> None:
        raise error("fabricated canceled scored worker")

    with ProcessRssSampler(read_rss_bytes=lambda: 100) as sampler:
        budget = ExecutionObserver(
            fixed_scoring_manifest().train_manifest, sampler, clock=lambda: 0.0
        )
        with pytest.raises(error):
            worker._score_held(
                trained,
                fixed_scoring_manifest(),
                budget,
                lambda: None,
                _ports(trained, identity, after),
            )
    assert _methods() == before and all(
        source.reads_input == source.reads_target == 1 for source in sources
    )


def test_should_abort_on_scientific_serialization_failure_after_all_actual_predictions(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    trained, identity, sources = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    original = worker.json.dumps
    before = _methods()

    def broken(value: Any, *args: Any, **kwargs: Any) -> Any:
        if type(value) is dict and "training_before_release" in value:
            raise OSError("fabricated scientific serialization failure")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(worker.json, "dumps", broken)
    with ProcessRssSampler(read_rss_bytes=lambda: 100) as sampler:
        budget = ExecutionObserver(
            fixed_scoring_manifest().train_manifest, sampler, clock=lambda: 0.0
        )
        with pytest.raises(OSError, match="serialization"):
            worker._score_held(
                trained, fixed_scoring_manifest(), budget, lambda: None, _ports(trained, identity)
            )
    assert _methods() == before and all(
        source.reads_input == source.reads_target == 1 for source in sources
    )


def test_should_refuse_private_development_training_with_all_public_default_worker_ports(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    trained, _, sources = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    before = _methods()
    with ProcessRssSampler(read_rss_bytes=lambda: 100) as sampler:
        budget = ExecutionObserver(
            fixed_scoring_manifest().train_manifest, sampler, clock=lambda: 0.0
        )
        with pytest.raises(ValueError, match="inventory|artifact"):
            worker._score_held(trained, fixed_scoring_manifest(), budget, lambda: None)
    assert _methods() == before and all(
        source.reads_input == source.reads_target == 0 for source in sources
    )
