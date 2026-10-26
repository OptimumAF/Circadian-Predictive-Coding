"""Reset reservation admission stops before any environment exists.

16 cases per platform:8 forbidden role/seed pairs never call the factory;
8 preserved routes each call a sentinel factory once and receive no environment.
Zero scene/reset/step/close/model/RNG/fit/SGD/forward/native work.
"""

from typing import Any

import pytest

from src.infra import maniskill_continuous as native


class FactoryBoundaryReached(RuntimeError):
    """The accepted constructor reached IO admission, without creating a scene."""


@pytest.mark.parametrize(
    "role,seed",
    [
        ("train", 29),
        ("inner_guard", 29),
        ("probe_train", 29),
        ("probe_test", 29),
        ("final_test", 29),
        ("train", 23),
        ("inner_guard", 23),
        ("final_test", 23),
    ],
)
def test_should_refuse_reserved_reset_role_combinations_before_environment_factory(
    monkeypatch, role, seed
):
    calls = []

    def forbidden(**options):
        calls.append(options)
        pytest.fail("reserved reset crossed the environment construction boundary")

    monkeypatch.setattr(native, "_make_environment", forbidden)
    context = f"pick-cube-inner-guard-{seed}" if role == "inner_guard" else "pick-cube"
    with pytest.raises(ValueError):
        native.ManiSkillPickCubeEpisode("reserved-reset", context, role=role, seed=seed)
    assert calls == []


@pytest.mark.parametrize(
    "role,seed,context",
    [
        ("train", 17, "pick-cube"),
        ("train", 19, "pick-cube-train-19"),
        ("probe_train", 23, "caller-probe-context-a"),
        ("probe_test", 23, "caller-probe-context-b"),
        ("probe_train", None, "generic-probe-train"),
        ("probe_test", None, "generic-probe-test"),
        ("inner_guard", 31, "pick-cube-inner-guard-31"),
        ("inner_guard", 37, "pick-cube-inner-guard-37"),
    ],
)
def test_should_preserve_allowed_probe_train_and_guard_routes_without_new_factory_options(
    monkeypatch, role, seed, context
):
    calls = []
    stopped = FactoryBoundaryReached("factory sentinel; no environment was created")

    def factory(**options):
        calls.append(options)
        raise stopped

    monkeypatch.setattr(native, "_make_environment", factory)
    options: dict[str, Any] = dict(role=role)
    if seed is not None:
        options["seed"] = seed
    # The None cases deliberately omit the keyword and exercise the unchanged
    # default17 probe route. Arbitrary probe contexts are not qualification.
    with pytest.raises(FactoryBoundaryReached) as observed:
        native.ManiSkillPickCubeEpisode("preserved-reset", context, **options)
    assert observed.value is stopped
    assert calls == [{}]
