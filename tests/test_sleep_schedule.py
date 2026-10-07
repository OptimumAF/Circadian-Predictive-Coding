"""Periodic intervals schedule attempts; adaptive and forced paths stay explicit."""

from __future__ import annotations

from typing import Any

import pytest

from src.app.sleep_schedule import decide_sleep_attempt


@pytest.mark.parametrize(
    "mode,epoch,interval,adaptive_due,force_periodic,expected",
    [
        ("components", 3, 0, True, False, (False, True, True, False)),
        ("components", 2, 2, False, True, (True, False, True, True)),
        ("components", 2, 2, False, False, (True, False, True, False)),
        ("components", 2, 2, True, True, (True, True, True, True)),
        ("components", 1, 2, False, True, (False, False, False, False)),
        ("legacy", 2, 2, False, True, (True, False, True, True)),
        ("disabled", 2, 2, True, True, (False, False, False, False)),
        ("components", 0, 2, True, False, (False, True, True, False)),
    ],
)
def test_sleep_attempt_matrix(
    mode: str,
    epoch: int,
    interval: int,
    adaptive_due: bool,
    force_periodic: bool,
    expected: tuple[bool, bool, bool, bool],
) -> None:
    decision = decide_sleep_attempt(
        sleep_mode=mode,
        completed_epochs=epoch,
        interval_epochs=interval,
        adaptive_due=adaptive_due,
        force_periodic=force_periodic,
    )
    assert (
        decision.periodic_due,
        decision.adaptive_due,
        decision.attempted,
        decision.force_sleep,
    ) == expected


@pytest.mark.parametrize(
    "changes,field",
    [
        ({"sleep_mode": "unknown"}, "sleep_mode"),
        ({"completed_epochs": -1}, "completed_epochs"),
        ({"interval_epochs": -1}, "interval_epochs"),
    ],
)
def test_sleep_attempt_rejects_invalid_schedule(changes: dict[str, Any], field: str) -> None:
    options: dict[str, Any] = dict(
        sleep_mode="components",
        completed_epochs=1,
        interval_epochs=2,
        adaptive_due=False,
        force_periodic=True,
    )
    options.update(changes)
    with pytest.raises(ValueError, match=field):
        decide_sleep_attempt(**options)
