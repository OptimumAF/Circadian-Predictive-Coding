"""Torch validates post-split proposals on detached head and RNG state."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.core.resnet50_variants import (  # noqa: E402
    CircadianHeadConfig,
    CircadianPredictiveCodingHead,
)


def _config(**changes: Any) -> CircadianHeadConfig:
    options: dict[str, Any] = dict(
        sleep_mode="components",
        sleep_enable_homeostasis=False,
        sleep_enable_chemical_reset=False,
        max_split_per_sleep=1,
        max_prune_per_sleep=1,
        split_threshold=0.8,
        prune_threshold=0.2,
        split_weight_norm_mix=0.0,
        split_importance_mix=0.0,
        prune_weight_norm_mix=0.0,
        prune_importance_mix=0.0,
        split_noise_scale=0.1,
    )
    options.update(changes)
    return CircadianHeadConfig(**options)


def _head(
    config: CircadianHeadConfig | None = None,
    *,
    width: int = 4,
    minimum: int = 3,
    maximum: int = 6,
    seed: int = 479,
) -> CircadianPredictiveCodingHead:
    return CircadianPredictiveCodingHead(
        feature_dim=3,
        hidden_dim=width,
        num_classes=2,
        device=torch.device("cpu"),
        seed=seed,
        config=config or _config(),
        min_hidden_dim=minimum,
        max_hidden_dim=maximum,
    )


def _assert_same_state(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert left.keys() == right.keys()
    for name, value in left.items():
        if torch.is_tensor(value):
            assert torch.equal(value, right[name]), name
        else:
            assert value == right[name], name


def test_invalid_post_split_prune_does_not_mutate_live_head_or_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = _head()
    head._chemical = torch.tensor([0.95, 0.6, 0.4, 0.0])
    before = head.snapshot_state()
    monkeypatch.setattr(head, "_select_prune_indices", lambda **_: (99,))

    with pytest.raises(ValueError, match="prune index"):
        head.sleep_event(force_sleep=True)
    _assert_same_state(head.snapshot_state(), before)


def test_post_split_prune_keeps_legacy_parent_or_child_eligibility() -> None:
    head = _head(_config(prune_threshold=0.6), minimum=3)
    head._chemical = torch.tensor([1.0, 0.9, 0.9, 0.9])
    control = _head(head.config, minimum=3)
    control.restore_state(head.snapshot_state())
    split = control._select_split_indices(max_split_limit=1)
    control._apply_split(split)
    assert split == (0,)
    assert float(control._chemical[0]) == float(control._chemical[4]) == 0.5
    assert control._chemical[0] <= 0.6 and control._chemical[4] <= 0.6
    prune = control._select_prune_indices(max_prune_limit=1)
    assert prune[0] in (0, 4)
    control._apply_prune(prune)

    event = head.sleep_event(force_sleep=True)
    assert event.split_indices == split
    assert event.pruned_indices == prune
    for name in (
        "weight_feature_hidden",
        "bias_hidden",
        "weight_hidden_output",
        "_chemical",
        "_split_cooldown",
        "_prune_cooldown",
    ):
        assert torch.equal(getattr(head, name), getattr(control, name)), name
    assert torch.equal(head._split_generator.get_state(), control._split_generator.get_state())


def test_post_split_prune_can_return_to_initial_minimum_width() -> None:
    head = _head(_config(prune_threshold=0.6), width=4, minimum=4)
    head._chemical = torch.tensor([1.0, 0.9, 0.9, 0.9])

    event = head.sleep_event(force_sleep=True)

    assert event.split_indices == (0,)
    assert len(event.pruned_indices) == 1
    assert event.new_hidden_dim == 4


def test_new_child_can_be_selected_by_post_split_prune() -> None:
    head = _head(
        _config(prune_threshold=0.6, prune_weight_norm_mix=1.0),
        seed=482,
    )
    head._chemical = torch.tensor([1.0, 0.9, 0.9, 0.9])

    event = head.sleep_event(force_sleep=True)

    assert event.split_indices == (0,)
    assert event.pruned_indices == (4,)
    assert event.new_hidden_dim == 4


def test_new_child_and_parent_obey_post_split_prune_cooldown() -> None:
    head = _head(_config(prune_threshold=0.6, prune_cooldown_steps=2))
    head._chemical = torch.tensor([1.0, 0.9, 0.9, 0.9])

    event = head.sleep_event(force_sleep=True)

    assert event.split_indices == (0,)
    assert event.pruned_indices == ()
    assert head.hidden_dim == 5


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("split_index", "split index"),
        ("split_budget", "split budget"),
        ("split_duplicate", "split indices must be unique"),
        ("split_noninteger", "split index"),
        ("fraction", "split budget"),
        ("maximum", "maximum hidden width"),
        ("split_cooldown", "split index is ineligible"),
        ("prune_index", "prune index"),
        ("prune_budget", "prune budget"),
        ("prune_duplicate", "prune indices must be unique"),
        ("minimum", "minimum hidden width"),
        ("prune_cooldown", "prune index is ineligible"),
        ("prune_age", "prune index is ineligible"),
    ],
)
def test_invalid_torch_builtin_proposal_is_rejected_before_live_mutation(
    monkeypatch: pytest.MonkeyPatch, case: str, message: str
) -> None:
    split_limit = 1
    if case == "minimum":
        split_limit = 0
    elif case in {"split_duplicate", "fraction"}:
        split_limit = 2
    config = _config(
        max_split_per_sleep=split_limit,
        max_prune_per_sleep=2 if case in {"minimum", "prune_duplicate"} else 1,
        prune_min_age_steps=2 if case == "prune_age" else 0,
        sleep_max_change_fraction=0.25 if case == "fraction" else 1.0,
    )
    head = _head(
        config,
        minimum=4 if case == "minimum" else 3,
        maximum=4 if case == "maximum" else 6,
    )
    head._chemical = torch.tensor([0.95, 0.6, 0.4, 0.0])
    if case == "split_cooldown":
        head._split_cooldown[0] = 2
    if case == "prune_cooldown":
        head._prune_cooldown[3] = 2
    if case.startswith("split") or case in {"fraction", "maximum"}:
        selected = {
            "split_index": (99,),
            "split_budget": (0, 1),
            "split_duplicate": (0, 0),
            "split_noninteger": (True,),
            "fraction": (0, 1),
        }.get(case, (0,))
        monkeypatch.setattr(head, "_select_split_indices", lambda **_: selected)
    else:
        selected = {
            "prune_index": (99,),
            "prune_budget": (2, 3),
            "prune_duplicate": (3, 3),
        }.get(case, (3,))
        monkeypatch.setattr(head, "_select_prune_indices", lambda **_: selected)
    before = head.snapshot_state()

    with pytest.raises(ValueError, match=message):
        head.sleep_event(force_sleep=True)
    _assert_same_state(head.snapshot_state(), before)
