"""Fixed, mutation-free audit of the existing reward-weighted ranking path."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork


def _model(backend: str, *, use_importance_in_ranking: bool) -> Any:
    split_mix = 0.20 if use_importance_in_ranking else 0.0
    prune_mix = 0.35 if use_importance_in_ranking else 0.0
    if backend == "numpy":
        numpy_config = CircadianConfig(
            use_reward_modulated_learning=True,
            importance_ema_decay=0.5,
            split_importance_mix=split_mix,
            prune_importance_mix=prune_mix,
            max_split_per_sleep=1,
            max_prune_per_sleep=1,
        )
        numpy_model = CircadianPredictiveCodingNetwork(
            2, 4, 41, circadian_config=numpy_config, min_hidden_dim=2, max_hidden_dim=5
        )
        numpy_model.weight_hidden_output[:] = 1.0
        return numpy_model
    torch = pytest.importorskip("torch")
    from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead

    torch_config = CircadianHeadConfig(
        use_reward_modulated_learning=True,
        importance_ema_decay=0.5,
        split_importance_mix=split_mix,
        prune_importance_mix=prune_mix,
        max_split_per_sleep=1,
        max_prune_per_sleep=1,
    )
    torch_model = CircadianPredictiveCodingHead(
        2, 4, 2, torch.device("cpu"), 41, config=torch_config, min_hidden_dim=2, max_hidden_dim=5
    )
    torch_model.weight_hidden_output[:] = torch.tensor([[1.0, 0.0]] * 4)
    return torch_model


def _update_importance(model: Any, backend: str, factors: tuple[float, float]) -> None:
    first = np.array([6.0, 0.0, 0.0, 0.0], dtype=np.float64)
    second = np.array([0.0, 2.5, 0.0, 0.0], dtype=np.float64)
    for values, factor in zip((first, second), factors, strict=True):
        if backend == "numpy":
            numpy_gradient = values.reshape(4, 1)
            model._update_importance_ema(numpy_gradient, reward_scale=factor)
        else:
            import torch

            torch_gradient = torch.as_tensor(
                np.repeat(values[:, None], 2, axis=1), dtype=torch.float32
            )
            model._update_importance_ema(torch_gradient, reward_scale=factor)


def _set_chemistry(model: Any, backend: str, values: tuple[float, ...]) -> None:
    if backend == "numpy":
        model._hidden_chemical = np.asarray(values, dtype=np.float64)
    else:
        import torch

        model._chemical = torch.tensor(values, dtype=torch.float32)


def _as_numpy(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    return values.detach().numpy().astype(np.float64)


@pytest.mark.parametrize("backend", ("numpy", "torch_cpu"))
def test_should_separate_reward_weighting_from_importance_scoring_at_fixed_caps(
    backend: str,
) -> None:
    no_importance = _model(backend, use_importance_in_ranking=False)
    plain_importance = _model(backend, use_importance_in_ranking=True)
    reward_weighted = _model(backend, use_importance_in_ranking=True)
    _update_importance(no_importance, backend, (1.0, 1.5))
    _update_importance(plain_importance, backend, (1.0, 1.0))
    _update_importance(reward_weighted, backend, (1.0, 1.5))
    np.testing.assert_allclose(_as_numpy(plain_importance._importance_ema), [1.5, 1.25, 0.0, 0.0])
    np.testing.assert_allclose(_as_numpy(reward_weighted._importance_ema), [1.5, 1.875, 0.0, 0.0])

    _set_chemistry(no_importance, backend, (1.0, 0.99, 0.0, 0.0))
    _set_chemistry(plain_importance, backend, (1.0, 0.99, 0.0, 0.0))
    _set_chemistry(reward_weighted, backend, (1.0, 0.99, 0.0, 0.0))
    assert no_importance._select_split_indices(max_split_limit=1) == (0,)
    assert plain_importance._select_split_indices(max_split_limit=1) == (0,)
    assert reward_weighted._select_split_indices(max_split_limit=1) == (1,)
    np.testing.assert_allclose(_as_numpy(no_importance._compute_split_scores())[:2], [1.0, 0.993])
    np.testing.assert_allclose(
        _as_numpy(plain_importance._compute_split_scores())[:2], [1.0, 0.9616666666666667]
    )
    np.testing.assert_allclose(
        _as_numpy(reward_weighted._compute_split_scores())[:2], [0.96, 0.995]
    )

    for model in (no_importance, plain_importance, reward_weighted):
        _set_chemistry(model, backend, (0.0, 0.01, 1.0, 1.0))
    assert no_importance._select_prune_indices(max_prune_limit=1) == (0,)
    assert plain_importance._select_prune_indices(max_prune_limit=1) == (1,)
    assert reward_weighted._select_prune_indices(max_prune_limit=1) == (0,)
    np.testing.assert_allclose(_as_numpy(no_importance._compute_prune_scores())[:2], [0.7, 0.693])
    np.testing.assert_allclose(
        _as_numpy(plain_importance._compute_prune_scores())[:2], [0.35, 0.4048333333333333]
    )
    np.testing.assert_allclose(
        _as_numpy(reward_weighted._compute_prune_scores())[:2], [0.42, 0.3465]
    )
    for model in (no_importance, plain_importance, reward_weighted):
        assert model.hidden_dim == 4
        assert model.get_neuron_lineage().neuron_ids == (0, 1, 2, 3)
        assert model.get_sleep_clocks().sleep_events == 0
        assert model.get_sleep_clocks().wake_batches == 0


@pytest.mark.parametrize("backend", ("numpy", "torch_cpu"))
def test_should_preserve_rank_under_constant_positive_reward_factor(backend: str) -> None:
    plain = _model(backend, use_importance_in_ranking=True)
    constant_reward = _model(backend, use_importance_in_ranking=True)
    _update_importance(plain, backend, (1.0, 1.0))
    _update_importance(constant_reward, backend, (1.5, 1.5))
    np.testing.assert_allclose(
        _as_numpy(constant_reward._importance_ema),
        1.5 * _as_numpy(plain._importance_ema),
    )
    for chemistry in ((1.0, 0.99, 0.0, 0.0), (0.0, 0.01, 1.0, 1.0)):
        _set_chemistry(plain, backend, chemistry)
        _set_chemistry(constant_reward, backend, chemistry)
        np.testing.assert_allclose(
            _as_numpy(plain._compute_split_scores()),
            _as_numpy(constant_reward._compute_split_scores()),
        )
        np.testing.assert_allclose(
            _as_numpy(plain._compute_prune_scores()),
            _as_numpy(constant_reward._compute_prune_scores()),
        )
