"""Compose the existing learners for the fixed sensor demonstration.

Inputs: one exact method name and a common finite 16→8→1 parameter bundle.
Outputs: an owned forkable native learner with the fixed sensor update recipe.
No data selection, evaluation, replay, scheduling, tuning or new learning rule.
Why: the older coupled pilot has a different geometry and learning rate; keeping
this composition separate preserves that recipe while sharing the native models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.actor_ports import ForkableLearner
    from src.core.coupled_energy import Array, CoupledParameters

METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


def validate_sensor_method(method: str) -> None:
    """Validate selection without importing numerical models or arrays."""
    if type(method) is not str or method not in METHODS:
        raise ValueError("method must be an exact string in bp, pc, neutral_cpc, gated_cpc")


def build_sensor_learner(
    method: str, initial: CoupledParameters
) -> ForkableLearner[Array, Array, Array, Any]:
    """Own the supplied initial law and retain fixed policy on every fork."""
    validate_sensor_method(method)
    from src.adapters.coupled_learners import CoupledLearner
    from src.adapters.matched_backprop_learner import MatchedBackpropLearner
    from src.core.coupled_circadian_predictive_coding import (
        CoupledCircadianConfig,
        CoupledCircadianPredictiveCodingNetwork,
    )
    from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
    from src.core.matched_backprop import MatchedBackpropNetwork

    # Use the native topology boundary for every method, including flexible BP.
    # Initialization is overwritten; it does not select a new initial law.
    model = CoupledPredictiveCodingNetwork(16, (8,), seed=0, precisions=(1.0,))
    model.restore_parameters(initial)
    if method == "bp":
        return MatchedBackpropLearner(MatchedBackpropNetwork(initial), learning_rate=0.1)
    if method in ("neutral_cpc", "gated_cpc"):
        config = (
            CoupledCircadianConfig.matched_pc_control()
            if method == "neutral_cpc"
            else CoupledCircadianConfig()
        )
        model = CoupledCircadianPredictiveCodingNetwork(
            16, (8,), seed=0, precisions=(1.0,), config=config
        )
        model.restore_parameters(initial)
    return CoupledLearner(
        model,
        learning_rate=0.1,
        inference_steps=4,
        inference_learning_rate=0.2,
        inference_policy=None,
    )
