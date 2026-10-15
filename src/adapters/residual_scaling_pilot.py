"""Fixed residual scaling composition over the shared isolated grid boundary."""

from typing import Any

from src.adapters.linear_scaling_pilot import (
    _execute_case,
    _execute_grid,
    describe_protocol as describe_mlp_protocol,
)
from src.app.linear_scaling_measurement import run_residual_scaling_case
from src.core.residual_linear_pc import (
    LinearResidualConfig,
    effective_residual_weights,
    initialize_residual_linear,
)
from src.shared.process_memory import ProcessRssSampler


def describe_protocol() -> dict[str, Any]:
    """Keep data/rate/grid fixed; only the declared residual topology changes."""
    return {
        **describe_mlp_protocol(),
        "protocol_id": "residual_linear_scaling_measurements_v1",
        "topology": "fixed_middle_identity",
        "depth_definition": "L=hidden_layers+1 total learned matrices",
        "mean_field_middle_branch_scale": "1/sqrt(width*L)",
        "mean_field_raw_sgd_factor": "gamma0^2*width",
        "reference_author_commit": "18a732e1d495d833cb97cf305f9d6fbca7bdddeb",
        "selection_note": "Fixed prior grid/data/rate; no score feedback or tuning.",
    }


def _run_one(row: dict[str, Any], started: float, sampler: ProcessRssSampler) -> None:
    _execute_case(
        row,
        started,
        sampler,
        lambda scheme, width, hidden, seed: initialize_residual_linear(
            LinearResidualConfig(scheme, 2, width, hidden), seed
        ),
        effective_residual_weights,
        run_residual_scaling_case,
    )
    if "initial_state" in row:
        row["initial_state"].update(
            depth=row["hidden_layers"] + 1,
            identity_layer_indices_zero_based=list(range(1, row["hidden_layers"])),
        )


def run_pilot() -> dict[str, Any]:
    return _execute_grid(_run_one, describe_protocol)
