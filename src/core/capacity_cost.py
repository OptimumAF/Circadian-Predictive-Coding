"""Dense binary parameter counts and declared stored-array capacity costs.

These immutable counts exclude temporary allocations, metadata, process memory,
FLOPs and representational quality. No models, measurements or budgets live here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityUsage:
    parameters: int
    parameter_bytes: int
    auxiliary_bytes: int

    def __post_init__(self) -> None:
        if type(self) is not CapacityUsage:
            raise ValueError("capacity usage requires its exact record type")
        for name in ("parameters", "parameter_bytes", "auxiliary_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"capacity {name} must be a native nonnegative integer")
        if self.parameters == 0 or self.parameter_bytes != 8 * self.parameters:
            raise ValueError("capacity requires positive float64 parameter counts and exact bytes")

    @property
    def total_array_bytes(self) -> int:
        return self.parameter_bytes + self.auxiliary_bytes


def count_dense_parameters(input_dim: int, widths: tuple[int, ...]) -> int:
    """Count all weights and biases for tanh hidden layers and one binary output."""
    if type(input_dim) is not int or input_dim < 1:
        raise ValueError("dense input dimension must be a positive native integer")
    if (
        type(widths) is not tuple
        or not widths
        or any(type(width) is not int or width < 1 for width in widths)
    ):
        raise ValueError("dense hidden widths must be a nonempty positive native integer tuple")
    return sum((lower + 1) * upper for lower, upper in zip((input_dim, *widths), (*widths, 1)))
