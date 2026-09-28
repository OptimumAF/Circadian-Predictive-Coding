# Module: `src/shared`

## Responsibilities

- Provide minimal cross-cutting runtime helpers used by multiple layers.
- Keep optional dependency loading logic isolated (for example torch/torchvision lazy imports).
- Observe current-process RSS on Windows/Linux for benchmark telemetry without an extra dependency.

## Inputs / Outputs

- Inputs: runtime dependency requests, device references, and a process-memory sampling interval.
- Outputs: loaded modules or clear runtime errors; current RSS and an observed high-water value, or `None` on unsupported hosts.

## Non-Responsibilities

- No model training logic.
- No orchestration of experiment workflows.
- No dataset generation.
- No exact allocation attribution, process-isolated peak guarantee, or GPU memory profiling.
