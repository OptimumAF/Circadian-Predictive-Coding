# Module: `src/shared`

## Responsibilities

- Provide minimal cross-cutting runtime helpers used by multiple layers.
- Keep optional dependency loading logic isolated (for example torch/torchvision lazy imports).

## Inputs / Outputs

- Inputs: runtime dependency requests and device references.
- Outputs: loaded modules or clear runtime errors.

## Non-Responsibilities

- No model training logic.
- No orchestration of experiment workflows.
- No dataset generation.
