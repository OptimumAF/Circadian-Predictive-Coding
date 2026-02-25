# Module: `src/app`

## Responsibilities

- Orchestrate end-to-end experiment flow
- Build comparable reports across learning approaches
- Run in-depth aggregate comparisons across seeds and dataset noise levels

## Inputs / Outputs

- Inputs: `ExperimentConfig`, optional adaptation policy
- Outputs: `ExperimentResult` and `InDepthComparisonResult` with aggregate metrics

## Non-Responsibilities

- Low-level math routines
- CLI argument parsing
- Environment variable parsing
