# Module: `src/app`

## Responsibilities

- Orchestrate end-to-end experiment flow
- Build comparable reports across learning approaches
- Run in-depth aggregate comparisons across seeds and dataset noise levels
- Run ResNet-50 speed/latency benchmarks with target-accuracy and circadian sleep metrics

## Inputs / Outputs

- Inputs: `ExperimentConfig`, optional adaptation policy
- Outputs: `ExperimentResult`, `InDepthComparisonResult`, and `ResNet50BenchmarkResult`

## Non-Responsibilities

- Low-level math routines
- CLI argument parsing
- Environment variable parsing
