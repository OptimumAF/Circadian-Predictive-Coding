# Module: `src/core`

## Responsibilities

- Define model behavior (`BackpropMLP`, `PredictiveCodingNetwork`)
- Define model behavior (`BackpropMLP`, `PredictiveCodingNetwork`, `CircadianPredictiveCodingNetwork`)
- Provide activation utilities
- Define neuron adaptation interfaces and traffic summaries

## Inputs / Outputs

- Inputs: numeric arrays, model hyperparameters
- Outputs: model predictions, train-step metrics, traffic summaries

## Non-Responsibilities

- CLI handling
- Environment configuration
- Dataset generation and external IO
