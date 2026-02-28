# Module: `src/core`

## Responsibilities

- Define model behavior (`BackpropMLP`, `PredictiveCodingNetwork`, `CircadianPredictiveCodingNetwork`)
- Define ResNet-50 benchmark variants (`BackpropResNet50Classifier`, `PredictiveCodingResNet50Classifier`, `CircadianPredictiveCodingResNet50Classifier`)
- Implement circadian mechanisms such as chemical gating, reward-modulated wake updates, and adaptive sleep budgeting
- Provide activation utilities
- Define neuron adaptation interfaces and traffic summaries

## Inputs / Outputs

- Inputs: numeric arrays, model hyperparameters
- Outputs: model predictions, train-step metrics, traffic summaries

## Non-Responsibilities

- CLI handling
- Environment configuration
- Dataset generation and external IO
