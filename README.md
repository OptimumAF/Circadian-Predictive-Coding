# Predictive Coding vs Backprop (Baseline)

This project compares two learning approaches on the same toy binary-classification task:
- A traditional one-hidden-layer neural network trained with backpropagation
- A predictive-coding style network with iterative hidden-state inference

It also includes a scaffold for future neuron creation/deletion policies based on neuron traffic.
It now also includes a circadian predictive-coding variant with chemical buildup and sleep consolidation.

## What It Does

- Builds a synthetic 2D dataset with two noisy clusters.
- Trains both models using the same train/test split.
- Reports learning progress (`loss` for backprop, `energy` for predictive coding) and test accuracy.
- Records hidden-layer traffic (mean absolute activation) for future structural adaptation policies.

## Project Structure

```text
src/
  core/       # Pure model logic and adaptation policy interface
  app/        # Experiment orchestration
  adapters/   # CLI adapter
  infra/      # Dataset generation
  config/     # Environment-backed settings
  shared/     # Cross-cutting helpers (small only)
tests/        # Unit and integration tests
docs/
  adr/        # Architecture decision records
  modules/    # Module-level responsibilities
scripts/      # Utility scripts (reserved)
```

## Requirements

- Python 3.11+

## Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How To Run

```powershell
python predictive_coding_experiment.py
```

Optional CLI arguments:

```powershell
python predictive_coding_experiment.py --samples 500 --epochs 200 --hidden-dim 16 --noise 0.75 --seed 7
```

## How To Test

```powershell
pytest -q
```

## Lint / Format / Type Check

This repo keeps commands explicit even if tools are not yet enforced in CI:

```powershell
ruff check .
ruff format .
mypy src tests
```

## Configuration

Environment variables are optional and documented in `.env.example`:

- `PC_BASE_SEED`
- `PC_DATASET_SIZE`
- `PC_EPOCHS`

## Common Workflows

1. Run baseline with defaults: `python predictive_coding_experiment.py`
2. Stress test with more epochs: `python predictive_coding_experiment.py --epochs 300`
3. Run tests before changes: `pytest -q`
4. Add a new neuron adaptation policy in `src/core/neuron_adaptation.py` and wire it into `run_experiment`.
5. Explore circadian behavior with `CircadianPredictiveCodingNetwork` in `src/core/circadian_predictive_coding.py`.

## Circadian Model Example

```python
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.infra.datasets import generate_two_cluster_dataset

dataset = generate_two_cluster_dataset(sample_count=300, noise_scale=0.8, seed=7)
model = CircadianPredictiveCodingNetwork(input_dim=2, hidden_dim=10, seed=13)

for _ in range(120):
    model.train_epoch(
        input_batch=dataset.train_input,
        target_batch=dataset.train_target,
        learning_rate=0.05,
        inference_steps=20,
        inference_learning_rate=0.2,
    )

# Sleep uses chemical buildup to split/prune neurons.
sleep_report = model.sleep_event()
print(sleep_report)
```
