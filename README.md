# Predictive Coding vs Backprop (Baseline)

This project compares three learning approaches on the same toy binary-classification task:
- A traditional one-hidden-layer neural network trained with backpropagation
- A predictive-coding style network with iterative hidden-state inference
- A circadian predictive-coding network with chemical state and sleep-time split/prune events

It also includes a scaffold for future neuron creation/deletion policies based on neuron traffic.
It now also includes a circadian predictive-coding variant with chemical buildup and sleep consolidation.

## What It Does

- Builds a synthetic 2D dataset with two noisy clusters.
- Trains all three models using the same train/test split.
- Reports learning progress (`loss` for backprop, `energy` for predictive coding variants) and test accuracy.
- Records hidden-layer traffic (mean absolute activation) for future structural adaptation policies.
- Supports in-depth aggregate comparisons across multiple seeds and noise levels.

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
- Optional for ResNet-50 benchmark: PyTorch + torchvision

## Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional ResNet benchmark dependencies:

```powershell
pip install -r requirements-resnet.txt
```

For NVIDIA GPU acceleration (recommended), install CUDA wheels in your venv:

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## How To Run

```powershell
python predictive_coding_experiment.py
```

Optional CLI arguments:

```powershell
python predictive_coding_experiment.py --samples 500 --epochs 200 --hidden-dim 16 --noise 0.75 --seed 7
```

In-depth multi-run comparison:

```powershell
python predictive_coding_experiment.py --mode indepth --samples 400 --epochs 160 --noise-levels 0.6,0.8,1.0 --seed-list 3,7,11,19,23 --sleep-interval 40
```

ResNet-50 speed benchmark:

```powershell
python resnet50_benchmark.py --train-samples 2000 --test-samples 500 --epochs 8 --batch-size 32 --image-size 96 --target-accuracy 0.99 --device auto
```

Tuned circadian run (GPU) with both split and prune while reaching high accuracy:

```powershell
python resnet50_benchmark.py --device cuda --backbone-weights imagenet --train-samples 2000 --test-samples 500 --classes 10 --image-size 96 --batch-size 64 --epochs 12 --target-accuracy 0.995 --backprop-freeze-backbone --pc-hidden-dim 256 --pc-lr 0.03 --pc-steps 10 --pc-inference-lr 0.15 --circ-hidden-dim 256 --circ-lr 0.03 --circ-steps 10 --circ-inference-lr 0.15 --circ-sleep-interval 2 --circ-min-hidden-dim 96 --circ-max-hidden-dim 768 --circ-split-threshold 0.8 --circ-prune-threshold 0.65 --circ-max-split-per-sleep 2 --circ-max-prune-per-sleep 2
```

ResNet-50 benchmark focuses on:
- training speed (`samples/s`, `ms/step`)
- inference speed (mean and p95 latency, inference `samples/s`)
- time/epochs to target accuracy (early stop)
- circadian sleep dynamics (split/prune totals and hidden size change)
- constant model comparison: each run always includes traditional backprop, traditional predictive coding, and circadian predictive coding

Harder benchmark mode (recommended to avoid trivial 1.0 accuracy):

```powershell
python resnet50_benchmark.py --device cuda --backbone-weights imagenet --train-samples 2000 --test-samples 500 --classes 10 --image-size 96 --batch-size 64 --dataset-difficulty medium --dataset-noise-std 0.07 --epochs 12 --target-accuracy -1 --backprop-freeze-backbone --pc-hidden-dim 256 --pc-lr 0.03 --pc-steps 10 --pc-inference-lr 0.15 --circ-hidden-dim 256 --circ-lr 0.03 --circ-steps 10 --circ-inference-lr 0.15 --circ-sleep-interval 2 --circ-min-hidden-dim 96 --circ-max-hidden-dim 768 --circ-split-threshold 0.8 --circ-prune-threshold 0.65 --circ-max-split-per-sleep 2 --circ-max-prune-per-sleep 2
```

Multi-seed CIFAR benchmark with JSON and CSV exports:

```powershell
python scripts/run_multiseed_resnet_benchmark.py --dataset-name cifar100 --dataset-train-subset-size 20000 --dataset-test-subset-size 5000 --epochs 12 --device cuda --output-prefix benchmark_multiseed_cifar100
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

Circadian model supports optional advanced mechanisms through `CircadianConfig`:
- adaptive split/prune thresholds via chemical percentiles
- adaptive sleep triggering (energy plateau + chemical variance)
- weight-norm-aware split/prune ranking
- gradient-importance-aware split/prune ranking
- split/prune hysteresis and per-neuron cooldown
- optional dual-timescale chemical dynamics (fast + slow accumulation)
- optional saturating chemical accumulation with configurable maximum
- adaptive per-neuron plasticity sensitivity (age + importance driven)
- sleep-phase scheduling (warm-up, split-first, prune-late)
- per-sleep structural-change caps and prune age floor
- gradual prune decay and delayed removal
- prioritized and class-balanced replay consolidation during sleep
- targeted homeostatic norm matching after sleep
- function-preserving split behavior (parent+child outgoing weights sum to original)

Predictive-coding binary training now uses BCE-consistent output gradients (`p - y`) while keeping a hidden-state error penalty in the reported energy metric.
