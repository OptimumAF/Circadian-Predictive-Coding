# Circadian Predictive Coding

Circadian Predictive Coding is a research-focused repository for biologically inspired neural learning with:

- chemical-gated plasticity
- sleep-phase structural adaptation (split/prune)
- direct comparison against traditional backpropagation and traditional predictive coding

The circadian model is the primary focus of this project. Backprop and predictive coding baselines are kept to ensure fair comparison and reproducible evaluation.

## Core Idea

The circadian algorithm models wake and sleep phases:

- Wake: train with predictive-coding updates while each neuron accumulates a chemical usage signal.
- Sleep: consolidate with architecture updates (split high-usage neurons, prune low-usage neurons), optional rollback, and homeostatic controls.

This lets model capacity adapt over time instead of staying fixed.

## Features

- NumPy circadian predictive coding baseline for small-scale experiments
- Torch ResNet-50 benchmark pipeline for speed and accuracy comparisons
- Adaptive sleep triggers, adaptive split/prune thresholds, dual-timescale chemical dynamics
- Function-preserving split behavior and guarded sleep rollback
- Multi-seed benchmark runner with JSON/CSV output

## Repository Layout

```text
src/
  core/       # Learning rules and model definitions
  app/        # Experiment and benchmark orchestration
  adapters/   # CLI entrypoints
  infra/      # Dataset and dataloader construction
  config/     # Environment-backed defaults
  shared/     # Small cross-cutting runtime helpers
tests/        # Unit/integration tests
docs/
  adr/        # Architecture decision records
  modules/    # Module responsibility docs
scripts/      # Reproducible benchmark scripts
```

## Quickstart

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional torch benchmark dependencies:

```powershell
pip install -r requirements-resnet.txt
```

For NVIDIA GPUs (example CUDA wheels):

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Running Experiments

Toy baseline comparison:

```powershell
python predictive_coding_experiment.py
```

In-depth toy comparison (multi-seed, multi-noise):

```powershell
python predictive_coding_experiment.py --mode indepth --samples 400 --epochs 160 --noise-levels 0.6,0.8,1.0 --seed-list 3,7,11,19,23 --sleep-interval 40
```

ResNet-50 benchmark (all 3 models):

```powershell
python resnet50_benchmark.py --dataset-name cifar100 --classes 100 --dataset-train-subset-size 20000 --dataset-test-subset-size 5000 --epochs 12 --device cuda
```

Multi-seed benchmark export:

```powershell
python scripts/run_multiseed_resnet_benchmark.py --dataset-name cifar100 --seeds 7,13,29 --dataset-train-subset-size 20000 --dataset-test-subset-size 5000 --epochs 12 --device cuda --output-prefix benchmark_multiseed_cifar100
```

## Reproducing The Current Strong Circadian Regime

This command runs a harder full CIFAR-100 setup where circadian adaptation is active:

```powershell
python resnet50_benchmark.py --dataset-name cifar100 --classes 100 --dataset-train-subset-size 0 --dataset-test-subset-size 0 --epochs 48 --device cuda --target-accuracy -1 --backbone-weights imagenet --backprop-freeze-backbone --batch-size 64 --image-size 96 --eval-batches 2 --inference-batches 20 --warmup-batches 5 --circ-force-sleep --circ-sleep-interval 2 --circ-enable-sleep-rollback --circ-sleep-rollback-tolerance 0.001 --circ-sleep-rollback-metric cross_entropy --circ-sleep-rollback-eval-batches 2 --circ-min-hidden-dim 320 --circ-max-hidden-dim 768 --circ-prune-min-age-steps 1200 --circ-sleep-warmup-steps 10 --circ-adaptive-split-percentile 82 --circ-adaptive-prune-percentile 8 --circ-split-hysteresis-margin 0.01 --circ-prune-hysteresis-margin 0.02 --circ-max-split-per-sleep 1 --circ-max-prune-per-sleep 1 --circ-sleep-split-only-until-fraction 0.70 --circ-sleep-prune-only-after-fraction 0.97 --circ-sleep-max-change-fraction 0.01 --circ-split-noise-scale 0.002 --circ-homeostatic-downscale-factor 0.999 --circ-homeostasis-target-input-norm 1.2 --circ-homeostasis-target-output-norm 1.0 --circ-homeostasis-strength 0.25 --circ-lr 0.028 --circ-steps 14 --circ-inference-lr 0.12
```

## Quality Commands

```powershell
ruff check .
ruff format .
mypy src tests scripts
pytest -q
```

## Open Source Standards

- License: [MIT](LICENSE)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Governance: [GOVERNANCE.md](GOVERNANCE.md)
- Support process: [SUPPORT.md](SUPPORT.md)

## Citation

If this repository contributes to your work, cite it using [CITATION.cff](CITATION.cff).
