"""One cold actual-Tensor qualification, zero updates and no games."""

from copy import deepcopy
from dataclasses import asdict
import json
import pickle

import numpy as np

from scripts.run_minigrid_linear_fit_control import read_training_sources
from src.adapters import minigrid_public_cue_health as health
from src.adapters.minigrid_public_cue_recurrent_learner import PublicCueRecurrentLearner
from src.adapters.minigrid_public_cue_recurrent_pilot import _typed_sources
from src.adapters.torch_activation_diagnostics import read_native_activations
from src.app.coupled_attribution import _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learnability_diagnostics import diagnose_activations
from src.infra.minigrid_affine_example import example_affine_training
from src.shared.torch_runtime import require_torch


def test_should_match_existing_native_health_and_preserve_cold_state_rng_and_budget(
    record_property,
):
    torch = require_torch()
    torch_rng = torch.get_rng_state().clone()
    numpy_rng = pickle.dumps(np.random.get_state())
    with example_affine_training() as path:
        episodes = read_training_sources(
            path,
            raw_frames=True,
            expected_sha256="5a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508",
        )
    packets = _typed_sources(episodes)
    signatures = health.shared._signature(packets)
    source = PublicCueRecurrentLearner()
    before = source.snapshot_state()
    saved, original = deepcopy(before), deepcopy(before)
    limits = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=0,
            max_replay_examples=0,
            max_wall_seconds=200,
            max_process_rss_bytes=768 * 1024**2,
        ),
        lambda: 0.0,
    )
    born, config, clock = limits.started_at, limits.budget, limits.clock

    def check():
        assert _same_value(source.snapshot_state(), before)
        assert _same_value(saved, original)
        assert health.shared._check_signature(packets, signatures)
        assert torch.equal(torch.get_rng_state(), torch_rng)
        assert pickle.dumps(np.random.get_state()) == numpy_rng
        assert limits.started_at == born and limits.budget is config and limits.clock is clock
        assert limits.updates_completed == limits.replay_examples_completed == 0

    report = health.read_checkpoint_health(
        source, saved, updates=0, sources=packets, budget=limits, check_boundary=check
    )
    assert report["status"] == "complete"
    for index, packet in enumerate(packets):
        actual = read_native_activations(deepcopy(saved.heads[index % 2]), packet.features.copy())
        expected = [asdict(value) for value in diagnose_activations((actual,))]
        assert report["scores"][index]["metrics"] == expected
        assert actual.shape == (9, 8) and actual.dtype == np.float64
    check()
    assert report["work"]["activation_reads_completed"] == 6
    assert report["work"]["forward_calls_completed"] == 54
    assert report["work"]["optimizer_steps"] == report["work"]["published_updates"] == 0
    record_property("cold_TRAIN_health", json.dumps(report, separators=(",", ":"), allow_nan=False))
