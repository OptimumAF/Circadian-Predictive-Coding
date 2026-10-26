"""Exercise complete scheduled pending state through ordinary continuation, using fake math."""

from types import SimpleNamespace

import numpy as np

from src.adapters.sensor_drift_demo import run_demo
from src.app.coupled_attribution import AttributionStream, _same_value
from src.core import coupled_energy
from src.core.matched_backprop import MATCHED_BACKPROP_LOSS_ID, MatchedBackpropNetwork


def test_should_resume_two_arrived_buffers_before_the_first_optimizer_step(monkeypatch):
    work = []

    def train(model, features, targets, rate):
        work.append((features.copy(), targets.copy()))
        return SimpleNamespace(loss_definition=MATCHED_BACKPROP_LOSS_ID, loss=0.5)

    monkeypatch.setattr(MatchedBackpropNetwork, "train_epoch", train)
    monkeypatch.setattr(
        coupled_energy, "predict_coupled", lambda parameters, features: np.full((4, 1), 0.5)
    )
    original_run = AttributionStream.run_until
    errors = []

    def pause_then_run(owner, target):
        original_run(owner, 2)
        before = owner._runtime.candidate_snapshot()
        assert len(before.state.buffered_batches) == 2
        try:
            pause = owner.pause()
            owner.resume(pause)
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            raise
        assert _same_value(before, owner._runtime.candidate_snapshot())
        original_run(owner, target)

    monkeypatch.setattr(AttributionStream, "run_until", pause_then_run)
    report = run_demo(scheduled=True, monitoring=True)
    assert report["status"] == "complete", (report["reason"], errors)
    assert report["charged_training_updates"] == 12 and len(work) == 3
