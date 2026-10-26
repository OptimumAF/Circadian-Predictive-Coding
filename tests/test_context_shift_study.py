"""Fixed study procedure with scalar data/stubs only; no real models/teachers."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scripts import context_shift_study as study
from src.app.coupled_attribution import AttributionBatch


def packets():
    def batch(n, start):
        return AttributionBatch(
            "train",
            "fixture",
            tuple(f"{n}-{r}" for r in range(n)),
            np.arange(start, start + n * 9, dtype=np.float64).reshape(n, 9),
            (np.arange(n, dtype=np.float64) % 6).reshape(n, 1),
            num_classes=6,
        )

    return batch(3, 0), batch(6, 100)


def test_should_fix_equal_exposure_unique_ids_and_exact_recurrence():
    original = packets()
    result = study.build_conditions(original)
    assert tuple(result) == study.CONDITIONS
    ids: set[str] = set()
    for name, batches in result.items():
        assert len(batches) == 12
        for i, b in enumerate(batches):
            assert b.role == "train" and b.num_classes == 6
            assert b.features.shape == (6, 9) and b.targets.shape == (6, 1)
            assert not ids.intersection(b.row_ids)
            ids.update(b.row_ids)
            expected = "legacy-B" if name == "recurring" and 4 <= i < 8 else "old-A"
            assert b.context == expected
    assert len(ids) == 216
    b = result["recurring"][4]
    np.testing.assert_array_equal(b.features[:3], original[0].features)
    np.testing.assert_array_equal(b.features[3:], original[0].features)
    np.testing.assert_array_equal(b.targets[:3], b.targets[3:])


def test_should_corrupt_only_first_label_in_every_stationary_noise_packet():
    result = study.build_conditions(packets())
    for clean, noisy in zip(result["stationary"], result["stationary_label_noise"]):
        np.testing.assert_array_equal(clean.features, noisy.features)
        assert noisy.targets[0, 0] == (clean.targets[0, 0] + 1) % 6
        np.testing.assert_array_equal(clean.targets[1:], noisy.targets[1:])


def test_should_detach_every_packet_from_inputs_and_other_exposures():
    original = packets()
    result = study.build_conditions(original)
    before = original[1].features.copy()
    result["stationary"][0].features[:] = -99
    result["stationary_label_noise"][0].targets[:] = -99
    np.testing.assert_array_equal(original[1].features, before)
    np.testing.assert_array_equal(result["stationary"][1].features, before)
    assert result["stationary_label_noise"][1].targets[0, 0] == 1
    assert original[1].targets[0, 0] == 0


@pytest.mark.parametrize(
    "which,field,value",
    [
        (0, "features", np.zeros((6, 9))),
        (1, "features", np.zeros((6, 8))),
        (0, "targets", np.zeros((3, 2))),
        (1, "targets", np.zeros((5, 1))),
        (0, "num_classes", 2),
        (1, "num_classes", 3),
    ],
)
def test_should_refuse_replaced_original_packet_shapes(which, field, value):
    data = list(packets())
    data[which] = replace(data[which], **{field: value})
    with pytest.raises(ValueError, match="original"):
        study.build_conditions(data)


def test_should_finish_all_conditions_before_any_fixed_task_readout(monkeypatch):
    events = []

    def train(initial, packets, sampler, require, row):
        events.append(("train", row["condition"]))
        return {k: (row["condition"], k) for k in study.ENDPOINTS}

    def read(saved, spec, require, row):
        events.append(("read", row["condition"]))
        assert tuple(saved) == (0, 4, 8, 12)

    monkeypatch.setattr(study, "train_condition", train)
    monkeypatch.setattr(study, "read_tasks", read)
    report: dict[str, Any] = {"cases": []}
    study.run_conditions(None, dict.fromkeys(study.CONDITIONS), None, None, lambda: None, report)
    assert events == [(phase, name) for phase in ("train", "read") for name in study.CONDITIONS]
    assert report["all_training_complete_before_task_readouts"] is True


@pytest.mark.parametrize("failure", study.CONDITIONS)
def test_should_preserve_failed_training_prefix_and_admit_no_task_readouts(monkeypatch, failure):
    events = []

    def train(initial, packets, sampler, require, row):
        events.append(row["condition"])
        row["paid_attempt"] = True
        if row["condition"] == failure:
            raise ValueError("paid failure")
        return {}

    def forbidden(*args):
        pytest.fail("readout admitted after failed training")

    monkeypatch.setattr(study, "train_condition", train)
    monkeypatch.setattr(study, "read_tasks", forbidden)
    report: dict[str, Any] = {"cases": []}
    with pytest.raises(ValueError, match="paid failure"):
        study.run_conditions(
            None, dict.fromkeys(study.CONDITIONS), None, None, lambda: None, report
        )
    assert events == list(study.CONDITIONS[: study.CONDITIONS.index(failure) + 1])
    assert report["cases"][-1]["paid_attempt"] is True
    assert "all_training_complete_before_task_readouts" not in report
