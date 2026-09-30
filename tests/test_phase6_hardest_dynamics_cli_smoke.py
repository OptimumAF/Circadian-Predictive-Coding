"""Bounded public GIF/HTML readback for both dynamics protocols."""

from __future__ import annotations

from ast import literal_eval
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
from PIL import Image

from scripts import generate_hardest_mode_dynamics as dynamics


REPOSITORY = Path(__file__).resolve().parents[1]


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite dynamics value: {value}")


def _payload(html: str) -> dict[str, Any]:
    line = next(line.strip() for line in html.splitlines() if "const payload = " in line)
    record = json.loads(
        line.removeprefix("const payload = ").removesuffix(";"), parse_constant=_reject_nonfinite
    )
    assert isinstance(record, dict)
    return record


@pytest.mark.parametrize(
    ("protocol", "evaluation_split"),
    [
        (dynamics.VALIDATION_PROTOCOL, "validation"),
        (dynamics.LEGACY_PROTOCOL, "test (legacy, intermediate access)"),
    ],
)
def test_should_read_tiny_public_dynamics_gif_and_html(
    protocol: str, evaluation_split: str, tmp_path: Path
) -> None:
    gif = tmp_path / f"{protocol}.gif"
    html_path = tmp_path / f"{protocol}.html"
    command = [
        sys.executable,
        "-m",
        "scripts.generate_hardest_mode_dynamics",
        "--tiny-smoke",
        "--protocol-id",
        protocol,
        "--seed",
        "7",
        "--snapshot-interval",
        "1",
        "--gif-duration-ms",
        "20",
        "--gif-output-path",
        str(gif),
        "--interactive-output-path",
        str(html_path),
    ]

    completed = subprocess.run(
        command, cwd=REPOSITORY, capture_output=True, text=True, timeout=30, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert f"Wrote {gif}" in completed.stdout
    assert f"Wrote {html_path}" in completed.stdout
    assert f"Protocol: {protocol}; intermediate split: {evaluation_split}" in completed.stdout
    fixture_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("Fixture: ")
    )
    assert fixture_line.startswith("Fixture: tiny_smoke_v1 ")
    config = json.loads(
        fixture_line.removeprefix("Fixture: tiny_smoke_v1 "), parse_constant=_reject_nonfinite
    )
    assert config["protocol_id"] == protocol
    assert config["seed"] == 7
    assert config["sample_count_phase_a"] == config["sample_count_phase_b"] == 40
    assert config["phase_a_epochs"] == config["phase_b_epochs"] == 2
    assert config["decision_grid_size"] == 8
    assert config["latency_repeats"] == 1
    assert config["sleep_interval_phase_a"] == config["sleep_interval_phase_b"] == 1
    final_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("Final test accuracy: ")
    )
    final_scores = literal_eval(final_line.removeprefix("Final test accuracy: "))
    assert set(final_scores) == set(dynamics.MODEL_ORDER)
    assert all(isfinite(score) and 0 <= score <= 1 for score in final_scores.values())

    html = html_path.read_text(encoding="utf-8")
    assert "Plotly.react" in html
    payload = _payload(html)
    assert "Tiny smoke fixture: 40 rows per phase" in html
    assert payload["fixture_id"] == "tiny_smoke_v1"
    assert payload["fixture_config"] == config
    assert payload["protocol_id"] == protocol
    assert payload["evaluation_split"] == evaluation_split
    assert payload["final_test_accuracy"] == final_scores
    assert payload["epochs"] == [1, 2, 3, 4]
    assert len(payload["frames"]) == 4
    assert payload["phase_b_start_index"] == 2
    assert set(payload["training_metric_ids"]) == set(dynamics.MODEL_ORDER)
    assert len(payload["phase_b_evaluation_labels"]) > 0
    assert all(len(frame["decision_map"]) == 8 for frame in payload["frames"])
    assert all(
        len(frame["circadian_predictions"]) == len(payload["phase_b_evaluation_labels"])
        for frame in payload["frames"]
    )
    if protocol == dynamics.VALIDATION_PROTOCOL:
        assert len(payload["split_hashes"]) == 6
        assert "phase_b_test_labels" not in payload
    else:
        assert payload["split_hashes"] == {}
        assert "legacy" in html.lower()

    with Image.open(gif) as animation:
        assert animation.format == "GIF"
        assert getattr(animation, "n_frames") == 4
        assert animation.size == (1180, 680)
        assert payload["comparison_scope"]["scope_id"].encode() in animation.info["comment"]

    before = (sha256(gif.read_bytes()).hexdigest(), sha256(html_path.read_bytes()).hexdigest())
    occupied = subprocess.run(
        command, cwd=REPOSITORY, capture_output=True, text=True, timeout=10, check=False
    )
    assert occupied.returncode != 0
    assert "Dynamics output already exists" in occupied.stderr
    assert before == (
        sha256(gif.read_bytes()).hexdigest(),
        sha256(html_path.read_bytes()).hexdigest(),
    )
