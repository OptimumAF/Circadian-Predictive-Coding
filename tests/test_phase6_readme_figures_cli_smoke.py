"""Read every public README figure output from a paired synthetic fixture.

The three equal summary rows are writer inputs, not benchmark measurements.
"""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from PIL import Image

from scripts import generate_readme_figures as figures


REPOSITORY = Path(__file__).resolve().parents[1]
PROTOCOL = "vision_guard_separated_unmatched_v2"


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite figure value: {value}")


def _paired_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    summary = tmp_path / "synthetic_figure_summary.csv"
    result = tmp_path / "synthetic_figure.json"
    rows = [
        {
            "model_name": name,
            "test_accuracy_mean": 0.4,
            "train_samples_per_second_mean": 100.0,
            "inference_latency_p95_ms_mean": 2.0,
            "balanced_score": 0.25,
        }
        for name in figures.MODEL_ORDER
    ]
    with summary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result.write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL,
                "dataset": {"name": "cifar100"},
                "summary": rows,
                "fixture_only": True,
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return summary, result


def _command(summary: Path, result: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.generate_readme_figures",
        "--summary-csv",
        str(summary),
        "--result-json",
        str(result),
        "--output-dir",
        str(output_dir),
        "--sleep-cycles",
        "2",
        "--start-hidden",
        "8",
        "--splits",
        "1",
        "--prunes",
        "1",
    ]


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_should_read_all_paired_figure_cli_outputs_without_overwrite(tmp_path: Path) -> None:
    summary, result = _paired_synthetic_inputs(tmp_path)
    output_dir = tmp_path / "new-figures"
    command = _command(summary, result, output_dir)

    completed = subprocess.run(
        command, cwd=REPOSITORY, capture_output=True, text=True, timeout=60, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == f"Wrote figures to {output_dir}"
    assert {path.name for path in output_dir.iterdir()} == set(figures.FIGURE_FILENAMES)
    provenance = json.loads(
        (output_dir / "provenance.json").read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite,
    )
    assert provenance == {
        "protocol_id": PROTOCOL,
        "summary_csv": str(summary),
        "summary_sha256": _digest(summary),
        "result_json": str(result),
        "result_sha256": _digest(result),
        "figures": list(figures.FIGURE_FILENAMES[:-1]),
        "sleep_gif_status": "illustrative, not observed training telemetry",
    }
    assert set(provenance["figures"]) == set(figures.FIGURE_FILENAMES) - {"provenance.json"}

    pngs = [name for name in figures.FIGURE_FILENAMES if name.endswith(".png")]
    htmls = [name for name in figures.FIGURE_FILENAMES if name.endswith(".html")]
    assert len(pngs) == len(htmls) == 4
    for name in pngs:
        with Image.open(output_dir / name) as image:
            assert image.format == "PNG"
            assert image.width >= 300 and image.height >= 200
            image.verify()
    for name in htmls:
        html = (output_dir / name).read_text(encoding="utf-8")
        assert "Plotly.newPlot" in html
        assert all(label in html for label in figures.MODEL_LABELS.values())
    assert "0.4" in (output_dir / "interactive_benchmark_accuracy.html").read_text(encoding="utf-8")
    assert "100" in (output_dir / "interactive_benchmark_train_speed.html").read_text(
        encoding="utf-8"
    )
    with Image.open(output_dir / "circadian_sleep_dynamics.gif") as animation:
        assert animation.format == "GIF"
        assert getattr(animation, "n_frames") == 3  # initial state and two illustrative cycles

    before = {name: _digest(output_dir / name) for name in figures.FIGURE_FILENAMES}
    occupied = subprocess.run(
        command, cwd=REPOSITORY, capture_output=True, text=True, timeout=15, check=False
    )
    assert occupied.returncode != 0
    assert "Figure output already exists" in occupied.stderr
    assert before == {name: _digest(output_dir / name) for name in before}


def test_should_reject_nonfinite_paired_metrics_before_rendering(tmp_path: Path) -> None:
    summary, result = _paired_synthetic_inputs(tmp_path)
    with summary.open(encoding="utf-8", newline="") as stream:
        rows: list[dict[str, Any]] = list(csv.DictReader(stream))
    rows[0]["train_samples_per_second_mean"] = "inf"
    with summary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["summary"][0]["train_samples_per_second_mean"] = float("inf")
    result.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "bad-figures"

    completed = subprocess.run(
        _command(summary, result, output_dir),
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode != 0
    assert "finite" in completed.stderr.lower()
    assert not output_dir.exists()
