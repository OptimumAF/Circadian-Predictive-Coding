"""Export only the pinned historical P6.10 presentation, never scientific readers.

Run with a Python containing optional ReportLab (the Codex bundled runtime is
supported). The project environment/dependencies stay unchanged. Pure projection
lives in app; file IO and optional graphics belong to this outer CLI boundary.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

from src.app.historical_outcome_view import build_historical_view, numeric_extent


SOURCE_BYTES = 39_777_631
SOURCE_SHA256 = "73f5892ef7d152a9a1ada89d59f877483e720026ca160a7afb146c53f4a7f4e8"
SOURCE = Path("artifacts/runs/p610-outcome-cost-pure/outcome-costs.result.json")
PAGE_WIDTH, PAGE_HEIGHT = 1800, 1120
COSTS = (
    "executed_optimizer_updates",
    "optimizer_latent_iterations",
    "applied_replay_updates",
    "rejected_replay_updates",
    "parameters_peak",
    "owned_array_bytes_after_b",
)
CELL_METRICS = (
    "a_after_a",
    "a_after_b",
    "b_after_b",
    "final_mean_task_accuracy",
    "signed_forgetting_a",
    "retention_ratio_a",
)
NOTICE = "HISTORICAL ARTIFACT RENDERING - no new experiment or protocol reproduction"


def read_pinned_body(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Bound read and exact whole identity; arbitrary/partial results are refused."""
    with path.open("rb") as stream:
        raw = stream.read(SOURCE_BYTES + 1)
    if len(raw) != SOURCE_BYTES or sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("historical source whole bytes/SHA256 differ from accepted P6.10 body")
    return raw, json.loads(raw)


def _json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _write(path: Path, data: bytes) -> dict[str, Any]:
    with path.open("xb") as stream:
        stream.write(data)
    return {"bytes": len(data), "sha256": sha256(data).hexdigest()}


def _series(family: dict[str, Any], kind: str, metric: str) -> list[dict[str, Any]]:
    return [
        {
            "label": item["name"] if kind == "arms" else item["left"] + " - " + item["right"],
            **next(value for value in item["metrics"] if value["metric"] == metric),
        }
        for item in family[kind]
    ]


def _cost_series(view: dict[str, Any], family: dict[str, Any], name: str) -> list[dict[str, Any]]:
    series = []
    for arm in family["arms"]:
        rows = [
            r for r in view["rows"] if r["family"] == family["name"] and r["arm"] == arm["name"]
        ]
        observations = []
        for row in rows:
            field = (
                {
                    "value": row["owned_retention"]["checkpoints"]["after_b"]["owned_array_bytes"],
                    "reason": row["owned_retention"]["checkpoints"]["after_b"].get(
                        "retention_status"
                    ),
                }
                if name == "owned_array_bytes_after_b"
                else row["resource_fields"][name]
            )
            observations.append(
                {"seed": row["seed"], "value": field["value"], "reason": field.get("reason")}
            )
        series.append(
            {
                "label": arm["name"],
                "summary": {
                    "observations": observations,
                    "mean": None,
                    "marginal_interval": None,
                    "simultaneous_interval": None,
                },
            }
        )
    return series


def _panel(
    drawing: Any, x: float, y: float, title: str, series: list[dict[str, Any]], contrast: bool
) -> dict[str, Any]:
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import String
    from reportlab.graphics.widgets.markers import makeMarker
    from reportlab.lib import colors

    selected = (
        "simultaneous_interval"
        if contrast and series[0].get("primary_endpoint")
        else "marginal_interval"
    )
    values: list[float | int | None] = []
    points: list[tuple[float, float]] = []
    means: list[tuple[float, float]] = []
    intervals: list[list[tuple[float, float]]] = []
    for index, item in enumerate(series):
        row_y = len(series) - 1 - index
        summary = item["summary"]
        observations = summary["observations"]
        values.extend(o["value"] for o in observations)
        known = sum(o["value"] is not None for o in observations)
        drawing.add(
            String(
                x,
                y + 42 + (row_y + 0.5) * 360 / max(len(series), 1),
                item["label"] + f"  [{known}/{len(observations)}]",
                fontName="Helvetica",
                fontSize=8,
            )
        )
        for seed_index, observation in enumerate(observations):
            if observation["value"] is not None:
                points.append(
                    (
                        observation["value"],
                        row_y + (seed_index - (len(observations) - 1) / 2) * 0.035,
                    )
                )
        if summary["mean"] is not None:
            values.append(summary["mean"])
            means.append((summary["mean"], row_y))
        interval = summary[selected]
        if interval is not None:
            values.extend((interval["lower"], interval["upper"]))
            intervals.append([(interval["lower"], row_y), (interval["upper"], row_y)])
    measured = [v for v in values if v is not None]
    low, high = numeric_extent(values) if measured else (-1.0, 1.0)
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = x + 235, y + 45, 310, 360
    plot.data = intervals + ([points] if points else []) + ([means] if means else [])
    if not plot.data:
        drawing.add(String(x + 235, y + 200, "All values unmeasured", fontSize=10))
    else:
        plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = low, high
        plot.xValueAxis.valueSteps = [low + (high - low) * n / 4 for n in range(5)]
        plot.xValueAxis.labelTextFormat = "%0.3g"
        plot.xValueAxis.labels.fontSize = 8
        plot.yValueAxis.valueMin, plot.yValueAxis.valueMax = -0.5, len(series) - 0.5
        plot.yValueAxis.visible = False
        for n in range(len(intervals)):
            plot.lines[n].strokeColor = colors.HexColor("#1b4f72")
            plot.lines[n].strokeWidth = 2
        for n, marker_size, color in (
            (len(intervals), 2.5, "#8294a0"),
            (len(intervals) + bool(points), 5, "#132f45"),
        ):
            if n < len(plot.data):
                plot.lines[n].strokeColor = None
                plot.lines[n].symbol = makeMarker(
                    "FilledCircle",
                    size=marker_size,
                    fillColor=colors.HexColor(color),
                    strokeColor=None,
                )
        drawing.add(plot)
    drawing.add(String(x, y + 440, title, fontName="Helvetica-Bold", fontSize=12))
    return {"metric": title, "interval": selected, "extent": [low, high], "series": series}


def _page(view: dict[str, Any], family: dict[str, Any], kind: str) -> tuple[Any, dict[str, Any]]:
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.lib import colors

    drawing = Drawing(PAGE_WIDTH, PAGE_HEIGHT)
    drawing.add(Rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fillColor=colors.white, strokeColor=None))
    drawing.add(
        String(28, 1075, family["name"] + " / " + kind, fontName="Helvetica-Bold", fontSize=24)
    )
    drawing.add(String(28, 1046, NOTICE, fontSize=13))
    metrics = COSTS if kind == "costs" else tuple(m["metric"] for m in family[kind][0]["metrics"])
    panels = []
    for n, metric in enumerate(metrics):
        series = (
            _cost_series(view, family, metric) if kind == "costs" else _series(family, kind, metric)
        )
        panels.append(
            _panel(
                drawing, 28 + n % 3 * 580, 550 - n // 3 * 485, metric, series, kind == "contrasts"
            )
        )
    lines = (
        "Dots: every original seed; bold dot: original mean; line: original marginal 95% Student-t interval. Retention is descriptive only.",
        "Contrasts: left minus right; primary lines: original Bonferroni 116-statement intervals; secondary lines: original marginal intervals.",
        "Costs: dots are recorded/derived work or separate after-B owned arrays, not CPU/FLOPs/RSS. All costs, stages and null reasons are in view.json.",
        "Axes include every point and original interval endpoint; [n/N] preserves missing counts. No repeats counted as independent replications.",
        "H1-H4 remain unresolved. Current G0/guard/source/scientific admission is open; historical bytes are not new protocol reproduction.",
    )
    for n, line in enumerate(lines):
        drawing.add(String(28, 72 - n * 13, line, fontSize=10))
    return drawing, {"family": family["name"], "kind": kind, "panels": panels}


def _markdown(view: dict[str, Any], pages: list[dict[str, Any]]) -> str:
    lines = [
        "# Historical confirmation outcomes, uncertainty and costs",
        "",
        NOTICE,
        "",
        "All six families, 560 cells, six cell metrics and 626 original arm/paired metric vectors are retained. No winner or new estimator. H1-H4 remain unresolved; the reopened owning-with correctness/G0 gates do not grant new source or scientific authority.",
        "",
        "Gray dots are all original seeds; bold dots are original means. Arm and secondary paired lines are marginal 95% Student-t intervals. Primary paired contrasts use the original 116-statement Bonferroni model-based intervals (left minus right). Constant/missing vectors keep absent intervals; retention stays descriptive. Independence/normality are original model assumptions, not established by this rendering. Repeats add no replications.",
        "",
        "Negative forgetting, above-one retention and both null metric reasons remain. Metrics are not clipped to [0,1]. Display labels may be rounded; view.json retains exact source numbers and the complete analysis contract/records, all 12,880 resource fields and 25,400 parameter-history points. Original state proofs remain in the whole pinned input.",
        "",
        "Optimizer work includes rolled-back replay; latent loops exclude prediction/guard/selection work. Owned arrays, one shared FIFO per context, Python overhead, checkpoint copies and sampled whole-process RSS have different scopes. Stage snapshots are not additive RAM. Per-arm wall time/RSS/guard duration and continuous parameter history remain unmeasured. Four original process segments, including repeats, remain separate in view.json.",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- [{p['family']} / {p['kind']}]({p['filename']})" for p in pages)
    lines.extend(
        [
            "",
            "## Separate original stage storage totals",
            "",
            "```json",
            json.dumps(view["stage_storage_totals"], indent=2),
            "```",
            "",
            "## All cell outcomes and selected cost columns",
            "",
            "Other cost fields, reasons and original interval/null status are fully retained in view.json.",
            "",
            "| Family | Seed | Arm | A after A | A after B | B after B | Final mean | Signed forgetting | Retention | Executed updates | Latent loops | Rolled back updates | Peak parameters | Owned B bytes | Failure |",
            "|" + "---|" * 15,
        ]
    )
    for row in view["rows"]:
        metrics = [
            str(v["value"]) if v["value"] is not None else "null: " + str(v["reason"])
            for v in (row["metrics"][name] for name in CELL_METRICS)
        ]
        costs = [
            row["resource_fields"][n]["value"]
            for n in (
                "executed_optimizer_updates",
                "optimizer_latent_iterations",
                "rejected_replay_updates",
                "parameters_peak",
            )
        ]
        values = [
            row["family"],
            row["seed"],
            row["arm"],
            *metrics,
            *costs,
            row["owned_retention"]["checkpoints"]["after_b"]["owned_array_bytes"],
            row["outcome"]["failure"],
        ]
        lines.append(
            "| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in values) + " |"
        )
    return "\n".join(lines) + "\n"


def export_figures(source: Path, output: Path) -> dict[str, Any]:
    raw, body = read_pinned_body(source)
    view = build_historical_view(body)
    from reportlab.graphics import renderPDF, renderSVG
    from reportlab.pdfgen.canvas import Canvas

    output.mkdir(parents=True, exist_ok=False)
    files, pages = {}, []
    pdf_path = output / "historical-outcomes-costs.pdf"
    canvas = Canvas(str(pdf_path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT), invariant=1)
    canvas.setTitle(NOTICE)
    for family in view["original_report_records"]["analysis"]["families"]:
        for kind in ("arms", "contrasts", "costs"):
            drawing, page = _page(view, family, kind)
            filename = family["name"] + "-" + kind + ".svg"
            files[filename] = _write(output / filename, renderSVG.drawToString(drawing).encode())
            renderPDF.draw(drawing, canvas, 0, 0)
            canvas.showPage()
            pages.append({**page, "filename": filename})
    canvas.save()
    files[pdf_path.name] = {
        "bytes": pdf_path.stat().st_size,
        "sha256": sha256(pdf_path.read_bytes()).hexdigest(),
    }
    files["view.json"] = _write(output / "view.json", _json(view))
    files["figures.json"] = _write(output / "figures.json", _json(pages))
    files["report.md"] = _write(output / "report.md", _markdown(view, pages).encode())
    if source.read_bytes() != raw:
        raise ValueError(
            "historical source changed during export; retain partial outputs, do not accept"
        )
    manifest = {
        "notice": NOTICE,
        "source": str(source.resolve()),
        "source_identity": {"bytes": len(raw), "sha256": SOURCE_SHA256},
        "coverage": view["coverage"],
        "python": sys.version,
        "reportlab": importlib.metadata.version("reportlab"),
        "new_experiment_or_reader_calls": 0,
        "files": files,
    }
    _write(output / "manifest.json", _json(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True, help="new, unoccupied directory")
    args = parser.parse_args()
    try:
        manifest = export_figures(args.source, args.output_dir)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f"historical figure export refused: {error}\n")
    print(json.dumps({"output": str(args.output_dir), "coverage": manifest["coverage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
