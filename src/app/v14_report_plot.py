"""Render fixed descriptive v14 metric ranges as standalone PNG figures.

Inputs are verified table rows and a declared metric. Output is PNG bytes.
This module does not read files, select cells, estimate uncertainty, or
change recorded metrics. The plotted bar spans observed minimum–maximum.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw, ImageFont


_WIDTH = 1180
_HEIGHT = 730
_PLOT_LEFT = 430
_PLOT_RIGHT = 1080
_FIRST_ROW_Y = 188
_ROW_STEP = 52
_METHOD_COLORS = {
    "backprop": "#37617a",
    "predictive_coding": "#715f9a",
    "circadian_predictive_coding": "#b55d33",
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _method_label(method: str) -> str:
    return {
        "backprop": "Backprop",
        "predictive_coding": "Predictive coding",
        "circadian_predictive_coding": "Circadian PC",
    }.get(method, method.replace("_", " ").title())


def _axis_domain(rows: list[dict[str, Any]], metric: str) -> tuple[float, float]:
    default_low, default_high = (-1.0, 1.0) if metric == "signed_forgetting" else (0.0, 1.0)
    low = min(default_low, *(row[metric]["min"] for row in rows))
    high = max(default_high, *(row[metric]["max"] for row in rows))
    return low, high


def render_metric_plot(
    rows: list[dict[str, Any]], *, metric: str, title: str, seed_count: int
) -> bytes:
    """Plot every arm/method mean and observed min–max without ranking."""
    low, high = _axis_domain(rows, metric)
    scale = (_PLOT_RIGHT - _PLOT_LEFT) / (high - low)

    def x(value: float) -> int:
        return round(_PLOT_LEFT + (value - low) * scale)

    image = Image.new("RGB", (_WIDTH, _HEIGHT), "#fbfaf7")
    draw = ImageDraw.Draw(image)
    draw.text((45, 33), title, font=_font(30), fill="#193246")
    draw.text(
        (45, 83),
        f"Observed mean and min-max across {seed_count} seeds · fixed source order",
        font=_font(17),
        fill="#586a75",
    )
    for tick_index in range(5):
        value = low + (high - low) * tick_index / 4
        position = x(value)
        draw.line((position, 146, position, 632), fill="#dce4e7", width=2)
        draw.text((position - 18, 646), f"{value:.2g}", font=_font(15), fill="#52636e")
    if low < 0 < high:
        draw.line((x(0.0), 146, x(0.0), 632), fill="#8fa3ae", width=3)
    for index, row in enumerate(rows):
        y = _FIRST_ROW_Y + index * _ROW_STEP
        color = _METHOD_COLORS.get(row["method"], "#37617a")
        label = f"{row['arm'].replace('_', ' ').title()}  /  {_method_label(row['method'])}"
        draw.text((45, y - 12), label, font=_font(16), fill="#263b4b")
        spread = row[metric]
        left, center, right = x(spread["min"]), x(spread["mean"]), x(spread["max"])
        draw.line((left, y, right, y), fill=color, width=5)
        draw.line((left, y - 8, left, y + 8), fill=color, width=3)
        draw.line((right, y - 8, right, y + 8), fill=color, width=3)
        draw.ellipse((center - 7, y - 7, center + 7, y + 7), fill=color, outline="#ffffff", width=2)
        if index in {2, 5}:
            draw.line((45, y + 26, 1080, y + 26), fill="#e6ebed", width=1)
    draw.text(
        (45, 690),
        "Dots: means  ·  bars: observed seed range  ·  no uncertainty interval or winner selected",
        font=_font(15),
        fill="#657580",
    )
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()
