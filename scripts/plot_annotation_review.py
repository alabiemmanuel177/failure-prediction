#!/usr/bin/env python3
"""Render a non-model SVG telemetry timeline for independent annotation review."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path

import yaml


PANELS = (
    ("motion", ("command_linear", "measured_linear"), "speed (m/s)"),
    ("localisation health", ("pose_covariance_trace",), "covariance trace"),
    ("LiDAR health", ("valid_return_fraction",), "valid fraction"),
    ("forward clearance", ("minimum_front_range",), "range (m)"),
)
COLORS = {
    "command_linear": "#2455a4",
    "measured_linear": "#2f855a",
    "pose_covariance_trace": "#7c3aed",
    "valid_return_fraction": "#0f766e",
    "minimum_front_range": "#b45309",
}


def finite_series(rows: list[dict[str, str]], feature: str, start: float, end: float):
    values = []
    for row in rows:
        if row["feature"] != feature:
            continue
        timestamp, value = float(row["timestamp"]), float(row["value"])
        if start <= timestamp <= end and math.isfinite(value):
            values.append((timestamp, value))
    if len(values) <= 1200:
        return values
    step = math.ceil(len(values) / 1200)
    return values[::step]


def render(annotation: dict, rows: list[dict[str, str]]) -> str:
    start = float(annotation["episode"]["start_time"])
    end = float(annotation["episode"]["end_time"])
    span = max(end - start, 1e-9)
    width, height = 1120, 820
    left, right, top = 100, 1080, 70
    panel_height, gap = 145, 35
    x = lambda value: left + (value - start) / span * (right - left)
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="20" font-weight="600">'
        f'Annotation review — {html.escape(annotation["run_id"])}</text>',
        f'<text x="{left}" y="50" font-family="sans-serif" font-size="13" fill="#444">'
        f'{html.escape(annotation["episode"]["termination_reason"])} · no model outputs shown</text>',
    ]
    event = next((item for item in annotation.get("events", []) if item.get("terminal")), None)
    onset = next((item.get("actual_onset") for item in annotation.get("injections", [])
                  if item.get("eligible") is True), None)
    for index, (title, features, unit) in enumerate(PANELS):
        y_top = top + index * (panel_height + gap)
        y_bottom = y_top + panel_height
        series = {feature: finite_series(rows, feature, start, end) for feature in features}
        all_values = [value for values in series.values() for _, value in values]
        if all_values:
            low, high = min(all_values), max(all_values)
            if low == high:
                low, high = low - 0.5, high + 0.5
            padding = 0.08 * (high - low)
            low, high = low - padding, high + padding
        else:
            low, high = 0.0, 1.0
        y = lambda value: y_bottom - (value - low) / (high - low) * panel_height
        elements.extend([
            f'<rect x="{left}" y="{y_top}" width="{right-left}" height="{panel_height}" fill="#fafafa" stroke="#aab2bd"/>',
            f'<text x="{left}" y="{y_top-10}" font-family="sans-serif" font-size="14" font-weight="600">{html.escape(title)}</text>',
            f'<text x="{left-12}" y="{y_top+8}" text-anchor="end" font-family="sans-serif" font-size="11">{high:.3g}</text>',
            f'<text x="{left-12}" y="{y_bottom}" text-anchor="end" font-family="sans-serif" font-size="11">{low:.3g}</text>',
            f'<text x="22" y="{(y_top+y_bottom)/2}" text-anchor="middle" font-family="sans-serif" font-size="11" transform="rotate(-90 22 {(y_top+y_bottom)/2})">{html.escape(unit)}</text>',
        ])
        if event:
            event_time = float(event["time"])
            warning_left = max(start, event_time - 10.0)
            warning_right = min(end, event_time - 1.0)
            if warning_right > warning_left:
                elements.append(
                    f'<rect x="{x(warning_left):.2f}" y="{y_top}" width="{x(warning_right)-x(warning_left):.2f}" height="{panel_height}" fill="#3a9d5d" opacity="0.10"/>'
                )
        for feature, values in series.items():
            if not values:
                continue
            points = " ".join(f"{x(t):.2f},{y(value):.2f}" for t, value in values)
            elements.append(
                f'<polyline points="{points}" fill="none" stroke="{COLORS[feature]}" stroke-width="2"/>'
            )
        if onset is not None:
            px = x(float(onset))
            elements.append(f'<line x1="{px:.2f}" y1="{y_top}" x2="{px:.2f}" y2="{y_bottom}" stroke="#e28a00" stroke-width="2" stroke-dasharray="6 5"/>')
        if event:
            px = x(float(event["time"]))
            elements.append(f'<line x1="{px:.2f}" y1="{y_top}" x2="{px:.2f}" y2="{y_bottom}" stroke="#b3261e" stroke-width="3"/>')
        legend_x = right
        for feature in reversed(features):
            legend_x -= 145
            elements.append(f'<line x1="{legend_x}" y1="{y_top-10}" x2="{legend_x+24}" y2="{y_top-10}" stroke="{COLORS[feature]}" stroke-width="3"/><text x="{legend_x+30}" y="{y_top-6}" font-family="sans-serif" font-size="11">{html.escape(feature)}</text>')
    axis_y = top + len(PANELS) * (panel_height + gap) - gap + 20
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = start + fraction * span
        elements.append(f'<text x="{x(value):.2f}" y="{axis_y}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.1f}</text>')
    elements.append(f'<text x="{(left+right)/2}" y="{axis_y+25}" text-anchor="middle" font-family="sans-serif" font-size="13">simulation time (s)</text>')
    legend_y = height - 18
    elements.append(f'<line x1="{left}" y1="{legend_y}" x2="{left+24}" y2="{legend_y}" stroke="#e28a00" stroke-width="2" stroke-dasharray="6 5"/><text x="{left+30}" y="{legend_y+4}" font-family="sans-serif" font-size="12">injection onset</text>')
    elements.append(f'<line x1="{left+170}" y1="{legend_y}" x2="{left+194}" y2="{legend_y}" stroke="#b3261e" stroke-width="3"/><text x="{left+200}" y="{legend_y+4}" font-family="sans-serif" font-size="12">terminal event</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">' + "".join(elements) + "</svg>\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.output.suffix.lower() != ".svg":
        raise SystemExit("output must be SVG")
    annotation = yaml.safe_load(args.annotation.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(args.telemetry.open(newline="", encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(annotation, rows), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
