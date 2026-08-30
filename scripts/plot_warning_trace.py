#!/usr/bin/env python3
"""Write a dependency-free SVG risk, alert, warning-window, and event timeline."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("run_id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [row for row in csv.DictReader(args.predictions.open(newline="", encoding="utf-8"))
            if row["run_id"] == args.run_id]
    if not rows:
        raise SystemExit(f"run_id not found: {args.run_id}")
    times = [float(row["decision_time"]) for row in rows]
    scores = [float(row["risk_score"]) for row in rows]
    alarms = [float(row["decision_time"]) for row in rows if row["alarm"].lower() == "true"]
    event_values = {row["primary_event_time"] for row in rows if row["primary_event_time"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.output.suffix.lower() != ".svg":
        raise SystemExit("dependency-free trace output must use the .svg extension")
    width, height = 1000, 450
    left, right, top, bottom = 85, 970, 45, 370
    low, high = min(times), max(times)
    span = high - low or 1.0
    x = lambda value: left + (float(value) - low) / span * (right - left)
    y = lambda value: bottom - max(0.0, min(1.0, float(value))) * (bottom - top)
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{left}" y="24" font-family="sans-serif" font-size="18" '
        f'font-weight="600">Warning trace — {html.escape(args.run_id)}</text>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        py = y(tick)
        elements.append(f'<line x1="{left}" y1="{py:.2f}" x2="{right}" y2="{py:.2f}" stroke="#d9dde3"/>')
        elements.append(f'<text x="{left-12}" y="{py+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:.2g}</text>')
    if event_values:
        event_time = float(next(iter(event_values)))
        window_left = max(left, x(event_time - 10.0))
        window_right = min(right, x(event_time - 1.0))
        if window_right > window_left:
            elements.append(
                f'<rect x="{window_left:.2f}" y="{top}" width="{window_right-window_left:.2f}" '
                f'height="{bottom-top}" fill="#3a9d5d" opacity="0.15"/>'
            )
        event_x = x(event_time)
        elements.append(f'<line x1="{event_x:.2f}" y1="{top}" x2="{event_x:.2f}" y2="{bottom}" stroke="#b3261e" stroke-width="3"/>')
    points = " ".join(f"{x(t):.2f},{y(s):.2f}" for t, s in zip(times, scores))
    elements.append(f'<polyline points="{points}" fill="none" stroke="#2455a4" stroke-width="2.5"/>')
    for alarm in alarms:
        alarm_x = x(alarm)
        elements.append(f'<line x1="{alarm_x:.2f}" y1="{top}" x2="{alarm_x:.2f}" y2="{bottom}" stroke="#e28a00" stroke-width="2" stroke-dasharray="7 5"/>')
    elements.extend([
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222"/>',
        f'<text x="{(left+right)/2:.2f}" y="420" text-anchor="middle" font-family="sans-serif" font-size="14">simulation time (s)</text>',
        f'<text x="20" y="{(top+bottom)/2:.2f}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 20 {(top+bottom)/2:.2f})">warning risk</text>',
        f'<text x="{left}" y="393" font-family="sans-serif" font-size="12">{low:.2f}</text>',
        f'<text x="{right}" y="393" text-anchor="end" font-family="sans-serif" font-size="12">{high:.2f}</text>',
        '<line x1="700" y1="24" x2="728" y2="24" stroke="#2455a4" stroke-width="3"/><text x="735" y="29" font-family="sans-serif" font-size="12">risk</text>',
        '<line x1="790" y1="24" x2="818" y2="24" stroke="#e28a00" stroke-width="2" stroke-dasharray="7 5"/><text x="825" y="29" font-family="sans-serif" font-size="12">alarm</text>',
        '<line x1="885" y1="24" x2="913" y2="24" stroke="#b3261e" stroke-width="3"/><text x="920" y="29" font-family="sans-serif" font-size="12">event</text>',
    ])
    args.output.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(elements) + "</svg>\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
