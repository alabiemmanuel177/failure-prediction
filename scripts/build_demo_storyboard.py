#!/usr/bin/env python3
"""Produce the 90-second synchronized demonstration assets for one held-out episode.

Inputs: an alarmed prediction table (frozen contract columns), the episode ``run_id``
and optionally a recovery decision log CSV (``time_seconds, event_type, action, detail``)
from the paired recovery campaign. Outputs (new directory, never overwritten):
  frames/frame_%04d.svg   progressive risk-trace frames (dependency-free SVG)
  timeline.svg            full-episode trace with warning and recovery markers
  shot_list.json          shots with demo timestamps synchronised to episode time
  ASSEMBLY.md             the ffmpeg command (documented, not executed)
Video assembly is documented but never executed here; ffmpeg presence is detected.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.release import sha256_file  # noqa: E402


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_episode(predictions: Path, run_id: str) -> list[dict[str, str]]:
    with predictions.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["run_id"] == run_id]
    if not rows:
        raise SystemExit(f"run_id not found: {run_id}")
    models = {row["model_id"] for row in rows}
    if len(models) != 1:
        raise SystemExit(f"episode rows mix models {sorted(models)}; filter the table to one model")
    rows.sort(key=lambda row: float(row["decision_time"]))
    return rows


def load_recovery_log(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        float(row["time_seconds"])
    return sorted(rows, key=lambda row: float(row["time_seconds"]))


def trace_svg(rows: list[dict[str, str]], upto: float | None, warning_time: float | None,
              event_time: float | None, actions: list[dict[str, str]], title: str) -> str:
    width, height = 1000, 450
    left, right, top, bottom = 85, 970, 45, 370
    times = [float(row["decision_time"]) for row in rows]
    low, high = min(times), max(times)
    span = high - low or 1.0
    x = lambda value: left + (float(value) - low) / span * (right - left)
    y = lambda value: bottom - max(0.0, min(1.0, float(value))) * (bottom - top)
    parts = [f'<rect width="{width}" height="{height}" fill="white"/>',
             f'<text x="{left}" y="24" font-family="sans-serif" font-size="18" font-weight="600">{html.escape(title)}</text>']
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{left}" y1="{y(tick):.2f}" x2="{right}" y2="{y(tick):.2f}" stroke="#d9dde3"/>')
        parts.append(f'<text x="{left-12}" y="{y(tick)+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:.2g}</text>')
    visible = [row for row in rows if upto is None or float(row["decision_time"]) <= upto]
    points = " ".join(f"{x(row['decision_time']):.2f},{y(row['risk_score']):.2f}" for row in visible)
    if points:
        parts.append(f'<polyline points="{points}" fill="none" stroke="#2455a4" stroke-width="2.5"/>')
    if warning_time is not None and (upto is None or warning_time <= upto):
        parts.append(f'<line x1="{x(warning_time):.2f}" y1="{top}" x2="{x(warning_time):.2f}" y2="{bottom}" stroke="#e28a00" stroke-width="3" stroke-dasharray="7 5"/>')
        parts.append(f'<text x="{x(warning_time)+4:.2f}" y="{top+14}" font-family="sans-serif" font-size="12" fill="#e28a00">warning {warning_time:.1f} s</text>')
    for index, action in enumerate(actions):
        t = float(action["time_seconds"])
        if upto is not None and t > upto:
            continue
        parts.append(f'<line x1="{x(t):.2f}" y1="{top}" x2="{x(t):.2f}" y2="{bottom}" stroke="#3a9d5d" stroke-width="2"/>')
        parts.append(f'<text x="{x(t)+4:.2f}" y="{top+32+14*index}" font-family="sans-serif" font-size="12" fill="#3a9d5d">{html.escape(action.get("action") or action.get("event_type", ""))} {t:.1f} s</text>')
    if event_time is not None and (upto is None or event_time <= upto):
        parts.append(f'<line x1="{x(event_time):.2f}" y1="{top}" x2="{x(event_time):.2f}" y2="{bottom}" stroke="#b3261e" stroke-width="3"/>')
    parts.extend([
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222"/>',
        f'<text x="{(left+right)/2:.2f}" y="420" text-anchor="middle" font-family="sans-serif" font-size="14">simulation time (s)</text>',
        f'<text x="20" y="{(top+bottom)/2:.2f}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 20 {(top+bottom)/2:.2f})">warning risk</text>',
    ])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            + "".join(parts) + "</svg>\n")


def build_storyboard(rows: list[dict[str, str]], actions: list[dict[str, str]], out: Path, *,
                     demo_seconds: float, fps: float, sources: dict[str, str]) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite storyboard directory: {out}")
    run_id = rows[0]["run_id"]
    times = [float(row["decision_time"]) for row in rows]
    low, high = min(times), max(times)
    events = {row["primary_event_time"] for row in rows if row["primary_event_time"] not in {"", "None"}}
    event_time = float(next(iter(events))) if events else None
    useful = [float(row["decision_time"]) for row in rows
              if _truth(row.get("alarm", "false")) and row["eligibility"] == "eligible_positive"]
    any_alarm = [float(row["decision_time"]) for row in rows if _truth(row.get("alarm", "false"))]
    warning_time = useful[0] if useful else (any_alarm[0] if any_alarm else None)
    lead_in, lead_out = 5.0, 5.0
    playable = demo_seconds - lead_in - lead_out
    scale = playable / ((high - low) or 1.0)
    to_demo = lambda t: round(lead_in + (t - low) * scale, 2)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True)
    frame_count = int(round(playable * fps))
    for index in range(frame_count + 1):
        upto = low + index / (frame_count or 1) * (high - low)
        (frames_dir / f"frame_{index:04d}.svg").write_text(
            trace_svg(rows, upto, warning_time, event_time, actions, f"{run_id} — t={upto:.1f} s"), encoding="utf-8")
    (out / "timeline.svg").write_text(
        trace_svg(rows, None, warning_time, event_time, actions, f"Episode timeline — {run_id}"), encoding="utf-8")
    shots = [{"shot": "title", "demo_start_s": 0.0, "demo_end_s": lead_in,
              "text": "The robot forecasts a navigation failure seconds early, then recovers."}]
    shots.append({"shot": "approach", "demo_start_s": lead_in, "demo_end_s": to_demo(warning_time) if warning_time else lead_in + playable,
                  "text": "Synchronized video with the live risk trace."})
    if warning_time is not None:
        shots.append({"shot": "warning", "episode_time_s": warning_time, "demo_start_s": to_demo(warning_time),
                      "lead_seconds_before_event": (event_time - warning_time) if event_time else None,
                      "text": "Calibrated risk crosses the frozen threshold with 2-of-3 persistence."})
    for action in actions:
        t = float(action["time_seconds"])
        shots.append({"shot": "recovery_action", "episode_time_s": t, "demo_start_s": to_demo(t),
                      "action": action.get("action"), "event_type": action.get("event_type"),
                      "detail": action.get("detail", "")})
    if event_time is not None:
        shots.append({"shot": "terminal_event", "episode_time_s": event_time, "demo_start_s": to_demo(event_time),
                      "event_class": rows[0].get("primary_event_class")})
    shots.append({"shot": "closing", "demo_start_s": demo_seconds - lead_out, "demo_end_s": demo_seconds,
                  "text": "Preprint, code, dataset card, one-command reproduction and limitations."})
    ffmpeg = shutil.which("ffmpeg")
    shot_list = {
        "run_id": run_id, "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "demo_seconds": demo_seconds, "fps": fps, "frame_count": frame_count + 1,
        "episode_time_span_s": [low, high], "warning_time_s": warning_time, "event_time_s": event_time,
        "recovery_actions": actions, "shots": shots, "sources": sources,
        "ffmpeg_available": ffmpeg is not None, "ffmpeg_path": ffmpeg, "video_assembled": False,
    }
    (out / "shot_list.json").write_text(json.dumps(shot_list, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "ASSEMBLY.md").write_text(
        f"# Demo assembly for {run_id}\n\n"
        f"ffmpeg detected: {'yes (' + ffmpeg + ')' if ffmpeg else 'no; install ffmpeg and rsvg-convert'}.\n\n"
        "Rasterise the SVG frames, then assemble at the storyboard frame rate and mux with the\n"
        "simulator screen capture (`sim.mp4`, trimmed to the same episode span):\n\n"
        "```bash\n"
        "for f in frames/frame_*.svg; do rsvg-convert -w 1000 -h 450 \"$f\" -o \"${f%.svg}.png\"; done\n"
        f"ffmpeg -framerate {fps:g} -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p risk_trace.mp4\n"
        "ffmpeg -i sim.mp4 -i risk_trace.mp4 -filter_complex \"[0:v][1:v]vstack=inputs=2\" -c:v libx264 demo_90s.mp4\n"
        "```\n\nThis command is documented only; the storyboard script never executes it.\n",
        encoding="utf-8")
    return shot_list


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("run_id")
    parser.add_argument("--recovery-log", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--demo-seconds", type=float, default=90.0)
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()
    rows = load_episode(args.predictions, args.run_id)
    if rows[0].get("split") not in {"held_out_map_test", ""}:
        print(f"note: episode split is {rows[0].get('split')!r}; the portfolio demo should use a held-out episode")
    actions = load_recovery_log(args.recovery_log)
    out = args.output_dir or (ROOT / "reports/demo" / args.run_id)
    sources = {"predictions": str(args.predictions), "predictions_sha256": sha256_file(args.predictions)}
    if args.recovery_log:
        sources.update({"recovery_log": str(args.recovery_log), "recovery_log_sha256": sha256_file(args.recovery_log)})
    shot_list = build_storyboard(rows, actions, out, demo_seconds=args.demo_seconds, fps=args.fps, sources=sources)
    print(json.dumps({"output": str(out), "frames": shot_list["frame_count"], "shots": len(shot_list["shots"]),
                      "ffmpeg_available": shot_list["ffmpeg_available"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
