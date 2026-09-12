#!/usr/bin/env python3
"""Render a recorded episode's bag as a synchronised top-down replay for the demo.

Reads the immutable rosbag (ground-truth pose, AMCL estimate, laser scan, planned path,
research2 events) and the storyboard's shot list, and writes one PNG per storyboard frame
at exactly the same episode times, so the replay can be stacked with the risk-trace
video. No simulator runs and no artefact is modified: this is a rendering of what was
recorded. Then assembles sim.mp4 and the stacked 90-second demo with ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

import rosbag2_py  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
from rosidl_runtime_py.utilities import get_message  # noqa: E402


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def read_bag(bag_dir: Path) -> dict:
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    data = {"gt": [], "amcl": [], "scan": [], "plan": [], "events": []}
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        t = t_ns / 1e9
        if topic == "/ground_truth_pose":
            m = deserialize_message(raw, get_message(types[topic]))
            data["gt"].append((t, m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation)))
        elif topic == "/amcl_pose":
            m = deserialize_message(raw, get_message(types[topic]))
            data["amcl"].append((t, m.pose.pose.position.x, m.pose.pose.position.y))
        elif topic == "/scan":
            m = deserialize_message(raw, get_message(types[topic]))
            data["scan"].append((t, float(m.angle_min), float(m.angle_increment), float(m.range_max), np.asarray(m.ranges, dtype=float)))
        elif topic == "/plan":
            m = deserialize_message(raw, get_message(types[topic]))
            data["plan"].append((t, [(p.pose.position.x, p.pose.position.y) for p in m.poses]))
        elif topic == "/research2/events":
            m = deserialize_message(raw, get_message(types[topic]))
            for s in m.status:
                kv = {v.key: v.value for v in s.values}
                data["events"].append((t, kv.get("event_type", s.name.split("/")[-1])))
    return data


def latest_before(items, t):
    chosen = None
    for item in items:
        if item[0] <= t:
            chosen = item
        else:
            break
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--storyboard", type=Path, required=True, help="directory with shot_list.json and risk_trace.mp4")
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--route-yaml", type=Path, required=True)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=450)
    args = parser.parse_args()

    shots = json.loads((args.storyboard / "shot_list.json").read_text())
    low, high = shots["episode_time_span_s"]
    frame_count = int(shots["frame_count"])
    fps = float(shots["fps"])
    warning_time = shots.get("warning_time_s")
    event_time = shots.get("event_time_s")
    run_id = shots["run_id"]

    map_meta = yaml.safe_load(args.map_yaml.read_text())
    grid = np.asarray(Image.open(args.map_yaml.parent / map_meta["image"]).convert("L"), dtype=float)
    if map_meta.get("negate"):
        grid = 255 - grid
    res = float(map_meta["resolution"]); ox, oy = map_meta["origin"][:2]
    extent = (ox, ox + grid.shape[1] * res, oy, oy + grid.shape[0] * res)
    routes = yaml.safe_load(args.route_yaml.read_text())["routes"]
    route = next(r for r in routes if r["route_id"] == args.route_id)

    data = read_bag(args.bag)
    inj_start = next((t for t, e in data["events"] if e == "injection_started"), None)
    inj_end = next((t for t, e in data["events"] if e == "injection_ended"), None)
    terminal = next((t for t, e in data["events"] if e == "terminal_event"), event_time)
    gt = np.asarray([(t, x, y) for t, x, y, _ in data["gt"]])

    out = args.storyboard / "frames_sim"
    out.mkdir(exist_ok=True)
    dpi = 100
    for index in range(frame_count):
        upto = low + index / max(frame_count - 1, 1) * (high - low)
        fig = plt.figure(figsize=(args.width / dpi, args.height / dpi), dpi=dpi)
        ax = fig.add_axes([0.02, 0.04, 0.60, 0.92])
        ax.imshow(grid, cmap="gray", extent=extent, origin="lower", vmin=0, vmax=255, interpolation="nearest")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.plot(route["start"]["x"], route["start"]["y"], marker="s", color="#2b8a3e", ms=7, ls="none")
        ax.plot(route["goal"]["x"], route["goal"]["y"], marker="*", color="#e67700", ms=12, ls="none")
        plan = latest_before(data["plan"], upto)
        if plan and plan[1]:
            px, py = zip(*plan[1]); ax.plot(px, py, color="#1c7ed6", lw=1.2, alpha=0.8)
        past = gt[gt[:, 0] <= upto]
        if len(past):
            ax.plot(past[:, 1], past[:, 2], color="#212529", lw=1.5)
            t, x, y, yaw = latest_before(data["gt"], upto)
            scan = latest_before(data["scan"], upto)
            if scan:
                _, amin, ainc, rmax, ranges = scan
                angles = amin + ainc * np.arange(len(ranges))
                ok = np.isfinite(ranges) & (ranges > 0.05) & (ranges < rmax)
                sx = x + ranges[ok] * np.cos(yaw + angles[ok]); sy = y + ranges[ok] * np.sin(yaw + angles[ok])
                ax.scatter(sx, sy, s=2, color="#e03131", alpha=0.6)
            ax.plot(x, y, marker="o", color="#212529", ms=8, ls="none")
            ax.arrow(x, y, 0.35 * math.cos(yaw), 0.35 * math.sin(yaw), width=0.04, color="#212529")
            amcl = latest_before(data["amcl"], upto)
            if amcl:
                ax.plot(amcl[1], amcl[2], marker="x", color="#1c7ed6", ms=8, ls="none")
        # window on the robot's neighbourhood
        cx, cy = (past[-1, 1], past[-1, 2]) if len(past) else (route["start"]["x"], route["start"]["y"])
        ax.set_xlim(cx - 4.0, cx + 4.0); ax.set_ylim(cy - 3.0, cy + 3.0)
        # status panel
        panel = fig.add_axes([0.64, 0.04, 0.34, 0.92]); panel.axis("off")
        lines = [f"episode {run_id[:8]}", f"map {map_meta.get('map_id', args.map_yaml.parent.name)}  route {args.route_id}",
                 f"t = {upto:5.1f} s (simulation time)", ""]
        fault_state = "not yet injected"
        if inj_start is not None and upto >= inj_start:
            fault_state = "planner oscillation ACTIVE" if (inj_end is None or upto < inj_end) else "injection ended"
        lines.append(f"fault: {fault_state}")
        if warning_time is not None:
            lines.append("warning: RAISED" if upto >= warning_time else f"warning: none (fires at {warning_time:.1f} s)")
            if upto >= warning_time and terminal:
                lines.append(f"lead time at event: {terminal - warning_time:.1f} s")
        if terminal is not None and upto >= terminal - 0.25:
            lines.append("TERMINAL EVENT: planner failure")
        lines += ["", "black: ground-truth path and pose", "blue x: AMCL estimate", "blue line: current Nav2 plan",
                  "red: laser returns", "green square: start   orange star: goal"]
        panel.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
        fig.savefig(out / f"frame_{index:04d}.png", dpi=dpi)
        plt.close(fig)
    print("rendered", frame_count, "frames ->", out)

    sim = args.storyboard / "sim.mp4"
    subprocess.run([args.ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", str(out / "frame_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(sim)], check=True)
    trace = args.storyboard / "risk_trace.mp4"
    stacked = args.storyboard / "demo_core.mp4"
    subprocess.run([args.ffmpeg, "-y", "-loglevel", "error", "-i", str(sim), "-i", str(trace), "-filter_complex",
                    "[0:v][1:v]vstack=inputs=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(stacked)], check=True)
    # title and closing cards (5 s each) around the synchronised core
    for name, text in (("title", shots["shots"][0]["text"]), ("closing", shots["shots"][-1]["text"])):
        fig = plt.figure(figsize=(args.width / dpi, 2 * args.height / dpi), dpi=dpi); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.text(0.5, 0.55, "Research 2: early failure prediction and recovery", ha="center", va="center", fontsize=22, weight="bold")
        ax.text(0.5, 0.42, text, ha="center", va="center", fontsize=15, wrap=True)
        ax.text(0.5, 0.30, f"held-out episode {run_id} (test map, planner oscillation)", ha="center", va="center", fontsize=11, color="#495057")
        fig.savefig(args.storyboard / f"{name}_card.png", dpi=dpi); plt.close(fig)
        subprocess.run([args.ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-i", str(args.storyboard / f"{name}_card.png"), "-t", "5",
                        "-r", f"{fps:g}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(args.storyboard / f"{name}_card.mp4")], check=True)
    concat = args.storyboard / "concat.txt"
    concat.write_text("".join(f"file '{p.name}'\n" for p in (args.storyboard / "title_card.mp4", stacked, args.storyboard / "closing_card.mp4")))
    demo = args.storyboard / "demo_90s.mp4"
    subprocess.run([args.ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat.resolve()), "-c", "copy", str(demo.resolve())], check=True)
    print("wrote", sim, stacked, demo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
