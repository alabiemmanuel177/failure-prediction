#!/usr/bin/env python3
"""Assemble the illustrative Gazebo re-run demo: screen capture synchronised with the
re-run's own live risk trace (failure-monitor sidecar), stacked, with title and closing
cards. Engineering material: labelled as an illustrative re-execution on every frame.
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--sidecar", type=Path, required=True)
    p.add_argument("--ffmpeg", default=str(Path.home() / ".local/bin/ffmpeg"))
    p.add_argument("--label", default="Illustrative re-execution (engineering run, validation map) - not the confirmatory episode")
    a = p.parse_args()
    out = a.capture_dir
    s = yaml.safe_load(a.summary.read_text()); d = json.loads(a.sidecar.read_text())
    rid = s["identity"]["run_id"]; env = s["environment"]; lab = s["label_only"]
    event_time = None
    for e in s.get("events", []) or []:
        if e.get("terminal"): event_time = float(e["time"])
    if event_time is None and lab.get("primary_event_time") is not None: event_time = float(lab["primary_event_time"])
    if event_time is None:
        # fall back to the terminal_event stamp in the events sidecar
        ev = json.load(open(s["provenance"]["event_sidecar"]))
        for x in ev:
            if x.get("event_type") == "terminal_event": event_time = x["stamp"]["sec"] + x["stamp"]["nanosec"] / 1e9
    table = out / "rerun_predictions.csv"
    cols = ["run_id","decision_index","decision_time","split","map_id","route_id","fault_family","severity","seed","protected_test_used","eligibility","label","primary_event_class","primary_event_time","model_id","raw_score","risk_score","alarm","persistent"]
    with table.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for dec in d["decisions"]:
            t = float(dec["decision_time"])
            if event_time is not None and t > event_time: continue
            elig = "eligible_positive" if (event_time is not None and event_time - 10.0 <= t <= event_time - 1.0) else ("excluded_too_late" if event_time is not None and t > event_time - 1.0 else "eligible_negative")
            w.writerow({"run_id": rid, "decision_index": dec["decision_index"], "decision_time": t, "split": env.get("split"), "map_id": env.get("map_id"), "route_id": env.get("route_id"), "fault_family": lab.get("fault_family"), "severity": lab.get("severity"), "seed": env.get("seed"), "protected_test_used": "false", "eligibility": elig, "label": 1 if elig == "eligible_positive" else 0, "primary_event_class": lab.get("primary_event_class") or "", "primary_event_time": "" if event_time is None else event_time, "model_id": "p3_causal_tcn", "raw_score": dec.get("raw_score"), "risk_score": dec.get("risk_score"), "alarm": "true" if dec.get("alarm") else "false", "persistent": "true" if dec.get("persistent") else "false"})
    sb = out / "storyboard"
    run([sys.executable, ROOT / "scripts/build_demo_storyboard.py", table, rid, "--output-dir", sb])
    shots = json.loads((sb / "shot_list.json").read_text()); low, high = shots["episode_time_span_s"]; fps = shots["fps"]; n = shots["frame_count"]
    import cairosvg
    for f in sorted((sb / "frames").glob("frame_*.svg")):
        cairosvg.svg2png(url=str(f), write_to=str(f.with_suffix(".png")), output_width=1000, output_height=450)
    run([a.ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", sb / "frames/frame_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", sb / "risk_trace.mp4"])
    # wall-clock alignment from the clock logger
    pairs = [(float(r["wall_s"]), float(r["sim_s"])) for r in csv.DictReader((out / "clock_wall.csv").open())]
    cap0 = float((out / "capture_wall_start.txt").read_text().strip())
    def wall_of(sim_t):
        best = min(pairs, key=lambda x: abs(x[1] - sim_t)); return best[0] + (sim_t - best[1])
    off_low, off_high = wall_of(low) - cap0, wall_of(high) - cap0
    demo_len = (n - 1) / fps
    factor = demo_len / max(off_high - off_low, 0.1)
    print(f"capture offsets {off_low:.2f}..{off_high:.2f} s -> retime x{factor:.2f} to {demo_len:.1f} s")
    # crop to the 3D viewport (window frame, title bar and toolbar removed), keep aspect, width 1000
    vf = f"trim=start={max(off_low,0):.3f}:end={off_high:.3f},setpts=(PTS-STARTPTS)*{factor:.5f},crop=996:716:16:100,scale=1000:-2"
    sim_h = 702
    label = a.label.replace(":", "\\:").replace("'", "")
    vf_label = vf + f",drawtext=text='{label}':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.55:x=10:y=h-th-10"
    sim = out / "sim_view.mp4"
    try:
        run([a.ffmpeg, "-y", "-loglevel", "error", "-i", out / "capture.mp4", "-vf", vf_label, "-r", f"{fps:g}", "-c:v", "libx264", "-pix_fmt", "yuv420p", sim])
    except subprocess.CalledProcessError:
        print("drawtext unavailable; label goes on the cards only"); run([a.ffmpeg, "-y", "-loglevel", "error", "-i", out / "capture.mp4", "-vf", vf, "-r", f"{fps:g}", "-c:v", "libx264", "-pix_fmt", "yuv420p", sim])
    core = out / "demo_core.mp4"
    run([a.ffmpeg, "-y", "-loglevel", "error", "-i", sim, "-i", sb / "risk_trace.mp4", "-filter_complex", "[0:v][1:v]vstack=inputs=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", core])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    warn = shots.get("warning_time_s"); ev = shots.get("event_time_s")
    cards = {"title": ["Research 2: early failure prediction and recovery", "Illustrative re-execution in Gazebo (engineering run on a validation map).", f"Same route, seed and injected fault as validation episode {env.get('route_id')}; frozen predictor in recommendation mode.", "Not the confirmatory episode; see the faithful replay for the study's numbers."],
             "closing": ["Result of this re-run", (f"warning {warn:.1f} s, terminal event {ev:.1f} s, lead {ev - warn:.1f} s" if warn and ev else "no useful warning in this re-run"), "Preprint, code, dataset card, one-command reproduction and limitations.", ""]}
    for name, lines in cards.items():
        fig = plt.figure(figsize=(10, (sim_h + 450) / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.text(0.5, 0.62, lines[0], ha="center", va="center", fontsize=22, weight="bold"); ax.text(0.5, 0.50, lines[1], ha="center", va="center", fontsize=14); ax.text(0.5, 0.42, lines[2], ha="center", va="center", fontsize=12, color="#495057"); ax.text(0.5, 0.35, lines[3], ha="center", va="center", fontsize=12, color="#c92a2a")
        fig.savefig(out / f"{name}_card.png", dpi=100); plt.close(fig)
        run([a.ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-i", out / f"{name}_card.png", "-t", "5", "-r", f"{fps:g}", "-c:v", "libx264", "-pix_fmt", "yuv420p", out / f"{name}_card.mp4"])
    (out / "concat.txt").write_text(f"file '{(out/'title_card.mp4').resolve()}'\nfile '{core.resolve()}'\nfile '{(out/'closing_card.mp4').resolve()}'\n")
    run([a.ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", out / "concat.txt", "-c", "copy", out / "demo_90s_gazebo.mp4"])
    print("wrote", out / "demo_90s_gazebo.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
