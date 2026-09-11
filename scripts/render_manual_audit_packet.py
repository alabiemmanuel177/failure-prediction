#!/usr/bin/env python3
"""Render all reviewer timelines and an index for the frozen audit sample."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    output_root = ROOT / "reports/manual-audit"
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "index.md"
    if index_path.exists():
        raise SystemExit(f"refusing to overwrite {index_path}")
    summaries = {}
    for path in (ROOT / "data/raw/summaries").glob("*.yaml"):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        summaries[document["identity"]["run_id"]] = document
    rows = []
    annotations = sorted(
        path for path in (ROOT / "data/annotations").glob("*.yaml")
        if path.name != "episode.template.yaml"
    )
    if len(annotations) != 20:
        raise SystemExit(f"expected 20 automatic annotations, found {len(annotations)}")
    for annotation_path in annotations:
        annotation = yaml.safe_load(annotation_path.read_text(encoding="utf-8"))
        run_id = annotation["run_id"]
        summary = summaries[run_id]
        plot_path = output_root / f"{run_id}.svg"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/plot_annotation_review.py"),
            str(annotation_path),
            str(ROOT / "data/audit/scalar_telemetry" / f"{run_id}.csv"),
            str(plot_path),
        ], check=True)
        rows.append(
            f"| {summary['identity']['episode_key']} | {summary['label_only']['fault_family']} | "
            f"{summary['outcome']['terminal_state']} | [{run_id}]({run_id}.svg) | "
            f"`data/annotations/{run_id}.yaml` |"
        )
    document = [
        "# Frozen 20-episode manual audit\n",
        "These timelines contain telemetry and label-only onset/event markers, but no model outputs. Review the MCAP when a plotted signal is ambiguous.\n",
        "| Episode key | Fault family | Outcome | Timeline | Automatic annotation |",
        "|---|---|---|---|---|",
        *rows,
        "\nTraining admission remains forbidden until `python3 scripts/check_manual_audit_gate.py` passes.\n",
    ]
    index_path.write_text("\n".join(document), encoding="utf-8")
    print(f"wrote reviewer index and {len(rows)} timelines to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
