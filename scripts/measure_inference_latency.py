#!/usr/bin/env python3
"""Measure per-decision inference latency on CPU with batch size 1.

  .venv/bin/python scripts/measure_inference_latency.py --model-dir models/<run_name> \\
      [--dataset <dataset_id>] [--decisions 500] [--output reports/latency/<name>.json]

Reports median, p95 and max (milliseconds) of feature-window preparation (mask +
normalisation), model forward, and alarm-policy step for one decision at a time,
plus the end-to-end sum. The deployment target is the robot host, so the model runs
on CPU with one thread regardless of the GPU. Windows come from ``decisions/`` (or
``sequences/``) of the given dataset when available, otherwise from a seeded
synthetic generator so the timing never depends on data content.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import AlarmPolicy, apply_alarm_policy  # noqa: E402
from src.models import common  # noqa: E402
from src.models.common import (  # noqa: E402
    ALARM_POLICY_PATH, load_dataset_manifest, load_episode_decisions, load_episode_sequences,
    publish_new_bytes,
)

sys.path.insert(0, str(ROOT / "scripts"))
from predict_decisions import LoadedModel  # noqa: E402


def collect_windows(args, feature_names, steps: int, channels: int) -> tuple[np.ndarray, str]:
    if args.dataset:
        manifest = load_dataset_manifest(args.dataset, args.derived_root)
        windows = []
        for record in manifest.records:
            if record.protected_test_used is not False:
                raise SystemExit("latency measurement never reads protected episodes")
            try:
                episode = load_episode_decisions(record, feature_names)
            except FileNotFoundError:
                episode = load_episode_sequences(record, feature_names)
            windows.append(episode.X)
            if sum(w.shape[0] for w in windows) >= args.decisions:
                break
        X = np.concatenate(windows)[:args.decisions]
        return X.astype(np.float32), f"dataset:{manifest.dataset_id}"
    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((args.decisions, steps, channels)).astype(np.float32)
    for position, name in enumerate(feature_names):
        if name.endswith("__missing"):
            X[..., position] = (rng.random((args.decisions, steps)) < 0.05).astype(np.float32)
        elif name.endswith("__age_seconds"):
            X[..., position] = np.abs(X[..., position]) * 0.2
    return X, "synthetic"


def summarise(samples_ns: list[int]) -> dict:
    values = np.asarray(samples_ns, dtype=np.float64) / 1e6
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "max_ms": float(values.max()), "mean_ms": float(values.mean()), "count": int(values.size)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--decisions", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None, help="immutable JSON report")
    parser.add_argument("--derived-root", type=Path, default=common.DERIVED_ROOT)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    import torch
    torch.set_num_threads(max(1, args.threads))
    model = LoadedModel(args.model_dir, "cpu")
    if model.is_oracle:
        raise SystemExit("the oracle has no inference path to time")
    inputs = model.checkpoint["config"].get("input", {})
    steps, channels = int(inputs.get("time_steps", 10)), int(inputs.get("channels", 84))
    X, source = collect_windows(args, model.feature_names, steps, channels)
    if X.shape[0] < 2:
        raise SystemExit("need at least two windows to time")

    alarm = yaml.safe_load(ALARM_POLICY_PATH.read_text(encoding="utf-8"))
    threshold = alarm.get("threshold")
    policy = AlarmPolicy(
        float(threshold) if threshold is not None else 0.5,
        int(alarm["persistence"]["required_above_threshold"]),
        int(alarm["persistence"]["decisions_considered"]),
        float(alarm["cooldown_seconds"]),
    )
    stride = float(inputs.get("stride_seconds", 0.5))
    trailing = max(policy.decisions_considered, int(policy.cooldown_seconds / stride) + 1)

    prep_ns, model_ns, policy_ns, total_ns = [], [], [], []
    history: list[dict] = []
    for index in range(X.shape[0]):
        window = X[index:index + 1]
        t0 = time.perf_counter_ns()
        prepared = model.prepare(window)
        t1 = time.perf_counter_ns()
        score = float(model.score_prepared(prepared, batch_size=1)[0])
        t2 = time.perf_counter_ns()
        history.append({"decision_time": index * stride, "risk_score": score})
        history = history[-trailing:]
        apply_alarm_policy(history, policy)
        t3 = time.perf_counter_ns()
        if index >= args.warmup:
            prep_ns.append(t1 - t0)
            model_ns.append(t2 - t1)
            policy_ns.append(t3 - t2)
            total_ns.append(t3 - t0)
    report = {
        "schema_version": 1,
        "model_dir": str(args.model_dir),
        "model_id": model.model_id,
        "checkpoint_sha256": model.checkpoint_sha256,
        "device": "cpu",
        "threads": max(1, args.threads),
        "batch_size": 1,
        "windows_source": source,
        "timed_decisions": len(total_ns),
        "warmup_decisions": args.warmup,
        "feature_preparation": summarise(prep_ns),
        "model_forward": summarise(model_ns),
        "alarm_policy": summarise(policy_ns),
        "end_to_end": summarise(total_ns),
        "parameter_count": model.record.get("parameter_count"),
        "cpu": platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
        "protected_test_used": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    text = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        publish_new_bytes(args.output, text.encode("utf-8"))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
