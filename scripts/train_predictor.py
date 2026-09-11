#!/usr/bin/env python3
"""Fit one learned predictor on development sequences and select on validation.

Implements the CLI in docs/model-development-contracts.md:

  .venv/bin/python scripts/train_predictor.py \\
      --model-id p3_causal_tcn --config configs/models/p3_causal_tcn.yaml \\
      --train-dataset balanced_pilot_v1-development-648 [--train-dataset ...] \\
      --selection-dataset balanced_validation_v1-validation-324 \\
      [--exclude-family <family>] [--ablation <name>] --seed 20260903 \\
      --output-root models/

Fitting reads ``sequences/`` of development datasets only; validation datasets are
used only to select the epoch (window-level AUPRC, own numpy implementation). The
event recall at the frozen alarm budget is logged every epoch for information and
is never used for selection here. Every output is immutable; an existing run
directory is refused.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import select_validation_threshold  # noqa: E402
from src.models import common  # noqa: E402
from src.models.common import (  # noqa: E402
    ALARM_POLICY_PATH, CLEAN_FAMILY, ELIGIBLE_NEGATIVE, ELIGIBLE_POSITIVE, LEARNED_MODEL_IDS,
    MODEL_IDS, ORACLE_MODEL_ID, EpisodeArrays, apply_feature_mask, apply_normalization,
    average_precision, auroc, build_model, build_training_set, count_parameters, fit_normalization,
    frozen_feature_names, git_commit, load_dataset_manifest, load_episode_decisions,
    load_model_config, load_split_sequences, masked_age_by_feature, primary_value_features,
    publish_new_bytes, resolve_feature_mask, seed_everything, sha256_bytes, sha256_file,
    torch_device, write_json,
)


def default_run_name(args: argparse.Namespace) -> str:
    parts = [args.model_id]
    if args.ablation:
        parts.append(f"abl-{args.ablation}")
    if args.exclude_family:
        parts.append(f"excl-{args.exclude_family}")
    parts.append(f"seed{args.seed}")
    return "__".join(parts)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def episode_summary(episodes: list[EpisodeArrays]) -> dict:
    families: dict[str, int] = {}
    for episode in episodes:
        families[episode.record.fault_family] = families.get(episode.record.fault_family, 0) + 1
    return {
        "episodes": len(episodes),
        "episodes_with_event": sum(1 for e in episodes if e.record.primary_event_class),
        "clean_episodes": sum(1 for e in episodes if e.record.is_clean),
        "positive_windows": int(sum((e.y == 1).sum() for e in episodes)),
        "negative_windows": int(sum((e.y == 0).sum() for e in episodes)),
        "by_fault_family": dict(sorted(families.items())),
        "maps": sorted({e.record.map_id for e in episodes}),
    }


# --------------------------------------------------------------------------- event-recall log


def event_recall_rows(
    episodes: list[EpisodeArrays], scores_by_run: dict[str, np.ndarray], *, grid: float,
) -> tuple[dict[str, list[dict]], set[str]]:
    rows: dict[str, list[dict]] = {}
    clean: set[str] = set()
    for episode in episodes:
        scores = scores_by_run[episode.record.run_id]
        if grid:
            scores = np.round(scores / grid) * grid
        scores = np.clip(scores, 0.0, 1.0)
        eligibility = episode.eligibility if episode.eligibility is not None else np.where(
            episode.y == 1, ELIGIBLE_POSITIVE, ELIGIBLE_NEGATIVE
        )
        rows[episode.record.run_id] = [
            {"decision_time": float(t), "risk_score": float(s), "eligibility": str(e),
             "primary_event_time": episode.record.primary_event_time}
            for t, s, e in zip(episode.decision_time, scores, eligibility)
        ]
        if episode.record.is_clean:
            clean.add(episode.record.run_id)
    return rows, clean


def log_event_recall(rows, clean, alarm: dict) -> dict:
    persistence = alarm["persistence"]
    try:
        report = select_validation_threshold(
            rows, clean,
            false_alert_budget=float(alarm["false_alert_budget_per_clean_mission"]),
            required_above=int(persistence["required_above_threshold"]),
            decisions_considered=int(persistence["decisions_considered"]),
            cooldown_seconds=float(alarm["cooldown_seconds"]),
        )
    except ValueError as error:
        return {"event_recall": None, "reason": str(error)}
    return {
        "threshold": report["threshold"],
        "event_recall": report["event_recall"],
        "false_alerts_per_clean_mission": report["false_alerts_per_clean_mission"],
        "median_useful_lead_seconds_detected": report["median_useful_lead_seconds_detected"],
    }


# --------------------------------------------------------------------------- training


def make_loss(loss_config: dict, pos_weight: float):
    import torch

    kind = str(loss_config.get("type", "weighted_bce"))
    if kind == "weighted_bce":
        def loss_fn(logits, targets, weights):
            raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            return (raw * weights).sum() / weights.sum()
        return loss_fn
    if kind == "focal":
        gamma = float(loss_config.get("focal_gamma", 2.0))
        alpha = float(loss_config.get("focal_alpha", 0.25))

        def loss_fn(logits, targets, weights):
            probability = torch.sigmoid(logits)
            pt = probability * targets + (1 - probability) * (1 - targets)
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            raw = -alpha_t * (1 - pt).clamp(min=1e-6) ** gamma * torch.log(pt.clamp(min=1e-6))
            # class balance comes from alpha: strip pos_weight, keep event weights
            balanced = torch.where(targets > 0.5, weights / pos_weight, weights)
            return (raw * balanced).sum() / balanced.sum()
        return loss_fn
    if kind == "mse":
        def loss_fn(errors, targets, weights):
            return (errors * weights).sum() / weights.sum()
        return loss_fn
    raise ValueError(f"unknown loss type {kind!r}")


def score_windows(model, X: np.ndarray, device, batch_size: int, *, reconstruction: bool) -> np.ndarray:
    import torch

    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.from_numpy(X[start:start + batch_size]).to(device)
            out = model(batch)
            outputs.append((out if reconstruction else torch.sigmoid(out)).float().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)


def train_learned(args, config, out_dir: Path, record: dict) -> dict:
    import torch

    training = config["training"]
    model_id = args.model_id
    reconstruction = model_id == "p2_reconstruction_ae"
    feature_names = frozen_feature_names()
    primary = primary_value_features()
    masked_age = masked_age_by_feature()
    mask = resolve_feature_mask(args.ablation, primary_features=primary)
    train_manifests = [load_dataset_manifest(d, args.derived_root) for d in args.train_dataset]
    selection_manifest = load_dataset_manifest(args.selection_dataset, args.derived_root)
    dev = load_split_sequences(train_manifests, "development", purpose="fitting",
                               feature_names=feature_names, exclude=args.exclude_family)
    val = load_split_sequences([selection_manifest], "validation", purpose="selection",
                               feature_names=feature_names, exclude=args.exclude_family)
    for episode in dev + val:
        episode.X = apply_feature_mask(episode.X, feature_names, mask, masked_age=masked_age)
    bundle = fit_normalization(np.concatenate([e.X for e in dev]), feature_names)
    for episode in dev + val:
        episode.X = apply_normalization(episode.X, feature_names, bundle)

    sampling = training.get("negative_sampling", {})
    train_set = build_training_set(
        dev, cap_per_episode=sampling.get("cap_per_episode"),
        phase_bins=int(sampling.get("phase_bins", 4)),
        positive_weighting=str(training.get("positive_weighting", "class")),
        seed=args.seed, negatives_only=reconstruction, clean_only=reconstruction,
    )
    X_val = np.concatenate([e.X for e in val])
    y_val = np.concatenate([e.y for e in val])
    if not (y_val == 1).any() or not (y_val == 0).any():
        raise SystemExit("validation selection set needs both positive and negative windows")

    # event-recall log basis: decisions/ when published, else eligible sequences (approximation)
    recall_config = config.get("selection", {}).get("event_recall_log", {}) or {}
    recall_enabled = bool(recall_config.get("enabled", True))
    recall_basis = None
    recall_episodes: list[EpisodeArrays] = []
    if recall_enabled:
        if all(e.record.decisions_path.is_file() for e in val):
            recall_basis = "decisions"
            for episode in val:
                decisions = load_episode_decisions(episode.record, feature_names)
                decisions.X = apply_normalization(
                    apply_feature_mask(decisions.X, feature_names, mask, masked_age=masked_age),
                    feature_names, bundle,
                )
                recall_episodes.append(decisions)
        else:
            recall_basis = "eligible_sequences_only_approximation"
            recall_episodes = val
    alarm = yaml.safe_load(ALARM_POLICY_PATH.read_text(encoding="utf-8"))
    grid = float(recall_config.get("score_grid", 0.01))

    device = torch_device(args.device or training.get("device"))
    model = build_model(model_id, config).to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    loss_fn = make_loss(training.get("loss", {}), train_set.summary["pos_weight"])
    X_train = torch.from_numpy(train_set.X).to(device)
    y_train = torch.from_numpy(train_set.y.astype(np.float32)).to(device)
    w_train = torch.from_numpy(train_set.weights.astype(np.float32)).to(device)
    batch_size = int(training["batch_size"])
    max_epochs = int(args.max_epochs or training["max_epochs"])
    stopping = training.get("early_stopping", {})
    patience = int(stopping.get("patience", 10))
    minimum_delta = float(stopping.get("minimum_delta", 0.0))
    clip = training.get("gradient_clip_norm")
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))

    history = []
    best = {"epoch": 0, "validation_auprc": -math.inf}
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        started = time.monotonic()
        model.train()
        permutation = torch.randperm(X_train.shape[0], generator=generator).to(device)
        total_loss, batches = 0.0, 0
        for start in range(0, X_train.shape[0], batch_size):
            index = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = model(X_train[index])
            loss = loss_fn(output, y_train[index], w_train[index])
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        scores = score_windows(model, X_val, device, 4096, reconstruction=reconstruction)
        val_auprc = average_precision(y_val, scores)
        val_auroc = auroc(y_val, scores)
        entry = {
            "epoch": epoch, "train_loss": total_loss / max(batches, 1),
            "validation_auprc": val_auprc, "validation_auroc": val_auroc,
            "seconds": round(time.monotonic() - started, 3),
        }
        if recall_enabled:
            scores_by_run = {}
            for episode in recall_episodes:
                run_scores = score_windows(model, episode.X, device, 4096, reconstruction=reconstruction)
                if reconstruction:   # monotone provisional scaling for the information-only sweep
                    lo, hi = float(np.min(scores)), float(np.max(scores))
                    run_scores = np.clip((run_scores - lo) / ((hi - lo) or 1.0), 0.0, 1.0)
                scores_by_run[episode.record.run_id] = run_scores
            rows, clean = event_recall_rows(recall_episodes, scores_by_run, grid=grid)
            entry["event_recall_at_frozen_budget"] = {
                "basis": recall_basis, "purpose": "information_only",
                **log_event_recall(rows, clean, alarm),
            }
        history.append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)
        improved = val_auprc > best["validation_auprc"] + minimum_delta
        if improved or best_state is None:
            best = {"epoch": epoch, "validation_auprc": val_auprc, "validation_auroc": val_auroc}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stopping after {epoch} epochs (best epoch {best['epoch']})", flush=True)
                break
    model.load_state_dict(best_state)

    checkpoint = {
        "model_id": model_id, "state_dict": best_state, "config": config,
        "feature_names": list(feature_names), "ablation": args.ablation,
        "masked_features": list(mask.masked_features), "visible_time_steps": mask.visible_time_steps,
        "normalization_sha256": bundle.sha256, "best_epoch": best["epoch"],
        "protected_test_used": False,
    }
    if reconstruction:
        from src.models.autoencoder import ErrorScaler
        scoring = config.get("scoring", {})
        dev_errors = score_windows(model, train_set.X, device, 4096, reconstruction=True)
        scaler = ErrorScaler.fit(
            dev_errors, lower_quantile=float(scoring.get("lower_quantile", 0.01)),
            upper_quantile=float(scoring.get("upper_quantile", 0.99)), split="development",
        )
        checkpoint["error_scaler"] = scaler.to_dict()
        record["error_scaler"] = scaler.to_dict()

    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    checkpoint_bytes = buffer.getvalue()
    checkpoint_sha = sha256_bytes(checkpoint_bytes)
    publish_new_bytes(out_dir / "checkpoint.pt", checkpoint_bytes)
    publish_new_bytes(out_dir / "checkpoint.sha256", f"{checkpoint_sha}  checkpoint.pt\n".encode())
    publish_new_bytes(out_dir / "normalization.json", bundle.canonical_json().encode("utf-8"))

    record.update({
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "parameter_count": parameter_count,
        "normalization_sha256": bundle.sha256,
        "checkpoint_sha256": checkpoint_sha,
        "feature_mask": {"ablation": args.ablation, "masked_features": list(mask.masked_features),
                         "visible_time_steps": mask.visible_time_steps},
        "train_datasets": [
            {"dataset_id": m.dataset_id, "manifest": str(m.path), "manifest_sha256": m.sha256,
             "episodes_in_manifest": len(m.records)} for m in train_manifests
        ],
        "selection_dataset": {
            "dataset_id": selection_manifest.dataset_id, "manifest": str(selection_manifest.path),
            "manifest_sha256": selection_manifest.sha256,
            "episodes_in_manifest": len(selection_manifest.records),
        },
        "development": episode_summary(dev),
        "validation": episode_summary(val),
        "fitting_windows": train_set.summary,
        "validation_windows": {"positive": int((y_val == 1).sum()), "negative": int((y_val == 0).sum())},
        "epochs_run": len(history),
        "max_epochs": max_epochs,
        "early_stopping": {"metric": "validation_auprc", "patience": patience,
                           "minimum_delta": minimum_delta, "best_epoch": best["epoch"]},
        "best_validation_auprc": best["validation_auprc"],
        "best_validation_auroc": best["validation_auroc"],
        "event_recall_log_basis": recall_basis,
        "history": history,
        "sequence_sha256_by_run": {
            e.record.run_id: e.sha256 for e in dev + val
        },
    })
    return record


def train_oracle(args, out_dir: Path, record: dict) -> dict:
    import torch

    checkpoint = {"model_id": ORACLE_MODEL_ID, "analysis_only": True, "state_dict": {},
                  "protected_test_used": False}
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    payload = buffer.getvalue()
    sha = sha256_bytes(payload)
    publish_new_bytes(out_dir / "checkpoint.pt", payload)
    publish_new_bytes(out_dir / "checkpoint.sha256", f"{sha}  checkpoint.pt\n".encode())
    record.update({
        "analysis_only": True, "parameter_count": 0, "checkpoint_sha256": sha,
        "normalization_sha256": None, "epochs_run": 0,
        "note": "P6 oracle reads label-only eligibility; upper bound for analysis, never deployable",
    })
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True, choices=MODEL_IDS)
    parser.add_argument("--config", type=Path, help="configs/models/<model_id>.yaml")
    parser.add_argument("--train-dataset", action="append", default=[],
                        help="development dataset id or extraction manifest path (repeatable)")
    parser.add_argument("--selection-dataset", help="validation dataset id or manifest path")
    parser.add_argument("--exclude-family", default=None,
                        help="unseen-family fold: drop this fault family from fitting and selection")
    parser.add_argument("--ablation", default=None, help="name from configs/ablations.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "models")
    parser.add_argument("--run-name", default=None, help="default: <model>__abl-..__excl-..__seed<n>")
    parser.add_argument("--max-epochs", type=int, default=None, help="override the config (recorded)")
    parser.add_argument("--device", default=None, help="cuda|cpu (default: config, else auto)")
    parser.add_argument("--derived-root", type=Path, default=common.DERIVED_ROOT)
    args = parser.parse_args()

    if args.model_id in LEARNED_MODEL_IDS and (
        not args.config or not args.train_dataset or not args.selection_dataset
    ):
        parser.error("learned models require --config, --train-dataset and --selection-dataset")
    if args.exclude_family == CLEAN_FAMILY:
        parser.error("the clean family cannot be excluded")
    if args.ablation:
        common.ablation_kind(args.ablation)   # unknown names fail before any data is read
    run_name = args.run_name or default_run_name(args)
    out_dir = args.output_root / run_name
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing model directory: {out_dir}")

    seeding = seed_everything(args.seed)
    started = time.monotonic()
    config = load_model_config(args.config, args.model_id) if args.config else {"model_id": args.model_id}
    record = {
        "schema_version": 1,
        "model_id": args.model_id,
        "run_name": run_name,
        "protected_test_used": False,
        "seed": args.seed,
        "seeding": seeding,
        "command": sys.argv,
        "config_path": str(args.config) if args.config else None,
        "config_sha256": sha256_file(args.config) if args.config else None,
        "config": config,
        "ablation": args.ablation,
        "ablation_kind": common.ablation_kind(args.ablation),
        "exclude_family": args.exclude_family,
        "feature_schema_sha256": sha256_file(common.FEATURE_SCHEMA_PATH),
        "leakage_denylist_sha256": sha256_file(ROOT / "configs/leakage_denylist.yaml"),
        "ablations_sha256": sha256_file(common.ABLATIONS_PATH),
        "alarm_policy_sha256": sha256_file(ALARM_POLICY_PATH),
        "feature_names": list(frozen_feature_names()),
        "git_commit": git_commit(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "started_utc": utc_now(),
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    try:
        if args.model_id == ORACLE_MODEL_ID:
            record = train_oracle(args, out_dir, record)
        else:
            record = train_learned(args, config, out_dir, record)
    except Exception:
        # leave partial outputs visible for inspection but mark the directory as failed
        (out_dir / "FAILED").write_text(utc_now() + "\n", encoding="utf-8")
        raise
    record["wall_seconds"] = round(time.monotonic() - started, 3)
    record["finished_utc"] = utc_now()
    record_sha = write_json(out_dir / "training_record.json", record)
    print(json.dumps({
        "model_dir": str(out_dir), "model_id": args.model_id,
        "best_validation_auprc": record.get("best_validation_auprc"),
        "best_epoch": record.get("early_stopping", {}).get("best_epoch"),
        "parameter_count": record.get("parameter_count"),
        "training_record_sha256": record_sha, "wall_seconds": record["wall_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
