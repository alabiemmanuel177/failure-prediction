#!/usr/bin/env python3
"""Score every label-grid decision of one or more datasets with a trained predictor.

  .venv/bin/python scripts/predict_decisions.py \\
      --model-dir models/<run_name> --dataset <dataset_id> [--dataset ...] \\
      --output reports/predictions/<name>.csv [--allow-protected-after-freeze]

Reads ``decisions/<run_id>.npz`` (every decision with a complete history, including
excluded ones so the alarm policy sees consecutive decisions), applies the frozen
feature mask and development normalisation stored with the model, and writes the
immutable prediction table with exactly the contract columns, sorted by
``run_id, decision_index``. ``risk_score`` equals ``raw_score`` until
``scripts/apply_calibration.py`` replaces it. A ``<output>.provenance.json`` sidecar
records SHA-256 of the checkpoint, manifests and every decision artifact.

Protected (held-out map test) episodes are refused unless
``--allow-protected-after-freeze`` is given *and* ``scripts/check_readiness.py
--stage confirmatory`` passes, mirroring ``scripts/assemble_episode_sequences.py``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import common  # noqa: E402
from src.models.common import (  # noqa: E402
    ORACLE_MODEL_ID, EpisodeArrays, FeatureMask, apply_feature_mask, apply_normalization,
    build_model, frozen_feature_names, load_dataset_manifest, load_episode_decisions,
    masked_age_by_feature, normalization_from_json, publish_new_bytes, sha256_bytes,
    sha256_file, torch_device,
)
from src.models.oracle import oracle_risk_scores  # noqa: E402
from src.protected_data import enforce_protected_boundary  # noqa: E402

COLUMNS = (
    "run_id", "decision_index", "decision_time", "split", "map_id", "route_id", "fault_family",
    "severity", "seed", "protected_test_used", "eligibility", "label", "primary_event_class",
    "primary_event_time", "model_id", "raw_score", "risk_score",
)


class LoadedModel:
    """Checkpoint + normalisation + mask, ready to score prepared windows."""

    def __init__(self, model_dir: Path, device: str | None = None):
        self.model_dir = model_dir
        record_path = model_dir / "training_record.json"
        if not record_path.is_file():
            raise FileNotFoundError(f"training_record.json missing in {model_dir}")
        self.record = json.loads(record_path.read_text(encoding="utf-8"))
        self.model_id = str(self.record["model_id"])
        checkpoint_path = model_dir / "checkpoint.pt"
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        declared = (model_dir / "checkpoint.sha256").read_text(encoding="utf-8").split()[0]
        if declared != self.checkpoint_sha256:
            raise ValueError(f"{checkpoint_path}: sha256 differs from checkpoint.sha256")
        if self.record.get("checkpoint_sha256") not in (None, self.checkpoint_sha256):
            raise ValueError("training_record checkpoint_sha256 differs from checkpoint.pt")
        import torch
        self.torch = torch
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if self.checkpoint.get("model_id") != self.model_id:
            raise ValueError("checkpoint model_id differs from training record")
        self.feature_names = frozen_feature_names()
        self.is_oracle = self.model_id == ORACLE_MODEL_ID
        if self.is_oracle:
            self.model = None
            self.bundle = None
            self.mask = FeatureMask(None, (), None)
            self.masked_age = {}
            self.device = None
            return
        if list(self.checkpoint["feature_names"]) != list(self.feature_names):
            raise ValueError("checkpoint feature order differs from the frozen model columns")
        bundle_text = (model_dir / "normalization.json").read_text(encoding="utf-8")
        self.bundle = normalization_from_json(bundle_text)
        if self.bundle.sha256 != self.checkpoint.get("normalization_sha256"):
            raise ValueError("normalization.json differs from the bundle recorded in the checkpoint")
        visible = self.checkpoint.get("visible_time_steps")
        self.mask = FeatureMask(
            self.checkpoint.get("ablation"), tuple(self.checkpoint.get("masked_features", ())),
            int(visible) if visible is not None else None,
        )
        self.masked_age = masked_age_by_feature()
        self.device = torch_device(device or "cpu")
        self.model = build_model(self.model_id, self.checkpoint["config"]).to(self.device)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.eval()
        self.reconstruction = self.model_id == "p2_reconstruction_ae"
        self.scaler = None
        if self.reconstruction:
            from src.models.autoencoder import ErrorScaler
            self.scaler = ErrorScaler.from_dict(self.checkpoint["error_scaler"])

    def prepare(self, X: np.ndarray) -> np.ndarray:
        masked = apply_feature_mask(X, self.feature_names, self.mask, masked_age=self.masked_age)
        return apply_normalization(masked, self.feature_names, self.bundle)

    def score_prepared(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        torch = self.torch
        outputs = []
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                batch = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size])).to(self.device)
                out = self.model(batch).float().cpu().numpy()
                outputs.append(self.scaler.transform(out) if self.reconstruction else 1 / (1 + np.exp(-out)))
        scores = np.concatenate(outputs) if outputs else np.zeros(0)
        return np.clip(scores.astype(np.float64), 0.0, 1.0)

    def score_episode(self, episode: EpisodeArrays) -> np.ndarray:
        if self.is_oracle:
            return oracle_risk_scores(episode.eligibility)
        if episode.X.shape[0] == 0:
            return np.zeros(0)
        return self.score_prepared(self.prepare(episode.X))


def format_float(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", action="append", required=True,
                        help="dataset id or extraction manifest path (repeatable)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-protected-after-freeze", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--derived-root", type=Path, default=common.DERIVED_ROOT)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable prediction table: {args.output}")
    provenance_path = args.output.with_name(args.output.name + ".provenance.json")
    if provenance_path.exists():
        raise SystemExit(f"refusing to overwrite {provenance_path}")

    manifests = [load_dataset_manifest(d, args.derived_root) for d in args.dataset]
    records = [record for manifest in manifests for record in manifest.records]
    protected_flags = {record.protected_test_used for record in records}
    confirmatory_gate_passed = False
    if True in protected_flags and args.allow_protected_after_freeze:
        confirmatory_gate_passed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_readiness.py"), "--stage", "confirmatory"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
    for record in records:
        try:
            enforce_protected_boundary(
                record.protected_test_used,
                explicitly_allowed=args.allow_protected_after_freeze,
                confirmatory_gate_passed=confirmatory_gate_passed,
            )
        except ValueError as error:
            raise SystemExit(f"{record.run_id}: {error}") from error
    protected_used = True in protected_flags

    model = LoadedModel(args.model_dir, args.device)
    if model.is_oracle and protected_used:
        raise SystemExit("the analysis-only oracle must never score protected episodes")
    if model.record.get("protected_test_used") is not False:
        raise SystemExit("refusing a model whose training record does not declare protected_test_used: false")

    rows: list[dict[str, str]] = []
    decision_sha: dict[str, str] = {}
    counts = {"decisions": 0, "eligible_positive": 0, "eligible_negative": 0, "excluded": 0}
    for record in sorted(records, key=lambda item: item.run_id):
        episode = load_episode_decisions(record, model.feature_names,
                                         allow_protected=record.protected_test_used is True)
        decision_sha[record.run_id] = episode.sha256
        scores = model.score_episode(episode)
        if scores.shape[0] != episode.y.shape[0]:
            raise AssertionError(f"{record.run_id}: score count differs from decision count")
        order = np.argsort(episode.decision_index, kind="stable")
        for position in order:
            eligibility = str(episode.eligibility[position])
            counts["decisions"] += 1
            counts[eligibility if eligibility in counts else "excluded"] += 1
            rows.append({
                "run_id": record.run_id,
                "decision_index": str(int(episode.decision_index[position])),
                "decision_time": repr(float(episode.decision_time[position])),
                "split": record.split,
                "map_id": record.map_id,
                "route_id": record.route_id,
                "fault_family": record.fault_family,
                "severity": record.severity,
                "seed": str(record.seed),
                "protected_test_used": "true" if record.protected_test_used is True else "false",
                "eligibility": eligibility,
                "label": str(int(episode.y[position])),
                "primary_event_class": record.primary_event_class or "",
                "primary_event_time": format_float(record.primary_event_time),
                "model_id": model.model_id,
                "raw_score": f"{float(scores[position]):.6f}",
                "risk_score": f"{float(scores[position]):.6f}",
            })
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue().encode("utf-8")
    publish_new_bytes(args.output, payload)
    provenance = {
        "schema_version": 1,
        "prediction_table": str(args.output),
        "prediction_table_sha256": sha256_bytes(payload),
        "model_dir": str(args.model_dir),
        "model_id": model.model_id,
        "checkpoint_sha256": model.checkpoint_sha256,
        "normalization_sha256": model.bundle.sha256 if model.bundle else None,
        "ablation": model.mask.name,
        "datasets": [{"dataset_id": m.dataset_id, "manifest": str(m.path), "manifest_sha256": m.sha256}
                     for m in manifests],
        "decisions_sha256_by_run": decision_sha,
        "counts": counts,
        "protected_test_used": protected_used,
        "confirmatory_gate_passed": confirmatory_gate_passed,
        "risk_score_equals_raw_score": True,
        "calibration_applied": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    publish_new_bytes(provenance_path, (json.dumps(provenance, sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps({"output": str(args.output), "rows": len(rows), "model_id": model.model_id,
                      **counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
