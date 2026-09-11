"""Shared, torch-free data plumbing for the learned predictors (work package M).

Everything here follows ``docs/model-development-contracts.md``: fitting reads
``sequences/`` of development datasets only, selection reads validation datasets
only, scoring reads ``decisions/``; the feature order is the frozen
``model_columns(primary)`` order and is asserted against every NPZ; ablations mask
channels *before* normalisation; normalisation is fitted on development data only
and stored as a ``NormalizationBundle``.

``torch`` is imported lazily inside the few helpers that need it so that the system
interpreter (no torch) can still import ``src.models``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from ..dataset_inventory import publish_new_bytes, sha256_file
from ..features import NormalizationBundle, load_primary_feature_set, model_columns
from ..features.schema import load_raw_feature_contract


ROOT = Path(__file__).resolve().parents[2]
FEATURE_SCHEMA_PATH = ROOT / "configs/feature_schema.yaml"
ABLATIONS_PATH = ROOT / "configs/ablations.yaml"
ALARM_POLICY_PATH = ROOT / "configs/alarm_policy.yaml"
DERIVED_ROOT = ROOT / "data/derived"

LEARNED_MODEL_IDS = ("p2_reconstruction_ae", "p3_causal_tcn", "p4_gru", "p5_compact_transformer")
ORACLE_MODEL_ID = "p6_oracle"
MODEL_IDS = (*LEARNED_MODEL_IDS, ORACLE_MODEL_ID)

ELIGIBLE_POSITIVE = "eligible_positive"
ELIGIBLE_NEGATIVE = "eligible_negative"
CLEAN_FAMILY = "none"


# --------------------------------------------------------------------------- hashing


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(document: Any) -> str:
    return sha256_text(json.dumps(document, sort_keys=True, separators=(",", ":")))


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=False,
            capture_output=True, text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


# --------------------------------------------------------------------------- seeding


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed python, numpy and (when available) torch; returns what was applied."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    applied: dict[str, Any] = {"seed": seed, "torch": False, "deterministic_algorithms": False}
    try:
        import torch
    except ImportError:  # pragma: no cover - system interpreter
        return applied
    torch.manual_seed(seed)
    applied["torch"] = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            applied["deterministic_algorithms"] = True
        except (RuntimeError, TypeError):  # pragma: no cover - older torch
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return applied


def count_parameters(module: Any, *, trainable_only: bool = False) -> int:
    return int(sum(
        parameter.numel() for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    ))


# --------------------------------------------------------------------------- features


def frozen_feature_names(schema_path: Path = FEATURE_SCHEMA_PATH) -> tuple[str, ...]:
    return model_columns(load_primary_feature_set(schema_path))


def primary_value_features(schema_path: Path = FEATURE_SCHEMA_PATH) -> tuple[str, ...]:
    return load_primary_feature_set(schema_path)


def masked_age_by_feature(
    schema_path: Path = FEATURE_SCHEMA_PATH, *, derived_masked_age: float = 0.0,
) -> dict[str, float]:
    """Age assigned to a masked feature: raw max_age_seconds, or 0.0 for derived ones."""
    raw = load_raw_feature_contract(schema_path)
    return {
        name: float(raw[name]["max_age_seconds"]) if name in raw else float(derived_masked_age)
        for name in load_primary_feature_set(schema_path)
    }


def assert_feature_names(observed: Sequence[str], expected: Sequence[str], *, context: str) -> None:
    observed_tuple = tuple(str(value) for value in observed)
    if observed_tuple != tuple(expected):
        raise ValueError(
            f"{context}: feature_names differ from the frozen model-column order "
            f"({len(observed_tuple)} vs {len(expected)} columns)"
        )


# --------------------------------------------------------------------------- manifests


@dataclass(frozen=True)
class EpisodeRecord:
    run_id: str
    split: str
    map_id: str
    route_id: str
    fault_family: str
    severity: str
    seed: int
    protected_test_used: object
    primary_event_class: str | None
    primary_event_time: float | None
    sequences_sha256: str | None
    dataset_id: str
    dataset_root: Path

    @property
    def sequences_path(self) -> Path:
        return self.dataset_root / "sequences" / f"{self.run_id}.npz"

    @property
    def decisions_path(self) -> Path:
        return self.dataset_root / "decisions" / f"{self.run_id}.npz"

    @property
    def is_clean(self) -> bool:
        return self.fault_family == CLEAN_FAMILY


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    path: Path
    sha256: str
    records: tuple[EpisodeRecord, ...]


def resolve_dataset(identifier: str | Path, derived_root: Path = DERIVED_ROOT) -> tuple[str, Path]:
    """Return (dataset_id, manifest path) for a dataset id or a manifest path."""
    candidate = Path(identifier)
    if candidate.suffix == ".jsonl" and candidate.is_file():
        return candidate.resolve().parent.name, candidate.resolve()
    manifest = derived_root / str(identifier) / "extraction_manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"extraction manifest not published for dataset {identifier!r}: {manifest} "
            "(the extraction driver writes it only after the whole inventory finishes)"
        )
    return str(identifier), manifest


def load_dataset_manifest(identifier: str | Path, derived_root: Path = DERIVED_ROOT) -> DatasetManifest:
    dataset_id, path = resolve_dataset(identifier, derived_root)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("run_id", "split", "map_id", "route_id", "fault_family", "severity", "seed"):
            if key not in row:
                raise ValueError(f"{path}:{line_number} lacks {key}")
        if "protected_test_used" not in row:
            raise ValueError(f"{path}:{line_number} lacks protected_test_used")
        event_time = row.get("primary_event_time")
        records.append(EpisodeRecord(
            run_id=str(row["run_id"]),
            split=str(row["split"]),
            map_id=str(row["map_id"]),
            route_id=str(row["route_id"]),
            fault_family=str(row["fault_family"]),
            severity=str(row["severity"]),
            seed=int(row["seed"]),
            protected_test_used=row["protected_test_used"],
            primary_event_class=row.get("primary_event_class"),
            primary_event_time=float(event_time) if event_time is not None else None,
            sequences_sha256=(row.get("artifact_sha256") or {}).get("sequences"),
            dataset_id=dataset_id,
            dataset_root=path.parent,
        ))
    if not records:
        raise ValueError(f"{path}: manifest is empty")
    if len({record.run_id for record in records}) != len(records):
        raise ValueError(f"{path}: duplicated run_id")
    return DatasetManifest(dataset_id, path, sha256_file(path), tuple(records))


def enforce_split(records: Iterable[EpisodeRecord], expected_split: str, *, purpose: str) -> None:
    """Refuse any row outside the expected split or with protected_test_used != False."""
    for record in records:
        if record.protected_test_used is not False:
            raise ValueError(
                f"{purpose}: {record.run_id} does not declare protected_test_used: false"
            )
        if record.split != expected_split:
            raise ValueError(
                f"{purpose}: {record.run_id} has split {record.split!r}; only "
                f"{expected_split!r} episodes may be used for {purpose}"
            )


def exclude_family(records: Sequence[EpisodeRecord], family: str | None, *, strict: bool = True) -> tuple[EpisodeRecord, ...]:
    """Drop every episode of ``family``.

    ``strict`` refuses a silent no-op (the family matches nothing). Callers that pool
    several manifests pass ``strict=False`` per manifest and apply the no-op check on
    the pooled result: a family may legitimately be absent from one manifest of the
    pool (the targeted development campaign carries no planner_oscillation episodes)
    while present in others.
    """
    if family is None:
        return tuple(records)
    if family == CLEAN_FAMILY:
        raise ValueError("the clean family cannot be excluded from fitting")
    kept = tuple(record for record in records if record.fault_family != family)
    if strict and len(kept) == len(records):
        raise ValueError(f"--exclude-family {family!r} matches no episode; refusing silently no-op fold")
    return kept


# --------------------------------------------------------------------------- arrays


@dataclass
class EpisodeArrays:
    record: EpisodeRecord
    X: np.ndarray                  # [n, T, C] float32
    y: np.ndarray                  # int8; sequences: {0,1}; decisions: {-1,0,1}
    decision_index: np.ndarray
    decision_time: np.ndarray
    eligibility: np.ndarray | None = None   # decisions only
    metadata: dict[str, Any] = field(default_factory=dict)
    sha256: str | None = None


def _load_npz(
    path: Path, expected_feature_names: Sequence[str], *, decisions: bool,
    allow_protected: bool = False,
) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as artifact:
        required = {"X", "y", "decision_index", "decision_time", "feature_names", "metadata_json"}
        if decisions:
            required.add("eligibility")
        missing = sorted(required - set(artifact.files))
        if missing:
            raise ValueError(f"{path}: artifact lacks arrays {missing}")
        arrays = {name: artifact[name] for name in required}
    assert_feature_names(arrays["feature_names"].tolist(), expected_feature_names, context=str(path))
    metadata = json.loads(str(arrays["metadata_json"].item()))
    protected = metadata.get("protected_test_used")
    if protected is not False and not (allow_protected and protected is True):
        raise ValueError(f"{path}: artifact does not declare protected_test_used: false")
    X = arrays["X"]
    if X.ndim != 3 or X.shape[2] != len(expected_feature_names):
        raise ValueError(f"{path}: X has shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{path}: X contains non-finite values")
    allowed = (-1, 0, 1) if decisions else (0, 1)
    if not np.isin(arrays["y"], allowed).all():
        raise ValueError(f"{path}: y outside {allowed}")
    return arrays, metadata


def load_episode_sequences(
    record: EpisodeRecord, expected_feature_names: Sequence[str], *, verify_sha256: bool = True,
) -> EpisodeArrays:
    path = record.sequences_path
    if not path.is_file():
        raise FileNotFoundError(f"sequence artifact missing for {record.run_id}: {path}")
    digest = sha256_file(path)
    if verify_sha256 and record.sequences_sha256 and digest != record.sequences_sha256:
        raise ValueError(f"{path}: sha256 differs from the extraction manifest")
    arrays, metadata = _load_npz(path, expected_feature_names, decisions=False)
    if metadata.get("run_id") != record.run_id or metadata.get("split") != record.split:
        raise ValueError(f"{path}: metadata identity differs from manifest row")
    return EpisodeArrays(
        record=record, X=arrays["X"].astype(np.float32, copy=False),
        y=arrays["y"].astype(np.int8), decision_index=arrays["decision_index"].astype(np.int64),
        decision_time=arrays["decision_time"].astype(np.float64), metadata=metadata, sha256=digest,
    )


def load_episode_decisions(
    record: EpisodeRecord, expected_feature_names: Sequence[str], *, allow_protected: bool = False,
) -> EpisodeArrays:
    """Load ``decisions/<run_id>.npz``; protected artifacts need the post-freeze path."""
    path = record.decisions_path
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"decisions/ artifacts are not published for dataset {record.dataset_id!r} "
            f"({path.parent}); run scripts/assemble_episode_decisions.py for that dataset "
            "before scoring"
        )
    if not path.is_file():
        raise FileNotFoundError(f"decision artifact missing for {record.run_id}: {path}")
    arrays, metadata = _load_npz(path, expected_feature_names, decisions=True,
                                 allow_protected=allow_protected)
    if metadata.get("run_id") != record.run_id:
        raise ValueError(f"{path}: metadata run_id differs from manifest row")
    eligibility = np.asarray([str(value) for value in arrays["eligibility"].tolist()])
    y = arrays["y"].astype(np.int8)
    positive = eligibility == ELIGIBLE_POSITIVE
    negative = eligibility == ELIGIBLE_NEGATIVE
    if not (np.all(y[positive] == 1) and np.all(y[negative] == 0)
            and np.all(y[~(positive | negative)] == -1)):
        raise ValueError(f"{path}: eligibility and y disagree")
    if np.any(np.diff(arrays["decision_time"].astype(float)) <= 0):
        raise ValueError(f"{path}: decisions are not strictly time ordered")
    return EpisodeArrays(
        record=record, X=arrays["X"].astype(np.float32, copy=False), y=y,
        decision_index=arrays["decision_index"].astype(np.int64),
        decision_time=arrays["decision_time"].astype(np.float64),
        eligibility=eligibility, metadata=metadata, sha256=sha256_file(path),
    )


def load_split_sequences(
    manifests: Sequence[DatasetManifest], expected_split: str, *, purpose: str,
    feature_names: Sequence[str], exclude: str | None = None,
) -> list[EpisodeArrays]:
    episodes = []
    total = kept_total = 0
    for manifest in manifests:
        records = exclude_family(manifest.records, exclude, strict=False)
        total += len(manifest.records)
        kept_total += len(records)
        enforce_split(records, expected_split, purpose=purpose)
        for record in records:
            episodes.append(load_episode_sequences(record, feature_names))
    if exclude is not None and kept_total == total:
        raise ValueError(f"--exclude-family {exclude!r} matches no episode across the {purpose} pool; "
                         "refusing silently no-op fold")
    return episodes


# --------------------------------------------------------------------------- ablations


@dataclass(frozen=True)
class FeatureMask:
    """Masked value features and the number of visible trailing time steps."""
    name: str | None
    masked_features: tuple[str, ...]
    visible_time_steps: int | None

    @property
    def is_identity(self) -> bool:
        return not self.masked_features and self.visible_time_steps is None


def load_ablations(path: Path = ABLATIONS_PATH) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "ablations" not in document or "feature_groups" not in document:
        raise ValueError(f"{path}: must declare feature_groups and ablations")
    return document


def ablation_kind(name: str | None, document: Mapping[str, Any] | None = None) -> str:
    if name is None:
        return "none"
    document = document or load_ablations()
    try:
        return str(document["ablations"][name]["kind"])
    except KeyError as error:
        raise ValueError(f"unknown ablation {name!r}; see configs/ablations.yaml") from error


def resolve_feature_mask(
    name: str | None, *, primary_features: Sequence[str], document: Mapping[str, Any] | None = None,
) -> FeatureMask:
    if name is None:
        return FeatureMask(None, (), None)
    document = document or load_ablations()
    spec = document["ablations"].get(name)
    if spec is None:
        raise ValueError(f"unknown ablation {name!r}; see configs/ablations.yaml")
    if spec.get("kind") != "feature":
        return FeatureMask(name, (), None)   # policy ablations leave the matrix untouched
    masked: list[str] = []
    for group in spec.get("remove_groups", []):
        members = document["feature_groups"].get(group)
        if members is None:
            raise ValueError(f"ablation {name!r} removes unknown group {group!r}")
        for feature in members:
            if feature not in primary_features:
                raise ValueError(f"group {group!r} lists non-primary feature {feature!r}")
            if feature not in masked:
                masked.append(feature)
    visible = spec.get("visible_time_steps")
    return FeatureMask(name, tuple(masked), int(visible) if visible is not None else None)


def apply_feature_mask(
    X: np.ndarray, feature_names: Sequence[str], mask: FeatureMask, *,
    masked_age: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Zero masked values, set missing=1 and age=max_age; mask early steps if requested."""
    if mask.is_identity:
        return X
    masked_age = masked_age or masked_age_by_feature()
    index = {name: position for position, name in enumerate(feature_names)}
    out = np.array(X, dtype=np.float32, copy=True)
    for feature in mask.masked_features:
        out[..., index[feature]] = 0.0
        out[..., index[f"{feature}__age_seconds"]] = masked_age[feature]
        out[..., index[f"{feature}__missing"]] = 1.0
    if mask.visible_time_steps is not None:
        steps = out.shape[-2]
        hidden = steps - int(mask.visible_time_steps)
        if hidden < 0:
            raise ValueError("visible_time_steps exceeds the window length")
        if hidden:
            for name, position in index.items():
                if name.endswith("__missing"):
                    out[..., :hidden, position] = 1.0
                elif name.endswith("__age_seconds"):
                    out[..., :hidden, position] = masked_age[name[: -len("__age_seconds")]]
                else:
                    out[..., :hidden, position] = 0.0
    return out


# --------------------------------------------------------------------------- normalisation


def fit_normalization(X: np.ndarray, feature_names: Sequence[str]) -> NormalizationBundle:
    """Development-only z-score bundle for value and age channels.

    Mirrors ``NormalizationBundle.fit`` (population variance, scale 1.0 when the
    variance is zero, missing samples excluded) using arrays; a feature with no
    observed development sample (for example a fully masked ablation group) gets
    mean 0 and scale 1 because its value channel is identically 0 after masking.
    """
    index = {name: position for position, name in enumerate(feature_names)}
    flat = np.asarray(X, dtype=np.float64).reshape(-1, len(feature_names))
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in feature_names:
        if name.endswith("__missing"):
            continue
        base = name[: -len("__age_seconds")] if name.endswith("__age_seconds") else name
        missing_column = index.get(f"{base}__missing")
        if name.endswith("__age_seconds") or missing_column is None:
            observed = flat[:, index[name]]
        else:
            observed = flat[flat[:, missing_column] == 0, index[name]]
        if observed.size == 0:
            means[name], scales[name] = 0.0, 1.0
            continue
        mean = float(observed.mean())
        std = float(math.sqrt(float(((observed - mean) ** 2).mean())))
        means[name] = mean
        scales[name] = std or 1.0
    return NormalizationBundle(means=means, scales=scales, fit_split="development")


def normalization_from_json(text: str) -> NormalizationBundle:
    document = json.loads(text)
    if document.get("fit_split") != "development":
        raise ValueError("normalization bundle must be fitted on development")
    return NormalizationBundle(
        means={str(k): float(v) for k, v in document["means"].items()},
        scales={str(k): float(v) for k, v in document["scales"].items()},
        fit_split="development",
    )


def apply_normalization(
    X: np.ndarray, feature_names: Sequence[str], bundle: NormalizationBundle,
) -> np.ndarray:
    """Vectorised ``NormalizationBundle.transform``: missing values become 0."""
    index = {name: position for position, name in enumerate(feature_names)}
    out = np.array(X, dtype=np.float32, copy=True)
    for name, mean in bundle.means.items():
        position = index.get(name)
        if position is None:
            raise ValueError(f"normalization bundle names unknown column {name!r}")
        column = (out[..., position] - mean) / bundle.scales[name]
        base = name[: -len("__age_seconds")] if name.endswith("__age_seconds") else name
        missing_column = index.get(f"{base}__missing")
        if not name.endswith("__age_seconds") and missing_column is not None:
            column = np.where(out[..., missing_column] > 0.5, 0.0, column)
        out[..., position] = column
    return out


def prepare_window(
    X: np.ndarray, feature_names: Sequence[str], mask: FeatureMask, bundle: NormalizationBundle,
    *, masked_age: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Mask first, normalise second (the frozen deployment order)."""
    return apply_normalization(apply_feature_mask(X, feature_names, mask, masked_age=masked_age),
                               feature_names, bundle)


# --------------------------------------------------------------------------- sampling


@dataclass
class TrainingSet:
    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    run_ids: np.ndarray
    summary: dict[str, Any]


def subsample_negatives(
    episode: EpisodeArrays, *, cap_per_episode: int | None, phase_bins: int, rng: np.random.Generator,
) -> np.ndarray:
    """Indices kept for fitting: every positive, negatives capped per episode.

    Negatives are thinned by operating phase: the episode is cut into
    ``phase_bins`` contiguous time chunks and the cap is spread evenly across them so
    long clean episodes neither dominate nor lose whole phases.
    """
    positive = np.flatnonzero(episode.y == 1)
    negative = np.flatnonzero(episode.y == 0)
    if cap_per_episode is None or negative.size <= cap_per_episode:
        return np.sort(np.concatenate([positive, negative]))
    order = negative[np.argsort(episode.decision_time[negative], kind="stable")]
    chunks = np.array_split(order, max(1, min(phase_bins, order.size)))
    quotas = [cap_per_episode // len(chunks)] * len(chunks)
    for offset in range(cap_per_episode - sum(quotas)):
        quotas[offset % len(chunks)] += 1
    kept = [positive]
    spare = 0
    for chunk, quota in zip(chunks, quotas):
        take = min(chunk.size, quota + spare)
        spare = quota + spare - take
        kept.append(rng.choice(chunk, size=take, replace=False))
    return np.sort(np.concatenate(kept))


def build_training_set(
    episodes: Sequence[EpisodeArrays], *, cap_per_episode: int | None, phase_bins: int,
    positive_weighting: str, seed: int, negatives_only: bool = False, clean_only: bool = False,
) -> TrainingSet:
    """Episode-grouped subsampling plus class/event weights from training data only."""
    if positive_weighting not in {"class", "event"}:
        raise ValueError("positive_weighting must be class or event")
    rng = np.random.default_rng(int(seed))
    natural_positive = sum(int((episode.y == 1).sum()) for episode in episodes)
    natural_negative = sum(int((episode.y == 0).sum()) for episode in episodes)
    xs, ys, ws, ids = [], [], [], []
    per_episode_positive: dict[str, int] = {}
    used_episodes = 0
    for episode in sorted(episodes, key=lambda item: item.record.run_id):
        if clean_only and not episode.record.is_clean:
            continue
        kept = subsample_negatives(episode, cap_per_episode=cap_per_episode,
                                   phase_bins=phase_bins, rng=rng)
        if negatives_only:
            kept = kept[episode.y[kept] == 0]
        if kept.size == 0:
            continue
        used_episodes += 1
        xs.append(episode.X[kept])
        ys.append(episode.y[kept])
        ids.append(np.full(kept.size, episode.record.run_id))
        per_episode_positive[episode.record.run_id] = int((episode.y[kept] == 1).sum())
    if not xs:
        raise ValueError("no fitting windows remain after subsampling")
    X = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int8)
    run_ids = np.concatenate(ids)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    weights = np.ones(y.shape[0], dtype=np.float32)
    pos_weight = negatives / positives if positives else 1.0
    if positives and positive_weighting == "event":
        event_episodes = [count for count in per_episode_positive.values() if count]
        mean_positive = float(np.mean(event_episodes))
        for position in np.flatnonzero(y == 1):
            weights[position] = mean_positive / per_episode_positive[str(run_ids[position])]
    weights[y == 1] *= pos_weight
    rebalanced = float(weights[y == 1].sum() / weights.sum()) if positives else 0.0
    natural_total = natural_positive + natural_negative
    summary = {
        "episodes_used": used_episodes,
        "natural_positive_windows": natural_positive,
        "natural_negative_windows": natural_negative,
        "natural_prevalence": natural_positive / natural_total if natural_total else None,
        "fitting_positive_windows": positives,
        "fitting_negative_windows": negatives,
        "subsampled_prevalence": positives / (positives + negatives) if positives + negatives else None,
        "rebalanced_prevalence": rebalanced,
        "positive_weighting": positive_weighting,
        "pos_weight": float(pos_weight),
        "negative_cap_per_episode": cap_per_episode,
        "phase_bins": phase_bins,
        "sampling_seed": int(seed),
    }
    return TrainingSet(X=X, y=y, weights=weights, run_ids=run_ids, summary=summary)


# --------------------------------------------------------------------------- metrics


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Window-level AUPRC (step-wise average precision, no interpolation)."""
    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y_true and scores must be 1-D arrays of equal length")
    positives = int((y == 1).sum())
    if positives == 0 or positives == y.size:
        return math.nan
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    s_sorted = s[order]
    true_positive = np.cumsum(y_sorted == 1)
    rank = np.arange(1, y.size + 1)
    # evaluate only at the last index of each distinct score (ties share a threshold)
    distinct = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])
    precision = true_positive[distinct] / rank[distinct]
    recall = true_positive[distinct] / positives
    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    if not positives or not negatives:
        return math.nan
    order = np.argsort(s, kind="stable")
    ranks = np.empty(s.size, dtype=np.float64)
    sorted_scores = s[order]
    start = 0
    while start < s.size:
        end = start
        while end + 1 < s.size and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


# --------------------------------------------------------------------------- torch glue


def torch_device(requested: str | None = None):
    import torch
    if requested in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def build_model(model_id: str, config: Mapping[str, Any]):
    """Instantiate a learned predictor from its YAML config (lazy torch import)."""
    architecture = dict(config.get("architecture", {}))
    inputs = dict(config.get("input", {}))
    channels = int(inputs.get("channels", 84))
    steps = int(inputs.get("time_steps", 10))
    if model_id == "p3_causal_tcn":
        from .tcn import CausalTCN
        return CausalTCN(channels, **architecture)
    if model_id == "p4_gru":
        from .recurrent import GRUPredictor
        return GRUPredictor(channels, **architecture)
    if model_id == "p5_compact_transformer":
        from .transformer import CompactCausalTransformer
        return CompactCausalTransformer(channels, time_steps=steps, **architecture)
    if model_id == "p2_reconstruction_ae":
        from .autoencoder import ReconstructionAutoencoder
        return ReconstructionAutoencoder(channels * steps, **architecture)
    raise ValueError(f"no learned architecture for model_id {model_id!r}")


def load_model_config(path: Path, expected_model_id: str | None = None) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "model_id" not in config:
        raise ValueError(f"{path}: model config must declare model_id")
    if expected_model_id and config["model_id"] != expected_model_id:
        raise ValueError(f"{path}: model_id {config['model_id']!r} differs from {expected_model_id!r}")
    return config


def write_json(path: Path, document: Any) -> str:
    payload = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode("utf-8")
    publish_new_bytes(path, payload)
    return sha256_bytes(payload)


__all__ = [
    "ABLATIONS_PATH", "ALARM_POLICY_PATH", "CLEAN_FAMILY", "DERIVED_ROOT", "ELIGIBLE_NEGATIVE",
    "ELIGIBLE_POSITIVE", "FEATURE_SCHEMA_PATH", "LEARNED_MODEL_IDS", "MODEL_IDS", "ORACLE_MODEL_ID",
    "ROOT", "DatasetManifest", "EpisodeArrays", "EpisodeRecord", "FeatureMask", "TrainingSet",
    "ablation_kind", "apply_feature_mask", "apply_normalization", "assert_feature_names", "auroc",
    "average_precision", "build_model", "build_training_set", "count_parameters", "enforce_split",
    "exclude_family", "fit_normalization", "frozen_feature_names", "git_commit", "load_ablations",
    "load_dataset_manifest", "load_episode_decisions", "load_episode_sequences", "load_model_config",
    "load_split_sequences", "masked_age_by_feature", "normalization_from_json", "prepare_window",
    "primary_value_features", "publish_new_bytes", "resolve_dataset", "resolve_feature_mask",
    "seed_everything", "sha256_bytes", "sha256_file", "sha256_json", "sha256_text",
    "subsample_negatives", "torch_device", "write_json",
]
