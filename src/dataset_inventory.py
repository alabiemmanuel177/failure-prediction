"""Build a hash-addressed episode inventory without extracting model features."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

import yaml


def publish_new_bytes(path: Path, payload: bytes) -> None:
    """Durably publish a new immutable artifact without an empty-file window."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _usable(summary: dict[str, Any], allow_protected: bool = False) -> bool:
    protected = summary.get("environment", {}).get("protected_test_used")
    return bool(
        summary.get("outcome", {}).get("terminal_state") != "invalid"
        and summary.get("provenance", {}).get("bag_mcap_count") == 1
        and summary.get("provenance", {}).get("bag_checksum_sha256")
        and (protected is False or (allow_protected and protected is True))
    )


def select_scientific_summaries(
    campaign_id: str,
    expected_keys: list[str],
    summaries: list[dict[str, Any]],
    replacement_specs: list[dict[str, Any]],
    allow_protected: bool = False,
) -> list[tuple[str, dict[str, Any], bool]]:
    """Resolve each design key to exactly one usable original or declared replacement."""
    summaries = [summary for summary in summaries if isinstance(summary, dict)]
    by_campaign_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for summary in summaries:
        identity = summary.get("identity", {})
        pair = (str(identity.get("campaign_id")), str(identity.get("episode_key")))
        by_campaign_key.setdefault(pair, []).append(summary)
    replacements = {str(item["original_episode_key"]): item for item in replacement_specs}
    selected = []
    for expected_key in expected_keys:
        originals = by_campaign_key.get((campaign_id, expected_key), [])
        spec = replacements.get(expected_key)
        missing_original_allowed = (
            not originals and spec is not None
            and (
                (spec.get("invalid_reason") == "ros_middleware_initialization_failure"
                 and spec.get("original_bag_mcap_count") == 0)
                or (spec.get("invalid_reason")
                    == "summary_and_bag_payload_loss_after_validation"
                    and spec.get("original_bag_mcap_count") == 1)
                or (spec.get("invalid_reason")
                    == "teardown_worker_isolation_failure_before_summary_publication"
                    and spec.get("original_bag_mcap_count") == 1)
            )
        )
        if len(originals) != 1 and not missing_original_allowed:
            raise ValueError(f"{expected_key}: expected one original summary, found {len(originals)}")
        original = originals[0] if originals else None
        if not spec and original is not None and _usable(original, allow_protected):
            selected.append((expected_key, original, False))
            continue
        if not spec:
            raise ValueError(f"{expected_key}: unusable original lacks a declared replacement")
        candidates = by_campaign_key.get((
            str(spec["replacement_campaign_id"]), str(spec["replacement_episode_key"])
        ), [])
        if len(candidates) != 1 or not _usable(candidates[0], allow_protected):
            raise ValueError(f"{expected_key}: declared replacement is absent or unusable")
        replacement = candidates[0]
        identity = replacement.get("identity", {})
        if (
            identity.get("replacement_for_episode_key") != expected_key
            or identity.get("replacement_for_run_id") != spec.get("original_run_id")
        ):
            raise ValueError(f"{expected_key}: replacement linkage differs from manifest")
        selected.append((expected_key, replacement, True))
    if len({item[1]["identity"]["run_id"] for item in selected}) != len(expected_keys):
        raise ValueError("one observed run resolves more than one scientific design cell")
    return selected


def episode_record(
    root: Path, design_key: str, summary: dict[str, Any], replacement: bool
) -> dict[str, Any]:
    identity = summary["identity"]
    environment = summary["environment"]
    label = summary["label_only"]
    outcome = summary["outcome"]
    provenance = summary["provenance"]
    bag_path = Path(provenance["bag_path"])
    health_path = Path(provenance["topic_health_sidecar"])
    event_path = Path(provenance["event_sidecar"])
    summary_path = root / "data/raw/summaries" / f"{identity['run_id']}.yaml"
    for name, path in {
        "bag": bag_path, "summary": summary_path,
        "topic-health sidecar": health_path, "event sidecar": event_path,
    }.items():
        if not path.exists():
            raise ValueError(f"{design_key}: {name} is missing: {path}")
    bag_bytes = sum(item.stat().st_size for item in bag_path.rglob("*") if item.is_file())
    return {
        "dataset_episode_key": design_key,
        "run_id": identity["run_id"],
        "observed_campaign_id": identity["campaign_id"],
        "observed_episode_key": identity["episode_key"],
        "infrastructure_replacement": replacement,
        "replacement_for_run_id": identity.get("replacement_for_run_id"),
        "map_id": environment["map_id"],
        "route_id": environment["route_id"],
        "system_id": environment["system_id"],
        "seed": environment["seed"],
        "split": environment["split"],
        "protected_test_used": environment["protected_test_used"],
        "fault_family": label["fault_family"],
        "severity": label.get("severity"),
        "primary_event_class": label.get("primary_event_class"),
        "terminal_state": outcome["terminal_state"],
        "success": outcome["success"],
        "collision": outcome["collision"],
        "timeout": outcome["timeout"],
        "duration_s": outcome["duration_s"],
        # Summaries created before the prospective profile transition did not carry
        # this field; those retained bags are the documented full_v1 default.
        "recording_profile": provenance.get("recording_profile", "full_v1"),
        "bag_bytes": bag_bytes,
        "bag_checksum_sha256": provenance["bag_checksum_sha256"],
        "summary_sha256": sha256_file(summary_path),
        "topic_health_sha256": sha256_file(health_path),
        "event_sidecar_sha256": sha256_file(event_path),
        "research1_platform_commit": identity["research1_platform_commit"],
        "research2_config_hash": identity["research2_config_hash"],
    }


def inventory_report(
    records: list[dict[str, Any]], *, dataset_id: str, source_hashes: dict[str, str],
    inventory_sha256: str,
    inventory_path: str = "data/manifests/balanced_pilot_v1.episodes.jsonl",
    collection_status: str = "development_collection_complete_training_admission_pending_human_gate",
    allow_protected: bool = False,
) -> dict[str, Any]:
    if not records:
        raise ValueError("dataset inventory is empty")
    protected_values = {item["protected_test_used"] for item in records}
    if protected_values == {True} and allow_protected:
        protected_flag = True
    elif protected_values == {False}:
        protected_flag = False
    else:
        raise ValueError("dataset inventory contains protected, mixed or unmarked episodes")
    count = len(records)
    events = sum(item["primary_event_class"] is not None for item in records)
    distributions = {}
    for output_name, field in (
        ("maps", "map_id"), ("routes", "route_id"), ("systems", "system_id"),
        ("fault_families", "fault_family"), ("terminal_states", "terminal_state"),
        ("event_classes", "primary_event_class"),
        ("recording_profiles", "recording_profile"),
    ):
        distributions[output_name] = dict(sorted(
            Counter(str(item[field]) for item in records if item[field] is not None).items()
        ))
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "analysis_unit": "episode",
        "status": collection_status,
        "protected_test_used": protected_flag,
        "counts": {
            "scientific_episodes": count,
            "terminal_events": events,
            "non_events": count - events,
            "infrastructure_replacements": sum(item["infrastructure_replacement"] for item in records),
        },
        "event_prevalence": events / count,
        "duration_seconds": {
            "total": sum(float(item["duration_s"]) for item in records),
            "minimum": min(float(item["duration_s"]) for item in records),
            "maximum": max(float(item["duration_s"]) for item in records),
        },
        "storage": {
            "bag_bytes": sum(int(item["bag_bytes"]) for item in records),
            "bag_gib": sum(int(item["bag_bytes"]) for item in records) / 2**30,
        },
        "distributions": distributions,
        "source_hashes": source_hashes,
        "episode_inventory": {
            "path": inventory_path,
            "sha256": inventory_sha256,
            "rows": count,
        },
        "label_status": {
            "episode_outcomes": "recorded_and_artifact_validated",
            "causal_window_labels": "not_admitted_pending_supervisor_and_second_review",
        },
        "training_admission": "forbidden_until_training_readiness_gate_passes",
    }


def jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for item in records
    )


def validate_inventory(
    root: Path, document: dict[str, Any], *, expected_split: str = "development",
    allow_protected: bool = False,
    source_paths: dict[str, Path] | None = None,
) -> list[str]:
    """Validate hashes, independence, and retained sidecars without reading 53 GiB of bags."""
    findings: list[str] = []
    inventory_info = document.get("episode_inventory", {})
    inventory_path = root / str(inventory_info.get("path", ""))
    if not inventory_path.is_file():
        return [f"episode inventory missing: {inventory_path}"]
    if sha256_file(inventory_path) != inventory_info.get("sha256"):
        findings.append("episode inventory SHA-256 differs from dataset manifest")
    records = []
    for line_number, line in enumerate(inventory_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError
            records.append(value)
        except (json.JSONDecodeError, ValueError):
            findings.append(f"episode inventory line {line_number} is not a JSON object")
    expected_rows = int(inventory_info.get("rows", -1))
    if len(records) != expected_rows:
        findings.append(f"episode inventory has {len(records)} rows, expected {expected_rows}")
    if len(records) != int(document.get("counts", {}).get("scientific_episodes", -1)):
        findings.append("episode inventory row count differs from scientific episode count")
    for key in ("run_id", "dataset_episode_key"):
        values = [item.get(key) for item in records]
        if len(set(values)) != len(values) or any(not value for value in values):
            findings.append(f"episode inventory {key} values are missing or duplicated")
    if any(item.get("split") != expected_split for item in records):
        findings.append(f"episode inventory contains a non-{expected_split} split")
    expected_protected = True if allow_protected and expected_split == "held_out_map_test" else False
    if any(item.get("protected_test_used") is not expected_protected for item in records):
        findings.append("episode inventory protection markers differ from the expected split")
    source_paths = source_paths or {
        "campaign_manifest_sha256": root / "data/manifests/balanced_pilot_v1.yaml",
        "pilot_report_sha256": root / "reports/pilot/balanced_pilot_v1.cumulative648.yaml",
    }
    for field, path in source_paths.items():
        if not path.is_file() or sha256_file(path) != document.get("source_hashes", {}).get(field):
            findings.append(f"source hash differs: {field}")
    for item in records:
        run_id = str(item.get("run_id", ""))
        summary_path = root / "data/raw/summaries" / f"{run_id}.yaml"
        if not summary_path.is_file():
            findings.append(f"{run_id}: retained summary missing")
            continue
        if sha256_file(summary_path) != item.get("summary_sha256"):
            findings.append(f"{run_id}: retained summary hash differs")
            continue
        summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
        for label, field, source_field in (
            ("topic-health", "topic_health_sha256", "topic_health_sidecar"),
            ("event", "event_sidecar_sha256", "event_sidecar"),
        ):
            sidecar = Path(summary["provenance"][source_field])
            if not sidecar.is_file() or sha256_file(sidecar) != item.get(field):
                findings.append(f"{run_id}: {label} sidecar missing or hash differs")
    return findings
