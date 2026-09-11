import json
from pathlib import Path

from scripts.monitor_campaign_status import append_history, snapshot


def append(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def test_monitor_resolves_declared_replacement(tmp_path):
    manifest = {
        "campaign_id": "parent",
        "expected_episode_count": 2,
        "infrastructure_replacements": [{
            "original_episode_key": "b",
            "replacement_campaign_id": "replacement",
            "replacement_episode_key": "b-r",
        }],
    }
    append(tmp_path / "logs/campaigns/parent.jsonl", {"episode_key": "a", "returncode": 0})
    append(tmp_path / "logs/campaigns/parent.jsonl", {"episode_key": "b", "returncode": 1})
    append(tmp_path / "logs/campaigns/replacement.jsonl", {"episode_key": "b-r", "returncode": 0})

    report = snapshot(manifest, root=tmp_path)
    assert report["state"] == "complete"
    assert report["counts"]["scientifically_resolved_design_keys"] == 2
    assert report["counts"]["unresolved_invalid_attempts"] == 0


def test_monitor_fails_closed_for_unresolved_invalid(tmp_path):
    manifest = {
        "campaign_id": "parent",
        "expected_episode_count": 1,
        "infrastructure_replacements": [],
    }
    append(tmp_path / "logs/campaigns/parent.jsonl", {"episode_key": "a", "returncode": 1})
    report = snapshot(manifest, root=tmp_path)
    assert report["state"] == "fail_stopped_unresolved_invalid"
    assert report["counts"]["scientifically_resolved_design_keys"] == 0


def test_monitor_history_is_append_only_jsonl(tmp_path):
    path = tmp_path / "history.jsonl"
    append_history(path, {"observed_utc": "first", "state": "running"})
    append_history(path, {"observed_utc": "second", "state": "complete"})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["observed_utc"] for row in rows] == ["first", "second"]


def test_monitor_fails_closed_for_duplicate_parent_ledger_key(tmp_path):
    manifest = {
        "campaign_id": "parent",
        "expected_episode_count": 1,
        "infrastructure_replacements": [],
    }
    append(tmp_path / "logs/campaigns/parent.jsonl", {"episode_key": "a", "returncode": 0})
    append(tmp_path / "logs/campaigns/parent.jsonl", {"episode_key": "a", "returncode": 0})
    report = snapshot(manifest, root=tmp_path)
    assert report["state"] == "fail_stopped_ledger_integrity"
    assert report["integrity"]["duplicate_parent_episode_keys"] == ["a"]


def test_monitor_fails_closed_for_off_manifest_key(tmp_path):
    manifest = {
        "campaign_id": "parent",
        "expected_episode_count": 1,
        "infrastructure_replacements": [],
        "episode_defaults": {},
        "design": {
            "seed_base": 1,
            "map_routes": {"dev_00": ["dev_00_r0"]},
            "conditions": [{
                "id": "clean", "family": "none", "severity": "none",
                "system": "s0", "replicates": [0],
            }],
        },
    }
    append(
        tmp_path / "logs/campaigns/parent.jsonl",
        {"episode_key": "not-in-design", "returncode": 0},
    )
    report = snapshot(manifest, root=tmp_path)
    assert report["state"] == "fail_stopped_ledger_integrity"
    assert report["integrity"]["unexpected_parent_episode_keys"] == ["not-in-design"]
    assert report["counts"]["remaining_design_keys"] == 1
