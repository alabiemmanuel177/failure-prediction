import hashlib

import pytest

from src.dataset_inventory import (
    episode_record, inventory_report, jsonl_bytes, publish_new_bytes,
    select_scientific_summaries, validate_inventory,
)


def summary(campaign, key, run, *, usable=True, protected=False, replacement_for=None):
    return {
        "identity": {
            "campaign_id": campaign, "episode_key": key, "run_id": run,
            "replacement_for_episode_key": replacement_for,
            "replacement_for_run_id": "bad-run" if replacement_for else None,
        },
        "environment": {"protected_test_used": protected},
        "outcome": {"terminal_state": "success" if usable else "invalid"},
        "provenance": {
            "bag_mcap_count": 1 if usable else 0,
            "bag_checksum_sha256": "hash" if usable else None,
        },
    }


def test_scientific_selection_resolves_one_declared_replacement():
    original = summary("pilot", "a", "run-a", usable=False)
    replacement = summary("replacement", "a-replacement", "run-r", replacement_for="a")
    replacement["identity"]["replacement_for_run_id"] = "run-a"
    selected = select_scientific_summaries("pilot", ["a"], [None, original, replacement], [{
        "original_episode_key": "a", "original_run_id": "run-a",
        "replacement_campaign_id": "replacement",
        "replacement_episode_key": "a-replacement",
    }])
    assert selected == [("a", replacement, True)]


def test_scientific_selection_allows_declared_unmaterialized_original():
    replacement = summary("replacement", "a-replacement", "run-r", replacement_for="a")
    replacement["identity"]["replacement_for_run_id"] = "run-a"
    selected = select_scientific_summaries("pilot", ["a"], [replacement], [{
        "original_episode_key": "a", "original_run_id": "run-a",
        "invalid_reason": "ros_middleware_initialization_failure",
        "original_bag_mcap_count": 0,
        "replacement_campaign_id": "replacement",
        "replacement_episode_key": "a-replacement",
    }])
    assert selected == [("a", replacement, True)]


def test_scientific_selection_rejects_protected_or_bad_linkage():
    protected = summary("pilot", "a", "run-a", protected=True)
    with pytest.raises(ValueError, match="lacks a declared replacement"):
        select_scientific_summaries("pilot", ["a"], [protected], [])


def test_inventory_report_is_episode_level_and_hash_addressed():
    records = [{
        "run_id": "run", "map_id": "dev", "route_id": "route", "system_id": "S0",
        "fault_family": "none", "primary_event_class": None, "terminal_state": "success",
        "recording_profile": "compact_v2", "protected_test_used": False,
        "infrastructure_replacement": False, "duration_s": 10.0, "bag_bytes": 100,
    }]
    payload = jsonl_bytes(records)
    digest = hashlib.sha256(payload).hexdigest()
    report = inventory_report(
        records, dataset_id="dataset", source_hashes={"source": "hash"},
        inventory_sha256=digest,
    )
    assert report["analysis_unit"] == "episode"
    assert report["counts"]["non_events"] == 1
    assert report["episode_inventory"]["sha256"] == digest
    assert report["protected_test_used"] is False


def test_legacy_recording_profile_defaults_to_full_v1(tmp_path):
    run_id = "legacy-run"
    bag = tmp_path / "data/raw/bags" / run_id
    bag.mkdir(parents=True)
    (bag / "data.mcap").write_bytes(b"bag")
    summary_path = tmp_path / "data/raw/summaries" / f"{run_id}.yaml"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text("legacy: true\n", encoding="utf-8")
    health = tmp_path / "health.json"; health.write_text("{}", encoding="utf-8")
    events = tmp_path / "events.json"; events.write_text("[]", encoding="utf-8")
    value = {
        "identity": {"run_id": run_id, "campaign_id": "pilot", "episode_key": "key",
                     "replacement_for_run_id": None, "research1_platform_commit": "commit",
                     "research2_config_hash": "config"},
        "environment": {"map_id": "dev", "route_id": "route", "system_id": "S0",
                        "seed": 1, "split": "development", "protected_test_used": False},
        "label_only": {"fault_family": "none", "severity": None,
                       "primary_event_class": None},
        "outcome": {"terminal_state": "success", "success": True, "collision": False,
                    "timeout": False, "duration_s": 1.0},
        "provenance": {"bag_path": str(bag), "bag_checksum_sha256": "hash",
                       "topic_health_sidecar": str(health), "event_sidecar": str(events)},
    }
    assert episode_record(tmp_path, "key", value, False)["recording_profile"] == "full_v1"


def test_inventory_validation_fails_on_hash_mismatch(tmp_path):
    inventory = tmp_path / "episodes.jsonl"
    inventory.write_text('{"run_id":"run","dataset_episode_key":"key",'
                         '"split":"development","protected_test_used":false}\n',
                         encoding="utf-8")
    document = {
        "counts": {"scientific_episodes": 1},
        "episode_inventory": {"path": "episodes.jsonl", "sha256": "wrong", "rows": 1},
        "source_hashes": {},
    }
    assert "episode inventory SHA-256 differs" in validate_inventory(tmp_path, document)[0]


def test_publish_new_bytes_is_durable_and_refuses_overwrite(tmp_path):
    target = tmp_path / "nested" / "inventory.jsonl"
    publish_new_bytes(target, b'{"run_id":"run"}\n')
    assert target.read_bytes() == b'{"run_id":"run"}\n'
    assert list(target.parent.glob(".*.tmp")) == []
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_new_bytes(target, b"replacement\n")
    assert target.read_bytes() == b'{"run_id":"run"}\n'
