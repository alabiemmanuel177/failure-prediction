import pytest

from src.experiments import summarize_pilot, summarize_pilot_wave


def summary(key, family="none", terminal="success", event=None, protected=False):
    return {
        "identity": {"campaign_id": "pilot", "episode_key": key},
        "environment": {"map_id": "dev_00", "protected_test_used": protected},
        "label_only": {"fault_family": family, "primary_event_class": event},
        "outcome": {"terminal_state": terminal, "duration_s": 10.0},
        "provenance": {"bag_mcap_count": 1, "bag_checksum_sha256": "abc"},
    }


def test_pilot_inventory_counts_independent_episodes():
    expected = [{"episode_key": "a"}, {"episode_key": "b"}]
    report = summarize_pilot(
        "pilot", expected,
        [summary("a"), summary("b", "lidar_dropout", "collision", "collision")],
    )
    assert report["complete_and_artifact_valid"] is True
    assert report["counts"] == {
        "ledger_artifact_invalid_attempts": 0,
        "expected": 2, "observed": 2, "usable": 2,
        "infrastructure_invalid_attempts": 0, "infrastructure_replacements": 0,
        "terminal_events": 1, "non_events": 1,
    }
    assert report["event_prevalence"] == 0.5


def test_pilot_summaries_ignore_empty_yaml_documents(tmp_path):
    report = summarize_pilot("pilot", [{"episode_key": "a"}], [None, summary("a")])
    assert report["complete_and_artifact_valid"] is True
    item = summary("a")
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "data.mcap").write_bytes(b"bag")
    item["provenance"].update({"bag_path": str(bag), "recording_profile": "compact_v2"})
    report = summarize_pilot_wave(
        campaign_id="pilot", wave=1,
        ledger_rows=[{"episode_key": "a", "returncode": 0}],
        summaries=[None, item], episodes_per_wave=1,
    )
    assert report["counts"]["usable"] == 1


def test_wave_accepts_declared_zero_bag_unmaterialized_attempt(tmp_path):
    replacement = summary("a-replacement")
    replacement["identity"].update({
        "campaign_id": "replacement", "replacement_for_episode_key": "a",
        "replacement_for_run_id": "run-a",
    })
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "data.mcap").write_bytes(b"bag")
    replacement["provenance"].update({
        "bag_path": str(bag), "recording_profile": "compact_v2",
    })
    spec = {
        "original_episode_key": "a", "original_run_id": "run-a",
        "invalid_reason": "ros_middleware_initialization_failure",
        "original_bag_mcap_count": 0,
        "replacement_campaign_id": "replacement",
        "replacement_episode_key": "a-replacement",
    }
    report = summarize_pilot_wave(
        campaign_id="pilot", wave=1,
        ledger_rows=[{"episode_key": "a", "returncode": 1}],
        summaries=[replacement], episodes_per_wave=1, replacement_specs=[spec],
    )
    assert report["counts"]["usable"] == 1
    assert report["counts"]["infrastructure_invalid_attempts"] == 1
    assert report["counts"]["infrastructure_replacements"] == 1


def test_pilot_inventory_exposes_duplicates_invalid_and_protected_use():
    expected = [{"episode_key": "a"}, {"episode_key": "b"}]
    report = summarize_pilot(
        "pilot", expected,
        [summary("a", terminal="invalid"), summary("a", protected=True)],
    )
    assert report["complete_and_artifact_valid"] is False
    assert report["protected_test_used"] is True
    assert report["integrity"]["duplicate_episode_keys"] == ["a"]
    assert report["integrity"]["missing_episode_keys"] == ["b"]


def test_wave_report_uses_ledger_slice_and_episode_artifacts(tmp_path):
    summaries = []
    ledger = []
    for index in range(4):
        key = f"episode-{index}"
        bag = tmp_path / key
        bag.mkdir()
        (bag / "data.mcap").write_bytes(b"x" * (index + 1))
        item = summary(
            key, terminal="collision" if index == 3 else "success",
            event="collision" if index == 3 else None,
        )
        item["provenance"].update({"bag_path": str(bag), "recording_profile": "compact_v1"})
        item["label_only"]["perception_metrics"] = {} if index % 2 else None
        summaries.append(item)
        ledger.append({"episode_key": key, "returncode": 0})
    report = summarize_pilot_wave(
        campaign_id="pilot", wave=2, ledger_rows=ledger, summaries=summaries,
        episodes_per_wave=2,
    )
    assert report["episode_keys"] == ["episode-2", "episode-3"]
    assert report["counts"]["terminal_events"] == 1
    assert report["counts"]["collisions"] == 1
    assert report["bag_gib"] == pytest.approx(7 / 2**30)
    assert report["perception_metric_summaries"] == 1


def test_wave_report_carries_dispatcher_provenance_only_for_parallel_waves(tmp_path):
    summaries = []
    ledger = []
    for index in range(2):
        key = f"episode-{index}"
        bag = tmp_path / key
        bag.mkdir()
        (bag / "data.mcap").write_bytes(b"x")
        item = summary(key)
        item["provenance"].update({"bag_path": str(bag), "recording_profile": "compact_v2"})
        summaries.append(item)
        ledger.append({
            "episode_key": key, "returncode": 0, "system": "s3" if index else "s0",
            "dispatcher": "run_campaign_parallel_v1", "concurrency_workers": 6,
            "system_concurrency": {"s3": 1}, "worker_slot": index * 3,
            "ros_domain_id": 60 + index * 3, "gz_partition": f"research2_w{index * 3}",
        })
    report = summarize_pilot_wave(
        campaign_id="pilot", wave=1, ledger_rows=ledger, summaries=summaries, episodes_per_wave=2,
    )
    assert report["execution_provenance"] == {
        "dispatcher": ["run_campaign_parallel_v1"], "dispatched_rows": 2, "sequential_rows": 0,
        "concurrency_workers": [6], "system_concurrency": {"s3": 1},
        "worker_slots": [0, 3], "ros_domain_ids": [60, 63],
        "gz_partitions": ["research2_w0", "research2_w3"],
        "episodes_by_system": {"s0": 1, "s3": 1},
    }
    sequential = summarize_pilot_wave(
        campaign_id="pilot", wave=1, summaries=summaries, episodes_per_wave=2,
        ledger_rows=[{"episode_key": row["episode_key"], "returncode": 0} for row in ledger],
    )
    assert "execution_provenance" not in sequential


def test_wave_report_fails_closed_on_failed_or_protected_row(tmp_path):
    ledger = [{"episode_key": "a", "returncode": 1}]
    with pytest.raises(ValueError, match="failed ledger"):
        summarize_pilot_wave(
            campaign_id="pilot", wave=1, ledger_rows=ledger,
            summaries=[summary("a")], episodes_per_wave=1,
        )


def test_inventory_substitutes_only_a_manifest_linked_usable_replacement(tmp_path):
    original = summary("a", terminal="invalid")
    original["identity"]["run_id"] = "original-run"
    original["provenance"].update({"bag_mcap_count": 0, "bag_checksum_sha256": None})
    replacement = summary("a-replacement")
    replacement["identity"].update({
        "campaign_id": "replacement-campaign",
        "replacement_for_episode_key": "a",
        "replacement_for_run_id": "original-run",
    })
    report = summarize_pilot(
        "pilot", [{"episode_key": "a"}], [original, replacement],
        [{
            "original_episode_key": "a", "original_run_id": "original-run",
            "replacement_campaign_id": "replacement-campaign",
            "replacement_episode_key": "a-replacement",
        }],
    )
    assert report["complete_and_artifact_valid"] is True
    assert report["counts"]["usable"] == 1
    assert report["counts"]["infrastructure_invalid_attempts"] == 1
    assert report["counts"]["infrastructure_replacements"] == 1
    assert report["by_attempt_terminal_state"] == {"invalid": 1}


def test_ledger_artifact_invalid_original_is_not_scientific_until_replaced():
    """An attempt that reached a terminal state but failed artifact validation (ledger
    returncode != 0) is excluded; the campaign is incomplete until its cell is replaced."""
    from src.experiments.pilot_audit import summarize_pilot
    expected = [{"episode_key": "m-r0-fam-r0"}, {"episode_key": "m-r0-fam-r1"}]
    def summary(key, run_id):
        return {"identity": {"campaign_id": "c", "episode_key": key, "run_id": run_id},
                "outcome": {"terminal_state": "success", "duration_s": 1.0},
                "label_only": {"fault_family": "fam", "primary_event_class": None},
                "environment": {"map_id": "m", "protected_test_used": False, "split": "validation"},
                "provenance": {"bag_mcap_count": 1, "bag_checksum_sha256": "deadbeef"}}
    summaries = [summary("m-r0-fam-r0", "a"), summary("m-r0-fam-r1", "b")]
    report = summarize_pilot("c", expected, summaries, ledger_invalid_keys={"m-r0-fam-r1"})
    assert report["counts"]["usable"] == 1
    assert report["counts"]["ledger_artifact_invalid_attempts"] == 1
    assert report["complete_and_artifact_valid"] is False
    assert summarize_pilot("c", expected, summaries)["complete_and_artifact_valid"] is True
