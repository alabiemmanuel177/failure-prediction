from pathlib import Path

import yaml

from src.artifact_integrity import audit_raw_payloads


def create_zero_run(root: Path, run_id: str):
    summary = root / "data/raw/summaries" / f"{run_id}.yaml"
    bag = root / "data/raw/bags" / run_id
    summary.parent.mkdir(parents=True, exist_ok=True)
    bag.mkdir(parents=True, exist_ok=True)
    summary.write_bytes(b"")
    (bag / f"{run_id}_0.mcap").write_bytes(b"")
    (bag / "metadata.yaml").write_bytes(b"")


def test_payload_audit_rejects_undeclared_zero_files(tmp_path):
    create_zero_run(tmp_path, "lost")
    report = audit_raw_payloads(tmp_path)
    assert report["passed"] is False
    assert report["unexpected_zero_length_runs"] == ["lost"]


def test_payload_audit_accepts_exact_declared_loss_evidence(tmp_path):
    create_zero_run(tmp_path, "lost")
    manifests = tmp_path / "data/manifests"
    manifests.mkdir(parents=True)
    (manifests / "campaign.yaml").write_text(yaml.safe_dump({
        "infrastructure_replacements": [{
            "original_run_id": "lost",
            "original_episode_key": "episode",
            "invalid_reason": "summary_and_bag_payload_loss_after_validation",
            "replacement_campaign_id": "replacement",
            "replacement_episode_key": "episode-r",
        }],
    }), encoding="utf-8")
    report = audit_raw_payloads(tmp_path)
    assert report["passed"] is True
    assert report["counts"]["declared_payload_loss_runs"] == 1
    assert report["declared_evidence"]["lost"] == ["mcap", "metadata", "summary"]


def test_payload_audit_accepts_declared_pre_goal_termination(tmp_path):
    # A killed pre-goal attempt leaves only an empty MCAP: no summary, no metadata.
    bag = tmp_path / "data/raw/bags/killed"
    bag.mkdir(parents=True)
    (bag / "killed_0.mcap").write_bytes(b"")
    (tmp_path / "data/raw/summaries").mkdir(parents=True)
    manifests = tmp_path / "data/manifests"
    manifests.mkdir(parents=True)
    assert audit_raw_payloads(tmp_path)["passed"] is False
    (manifests / "interruption.yaml").write_text(yaml.safe_dump({
        "infrastructure_replacements": [{
            "original_run_id": "killed",
            "original_episode_key": "episode",
            "invalid_reason": "pre_goal_process_termination",
            "replacement_campaign_id": "parent",
            "replacement_episode_key": "episode",
        }],
    }), encoding="utf-8")
    report = audit_raw_payloads(tmp_path)
    assert report["passed"] is True
    assert report["declared_evidence"]["killed"] == ["mcap", "no_metadata_file", "no_summary_file"]
    # The same declaration is incomplete for a run that also has an empty summary.
    (tmp_path / "data/raw/summaries/killed.yaml").write_bytes(b"")
    assert audit_raw_payloads(tmp_path)["passed"] is False
