from pathlib import Path

import yaml

from scripts import run_live_integrity_campaign as live
from scripts.run_balanced_pilot_replacements import summary_for_episode


def write_summary(root: Path, name: str, campaign: str, episode: str) -> Path:
    path = root / "summaries" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "identity": {"campaign_id": campaign, "episode_key": episode},
    }), encoding="utf-8")
    return path


def test_summary_scanners_ignore_empty_yaml(tmp_path, monkeypatch):
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    (summaries / "empty.yaml").write_text("", encoding="utf-8")
    expected = write_summary(tmp_path, "valid.yaml", "campaign", "episode")

    assert summary_for_episode("campaign", "episode", tmp_path)["identity"][
        "episode_key"
    ] == "episode"

    observed = []

    class Result:
        returncode = 0

    def fake_run(command, check):
        observed.append((command, check))
        return Result()

    monkeypatch.setattr(live.subprocess, "run", fake_run)
    assert live.validate_completed_artifact("campaign", "episode", tmp_path) == 0
    assert observed[0][0][-1] == str(expected)


def test_summary_for_episode_returns_none_when_only_empty_yaml_exists(tmp_path):
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    (summaries / "empty.yaml").write_text("", encoding="utf-8")
    assert summary_for_episode("campaign", "episode", tmp_path) is None
