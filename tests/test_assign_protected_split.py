import hashlib
from pathlib import Path
import shutil

import pytest
import yaml

from confirmatory_fixtures import ROOT, frozen_model_freeze, unassigned_splits_text
from scripts.assign_protected_split import main, replace_held_out_block
from scripts.research_log import read_records, verify


def fake_research1(root: Path, *, routes_per_map: int = 8, maps=("test_00", "test_01", "test_02")) -> Path:
    for map_id in maps:
        map_dir = root / "data/test" / map_id
        map_dir.mkdir(parents=True)
        (map_dir / "map.yaml").write_text(f"image: map.pgm\nmap: {map_id}\n", encoding="utf-8")
        (map_dir / "map.pgm").write_bytes(b"P5 1 1 255\n\x00")
        routes = [
            {"route_id": f"{map_id}_r{index}", "start": {"x": 0.0, "y": 0.0, "yaw": 0.0},
             "goal": {"x": 1.0, "y": float(index)}, "shortest_path_m": 1.0 + index,
             "length_class": "short", "tags": [], "s0_clean_verified": True}
            for index in range(routes_per_map)
        ]
        (root / "configs/routes").mkdir(parents=True, exist_ok=True)
        (root / "configs/routes" / f"{map_id}.yaml").write_text(
            yaml.safe_dump({"map_id": map_id, "frozen": True, "map_hash": "x", "routes": routes}),
            encoding="utf-8",
        )
    return root


def fake_repo(tmp_path: Path, *, frozen: bool = True) -> dict[str, Path]:
    paths = {
        "splits": tmp_path / "repo/data/manifests/splits.template.yaml",
        "freeze": tmp_path / "repo/configs/model_freeze.yaml",
        "alarm": tmp_path / "repo/configs/alarm_policy.yaml",
        "record": tmp_path / "repo/reports/confirmatory/protected_split_assignment.yaml",
        "log": tmp_path / "repo/logs/research-log.jsonl",
    }
    paths["splits"].parent.mkdir(parents=True)
    paths["freeze"].parent.mkdir(parents=True)
    # The workspace manifest may already carry the post-freeze assignment; the fake
    # repository always starts from the pre-assignment block.
    paths["splits"].write_text(unassigned_splits_text(), encoding="utf-8")
    if frozen:
        paths["freeze"].write_text(yaml.safe_dump(frozen_model_freeze()), encoding="utf-8")
    alarm = yaml.safe_load((ROOT / "configs/alarm_policy.yaml").read_text(encoding="utf-8"))
    alarm["threshold"] = 0.42
    paths["alarm"].write_text(yaml.safe_dump(alarm), encoding="utf-8")
    return paths


def argv(paths: dict[str, Path], research1: Path, *extra: str) -> list[str]:
    return [
        "--research1-root", str(research1), "--splits", str(paths["splits"]),
        "--model-freeze", str(paths["freeze"]), "--alarm", str(paths["alarm"]),
        "--record", str(paths["record"]), "--research-log", str(paths["log"]),
        "--no-readiness-subprocess", *extra,
    ]


def test_dry_run_never_reads_the_test_directory(tmp_path, capsys):
    paths = fake_repo(tmp_path, frozen=False)
    missing_root = tmp_path / "does-not-exist"
    assert main(argv(paths, missing_root, "--dry-run")) == 1
    output = capsys.readouterr().out
    assert '"test_directory_read": false' in output
    assert "model freeze missing" in output
    paths = fake_repo(tmp_path / "frozen", frozen=True)
    assert main(argv(paths, missing_root, "--dry-run")) == 0
    assert not paths["record"].exists()


def test_refuses_before_freeze_and_without_confirmation(tmp_path):
    research1 = fake_research1(tmp_path / "r1")
    paths = fake_repo(tmp_path, frozen=False)
    with pytest.raises(SystemExit, match="model freeze missing"):
        main(argv(paths, research1))
    paths = fake_repo(tmp_path / "frozen", frozen=True)
    with pytest.raises(SystemExit, match="--i-confirm-model-freeze"):
        main(argv(paths, research1))
    assert not paths["record"].exists()


def test_assigns_three_maps_times_six_routes_preserving_other_content(tmp_path):
    # Protocol Amendment PA-2026-09-03-02: six frozen routes per held-out map.
    research1 = fake_research1(tmp_path / "r1", routes_per_map=6)
    paths = fake_repo(tmp_path)
    before = paths["splits"].read_text(encoding="utf-8")
    assert main(argv(paths, research1, "--i-confirm-model-freeze")) == 0
    after = paths["splits"].read_text(encoding="utf-8")
    document = yaml.safe_load(after)
    split = document["held_out_map_test"]
    assert split["maps"] == ["test_00", "test_01", "test_02"]
    assert len(split["routes"]) == 18
    assert split["status"] == "assigned_after_model_freeze"
    assert split["model_freeze_sha256"] == hashlib.sha256(paths["freeze"].read_bytes()).hexdigest()
    original = yaml.safe_load(before)
    for key in original:
        if key != "held_out_map_test":
            assert original[key] == document[key]
    # Byte-for-byte preservation outside the replaced block.
    head, _, _ = before.partition("held_out_map_test:")
    assert after.startswith(head)
    tail = before[before.index("natural_failure_audit:"):]
    assert after.endswith(tail)
    record = yaml.safe_load(paths["record"].read_text(encoding="utf-8"))
    assert record["protected_test_used"] is True
    assert record["protected_outcomes_consulted"] is False
    assert len(record["maps"]) == 3
    assert set(record["maps"][0]["file_sha256"]) == {"map.yaml", "map.pgm", "routes.yaml"}
    records = read_records(paths["log"])
    verify(records)
    assert records[-1]["kind"] == "protocol_change"
    assert records[-1]["metadata"]["route_count"] == 18
    with pytest.raises(SystemExit, match="already assigned"):
        main(argv(paths, research1, "--i-confirm-model-freeze"))


def test_fails_closed_when_a_map_has_fewer_routes_than_required(tmp_path):
    research1 = fake_research1(tmp_path / "r1", routes_per_map=5)
    paths = fake_repo(tmp_path)
    before = paths["splits"].read_bytes()
    with pytest.raises(SystemExit, match="never invented"):
        main(argv(paths, research1, "--i-confirm-model-freeze"))
    assert paths["splits"].read_bytes() == before
    assert not paths["record"].exists()
    assert not paths["log"].exists()


def test_fails_closed_on_wrong_map_count(tmp_path):
    research1 = fake_research1(tmp_path / "r1", maps=("test_00", "test_01"))
    paths = fake_repo(tmp_path)
    with pytest.raises(SystemExit, match="expected exactly 3 protected maps"):
        main(argv(paths, research1, "--i-confirm-model-freeze"))
    assert not paths["record"].exists()


def test_block_replacement_only_touches_the_held_out_block():
    text = "a: 1\nheld_out_map_test:\n  maps: []\n  routes: []\nb:\n  c: 2\n"
    replaced = replace_held_out_block(text, "held_out_map_test:\n  maps: [m]\n")
    assert replaced == "a: 1\nheld_out_map_test:\n  maps: [m]\nb:\n  c: 2\n"
