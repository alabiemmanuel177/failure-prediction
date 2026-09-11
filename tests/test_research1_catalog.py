import csv
from pathlib import Path

import yaml

from src.research1_catalog import build_structural_catalog, validate_structural_catalog


def write_identity(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "run_id", "map_id", "route_id", "system_id", "seed",
                "terminal_state", "success", "shift_family", "severity",
            ],
        )
        writer.writeheader()
        writer.writerow(values)


def write_metadata(path: Path, topics: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "rosbag2_bagfile_information": {
            "message_count": len(topics),
            "duration": {"nanoseconds": 10},
            "topics_with_message_count": [{
                "topic_metadata": {"name": topic}, "message_count": 1,
            } for topic in topics],
        },
    }), encoding="utf-8")


def test_catalog_exports_only_split_safe_structure_and_never_outcomes(tmp_path):
    required = {"/cmd_vel", "/odom"}
    for run_id, map_id in (("dev-run", "dev_00"), ("test-run", "test_00")):
        write_identity(tmp_path / f"results/raw/{run_id}.csv", {
            "run_id": run_id, "map_id": map_id, "route_id": f"{map_id}_r0",
            "system_id": "S0", "seed": "1", "terminal_state": "collision",
            "success": "false", "shift_family": "fog", "severity": "high",
        })
        write_metadata(tmp_path / f"results/bags/{run_id}/metadata.yaml", sorted(required))
    records, counts = build_structural_catalog(tmp_path, required_topics=required)
    assert [item["run_id"] for item in records] == ["dev-run"]
    assert records[0]["split"] == "development"
    assert records[0]["protected_outcomes_consulted"] is False
    assert not {"terminal_state", "success", "shift_family", "severity"} & set(records[0])
    assert records[0]["required_topic_message_counts"] == {"/cmd_vel": 1, "/odom": 1}
    assert "topic_message_counts" not in records[0]
    assert counts["development_candidates"] == 1
    assert counts["validation_candidates"] == 0
    assert validate_structural_catalog(
        records, research1_root=tmp_path, required_topics=required,
    ) == []

    metadata = tmp_path / "results/bags/dev-run/metadata.yaml"
    metadata.write_text(metadata.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    findings = validate_structural_catalog(
        records, research1_root=tmp_path, required_topics=required,
    )
    assert "row 0 source metadata checksum changed" in findings
