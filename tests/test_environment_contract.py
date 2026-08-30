from pathlib import Path
import subprocess

from scripts.validate_episode_artifacts import raw_deployed_tolerance


ROOT = Path(__file__).resolve().parents[1]


def test_research2_environment_exposes_research1_measurement_library():
    result = subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/env_research2.sh && python3 -c 'import rcn; from episode_logger.monitor import EpisodeMonitor'",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_recorder_uses_explicit_sensor_qos_overrides():
    script = (ROOT / "scripts/record_research2_bag.sh").read_text(encoding="utf-8")
    assert "--qos-profile-overrides-path" in script
    qos = (ROOT / "configs/recording_qos.yaml").read_text(encoding="utf-8")
    for topic in ("/scan:", "/odom:", "/camera/image:", "/semantic/confidence:"):
        assert topic in qos
    assert qos.count("reliability: best_effort") >= 4


def test_only_declared_camera_dropout_expands_raw_deployed_tolerance():
    ordinary = {"label_only": {"fault_family": "none", "parameters": {}}}
    camera = {
        "label_only": {
            "fault_family": "camera_occlusion",
            "parameters": {"dropout_probability": 0.20},
        }
    }
    assert raw_deployed_tolerance(
        raw_topic="/research2/raw/camera/image", raw_count=200, summary=ordinary
    ) == 10
    assert raw_deployed_tolerance(
        raw_topic="/research2/raw/camera/image", raw_count=200, summary=camera
    ) == 60
    assert raw_deployed_tolerance(
        raw_topic="/research2/raw/scan", raw_count=200, summary=camera
    ) == 10
