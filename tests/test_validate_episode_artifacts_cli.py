from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_empty_summary_is_a_controlled_integrity_failure(tmp_path):
    summary = tmp_path / "empty.yaml"
    summary.write_bytes(b"")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_episode_artifacts.py"), str(summary)],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 1
    assert "summary YAML is empty or is not a mapping" in result.stdout
    assert "Traceback" not in result.stderr
