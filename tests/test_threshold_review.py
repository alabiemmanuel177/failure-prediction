from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_unreviewed_threshold_template_fails_closed():
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/validate_threshold_review.py"),
        str(ROOT / "configs/event_threshold_review.template.yaml"),
    ], check=False, capture_output=True, text=True)
    assert result.returncode == 1
    assert "REVIEW INVALID" in result.stdout
