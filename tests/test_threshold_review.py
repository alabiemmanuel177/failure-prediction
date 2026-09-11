from pathlib import Path
import copy
import subprocess
import sys

import yaml

from src.labels.threshold_review import (
    validate_researcher_threshold_review,
    validate_threshold_review,
)

ROOT = Path(__file__).resolve().parents[1]


def test_unreviewed_threshold_template_fails_closed():
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/validate_threshold_review.py"),
        str(ROOT / "configs/event_threshold_review.template.yaml"),
    ], check=False, capture_output=True, text=True)
    assert result.returncode == 1
    assert "REVIEW INVALID" in result.stdout


def test_recorded_researcher_approval_is_valid_for_protocol_1_1():
    researcher = yaml.safe_load(
        (ROOT / "configs/event_threshold_review.2026-08-30.researcher.yaml")
        .read_text(encoding="utf-8")
    )
    assert validate_researcher_threshold_review(researcher) == []


def accepted_review():
    researcher = yaml.safe_load(
        (ROOT / "configs/event_threshold_review.2026-08-30.researcher.yaml")
        .read_text(encoding="utf-8")
    )
    approved = researcher["decisions"]
    review = yaml.safe_load(
        (ROOT / "configs/event_threshold_review.template.yaml").read_text(encoding="utf-8")
    )
    review.update({
        "reviewer_name": "Independent Supervisor",
        "reviewer_role": "Supervisor",
        "reviewed_utc": "2026-09-01",
        "overall_decision": "approved",
    })
    for event, decision in review["decisions"].items():
        decision["decision"] = "accept"
        decision["rationale"] = "Prospectively reviewed and accepted."
        for field, value in approved[event].items():
            if field not in {"rationale", "decision"}:
                decision[field] = value
    return review, approved


def test_exact_distinct_supervisor_acceptance_is_valid():
    review, approved = accepted_review()
    assert validate_threshold_review(
        review, approved, researcher_name="Emmanuel Alabi Olasubomi"
    ) == []


def test_accept_cannot_silently_change_value_or_reuse_researcher_identity():
    review, approved = accepted_review()
    altered = copy.deepcopy(review)
    altered["decisions"]["localisation_loss"]["translation_error_m"] = 0.75
    altered["reviewer_name"] = "Emmanuel Alabi Olasubomi"
    findings = validate_threshold_review(
        altered, approved, researcher_name="Emmanuel Alabi Olasubomi"
    )
    assert "supervisor reviewer must be distinct from the researcher" in findings
    assert any("translation_error_m differs" in finding for finding in findings)
