from pathlib import Path

import pytest
import yaml

from confirmatory_fixtures import FAMILIES, ROOT, held_out_episodes, table_rows
from scripts.analyze_hypotheses import assemble, main, render_markdown
from scripts.evaluate_confirmatory import build_report


def held_out_report(*, fixture: bool) -> dict:
    episodes = held_out_episodes()
    alarm = yaml.safe_load((ROOT / "configs/alarm_policy.yaml").read_text(encoding="utf-8"))
    alarm["threshold"] = 0.5
    tables = {
        "p3": table_rows(episodes, model_id="p3_causal_tcn", protected=True, detect_probability=0.95,
                         false_alert_probability=0.05, seed=1),
        "p1": table_rows(episodes, model_id="p1_threshold_rules", protected=True, detect_probability=0.4,
                         false_alert_probability=0.1, seed=2),
    }
    return build_report(tables=tables, threshold=0.5, alarm_policy=alarm,
                        bootstrap={"replicates": 30, "seed": 1}, engineering_fixture=fixture,
                        model_freeze_sha256="f" * 64)


def unseen_report(*, complete: bool) -> dict:
    return {
        "complete": complete, "engineering_fixture": not complete, "families": list(FAMILIES),
        "h5": {"count": 5, "families_where_p3_exceeds_p1": list(FAMILIES[:5]),
               "supported": True if complete else None,
               "heterogeneity": {"per_family_differences": {f: 0.2 for f in FAMILIES}, "range": [0.2, 0.2]}},
    }


def test_assembles_labels_and_statuses():
    report = assemble(held_out_report(fixture=False), unseen_report(complete=True), None, None)
    rows = {row["id"]: row for row in report["hypotheses"]}
    assert rows["H1"]["label"] == "confirmatory"
    assert rows["H1"]["status"] in {"supported", "not_supported"}
    assert rows["H1"]["confidence_interval"] is not None
    assert all(rows[key]["label"] == "supporting" for key in ("H2", "H3", "H4", "H5"))
    assert rows["H5"]["status"] == "supported" and rows["H5"]["estimate"] == 5
    assert rows["H3"]["status"] == "not_evaluable"  # no validation table supplied
    assert rows["H4"]["status"] == "pending"
    assert rows["H6"]["label"] == "confirmatory" and rows["H6"]["status"] == "pending"
    assert report["research_evidence"] is True
    assert report["confirmatory_claims"] == ["H1", "H6"]
    assert all(item["label"] == "exploratory" for item in report["exploratory"])
    assert any("timeout" in item["item"] for item in report["exploratory"])


def test_fixture_inputs_are_flagged_and_pending():
    report = assemble(held_out_report(fixture=True), unseen_report(complete=False), None, None)
    assert report["research_evidence"] is False
    rows = {row["id"]: row for row in report["hypotheses"]}
    assert rows["H1"]["status"] == "pending"
    assert rows["H5"]["status"] == "pending"
    markdown = render_markdown(report)
    assert "Engineering fixture" in markdown
    assert "| H1 | confirmatory |" in markdown
    assert "| H5 | supporting |" in markdown


def test_cli_writes_immutable_yaml_and_markdown(tmp_path):
    held_out = tmp_path / "held_out_map.yaml"
    unseen = tmp_path / "unseen_family.yaml"
    held_out.write_text(yaml.safe_dump(held_out_report(fixture=False)), encoding="utf-8")
    unseen.write_text(yaml.safe_dump(unseen_report(complete=True)), encoding="utf-8")
    natural = tmp_path / "natural_failure_audit.yaml"
    natural.write_text(yaml.safe_dump({
        "totals": {"natural_failures": 3},
        "warning_performance": [{"model_id": "p3_causal_tcn",
                                 "natural_failures_excluding_timeouts": {"event_recall": 0.5, "event_count": 2}}],
    }), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    argv = ["--held-out", str(held_out), "--unseen-family", str(unseen), "--natural-failures", str(natural),
            "--recovery", str(tmp_path / "missing.yaml"), "--output-dir", str(out)]
    assert main(argv) == 0
    report = yaml.safe_load((out / "hypotheses.yaml").read_text(encoding="utf-8"))
    assert str(held_out) in report["inputs_sha256"]
    assert report["model_freeze_sha256"] == "f" * 64
    markdown = (out / "hypotheses.md").read_text(encoding="utf-8")
    assert "| ID | Label | Hypothesis |" in markdown
    assert "natural (no-injection) failures" in markdown
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(argv)
    with pytest.raises(SystemExit, match="no confirmatory reports"):
        main(["--held-out", str(tmp_path / "a.yaml"), "--unseen-family", str(tmp_path / "b.yaml"),
              "--output-dir", str(tmp_path / "other")])
