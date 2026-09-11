"""State and safe-write operations for the local manual-audit review app."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from src.labels.audit_gate import validate_review_pair
from src.labels.threshold_review import validate_threshold_review


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def atomic_yaml(path: Path, value: dict[str, Any]) -> None:
    """Write one review artifact atomically without following a caller path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            yaml.safe_dump(value, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ReviewStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.annotations = self.root / "data/annotations"
        self.reviewed = self.annotations / "reviewed"
        self.disagreements = self.annotations / "review-disagreements"
        self.taxonomy = load_yaml(self.root / "configs/failure_taxonomy.yaml")
        self.researcher = load_yaml(
            self.root / "configs/event_threshold_review.2026-08-30.researcher.yaml"
        )

    def automatic_paths(self) -> list[Path]:
        return sorted(
            path for path in self.annotations.glob("*.yaml")
            if path.name != "episode.template.yaml"
        )

    def automatic(self, run_id: str) -> dict[str, Any]:
        path = self.annotations / f"{run_id}.yaml"
        allowed = {item.stem for item in self.automatic_paths()}
        if run_id not in allowed or not path.exists():
            raise KeyError("unknown frozen-audit run_id")
        return load_yaml(path)

    def _summaries(self) -> dict[str, dict[str, Any]]:
        result = {}
        for automatic_path in self.automatic_paths():
            path = self.root / "data/raw/summaries" / automatic_path.name
            if not path.is_file():
                continue
            try:
                value = load_yaml(path)
            except (OSError, ValueError, yaml.YAMLError):
                # A campaign may have an unrelated summary open while the frozen
                # review packet is served. Missing/incomplete packet summaries fall
                # back to the immutable automatic annotation below.
                continue
            run_id = value.get("identity", {}).get("run_id")
            if run_id:
                result[str(run_id)] = value
        return result

    def episode_state(self) -> list[dict[str, Any]]:
        summaries = self._summaries()
        result = []
        for path in self.automatic_paths():
            automatic = load_yaml(path)
            run_id = str(automatic["run_id"])
            reviewed_path = self.reviewed / path.name
            reviewed = load_yaml(reviewed_path) if reviewed_path.exists() else None
            review = reviewed.get("review", {}) if reviewed else {}
            summary = summaries.get(run_id, {})
            injections = automatic.get("injections", [])
            injection = injections[0] if injections and isinstance(injections[0], dict) else {}
            summary_labels = summary.get("label_only", {})
            fault_family = summary_labels.get("fault_family") or injection.get("family") or "none"
            fault_severity = summary_labels.get("fault_severity")
            if fault_severity is None:
                fault_severity = injection.get("severity")
            operational_path = (
                self.root / "reports/manual-audit/operational-events" / path.name
            )
            operational = load_yaml(operational_path) if operational_path.exists() else {}
            result.append({
                "run_id": run_id,
                "episode_key": summary.get("identity", {}).get("episode_key", run_id),
                "map_id": summary.get("identity", {}).get("map_id"),
                "route_id": summary.get("identity", {}).get("route_id"),
                "fault_family": fault_family,
                "fault_severity": fault_severity,
                "outcome": summary.get("outcome", {}).get("terminal_state"),
                "automatic": automatic,
                "operational_evidence": operational,
                "review": review,
                "review_stage": (
                    "complete" if review.get("adjudication_status") in {"agreed", "adjudicated"}
                    else "first_complete" if review.get("first_reviewer")
                    else "pending"
                ),
                "timeline_url": f"/timelines/{run_id}.svg",
            })
        return result

    def state(self) -> dict[str, Any]:
        episodes = self.episode_state()
        complete = sum(item["review_stage"] == "complete" for item in episodes)
        first = sum(item["review_stage"] in {"first_complete", "complete"} for item in episodes)
        supervisor_path = self.root / "configs/event_threshold_review.supervisor.yaml"
        supervisor = load_yaml(supervisor_path) if supervisor_path.exists() else None
        supervisor_findings = []
        if supervisor:
            supervisor_findings = validate_threshold_review(
                supervisor,
                self.researcher["decisions"],
                researcher_name=str(self.researcher["reviewer_name"]),
            )
        return {
            "episodes": episodes,
            "progress": {"first_review": first, "fully_reviewed": complete, "total": len(episodes)},
            "researcher_name": self.researcher["reviewer_name"],
            "thresholds": self.researcher["decisions"],
            "supervisor_review": supervisor,
            "supervisor_review_valid": bool(supervisor) and not supervisor_findings,
            "supervisor_findings": supervisor_findings,
        }

    def record_review(
        self,
        run_id: str,
        *,
        reviewer_name: str,
        decision: str,
        notes: str,
        evidence_checked: bool,
    ) -> dict[str, Any]:
        reviewer_name = reviewer_name.strip()
        notes = notes.strip()
        if len(reviewer_name) < 2:
            raise ValueError("Enter the reviewer's full name.")
        if decision not in {"agree", "disagree"}:
            raise ValueError("Decision must be agree or disagree.")
        if not evidence_checked:
            raise ValueError("Confirm that the timeline and causal evidence were inspected.")
        if decision == "disagree" and len(notes) < 8:
            raise ValueError("Describe the disagreement before flagging the episode.")

        automatic = self.automatic(run_id)
        output = self.reviewed / f"{run_id}.yaml"
        reviewed = load_yaml(output) if output.exists() else copy.deepcopy(automatic)
        review = reviewed.setdefault("review", {})
        first = str(review.get("first_reviewer") or "").strip()
        second = str(review.get("second_reviewer") or "").strip()
        if review.get("adjudication_status") in {"agreed", "adjudicated"}:
            raise ValueError("This episode already has two completed reviews.")

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if decision == "disagree":
            disagreement = {
                "schema_version": 1,
                "run_id": run_id,
                "reviewer_name": reviewer_name,
                "reviewed_utc": now,
                "decision": "disagree",
                "notes": notes,
                "automatic_causal_fields_preserved": True,
            }
            stamp = now.replace(":", "").replace("-", "")
            atomic_yaml(self.disagreements / f"{run_id}.{stamp}.yaml", disagreement)
            review.setdefault("disagreements", []).append(disagreement)
            review["adjudication_status"] = "pending"
            atomic_yaml(output, reviewed)
            return {"status": "flagged", "message": "Disagreement recorded; adjudication is required."}

        if not first:
            review.update({
                "first_reviewer": reviewer_name,
                "first_reviewed_utc": now,
                "first_review_notes": notes or None,
                "second_reviewer": None,
                "adjudication_status": "pending",
                "disagreements": review.get("disagreements", []),
            })
            atomic_yaml(output, reviewed)
            return {"status": "first_complete", "message": "First review recorded."}
        if reviewer_name.casefold() == first.casefold():
            raise ValueError("The second reviewer must be a different person.")
        if second:
            raise ValueError("A second reviewer has already been recorded.")
        review.update({
            "second_reviewer": reviewer_name,
            "second_reviewed_utc": now,
            "second_review_notes": notes or None,
            "adjudication_status": "agreed",
        })
        errors = validate_review_pair(automatic, reviewed, self.taxonomy)
        if errors:
            raise ValueError("; ".join(errors))
        atomic_yaml(output, reviewed)
        return {"status": "complete", "message": "Independent agreement recorded."}

    def record_supervisor_review(
        self,
        *,
        reviewer_name: str,
        rationale: str,
        role_confirmed: bool,
        protected_confirmed: bool,
    ) -> dict[str, Any]:
        reviewer_name = reviewer_name.strip()
        rationale = rationale.strip()
        if not role_confirmed:
            raise ValueError("The signer must confirm they are the research supervisor.")
        if not protected_confirmed:
            raise ValueError("Confirm that protected outcomes were not consulted.")
        if len(reviewer_name) < 2:
            raise ValueError("Enter the supervisor's full name.")
        if len(rationale) < 12:
            raise ValueError("Enter a short methodological rationale.")
        if reviewer_name.casefold() == str(self.researcher["reviewer_name"]).casefold():
            raise ValueError("The supervisor must be distinct from the researcher.")

        review = copy.deepcopy(load_yaml(self.root / "configs/event_threshold_review.template.yaml"))
        review.update({
            "reviewer_name": reviewer_name,
            "reviewer_role": "Supervisor",
            "reviewed_utc": datetime.now(timezone.utc).date().isoformat(),
            "protected_outcomes_consulted": False,
            "overall_decision": "approved",
        })
        for event, values in review["decisions"].items():
            values.update(copy.deepcopy(self.researcher["decisions"][event]))
            values["decision"] = "accept"
            values["rationale"] = rationale
        findings = validate_threshold_review(
            review,
            self.researcher["decisions"],
            researcher_name=str(self.researcher["reviewer_name"]),
        )
        if findings:
            raise ValueError("; ".join(findings))
        path = self.root / "configs/event_threshold_review.supervisor.yaml"
        if path.exists():
            raise ValueError("A supervisor review already exists; refusing to overwrite it.")
        atomic_yaml(path, review)
        return {"status": "complete", "message": "Supervisor threshold approval recorded."}
