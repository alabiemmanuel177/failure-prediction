"""Recovery-policy plumbing shared by the episode runner, campaign runners and reports.

Pure Python (no ROS): policy identifiers, the command-line and launch arguments that
carry a policy into one episode, and the label-only ``system`` block that the episode
summary records from the online monitor and recovery-manager sidecars.

Policy identifiers
------------------
``R0``            default Nav2 recovery tree, no monitor, no predictor (unchanged path)
``R1``            predictor + fixed conservative stop and resume (deferred, PA-2026-09-03-04)
``R2``            predictor + rule-matched recovery by diagnosed signal group
``R3``            predictor + cost-sensitive learned selector
``RP_<action>``   recovery pilot: the frozen predictor's alarm forces one guard-checked
                  action (``recovery_pilot_v1``); a guard-rejected forced action is
                  recorded as ``guard_rejected`` and executed as ``controlled_stop``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .guards import ACTIONS


DEFAULT_POLICY_ID = "R0"
PAIRED_POLICY_IDS = ("R0", "R1", "R2", "R3")
FORCEABLE_ACTIONS = tuple(action for action in ACTIONS if action != "request_assistance")
PILOT_POLICY_PREFIX = "RP_"
PILOT_POLICY_IDS = tuple(f"{PILOT_POLICY_PREFIX}{action}" for action in FORCEABLE_ACTIONS)
POLICY_IDS = (*PAIRED_POLICY_IDS, *PILOT_POLICY_IDS)

MONITOR_SIDECAR_DIR = "logs/failure-monitor"
MANAGER_SIDECAR_DIR = "logs/recovery"
RECOVERY_LAUNCH_ARGUMENTS = (
    "recovery_policy", "recovery_smoke_unfrozen", "recovery_model_dir", "recovery_calibrator",
    "recovery_selector_model", "recovery_live_execution", "recovery_relocalisation_available",
    "recovery_live_evidence", "goal_x", "goal_y", "goal_yaw",
)
# Engineering live smokes (docs/recovery-live-bringup.md section 4) may point the manager at
# a provisional evidence file, but only when nothing can land in data/raw: the campaign id
# must carry this prefix and the output root must be the engineering smoke root.
ENGINEERING_SMOKE_CAMPAIGN_PREFIX = "recovery_smoke_"
ENGINEERING_SMOKE_OUTPUT_ROOT = "data/raw_engineering_smoke"
SIGNED_LIVE_EVIDENCE = "configs/recovery_live_evidence.yaml"


def validate_policy_id(policy_id: str) -> str:
    if policy_id not in POLICY_IDS:
        raise ValueError(
            f"unknown recovery policy {policy_id!r}; expected one of {', '.join(POLICY_IDS)}"
        )
    return policy_id


def forced_action(policy_id: str) -> str | None:
    """The action a pilot policy forces, or None for R0-R3."""
    validate_policy_id(policy_id)
    if policy_id.startswith(PILOT_POLICY_PREFIX):
        return policy_id[len(PILOT_POLICY_PREFIX):]
    return None


def policy_uses_predictor(policy_id: str) -> bool:
    return validate_policy_id(policy_id) != DEFAULT_POLICY_ID


def validate_evidence_override(
    path: str, *, campaign_id: str, output_root: Path, root: Path,
) -> Path:
    """Admit a provisional live-evidence file for an engineering smoke episode only.

    Raises ``ValueError`` unless the campaign id starts with ``recovery_smoke_``, the output
    root is ``data/raw_engineering_smoke`` under ``root``, the file exists and it is not the
    researcher-signed ``configs/recovery_live_evidence.yaml`` itself.
    """
    if not str(campaign_id).startswith(ENGINEERING_SMOKE_CAMPAIGN_PREFIX):
        raise ValueError(
            "--recovery-evidence-override is refused: --campaign-id must start with "
            f"{ENGINEERING_SMOKE_CAMPAIGN_PREFIX!r} (got {campaign_id!r})"
        )
    expected_root = (Path(root) / ENGINEERING_SMOKE_OUTPUT_ROOT).resolve()
    if Path(output_root).resolve() != expected_root:
        raise ValueError(
            "--recovery-evidence-override is refused: --output-root must be "
            f"{ENGINEERING_SMOKE_OUTPUT_ROOT} (got {output_root})"
        )
    evidence = Path(path)
    if not evidence.is_absolute():
        evidence = Path(root) / evidence
    evidence = evidence.resolve()
    if evidence == (Path(root) / SIGNED_LIVE_EVIDENCE).resolve():
        raise ValueError(
            "--recovery-evidence-override must not name the signed evidence file "
            f"{SIGNED_LIVE_EVIDENCE}; drop the flag to use it"
        )
    if not evidence.is_file():
        raise ValueError(f"--recovery-evidence-override file does not exist: {evidence}")
    return evidence


def recovery_cli_arguments(episode: Mapping[str, Any]) -> list[str]:
    """Extra ``run_research2_episode.py`` arguments for one expanded campaign episode.

    Empty unless the episode carries ``recovery_policy_id`` (a manifest that declares
    ``recovery_policies``), so every other campaign's command line is unchanged.
    """
    policy_id = episode.get("recovery_policy_id")
    if policy_id is None:
        return []
    arguments = ["--recovery-policy", validate_policy_id(str(policy_id))]
    for key, flag in (
        ("recovery_selector_model", "--recovery-selector-model"),
        ("recovery_model_dir", "--recovery-model-dir"),
        ("recovery_calibrator", "--recovery-calibrator"),
    ):
        if episode.get(key):
            arguments.extend([flag, str(episode[key])])
    if episode.get("recovery_live_execution") is True:
        arguments.append("--recovery-live-execution")
    if episode.get("recovery_smoke_unfrozen") is True:
        arguments.append("--recovery-smoke-unfrozen")
    return arguments


def recovery_launch_arguments(
    policy_id: str, *, goal: Mapping[str, float], smoke_unfrozen: bool = False,
    model_dir: str = "", calibrator: str = "", selector_model: str = "",
    live_execution: bool = False, relocalisation_available: bool = False,
    evidence_override: str = "",
) -> list[str]:
    """``key:=value`` launch arguments appended only when a policy was requested."""
    validate_policy_id(policy_id)
    values = {
        "recovery_policy": policy_id,
        "recovery_smoke_unfrozen": "true" if smoke_unfrozen else "false",
        "recovery_model_dir": model_dir or "",
        "recovery_calibrator": calibrator or "",
        "recovery_selector_model": selector_model or "",
        "recovery_live_execution": "true" if live_execution else "false",
        "recovery_relocalisation_available": "true" if relocalisation_available else "false",
        "recovery_live_evidence": str(evidence_override or ""),
        "goal_x": repr(float(goal["x"])),
        "goal_y": repr(float(goal["y"])),
        "goal_yaw": repr(float(goal.get("yaw", 0.0))),
    }
    assert tuple(values) == RECOVERY_LAUNCH_ARGUMENTS
    # ``ros2 launch`` rejects ``key:=`` (an empty value) as malformed; an omitted key
    # takes the launch file's declared default, which is the same empty string.
    return [f"{key}:={value}" for key, value in values.items() if value != ""]


# action_msgs/msg/GoalStatus codes for NavigateToPose goals.
GOAL_STATUS_ACCEPTED, GOAL_STATUS_EXECUTING, GOAL_STATUS_CANCELING = 1, 2, 3
GOAL_STATUS_SUCCEEDED, GOAL_STATUS_CANCELED, GOAL_STATUS_ABORTED = 4, 5, 6
RESUMED_MISSION_GRACE_SECONDS = 30.0


def resumed_mission_terminal(
    statuses: Sequence[tuple[bytes, float, int]], own_goal_id: bytes, *, now: float,
    last_activity: float, grace_seconds: float = RESUMED_MISSION_GRACE_SECONDS,
) -> tuple[str | None, float]:
    """Follow the mission after a live recovery cancelled the runner's own goal.

    The recovery manager resumes with a fresh ``navigate_to_pose`` goal that the runner
    does not own, so the runner watches the action status array instead. ``statuses``
    holds ``(goal_id, stamp_seconds, status)`` for every goal Nav2 still reports; the
    newest goal other than the runner's decides: succeeded -> ``success``, aborted ->
    ``planner_failure``, active -> keep waiting, cancelled (another intervention) ->
    wait for a further resumption. With no resumption within ``grace_seconds`` of the
    last activity (``request_assistance`` never resumes) the mission is a ``timeout``.
    Returns ``(terminal_or_None, last_activity)``.
    """
    others = [item for item in statuses if item[0] != own_goal_id]
    if others:
        _goal, _stamp, status = max(others, key=lambda item: item[1])
        if status in (GOAL_STATUS_ACCEPTED, GOAL_STATUS_EXECUTING, GOAL_STATUS_CANCELING):
            return None, now
        if status == GOAL_STATUS_SUCCEEDED:
            return "success", now
        if status == GOAL_STATUS_ABORTED:
            return "planner_failure", now
    if now - last_activity >= grace_seconds:
        return "timeout", last_activity
    return None, last_activity


def sidecar_paths(root: Path, run_id: str) -> dict[str, Path]:
    return {
        "monitor": root / MONITOR_SIDECAR_DIR / f"{run_id}.json",
        "manager": root / MANAGER_SIDECAR_DIR / f"{run_id}.json",
    }


def promote_partial_sidecar(final: Path) -> Path | None:
    """Rename ``<run_id>.partial.json`` to its final name when the node was killed first."""
    partial = final.with_name(final.name.replace(".json", ".partial.json"))
    if partial.exists() and not final.exists():
        partial.replace(final)
    return final if final.exists() else None


def load_sidecar(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def summarise_recovery_sidecars(
    monitor: Mapping[str, Any] | None, manager: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compact label-only recovery outcome fields for the episode summary."""
    alarms = list((monitor or {}).get("alarms") or [])
    decisions = list((manager or {}).get("decisions") or [])
    actions = []
    guard_rejections = []
    execution_results = []
    guard_violation = False
    for decision in decisions:
        execution = decision.get("execution") or {}
        action = str(decision.get("recommended_action", "none"))
        guard_results = decision.get("guard_results") or {}
        rejected = sorted(name for name, result in guard_results.items()
                          if not (result or {}).get("eligible"))
        if rejected or decision.get("guard_rejected"):
            guard_rejections.append({
                "warning_id": decision.get("warning_id"),
                "forced_action": decision.get("forced_action"),
                "forced_action_rejected": bool(decision.get("guard_rejected")),
                "rejected_actions": rejected,
            })
        performed = bool(execution.get("execution_performed"))
        status = str(execution.get("status", "not_executed"))
        if performed and action in guard_results and not guard_results[action].get("eligible"):
            guard_violation = True
        actions.append({
            "warning_id": decision.get("warning_id"),
            "decision_time": decision.get("decision_time"),
            "action": action,
            "forced_action": decision.get("forced_action"),
            "guard_rejected": bool(decision.get("guard_rejected")),
            "execution_status": status,
            "execution_performed": performed,
            "seconds": execution.get("seconds"),
        })
        execution_results.append({
            "warning_id": decision.get("warning_id"), "action": action, "status": status,
            "failed": status == "failed",
        })
    return {
        "warnings_issued": len(alarms),
        "first_warning_time": (
            float(alarms[0]["decision_time"]) if alarms else None
        ),
        "first_warning_risk": (float(alarms[0]["risk_score"]) if alarms else None),
        "decisions_scored": int((monitor or {}).get("decision_count") or 0),
        "engineering_smoke": bool((monitor or {}).get("engineering_smoke") or False),
        "actions": actions,
        "guard_rejections": guard_rejections,
        "execution_results": execution_results,
        "intervention_count": sum(1 for item in actions if item["execution_performed"]),
        "guard_violation": guard_violation,
        "live_execution_enabled": bool(((manager or {}).get("gate") or {}).get("enabled")),
        "monitor_sidecar_present": monitor is not None,
        "manager_sidecar_present": manager is not None,
    }


def recovery_system_block(
    policy_id: str, *, root: Path, run_id: str, live_execution_requested: bool,
    smoke_unfrozen: bool, evidence_override: str | None = None,
) -> dict[str, Any]:
    """The ``system`` summary block for an episode that requested a recovery policy."""
    validate_policy_id(policy_id)
    paths = sidecar_paths(root, run_id)
    monitor = manager = None
    if policy_uses_predictor(policy_id):
        monitor = load_sidecar(promote_partial_sidecar(paths["monitor"]))
        manager = load_sidecar(promote_partial_sidecar(paths["manager"]))
    return {
        "recovery_policy_id": policy_id,
        "online_failure_monitor": policy_uses_predictor(policy_id),
        "recovery_manager": policy_uses_predictor(policy_id),
        "recovery_forced_action": forced_action(policy_id),
        "recovery_live_execution_requested": bool(live_execution_requested),
        "recovery_smoke_unfrozen": bool(smoke_unfrozen),
        # Engineering smokes only: the provisional evidence file the manager was pointed at.
        "recovery_live_evidence_override": str(evidence_override) if evidence_override else None,
        "monitor_sidecar": str(paths["monitor"]) if policy_uses_predictor(policy_id) else None,
        "manager_sidecar": str(paths["manager"]) if policy_uses_predictor(policy_id) else None,
        "recovery": (
            summarise_recovery_sidecars(monitor, manager)
            if policy_uses_predictor(policy_id) else None
        ),
    }


__all__ = [
    "DEFAULT_POLICY_ID", "ENGINEERING_SMOKE_CAMPAIGN_PREFIX", "ENGINEERING_SMOKE_OUTPUT_ROOT",
    "FORCEABLE_ACTIONS", "PAIRED_POLICY_IDS", "PILOT_POLICY_IDS",
    "PILOT_POLICY_PREFIX", "POLICY_IDS", "RECOVERY_LAUNCH_ARGUMENTS", "forced_action",
    "load_sidecar", "policy_uses_predictor", "promote_partial_sidecar",
    "recovery_cli_arguments", "recovery_launch_arguments", "recovery_system_block",
    "resumed_mission_terminal", "sidecar_paths", "summarise_recovery_sidecars", "validate_evidence_override",
    "validate_policy_id",
]
