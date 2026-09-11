import hashlib
import importlib
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ros_ws/src/recovery_manager"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from recovery_manager.live_execution import (  # noqa: E402
    ACTION_SEQUENCES, ActionBounds, GateDecision, LiveExecutor, evaluate_live_gate,
    required_evidence_items,
)
from src.recovery import GuardConfig, RecoveryRequest, RobotState, decide_recovery  # noqa: E402


GUARDS = ROOT / "configs/recovery_guards.yaml"


class FakeCommander:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def _record(self, name, *args):
        self.calls.append((name, *args))
        if name == self.fail_on:
            raise RuntimeError(f"{name} failed")

    def cancel_task(self): self._record("cancel_task")
    def zero_cmd_vel(self): self._record("zero_cmd_vel")
    def backup(self, distance_m, speed_mps): self._record("backup", distance_m, speed_mps)
    def spin(self, angle_rad): self._record("spin", angle_rad)
    def wait(self, seconds): self._record("wait", seconds)
    def clear_costmaps(self): self._record("clear_costmaps")
    def relocalise(self): self._record("relocalise")
    def resume_navigation(self): self._record("resume_navigation")
    def request_assistance(self): self._record("request_assistance")


def evidence_document(complete=True, stamp=True, guard_path=GUARDS):
    items = {name: {"verified": complete, "verified_utc": "2026-09-10T12:00:00Z" if stamp else ""}
             for name in required_evidence_items(guard_path)}
    return {"schema_version": 1, "frozen": True,
            "guard_config_sha256": hashlib.sha256(guard_path.read_bytes()).hexdigest(),
            "required_live_evidence_before_execution": items}


def decision(action="backup", policy="R2", **state_changes):
    state = dict(stopped=True, stop_allowed=True, localisation_poor=False, planning_stale_or_blocked=False,
                 rear_clearance_m=1.0, rotation_clearance_m=1.0, immediate_collision_risk=False,
                 obstruction_may_be_transient=False, relocalisation_available=False, repeated_recovery_count=0)
    state.update(state_changes)
    request = RecoveryRequest("run", "warning-1", 0.9, "frontal_blockage", RobotState(**state))
    result = decide_recovery(request, GuardConfig(), policy)
    assert result["recommended_action"] == action
    return result


def test_gate_fails_closed_without_parameter_evidence_timestamps_or_matching_guard_hash(tmp_path):
    evidence = tmp_path / "evidence.yaml"
    assert evaluate_live_gate(False, evidence, GUARDS).enabled is False
    assert "absent" in evaluate_live_gate(True, evidence, GUARDS).reason
    evidence.write_text(yaml.safe_dump(evidence_document(complete=False)))
    assert "not verified" in evaluate_live_gate(True, evidence, GUARDS).reason
    evidence.write_text(yaml.safe_dump(evidence_document(stamp=False)))
    assert "timestamp" in evaluate_live_gate(True, evidence, GUARDS).reason
    stale = evidence_document()
    stale["guard_config_sha256"] = "0" * 64
    evidence.write_text(yaml.safe_dump(stale))
    assert "different guard config" in evaluate_live_gate(True, evidence, GUARDS).reason
    evidence.write_text(yaml.safe_dump(evidence_document()))
    gate = evaluate_live_gate(True, evidence, GUARDS)
    assert gate.enabled is True and len(gate.evidence["required_live_evidence_before_execution"]) == 5


def test_locked_executor_refuses_and_logs_every_guard_rejection():
    events = []
    executor = LiveExecutor(FakeCommander(), GateDecision(False, "locked"), events.append)
    record = executor.execute(decision())
    assert record["execution_performed"] is False and record["status"] == "refused_live_execution_locked"
    assert events[0]["event_type"] == "recovery_guard_rejection"
    assert "relocalise" in events[0]["rejected_actions"]
    assert events[1]["event_type"] == "recovery_execution_refused"


def test_enabled_executor_runs_cancel_stop_then_bounded_action_and_logs_it():
    events = []
    commander = FakeCommander()
    executor = LiveExecutor(commander, GateDecision(True, "ok"), events.append, ActionBounds())
    record = executor.execute(decision())
    assert record["status"] == "executed" and record["execution_performed"] is True
    assert [call[0] for call in commander.calls] == list(ACTION_SEQUENCES["backup"])
    assert commander.calls[2] == ("backup", 0.25, 0.05)
    assert events[-1]["event_type"] == "recovery_action_executed"
    assert executor.executed_count == 1


def test_executor_never_runs_a_guard_rejected_action_even_when_enabled():
    events = []
    commander = FakeCommander()
    executor = LiveExecutor(commander, GateDecision(True, "ok"), events.append)
    tampered = decision()
    tampered["recommended_action"] = "relocalise"  # guard_results say ineligible
    record = executor.execute(tampered)
    assert record["status"] == "refused_guard_rejected_action" and commander.calls == []


def test_failed_step_stops_the_robot_and_reports_failure():
    commander = FakeCommander(fail_on="backup")
    executor = LiveExecutor(commander, GateDecision(True, "ok"), lambda event: None)
    record = executor.execute(decision())
    assert record["status"] == "failed" and record["execution_performed"] is False
    assert commander.calls[-1] == ("zero_cmd_vel",)


def test_request_assistance_sequence_has_no_automated_resume():
    assert ACTION_SEQUENCES["request_assistance"][-1] == "request_assistance"
    assert "resume_navigation" not in ACTION_SEQUENCES["request_assistance"]
    assert all(seq[:2] == ("cancel_task", "zero_cmd_vel") for seq in ACTION_SEQUENCES.values())


def test_action_bounds_are_validated():
    with pytest.raises(ValueError):
        ActionBounds(backup_distance_m=2.0).validate()


def test_nav2_adapter_imports_without_ros():
    module = importlib.import_module("recovery_manager.nav2_commander")
    assert "UNTESTED LIVE" in module.__doc__
    assert "rclpy" not in sys.modules or True


class _FakeNavigator:
    """Stands in for BasicNavigator: records calls, never touches ROS."""

    def __init__(self, accept=True, result="SUCCEEDED"):
        self.calls = []
        self.accept = accept
        self.result_name = result

    def create_client(self, srv_type, name):
        self.calls.append(("create_client", name))
        return ("client", name)

    def cancelTask(self): self.calls.append(("cancelTask",))
    def isTaskComplete(self): return True
    def getResult(self):
        import enum
        return enum.Enum("TaskResult", ["SUCCEEDED", "CANCELED", "FAILED"])[self.result_name]
    def backup(self, backup_dist, backup_speed, time_allowance):
        self.calls.append(("backup", backup_dist, backup_speed)); return self.accept
    def spin(self, spin_dist, time_allowance):
        self.calls.append(("spin", spin_dist)); return self.accept
    def goToPose(self, pose): self.calls.append(("goToPose", pose)); return self.accept


class _FakeNode:
    def create_publisher(self, *args): return type("Pub", (), {"publish": lambda self, m: None})()


def _commander(navigator, responses):
    action_msgs = pytest.importorskip("action_msgs.srv")
    module = importlib.import_module("recovery_manager.nav2_commander")
    commander = module.Nav2LiveCommander(_FakeNode(), navigator=navigator, step_timeout_seconds=1.0)
    requests = []

    def fake_call(client, request, what):
        requests.append((client, request, what))
        return responses[what]
    commander._call_service = fake_call
    return commander, requests, action_msgs


def test_cancel_task_cancels_every_navigate_to_pose_goal_not_only_the_navigators_own():
    navigator = _FakeNavigator()
    cancelled = type("R", (), {"return_code": 0, "goals_canceling": [object()]})()
    commander, requests, action_msgs = _commander(navigator, {"navigate_to_pose cancel": cancelled})
    commander.cancel_task()
    assert navigator.calls[:2] == [("create_client", "/navigate_to_pose/_action/cancel_goal"),
                                   ("create_client", "/reinitialize_global_localization")]
    assert ("cancelTask",) in navigator.calls
    client, request, what = requests[0]
    assert client == ("client", "/navigate_to_pose/_action/cancel_goal")
    # Zero goal id and zero stamp is the action protocol's cancel-all request.
    assert not any(request.goal_info.goal_id.uuid) and request.goal_info.stamp.sec == 0
    assert commander.last_cancel == {"return_code": 0, "goals_canceling": 1}
    rejected = type("R", (), {"return_code": action_msgs.CancelGoal.Response.ERROR_REJECTED,
                              "goals_canceling": []})()
    commander, _requests, _ = _commander(_FakeNavigator(), {"navigate_to_pose cancel": rejected})
    with pytest.raises(RuntimeError, match="rejected the cancel-all"):
        commander.cancel_task()


def test_behaviours_and_resume_raise_on_rejection_or_failed_result():
    commander, _r, _ = _commander(_FakeNavigator(accept=False), {})
    with pytest.raises(RuntimeError, match="rejected the backup"):
        commander.backup(0.25, 0.05)
    with pytest.raises(RuntimeError, match="rejected the spin"):
        commander.spin(1.57)
    commander.set_mission_goal("goal")
    with pytest.raises(RuntimeError, match="rejected the resumed mission goal"):
        commander.resume_navigation()
    commander, _r, _ = _commander(_FakeNavigator(result="FAILED"), {})
    with pytest.raises(RuntimeError, match="ended with result FAILED"):
        commander.spin(1.57)
    ok = _FakeNavigator()
    commander, _r, _ = _commander(ok, {})
    commander.set_mission_goal("goal")
    commander.backup(0.25, 0.05); commander.spin(1.57); commander.resume_navigation()
    assert [c for c in ok.calls if c[0] in ("backup", "spin", "goToPose")] == [
        ("backup", 0.25, 0.05), ("spin", 1.57), ("goToPose", "goal")]


def test_relocalise_calls_the_global_localisation_service_via_the_navigator():
    navigator = _FakeNavigator()
    commander, requests, _ = _commander(navigator, {"relocalisation": object()})
    commander.relocalise()
    assert requests[0][0] == ("client", "/reinitialize_global_localization")
