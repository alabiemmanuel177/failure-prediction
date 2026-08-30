from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bt_wall_timestamp_is_not_used_to_arm_signal_fault_schedule():
    source = (
        ROOT / "ros_ws/src/failure_experiment/failure_experiment/signal_proxy.py"
    ).read_text(encoding="utf-8")
    body = source.split("def on_bt", 1)[1].split("def on_command", 1)[0]
    assert "self.get_clock().now()" in body
    assert "schedule.arm(stamp_seconds(message.timestamp))" not in body


def test_bt_wall_timestamp_is_not_used_to_arm_environment_fault_schedule():
    source = (
        ROOT / "ros_ws/src/failure_experiment/failure_experiment/environment_fault.py"
    ).read_text(encoding="utf-8")
    body = source.split("def on_bt", 1)[1].split("def _spawn", 1)[0]
    assert "self.get_clock().now()" in body
    assert "schedule.arm(stamp_seconds(message.timestamp))" not in body
