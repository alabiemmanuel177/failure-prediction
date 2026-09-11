"""Nav2 Simple Commander adapter for the guarded live executor.

UNTESTED LIVE until docs/recovery-live-evidence-draft.md records a passing engineering
smoke. Every ROS/Nav2 import is lazy so that importing this module never requires rclpy.
Construct only inside a running node after ``evaluate_live_gate`` has authorised
execution. Nav2 collision checking is never bypassed: backup and spin are the stock
Nav2 behaviours with bounded arguments, relocalisation uses
``/reinitialize_global_localization`` (or an ``initialpose`` republish when a last
known pose is supplied) and every step waits for task completion or a timeout.

Two live findings shaped this adapter (docs/recovery-live-evidence-draft.md):

* ``BasicNavigator.cancelTask`` cancels only a goal the navigator itself sent. The
  mission goal belongs to the episode runner, so ``cancel_task`` additionally sends a
  cancel-all request (zero goal id and stamp) to ``navigate_to_pose``.
* The manager's request callback blocks for the whole sequence, so a service future on
  the manager node never completes. Service clients therefore live on the navigator
  node, which is spun explicitly while waiting.
"""

from __future__ import annotations

import time
from typing import Any


class Nav2LiveCommander:
    def __init__(self, node: Any, *, step_timeout_seconds: float = 10.0,
                 last_known_pose: Any | None = None, navigator: Any | None = None) -> None:
        from action_msgs.srv import CancelGoal  # lazy
        from geometry_msgs.msg import Twist  # lazy
        from std_srvs.srv import Empty  # lazy

        if navigator is None:
            from nav2_simple_commander.robot_navigator import BasicNavigator  # lazy
            navigator = BasicNavigator()
        self.node = node
        self.navigator = navigator
        self.step_timeout_seconds = step_timeout_seconds
        self.last_known_pose = last_known_pose
        self._twist_type = Twist
        self._cancel_type = CancelGoal
        self._empty_type = Empty
        self.cmd_vel = node.create_publisher(Twist, "/cmd_vel", 10)
        self.cancel_all = navigator.create_client(CancelGoal, "/navigate_to_pose/_action/cancel_goal")
        self.reinitialize = navigator.create_client(Empty, "/reinitialize_global_localization")
        self.initialpose = None
        if last_known_pose is not None:
            from geometry_msgs.msg import PoseWithCovarianceStamped  # lazy
            self.initialpose = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self._last_goal = None
        self.last_cancel: dict[str, int] | None = None

    # ------------------------------------------------------------------ waiting
    def _call_service(self, client: Any, request: Any, what: str) -> Any:
        """Call a service from inside a blocked callback by spinning the navigator node."""
        import rclpy  # lazy
        if not client.wait_for_service(timeout_sec=self.step_timeout_seconds):
            raise RuntimeError(f"{what} service unavailable")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.navigator, future, timeout_sec=self.step_timeout_seconds)
        if not future.done():
            raise TimeoutError(f"{what} service did not respond")
        return future.result()

    def _wait_task(self, what: str) -> None:
        deadline = time.monotonic() + self.step_timeout_seconds
        while not self.navigator.isTaskComplete():
            if time.monotonic() > deadline:
                self.navigator.cancelTask()
                raise TimeoutError(f"Nav2 {what} exceeded the bounded step timeout")
            time.sleep(0.05)
        result = self.navigator.getResult()
        if getattr(result, "name", str(result)) != "SUCCEEDED":
            raise RuntimeError(f"Nav2 {what} ended with result {getattr(result, 'name', result)}")

    # ------------------------------------------------------------------ steps
    def cancel_task(self) -> None:
        # The navigator's own task first (a behaviour in flight), then every
        # navigate_to_pose goal including the episode runner's mission goal.
        self.navigator.cancelTask()
        response = self._call_service(
            self.cancel_all, self._cancel_type.Request(), "navigate_to_pose cancel",
        )
        code = int(response.return_code)
        self.last_cancel = {"return_code": code, "goals_canceling": len(response.goals_canceling)}
        if code == self._cancel_type.Response.ERROR_REJECTED:
            raise RuntimeError("Nav2 rejected the cancel-all request")

    def zero_cmd_vel(self) -> None:
        for _ in range(3):
            self.cmd_vel.publish(self._twist_type())
            time.sleep(0.02)

    def backup(self, distance_m: float, speed_mps: float) -> None:
        if not self.navigator.backup(backup_dist=float(distance_m), backup_speed=float(speed_mps),
                                     time_allowance=int(self.step_timeout_seconds)):
            raise RuntimeError("Nav2 rejected the backup request")
        self._wait_task("backup")

    def spin(self, angle_rad: float) -> None:
        if not self.navigator.spin(spin_dist=float(angle_rad),
                                   time_allowance=int(self.step_timeout_seconds)):
            raise RuntimeError("Nav2 rejected the spin request")
        self._wait_task("spin")

    def wait(self, seconds: float) -> None:
        time.sleep(float(seconds))

    def clear_costmaps(self) -> None:
        self.navigator.clearAllCostmaps()

    def relocalise(self) -> None:
        if self.initialpose is not None and self.last_known_pose is not None:
            self.initialpose.publish(self.last_known_pose)
            return
        self._call_service(self.reinitialize, self._empty_type.Request(), "relocalisation")

    def resume_navigation(self) -> None:
        # The episode runner owns the mission goal; resumption is a fresh goToPose to the
        # goal it recorded on the request, which the node stores before executing.
        if self._last_goal is None:
            raise RuntimeError("no mission goal recorded for resumption")
        if not self.navigator.goToPose(self._last_goal):
            raise RuntimeError("Nav2 rejected the resumed mission goal")

    def request_assistance(self) -> None:
        self.cancel_task()
        self.zero_cmd_vel()

    def set_mission_goal(self, goal: Any) -> None:
        self._last_goal = goal
