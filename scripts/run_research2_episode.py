#!/usr/bin/env python3
"""Run one Research 2 episode on the pinned Research 1 platform.

This is intentionally separate from Research 1's immutable controller. It reuses its
maps, routes, Nav2 configuration, monitor, semantic stack, and provenance utilities,
while routing faultable streams through the Research 2 overlay and retaining every bag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid

import yaml


ROOT = Path(__file__).resolve().parents[1]
ROS_PACKAGE = ROOT / "ros_ws" / "src" / "failure_experiment"
if str(ROS_PACKAGE) not in sys.path:
    sys.path.insert(0, str(ROS_PACKAGE))

from failure_experiment.events import EVENT_TOPIC, event_message
from failure_experiment.parameters import load_fault_parameters
from failure_experiment.signal_proxy import EVENT_QOS

sys.path.insert(0, str(ROOT))
from src.platform_boundary import BoundaryError, validate_platform


def validate_boundary(lock: dict, map_id: str) -> tuple[Path, str, str]:
    try:
        research1, actual_commit = validate_platform(lock)
    except BoundaryError as error:
        raise SystemExit(str(error)) from error
    split = {"dev": "development", "val": "validation", "test": "test"}.get(
        map_id.split("_", 1)[0]
    )
    if split not in lock["allowed_splits"]:
        raise SystemExit(f"map {map_id} belongs to forbidden or unknown split {split!r}")
    return research1, split, actual_commit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_id", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--system", default="s3")
    parser.add_argument("--family", required=True)
    parser.add_argument("--severity", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--clean-prefix-seconds", type=float, default=10.0)
    parser.add_argument("--planned-onset-seconds", type=float, default=15.0)
    parser.add_argument("--maximum-duration-seconds", type=float, default=20.0)
    parser.add_argument("--maximum-wait-seconds", type=float, default=10.0)
    parser.add_argument("--injection-x", type=float)
    parser.add_argument("--injection-y", type=float)
    parser.add_argument("--injection-yaw", type=float, default=0.0)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--campaign-id", default="manual")
    parser.add_argument("--episode-key", default=None)
    args = parser.parse_args()

    lock = yaml.safe_load((ROOT / "integration" / "research1.lock.yaml").read_text())
    research1, split, research1_head = validate_boundary(lock, args.map_id)
    load_fault_parameters(ROOT, args.family, args.severity)
    if args.family in {"dynamic_blockage", "planner_oscillation"} and (
        args.injection_x is None or args.injection_y is None
    ):
        raise SystemExit("environment faults require explicit --injection-x and --injection-y")

    # Imports happen after the environment and Research 1 boundary are validated.
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from tf2_msgs.msg import TFMessage

    from episode_logger.monitor import EpisodeMonitor
    from experiment_controller import preflight
    from experiment_controller.run_episode import (
        AMCL_TIMEOUT_S,
        NAV2_READY_TIMEOUT_S,
        PREFLIGHT_TIMEOUT_S,
        SETTLE_S,
        EpisodeFinished,
        EpisodeInvalid,
        Stack,
        nav2_bringup_ready,
        load_route,
        load_system_config,
        provenance,
        write_episode_params,
    )
    from rcn.metrics import spl as compute_spl

    run_id = str(uuid.uuid4())
    output_root = args.output_root.resolve()
    summary_dir = output_root / "summaries"
    bag_dir = output_root / "bags" / run_id
    log_dir = output_root / "logs" / run_id
    event_sidecar = output_root / "events" / f"{run_id}.json"
    for path in (summary_dir, bag_dir.parent, log_dir, event_sidecar.parent):
        path.mkdir(parents=True, exist_ok=True)

    map_dir = research1 / "data" / split / args.map_id
    route = load_route(map_dir, args.route)
    goal_yaw = math.atan2(
        float(route["goal"]["y"]) - float(route["start"]["y"]),
        float(route["goal"]["x"]) - float(route["start"]["x"]),
    )
    map_spec = route["_map"]
    system_path = research1 / "configs" / "systems" / f"{args.system.lower()}.yaml"
    system = load_system_config(system_path)
    if args.family in {"camera_occlusion", "semantic_corruption"} and not system.get(
        "perception", {}
    ).get("enabled"):
        raise SystemExit(f"{args.family} is irrelevant to system {system['system_id']}")

    prov = provenance()
    timeout_s = float(map_spec.get("episode_timeout_s") or 180.0)
    goal_tolerance = float(map_spec.get("goal_tolerance_m", 0.25))
    stack = Stack(log_dir)
    terminal = "invalid"
    invalid_reason = "logging_failure"
    planning_failures = 0
    measurements = None
    goal_distance_gt = None
    duration = 0.0
    recorded_events: list[dict] = []

    try:
        rclpy.init()
        launch = [
            "ros2", "launch", "failure_experiment", "shared_sim.launch.py",
            f"world:={args.map_id}",
            f"x_pose:={route['start']['x']}",
            f"y_pose:={route['start']['y']}",
            f"yaw:={route['start'].get('yaw', 0.0)}",
            f"research2_root:={ROOT}",
            f"run_id:={run_id}",
            f"family:={args.family}",
            f"severity:={args.severity}",
            f"seed:={args.seed}",
            f"source_commit:={lock['commit_sha']}",
            f"clean_prefix_seconds:={args.clean_prefix_seconds}",
            f"planned_onset_seconds:={args.planned_onset_seconds}",
            f"maximum_duration_seconds:={args.maximum_duration_seconds}",
            f"maximum_wait_seconds:={args.maximum_wait_seconds}",
        ]
        if args.injection_x is not None:
            launch.extend([
                f"injection_x:={args.injection_x}", f"injection_y:={args.injection_y}",
                f"injection_yaw:={args.injection_yaw}",
            ])
        stack.launch("sim", launch)

        required = dict(preflight.MANDATORY)
        required.update(preflight.EVALUATION_ONLY)
        if system.get("perception", {}).get("enabled"):
            required.update(preflight.PERCEPTION)
        readiness = preflight.wait_until_ready(required, timeout_s=PREFLIGHT_TIMEOUT_S)
        if not readiness.ok:
            invalid_reason = "missing_mandatory_topic_before_goal"
            raise EpisodeInvalid("preflight failed:\n" + readiness.summary())

        params_path = write_episode_params(
            research1 / "configs" / "nav2" / "nav2_common.yaml",
            log_dir / "nav2_params.yaml",
            float(route["start"]["x"]), float(route["start"]["y"]),
            float(route["start"].get("yaw", 0.0)), system,
        )
        stack.launch("nav2", [
            "ros2", "launch", "simulation_worlds", "nav2.launch.py",
            f"map:={map_dir / 'map.yaml'}", f"params_file:={params_path}",
        ])

        monitor = EpisodeMonitor()
        event_publisher = monitor.create_publisher(
            __import__("diagnostic_msgs.msg", fromlist=["DiagnosticArray"]).DiagnosticArray,
            EVENT_TOPIC, EVENT_QOS,
        )
        executor = SingleThreadedExecutor()
        tf_edges: dict[str, tuple[str, float]] = {}

        def observe_tf(message: TFMessage) -> None:
            for transform in message.transforms:
                tf_edges[transform.child_frame_id] = (
                    transform.header.frame_id,
                    float(transform.header.stamp.sec)
                    + float(transform.header.stamp.nanosec) * 1e-9,
                )

        tf_observer = monitor.create_subscription(
            TFMessage, "/tf", observe_tf, qos_profile_sensor_data
        )
        executor.add_node(monitor)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        def emit(event_type: str, reason: str = "", parameters: dict | None = None) -> None:
            stamp = monitor.get_clock().now().to_msg()
            payload = {
                "event_type": event_type,
                "reason": reason,
                "parameters": parameters or {},
                "stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec},
            }
            recorded_events.append(payload)
            event_publisher.publish(event_message(
                stamp=stamp, event_type=event_type, run_id=run_id,
                family=args.family, severity=args.severity, seed=args.seed,
                reason=reason, parameters=parameters,
                source_commit=lock["commit_sha"],
            ))

        nav_client = ActionClient(monitor, NavigateToPose, "navigate_to_pose")
        # The G6 Nav2 parameters intentionally launch on wall time to avoid Jazzy's
        # action-server timer race. Research 1's frozen readiness routine waits for
        # every lifecycle node, then switches all Nav2 nodes to simulation time.
        bringup_failure = nav2_bringup_ready(
            nav_client, monitor, NAV2_READY_TIMEOUT_S
        )
        if bringup_failure is not None:
            invalid_reason = "missing_mandatory_topic_before_goal"
            raise EpisodeInvalid(bringup_failure)

        initial_pose_pub = monitor.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        start_yaw = float(route["start"].get("yaw", 0.0))
        initial = PoseWithCovarianceStamped()
        initial.header.frame_id = "map"
        initial.pose.pose.position.x = float(route["start"]["x"])
        initial.pose.pose.position.y = float(route["start"]["y"])
        initial.pose.pose.orientation.z = math.sin(start_yaw / 2)
        initial.pose.pose.orientation.w = math.cos(start_yaw / 2)
        initial.pose.covariance[0] = initial.pose.covariance[7] = 0.25
        initial.pose.covariance[35] = 0.068
        deadline = time.time() + AMCL_TIMEOUT_S
        while time.time() < deadline and not monitor.amcl_converged():
            initial.header.stamp = monitor.get_clock().now().to_msg()
            initial_pose_pub.publish(initial)
            time.sleep(1.0)
        if not monitor.amcl_converged():
            invalid_reason = "missing_mandatory_topic_before_goal"
            raise EpisodeInvalid("AMCL did not converge before the goal")
        time.sleep(SETTLE_S)

        if system.get("perception", {}).get("enabled"):
            perception_command = [
                "ros2", "run", "semantic_perception", "live_risk_node",
                "--ros-args", "-p", "use_sim_time:=true",
                "-p", f"repo_root:={research1}", "-p", f"system_config:={system_path}",
                "-p", "rgb_topic:=/camera/image",
            ]
            if args.family == "semantic_corruption":
                perception_command.extend([
                    "-r", "/semantic/risk_grid:=/research2/raw/semantic/risk_grid",
                    "-r", "/semantic/risk_grid_odom:=/research2/raw/semantic/risk_grid_odom",
                ])
            stack.launch("perception", perception_command)
            semantic_readiness = preflight.wait_until_ready(
                preflight.SEMANTIC_OUTPUTS, timeout_s=PREFLIGHT_TIMEOUT_S
            )
            if not semantic_readiness.ok:
                invalid_reason = "missing_mandatory_topic_before_goal"
                raise EpisodeInvalid("semantic preflight failed:\n" + semantic_readiness.summary())

        # AMCL agreement alone does not prove that Nav2 can currently traverse the
        # complete map -> odom -> base_link chain.  A Jazzy clock transition can clear
        # one TF buffer after convergence and before dispatch.  Retry the initial pose
        # and require the exact transform consumed by bt_navigator before recording.
        def wait_for_fresh_navigation_tf(timeout_seconds: float = 15.0) -> bool:
            deadline = time.time() + timeout_seconds
            next_pose_retry = 0.0
            stable_since = None
            while time.time() < deadline:
                now = monitor.get_clock().now().nanoseconds * 1e-9
                map_odom = tf_edges.get("odom")
                odom_base = tf_edges.get("base_footprint")
                fresh = bool(
                    map_odom and map_odom[0] == "map"
                    and odom_base and odom_base[0] == "odom"
                    # AMCL publishes map -> odom ahead by transform_tolerance
                    # (0.9 s observed on the pinned platform). It is fresh, not
                    # future leakage; allow the configured prediction interval.
                    and -1.5 <= now - map_odom[1] <= 0.5
                    and -0.05 <= now - odom_base[1] <= 0.5
                    and monitor.amcl_converged()
                )
                if fresh:
                    stable_since = stable_since or time.time()
                    if time.time() - stable_since >= 1.0:
                        return True
                else:
                    stable_since = None
                if time.time() >= next_pose_retry:
                    initial.header.stamp = monitor.get_clock().now().to_msg()
                    initial_pose_pub.publish(initial)
                    next_pose_retry = time.time() + 1.0
                time.sleep(0.1)
            return False

        if not wait_for_fresh_navigation_tf():
            invalid_reason = "missing_mandatory_topic_before_goal"
            raise EpisodeInvalid("fresh map -> base_link TF unavailable before recording")

        # All required publishers now exist. Start recording, then mark the exact
        # interval independently counted by TopicHealth. The marker itself is retained
        # in MCAP so reconciliation can use recorder timestamps rather than estimates.
        stack.launch("bag", [
            "bash", str(ROOT / "scripts" / "record_research2_bag.sh"), run_id,
            str(bag_dir), lock["commit_sha"], research1_head,
        ])
        time.sleep(2.0)
        emit("recording_window_started")
        time.sleep(0.2)

        # Recorder discovery is deliberately bounded but long enough for a TF clock
        # discontinuity to recur. Recheck at the actual dispatch boundary.
        if not wait_for_fresh_navigation_tf():
            emit("recording_window_ended")
            invalid_reason = "missing_mandatory_topic_before_goal"
            raise EpisodeInvalid("fresh map -> base_link TF unavailable before goal dispatch")

        monitor.start()
        start_wall = time.time()
        emit("episode_started")
        if args.family == "none":
            emit("no_injection_control")
        emit("goal_dispatched")
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = monitor.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(route["goal"]["x"])
        goal.pose.pose.position.y = float(route["goal"]["y"])
        goal.pose.pose.orientation.z = math.sin(goal_yaw / 2)
        goal.pose.pose.orientation.w = math.cos(goal_yaw / 2)
        send_future = nav_client.send_goal_async(goal)
        while not send_future.done() and time.time() - start_wall < timeout_s:
            time.sleep(0.1)
        handle = send_future.result() if send_future.done() else None
        if handle is None or not handle.accepted:
            terminal = "planner_failure"
            planning_failures = 1
        else:
            result_future = handle.get_result_async()
            terminal = "timeout"
            while time.time() - start_wall < timeout_s:
                if monitor.snapshot().collision:
                    terminal = "collision"
                    handle.cancel_goal_async()
                    break
                if result_future.done():
                    status = result_future.result().status
                    if status == 4:
                        terminal = "success"
                    elif status == 6:
                        elapsed = time.time() - start_wall
                        if args.family != "none" and elapsed < args.clean_prefix_seconds:
                            terminal = "invalid"
                            invalid_reason = "navigation_aborted_before_fault_eligibility"
                        else:
                            terminal = "planner_failure"
                            planning_failures = 1
                    break
                time.sleep(0.1)
            else:
                handle.cancel_goal_async()

        goal_distance_gt = monitor.distance_to(
            float(route["goal"]["x"]), float(route["goal"]["y"])
        )
        claimed = terminal == "success"
        reached = goal_distance_gt <= goal_tolerance
        success = claimed and reached
        if claimed and not reached:
            terminal = "false_arrival"
        measurements = monitor.snapshot()
        if measurements.collision and success:
            success, terminal = False, "collision"
        emit("terminal_event", terminal, {
            "terminal_state": terminal,
            "goal_distance_gt_m": goal_distance_gt,
            "success": success,
        })
        time.sleep(0.5)
        emit("recording_window_ended")
        time.sleep(0.2)
        duration = time.time() - start_wall
        raise EpisodeFinished()

    except EpisodeFinished:
        pass
    except EpisodeInvalid as error:
        terminal = "invalid"
        print(f"episode invalid: {error}", file=sys.stderr)
    finally:
        try:
            monitor.stop()
            measurements = measurements or monitor.snapshot()
            executor.shutdown()
            monitor.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        stack.shutdown()
        health_root = ROOT / "logs" / "topic-health"
        health_partial = health_root / f"{run_id}.partial.json"
        health_final = health_root / f"{run_id}.json"
        if health_partial.exists() and not health_final.exists():
            health_partial.replace(health_final)

    measurement = measurements
    collided = bool(measurement.collision) if measurement else False
    success = terminal == "success" and not collided
    shortest = float(route.get("shortest_path_m") or 0.0)
    executed = round(float(measurement.path_length_m) if measurement else 0.0, 4)
    primary_event = {
        "collision": "collision",
        "planner_failure": "navigation_abort",
        "timeout": "mission_timeout",
        "false_arrival": "false_arrival",
    }.get(terminal)
    fault_path = ROOT / "configs" / "faults" / f"{args.family}.yaml"
    config_material = (
        fault_path.read_bytes() if fault_path.exists() else b"none"
    ) + (ROOT / "configs" / "failure_events.yaml").read_bytes()
    bag_files = sorted(bag_dir.glob("*.mcap")) if bag_dir.exists() else []
    bag_hash = hashlib.sha256()
    for bag_file in bag_files:
        with bag_file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                bag_hash.update(chunk)
    health_sidecar = ROOT / "logs" / "topic-health" / f"{run_id}.json"
    summary = {
        "schema_version": 1,
        "identity": {
            "run_id": run_id,
            "campaign_id": args.campaign_id,
            "episode_key": args.episode_key,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "protocol_version": "1.0",
            "research1_platform_commit": lock["commit_sha"],
            "research1_repository_head": research1_head,
            "research2_config_hash": hashlib.sha256(config_material).hexdigest(),
        },
        "environment": {
            "map_id": args.map_id,
            "route_id": args.route,
            "start_pose": {
                "x": float(route["start"]["x"]), "y": float(route["start"]["y"]),
                "yaw": float(route["start"].get("yaw", 0.0)),
            },
            "goal_pose": {
                "x": float(route["goal"]["x"]), "y": float(route["goal"]["y"]),
                "yaw": goal_yaw,
            },
            "seed": args.seed,
            "system_id": system["system_id"],
            "split": split,
            "protected_test_used": False,
        },
        "label_only": {
            "fault_family": args.family,
            "severity": args.severity,
            "parameters": load_fault_parameters(ROOT, args.family, args.severity),
            "planned_onset_seconds": args.planned_onset_seconds,
            "clean_prefix_seconds": args.clean_prefix_seconds,
            "primary_event_class": primary_event,
        },
        "outcome": {
            "terminal_state": terminal,
            "success": success,
            "collision": collided,
            "timeout": terminal == "timeout",
            "invalid_reason": invalid_reason if terminal == "invalid" else None,
            "duration_s": round(duration, 3),
            "path_length_m": executed,
            "shortest_path_m": shortest,
            "spl": compute_spl(success, shortest, executed) if terminal != "invalid" else 0.0,
            "minimum_clearance_m": (
                round(measurement.final_clearance, 4)
                if measurement and measurement.final_clearance is not None else -1.0
            ),
            "near_collision_count": measurement.near_collision_count if measurement else 0,
            "collision_count": measurement.collision_count if measurement else 0,
            "planning_failures": planning_failures,
            "localisation_error_mean_m": round(measurement.localisation_error_mean_m, 4) if measurement else 0.0,
            "localisation_error_max_m": round(measurement.localisation_error_max_m, 4) if measurement else 0.0,
            "goal_distance_gt_m": round(goal_distance_gt, 4) if goal_distance_gt is not None else -1.0,
        },
        "provenance": {
            "container_digest": prov["container_digest"],
            "host": prov["host"],
            "gpu": prov["gpu"],
            "ros_distro": prov["ros_distro"],
            "gazebo_version": prov["gazebo_version"],
            "bag_path": str(bag_dir),
            "bag_mcap_count": len(bag_files),
            "bag_checksum_sha256": bag_hash.hexdigest() if bag_files else None,
            "topic_health_sidecar": str(health_sidecar),
            "event_sidecar": str(event_sidecar),
        },
    }
    summary_path = summary_dir / f"{run_id}.yaml"
    with summary_path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(summary, stream, sort_keys=False)
    event_sidecar.write_text(json.dumps(recorded_events, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "run_id": run_id, "summary": str(summary_path), "bag": str(bag_dir),
        "events": str(event_sidecar), "terminal_state": terminal, "success": success,
    }, indent=2))
    return 0 if terminal != "invalid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
