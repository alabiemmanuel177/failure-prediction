"""Past-only temporal features derived from synchronized decision rows."""

from __future__ import annotations

from collections import deque
import math
from typing import Mapping, Sequence


def _observed(row: Mapping[str, object], feature: str) -> bool:
    return feature in row and int(row.get(f"{feature}__missing", 0)) == 0


def _slope(points: Sequence[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    mean_t = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_t) ** 2 for point in points)
    if denominator == 0:
        return None
    return sum((t - mean_t) * (value - mean_y) for t, value in points) / denominator


def derive_window_features(
    rows: Sequence[Mapping[str, object]], *, goal_x: float, goal_y: float,
    history_seconds: float = 5.0,
) -> list[dict[str, object]]:
    if history_seconds <= 0:
        raise ValueError("history_seconds must be positive")
    history: deque[dict[str, object]] = deque()
    output = []
    previous_time = float("-inf")
    previous_amcl: tuple[float, float] | None = None
    for source in rows:
        row = dict(source)
        now = float(row["decision_time"])
        if now <= previous_time:
            raise ValueError("decision rows must be strictly ordered")
        history.append(row)
        while history and float(history[0]["decision_time"]) < now - history_seconds:
            history.popleft()

        def emit(name: str, value: float | None) -> None:
            row[name] = 0.0 if value is None else float(value)
            row[f"{name}__missing"] = 1 if value is None else 0
            row[f"{name}__age_seconds"] = 0.0

        tracking = None
        if _observed(row, "command_linear") and _observed(row, "measured_linear"):
            tracking = abs(float(row["command_linear"]) - float(row["measured_linear"]))
        emit("tracking_error", tracking)
        stopped = None
        if _observed(row, "command_linear") and _observed(row, "measured_linear"):
            stopped = float(
                abs(float(row["command_linear"])) >= 0.05
                and abs(float(row["measured_linear"])) < 0.01
            )
        emit("stopped_while_commanded", stopped)

        velocity_points = [
            (float(past["decision_time"]), float(past["measured_linear"]))
            for past in history if _observed(past, "measured_linear")
        ]
        accelerations = []
        for (first_t, first_v), (second_t, second_v) in zip(
            velocity_points, velocity_points[1:]
        ):
            if second_t > first_t:
                accelerations.append((second_t, (second_v - first_v) / (second_t - first_t)))
        emit("jerk", _slope(accelerations))

        # Goal context is map-frame: the frozen route goal is compared with the
        # robot's own map-frame pose estimate (/amcl_pose). Odometry starts at the
        # odom-frame origin and must never be compared with a map-frame goal.
        goal_distance = None
        if _observed(row, "amcl_x") and _observed(row, "amcl_y"):
            goal_distance = math.dist(
                (float(row["amcl_x"]), float(row["amcl_y"])), (goal_x, goal_y)
            )
        emit("goal_distance", goal_distance)
        progress_points = []
        for past in history:
            if _observed(past, "amcl_x") and _observed(past, "amcl_y"):
                distance = math.dist(
                    (float(past["amcl_x"]), float(past["amcl_y"])), (goal_x, goal_y)
                )
                progress_points.append((float(past["decision_time"]), distance))
        distance_slope = _slope(progress_points)
        emit("remaining_distance_slope", distance_slope)
        emit("progress_slope", None if distance_slope is None else -distance_slope)

        angular = [float(past["command_angular"]) for past in history
                   if _observed(past, "command_angular") and abs(float(past["command_angular"])) >= 0.05]
        sign_changes = sum(a * b < 0 for a, b in zip(angular, angular[1:]))
        emit("command_sign_changes", float(sign_changes) if angular else None)

        pose_jump = None
        if _observed(row, "amcl_x") and _observed(row, "amcl_y"):
            current_amcl = (float(row["amcl_x"]), float(row["amcl_y"]))
            if previous_amcl is not None:
                pose_jump = math.dist(previous_amcl, current_amcl)
            previous_amcl = current_amcl
        emit("pose_jump", pose_jump)

        scan_points = [(float(past["decision_time"]), float(past["valid_return_fraction"]))
                       for past in history if _observed(past, "valid_return_fraction")]
        emit("valid_return_trend", _slope(scan_points))
        near_points = [
            (float(past["decision_time"]), float(past["minimum_front_range"]))
            for past in history if _observed(past, "minimum_front_range")
        ]
        emit("near_obstacle_trend", _slope(near_points))
        imbalance = None
        if _observed(row, "minimum_left_range") and _observed(row, "minimum_right_range"):
            imbalance = abs(
                float(row["minimum_left_range"]) - float(row["minimum_right_range"])
            )
        emit("left_right_imbalance", imbalance)

        path_ages = [
            (float(past["decision_time"]), float(past["global_path_length__age_seconds"]))
            for past in history if _observed(past, "global_path_length")
        ]
        replan_rate = None
        if len(path_ages) >= 2 and path_ages[-1][0] > path_ages[0][0]:
            resets = sum(
                current_age < previous_age
                for (_, previous_age), (_, current_age) in zip(path_ages, path_ages[1:])
            )
            replan_rate = resets / (path_ages[-1][0] - path_ages[0][0])
        emit("replan_rate", replan_rate)
        output.append(row)
        previous_time = now
    return output
