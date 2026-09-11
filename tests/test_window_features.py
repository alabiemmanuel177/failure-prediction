from src.features import derive_window_features


def row(time, x, angular=0.0, valid=1.0):
    return {
        "decision_time": time,
        "command_linear": 0.2, "measured_linear": 0.2,
        "command_angular": angular, "odom_x": x, "odom_y": 0.0,
        "amcl_x": x, "amcl_y": 0.0, "valid_return_fraction": valid,
        "minimum_front_range": 4.0 - x,
        "minimum_left_range": 2.0,
        "minimum_right_range": 1.0,
        "global_path_length": 4.0,
        "global_path_length__age_seconds": time % 2,
    }


def test_window_features_use_only_accumulated_past_rows():
    output = derive_window_features([
        row(0.0, 0.0, -0.2, 1.0),
        row(1.0, 1.0, 0.2, 0.8),
        row(2.0, 2.0, -0.2, 0.6),
    ], goal_x=4.0, goal_y=0.0, history_seconds=5.0)
    assert output[0]["progress_slope__missing"] == 1
    assert output[1]["progress_slope"] == 1.0
    assert output[2]["command_sign_changes"] == 2.0
    assert output[2]["valid_return_trend"] == -0.2
    assert output[2]["pose_jump"] == 1.0
    assert output[2]["near_obstacle_trend"] == -1.0
    assert output[2]["left_right_imbalance"] == 1.0
    assert output[2]["replan_rate"] == 0.5


def test_stopped_while_commanded_is_explicit():
    source = row(0.0, 0.0)
    source["measured_linear"] = 0.0
    output = derive_window_features([source], goal_x=1.0, goal_y=0.0)
    assert output[0]["stopped_while_commanded"] == 1.0


def test_jerk_uses_only_past_velocity_history():
    rows = [row(0.0, 0.0), row(1.0, 0.1), row(2.0, 0.3)]
    rows[0]["measured_linear"] = 0.0
    rows[1]["measured_linear"] = 1.0
    rows[2]["measured_linear"] = 3.0
    output = derive_window_features(rows, goal_x=4.0, goal_y=0.0)
    assert output[1]["jerk__missing"] == 1
    assert output[2]["jerk"] == 1.0
