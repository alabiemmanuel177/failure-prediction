from src.features import derive_window_features


def row(time, x, angular=0.0, valid=1.0):
    return {
        "decision_time": time,
        "command_linear": 0.2, "measured_linear": 0.2,
        "command_angular": angular, "odom_x": x, "odom_y": 0.0,
        "amcl_x": x, "amcl_y": 0.0, "valid_return_fraction": valid,
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


def test_stopped_while_commanded_is_explicit():
    source = row(0.0, 0.0)
    source["measured_linear"] = 0.0
    output = derive_window_features([source], goal_x=1.0, goal_y=0.0)
    assert output[0]["stopped_while_commanded"] == 1.0
