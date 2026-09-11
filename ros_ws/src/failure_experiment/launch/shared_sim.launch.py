"""Research 1 simulator with faultable sensor streams routed through Research 2."""

import os
import pathlib
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


TB3 = "/opt/ros/jazzy/share/nav2_minimal_tb3_sim"


def _resolve_xacro(path: str, **args) -> str:
    import subprocess

    command = ["xacro", path] + [f"{key}:={value}" for key, value in args.items()]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    descriptor, destination = tempfile.mkstemp(suffix=".sdf", prefix=pathlib.Path(path).stem + "_")
    with os.fdopen(descriptor, "w") as stream:
        stream.write(output)
    return destination


def launch_setup(context, *_args, **_kwargs):
    simulation_share = get_package_share_directory("simulation_worlds")
    robot_share = get_package_share_directory("robot_description")
    overlay_share = get_package_share_directory("failure_experiment")
    world_name = LaunchConfiguration("world").perform(context)
    world_path = LaunchConfiguration("world_path").perform(context)
    if world_path:
        if not os.path.isabs(world_path) or not os.path.isfile(world_path):
            raise FileNotFoundError(f"world_path must be an existing absolute file: {world_path!r}")
        world_sdf = world_path
    else:
        plain = os.path.join(simulation_share, "worlds", f"{world_name}.sdf")
        xacro = os.path.join(simulation_share, "worlds", f"{world_name}.sdf.xacro")
        if os.path.exists(plain):
            world_sdf = plain
        elif os.path.exists(xacro):
            world_sdf = _resolve_xacro(xacro, headless="true")
        else:
            raise FileNotFoundError(f"Research 1 has no world {world_name!r}")
    robot_sdf = _resolve_xacro(
        os.path.join(robot_share, "urdf", "rcn_waffle.sdf.xacro"), namespace=""
    )
    return [
        ExecuteProcess(
            cmd=["gz", "sim", "-r", "-s", "--headless-rendering", "-v", "1", world_sdf],
            output="screen",
        ),
        Node(
            package="ros_gz_sim", executable="create", output="screen",
            arguments=[
                "-name", "turtlebot3_waffle", "-file", robot_sdf,
                "-x", LaunchConfiguration("x_pose"),
                "-y", LaunchConfiguration("y_pose"),
                "-z", "0.01", "-Y", LaunchConfiguration("yaw"),
            ],
        ),
        Node(
            package="ros_gz_bridge", executable="parameter_bridge", output="screen",
            parameters=[{
                "config_file": os.path.join(overlay_share, "config", "raw_bridge.yaml"),
                "expand_gz_topic_names": True,
                "use_sim_time": True,
            }],
        ),
        Node(
            package="robot_state_publisher", executable="robot_state_publisher", output="screen",
            parameters=[{
                "use_sim_time": True,
                "robot_description": pathlib.Path(
                    os.path.join(TB3, "urdf", "turtlebot3_waffle.urdf")
                ).read_text(),
            }],
        ),
    ]


def common_parameters():
    return {
        "use_sim_time": True,
        "research2_root": LaunchConfiguration("research2_root"),
        "run_id": LaunchConfiguration("run_id"),
        "family": LaunchConfiguration("family"),
        "severity": LaunchConfiguration("severity"),
        "seed": LaunchConfiguration("seed"),
        "source_commit": LaunchConfiguration("source_commit"),
        "clean_prefix_seconds": LaunchConfiguration("clean_prefix_seconds"),
        "planned_onset_seconds": LaunchConfiguration("planned_onset_seconds"),
        "maximum_duration_seconds": LaunchConfiguration("maximum_duration_seconds"),
        "maximum_wait_seconds": LaunchConfiguration("maximum_wait_seconds"),
        "minimum_command_speed_mps": LaunchConfiguration("minimum_command_speed_mps"),
    }


def recovery_arguments():
    """Closed-loop recovery (work package Q). Defaults reproduce the R0 path exactly."""
    return [
        DeclareLaunchArgument("recovery_policy", default_value="R0"),
        DeclareLaunchArgument("recovery_smoke_unfrozen", default_value="false"),
        DeclareLaunchArgument("recovery_model_dir", default_value=""),
        DeclareLaunchArgument("recovery_calibrator", default_value=""),
        DeclareLaunchArgument("recovery_selector_model", default_value=""),
        DeclareLaunchArgument("recovery_live_execution", default_value="false"),
        DeclareLaunchArgument("recovery_relocalisation_available", default_value="false"),
        # Empty => the manager reads the signed configs/recovery_live_evidence.yaml.
        DeclareLaunchArgument("recovery_live_evidence", default_value=""),
        DeclareLaunchArgument("goal_x", default_value="0.0"),
        DeclareLaunchArgument("goal_y", default_value="0.0"),
        DeclareLaunchArgument("goal_yaw", default_value="0.0"),
    ]


def recovery_nodes():
    """Online failure monitor and recovery manager; launched only when policy != R0."""
    from launch_ros.parameter_descriptions import ParameterValue

    predictor_policy = IfCondition(PythonExpression([
        "'", LaunchConfiguration("recovery_policy"), "' != 'R0'"
    ]))

    def text(name):
        return ParameterValue(LaunchConfiguration(name), value_type=str)

    # The monitor scores with torch, which only the project venv provides (the venv
    # includes the system site-packages, so rclpy and the same numpy are available).
    # The console script is a Python file, so the interpreter is a plain prefix.
    venv_python = os.path.join(os.environ.get("RESEARCH2_ROOT", ""), ".venv", "bin", "python")
    monitor_prefix = venv_python if os.path.isfile(venv_python) else ""

    return [
        Node(
            package="failure_monitor", executable="failure_monitor", output="screen",
            condition=predictor_policy, prefix=monitor_prefix,
            parameters=[{
                "use_sim_time": True,
                "research2_root": text("research2_root"),
                "run_id": text("run_id"),
                "recovery_policy": text("recovery_policy"),
                "smoke_unfrozen": LaunchConfiguration("recovery_smoke_unfrozen"),
                "model_dir": text("recovery_model_dir"),
                "calibrator": text("recovery_calibrator"),
                "goal_x": LaunchConfiguration("goal_x"),
                "goal_y": LaunchConfiguration("goal_y"),
                "relocalisation_available": LaunchConfiguration("recovery_relocalisation_available"),
            }],
        ),
        Node(
            package="recovery_manager", executable="recovery_manager", output="screen",
            condition=predictor_policy,
            parameters=[{
                "use_sim_time": True,
                "run_id": text("run_id"),
                "policy_id": text("recovery_policy"),
                "selector_model": text("recovery_selector_model"),
                "live_execution": LaunchConfiguration("recovery_live_execution"),
                "live_evidence": text("recovery_live_evidence"),
                "goal_x": LaunchConfiguration("goal_x"),
                "goal_y": LaunchConfiguration("goal_y"),
                "goal_yaw": LaunchConfiguration("goal_yaw"),
            }],
        ),
    ]


def generate_launch_description():
    environment_family = PythonExpression([
        "'", LaunchConfiguration("family"), "' in ['dynamic_blockage', 'planner_oscillation']"
    ])
    arguments = [
        DeclareLaunchArgument("world", default_value="dev_00"),
        DeclareLaunchArgument("world_path", default_value=""),
        DeclareLaunchArgument("x_pose", default_value="-2.0"),
        DeclareLaunchArgument("y_pose", default_value="-0.5"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("research2_root", default_value=os.environ.get("RESEARCH2_ROOT", "")),
        DeclareLaunchArgument("run_id"),
        DeclareLaunchArgument("family", default_value="none"),
        DeclareLaunchArgument("severity", default_value="none"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("source_commit", default_value="unknown"),
        DeclareLaunchArgument("clean_prefix_seconds", default_value="10.0"),
        DeclareLaunchArgument("planned_onset_seconds", default_value="15.0"),
        DeclareLaunchArgument("maximum_duration_seconds", default_value="20.0"),
        DeclareLaunchArgument("maximum_wait_seconds", default_value="10.0"),
        DeclareLaunchArgument("minimum_command_speed_mps", default_value="0.05"),
        DeclareLaunchArgument("injection_x", default_value="0.0"),
        DeclareLaunchArgument("injection_y", default_value="0.0"),
        DeclareLaunchArgument("injection_yaw", default_value="0.0"),
        DeclareLaunchArgument("placement_mode", default_value="explicit"),
        DeclareLaunchArgument("route_fraction", default_value="0.55"),
        *recovery_arguments(),
    ]
    return LaunchDescription([
        SetEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH",
            f"{TB3}/models:{os.path.dirname(TB3)}:" + os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
        ),
        *arguments,
        OpaqueFunction(function=launch_setup),
        Node(
            package="failure_experiment", executable="signal_proxy", output="screen",
            parameters=[common_parameters()],
        ),
        Node(
            package="failure_experiment", executable="topic_health", output="screen",
            parameters=[{
                "use_sim_time": True,
                "run_id": LaunchConfiguration("run_id"),
            }],
        ),
        Node(
            package="failure_experiment", executable="semantic_summary", output="screen",
            parameters=[{"use_sim_time": True}],
        ),
        Node(
            package="failure_experiment", executable="environment_fault", output="screen",
            condition=IfCondition(environment_family),
            parameters=[{
                **common_parameters(),
                "injection_x": LaunchConfiguration("injection_x"),
                "injection_y": LaunchConfiguration("injection_y"),
                "injection_yaw": LaunchConfiguration("injection_yaw"),
                "placement_mode": LaunchConfiguration("placement_mode"),
                "route_fraction": LaunchConfiguration("route_fraction"),
            }],
        ),
        *recovery_nodes(),
    ])
