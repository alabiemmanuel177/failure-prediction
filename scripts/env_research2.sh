#!/usr/bin/env bash
# Source Research 1 and this overlay without relying on the caller's current directory.

if [ -n "${ZSH_VERSION:-}" ]; then
  R2_ENV_SCRIPT="${(%):-%N}"
  R2_SETUP_EXTENSION="zsh"
else
  R2_ENV_SCRIPT="${BASH_SOURCE[0]}"
  R2_SETUP_EXTENSION="bash"
fi
R2_WORKSPACE_ROOT="$(cd "$(dirname "$R2_ENV_SCRIPT")/.." && pwd)"
R1_WORKSPACE_ROOT="/home/eao/risk-calibrated-nav"

set +u
unset COLCON_CURRENT_PREFIX
source "/opt/ros/jazzy/setup.$R2_SETUP_EXTENSION"
source "$R1_WORKSPACE_ROOT/install/setup.$R2_SETUP_EXTENSION"
if [ -f "$R2_WORKSPACE_ROOT/ros_ws/install/setup.$R2_SETUP_EXTENSION" ]; then
  source "$R2_WORKSPACE_ROOT/ros_ws/install/setup.$R2_SETUP_EXTENSION"
fi
set -u

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$R1_WORKSPACE_ROOT/configs/dds/cyclonedds.xml"
# Research 1, Research 3, and Research 4 may run on this workstation. A shared
# domain allowed a foreign /clock to stamp a Research 2 goal hundreds of seconds
# ahead of its TF stream. Use an R2-specific domain unless explicitly overridden
# through the equally specific variable below.
export ROS_DOMAIN_ID="${RESEARCH2_ROS_DOMAIN_ID:-52}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export GZ_IP=127.0.0.1
export GZ_PARTITION="${GZ_PARTITION:-research2}"
export GZ_SIM_RESOURCE_PATH="/opt/ros/jazzy/share/nav2_minimal_tb3_sim/models:/opt/ros/jazzy/share:${GZ_SIM_RESOURCE_PATH:-}"
export GZ_SIM_HEADLESS=1
export PYTHONHASHSEED=0
export RESEARCH1_ROOT="$R1_WORKSPACE_ROOT"
export RESEARCH2_ROOT="$R2_WORKSPACE_ROOT"
# Research 1 ROS packages import the repository-level `rcn` measurement library.
# Colcon installs the ROS packages but does not install that library, so expose the
# locked repository root explicitly for direct episode runs and campaign workers.
export PYTHONPATH="$R2_WORKSPACE_ROOT:$R1_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$R2_WORKSPACE_ROOT/logs/ros}"
export GZ_HOMEDIR="${GZ_HOMEDIR:-$R2_WORKSPACE_ROOT/logs/gazebo}"
mkdir -p "$ROS_LOG_DIR" "$GZ_HOMEDIR"
