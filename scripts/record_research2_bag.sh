#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 RUN_ID OUTPUT_DIRECTORY PLATFORM_COMMIT REPOSITORY_HEAD" >&2
  exit 2
fi

R2_RECORD_RUN_ID="$1"
R2_RECORD_OUTPUT="$2"
R2_PLATFORM_COMMIT="$3"
R2_REPOSITORY_HEAD="$4"
if [ -e "$R2_RECORD_OUTPUT" ]; then
  echo "refusing to overwrite existing bag path: $R2_RECORD_OUTPUT" >&2
  exit 1
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_research2.sh"

exec ros2 bag record \
  --storage mcap \
  --storage-preset-profile zstd_fast \
  --use-sim-time \
  --include-unpublished-topics \
  --qos-profile-overrides-path "$RESEARCH2_ROOT/configs/recording_qos.yaml" \
  --output "$R2_RECORD_OUTPUT" \
  --custom-data \
    "run_id=$R2_RECORD_RUN_ID" \
    "protocol_version=1.0" \
    "research1_platform_commit=$R2_PLATFORM_COMMIT" \
    "research1_repository_head=$R2_REPOSITORY_HEAD" \
  --topics \
    /clock \
    /research2/events \
    /diagnostics \
    /research2/raw/camera/image \
    /research2/raw/scan \
    /research2/raw/odom \
    /research2/raw/semantic/risk_grid \
    /research2/raw/semantic/risk_grid_odom \
    /camera/image \
    /camera/depth_image \
    /camera/camera_info \
    /scan \
    /imu \
    /odom \
    /tf \
    /tf_static \
    /amcl_pose \
    /initialpose \
    /particle_cloud \
    /plan \
    /local_plan \
    /cmd_vel \
    /behavior_tree_log \
    /global_costmap/costmap \
    /local_costmap/costmap \
    /semantic/classes \
    /semantic/confidence \
    /semantic/uncertainty \
    /semantic/inference_latency_ms \
    /semantic/risk_grid \
    /semantic/risk_grid_odom \
    /collision_event \
    /ground_truth_pose \
    /camera/segmentation
