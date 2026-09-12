# Illustrative Gazebo re-execution (engineering run)

Not the confirmatory episode. This is a fresh engineering run of validation episode
val_02-val_02_r0-oscillation-r0 (same map, route, seed 920007 and injected planner-oscillation
fault) with the Gazebo GUI attached and screen-captured, and the frozen predictor running in
recommendation mode (policy R2 without live execution). It was written under
data/raw_engineering_smoke and entered no ledger, inventory or dataset.

Run 40f66aa7: warnings at 62.4 s (outside the 10 s window, a false alert by the preregistered
definition) and 72.9 s (useful, 6.3 s before the planner failure at 79.1 s). The first take
(a75e4a6c) had a 1.8 s lead and a wide static camera; it is retained in the sibling directory.

Files: capture.mp4 (raw screen capture), sim_view.mp4 (cropped, synchronised to the risk trace via
clock_wall.csv), storyboard/ (risk-trace frames from the run's own failure-monitor sidecar),
demo_90s_gazebo.mp4 (title card, stacked view, closing card). Built by
scripts/demo/capture_gazebo_demo.sh and scripts/demo/assemble_gazebo_demo.py.
