#!/usr/bin/env bash
# Illustrative re-run of one episode with the Gazebo GUI attached and screen-captured.
# Engineering smoke: writes only under data/raw_engineering_smoke; enters no ledger.
set -u
ROOT=/home/eao/failure-prediction; cd "$ROOT" || exit 1
OUT=${OUT:-$ROOT/reports/storyboard/gazebo_rerun_val02_r0}; mkdir -p "$OUT"
export DISPLAY=:1
source scripts/env_research2.sh
FFMPEG=$HOME/.local/bin/ffmpeg
say(){ echo "$(date -u +%FT%TZ) $*"; }
say "clock logger"
python3 scripts/demo/clock_wall_logger.py "$OUT/clock_wall.csv" > "$OUT/clock_logger.log" 2>&1 &
CLK=$!
say "episode runner (engineering smoke, R2 recommendation-only: monitor active, no execution)"
python3 scripts/run_research2_episode.py --campaign-id recovery_smoke_demo_val02 --episode-key demo-val_02-val_02_r0-oscillation-r0 \
  --map val_02 --route val_02_r0 --system s0 --family planner_oscillation --severity medium --seed 920007 \
  --clean-prefix-seconds 5.0 --planned-onset-seconds 8.0 --maximum-duration-seconds 20.0 --maximum-wait-seconds 5.0 \
  --placement-mode path_fraction --route-fraction 0.55 --recording-profile compact_v2 \
  --output-root data/raw_engineering_smoke --recovery-policy R2 > "$OUT/episode.log" 2>&1 &
RUN=$!
say "waiting for the simulation world"
for i in $(seq 1 120); do gz topic -l 2>/dev/null | grep -q '/world/default/clock' && break; sleep 1; done
say "launching the GUI"
gz sim -g --gui-config configs/demo/gazebo_gui_demo.config > "$OUT/gui.log" 2>&1 &
GUI=$!
for i in $(seq 1 60); do xwininfo -root -tree 2>/dev/null | grep -q 'Gazebo' && break; sleep 1; done
sleep 4
WID=$(xwininfo -root -tree | grep -i 'gazebo' | head -n 1 | awk '{print $1}')
eval $(xwininfo -id "$WID" | awk '/Absolute upper-left X/{print "WX="$4} /Absolute upper-left Y/{print "WY="$4} /Width/{print "WW="$2} /Height/{print "WH="$2}')
WW=$(( WW / 2 * 2 )); WH=$(( WH / 2 * 2 ))
say "window $WID at ${WX},${WY} ${WW}x${WH}"
# follow the robot once it exists
for i in $(seq 1 60); do gz model --list 2>/dev/null | grep -q turtlebot3_waffle && break; sleep 1; done
follow(){ gz service -s /gui/follow --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean --timeout 3000 --req 'data: "turtlebot3_waffle"' >> "$OUT/gui.log" 2>&1; gz service -s /gui/follow/offset --reqtype gz.msgs.Vector3d --reptype gz.msgs.Boolean --timeout 3000 --req 'x: -2.6, y: 0.0, z: 1.5' >> "$OUT/gui.log" 2>&1; }
follow
# the GUI scene may not hold the robot yet; repeat the follow request as the scene fills
( for d in 4 8 14 22 32; do sleep $d; follow; done ) &
CAP_START=$(date +%s.%N); echo "$CAP_START" > "$OUT/capture_wall_start.txt"
say "capturing"
"$FFMPEG" -y -loglevel error -f x11grab -framerate 30 -video_size "${WW}x${WH}" -i ":1+${WX},${WY}" -c:v libx264 -preset veryfast -pix_fmt yuv420p "$OUT/capture.mp4" > "$OUT/ffmpeg.log" 2>&1 &
FF=$!
wait $RUN; RC=$?
say "episode runner exited rc=$RC"
kill -INT $FF 2>/dev/null; wait $FF 2>/dev/null
kill $GUI 2>/dev/null; kill -INT $CLK 2>/dev/null
say "done: $(ls -la "$OUT/capture.mp4" | awk '{print $5}') bytes"
