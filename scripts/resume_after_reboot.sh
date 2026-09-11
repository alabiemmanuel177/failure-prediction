#!/usr/bin/env bash
# Resume the Research 2 post-freeze plan after a host reboot.
#
# Installed as the user service research2-resume.service (ops/). It waits until the
# AMD GPU initialises and PyTorch can run a kernel, verifies no campaign runner is
# already active, then starts the post-freeze confirmatory plan detached at four
# workers (within the six admitted by PA-2026-09-04-02; reduced after the AMDGPU
# wedge of 4 September). It never starts anything twice: the plan itself is idempotent
# and refuses while a runner lock is held.
set -u
ROOT=/home/eao/failure-prediction
LOG=$ROOT/logs/resume_after_reboot.log
WORKERS=${RESEARCH2_RESUME_WORKERS:-4}
# Severity phase A runs six-wide (PA-2026-09-04-02) since the iGPU wedge cause was removed on 6 September.
SEVERITY_WORKERS=${RESEARCH2_SEVERITY_WORKERS:-6}
cd "$ROOT" || exit 1
echo "$(date -u +%FT%TZ) resume service started (boot $(uptime -s))" >> "$LOG"

# 1. Wait for a healthy GPU (up to 30 minutes).
for attempt in $(seq 1 60); do
  if [ -e /dev/kfd ] && timeout 60 "$ROOT/.venv/bin/python" - <<'EOF' >> "$LOG" 2>&1
import torch
assert torch.cuda.is_available()
x = torch.randn(1024, 1024, device="cuda")
assert float((x @ x).sum()) != 0.0
print("gpu healthy:", torch.cuda.get_device_name(0))
EOF
  then
    break
  fi
  echo "$(date -u +%FT%TZ) gpu not ready (attempt $attempt)" >> "$LOG"
  sleep 30
done

# 2. Never overlap a running campaign.
if pgrep -f "run_balanced_pilot_continuous.py|run_campaign_parallel.py|run_post_freeze_pla[n].py" >/dev/null; then
  echo "$(date -u +%FT%TZ) a runner or plan is already active; not starting another" >> "$LOG"
  exit 0
fi
if [ ! -f "$ROOT/configs/model_freeze.yaml" ]; then
  echo "$(date -u +%FT%TZ) no model freeze; nothing to resume" >> "$LOG"
  exit 0
fi

# 3. Start the plan detached (severity stress is scheduled after the recovery bring-up).
# The plan runs as its own transient user unit: a oneshot service's cgroup is killed
# when the service exits, so a detached child would not survive this script.
systemd-run --user --unit=research2-post-freeze-plan --property=WorkingDirectory="$ROOT" --setenv=RESEARCH2_AUTO_REBOOT=1 --property=Restart=on-failure --property=RestartSec=300 --property=RestartPreventExitStatus=3 \
  /usr/bin/bash -lc "exec python3 $ROOT/scripts/run_post_freeze_plan.py --workers $WORKERS --skip-severity >> $ROOT/logs/post_freeze_plan.autoresume.log 2>&1" >> "$LOG" 2>&1
echo "$(date -u +%FT%TZ) started research2-post-freeze-plan.service (workers $WORKERS)" >> "$LOG"

# 4. Recreate the 30-minute read-only campaign monitor timer (transient; lost on reboot).
systemd-run --user --unit=research2-heldout-monitor --on-active=2m --on-unit-active=30m \
  --timer-property=AccuracySec=1m /usr/bin/python3 "$ROOT/scripts/monitor_campaign_status.py" \
  --manifest "$ROOT/data/manifests/held_out_map_v1.yaml" \
  --output "$ROOT/reports/status/held_out_map_monitor.yaml" >> "$LOG" 2>&1 || true

# 4a. Hourly artefact mirror to the HDD (transient timer; lost on reboot).
systemd-run --user --unit=research2-backup --on-active=5m --on-unit-active=1h --timer-property=AccuracySec=5m \
  /usr/bin/bash "$ROOT/scripts/backup_to_hdd.sh" >> "$LOG" 2>&1 || true

# 4b. Sweep model directories left incomplete by the crash (no checkpoint): the trainer
#     refuses to overwrite them, so they are moved aside and refitted from scratch.
for d in "$ROOT"/models/ablations/*/* "$ROOT"/models/unseen_family/*/*/*; do
  if [ -d "$d" ] && [ ! -f "$d/checkpoint.pt" ]; then
    mkdir -p "$ROOT/models/_incomplete_runs"
    mv "$d" "$ROOT/models/_incomplete_runs/$(basename "$(dirname "$d")")__$(basename "$d")__$(date -u +%Y%m%dT%H%M%SZ)"
    echo "$(date -u +%FT%TZ) moved incomplete model dir aside: $d" >> "$LOG"
  fi
done

# 5. Companion units that a wedge/reboot also kills: severity stress (simulators) and
#    the ablation suite (GPU). Each is idempotent and skipped once its output exists.
if [ -f "$ROOT/data/manifests/severity_stress_v1.yaml" ] && [ ! -f "$ROOT/reports/confirmatory/severity_stress_v1.cumulative1512.yaml" ]; then
  systemd-run --user --unit=research2-severity --property=WorkingDirectory="$ROOT" --setenv=RESEARCH2_AUTO_REBOOT=1 \
    --property=Restart=on-failure --property=RestartSec=300 --property=RestartPreventExitStatus=3 \
    /usr/bin/bash -lc "exec python3 $ROOT/scripts/run_post_freeze_plan.py --workers $SEVERITY_WORKERS --only-severity >> $ROOT/logs/severity_plan.log 2>&1" >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) started research2-severity.service" >> "$LOG"
fi
if [ -f "$ROOT/data/manifests/recovery_pilot_v1.yaml" ] && [ ! -f "$ROOT/reports/recovery/paired_recovery.yaml" ]; then
  systemd-run --user --unit=research2-recovery --property=WorkingDirectory="$ROOT" --setenv=RESEARCH2_AUTO_REBOOT=1 \
    --property=Restart=on-failure --property=RestartSec=300 --property=RestartPreventExitStatus=3 \
    /usr/bin/bash -lc "exec python3 $ROOT/scripts/run_recovery_plan.py >> $ROOT/logs/recovery_plan.log 2>&1" >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) started research2-recovery.service" >> "$LOG"
fi
if [ ! -f "$ROOT/reports/ablations/validation_ablations.yaml" ]; then
  systemd-run --user --unit=research2-ablations --property=WorkingDirectory="$ROOT" \
    /usr/bin/bash -lc "exec nice -n 19 python3 $ROOT/scripts/run_ablations.py --primary-predictions reports/predictions/final_v1_seed20260904/p3_causal_tcn.validation.calibrated.csv --config configs/models/p3_causal_tcn.yaml --train-dataset balanced_pilot_v1-development-648 --train-dataset targeted_development_v1-development-1212 --train-dataset development_supplement_v1-development-504 --selection-dataset balanced_validation_v1-validation-324 --seed 20260904 >> $ROOT/logs/model_development/ablations_final.log 2>&1" >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) started research2-ablations.service" >> "$LOG"
fi
exit 0
