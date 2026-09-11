#!/usr/bin/env bash
# Hourly mirror of the immutable research artefacts to the 1 TB HDD (/mnt/hdd).
# Never deletes on the destination (no --delete): a quarantine or move on the NVMe
# leaves the HDD copy in place. Low I/O priority so campaigns are unaffected.
set -u
SRC=/home/eao/failure-prediction
DST=/mnt/hdd/failure-prediction-backup
LOG=$SRC/logs/backup_to_hdd.log
mountpoint -q /mnt/hdd || { echo "$(date -u +%FT%TZ) /mnt/hdd not mounted; skipping" >> "$LOG"; exit 0; }
mkdir -p "$DST"
echo "$(date -u +%FT%TZ) backup start" >> "$LOG"
nice -n 19 ionice -c 3 rsync -a --no-inc-recursive --exclude 'data/raw/bags/*/*.mcap.active' \
  "$SRC/data/raw" "$SRC/data/raw_unrecorded_attempts" "$SRC/data/derived" "$SRC/data/manifests" \
  "$SRC/logs" "$SRC/reports" "$SRC/configs" "$SRC/models" "$SRC/docs" "$DST/" >> "$LOG" 2>&1
echo "$(date -u +%FT%TZ) backup done rc=$? ($(du -sh "$DST" 2>/dev/null | cut -f1))" >> "$LOG"
