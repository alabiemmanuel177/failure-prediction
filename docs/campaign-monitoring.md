# Campaign monitoring

`scripts/monitor_campaign_status.py` is an observe-only reconciler for long-running
preprotected campaigns. It reads the frozen campaign manifest, parent and replacement
ledgers, and the campaign's advisory `flock`; it writes one atomic YAML snapshot and
appends the same observation to an `fsync`-backed JSONL history.

The reconciler expands the manifest design and fails closed if the parent ledger
contains a duplicate episode key or any key outside that design. Progress therefore
means resolved preregistered design cells, not merely a convenient ledger line count.

It does not launch ROS, start or stop a process, remove a lock, retry an episode, create
a replacement, or inspect protected data. A quiet process is therefore never treated as
failed. A failed parent row is unresolved until exactly one manifest-declared replacement
has a successful ledger row. A historically successful parent row that is explicitly
declared invalid after an integrity audit is excluded from the successful-artifact count
and handled by the same replacement rule.

Run one check with:

```bash
make monitor-validation
```

After validation has been finalized and the targeted-development runner has started,
the equivalent one-shot check is:

```bash
make monitor-targeted
```

The current 30-minute workstation monitor was created as a transient user timer:

```bash
systemd-run --user \
  --unit=research2-validation-monitor \
  --on-active=30m \
  --on-unit-active=30m \
  --timer-property=AccuracySec=1m \
  /usr/bin/python3 \
  /home/eao/failure-prediction/scripts/monitor_campaign_status.py \
  --manifest /home/eao/failure-prediction/data/manifests/balanced_validation_v1.yaml \
  --output /home/eao/failure-prediction/reports/status/balanced_validation_monitor.yaml
```

Because the timer is transient, it must be recreated after a reboot. Its output is
`reports/status/balanced_validation_monitor.yaml`; its append-only history is
`logs/monitoring/balanced_validation_v1.jsonl`. Operational decisions still use the
immutable campaign ledger, replacement manifests and generated wave reports as the
authoritative record; the monitor report is a convenience snapshot, not a scientific
result.

Do not repoint the validation timer while validation is live. Once its cumulative-324
report and immutable inventory validate, stop that timer and create a separate
`research2-targeted-monitor` timer using the targeted manifest and
`reports/status/targeted_development_monitor.yaml`. The monitor remains observe-only;
campaign transition, finalization and replacement decisions stay explicit and audited.
The targeted runner independently enforces this boundary and refuses collection unless
the cumulative validation report and hash-addressed 324-episode inventory both pass.
