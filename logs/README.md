# Research logs

`research-log.jsonl` is an append-only, hash-chained record of protocol decisions, changes, exclusions, data issues, experiments, and reviews. Do not edit existing lines. Add records with:

```bash
python3 scripts/research_log.py add \
  --kind decision \
  --actor "Emmanuel Alabi Olasubomi" \
  --message "Frozen event precedence for pilot" \
  --metadata '{"protocol_version":"1.1"}'
```

Verify integrity with:

```bash
python3 scripts/research_log.py verify
```

Episode-level runtime events belong in immutable episode manifests and bags. This log is for research decisions and audit history, not high-rate telemetry.

