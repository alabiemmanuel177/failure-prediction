#!/usr/bin/env python3
"""Write the outcome-free completion record of a protected confirmatory campaign.

The pilot summariser deliberately refuses to call a protected campaign complete and
reports outcome distributions, which must not be inspected before the frozen
evaluation. This finaliser resolves every design cell through the same exact-once
rules (originals, declared replacements from child manifests) and writes only
counts and integrity: expected, observed, usable, invalid attempts, replacements,
per-map episode counts, and the missing/unexpected/duplicate/replacement-error lists.
No terminal states, events, durations or prevalence are written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402
from src.experiments import campaign_episodes, ledger_invalid_episode_keys, summarize_pilot  # noqa: E402
from src.experiments.campaigns import declared_replacements  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-root", type=Path, default=ROOT / "data/raw/summaries")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protected_test_used") is not True:
        raise SystemExit("this finaliser is only for protected confirmatory campaigns")
    expected = campaign_episodes(manifest)
    summaries = [yaml.safe_load(path.read_text(encoding="utf-8"))
                 for path in sorted(args.summary_root.glob("*.yaml"))]
    full = summarize_pilot(manifest["campaign_id"], expected, summaries, declared_replacements(ROOT, manifest),
                           ledger_invalid_keys=ledger_invalid_episode_keys(ROOT, manifest))
    counts = full["counts"]
    integrity = full["integrity"]
    complete = (
        counts["usable"] == counts["expected"] == len(expected)
        and not integrity["missing_episode_keys"] and not integrity["unexpected_episode_keys"]
        and not integrity["duplicate_episode_keys"] and not integrity["replacement_errors"]
        and full["protected_test_used"] is True
    )
    record = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "analysis_unit": "episode",
        "kind": "protected_confirmatory_completion_record",
        "protected_test_used": True,
        "protected_outcomes_consulted": False,
        "outcome_fields_omitted_by_design": True,
        "complete_and_artifact_valid": complete,
        "counts": {k: counts[k] for k in ("expected", "observed", "usable", "infrastructure_invalid_attempts",
                                           "infrastructure_replacements")},
        "episodes_by_map": full.get("by_map") if isinstance(full.get("by_map"), dict)
        and all(isinstance(v, int) for v in full.get("by_map", {}).values()) else None,
        "integrity": integrity,
        "campaign_manifest_sha256": sha256_file(args.manifest),
        "replacement_manifests": sorted(
            str(p.relative_to(ROOT)) for p in (ROOT / "data/manifests").glob("*_replacements_v*.yaml")
            if (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("parent_campaign_id") == manifest["campaign_id"]
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = args.output or (ROOT / "reports/confirmatory" / f"{manifest['campaign_id']}.cumulative{len(expected)}.yaml")
    publish_new_bytes(output, yaml.safe_dump(record, sort_keys=False).encode("utf-8"))
    print(f"{'COMPLETE' if complete else 'INCOMPLETE'}: {manifest['campaign_id']} usable {counts['usable']}/{counts['expected']}, "
          f"invalid attempts {counts['infrastructure_invalid_attempts']}, replacements {counts['infrastructure_replacements']} -> {output}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
