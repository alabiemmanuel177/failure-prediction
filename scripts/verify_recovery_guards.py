#!/usr/bin/env python3
"""Generate non-model exhaustive evidence for frozen recovery guards."""

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import sha256_file  # noqa: E402
from src.recovery import GuardConfig  # noqa: E402
from src.recovery.verification import verify_guard_space  # noqa: E402


def main() -> int:
    config_path = ROOT / "configs/recovery_guards.yaml"
    output_path = ROOT / "reports/recovery/guard_verification.yaml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = GuardConfig(
        minimum_rear_clearance_m=float(document["minimum_rear_clearance_m"]),
        minimum_rotation_clearance_m=float(document["minimum_rotation_clearance_m"]),
        maximum_repeated_recoveries=int(document["maximum_repeated_recoveries"]),
    )
    verification = verify_guard_space(config)
    report = {
        "schema_version": 1,
        "evidence_type": "non_model_exhaustive_guard_verification",
        "protected_test_used": False,
        "live_execution_performed": False,
        "guard_config": str(config_path.relative_to(ROOT)),
        "guard_config_sha256": sha256_file(config_path),
        **verification,
        "admission_effect": (
            "guard_engineering_complete; paired closed-loop recovery remains post-model"
        ),
    }
    if output_path.exists():
        existing = yaml.safe_load(output_path.read_text(encoding="utf-8"))
        if existing != report:
            raise SystemExit(f"refusing to overwrite differing evidence: {output_path}")
        print(f"guard evidence already current: {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
        print(f"wrote exhaustive guard evidence to {output_path}")
    if not report["passed"]:
        print(f"RECOVERY GUARD VERIFICATION FAILED: {report['violation_count']} violations")
        return 1
    print(
        f"RECOVERY GUARD VERIFICATION PASSED: {report['states_checked']} states; "
        f"{report['policy_decisions_checked']} policy decisions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
