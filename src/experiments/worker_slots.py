"""Per-worker-slot simulator isolation for parallel Research 2 campaign execution.

Pure helpers shared by ``scripts/run_campaign_parallel.py`` (which builds one
environment per slot) and ``scripts/run_research2_episode.py`` (which, only when a
slot is declared, verifies that it really was launched inside that slot's isolated
middleware and records the slot in its provenance). No ROS import happens here.

Isolation follows Research 1's proven six-worker operation: one ROS domain per
worker starting at a domain base of 60, one Gazebo transport partition per worker,
and per-worker ROS/Gazebo home directories. Research 2's sequential campaigns own
domain 52 (``scripts/env_research2.sh``); no slot may ever land there.
"""

from __future__ import annotations

import os
from pathlib import Path

SEQUENTIAL_DOMAIN = 52
DEFAULT_DOMAIN_BASE = 60
MAXIMUM_DOMAIN = 232
SLOT_VARIABLE = "RESEARCH2_WORKER_SLOT"
WORKERS_VARIABLE = "RESEARCH2_CONCURRENCY_WORKERS"
PARTITION_PREFIX = "research2_w"


def slot_partition(slot: int) -> str:
    return f"{PARTITION_PREFIX}{int(slot)}"


def slot_domain(slot: int, domain_base: int = DEFAULT_DOMAIN_BASE) -> int:
    return int(domain_base) + int(slot)


def slot_plan(
    workers: int, root: Path, *, domain_base: int = DEFAULT_DOMAIN_BASE,
) -> list[dict]:
    """Describe every slot's isolated middleware without touching the environment."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if domain_base < 0 or slot_domain(workers - 1, domain_base) > MAXIMUM_DOMAIN:
        raise ValueError(
            f"domain base {domain_base} with {workers} workers leaves the ROS domain range"
        )
    plan = []
    for slot in range(workers):
        domain = slot_domain(slot, domain_base)
        if domain == SEQUENTIAL_DOMAIN:
            raise ValueError(
                f"slot {slot} would use ROS domain {SEQUENTIAL_DOMAIN}, which belongs to "
                "the sequential Research 2 runner"
            )
        partition = slot_partition(slot)
        plan.append({
            "slot": slot,
            "ros_domain_id": domain,
            "gz_partition": partition,
            "ros_log_dir": str(Path(root) / "logs" / "ros" / "parallel" / partition),
            "gz_homedir": str(Path(root) / "logs" / "gazebo" / "parallel" / partition),
        })
    return plan


def slot_environment(
    slot: int, workers: int, root: Path, *, domain_base: int = DEFAULT_DOMAIN_BASE,
    base_environment: dict | None = None,
) -> dict[str, str]:
    """Return the complete process environment for one worker slot.

    ``ROS_DOMAIN_ID`` and ``RESEARCH2_ROS_DOMAIN_ID`` are both set because the bag
    recorder (``scripts/record_research2_bag.sh``) re-sources ``env_research2.sh``,
    which derives the domain from the latter. ``GZ_PARTITION``, ``ROS_LOG_DIR`` and
    ``GZ_HOMEDIR`` are honoured by that script when already present.
    """
    entry = slot_plan(workers, root, domain_base=domain_base)[slot]
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update({
        "ROS_DOMAIN_ID": str(entry["ros_domain_id"]),
        "RESEARCH2_ROS_DOMAIN_ID": str(entry["ros_domain_id"]),
        "GZ_PARTITION": entry["gz_partition"],
        "ROS_LOG_DIR": entry["ros_log_dir"],
        "GZ_HOMEDIR": entry["gz_homedir"],
        SLOT_VARIABLE: str(slot),
        WORKERS_VARIABLE: str(workers),
    })
    return environment


def worker_slot_findings(environ: dict) -> list[str]:
    """Fail-closed consistency check for an episode launched inside a worker slot.

    Returns an empty list when no slot is declared (the sequential path) or when the
    declared slot's domain and partition are exactly what ``slot_environment`` set.
    """
    raw_slot = environ.get(SLOT_VARIABLE)
    if raw_slot is None:
        return []
    findings: list[str] = []
    try:
        slot = int(raw_slot)
    except ValueError:
        return [f"{SLOT_VARIABLE} must be an integer, got {raw_slot!r}"]
    if slot < 0:
        findings.append(f"{SLOT_VARIABLE} must be non-negative")
    domain = environ.get("ROS_DOMAIN_ID")
    research2_domain = environ.get("RESEARCH2_ROS_DOMAIN_ID")
    if domain is None or domain != research2_domain:
        findings.append(
            "ROS_DOMAIN_ID and RESEARCH2_ROS_DOMAIN_ID must agree inside a worker slot"
        )
    if domain == str(SEQUENTIAL_DOMAIN):
        findings.append(
            f"worker slot {slot} may not use the sequential ROS domain {SEQUENTIAL_DOMAIN}"
        )
    if environ.get("GZ_PARTITION") != slot_partition(slot):
        findings.append(
            f"GZ_PARTITION must be {slot_partition(slot)} for worker slot {slot}"
        )
    workers = environ.get(WORKERS_VARIABLE)
    try:
        if workers is None or int(workers) <= slot:
            findings.append(f"{WORKERS_VARIABLE} must exceed the slot index")
    except ValueError:
        findings.append(f"{WORKERS_VARIABLE} must be an integer")
    return findings


def worker_slot_provenance(environ: dict) -> dict:
    """Provenance fields for an episode run inside a slot; empty on the sequential path."""
    if environ.get(SLOT_VARIABLE) is None:
        return {}
    return {
        "worker_slot": int(environ[SLOT_VARIABLE]),
        "concurrency_workers": int(environ[WORKERS_VARIABLE]),
    }
