#!/usr/bin/env python3
"""Execute a Research 2 campaign manifest with several isolated simulator workers.

The dispatcher keeps every guarantee of ``scripts/run_balanced_pilot.py`` while
running ``N`` worker slots concurrently:

* manifest expansion, ordering and validation are imported from
  ``src.experiments`` and ``scripts/run_balanced_pilot.py`` (never copied);
* every slot runs ``scripts/run_research2_episode.py`` exactly as the sequential
  runner does, inside its own ROS domain (``60 + slot``), Gazebo partition
  (``research2_w<slot>``) and per-slot ROS/Gazebo home directories;
* one file lock serialises pending-key claiming and ledger appends, so an episode
  key can never be attempted twice, and the ledger stays exact-once;
* the manifest's free-space reserve is checked before every claim;
* an invalid episode (or a failed artifact validation) stops all further claiming
  while running episodes finish and are recorded;
* per-episode artifact validation and the wave/cumulative reports are the sequential
  path's own scripts, invoked at every multiple of the manifest wave size;
* each ledger row records ``concurrency_workers``, ``worker_slot``, ``ros_domain_id``,
  ``gz_partition``, the episode's ``system`` and the ``system_concurrency`` caps in
  force;
* per-system concurrency caps (``--max-system-concurrency s3=1``, repeatable) bound how
  many episodes of one ``system`` may run at the same time across all slots. The
  manifest's ``execution_policy.system_concurrency`` block supplies the default caps and
  an admitting amendment's ``decisions.concurrency.system_concurrency`` block is a floor
  of restrictions; the flag may only tighten either. A slot whose every pending episode
  is cap-blocked waits for a running episode of that system to finish rather than
  exiting;
* phased execution (``--systems s0`` then ``--systems s3``) restricts claiming to the
  episodes whose ``system`` is listed. Episodes of other systems are left pending and
  are never marked attempted; when only such episodes remain the run stops with
  ``systems_drained`` (a normal stop) and the final JSON lists ``remaining_by_system``
  so the next phase can be started. An amendment admitting a sequential manifest must
  declare ``decisions.concurrency.phased_execution`` (``{s0: 6, s3: 1}``) for a
  ``--systems`` run, and ``--workers`` may not exceed the cap of any requested system;
  ``--workers 1 --systems s3`` therefore runs GPU-perception episodes strictly one at a
  time in the single slot 0 (ROS domain 60, partition ``research2_w0``), with every
  row recording ``concurrency_workers: 1``, and the dispatcher still refuses while any
  other Research 2 runner holds a campaign lock. Every row records ``systems_filter``.

Wave and cumulative reports slice the ledger in *ledger order*: wave ``n`` is ledger
rows ``[(n-1)*wave_size, n*wave_size)`` (``summarize_pilot_wave`` selects rows by
position, and the cumulative summary counts every completed row), so a report is
produced whenever a ledger slice is complete, regardless of the manifest's design
order. Under a phased run the slices differ in composition from the design's balanced
waves (phase A waves hold only S0 episodes; a slice left partial when phase A drains
is completed by the first rows of phase B and reported then). Reports are never
produced for an incomplete slice, and the wave/system composition of every wave is
visible in the wave report's ``execution_provenance.episodes_by_system``.

A manifest preregistered with ``execution_policy.concurrency: 1`` is refused unless
``--concurrency-override`` is given *and* the manifest carries
``parallel_execution_admitted_by`` naming an approved protocol amendment that lists
the campaign under ``decisions.parallel_execution_admitted_campaigns``. The
dispatcher also refuses while any other Research 2 runner (sequential or parallel)
holds a campaign lock, and refuses protected campaigns unless the confirmatory gate
passes. ``--dry-run`` prints the slot plan and exits without locking anything.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.monitor_campaign_status import lock_is_held  # noqa: E402
from scripts.run_balanced_pilot import (  # noqa: E402
    supplement_prerequisite_findings, targeted_prerequisite_findings,
)
from scripts.run_balanced_pilot_continuous import (  # noqa: E402
    finalize_confirmatory_wave, finalize_wave, jsonl,
)
from scripts.run_live_integrity_campaign import (  # noqa: E402
    append_ledger, attempted_keys, episode_command, research1_campaign_active,
    validate_completed_artifact,
)
from src.experiments import (  # noqa: E402
    CONCURRENCY_CHECK_KIND, PAIRED_RECOVERY_KIND, RECOVERY_PILOT_KIND, balanced_execution_order,
    expand_balanced_pilot, expand_recovery_policies, validate_paired_recovery, validate_recovery_pilot,
    normalise_system_concurrency, system_concurrency_findings, targeted_execution_order,
    validate_balanced_pilot, validate_concurrency_check, validate_confirmatory_campaign,
    validate_development_supplement, validate_parallel_execution_admission,
    validate_targeted_campaign,
)
from src.experiments.campaigns import (  # noqa: E402
    SYSTEM_ID_PATTERN, admission_condition_report_path, parallel_admission_terms,
)
from src.experiments.worker_slots import (  # noqa: E402
    DEFAULT_DOMAIN_BASE, slot_environment, slot_plan,
)
from src.protected_data import held_out_campaign_gate  # noqa: E402

DISPATCHER_ID = "run_campaign_parallel_v1"
SYSTEMS_DRAINED = "systems_drained"  # every pending episode lies outside --systems
DRAIN_REQUESTED = "drain_requested"  # operator marker: finish in-flight episodes, claim no more
NORMAL_STOPS = {"queue_exhausted", "claim_limit_reached", SYSTEMS_DRAINED, DRAIN_REQUESTED}
CAP_BLOCKED = "cap_blocked"  # claim() result: pending work exists but every item is capped
WAVE_REPORTING = "ledger_order_slices"
DEFAULT_CAP_POLL_SECONDS = 1.0


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def classify_manifest(document: dict) -> dict[str, bool]:
    kind = document.get("campaign_kind")
    supplement = kind == "development_supplement"
    return {
        # The paired recovery campaign is protected held-out work: it takes the
        # confirmatory readiness gate and the outcome-free wave finaliser.
        "confirmatory": kind in ("held_out_confirmatory", PAIRED_RECOVERY_KIND),
        "supplement": supplement,
        "targeted": kind == "targeted_event_floor" or supplement,
        "concurrency_check": kind == CONCURRENCY_CHECK_KIND,
        "recovery_pilot": kind == RECOVERY_PILOT_KIND,
        "paired_recovery": kind == PAIRED_RECOVERY_KIND,
    }


def manifest_findings(document: dict, splits: dict) -> list[str]:
    kinds = classify_manifest(document)
    if kinds["recovery_pilot"]:
        return validate_recovery_pilot(document, splits)
    if kinds["paired_recovery"]:
        return validate_paired_recovery(document, splits)
    if kinds["confirmatory"]:
        return validate_confirmatory_campaign(document, splits)
    if kinds["concurrency_check"]:
        return validate_concurrency_check(document, splits)
    if kinds["supplement"]:
        return validate_development_supplement(document, splits)
    if kinds["targeted"]:
        return validate_targeted_campaign(document, splits)
    return validate_balanced_pilot(document, splits)


def ordered_episodes(document: dict) -> list[dict]:
    kinds = classify_manifest(document)
    expanded = expand_balanced_pilot(document)
    if kinds["recovery_pilot"] or kinds["paired_recovery"]:
        # Pair-complete order: every policy of one map/route/seed/fault cell consecutively.
        return expand_recovery_policies(document, targeted_execution_order(expanded))
    if kinds["targeted"] or kinds["confirmatory"] or kinds["concurrency_check"]:
        return targeted_execution_order(expanded)
    return balanced_execution_order(expanded)


def load_amendments(root: Path) -> dict[str, dict]:
    amendments: dict[str, dict] = {}
    for path in sorted((root / "configs").glob("protocol_amendment_*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and isinstance(document.get("amendment_id"), str):
            amendments[document["amendment_id"]] = document
    return amendments


def load_check_reports(root: Path, amendments: dict) -> dict[str, dict]:
    """Shift-check reports named by amendment admission conditions, when present."""
    reports: dict[str, dict] = {}
    for amendment in amendments.values():
        path = admission_condition_report_path(
            parallel_admission_terms(amendment)["admission_condition"]
        )
        if path is None or path in reports:
            continue
        candidate = root / path
        if candidate.is_file():
            loaded = yaml.safe_load(candidate.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                reports[path] = loaded
    return reports


def concurrency_admission(
    document: dict, requested_workers: int | None, *, override: bool, amendments: dict,
    check_reports: dict | None = None, systems: list[str] | None = None,
) -> tuple[int, list[str]]:
    """Resolve the worker count and the reasons it is (not) admitted.

    ``systems`` is the ``--systems`` phase restriction; for an amendment-admitted
    sequential manifest it must match the amendment's ``phased_execution`` block. A
    manifest preregistered with its own multi-worker policy carries no phase caps, so
    ``--systems`` there is only a claim filter under the preregistered concurrency.
    """
    findings: list[str] = []
    policy = document.get("execution_policy", {})
    try:
        declared = int(policy.get("concurrency", 1))
    except (TypeError, ValueError):
        return 1, ["execution_policy.concurrency must be an integer"]
    workers = requested_workers if requested_workers is not None else declared
    if workers < 1:
        findings.append("worker count must be positive")
    if declared < 2:
        if not override:
            findings.append(
                "manifest preregisters execution_policy.concurrency 1 (sequential); parallel "
                "execution requires --concurrency-override together with "
                "parallel_execution_admitted_by"
            )
        else:
            findings.extend(validate_parallel_execution_admission(
                document, amendments, workers=workers, check_reports=check_reports,
                systems=systems,
            ))
    elif workers > declared:
        findings.append(
            f"requested {workers} workers exceeds the preregistered concurrency {declared}"
        )
    return workers, findings


def amendment_system_concurrency(
    document: dict, amendments: dict, *, override: bool,
) -> dict[str, int]:
    """Cap floor declared by the amendment that admits a sequential manifest.

    Only a manifest that runs in parallel *because* an amendment admits it (concurrency
    1 plus ``--concurrency-override``) inherits the amendment's
    ``decisions.concurrency.system_concurrency``; a manifest preregistered with its own
    multi-worker policy carries its own caps.
    """
    try:
        declared = int(document.get("execution_policy", {}).get("concurrency", 1))
    except (TypeError, ValueError):
        declared = 1
    if declared >= 2 or not override:
        return {}
    amendment = amendments.get(document.get("parallel_execution_admitted_by"))
    if not isinstance(amendment, dict):
        return {}
    return normalise_system_concurrency(parallel_admission_terms(amendment)["system_concurrency"])


def amendment_phased_execution(
    document: dict, amendments: dict, *, override: bool,
) -> dict[str, int]:
    """Phase caps declared by the amendment that admits a sequential manifest (else {})."""
    try:
        declared = int(document.get("execution_policy", {}).get("concurrency", 1))
    except (TypeError, ValueError):
        declared = 1
    if declared >= 2 or not override:
        return {}
    amendment = amendments.get(document.get("parallel_execution_admitted_by"))
    if not isinstance(amendment, dict):
        return {}
    return normalise_system_concurrency(parallel_admission_terms(amendment)["phased_execution"])


def parse_systems_flag(value: str | None) -> list[str] | None:
    """Parse ``--systems s0[,s3]`` into a sorted list of system ids (None when absent)."""
    if value is None:
        return None
    systems = sorted({item.strip().lower() for item in str(value).split(",") if item.strip()})
    if not systems:
        raise ValueError("--systems expects a comma-separated list such as s0 or s0,s3")
    for system in systems:
        if not re.match(SYSTEM_ID_PATTERN, system):
            raise ValueError(f"--systems: system id must look like s0..s9, got {system!r}")
    return systems


def pending_by_system(episodes: list[dict], attempted: set[str]) -> dict[str, int]:
    """Count the not-yet-attempted episodes of every system in the campaign (zeros kept)."""
    counts = {str(episode.get("system", "")).lower(): 0 for episode in episodes}
    for episode in episodes:
        if episode["episode_key"] not in attempted:
            counts[str(episode.get("system", "")).lower()] += 1
    return dict(sorted(counts.items()))


def other_runner_locks(root: Path, campaign_id: str) -> list[str]:
    """Campaign lock files held by another Research 2 runner, including ours if stale."""
    held = []
    lock_root = root / "logs" / "campaigns"
    if not lock_root.is_dir():
        return held
    for path in sorted(lock_root.glob("*.lock")):
        if path.name.endswith(".ledger.lock"):
            continue
        if lock_is_held(path):
            held.append(str(path.relative_to(root)))
    return held


# -- per-system concurrency caps ---------------------------------------------
def parse_system_concurrency_flags(values: list[str] | None) -> dict[str, int]:
    """Parse repeated ``--max-system-concurrency SYSTEM=N`` values into a cap block."""
    caps: dict[str, int] = {}
    for raw in values or []:
        system, separator, count = str(raw).partition("=")
        system = system.strip().lower()
        if not separator or not system or not count.strip():
            raise ValueError(f"--max-system-concurrency expects SYSTEM=N, got {raw!r}")
        try:
            cap = int(count.strip())
        except ValueError as error:
            raise ValueError(f"--max-system-concurrency cap must be an integer: {raw!r}") from error
        if system in caps and caps[system] != cap:
            raise ValueError(f"--max-system-concurrency names {system} twice with different caps")
        caps[system] = cap
    findings = system_concurrency_findings(caps, where="--max-system-concurrency")
    if findings:
        raise ValueError("; ".join(findings))
    return caps


def resolve_system_concurrency(
    *, manifest_caps: dict | None, cli_caps: dict | None, amendment_caps: dict | None = None,
) -> dict[str, int]:
    """Effective per-system caps: the strictest of every source, never loosened by the CLI.

    The manifest's ``execution_policy.system_concurrency`` block is the preregistered
    default; an admitting amendment's ``system_concurrency`` block is a floor of
    restrictions (the stricter of the two wins silently, since neither is operator
    input); ``--max-system-concurrency`` may add caps or tighten declared ones, and a
    flag looser than a declared cap is refused rather than ignored.
    """
    declared: dict[str, int] = {}
    for source in (normalise_system_concurrency(manifest_caps), normalise_system_concurrency(amendment_caps)):
        for system, cap in source.items():
            declared[system] = min(cap, declared.get(system, cap))
    effective = dict(declared)
    for system, cap in normalise_system_concurrency(cli_caps).items():
        if system in declared and cap > declared[system]:
            raise ValueError(
                f"--max-system-concurrency {system}={cap} would loosen the declared cap "
                f"{system}={declared[system]}; the flag may only tighten"
            )
        effective[system] = min(cap, declared.get(system, cap))
    return dict(sorted(effective.items()))


def initial_claims_with_caps(
    pending: list[dict], workers: int, caps: dict[str, int],
) -> dict[str, list[str]]:
    """First claim of every slot if all slots start together and nothing finishes.

    A slot that finds every remaining episode cap-blocked at start is shown as
    waiting (an empty list); the exact later interleaving depends on run times.
    """
    running: dict[str, int] = {}
    taken: set[str] = set()
    assignment: dict[str, list[str]] = {}
    for slot in range(workers):
        assignment[f"slot{slot}"] = []
        for episode in pending:
            key = episode["episode_key"]
            system = str(episode.get("system", "")).lower()
            if key in taken or running.get(system, 0) >= caps.get(system, workers + 1):
                continue
            taken.add(key)
            running[system] = running.get(system, 0) + 1
            assignment[f"slot{slot}"].append(key)
            break
    return assignment


@dataclass
class DispatchResult:
    claimed: int = 0
    recorded: int = 0
    invalid: int = 0
    stop_reason: str | None = None
    waves_finalized: list[int] = field(default_factory=list)
    waves_skipped_invalid: list[int] = field(default_factory=list)
    waves_complete_at_start: list[int] = field(default_factory=list)
    systems: list[str] | None = None
    remaining_by_system: dict[str, int] = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        return sum(self.remaining_by_system.values())

    @property
    def returncode(self) -> int:
        if self.invalid:
            return 1
        if self.stop_reason and self.stop_reason not in NORMAL_STOPS:
            return 2
        return 0


class ParallelDispatcher:
    """Exact-once work queue over isolated worker slots.

    ``run_episode(episode, slot)`` returns the episode process return code;
    ``validate_artifact(episode)`` returns the validator return code; ``disk_free()``
    returns free bytes on the raw volume; ``finalize(wave, completed)`` produces the
    wave/cumulative reports. All four are injectable so the queue can be exercised
    with fakes and without ROS.

    ``system_concurrency`` (``{"s3": 1}``) caps how many in-flight episodes may share a
    ``system``; a claim that finds only cap-blocked work returns ``CAP_BLOCKED`` and the
    slot waits (``cap_poll_seconds`` at most between checks, woken early by any
    record) instead of exiting, so the queue still drains exactly once.

    ``systems`` (``["s0"]``) restricts claiming to episodes of those systems: the
    others stay pending and untouched, and once only they remain the run stops with
    ``SYSTEMS_DRAINED``. Wave reports are ledger-order slices, so they are produced
    whenever a slice fills, whatever the design order of the rows in it.
    """

    def __init__(
        self, *, campaign_id: str, episodes: list[dict], ledger: Path, workers: int,
        plan: list[dict], run_episode: Callable[[dict, int], int],
        validate_artifact: Callable[[dict], int], disk_free: Callable[[], int],
        minimum_free_bytes: float, wave_size: int,
        finalize: Callable[[int, int], None] | None = None,
        stop_on_first_invalid: bool = True, max_claims: int | None = None,
        claims_lock: Path | None = None, extra_ledger_fields: dict | None = None,
        system_concurrency: dict[str, int] | None = None,
        cap_poll_seconds: float = DEFAULT_CAP_POLL_SECONDS,
        systems: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        if len(plan) != workers:
            raise ValueError("slot plan must describe every worker")
        if wave_size < 1:
            raise ValueError("wave size must be positive")
        cap_findings = system_concurrency_findings(system_concurrency, where="system_concurrency")
        if cap_findings:
            raise ValueError("; ".join(cap_findings))
        if cap_poll_seconds <= 0:
            raise ValueError("cap poll interval must be positive")
        if systems is not None:
            systems = sorted({str(system).lower() for system in systems})
            if not systems:
                raise ValueError("systems filter must name at least one system")
        self.systems: list[str] | None = systems
        self.campaign_id = campaign_id
        self.episodes = episodes
        self.ledger = ledger
        self.workers = workers
        self.plan = plan
        self.run_episode = run_episode
        self.validate_artifact = validate_artifact
        self.disk_free = disk_free
        self.minimum_free_bytes = float(minimum_free_bytes)
        self.drain_marker: Path | None = None
        self.start_stagger_seconds: float = 0.0
        self.wave_size = wave_size
        self.finalize = finalize
        self.stop_on_first_invalid = stop_on_first_invalid
        self.max_claims = max_claims
        self.claims_lock = claims_lock or ledger.with_suffix(".ledger.lock")
        self.extra_ledger_fields = dict(extra_ledger_fields or {})
        self.system_concurrency = dict(sorted(normalise_system_concurrency(system_concurrency).items()))
        self.cap_poll_seconds = float(cap_poll_seconds)
        self._memory_lock = threading.Lock()
        self._stop = threading.Event()
        self._in_flight: set[str] = set()
        self._in_flight_systems: dict[str, int] = {}
        self._release = threading.Condition()
        self._result = DispatchResult(systems=list(systems) if systems else None)
        self._finalized_waves: set[int] = set()
        self._peak_in_flight_systems: dict[str, int] = {}

    # -- exact-once claiming -------------------------------------------------
    def _locked(self):
        """Hold the single claims/ledger file lock (and the in-process mutex)."""
        dispatcher = self

        class _Guard:
            def __enter__(self_inner):
                dispatcher._memory_lock.acquire()
                dispatcher.claims_lock.parent.mkdir(parents=True, exist_ok=True)
                self_inner.stream = dispatcher.claims_lock.open("a+", encoding="utf-8")
                fcntl.flock(self_inner.stream.fileno(), fcntl.LOCK_EX)
                return self_inner

            def __exit__(self_inner, *_exc):
                fcntl.flock(self_inner.stream.fileno(), fcntl.LOCK_UN)
                self_inner.stream.close()
                dispatcher._memory_lock.release()
                return False

        return _Guard()

    @staticmethod
    def _system_of(episode: dict) -> str:
        return str(episode.get("system", "")).lower()

    def _cap_blocked(self, episode: dict) -> bool:
        system = self._system_of(episode)
        cap = self.system_concurrency.get(system)
        return cap is not None and self._in_flight_systems.get(system, 0) >= cap

    def _outside_phase(self, episode: dict) -> bool:
        return self.systems is not None and self._system_of(episode) not in self.systems

    def claim(self) -> dict | str | None:
        """Claim the next pending episode: a dict, ``CAP_BLOCKED`` to wait, or None to stop."""
        with self._locked():
            if self._stop.is_set():
                return None
            if self.max_claims is not None and self._result.claimed >= self.max_claims:
                self._stop_with("claim_limit_reached")
                return None
            if self.drain_marker is not None and self.drain_marker.exists():
                self._stop_with(DRAIN_REQUESTED)
                return None
            free = self.disk_free()
            if free < self.minimum_free_bytes:
                self._stop_with(
                    f"free-space reserve reached: {free / 2**30:.1f} GiB available, "
                    f"{self.minimum_free_bytes / 2**30:.1f} GiB required"
                )
                return None
            attempted = attempted_keys(self.ledger)
            blocked = False
            deferred = False
            for episode in self.episodes:
                key = episode["episode_key"]
                if key in attempted or key in self._in_flight:
                    continue
                if self._outside_phase(episode):
                    deferred = True  # left pending for a later phase, never attempted
                    continue
                if self._cap_blocked(episode):
                    blocked = True
                    continue
                self._in_flight.add(key)
                system = self._system_of(episode)
                running = self._in_flight_systems.get(system, 0) + 1
                self._in_flight_systems[system] = running
                self._peak_in_flight_systems[system] = max(
                    running, self._peak_in_flight_systems.get(system, 0)
                )
                self._result.claimed += 1
                return episode
            if blocked:
                # Only capped systems remain and their cap is fully occupied by running
                # episodes; the slot waits for one of them to be recorded.
                return CAP_BLOCKED
            self._stop_with(SYSTEMS_DRAINED if deferred else "queue_exhausted")
            return None

    def _stop_with(self, reason: str) -> None:
        if not self._stop.is_set():
            self._result.stop_reason = reason
            self._stop.set()
        with self._release:
            self._release.notify_all()

    def remaining_by_system(self) -> dict[str, int]:
        """Pending (never attempted) episodes per system, read from the ledger."""
        with self._locked():
            attempted = attempted_keys(self.ledger)
        return pending_by_system(self.episodes, attempted)

    @property
    def peak_in_flight_by_system(self) -> dict[str, int]:
        """Largest number of simultaneously running episodes seen per system."""
        return dict(sorted(self._peak_in_flight_systems.items()))

    def record(self, episode: dict, slot: int, row: dict) -> int:
        """Append one ledger row under the file lock and return the ledger length."""
        key = episode["episode_key"]
        with self._locked():
            if key in attempted_keys(self.ledger):
                raise RuntimeError(f"ledger already contains {key}; refusing a second row")
            entry = self.plan[slot]
            append_ledger(self.ledger, {
                **row,
                "campaign_id": self.campaign_id,
                "episode_key": key,
                "system": self._system_of(episode),
                "concurrency_workers": self.workers,
                "system_concurrency": dict(self.system_concurrency),
                "systems_filter": list(self.systems) if self.systems else None,
                "worker_slot": slot,
                "ros_domain_id": entry["ros_domain_id"],
                "gz_partition": entry["gz_partition"],
                "dispatcher": DISPATCHER_ID,
                **self.extra_ledger_fields,
            })
            if key in self._in_flight:
                self._in_flight.discard(key)
                system = self._system_of(episode)
                self._in_flight_systems[system] = max(self._in_flight_systems.get(system, 0) - 1, 0)
            self._result.recorded += 1
            if row.get("returncode"):
                self._result.invalid += 1
                if self.stop_on_first_invalid:
                    self._stop_with(f"invalid episode {key}; retained for audit")
            length = len(jsonl(self.ledger))
        with self._release:
            self._release.notify_all()
        return length

    # -- worker loop -------------------------------------------------------
    def _worker(self, slot: int) -> None:
        # Stagger the first claim per slot: six Nav2/Gazebo stacks brought up in the same
        # second (every dispatcher restart) produced clusters of pre-goal lifecycle
        # invalids (recovery pilot, 7 September: six of six slots at 21:12:20Z).
        if self.start_stagger_seconds > 0 and slot > 0:
            time.sleep(slot * self.start_stagger_seconds)
        while True:
            episode = self.claim()
            if episode is None:
                return
            if episode == CAP_BLOCKED:
                with self._release:
                    self._release.wait(timeout=self.cap_poll_seconds)
                continue
            started = utc_now()
            try:
                episode_rc = int(self.run_episode(episode, slot))
            except Exception as error:  # noqa: BLE001 - a crash is an invalid attempt
                episode_rc = 70
                print(f"slot {slot}: episode runner crashed: {error}", file=sys.stderr)
            artifact_rc = None
            if episode_rc == 0:
                try:
                    artifact_rc = int(self.validate_artifact(episode))
                except Exception as error:  # noqa: BLE001
                    artifact_rc = 71
                    print(f"slot {slot}: artifact validation crashed: {error}", file=sys.stderr)
            effective = episode_rc or artifact_rc or 0
            self.record(episode, slot, {
                "started_utc": started,
                "finished_utc": utc_now(),
                "returncode": effective,
                "episode_returncode": episode_rc,
                "artifact_validation_returncode": artifact_rc,
            })

    def _finalize_complete_waves(self) -> None:
        if self.finalize is None:
            return
        with self._locked():
            rows = jsonl(self.ledger)
        for wave in range(1, len(rows) // self.wave_size + 1):
            if wave in self._finalized_waves:
                continue
            self._finalized_waves.add(wave)
            slice_rows = rows[(wave - 1) * self.wave_size:wave * self.wave_size]
            if any(row.get("returncode") for row in slice_rows):
                # The sequential controller never reaches a wave boundary past an
                # invalid row; leave this wave's reports to the researcher's audit.
                self._result.waves_skipped_invalid.append(wave)
                continue
            self.finalize(wave, wave * self.wave_size)
            self._result.waves_finalized.append(wave)

    def run(self) -> DispatchResult:
        with self._locked():
            rows_at_start = len(jsonl(self.ledger))
        self._result.waves_complete_at_start = list(range(1, rows_at_start // self.wave_size + 1))
        # Finalise waves already complete in the ledger BEFORE any slot claims work: a
        # finaliser failure (7 September: a NUL-filled research-log line after a host
        # reset) must stop the run with nothing in flight, never orphan an unrecorded
        # attempt whose subprocess outlives the crashed dispatcher.
        self._finalize_complete_waves()
        threads = [
            threading.Thread(target=self._worker, args=(slot,), name=f"slot-{slot}", daemon=True)
            for slot in range(self.workers)
        ]
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=0.2)
            self._finalize_complete_waves()
        self._finalize_complete_waves()
        if self._result.stop_reason is None:
            self._result.stop_reason = "queue_exhausted"
        self._result.remaining_by_system = self.remaining_by_system()
        return self._result


def build_episode_runner(
    campaign_id: str, output_root: Path, environments: list[dict[str, str]], log_root: Path,
) -> Callable[[dict, int], int]:
    """Launch ``run_research2_episode.py`` exactly as the sequential runner does."""

    def run(episode: dict, slot: int) -> int:
        slot_log = log_root / f"slot{slot}"
        slot_log.mkdir(parents=True, exist_ok=True)
        log_path = slot_log / f"{episode['episode_key']}.log"
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"# {utc_now()} slot={slot} {' '.join(environments[slot][k] for k in ('ROS_DOMAIN_ID', 'GZ_PARTITION'))}\n")
            stream.flush()
            result = subprocess.run(
                episode_command(campaign_id, episode, output_root), check=False,
                env=environments[slot], stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        return result.returncode

    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--start-stagger-seconds", type=float, default=20.0,
                        help="delay slot k's first simulator start by k times this many seconds")
    parser.add_argument("--workers", type=int, default=None,
                        help="worker slots; defaults to execution_policy.concurrency")
    parser.add_argument("--domain-base", type=int, default=DEFAULT_DOMAIN_BASE)
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="claim at most this many episodes in this invocation")
    parser.add_argument("--concurrency-override", action="store_true",
                        help="run a concurrency-1 manifest in parallel; requires "
                             "parallel_execution_admitted_by in the manifest")
    parser.add_argument("--max-system-concurrency", action="append", default=None,
                        metavar="SYSTEM=N",
                        help="run at most N episodes of SYSTEM at once (repeatable, e.g. "
                             "s3=1); may only tighten the manifest's or amendment's caps")
    parser.add_argument("--systems", default=None, metavar="s0[,s3]",
                        help="claim only episodes of these systems (phased execution); "
                             "the others stay pending for a later phase and the run stops "
                             "with systems_drained once only they remain")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        cli_caps = parse_system_concurrency_flags(args.max_system_concurrency)
        systems = parse_systems_flag(args.systems)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    document = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    splits = yaml.safe_load(
        (ROOT / "data/manifests/splits.template.yaml").read_text(encoding="utf-8")
    )
    kinds = classify_manifest(document)
    findings = manifest_findings(document, splits)
    if findings:
        raise SystemExit("invalid campaign manifest:\n- " + "\n- ".join(findings))
    episodes = ordered_episodes(document)
    expected = int(document["expected_episode_count"])
    if len(episodes) != expected:
        raise SystemExit(f"expanded {len(episodes)} episodes, expected {expected}")
    if len({item["episode_key"] for item in episodes}) != len(episodes):
        raise SystemExit("episode keys are not unique")
    if kinds["recovery_pilot"] or kinds["paired_recovery"]:
        # Policies of one pairing cell share the seed by design; (seed, policy) is unique.
        if len({(item["seed"], item["recovery_policy_id"]) for item in episodes}) != len(episodes):
            raise SystemExit("seed/policy combinations are not unique")
    elif len({item["seed"] for item in episodes}) != len(episodes):
        raise SystemExit("seeds are not unique")
    campaign_id = str(document["campaign_id"])
    amendments = load_amendments(ROOT)
    workers, admission = concurrency_admission(
        document, args.workers, override=args.concurrency_override, amendments=amendments,
        check_reports=load_check_reports(ROOT, amendments), systems=systems,
    )
    if admission:
        raise SystemExit("parallel execution is not admitted:\n- " + "\n- ".join(admission))
    campaign_systems = sorted({str(item.get("system", "")).lower() for item in episodes})
    if systems is not None:
        unknown = [system for system in systems if system not in campaign_systems]
        if unknown:
            raise SystemExit(
                f"--systems names {', '.join(unknown)}, which the campaign does not contain "
                f"(systems: {', '.join(campaign_systems)})"
            )
    policy = document["execution_policy"]
    manifest_caps = policy.get("system_concurrency")
    cap_findings = system_concurrency_findings(
        manifest_caps, where="execution_policy.system_concurrency",
    )
    if cap_findings:
        raise SystemExit("invalid campaign manifest:\n- " + "\n- ".join(cap_findings))
    amendment_caps = amendment_system_concurrency(
        document, amendments, override=args.concurrency_override,
    )
    phase_caps = amendment_phased_execution(
        document, amendments, override=args.concurrency_override,
    )
    try:
        system_caps = resolve_system_concurrency(
            manifest_caps=manifest_caps, cli_caps=cli_caps, amendment_caps=amendment_caps,
        )
        plan = slot_plan(workers, ROOT, domain_base=args.domain_base)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    ledger = ROOT / "logs/campaigns" / f"{campaign_id}.jsonl"
    attempted = attempted_keys(ledger)
    pending = [episode for episode in episodes if episode["episode_key"] not in attempted]
    pending_counts = pending_by_system(episodes, attempted)
    claimable = [
        episode for episode in pending
        if systems is None or str(episode.get("system", "")).lower() in systems
    ]
    wave_size = int(document["execution_policy"]["maximum_episodes_per_invocation"])
    minimum_free = float(policy["minimum_free_space_gib_before_episode"]) * 2**30
    held_locks = other_runner_locks(ROOT, campaign_id)
    initial_assignment = initial_claims_with_caps(claimable, workers, system_caps)
    print(json.dumps({
        "campaign_id": campaign_id,
        "campaign_kind": document.get("campaign_kind"),
        "dispatcher": DISPATCHER_ID,
        "total": len(episodes),
        "attempted": len(attempted),
        "pending": len(pending),
        "pending_by_system": pending_counts,
        "systems": systems,
        "claimable_in_systems": len(claimable),
        "phased_execution": phase_caps,
        "workers": workers,
        "preregistered_concurrency": policy.get("concurrency"),
        "concurrency_override": args.concurrency_override,
        "system_concurrency": system_caps,
        "system_concurrency_sources": {
            "manifest": normalise_system_concurrency(manifest_caps),
            "amendment_floor": normalise_system_concurrency(amendment_caps),
            "cli": cli_caps,
        },
        "wave_size": wave_size,
        "start_stagger_seconds": args.start_stagger_seconds,
        "wave_reporting": WAVE_REPORTING,
        "minimum_free_space_gib": policy["minimum_free_space_gib_before_episode"],
        "stop_on_first_invalid": bool(policy.get("stop_on_first_invalid", True)),
        "protected_test_used": bool(document.get("protected_test_used")),
        "slot_plan": plan,
        "first_claims_if_all_slots_start_together": initial_assignment,
        "other_runner_locks_held": held_locks,
        "would_refuse_now": bool(held_locks),
    }, indent=2, sort_keys=False))
    if args.dry_run:
        return 0

    if held_locks:
        raise SystemExit(
            "another Research 2 runner holds a campaign lock: " + ", ".join(held_locks)
        )
    validator_args: tuple[str, ...] = ()
    if kinds["confirmatory"] or document.get("protected_test_used") is True:
        gate_findings = held_out_campaign_gate(ROOT, list(document["design"]["map_routes"]))
        if gate_findings:
            raise SystemExit(
                "protected campaign is blocked until the model freeze and protected split "
                "assignment pass the confirmatory readiness gate:\n- "
                + "\n- ".join(gate_findings)
            )
        validator_args = ("--allow-protected-after-freeze",)
    if kinds["targeted"]:
        prerequisite = (
            supplement_prerequisite_findings(ROOT) if kinds["supplement"]
            else targeted_prerequisite_findings(ROOT)
        )
        if prerequisite:
            raise SystemExit(
                "targeted development is blocked until validation is finalized:\n- "
                + "\n- ".join(prerequisite)
            )
        inventory_gate = subprocess.run([
            sys.executable, str(ROOT / "scripts/validate_validation_dataset_inventory.py"),
        ], check=False)
        if inventory_gate.returncode:
            raise SystemExit("targeted development is blocked: validation inventory validation failed")
    drain_path = ROOT / "logs/campaigns" / f"{campaign_id}.drain"
    pause_path = ROOT / "logs/campaigns" / f"{campaign_id}.pause"
    if pause_path.exists():
        raise SystemExit(
            f"campaign pause marker present: {pause_path}; inspect its reason before resuming"
        )
    lock_path = ROOT / "logs/campaigns" / f"{campaign_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another {campaign_id} runner holds {lock_path}")
    if other_runner_locks(ROOT, campaign_id) != [str(lock_path.relative_to(ROOT))]:
        raise SystemExit("another Research 2 runner took a campaign lock; refusing to start")
    gate = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_readiness.py"), "--stage",
         "confirmatory" if kinds["confirmatory"] else "extraction"],
        check=False,
    )
    if gate.returncode:
        raise SystemExit("campaign is blocked until the readiness gate passes")
    if os.environ.get("ROS_DISTRO") != "jazzy":
        raise SystemExit("source scripts/env_research2.sh; campaign requires ROS 2 Jazzy")
    if research1_campaign_active():
        raise SystemExit("Research 1 confirmatory service is active")
    output_root = args.output_root.resolve()
    environments = [
        slot_environment(slot, workers, ROOT, domain_base=args.domain_base)
        for slot in range(workers)
    ]
    for entry in plan:
        Path(entry["ros_log_dir"]).mkdir(parents=True, exist_ok=True)
        Path(entry["gz_homedir"]).mkdir(parents=True, exist_ok=True)
    log_root = ROOT / "logs/campaigns" / f"{campaign_id}.parallel"
    allowed = ["development"] if not kinds["confirmatory"] else ["held_out_map_test"]
    split_name = str(document.get("allowed_splits", allowed)[0])
    report_root = ROOT / (
        "reports/pilot" if split_name == "development"
        else "reports/confirmatory" if kinds["confirmatory"] else "reports/validation"
    )

    def finalize(wave: int, completed: int) -> None:
        if kinds["confirmatory"]:
            finalize_confirmatory_wave(
                campaign_id=campaign_id, wave=wave, completed=completed, ledger=ledger,
                log=ROOT / "logs/research-log.jsonl",
            )
        else:
            finalize_wave(
                campaign_id=campaign_id, wave=wave, completed=completed,
                manifest=args.manifest, report_root=report_root,
                log=ROOT / "logs/research-log.jsonl", split_name=split_name,
            )

    dispatcher = ParallelDispatcher(
        campaign_id=campaign_id, episodes=episodes, ledger=ledger, workers=workers,
        plan=plan,
        run_episode=build_episode_runner(campaign_id, output_root, environments, log_root),
        validate_artifact=lambda episode: validate_completed_artifact(
            campaign_id, episode["episode_key"], output_root, validator_args
        ),
        disk_free=lambda: shutil.disk_usage(output_root).free,
        minimum_free_bytes=minimum_free, wave_size=wave_size, finalize=finalize,
        stop_on_first_invalid=bool(policy.get("stop_on_first_invalid", True)),
        max_claims=args.max_episodes,
        extra_ledger_fields={"domain_base": args.domain_base},
        system_concurrency=system_caps, systems=systems,
    )
    dispatcher.drain_marker = drain_path
    dispatcher.start_stagger_seconds = max(0.0, float(args.start_stagger_seconds))
    result = dispatcher.run()
    ledger_rows = len(jsonl(ledger))
    print(json.dumps({
        "campaign_id": campaign_id,
        "claimed": result.claimed,
        "recorded": result.recorded,
        "invalid": result.invalid,
        "waves_finalized": result.waves_finalized,
        "waves_skipped_invalid": result.waves_skipped_invalid,
        "waves_complete_at_start": result.waves_complete_at_start,
        "wave_reporting": WAVE_REPORTING,
        "partial_wave_rows": ledger_rows % wave_size,
        "stop_reason": result.stop_reason,
        "ledger_rows": ledger_rows,
        "concurrency_workers": workers,
        "systems": systems,
        "phased_execution": phase_caps,
        "system_concurrency": system_caps,
        "peak_in_flight_by_system": dispatcher.peak_in_flight_by_system,
        "remaining_by_system": result.remaining_by_system,
        "remaining": result.remaining,
        "campaign_complete": result.remaining == 0,
    }, indent=2, sort_keys=True))
    if result.invalid:
        raise SystemExit(
            f"campaign stopped on invalid episode ({result.stop_reason}); retained for audit"
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
