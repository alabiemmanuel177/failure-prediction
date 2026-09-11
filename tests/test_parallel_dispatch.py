"""Parallel campaign dispatcher: isolation, exact-once claiming, refusals (no ROS)."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

import pytest
import yaml

from scripts.run_campaign_parallel import (
    CAP_BLOCKED, DISPATCHER_ID, SYSTEMS_DRAINED, WAVE_REPORTING, ParallelDispatcher,
    amendment_phased_execution, amendment_system_concurrency, concurrency_admission,
    initial_claims_with_caps, load_amendments, main as dispatcher_main, ordered_episodes,
    other_runner_locks, parse_system_concurrency_flags, parse_systems_flag,
    pending_by_system, resolve_system_concurrency,
)
from scripts.run_live_integrity_campaign import attempted_keys
from src.experiments import (
    expand_balanced_pilot, targeted_execution_order, validate_parallel_execution_admission,
)
from src.experiments.worker_slots import (
    SEQUENTIAL_DOMAIN, slot_environment, slot_plan, worker_slot_findings,
    worker_slot_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
TARGETED = ROOT / "data/manifests/targeted_development_v1.yaml"
SHIFT_CHECK = ROOT / "data/manifests/concurrency_shift_check_v1.yaml"
SHIFT_CHECK_V2 = ROOT / "data/manifests/concurrency_shift_check_v2.yaml"


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fake_episodes(count: int) -> list[dict]:
    return [
        {"episode_key": f"k{index:02d}", "map": "dev_00", "route": "dev_00_r0", "system": "s0",
         "family": "none", "severity": "none", "seed": 4_000_000 + index}
        for index in range(count)
    ]


def mixed_system_episodes(count: int, systems: tuple[str, ...] = ("s0", "s3", "s0")) -> list[dict]:
    """Fake queue whose systems cycle through ``systems``, in dispatcher order."""
    return [
        {**episode, "system": systems[index % len(systems)]}
        for index, episode in enumerate(fake_episodes(count))
    ]


def make_dispatcher(tmp_path: Path, episodes: list[dict], *, workers: int, run, validate=None,
                    disk_free=None, minimum_free=0, wave_size=12, finalize=None,
                    max_claims=None, system_concurrency=None,
                    cap_poll_seconds=0.01, systems=None) -> ParallelDispatcher:
    ledger = tmp_path / "logs/campaigns/fake.jsonl"
    return ParallelDispatcher(
        campaign_id="fake", episodes=episodes, ledger=ledger, workers=workers,
        plan=slot_plan(workers, tmp_path), run_episode=run,
        validate_artifact=validate or (lambda episode: 0),
        disk_free=disk_free or (lambda: 10**15), minimum_free_bytes=minimum_free,
        wave_size=wave_size, finalize=finalize, max_claims=max_claims,
        system_concurrency=system_concurrency, cap_poll_seconds=cap_poll_seconds,
        systems=systems,
    )


class ConcurrencyProbe:
    """Records, per system, the largest number of episodes running at once."""

    def __init__(self, hold_seconds: float = 0.01) -> None:
        self.hold_seconds = hold_seconds
        self.lock = threading.Lock()
        self.running: dict[str, int] = {}
        self.peak: dict[str, int] = {}
        self.calls: dict[str, int] = {}

    def __call__(self, episode: dict, slot: int) -> int:
        system = episode["system"]
        with self.lock:
            self.calls[episode["episode_key"]] = self.calls.get(episode["episode_key"], 0) + 1
            self.running[system] = self.running.get(system, 0) + 1
            self.peak[system] = max(self.peak[system] if system in self.peak else 0, self.running[system])
        time.sleep(self.hold_seconds)
        with self.lock:
            self.running[system] -= 1
        return 0


# ----------------------------------------------------------------- slot isolation
def test_slot_plan_isolates_six_workers_like_research1(tmp_path):
    plan = slot_plan(6, tmp_path)
    assert [entry["ros_domain_id"] for entry in plan] == [60, 61, 62, 63, 64, 65]
    assert [entry["gz_partition"] for entry in plan] == [f"research2_w{i}" for i in range(6)]
    assert len({entry["ros_log_dir"] for entry in plan}) == 6
    assert len({entry["gz_homedir"] for entry in plan}) == 6
    assert SEQUENTIAL_DOMAIN not in {entry["ros_domain_id"] for entry in plan}

    base = {"ROS_DOMAIN_ID": "52", "RESEARCH2_ROS_DOMAIN_ID": "52", "GZ_PARTITION": "research2",
            "ROS_LOG_DIR": str(tmp_path / "logs/ros"), "PATH": "/usr/bin"}
    environment = slot_environment(3, 6, tmp_path, base_environment=base)
    assert environment["ROS_DOMAIN_ID"] == environment["RESEARCH2_ROS_DOMAIN_ID"] == "63"
    assert environment["GZ_PARTITION"] == "research2_w3"
    assert environment["ROS_LOG_DIR"].endswith("parallel/research2_w3")
    assert environment["GZ_HOMEDIR"].endswith("parallel/research2_w3")
    assert environment["RESEARCH2_WORKER_SLOT"] == "3"
    assert environment["RESEARCH2_CONCURRENCY_WORKERS"] == "6"
    assert environment["PATH"] == "/usr/bin"
    assert base["ROS_DOMAIN_ID"] == "52"  # the caller's environment is untouched
    assert worker_slot_findings(environment) == []
    assert worker_slot_provenance(environment) == {"worker_slot": 3, "concurrency_workers": 6}


def test_slot_plan_never_lands_on_the_sequential_domain(tmp_path):
    with pytest.raises(ValueError, match="domain 52"):
        slot_plan(6, tmp_path, domain_base=50)
    with pytest.raises(ValueError, match="domain range"):
        slot_plan(6, tmp_path, domain_base=230)


def test_sequential_episode_path_is_unaffected_by_slot_helpers():
    sequential = {"ROS_DOMAIN_ID": "52", "RESEARCH2_ROS_DOMAIN_ID": "52", "GZ_PARTITION": "research2"}
    assert worker_slot_findings(sequential) == []
    assert worker_slot_provenance(sequential) == {}
    assert worker_slot_findings({}) == []


def test_worker_slot_environment_fails_closed_when_inconsistent():
    on_sequential_domain = {
        "RESEARCH2_WORKER_SLOT": "0", "RESEARCH2_CONCURRENCY_WORKERS": "6",
        "ROS_DOMAIN_ID": "52", "RESEARCH2_ROS_DOMAIN_ID": "52", "GZ_PARTITION": "research2_w0",
    }
    findings = worker_slot_findings(on_sequential_domain)
    assert any("sequential ROS domain 52" in item for item in findings)
    wrong_partition = {**on_sequential_domain, "ROS_DOMAIN_ID": "60", "RESEARCH2_ROS_DOMAIN_ID": "60",
                       "GZ_PARTITION": "research2"}
    assert worker_slot_findings(wrong_partition) == ["GZ_PARTITION must be research2_w0 for worker slot 0"]
    bag_recorder_would_resource_domain_52 = {**wrong_partition, "GZ_PARTITION": "research2_w0",
                                             "RESEARCH2_ROS_DOMAIN_ID": "52"}
    assert any("must agree" in item for item in worker_slot_findings(bag_recorder_would_resource_domain_52))


# ------------------------------------------------------------ exact-once claiming
def test_dispatcher_attempts_every_pending_key_exactly_once(tmp_path):
    episodes = fake_episodes(30)
    ledger = tmp_path / "logs/campaigns/fake.jsonl"
    ledger.parent.mkdir(parents=True)
    with ledger.open("w", encoding="utf-8") as stream:
        for episode in episodes[:4]:
            stream.write(json.dumps({"campaign_id": "fake", "episode_key": episode["episode_key"],
                                     "returncode": 0}) + "\n")
    calls: dict[str, int] = {}
    calls_lock = threading.Lock()
    rng = random.Random(7)

    def run(episode, slot):
        with calls_lock:
            calls[episode["episode_key"]] = calls.get(episode["episode_key"], 0) + 1
        time.sleep(rng.random() * 0.005)
        return 0

    finalized: list[tuple[int, int]] = []
    dispatcher = make_dispatcher(tmp_path, episodes, workers=6, run=run, wave_size=12,
                                 finalize=lambda wave, completed: finalized.append((wave, completed)))
    result = dispatcher.run()
    rows = read_ledger(ledger)
    assert result.recorded == 26 and result.claimed == 26 and result.invalid == 0
    assert result.returncode == 0 and result.stop_reason == "queue_exhausted"
    assert len(rows) == 30
    assert sorted(row["episode_key"] for row in rows) == sorted(e["episode_key"] for e in episodes)
    assert all(count == 1 for count in calls.values()) and len(calls) == 26
    assert set(calls) == {e["episode_key"] for e in episodes[4:]}
    for row in rows[4:]:
        assert row["concurrency_workers"] == 6
        assert 0 <= row["worker_slot"] < 6
        assert row["ros_domain_id"] == 60 + row["worker_slot"]
        assert row["gz_partition"] == f"research2_w{row['worker_slot']}"
        assert row["dispatcher"] == DISPATCHER_ID
        assert row["returncode"] == 0 and row["artifact_validation_returncode"] == 0
        assert {"started_utc", "finished_utc", "episode_returncode"} <= set(row)
    assert len({row["worker_slot"] for row in rows[4:]}) > 1
    assert finalized == [(1, 12), (2, 24)]
    assert result.waves_finalized == [1, 2]


def test_invalid_episode_stops_new_claims_but_records_running_ones(tmp_path):
    episodes = fake_episodes(40)

    def run(episode, slot):
        if episode["episode_key"] == "k03":
            time.sleep(0.01)
            return 1
        time.sleep(0.05)
        return 0

    finalized: list[int] = []
    dispatcher = make_dispatcher(tmp_path, episodes, workers=3, run=run, wave_size=3,
                                 finalize=lambda wave, completed: finalized.append(wave))
    result = dispatcher.run()
    rows = read_ledger(tmp_path / "logs/campaigns/fake.jsonl")
    assert result.invalid == 1 and result.returncode == 1
    # a wave slice containing the invalid row is never reported as a clean wave
    invalid_wave = next(index for index, row in enumerate(rows) if row["returncode"]) // 3 + 1
    assert invalid_wave not in finalized and invalid_wave in result.waves_skipped_invalid
    assert "invalid episode k03" in result.stop_reason
    assert len(rows) == result.claimed == result.recorded
    assert 3 <= len(rows) <= 6  # the three in flight at the stop finished and were recorded
    assert len({row["episode_key"] for row in rows}) == len(rows)
    invalid_rows = [row for row in rows if row["returncode"]]
    assert [row["episode_key"] for row in invalid_rows] == ["k03"]
    assert invalid_rows[0]["artifact_validation_returncode"] is None


def test_failed_artifact_validation_counts_as_invalid(tmp_path):
    episodes = fake_episodes(6)
    dispatcher = make_dispatcher(
        tmp_path, episodes, workers=2, run=lambda episode, slot: 0,
        validate=lambda episode: 3 if episode["episode_key"] == "k01" else 0,
    )
    result = dispatcher.run()
    rows = {row["episode_key"]: row for row in read_ledger(tmp_path / "logs/campaigns/fake.jsonl")}
    assert result.invalid == 1
    assert rows["k01"]["returncode"] == 3 and rows["k01"]["episode_returncode"] == 0
    assert rows["k01"]["artifact_validation_returncode"] == 3


def test_free_space_reserve_is_checked_before_every_claim(tmp_path):
    episodes = fake_episodes(6)
    dispatcher = make_dispatcher(
        tmp_path, episodes, workers=2, run=lambda episode, slot: 0,
        disk_free=lambda: 90 * 2**30, minimum_free=100 * 2**30,
    )
    result = dispatcher.run()
    assert result.claimed == 0 and result.recorded == 0
    assert "free-space reserve reached" in result.stop_reason
    assert result.returncode == 2
    assert read_ledger(tmp_path / "logs/campaigns/fake.jsonl") == []


def test_claim_limit_and_duplicate_record_refusal(tmp_path):
    episodes = fake_episodes(10)
    dispatcher = make_dispatcher(tmp_path, episodes, workers=4, run=lambda episode, slot: 0,
                                 max_claims=5)
    result = dispatcher.run()
    assert result.claimed == result.recorded == 5
    assert result.stop_reason == "claim_limit_reached" and result.returncode == 0
    with pytest.raises(RuntimeError, match="already contains k00"):
        dispatcher.record(episodes[0], 0, {"returncode": 0})


# ---------------------------------------------------------- per-system caps
def test_system_cap_is_never_exceeded_while_the_queue_still_drains_exactly_once(tmp_path):
    episodes = mixed_system_episodes(36)  # 12 s3 episodes interleaved with 24 s0
    probe = ConcurrencyProbe(hold_seconds=0.01)
    finalized: list[tuple[int, int]] = []
    dispatcher = make_dispatcher(
        tmp_path, episodes, workers=6, run=probe, wave_size=12,
        finalize=lambda wave, completed: finalized.append((wave, completed)),
        system_concurrency={"s3": 1},
    )
    result = dispatcher.run()
    rows = read_ledger(tmp_path / "logs/campaigns/fake.jsonl")
    assert result.claimed == result.recorded == 36 and result.invalid == 0
    assert result.stop_reason == "queue_exhausted" and result.returncode == 0
    assert sorted(row["episode_key"] for row in rows) == sorted(e["episode_key"] for e in episodes)
    assert all(count == 1 for count in probe.calls.values()) and len(probe.calls) == 36
    assert probe.peak["s3"] == 1  # the cap held in the runner itself, not only in bookkeeping
    assert probe.peak["s0"] > 1  # uncapped systems still ran concurrently
    assert dispatcher.peak_in_flight_by_system == {"s0": probe.peak["s0"], "s3": 1}
    for row in rows:
        assert row["system_concurrency"] == {"s3": 1}
        assert row["system"] in {"s0", "s3"}
        assert row["concurrency_workers"] == 6 and row["dispatcher"] == DISPATCHER_ID
    assert finalized == [(1, 12), (2, 24), (3, 36)]


def test_capped_slots_wait_for_the_running_episode_instead_of_exiting(tmp_path):
    episodes = mixed_system_episodes(5, systems=("s3",))  # nothing but capped work
    probe = ConcurrencyProbe(hold_seconds=0.02)
    dispatcher = make_dispatcher(tmp_path, episodes, workers=3, run=probe,
                                 system_concurrency={"s3": 1})
    result = dispatcher.run()
    rows = read_ledger(tmp_path / "logs/campaigns/fake.jsonl")
    assert result.recorded == 5 and result.stop_reason == "queue_exhausted"
    assert [row["episode_key"] for row in rows] == [e["episode_key"] for e in episodes]
    assert probe.peak == {"s3": 1}
    # every finish (record) can be followed by at most one start; the ledger is serial
    for earlier, later in zip(rows, rows[1:]):
        assert earlier["finished_utc"] <= later["started_utc"] or earlier["started_utc"] <= later["started_utc"]


def test_claim_reports_cap_blocked_rather_than_queue_exhausted(tmp_path):
    episodes = mixed_system_episodes(3, systems=("s3",))
    dispatcher = make_dispatcher(tmp_path, episodes, workers=2, run=lambda e, s: 0,
                                 system_concurrency={"s3": 1})
    first = dispatcher.claim()
    assert first["episode_key"] == "k00"
    assert dispatcher.claim() == CAP_BLOCKED  # k01 and k02 exist but the cap is occupied
    assert dispatcher.claim() == CAP_BLOCKED
    assert dispatcher._result.stop_reason is None  # waiting, not stopping
    dispatcher.record(first, 0, {"returncode": 0})
    second = dispatcher.claim()
    assert second["episode_key"] == "k01"
    row = read_ledger(tmp_path / "logs/campaigns/fake.jsonl")[0]
    assert row["system"] == "s3" and row["system_concurrency"] == {"s3": 1}


def test_cap_blocked_slots_still_honour_stop_on_invalid(tmp_path):
    episodes = mixed_system_episodes(6, systems=("s3",))

    def run(episode, slot):
        time.sleep(0.01)
        return 1 if episode["episode_key"] == "k00" else 0

    dispatcher = make_dispatcher(tmp_path, episodes, workers=3, run=run,
                                 system_concurrency={"s3": 1})
    result = dispatcher.run()
    rows = read_ledger(tmp_path / "logs/campaigns/fake.jsonl")
    assert result.invalid == 1 and len(rows) == 1 and rows[0]["episode_key"] == "k00"
    assert "invalid episode k00" in result.stop_reason


def test_dispatcher_rejects_malformed_caps(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        make_dispatcher(tmp_path, fake_episodes(2), workers=2, run=lambda e, s: 0,
                        system_concurrency={"s3": 0})
    with pytest.raises(ValueError, match="s0..s9"):
        make_dispatcher(tmp_path, fake_episodes(2), workers=2, run=lambda e, s: 0,
                        system_concurrency={"gpu": 1})
    # an episode without a system is never capped
    dispatcher = make_dispatcher(tmp_path, fake_episodes(2), workers=2, run=lambda e, s: 0,
                                 system_concurrency={"s3": 1})
    assert dispatcher.run().recorded == 2


def test_cap_flag_parsing_and_resolution():
    assert parse_system_concurrency_flags(None) == {}
    assert parse_system_concurrency_flags(["s3=1", "S0=2", "s3=1"]) == {"s3": 1, "s0": 2}
    for bad in (["s3"], ["=1"], ["s3=one"], ["s3=0"], ["gpu=1"]):
        with pytest.raises(ValueError):
            parse_system_concurrency_flags(bad)
    with pytest.raises(ValueError, match="twice"):
        parse_system_concurrency_flags(["s3=1", "s3=2"])

    # manifest defaults; the CLI may add or tighten, never loosen
    assert resolve_system_concurrency(manifest_caps={"s3": 1}, cli_caps=None) == {"s3": 1}
    assert resolve_system_concurrency(manifest_caps={"s3": 2}, cli_caps={"s3": 1}) == {"s3": 1}
    assert resolve_system_concurrency(manifest_caps={"s3": 1}, cli_caps={"s0": 3}) == {"s0": 3, "s3": 1}
    with pytest.raises(ValueError, match="may only tighten"):
        resolve_system_concurrency(manifest_caps={"s3": 1}, cli_caps={"s3": 2})
    # an admitting amendment is a floor of restrictions
    assert resolve_system_concurrency(manifest_caps=None, cli_caps=None, amendment_caps={"s3": 1}) == {"s3": 1}
    assert resolve_system_concurrency(manifest_caps={"s3": 3}, cli_caps=None, amendment_caps={"s3": 1}) == {"s3": 1}
    assert resolve_system_concurrency(manifest_caps={"s3": 1}, cli_caps=None, amendment_caps={"s3": 3}) == {"s3": 1}
    with pytest.raises(ValueError, match="may only tighten"):
        resolve_system_concurrency(manifest_caps=None, cli_caps={"s3": 2}, amendment_caps={"s3": 1})


def test_initial_claims_respect_caps():
    pending = mixed_system_episodes(6, systems=("s3",))
    assert initial_claims_with_caps(pending, 3, {"s3": 1}) == {
        "slot0": ["k00"], "slot1": [], "slot2": [],
    }
    pending = mixed_system_episodes(6, systems=("s3", "s0"))
    assert initial_claims_with_caps(pending, 3, {"s3": 1}) == {
        "slot0": ["k00"], "slot1": ["k01"], "slot2": ["k03"],
    }
    assert initial_claims_with_caps(pending, 2, {}) == {"slot0": ["k00"], "slot1": ["k01"]}


def test_amendment_system_concurrency_applies_only_to_admitted_sequential_manifests():
    amendment = {"amendment_id": "PA-2026-09-10-01", "status": "approved", "decisions": {
        "concurrency": {"admitted_workers": 6, "system_concurrency": {"s3": 1},
                        "campaigns_admitted_if_check_passes": ["targeted_development_v1"]},
    }}
    amendments = {"PA-2026-09-10-01": amendment}
    sequential = {**yaml.safe_load(TARGETED.read_text(encoding="utf-8")),
                  "parallel_execution_admitted_by": "PA-2026-09-10-01"}
    assert amendment_system_concurrency(sequential, amendments, override=True) == {"s3": 1}
    assert amendment_system_concurrency(sequential, amendments, override=False) == {}
    parallel = yaml.safe_load(SHIFT_CHECK_V2.read_text(encoding="utf-8"))
    assert amendment_system_concurrency(parallel, amendments, override=True) == {}
    # a malformed amendment block is an admission finding
    broken = {**amendment, "decisions": {"concurrency": {
        **amendment["decisions"]["concurrency"], "system_concurrency": {"s3": "one"},
    }}}
    findings = validate_parallel_execution_admission(sequential, {"PA-2026-09-10-01": broken}, workers=6)
    assert any("system_concurrency" in item and "positive integer" in item for item in findings)


def test_admission_condition_may_name_the_v2_check_report():
    sequential = {**yaml.safe_load(TARGETED.read_text(encoding="utf-8")),
                  "parallel_execution_admitted_by": "PA-2026-09-10-02"}
    v2_report = "reports/integrity/concurrency_shift_check_v2.yaml"
    amendment = {"amendment_id": "PA-2026-09-10-02", "status": "approved", "decisions": {
        "concurrency": {
            "admitted_workers": 6, "system_concurrency": {"s3": 1},
            "admission_condition": f"{v2_report} passed true",
            "campaigns_admitted_if_check_passes": ["targeted_development_v1"],
        },
    }}
    amendments = {"PA-2026-09-10-02": amendment}
    assert validate_parallel_execution_admission(
        sequential, amendments, workers=6, check_reports={v2_report: {"passed": True}},
    ) == []
    v1_only = {"reports/integrity/concurrency_shift_check_v1.yaml": {"passed": True}}
    assert validate_parallel_execution_admission(sequential, amendments, workers=6, check_reports=v1_only) == [
        f"admission condition of PA-2026-09-10-02 is not met: {v2_report} passed true"
    ]
    assert validate_parallel_execution_admission(
        sequential, amendments, workers=6, check_reports={v2_report: {"passed": False}},
    )


def test_dispatcher_dry_run_reads_v2_caps_and_refuses_loosening():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_campaign_parallel.py"),
         "--manifest", str(SHIFT_CHECK_V2), "--workers", "6", "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["workers"] == 6 and report["total"] == 36
    assert report["system_concurrency"] == {"s3": 1}
    assert report["system_concurrency_sources"]["manifest"] == {"s3": 1}
    assert report["system_concurrency_sources"]["cli"] == {}
    loosen = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_campaign_parallel.py"),
         "--manifest", str(SHIFT_CHECK_V2), "--dry-run", "--max-system-concurrency", "s3=2"],
        capture_output=True, text=True, check=False,
    )
    assert loosen.returncode != 0 and "may only tighten" in loosen.stderr
    tighten = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_campaign_parallel.py"),
         "--manifest", str(SHIFT_CHECK), "--dry-run", "--max-system-concurrency", "s3=1",
         "--max-system-concurrency", "s0=2"],
        capture_output=True, text=True, check=False,
    )
    assert tighten.returncode == 0, tighten.stderr
    assert json.loads(tighten.stdout)["system_concurrency"] == {"s0": 2, "s3": 1}
    lock_path = ROOT / "logs/campaigns/concurrency_shift_check_v2.lock"
    assert not lock_path.exists() or not _held(lock_path)


# --------------------------------------------------------------------- refusals
def test_concurrency_admission_requires_override_and_amendment():
    sequential = yaml.safe_load(TARGETED.read_text(encoding="utf-8"))
    assert sequential["execution_policy"]["concurrency"] == 1
    workers, findings = concurrency_admission(sequential, None, override=False, amendments={})
    assert workers == 1 and any("--concurrency-override" in item for item in findings)
    workers, findings = concurrency_admission(sequential, 6, override=True, amendments={})
    assert any("parallel_execution_admitted_by" in item for item in findings)

    admitted = {**sequential, "parallel_execution_admitted_by": "PA-2026-09-10-01"}
    report_path = "reports/integrity/concurrency_shift_check_v1.yaml"
    amendment = {"amendment_id": "PA-2026-09-10-01", "status": "approved", "decisions": {
        "concurrency": {
            "admitted_workers": 6, "ros_domains": [60, 61, 62, 63, 64, 65],
            "admission_condition": f"{report_path} passed true",
            "campaigns_admitted_if_check_passes": ["targeted_development_v1"],
        },
    }}
    passed = {report_path: {"passed": True}}
    workers, findings = concurrency_admission(
        admitted, 6, override=True, amendments={"PA-2026-09-10-01": amendment}, check_reports=passed,
    )
    assert workers == 6 and findings == []
    _, findings = concurrency_admission(
        admitted, 6, override=True, amendments={"PA-2026-09-10-01": amendment}, check_reports={},
    )
    assert findings == [f"admission condition of PA-2026-09-10-01 is not met: {report_path} passed true"]
    _, findings = concurrency_admission(
        admitted, 6, override=True, amendments={"PA-2026-09-10-01": amendment},
        check_reports={report_path: {"passed": False}},
    )
    assert any("is not met" in item for item in findings)
    _, findings = concurrency_admission(
        admitted, 7, override=True, amendments={"PA-2026-09-10-01": amendment}, check_reports=passed,
    )
    assert any("exceeds the 6 admitted" in item for item in findings)
    _, findings = concurrency_admission(admitted, 6, override=True, amendments={
        "PA-2026-09-10-01": {**amendment, "status": "draft"},
    }, check_reports=passed)
    assert any("not approved" in item for item in findings)
    _, findings = concurrency_admission(admitted, 6, override=True, amendments={
        "PA-2026-09-10-01": {**amendment, "decisions": {"parallel_execution_admitted_campaigns": []}},
    })
    assert any("does not admit" in item for item in findings)
    flat = {**amendment, "decisions": {"parallel_execution_admitted_campaigns": ["targeted_development_v1"]}}
    _, findings = concurrency_admission(admitted, 6, override=True, amendments={"PA-2026-09-10-01": flat})
    assert findings == []

    parallel = yaml.safe_load(SHIFT_CHECK.read_text(encoding="utf-8"))
    workers, findings = concurrency_admission(parallel, None, override=False, amendments={})
    assert workers == 6 and findings == []
    workers, findings = concurrency_admission(parallel, 3, override=False, amendments={})
    assert workers == 3 and findings == []
    _, findings = concurrency_admission(parallel, 8, override=False, amendments={})
    assert any("exceeds the preregistered concurrency" in item for item in findings)


def test_real_amendment_admits_supplement_only_after_the_check_report_passes():
    amendments = load_amendments(ROOT)
    assert "PA-2026-09-03-04" in amendments
    supplement = yaml.safe_load(
        (ROOT / "data/manifests/development_supplement_v1.yaml").read_text(encoding="utf-8")
    )
    admitted = {**supplement, "parallel_execution_admitted_by": "PA-2026-09-03-04"}
    report_path = "reports/integrity/concurrency_shift_check_v1.yaml"
    workers, findings = concurrency_admission(
        admitted, None, override=True, amendments=amendments,
        check_reports={report_path: {"passed": True}},
    )
    assert workers == 1  # a sequential manifest keeps concurrency 1 unless --workers is given
    workers, findings = concurrency_admission(
        admitted, 6, override=True, amendments=amendments,
        check_reports={report_path: {"passed": True}},
    )
    assert workers == 6 and findings == []
    _, findings = concurrency_admission(
        admitted, 6, override=True, amendments=amendments, check_reports={},
    )
    assert findings == [f"admission condition of PA-2026-09-03-04 is not met: {report_path} passed true"]
    targeted = {**yaml.safe_load(TARGETED.read_text(encoding="utf-8")),
                "parallel_execution_admitted_by": "PA-2026-09-03-04"}
    _, findings = concurrency_admission(
        targeted, 6, override=True, amendments=amendments,
        check_reports={report_path: {"passed": True}},
    )
    assert any("does not admit 'targeted_development_v1'" in item for item in findings)
    # The check itself is never gated on its own report.
    check = yaml.safe_load(SHIFT_CHECK.read_text(encoding="utf-8"))
    assert concurrency_admission(check, None, override=False, amendments=amendments) == (6, [])


def test_parallel_admission_field_format():
    assert validate_parallel_execution_admission({}) == [
        "parallel_execution_admitted_by must name one protocol amendment id (PA-YYYY-MM-DD-NN)"
    ]
    assert validate_parallel_execution_admission({"parallel_execution_admitted_by": "amendment 3"})
    assert validate_parallel_execution_admission({"parallel_execution_admitted_by": "PA-2026-09-10-01"}) == []
    findings = validate_parallel_execution_admission(
        {"parallel_execution_admitted_by": "PA-2026-09-10-01", "campaign_id": "x"}, amendments={},
    )
    assert findings == ["protocol amendment PA-2026-09-10-01 is not on file under configs/"]


def test_other_runner_locks_detects_any_held_campaign_lock(tmp_path):
    lock_root = tmp_path / "logs/campaigns"
    lock_root.mkdir(parents=True)
    held = lock_root / "targeted_development_v1.lock"
    (lock_root / "idle.lock").touch()
    with held.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert other_runner_locks(tmp_path, "concurrency_shift_check_v1") == [
            "logs/campaigns/targeted_development_v1.lock"
        ]
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    assert other_runner_locks(tmp_path, "concurrency_shift_check_v1") == []


def test_dispatcher_refuses_sequential_manifest_without_override(capsys):
    with pytest.raises(SystemExit) as failure:
        dispatcher_main(["--manifest", str(TARGETED), "--dry-run"])
    assert "parallel execution is not admitted" in str(failure.value)
    assert "--concurrency-override" in str(failure.value)
    with pytest.raises(SystemExit) as failure:
        dispatcher_main(["--manifest", str(TARGETED), "--dry-run", "--concurrency-override"])
    assert "parallel_execution_admitted_by" in str(failure.value)


def test_dispatcher_dry_run_prints_six_slot_plan_without_locking():
    lock_path = ROOT / "logs/campaigns/concurrency_shift_check_v1.lock"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_campaign_parallel.py"),
         "--manifest", str(SHIFT_CHECK), "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["workers"] == 6 and report["total"] == 36 and report["wave_size"] == 12
    assert [entry["ros_domain_id"] for entry in report["slot_plan"]] == [60, 61, 62, 63, 64, 65]
    assert [entry["gz_partition"] for entry in report["slot_plan"]] == [f"research2_w{i}" for i in range(6)]
    assert report["protected_test_used"] is False
    assert len(report["first_claims_if_all_slots_start_together"]) == 6
    assert not lock_path.exists() or not _held(lock_path)


def _held(path: Path) -> bool:
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return False


def test_ordered_episodes_matches_sequential_ordering():
    document = yaml.safe_load(TARGETED.read_text(encoding="utf-8"))
    assert [item["episode_key"] for item in ordered_episodes(document)] == [
        item["episode_key"] for item in targeted_execution_order(expand_balanced_pilot(document))
    ]


# ------------------------------------------------ sequential runner unchanged
def test_sequential_runner_is_not_coupled_to_the_dispatcher():
    source = (ROOT / "scripts/run_balanced_pilot.py").read_text(encoding="utf-8")
    assert "worker_slots" not in source and "run_campaign_parallel" not in source
    assert "concurrency" not in source
    episode_runner = (ROOT / "scripts/run_research2_episode.py").read_text(encoding="utf-8")
    # The only slot-aware lines are guarded by the slot variable and inert without it.
    assert episode_runner.count("worker_slot_findings(os.environ)") == 1
    assert episode_runner.count("**worker_slot_provenance(os.environ)") == 1


def test_sequential_dry_run_output_matches_manifest_and_ledger_exactly():
    """run_balanced_pilot.py --dry-run on the targeted manifest is unchanged in form and values."""
    document = yaml.safe_load(TARGETED.read_text(encoding="utf-8"))
    ledger = ROOT / "logs/campaigns" / f"{document['campaign_id']}.jsonl"
    for _attempt in range(3):
        before = attempted_keys(ledger)
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/run_balanced_pilot.py"),
             "--manifest", str(TARGETED), "--dry-run"],
            capture_output=True, text=True, check=False,
        )
        after = attempted_keys(ledger)
        if before == after:
            break
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    episodes = targeted_execution_order(expand_balanced_pilot(document))
    pending = [item for item in episodes if item["episode_key"] not in before]
    wave = int(document["execution_policy"]["maximum_episodes_per_invocation"])
    remainder = len(before) % wave
    limit = min(wave, wave - remainder if remainder else wave)
    family_counts: dict[str, int] = {}
    for item in episodes:
        family_counts[item["family"]] = family_counts.get(item["family"], 0) + 1
    assert report == {
        "attempted": len(before),
        "campaign_id": "targeted_development_v1",
        "family_counts": family_counts,
        "pending": len(pending),
        "protected_test_used": False,
        "scheduled_this_invocation": len(pending[:limit]),
        "total": 1212,
    }
    assert result.stdout == json.dumps(report, indent=2, sort_keys=True) + "\n"


# ------------------------------------------------------------ phased execution
def phased_amendment(amendment_id: str = "PA-2026-09-10-03", **overrides) -> dict:
    concurrency = {
        "admitted_workers": 6, "ros_domains": [60, 61, 62, 63, 64, 65],
        "system_concurrency": {"s3": 1}, "phased_execution": {"s0": 6, "s3": 1},
        "campaigns_admitted_if_check_passes": ["targeted_development_v1"],
        **overrides,
    }
    return {"amendment_id": amendment_id, "status": "approved",
            "decisions": {"concurrency": concurrency}}


def test_systems_flag_parsing():
    assert parse_systems_flag(None) is None
    assert parse_systems_flag("s0") == ["s0"]
    assert parse_systems_flag("S3, s0,s0") == ["s0", "s3"]
    for bad in ("", " , ", "gpu", "s0,perception"):
        with pytest.raises(ValueError):
            parse_systems_flag(bad)


def test_phase_a_claims_only_s0_and_leaves_s3_pending_untouched(tmp_path):
    episodes = mixed_system_episodes(36)  # 24 s0, 12 s3 interleaved in design order
    probe = ConcurrencyProbe(hold_seconds=0.005)
    finalized: list[tuple[int, int]] = []
    dispatcher = make_dispatcher(
        tmp_path, episodes, workers=6, run=probe, wave_size=12, systems=["s0"],
        finalize=lambda wave, completed: finalized.append((wave, completed)),
        system_concurrency={"s3": 1},
    )
    result = dispatcher.run()
    rows = read_ledger(tmp_path / "logs/campaigns/fake.jsonl")
    assert result.claimed == result.recorded == 24 and result.invalid == 0
    assert result.stop_reason == SYSTEMS_DRAINED and result.returncode == 0
    assert result.systems == ["s0"]
    assert {row["system"] for row in rows} == {"s0"}
    assert set(probe.calls) == {e["episode_key"] for e in episodes if e["system"] == "s0"}
    assert all(count == 1 for count in probe.calls.values())
    assert "s3" not in probe.peak and probe.peak["s0"] > 1
    # untouched S3 work is reported so phase B can be started, never marked attempted
    assert result.remaining_by_system == {"s0": 0, "s3": 12} and result.remaining == 12
    assert attempted_keys(tmp_path / "logs/campaigns/fake.jsonl") == set(probe.calls)
    for row in rows:
        assert row["systems_filter"] == ["s0"] and row["concurrency_workers"] == 6
    # ledger-order wave slices: two complete S0-only waves, produced in ledger order
    assert finalized == [(1, 12), (2, 24)] and result.waves_finalized == [1, 2]
    assert result.waves_complete_at_start == []


def test_phase_b_runs_remaining_s3_strictly_sequentially_in_slot_zero(tmp_path):
    episodes = mixed_system_episodes(36)
    make_dispatcher(tmp_path, episodes, workers=6, run=ConcurrencyProbe(0.001), wave_size=12,
                    systems=["s0"]).run()
    probe = ConcurrencyProbe(hold_seconds=0.01)
    finalized: list[tuple[int, int]] = []
    phase_b = make_dispatcher(
        tmp_path, episodes, workers=1, run=probe, wave_size=12, systems=["s3"],
        finalize=lambda wave, completed: finalized.append((wave, completed)),
        system_concurrency={"s3": 1},
    )
    result = phase_b.run()
    rows = read_ledger(tmp_path / "logs/campaigns/fake.jsonl")
    assert result.claimed == result.recorded == 12 and result.invalid == 0
    assert result.stop_reason == "queue_exhausted"  # nothing is left outside the phase
    assert result.remaining_by_system == {"s0": 0, "s3": 0} and result.remaining == 0
    assert len(rows) == 36 and len({row["episode_key"] for row in rows}) == 36
    phase_b_rows = rows[24:]
    assert [row["episode_key"] for row in phase_b_rows] == [
        e["episode_key"] for e in episodes if e["system"] == "s3"
    ]
    assert probe.peak == {"s3": 1} and phase_b.peak_in_flight_by_system == {"s3": 1}
    for row in phase_b_rows:
        # sequential-runner isolation semantics: one slot, domain 60, nothing else
        assert row["concurrency_workers"] == 1 and row["worker_slot"] == 0
        assert row["ros_domain_id"] == 60 and row["gz_partition"] == "research2_w0"
        assert row["systems_filter"] == ["s3"] and row["system"] == "s3"
    for earlier, later in zip(phase_b_rows, phase_b_rows[1:]):
        assert earlier["finished_utc"] <= later["started_utc"]
    # waves 1-2 were complete before phase B; wave 3 is the S3-only slice reported now
    assert result.waves_complete_at_start == [1, 2]
    assert finalized[-1] == (3, 36) and result.waves_finalized[-1] == 3


def test_partial_wave_left_by_phase_a_is_completed_and_reported_by_phase_b(tmp_path):
    episodes = mixed_system_episodes(30)  # 20 s0 + 10 s3: phase A leaves 8 rows past wave 1
    finalized_a: list[tuple[int, int]] = []
    result_a = make_dispatcher(
        tmp_path, episodes, workers=4, run=lambda e, s: 0, wave_size=12, systems=["s0"],
        finalize=lambda wave, completed: finalized_a.append((wave, completed)),
    ).run()
    assert result_a.recorded == 20 and finalized_a == [(1, 12)]
    assert result_a.stop_reason == SYSTEMS_DRAINED
    assert result_a.remaining_by_system == {"s0": 0, "s3": 10}
    finalized_b: list[tuple[int, int]] = []
    result_b = make_dispatcher(
        tmp_path, episodes, workers=1, run=lambda e, s: 0, wave_size=12, systems=["s3"],
        finalize=lambda wave, completed: finalized_b.append((wave, completed)),
    ).run()
    rows = read_ledger(tmp_path / "logs/campaigns/fake.jsonl")
    assert result_b.recorded == 10 and len(rows) == 30
    # wave 2 is a mixed ledger-order slice (8 s0 rows from phase A + 4 s3 rows from phase B)
    wave_two = rows[12:24]
    assert [row["system"] for row in wave_two] == ["s0"] * 8 + ["s3"] * 4
    assert (2, 24) in finalized_b and result_b.waves_complete_at_start == [1]
    assert 3 not in result_b.waves_finalized  # 6 trailing rows: no incomplete slice is reported
    assert result_b.remaining_by_system == {"s0": 0, "s3": 0}


def test_systems_filter_and_pending_by_system_helpers(tmp_path):
    episodes = mixed_system_episodes(6)
    assert pending_by_system(episodes, set()) == {"s0": 4, "s3": 2}
    assert pending_by_system(episodes, {"k00", "k01"}) == {"s0": 3, "s3": 1}
    with pytest.raises(ValueError, match="at least one system"):
        make_dispatcher(tmp_path, episodes, workers=2, run=lambda e, s: 0, systems=[])
    # an invalid episode inside the phase still stops the phase and leaves the rest pending
    dispatcher = make_dispatcher(tmp_path, episodes, workers=1, systems=["s0"],
                                 run=lambda e, s: 1 if e["episode_key"] == "k00" else 0)
    result = dispatcher.run()
    assert result.invalid == 1 and result.recorded == 1
    assert result.remaining_by_system == {"s0": 3, "s3": 2}


def test_phased_execution_admission_terms():
    sequential = {**yaml.safe_load(TARGETED.read_text(encoding="utf-8")),
                  "parallel_execution_admitted_by": "PA-2026-09-10-03"}
    amendments = {"PA-2026-09-10-03": phased_amendment()}
    # an unrestricted run is untouched by the phased block
    assert validate_parallel_execution_admission(sequential, amendments, workers=6) == []
    # phase A: S0 six wide; phase B: S3 strictly one at a time
    assert validate_parallel_execution_admission(sequential, amendments, workers=6, systems=["s0"]) == []
    assert validate_parallel_execution_admission(sequential, amendments, workers=1, systems=["s3"]) == []
    findings = validate_parallel_execution_admission(sequential, amendments, workers=2, systems=["s3"])
    assert findings == ["requested 2 workers exceeds the phase cap 1 for s3 in PA-2026-09-10-03"]
    findings = validate_parallel_execution_admission(sequential, amendments, workers=6, systems=["s0", "s3"])
    assert findings == ["requested 6 workers exceeds the phase cap 1 for s3 in PA-2026-09-10-03"]
    findings = validate_parallel_execution_admission(sequential, amendments, workers=1, systems=["s1"])
    assert findings == ["PA-2026-09-10-03 decisions.concurrency.phased_execution declares no phase for s1"]
    # without a phased block a restricted run is not admitted (fail closed)
    unphased = {"PA-2026-09-10-03": phased_amendment(phased_execution=None)}
    assert validate_parallel_execution_admission(sequential, unphased, workers=6) == []
    findings = validate_parallel_execution_admission(sequential, unphased, workers=1, systems=["s3"])
    assert findings == [
        "PA-2026-09-10-03 declares no decisions.concurrency.phased_execution; a run "
        "restricted to s3 is not admitted"
    ]
    # malformed or over-admitted phase caps are findings even for an unrestricted run
    malformed = {"PA-2026-09-10-03": phased_amendment(phased_execution={"s3": "one"})}
    findings = validate_parallel_execution_admission(sequential, malformed, workers=6)
    assert any("phased_execution" in item and "positive integer" in item for item in findings)
    too_wide = {"PA-2026-09-10-03": phased_amendment(phased_execution={"s0": 8, "s3": 1})}
    findings = validate_parallel_execution_admission(sequential, too_wide, workers=6)
    assert findings == [
        "PA-2026-09-10-03 decisions.concurrency.phased_execution: phase cap for s0 (8) "
        "exceeds the 6 admitted workers"
    ]
    # the dispatcher's admission wrapper passes the phase through
    workers, findings = concurrency_admission(sequential, 1, override=True, amendments=amendments,
                                              systems=["s3"])
    assert (workers, findings) == (1, [])
    _, findings = concurrency_admission(sequential, 6, override=True, amendments=amendments,
                                        systems=["s3"])
    assert any("phase cap 1 for s3" in item for item in findings)
    assert amendment_phased_execution(sequential, amendments, override=True) == {"s0": 6, "s3": 1}
    assert amendment_phased_execution(sequential, amendments, override=False) == {}
    parallel = yaml.safe_load(SHIFT_CHECK_V2.read_text(encoding="utf-8"))
    assert amendment_phased_execution(parallel, amendments, override=True) == {}


def test_real_amendments_declare_no_phased_execution_yet():
    """Until the researcher approves it, no on-file amendment admits a --systems phase."""
    amendments = load_amendments(ROOT)
    supplement = yaml.safe_load(
        (ROOT / "data/manifests/development_supplement_v1.yaml").read_text(encoding="utf-8")
    )
    admitted = {**supplement, "parallel_execution_admitted_by": "PA-2026-09-03-04"}
    report_path = "reports/integrity/concurrency_shift_check_v1.yaml"
    _, findings = concurrency_admission(
        admitted, 6, override=True, amendments=amendments,
        check_reports={report_path: {"passed": True}}, systems=["s0"],
    )
    assert findings == [
        "PA-2026-09-03-04 declares no decisions.concurrency.phased_execution; a run "
        "restricted to s0 is not admitted"
    ]


def test_dispatcher_dry_run_reports_phase_and_remaining_by_system():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_campaign_parallel.py"),
         "--manifest", str(SHIFT_CHECK_V2), "--workers", "1", "--systems", "s3", "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["workers"] == 1 and report["systems"] == ["s3"]
    assert report["wave_reporting"] == WAVE_REPORTING
    assert set(report["pending_by_system"]) == {"s0", "s3"}
    assert report["claimable_in_systems"] == report["pending_by_system"]["s3"]
    assert sum(report["pending_by_system"].values()) == report["pending"]
    assert [entry["ros_domain_id"] for entry in report["slot_plan"]] == [60]
    assert report["slot_plan"][0]["gz_partition"] == "research2_w0"
    claims = report["first_claims_if_all_slots_start_together"]
    assert list(claims) == ["slot0"]
    unknown = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_campaign_parallel.py"),
         "--manifest", str(SHIFT_CHECK_V2), "--systems", "s7", "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert unknown.returncode != 0 and "does not contain" in unknown.stderr
    malformed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_campaign_parallel.py"),
         "--manifest", str(SHIFT_CHECK_V2), "--systems", "gpu", "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert malformed.returncode != 0 and "s0..s9" in malformed.stderr
    lock_path = ROOT / "logs/campaigns/concurrency_shift_check_v2.lock"
    assert not lock_path.exists() or not _held(lock_path)


# ------------------------------------------------ recovery campaigns (pilot + paired)

RECOVERY_PILOT = ROOT / "data/manifests/recovery_pilot_v1.yaml"
PAIRED_RECOVERY = ROOT / "data/manifests/paired_recovery_v2.yaml"


def _splits():
    return yaml.safe_load((ROOT / "data/manifests/splits.template.yaml").read_text(encoding="utf-8"))


def test_recovery_pilot_manifest_expands_per_policy_with_recovery_arguments():
    from scripts.run_campaign_parallel import classify_manifest, manifest_findings, ordered_episodes
    from scripts.run_live_integrity_campaign import episode_command
    document = yaml.safe_load(RECOVERY_PILOT.read_text(encoding="utf-8"))
    kinds = classify_manifest(document)
    assert kinds["recovery_pilot"] and not kinds["confirmatory"] and not kinds["paired_recovery"]
    assert manifest_findings(document, _splits()) == []
    episodes = ordered_episodes(document)
    assert len(episodes) == 882 == len({e["episode_key"] for e in episodes})
    # every policy of one pairing cell runs consecutively and shares the seed
    first_cell = episodes[:7]
    assert len({e["pair_key"] for e in first_cell}) == 1 and len({e["seed"] for e in first_cell}) == 1
    assert [e["recovery_policy_id"] for e in first_cell][0] == "R0"
    r0, forced = first_cell[0], first_cell[1]
    assert "--recovery-policy" not in episode_command("c", {**r0, "recovery_policy_id": None}, ROOT)
    command = episode_command("c", forced, ROOT)
    assert command[command.index("--recovery-policy") + 1] == forced["recovery_policy_id"]
    assert "--recovery-live-execution" in command
    assert "--recovery-live-execution" not in episode_command("c", r0, ROOT)


def test_paired_recovery_manifest_is_protected_and_pair_complete():
    from scripts.run_campaign_parallel import classify_manifest, manifest_findings, ordered_episodes
    document = yaml.safe_load(PAIRED_RECOVERY.read_text(encoding="utf-8"))
    kinds = classify_manifest(document)
    assert kinds["paired_recovery"] and kinds["confirmatory"]
    assert manifest_findings(document, _splits()) == []
    episodes = ordered_episodes(document)
    assert len(episodes) == 1512 == len({e["episode_key"] for e in episodes})
    assert len({(e["seed"], e["recovery_policy_id"]) for e in episodes}) == 1512
    assert {e["recovery_policy_id"] for e in episodes} == {"R0", "R2", "R3"}
    r3 = next(e for e in episodes if e["recovery_policy_id"] == "R3")
    assert r3["recovery_selector_model"].endswith("r3_cost_sensitive_ridge_v1.json")


def test_campaign_episodes_is_identity_for_plain_manifests():
    from src.experiments import campaign_episodes
    document = yaml.safe_load(TARGETED.read_text(encoding="utf-8"))
    assert campaign_episodes(document) == expand_balanced_pilot(document)


def test_finaliser_failure_at_start_claims_nothing(tmp_path):
    """A finaliser that fails on already-complete waves must stop the run before any slot claims."""
    from scripts.run_campaign_parallel import ParallelDispatcher
    episodes = [{"episode_key": f"k{i}", "system": "s0", "seed": i} for i in range(4)]
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("".join(json.dumps({"episode_key": f"k{i}", "returncode": 0}) + "\n" for i in range(2)))
    started = []

    def run_episode(episode, slot):
        started.append(episode["episode_key"])
        return 0

    def finalize(wave, completed):
        raise RuntimeError("research log unreadable")

    dispatcher = ParallelDispatcher(
        campaign_id="c", episodes=episodes, ledger=ledger, workers=2, plan=[{}, {}],
        run_episode=run_episode, validate_artifact=lambda episode: 0, disk_free=lambda: 10**12,
        minimum_free_bytes=0, wave_size=2, finalize=finalize,
    )
    with pytest.raises(RuntimeError, match="research log unreadable"):
        dispatcher.run()
    assert started == []
    assert len([line for line in ledger.read_text().splitlines() if line.strip()]) == 2
