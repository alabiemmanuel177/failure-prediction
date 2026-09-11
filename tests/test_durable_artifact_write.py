import os

import pytest

from scripts.run_research2_episode import configure_worker_identity, durable_atomic_write


def test_durable_atomic_write_publishes_complete_payload(tmp_path):
    target = tmp_path / "summary.yaml"
    durable_atomic_write(target, "identity:\n  run_id: run\n")
    assert target.read_text(encoding="utf-8") == "identity:\n  run_id: run\n"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_durable_atomic_write_refuses_overwrite(tmp_path):
    target = tmp_path / "summary.yaml"
    target.write_text("original\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        durable_atomic_write(target, "replacement\n")
    assert target.read_text(encoding="utf-8") == "original\n"


def test_configure_worker_identity_overrides_legacy_global_sweep(monkeypatch):
    monkeypatch.setenv("RCN_WORKER_ID", "inherited-worker")
    worker_id = configure_worker_identity("01234567-89ab-cdef-0123-456789abcdef")
    assert worker_id == "research2-01234567-89ab-cdef-0123-456789abcdef"
    assert worker_id == os.environ["RCN_WORKER_ID"]
