import copy

import pytest

from scripts.research_log import append_record, read_records, verify


def test_hash_chained_log_detects_tampering(tmp_path):
    path = tmp_path / "research.jsonl"
    append_record(path, kind="decision", actor="test", message="first", metadata={})
    append_record(path, kind="review", actor="test", message="second", metadata={})
    records = read_records(path)
    verify(records)

    tampered = copy.deepcopy(records)
    tampered[0]["message"] = "changed"
    with pytest.raises(ValueError, match="record_hash is invalid"):
        verify(tampered)

