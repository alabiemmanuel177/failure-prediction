from scripts.validate_dataset_splits import assignments

import pytest


def test_assignment_validator_rejects_cross_split_map_or_route():
    document = {
        "development": {"maps": ["m"], "routes": ["r"]},
        "validation": {"maps": ["m"], "routes": []},
        "held_out_map_test": {"maps": [], "routes": []},
    }
    with pytest.raises(ValueError):
        assignments(document)


def test_assignment_validator_returns_exact_mapping():
    document = {
        "development": {"maps": ["d"], "routes": ["dr"]},
        "validation": {"maps": ["v"], "routes": ["vr"]},
        "held_out_map_test": {"maps": [], "routes": []},
    }
    maps, routes = assignments(document)
    assert maps == {"d": "development", "v": "validation"}
    assert routes == {"dr": "development", "vr": "validation"}
