from scripts.finalize_nonmodel_collection import collection_spec, report_is_complete


def test_collection_specs_keep_validation_and_targeted_artifacts_separate(tmp_path):
    validation = collection_spec("validation", tmp_path)
    targeted = collection_spec("targeted", tmp_path)
    assert validation.expected == 324
    assert targeted.expected == 1212
    assert validation.inventory != targeted.inventory
    assert validation.dataset_manifest != targeted.dataset_manifest


def test_finalizer_requires_full_artifact_valid_report(tmp_path):
    spec = collection_spec("validation", tmp_path)
    spec.report.parent.mkdir(parents=True)
    spec.report.write_text(
        "complete_and_artifact_valid: true\n"
        "protected_test_used: false\n"
        "counts: {expected: 324, observed: 324, usable: 324}\n"
        "integrity:\n"
        "  missing_episode_keys: []\n"
        "  unexpected_episode_keys: []\n"
        "  duplicate_episode_keys: []\n"
        "  replacement_errors: []\n",
        encoding="utf-8",
    )
    assert report_is_complete(spec)
    spec.report.write_text(
        spec.report.read_text(encoding="utf-8").replace("usable: 324", "usable: 323"),
        encoding="utf-8",
    )
    assert not report_is_complete(spec)
