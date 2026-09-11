from scripts.plot_annotation_review import render


def test_review_plot_contains_causal_markers_and_no_model_language():
    annotation = {
        "run_id": "run-1",
        "episode": {"start_time": 0.0, "end_time": 20.0, "termination_reason": "collision"},
        "injections": [{"eligible": True, "actual_onset": 5.0}],
        "events": [{"terminal": True, "time": 18.0, "class": "collision"}],
    }
    rows = [
        {"feature": "command_linear", "timestamp": "1", "value": "0.1"},
        {"feature": "measured_linear", "timestamp": "1", "value": "0.09"},
        {"feature": "pose_covariance_trace", "timestamp": "2", "value": "0.2"},
        {"feature": "valid_return_fraction", "timestamp": "2", "value": "0.8"},
        {"feature": "minimum_front_range", "timestamp": "2", "value": "1.5"},
    ]
    svg = render(annotation, rows)
    assert svg.startswith("<svg")
    assert "injection onset" in svg
    assert "terminal event" in svg
    assert "no model outputs shown" in svg
