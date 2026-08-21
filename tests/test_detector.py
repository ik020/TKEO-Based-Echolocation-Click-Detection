
import numpy as np
import pytest
from src.detection.detector import detect_clicks


def test_no_detections_when_all_below_threshold():
    """If nothing exceeds the threshold, no detections should be returned."""
    fdr_curve = np.array([0.1, 0.2, 0.15, 0.1])
    t = np.array([0.0, 1.0, 2.0, 3.0])
    detections = detect_clicks(fdr_curve, t, threshold=0.5)
    assert len(detections) == 0


def test_single_region_detected_once():
    """A single contiguous above-threshold region should yield exactly one detection."""
    fdr_curve = np.array([0.1, 0.6, 0.9, 0.7, 0.1])
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    detections = detect_clicks(fdr_curve, t, threshold=0.5)
    assert len(detections) == 1
    # The detected time should correspond to the max value in the region (index 2, value 0.9)
    assert detections[0]["time"] == 2.0
    assert detections[0]["fdr_value"] == 0.9


def test_multiple_separate_regions_detected_separately():
    """Two separate above-threshold bumps should yield two detections, not one."""
    fdr_curve = np.array([0.1, 0.6, 0.1, 0.1, 0.7, 0.1])
    t = np.arange(6, dtype=float)
    detections = detect_clicks(fdr_curve, t, threshold=0.5)
    assert len(detections) == 2
    assert detections[0]["time"] == 1.0
    assert detections[1]["time"] == 4.0


def test_region_touching_start_of_signal():
    """A region that starts at index 0 (already above threshold) should still be detected."""
    fdr_curve = np.array([0.9, 0.6, 0.1, 0.1])
    t = np.arange(4, dtype=float)
    detections = detect_clicks(fdr_curve, t, threshold=0.5)
    assert len(detections) == 1
    assert detections[0]["time"] == 0.0


def test_region_touching_end_of_signal():
    """A region that extends to the last sample (still above threshold) should still be detected."""
    fdr_curve = np.array([0.1, 0.1, 0.6, 0.9])
    t = np.arange(4, dtype=float)
    detections = detect_clicks(fdr_curve, t, threshold=0.5)
    assert len(detections) == 1
    assert detections[0]["time"] == 3.0


def test_detected_time_is_local_maximum_not_first_crossing():
    """The reported time should be where FDR is highest within the region, not just where it first crosses threshold."""
    fdr_curve = np.array([0.1, 0.55, 0.95, 0.6, 0.1])  # peak is in the middle, not at the edges
    t = np.arange(5, dtype=float)
    detections = detect_clicks(fdr_curve, t, threshold=0.5)
    assert len(detections) == 1
    assert detections[0]["time"] == 2.0  # index of the 0.95 peak