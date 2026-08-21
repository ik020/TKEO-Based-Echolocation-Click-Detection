# tests/test_detector.py

import numpy as np
import pytest
from src.detection.detector import detect_clicks


def test_no_detections_when_all_below_threshold():
    fdr_curve = np.array([0.1, 0.2, 0.15, 0.1])
    t = np.array([0.0, 1.0, 2.0, 3.0])
    detections = detect_clicks(fdr_curve, t, threshold=0.5)
    assert len(detections) == 0


def test_single_region_detected_once():
    fdr_curve = np.array([0.1, 0.6, 0.9, 0.7, 0.1])
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    detections = detect_clicks(fdr_curve, t, threshold=0.5)
    assert len(detections) == 1
    assert detections[0]["time"] == 2.0
    assert detections[0]["fdr_value"] == 0.9


def test_multiple_separate_regions_detected_separately():
    fdr_curve = np.array([0.1, 0.6, 0.1, 0.1, 0.7, 0.1])
    t = np.arange(6, dtype=float)
    detections = detect_clicks(fdr_curve, t, threshold=0.5)
    assert len(detections) == 2
    assert detections[0]["time"] == 1.0
    assert detections[1]["time"] == 4.0


def test_region_touching_start_of_signal():
    fdr_curve = np.array([0.9, 0.6, 0.1, 0.1])
    t = np.arange(4, dtype=float)
    detections = detect_clicks(fdr_curve, t, threshold=0.5)
    assert len(detections) == 1
    assert detections[0]["time"] == 0.0


def test_region_touching_end_of_signal():
    fdr_curve = np.array([0.1, 0.1, 0.6, 0.9])
    t = np.arange(4, dtype=float)
    detections = detect_clicks(fdr_curve, t, threshold=0.5)
    assert len(detections) == 1
    assert detections[0]["time"] == 3.0


def test_detected_time_is_local_maximum_not_first_crossing():
    fdr_curve = np.array([0.1, 0.55, 0.95, 0.6, 0.1])
    t = np.arange(5, dtype=float)
    detections = detect_clicks(fdr_curve, t, threshold=0.5)
    assert len(detections) == 1
    assert detections[0]["time"] == 2.0


def test_refractory_period_merges_nearby_regions():
    fdr_curve = np.array([0.1, 0.6, 0.1, 0.6, 0.1])
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    detections = detect_clicks(fdr_curve, t, threshold=0.5, refractory_period=2.5)
    assert len(detections) == 1
    assert detections[0]["time"] == 1.0


def test_refractory_period_does_not_merge_distant_regions():
    fdr_curve = np.array([0.1, 0.6, 0.1, 0.1, 0.1, 0.6, 0.1])
    t = np.arange(7, dtype=float)
    detections = detect_clicks(fdr_curve, t, threshold=0.5, refractory_period=2.0)
    assert len(detections) == 2


def test_no_refractory_period_preserves_old_behavior():
    fdr_curve = np.array([0.1, 0.6, 0.1, 0.6, 0.1])
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    detections = detect_clicks(fdr_curve, t, threshold=0.5, refractory_period=None)
    assert len(detections) == 2