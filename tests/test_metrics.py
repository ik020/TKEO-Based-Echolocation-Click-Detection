import numpy as np
import pytest
from src.evaluation.metrics import evaluate_detections


def test_perfect_detection():
    true_times = [1.0, 2.0, 3.0]
    detected_times = [1.0, 2.0, 3.0]
    result = evaluate_detections(true_times, detected_times, tolerance=0.1)
    assert result["true_positives"] == 3
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_missed_click_counts_as_false_negative():
    true_times = [1.0, 2.0, 3.0]
    detected_times = [1.0, 3.0]
    result = evaluate_detections(true_times, detected_times, tolerance=0.1)
    assert result["true_positives"] == 2
    assert result["false_negatives"] == 1
    assert result["false_positives"] == 0


def test_extra_detection_counts_as_false_positive():
    true_times = [1.0, 2.0]
    detected_times = [1.0, 2.0, 5.0]
    result = evaluate_detections(true_times, detected_times, tolerance=0.1)
    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 0


def test_detection_within_tolerance_still_matches():
    true_times = [5.0]
    detected_times = [5.03]
    result = evaluate_detections(true_times, detected_times, tolerance=0.05)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0


def test_detection_outside_tolerance_does_not_match():
    true_times = [5.0]
    detected_times = [5.5]
    result = evaluate_detections(true_times, detected_times, tolerance=0.05)
    assert result["true_positives"] == 0
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1


def test_no_double_counting_one_detection_for_multiple_true_clicks():
    true_times = [1.0, 1.02]
    detected_times = [1.01]
    result = evaluate_detections(true_times, detected_times, tolerance=0.05)
    assert result["true_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["false_positives"] == 0


def test_empty_inputs():
    result = evaluate_detections([], [], tolerance=0.1)
    assert result["true_positives"] == 0
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0