
import numpy as np


def evaluate_detections(true_times, detected_times, tolerance):
    true_times = list(true_times)
    detected_times = list(detected_times)

    unmatched_true = true_times.copy()
    unmatched_detected = detected_times.copy()
    matches = []

    for true_t in true_times:
        if true_t not in unmatched_true:
            continue  

        candidates = [
            d for d in unmatched_detected
            if abs(d - true_t) <= tolerance
        ]
        if candidates:
            # Pick the closest candidate
            best = min(candidates, key=lambda d: abs(d - true_t))
            matches.append((true_t, best))
            unmatched_true.remove(true_t)
            unmatched_detected.remove(best)

    true_positives = len(matches)
    false_negatives = len(unmatched_true)      # true clicks with no matching detection
    false_positives = len(unmatched_detected)  # detections with no matching true click

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }