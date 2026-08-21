import numpy as np
from src.signal.multi_click import generate_multi_click_signal
from src.tkeo.operator import tkeo
from src.filters.maf1 import maf1
from src.filters.maf2 import maf2
from src.detection.fdr import fdr, fdr_peak
from src.detection.detector import detect_clicks
from src.evaluation.metrics import evaluate_detections
from src.config import F0, SIGMA, A, PHI, FS


def test_full_pipeline_detects_all_clicks_with_fixed_seed():
    np.random.seed(42)

    click_times = [5e-3, 15e-3, 25e-3, 35e-3, 45e-3]
    duration = 60e-3
    snr_db = 5

    t, x_noisy, true_times = generate_multi_click_signal(
        click_times, FS, duration, snr_db, SIGMA, F0, PHI, A
    )

    psi_raw = tkeo(x_noisy, FS)

    sigma_tk_seconds = SIGMA / np.sqrt(2)
    sigma_g = sigma_tk_seconds * FS
    N = int(np.ceil(5 * sigma_g))

    psi_maf1 = maf1(psi_raw, sigma_g, N)
    psi_maf2 = maf2(psi_raw, N)
    fdr_curve = fdr(psi_maf1, psi_maf2)
    peak_value = fdr_peak(sigma_g, N)

    threshold = 0.35 * peak_value
    refractory_period = 2 * SIGMA

    detections = detect_clicks(fdr_curve, t, threshold, refractory_period=refractory_period)

    edge_margin = N / FS
    detected_times = [
        d["time"] for d in detections
        if edge_margin <= d["time"] <= (duration - edge_margin)
    ]

    results = evaluate_detections(true_times, detected_times, tolerance=2 * SIGMA)

    assert results["recall"] >= 0.8
    assert results["precision"] >= 0.8