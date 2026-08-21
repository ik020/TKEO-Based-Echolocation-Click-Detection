# experiments/08_evaluation.py

import os
import numpy as np
import matplotlib.pyplot as plt

from src.signal.multi_click import generate_multi_click_signal
from src.tkeo.operator import tkeo
from src.filters.maf1 import maf1
from src.filters.maf2 import maf2
from src.detection.fdr import fdr, fdr_peak
from src.detection.detector import detect_clicks
from src.evaluation.metrics import evaluate_detections
from src.config import F0, SIGMA, A, PHI, FS, FIGURES_DIR, METRICS_DIR

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

threshold_fraction = 0.35
threshold = threshold_fraction * peak_value

refractory_period = 2 * SIGMA
detections = detect_clicks(fdr_curve, t, threshold, refractory_period=refractory_period)

edge_margin = N / FS
detected_times = [
    d["time"] for d in detections
    if edge_margin <= d["time"] <= (duration - edge_margin)
]
excluded_count = len(detections) - len(detected_times)

results = evaluate_detections(true_times, detected_times, tolerance=2 * SIGMA)

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

axes[0].plot(t * 1000, x_noisy, color="tab:red", alpha=0.6)
for tt in true_times:
    axes[0].axvline(tt * 1000, color="green", linestyle=":", alpha=0.7)
axes[0].set_ylabel("Noisy Signal")
axes[0].set_title(f"Multi-Click Signal ({snr_db} dB SNR) - green dashed = ground truth")
axes[0].grid()

axes[1].plot(t * 1000, fdr_curve, color="tab:blue")
axes[1].axhline(threshold, color="black", linestyle="--", label=f"Threshold = {threshold:.3f}")
axes[1].axvline(edge_margin * 1000, color="gray", linestyle=":", alpha=0.5)
axes[1].axvline((duration - edge_margin) * 1000, color="gray", linestyle=":", alpha=0.5)
for d in detections:
    axes[1].plot(d["time"] * 1000, d["fdr_value"], "ro", markersize=8)
axes[1].set_ylabel("FDR")
axes[1].set_title("FDR with Detections (red dots) - gray dashed = edge exclusion zone")
axes[1].legend()
axes[1].grid()

plt.xlabel("Time (ms)")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/phase8_evaluation.png", dpi=200)
plt.show()

print(f"Excluded {excluded_count} edge-artifact detection(s) (within {edge_margin*1000:.2f}ms of boundaries)")
print(f"True clicks: {[f'{tt*1000:.2f}ms' for tt in true_times]}")
print(f"Detected (after edge exclusion): {[f'{dt*1000:.2f}ms' for dt in detected_times]}")
print(f"TP={results['true_positives']}, FP={results['false_positives']}, FN={results['false_negatives']}")
print(f"Precision={results['precision']:.3f}, Recall={results['recall']:.3f}, F1={results['f1']:.3f}")

np.save(f"{METRICS_DIR}/phase8_fdr_curve.npy", fdr_curve)