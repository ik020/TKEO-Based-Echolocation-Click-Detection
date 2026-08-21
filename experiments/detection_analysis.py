
import os
import numpy as np
import matplotlib.pyplot as plt

from src.signal.gabor import generate_gabor_click
from src.noise.noise import add_noise
from src.tkeo.operator import tkeo
from src.filters.maf1 import maf1
from src.filters.maf2 import maf2
from src.detection.fdr import fdr, fdr_peak
from src.detection.detector import detect_clicks
from src.config import F0, T0, SIGMA, A, PHI, FS, FIGURES_DIR, METRICS_DIR

DURATION_LONG = 50e-3

t, x = generate_gabor_click(F0, T0, SIGMA, A, PHI, DURATION_LONG, FS)
snr_db = 5
x_noisy = add_noise(x, snr_db=snr_db)
psi_raw = tkeo(x_noisy, FS)

sigma_tk_seconds = SIGMA / np.sqrt(2)
sigma_g = sigma_tk_seconds * FS
N = int(np.ceil(5 * sigma_g))

psi_maf1 = maf1(psi_raw, sigma_g, N)
psi_maf2 = maf2(psi_raw, N)
fdr_curve = fdr(psi_maf1, psi_maf2)
peak_value = fdr_peak(sigma_g, N)

threshold_fraction = 0.5
threshold = threshold_fraction * peak_value


detections = detect_clicks(fdr_curve, t, threshold)

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(t * 1000, x_noisy, color="tab:red", alpha=0.6)
axes[0].set_ylabel("Noisy Signal")
axes[0].set_title(f"Noisy Waveform ({snr_db} dB SNR)")
axes[0].grid()

axes[1].plot(t * 1000, fdr_curve, color="tab:blue")
axes[1].axhline(threshold, color="black", linestyle="--", label=f"Threshold = {threshold:.3f}")
for d in detections:
    axes[1].plot(d["time"] * 1000, d["fdr_value"], "ro", markersize=8)
axes[1].set_ylabel("FDR")
axes[1].set_title(f"FDR with Detected Clicks (threshold = {threshold_fraction} * FDR_peak)")
axes[1].legend()
axes[1].grid()

plt.xlabel("Time (ms)")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/phase7_detection.png", dpi=200)
plt.show()

print(f"FDR_peak: {peak_value:.4f}, Threshold: {threshold:.4f}")
print(f"Number of detections: {len(detections)}")
for i, d in enumerate(detections):
    print(f"  Detection {i+1}: time={d['time']*1000:.3f} ms, fdr_value={d['fdr_value']:.4f}, "
          f"region=[{d['start_time']*1000:.3f}, {d['end_time']*1000:.3f}] ms")

np.save(f"{METRICS_DIR}/phase7_fdr_curve.npy", fdr_curve)