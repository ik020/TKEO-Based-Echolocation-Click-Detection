
import os
import numpy as np
import matplotlib.pyplot as plt

from src.signal.gabor import generate_gabor_click
from src.noise.noise import add_noise
from src.tkeo.operator import tkeo
from src.filters.maf1 import maf1
from src.filters.maf2 import maf2
from src.detection.fdr import fdr, fdr_peak
from src.config import F0, T0, SIGMA, A, PHI, DURATION, FS, FIGURES_DIR, METRICS_DIR

# Generate noisy signal and TKEO 
t, x = generate_gabor_click(F0, T0, SIGMA, A, PHI, DURATION, FS)
snr_db = 5
x_noisy = add_noise(x, snr_db=snr_db)
psi_raw = tkeo(x_noisy, FS)

# Filter parameters 
sigma_g = 5e-4  
N = 21

psi_maf1 = maf1(psi_raw, sigma_g, N)
psi_maf2 = maf2(psi_raw, N)

# Compute FDR curve and FDR_peak 
fdr_curve = fdr(psi_maf1, psi_maf2)
peak_value = fdr_peak(sigma_g, N, FS)

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Plot MAF1/MAF2 and resulting FDR curve
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(t * 1000, psi_maf1, color="tab:purple", label="MAF1 (Gaussian)")
axes[0].plot(t * 1000, psi_maf2, color="tab:green", label="MAF2 (Rectangular)")
axes[0].set_ylabel("TKEO (filtered)")
axes[0].set_title(f"MAF1 vs MAF2 at {snr_db} dB SNR")
axes[0].legend()
axes[0].grid()

axes[1].plot(t * 1000, fdr_curve, color="tab:blue")
axes[1].axhline(peak_value, color="black", linestyle="--", label=f"FDR_peak = {peak_value:.3f}")
axes[1].set_ylabel("FDR")
axes[1].set_title("Filter Difference Ratio")
axes[1].legend()
axes[1].grid()

axes[2].plot(t * 1000, x_noisy, color="tab:red", alpha=0.6)
axes[2].set_ylabel("Noisy Signal")
axes[2].set_title("Original Noisy Waveform (reference)")
axes[2].grid()

plt.xlabel("Time (ms)")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/phase6_fdr.png", dpi=200)
plt.show()

np.save(f"{METRICS_DIR}/phase6_fdr_curve.npy", fdr_curve)
print(f"FDR_peak for this filter combination: {peak_value:.4f}")