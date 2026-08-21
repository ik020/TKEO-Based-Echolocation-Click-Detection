import os
import numpy as np
import matplotlib.pyplot as plt

from src.signal.gabor import generate_gabor_click
from src.noise.noise import add_noise
from src.tkeo.operator import tkeo
from src.filters.maf1 import maf1
from src.config import F0, T0, SIGMA, A, PHI, DURATION, FS, FIGURES_DIR, METRICS_DIR

# Generate clean signal, add noise (use a fixed SNR to test filtering)
t, x = generate_gabor_click(F0, T0, SIGMA, A, PHI, DURATION, FS)
snr_db = 5
x_noisy = add_noise(x, snr_db=snr_db)

# Compute TKEO before and after MAF1 smoothing
psi_raw = tkeo(x_noisy, FS)

sigma_g = 5
N = 21
psi_smoothed = maf1(psi_raw, sigma_g, N)

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Plot raw vs smoothed TKEO
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(t * 1000, psi_raw, color="tab:red")
axes[0].set_ylabel("TKEO (raw)")
axes[0].set_title(f"Raw TKEO at {snr_db} dB SNR")
axes[0].grid()

axes[1].plot(t * 1000, psi_smoothed, color="tab:purple")
axes[1].set_ylabel("TKEO (MAF1 smoothed)")
axes[1].set_title(f"MAF1 Smoothed (sigma_g={sigma_g}, N={N})")
axes[1].grid()

plt.xlabel("Time (ms)")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/phase4_maf1.png", dpi=200)
plt.show()

np.save(f"{METRICS_DIR}/phase4_psi_raw.npy", psi_raw)
np.save(f"{METRICS_DIR}/phase4_psi_smoothed.npy", psi_smoothed)