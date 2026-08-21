import os
import numpy as np
import matplotlib.pyplot as plt

from src.signal.gabor import generate_gabor_click
from src.noise.noise import add_noise
from src.tkeo.operator import tkeo
from src.filters.maf1 import maf1
from src.filters.maf2 import maf2
from src.config import F0, T0, SIGMA, A, PHI, DURATION, FS, FIGURES_DIR, METRICS_DIR

# Generate clean signal, add noise 
t, x = generate_gabor_click(F0, T0, SIGMA, A, PHI, DURATION, FS)
snr_db = 5
x_noisy = add_noise(x, snr_db=snr_db)

# Compute TKEO, then both filters
psi_raw = tkeo(x_noisy, FS)

sigma_g, N_gauss = 5, 21
N_rect = 21

psi_maf1 = maf1(psi_raw, sigma_g, N_gauss)
psi_maf2 = maf2(psi_raw, N_rect)

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Plot raw, MAF1, MAF2 side by side for comparison
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(t * 1000, psi_raw, color="tab:red")
axes[0].set_ylabel("TKEO (raw)")
axes[0].set_title(f"Raw TKEO at {snr_db} dB SNR")
axes[0].grid()

axes[1].plot(t * 1000, psi_maf1, color="tab:purple")
axes[1].set_ylabel("MAF1 (Gaussian)")
axes[1].set_title(f"MAF1 Smoothed (sigma_g={sigma_g}, N={N_gauss})")
axes[1].grid()

axes[2].plot(t * 1000, psi_maf2, color="tab:green")
axes[2].set_ylabel("MAF2 (Rectangular)")
axes[2].set_title(f"MAF2 Smoothed (N={N_rect})")
axes[2].grid()

plt.xlabel("Time (ms)")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/phase5_maf2.png", dpi=200)
plt.show()

np.save(f"{METRICS_DIR}/phase5_psi_raw.npy", psi_raw)
np.save(f"{METRICS_DIR}/phase5_psi_maf1.npy", psi_maf1)
np.save(f"{METRICS_DIR}/phase5_psi_maf2.npy", psi_maf2)