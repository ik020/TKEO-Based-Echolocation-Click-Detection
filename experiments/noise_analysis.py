import os
import numpy as np
import matplotlib.pyplot as plt

from src.signal.gabor import generate_gabor_click
from src.noise.noise import add_noise
from src.tkeo.operator import tkeo
from src.config import F0, T0, SIGMA, A, PHI, DURATION, FS, SNR_LEVELS_DB, FIGURES_DIR, METRICS_DIR

t, x = generate_gabor_click(F0, T0, SIGMA, A, PHI, DURATION, FS)

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

fig, axes = plt.subplots(len(SNR_LEVELS_DB), 2, figsize=(12, 10), sharex=True)

for i, snr_db in enumerate(SNR_LEVELS_DB):
    x_noisy = add_noise(x, snr_db)
    psi_noisy = tkeo(x_noisy, FS)

    axes[i, 0].plot(t * 1000, x_noisy, color="tab:blue")
    axes[i, 0].set_ylabel(f"{snr_db} dB")
    if i == 0:
        axes[i, 0].set_title("Noisy Signal")
    axes[i, 0].grid()

    axes[i, 1].plot(t * 1000, psi_noisy, color="tab:red")
    if i == 0:
        axes[i, 1].set_title("TKEO")
    axes[i, 1].grid()

    np.save(f"{METRICS_DIR}/phase3_signal_snr{snr_db}.npy", x_noisy)
    np.save(f"{METRICS_DIR}/phase3_tkeo_snr{snr_db}.npy", psi_noisy)

plt.xlabel("Time (ms)")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/phase3_noise_analysis.png", dpi=200)
plt.show()