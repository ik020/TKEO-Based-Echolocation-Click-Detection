
import os
import numpy as np
import matplotlib.pyplot as plt

from src.signal.gabor import generate_gabor_click
from src.config import F0, T0, SIGMA, A, PHI, DURATION, FS, FIGURES_DIR, METRICS_DIR

# Generate Signal
t, x = generate_gabor_click(F0, T0, SIGMA, A, PHI, DURATION, FS)

# Ensure Ouput Signal exists
os.makedirs(
	FIGURES_DIR,
	exist_ok=True
)

os.makedirs(
	METRICS_DIR,
	exist_ok=True
)

# Plots
plt.figure(figsize=(10, 4))
plt.plot(t * 1000, x)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title("Phase 1: Synthetic Gabor Click")
plt.grid()
plt.savefig(
	f"{FIGURES_DIR}/phase1_gabor_click.png",
	dpi = 200
)
plt.show()

# Save raw data
np.save(f"{METRICS_DIR}/phase1_time.npy", t)
np.save(f"{METRICS_DIR}/phase1_signal.npy", x)