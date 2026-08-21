import os
import numpy as np
import matplotlib.pyplot as plt

from src.signal.gabor import generate_gabor_click
from src.tkeo.operator import first_derivative, second_derivative, tkeo
from src.config import F0, T0, SIGMA, A, PHI, DURATION, FS, FIGURES_DIR, METRICS_DIR

# Generate Signal 
t, x = generate_gabor_click(F0, T0, SIGMA, A, PHI, DURATION, FS)

# Compute derivatives and TKEO
x_prime = first_derivative(x, FS)
x_double_prime = second_derivative(x, FS)
psi = tkeo(x, FS)

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Plots 
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

axes[0].plot(t * 1000, x, color="tab:blue")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Phase 2: Gabor Click")
axes[0].grid()

axes[1].plot(t * 1000, x_prime, color="tab:orange")
axes[1].set_ylabel("x'")
axes[1].set_title("First Derivative")
axes[1].grid()

axes[2].plot(t * 1000, x_double_prime, color="tab:green")
axes[2].set_ylabel("x''")
axes[2].set_title("Second Derivative")
axes[2].grid()

axes[3].plot(t * 1000, psi, color="tab:red")
axes[3].set_ylabel("TKEO")
axes[3].set_title("Teager-Kaiser Energy Operator")
axes[3].grid()

plt.xlabel("Time (ms)")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/phase2_tkeo.png", dpi=200)
plt.show()

# Save raw data
np.save(f"{METRICS_DIR}/phase2_time.npy", t)
np.save(f"{METRICS_DIR}/phase2_x_prime.npy", x_prime)
np.save(f"{METRICS_DIR}/phase2_x_double_prime.npy", x_double_prime)
np.save(f"{METRICS_DIR}/phase2_tkeo.npy", psi)