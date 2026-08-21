import numpy as np

def fdr(maf1_output, maf2_output):
    if len(maf1_output) != len(maf2_output):
        raise ValueError("maf1_output and maf2_output must be the same length")

    valid = (maf1_output > 0) & (maf2_output > 0) & (maf1_output > maf2_output)

    fdr_values = np.zeros_like(maf1_output, dtype=float)
    fdr_values[valid] = (maf1_output[valid] - maf2_output[valid]) / maf1_output[valid]

    return fdr_values

def fdr_peak(sigma_g, N, fs):
    Ts = 1 / fs
    n = np.arange(-N, N + 1)

    # MAF1 impulse response (Eq. 12) - Gaussian
    h_maf1 = (Ts / (sigma_g * np.sqrt(2 * np.pi))) * np.exp(-((n * Ts) ** 2) / (2 * sigma_g ** 2))

    # MAF2 impulse response (Eq. 13) — average of h_maf1 values (rectangular, same gain)
    h_maf2 = np.full_like(h_maf1, np.mean(h_maf1))

    # Evaluate at n=0 (center index)
    center_idx = N  # since n runs from -N to N, index N corresponds to n=0
    h1_0 = h_maf1[center_idx]
    h2_0 = h_maf2[center_idx]

    return (h1_0 - h2_0) / h1_0
