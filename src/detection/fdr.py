
import numpy as np
from src.filters.maf1 import maf1
from src.filters.maf2 import maf2


def fdr(maf1_output, maf2_output):
    if len(maf1_output) != len(maf2_output):
        raise ValueError("maf1_output and maf2_output must be the same length")

    valid = (maf1_output > 0) & (maf2_output > 0) & (maf1_output > maf2_output)
    fdr_values = np.zeros_like(maf1_output, dtype=float)
    fdr_values[valid] = (maf1_output[valid] - maf2_output[valid]) / maf1_output[valid]
    return fdr_values


def fdr_peak(sigma_g, N):

    length = 4 * N + 1  # generous padding so convolution edges don't affect the center
    impulse = np.zeros(length)
    center = length // 2
    impulse[center] = 1.0

    h1 = maf1(impulse, sigma_g, N)
    h2 = maf2(impulse, N)

    return (h1[center] - h2[center]) / h1[center]