import numpy as np

def maf2(signal, N):
    kernel = np.ones(N) / N  # uniform weights, normalized to sum to 1
    smoothed = np.convolve(signal, kernel, mode="same")
    return smoothed