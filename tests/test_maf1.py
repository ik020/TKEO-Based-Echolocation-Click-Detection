import numpy as np
import pytest
from src.filters.maf1 import maf1


def test_output_shape_matches_input():
    """maf1 should return an array the same length as the input signal."""
    signal = np.random.normal(0, 1, 1000)
    smoothed = maf1(signal, sigma_g=5, N=21)
    assert len(smoothed) == len(signal)


def test_constant_signal_unchanged():
    """Smoothing a constant signal should return (approximately) the same constant."""
    signal = np.full(500, 3.0)
    smoothed = maf1(signal, sigma_g=5, N=21)
    # Edges may deviate slightly due to convolution boundary effects; check interior
    assert np.allclose(smoothed[20:-20], 3.0, atol=1e-6)


def test_smoothing_reduces_noise_variance():
    """Smoothing should reduce the variance of a noisy signal (that's the whole point)."""
    np.random.seed(42)
    noisy = np.random.normal(0, 1, 1000)
    smoothed = maf1(noisy, sigma_g=5, N=21)
    assert np.var(smoothed) < np.var(noisy)


def test_kernel_normalization_preserves_dc_level():
    """The kernel should sum to 1, so a signal's average level shouldn't shift."""
    signal = np.random.normal(5, 1, 1000)  # noisy signal with mean ~5
    smoothed = maf1(signal, sigma_g=5, N=21)
    assert abs(np.mean(smoothed) - np.mean(signal)) < 0.5


def test_larger_sigma_smooths_more():
    """A larger sigma_g should produce a smoother (lower variance) output."""
    np.random.seed(0)
    signal = np.random.normal(0, 1, 1000)
    smoothed_small_sigma = maf1(signal, sigma_g=1, N=21)
    smoothed_large_sigma = maf1(signal, sigma_g=10, N=41)
    assert np.var(smoothed_large_sigma) < np.var(smoothed_small_sigma)