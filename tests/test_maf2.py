import numpy as np
import pytest
from src.filters.maf2 import maf2


def test_output_shape_matches_input():
    """maf2 should return an array the same length as the input signal."""
    signal = np.random.normal(0, 1, 1000)
    smoothed = maf2(signal, N=21)
    assert len(smoothed) == len(signal)


def test_constant_signal_unchanged():
    """Smoothing a constant signal should return (approximately) the same constant."""
    signal = np.full(500, 3.0)
    smoothed = maf2(signal, N=21)
    assert np.allclose(smoothed[20:-20], 3.0, atol=1e-6)


def test_smoothing_reduces_noise_variance():
    """Smoothing should reduce the variance of a noisy signal."""
    np.random.seed(42)
    noisy = np.random.normal(0, 1, 1000)
    smoothed = maf2(noisy, N=21)
    assert np.var(smoothed) < np.var(noisy)


def test_kernel_normalization_preserves_dc_level():
    """The kernel should sum to 1, so a signal's average level shouldn't shift."""
    signal = np.random.normal(5, 1, 1000)
    smoothed = maf2(signal, N=21)
    assert abs(np.mean(smoothed) - np.mean(signal)) < 0.5


def test_larger_window_smooths_more():
    """A larger window size N should produce a smoother (lower variance) output."""
    np.random.seed(0)
    signal = np.random.normal(0, 1, 1000)
    smoothed_small_N = maf2(signal, N=5)
    smoothed_large_N = maf2(signal, N=51)
    assert np.var(smoothed_large_N) < np.var(smoothed_small_N)


def test_maf2_matches_manual_average_at_center():
    """Sanity check: at a point far from edges, output should equal the manual mean of that window."""
    np.random.seed(1)
    signal = np.random.normal(0, 1, 1000)
    N = 21
    smoothed = maf2(signal, N=N)
    center_idx = 500
    half = N // 2
    manual_avg = np.mean(signal[center_idx - half: center_idx + half + 1])
    assert np.isclose(smoothed[center_idx], manual_avg, atol=1e-6)