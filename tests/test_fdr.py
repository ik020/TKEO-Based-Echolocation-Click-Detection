
import numpy as np
import pytest
from src.detection.fdr import fdr, fdr_peak


def test_output_shape_matches_input():
    """fdr() should return an array the same length as the inputs."""
    maf1_out = np.array([1.0, 2.0, 3.0, 4.0])
    maf2_out = np.array([0.5, 1.0, 1.5, 2.0])
    result = fdr(maf1_out, maf2_out)
    assert len(result) == len(maf1_out)


def test_mismatched_lengths_raises():
    """fdr() should raise if maf1 and maf2 outputs have different lengths."""
    maf1_out = np.array([1.0, 2.0, 3.0])
    maf2_out = np.array([0.5, 1.0])
    with pytest.raises(ValueError):
        fdr(maf1_out, maf2_out)


def test_fdr_matches_manual_calculation():
    """Sanity check against a hand-computed example where bypass conditions all hold."""
    maf1_out = np.array([4.0])
    maf2_out = np.array([2.0])
    result = fdr(maf1_out, maf2_out)
    expected = (4.0 - 2.0) / 4.0  # = 0.5
    assert np.isclose(result[0], expected)


def test_bypass_when_maf1_nonpositive():
    """If MAF1(n) <= 0, FDR should be bypassed (set to 0), regardless of MAF2."""
    maf1_out = np.array([-1.0, 0.0])
    maf2_out = np.array([0.5, 0.5])
    result = fdr(maf1_out, maf2_out)
    assert np.all(result == 0.0)


def test_bypass_when_maf2_nonpositive():
    """If MAF2(n) <= 0, FDR should be bypassed (set to 0)."""
    maf1_out = np.array([2.0, 2.0])
    maf2_out = np.array([-0.5, 0.0])
    result = fdr(maf1_out, maf2_out)
    assert np.all(result == 0.0)


def test_bypass_when_maf1_not_greater_than_maf2():
    """If MAF1(n) <= MAF2(n), FDR should be bypassed (set to 0), per property (iv)."""
    maf1_out = np.array([1.0, 2.0])
    maf2_out = np.array([1.0, 3.0])  # equal, then MAF2 > MAF1
    result = fdr(maf1_out, maf2_out)
    assert np.all(result == 0.0)


def test_fdr_values_within_valid_range():
    """Wherever FDR is computed (not bypassed), it should fall within [0, 1]."""
    np.random.seed(0)
    maf1_out = np.abs(np.random.normal(5, 2, 1000))
    maf2_out = np.abs(np.random.normal(2, 1, 1000))
    result = fdr(maf1_out, maf2_out)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)

def test_fdr_peak_within_valid_range():
    peak = fdr_peak(sigma_g=5, N=21)
    assert 0.0 <= peak <= 1.0


def test_fdr_peak_increases_with_sharper_gaussian():
    """A smaller sigma_g (sharper Gaussian, same N) should give a higher FDR_peak."""
    peak_sharp = fdr_peak(sigma_g=2, N=21)
    peak_wide = fdr_peak(sigma_g=10, N=21)
    assert peak_sharp > peak_wide