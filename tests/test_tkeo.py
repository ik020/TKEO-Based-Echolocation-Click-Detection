
import numpy as np
import pytest
from src.tkeo.operator import first_derivative, second_derivative, tkeo
from src.signal.gabor import generate_gabor_click
from src.config import F0, T0, SIGMA, A, PHI, DURATION, FS


def test_output_shapes_match_input():
    signal = np.sin(2 * np.pi * 1000 * np.arange(0, 0.01, 1/FS))
    assert len(first_derivative(signal, FS)) == len(signal)
    assert len(second_derivative(signal, FS)) == len(signal)
    assert len(tkeo(signal, FS)) == len(signal)


def test_tkeo_zero_for_constant_signal():
    signal = np.full(1000, 5.0)  # constant value, arbitrary length
    psi = tkeo(signal, FS)
    assert np.allclose(psi, 0.0, atol=1e-8)


def test_derivative_of_linear_signal():
    t = np.arange(0, 0.01, 1/FS)
    signal = t.copy()  # x(t) = t
    x_prime = first_derivative(signal, FS)
    # Check interior points only; np.gradient is less accurate at the edges
    assert np.allclose(x_prime[5:-5], 1.0, atol=1e-6)


def test_second_derivative_of_linear_signal_is_zero():
    t = np.arange(0, 0.01, 1/FS)
    signal = t.copy()
    x_double_prime = second_derivative(signal, FS)
    assert np.allclose(x_double_prime[5:-5], 0.0, atol=1e-4)


def test_tkeo_peak_near_gabor_click_center():
    t, x = generate_gabor_click(F0, T0, SIGMA, A, PHI, DURATION, FS)
    psi = tkeo(x, FS)
    peak_index = np.argmax(psi)
    peak_time = t[peak_index]
    assert abs(peak_time - T0) < 2 * SIGMA


def test_tkeo_nonnegative_for_gabor_click():
    t, x = generate_gabor_click(F0, T0, SIGMA, A, PHI, DURATION, FS)
    psi = tkeo(x, FS)
    # Allow tiny negative values from numerical derivative noise
    assert np.min(psi) > -1e-3