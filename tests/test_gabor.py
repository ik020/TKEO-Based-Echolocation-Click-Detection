import numpy as np
import pytest
from src.signal.gabor import generate_gabor_click

# Test Parameters
f0 = 20_000       # carrier frequency: 20 kHz
t0 = 5e-3         # center time: 5 ms
sigma = 0.5e-3    # envelope width
A = 1.0
phi = 0.0
duration = 10e-3  # total signal length: 10 ms
fs = 200_000      # sampling frequency


def call():
    return generate_gabor_click(
        f0=f0, 
        t0=t0, 
        sigma=sigma, 
        A=A, 
        phi=phi, 
        duration=duration, 
        fs=fs
    )


def test_output_shapes_match():
    t, x = call()
    assert len(t) == len(x)


def test_signal_length_matches_duration():
    t, x = call()
    expected_samples = int(duration * fs)
    assert abs(len(x) - expected_samples) <= 1


def test_peak_occurs_near_center_time():
    t, x = call()
    peak_index = np.argmax(np.abs(x))
    peak_time = t[peak_index]
    assert abs(peak_time - t0) < 2 * sigma


def test_amplitude_within_bounds():
    t, x = call()
    assert np.max(np.abs(x)) <= A + 1e-6


def test_signal_decays_at_edges():
    t, x = call()
    edge_samples = np.concatenate([x[:10], x[-10:]])
    assert np.all(np.abs(edge_samples) < 0.05 * A)


def test_time_vector_is_monotonic():
    t, x = call()
    assert np.all(np.diff(t) > 0)


def test_zero_sigma_raises_or_handled():
    with pytest.raises((ValueError, ZeroDivisionError)):
        generate_gabor_click(f0=f0, t0=t0, sigma=0, A=A, phi=phi, duration=duration, fs=fs)