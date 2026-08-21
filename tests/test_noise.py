import numpy as np
import pytest
from src.noise.noise import add_noise
from src.signal.gabor import generate_gabor_click
from src.config import F0, T0, SIGMA, A, PHI, DURATION, FS


def test_output_shape_matches_input():
    """add_noise should return an array the same length as the input signal."""
    t, x = generate_gabor_click(F0, T0, SIGMA, A, PHI, DURATION, FS)
    x_noisy = add_noise(x, snr_db=10)
    assert len(x_noisy) == len(x)


def test_achieved_snr_close_to_target():
    """The measured SNR of the output should be close to the requested SNR."""
    t, x = generate_gabor_click(F0, T0, SIGMA, A, PHI, DURATION, FS)
    target_snr_db = 10
    x_noisy = add_noise(x, snr_db=target_snr_db)

    noise = x_noisy - x
    signal_power = np.mean(x ** 2)
    noise_power = np.mean(noise ** 2)
    achieved_snr_db = 10 * np.log10(signal_power / noise_power)

    # Allow some tolerance since noise is randomly generated each call
    assert abs(achieved_snr_db - target_snr_db) < 1.5


def test_higher_snr_means_less_noise_power():
    """A higher requested SNR should result in lower noise power than a lower SNR."""
    t, x = generate_gabor_click(F0, T0, SIGMA, A, PHI, DURATION, FS)

    x_noisy_high_snr = add_noise(x, snr_db=20)
    x_noisy_low_snr = add_noise(x, snr_db=0)

    noise_high = x_noisy_high_snr - x
    noise_low = x_noisy_low_snr - x

    power_high = np.mean(noise_high ** 2)
    power_low = np.mean(noise_low ** 2)

    assert power_high < power_low


def test_zero_signal_produces_zero_noise():
    """If the signal has zero power, no noise should be added (noise_power = 0)."""
    signal = np.zeros(1000)
    x_noisy = add_noise(signal, snr_db=10)
    assert np.allclose(x_noisy, 0.0, atol=1e-8)


def test_repeated_calls_give_different_noise():
    """Since noise is random, two calls with the same inputs should NOT be identical."""
    t, x = generate_gabor_click(F0, T0, SIGMA, A, PHI, DURATION, FS)
    x_noisy_1 = add_noise(x, snr_db=10)
    x_noisy_2 = add_noise(x, snr_db=10)
    assert not np.allclose(x_noisy_1, x_noisy_2)