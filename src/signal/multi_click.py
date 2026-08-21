# src/signal/multi_click.py

import numpy as np
from src.signal.gabor import generate_gabor_click
from src.noise.noise import add_noise


def generate_multi_click_signal(click_times, fs, duration, snr_db,
                                  sigma, f0, phi, A=1.0):
    t = np.arange(0, duration, 1 / fs)
    clean_signal = np.zeros_like(t)

    for t0 in click_times:
        _, click = generate_gabor_click(f0, t0, sigma, A, phi, duration, fs)
        clean_signal += click

    noisy_signal = add_noise(clean_signal, snr_db=snr_db)

    return t, noisy_signal, click_times