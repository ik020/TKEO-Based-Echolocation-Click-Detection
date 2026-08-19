
import numpy as np

def generate_gabor_click(f0, t0, sigma, A, phi, duration, fs):
	t = np.arange(0, duration, 1/fs)
	envelope = A * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))
	carrier = np.cos(2 * np.pi * f0 * (t - t0) + phi)
	x = envelope * carrier
	return t, x

