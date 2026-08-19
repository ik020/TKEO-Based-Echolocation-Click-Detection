import numpy as np

def first_derivative(signal, fs):
    return np.gradient(signal, 1/fs)

def second_derivative(signal, fs):
    return np.gradient(first_derivative(signal, fs), 1/fs)

def tkeo(signal, fs):
    x_prime = first_derivative(signal, fs)
    x_double_prime = second_derivative(signal, fs)
    return x_prime**2 - signal * x_double_prime