
def maf1(signal, sigma_g, N):
    half = N // 2
    n = np.arange(-half, half + 1)
    kernel = np.exp(-(n ** 2) / (2 * sigma_g ** 2))
    kernel = kernel / np.sum(kernel)  # normalize so it doesn't change signal scale
    smoothed = np.convolve(signal, kernel, mode="same")
    return smoothed