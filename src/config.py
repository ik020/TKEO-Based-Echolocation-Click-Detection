
'''
Shared configuration parameters for the simulation.
'''

'''Gabor Click Parameters'''
F0 = 20_000 # center frequency (Hz)
T0 = 0.005 # click center time (s)
SIGMA = 0.0005 # standard deviation of the Gaussian envelope (s)
A = 1 # amplitude of the click
PHI = 0 # phase of the click (radians)
DURATION = 50e-3 # duration of the click (s)
FS = 500_000 # sampling frequency (Hz)

'''Paths'''
FIGURES_DIR = "results/figures"
METRICS_DIR = "results/metrics"

'''SNR levels for noise addition'''
SNR_LEVELS_DB = [20, 10, 5, 0]

