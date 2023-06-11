import numpy as np

def snr(signal):
    As = np.max(signal) - np.min(signal)
    sigmaN = 0
    SNR = 20*np.log10(As/(4*sigmaN))
    return SNR