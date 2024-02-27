import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft, fftshift
from scipy.signal import welch, find_peaks


def get_fft_values(x):
    fxt = fft(x)
    fxt = 2.0 / len(fxt) * np.abs(fxt[0 : len(fxt) // 2])
    return fxt


def get_psd_values(x):
    # fs : sampling freq by default 1
    _, psd_values = welch(x, fs=1)
    return psd_values


def get_first_n_peaks(x, no_peaks=5):
    x_ = list(x)

    if len(x_) >= no_peaks:
        return x_[:no_peaks]
    else:
        missing_no_peaks = no_peaks - len(x_)
        return x_ + [0] * missing_no_peaks


def get_features(x):
    indices_peaks, _ = find_peaks(x)
    peaks_x = get_first_n_peaks(x[indices_peaks])
    return peaks_x


#### test
# t_n = 10
# N = 1000
# T = t_n / N
# f_s = 1 / T

# x_value = np.linspace(0, t_n, N)
# amplitudes = [4, 6, 8, 10, 14]
# frequencies = [6.5, 5, 3, 1.5, 1]
# y_values = [
#     amplitudes[ii] * np.sin(2 * np.pi * frequencies[ii] * x_value)
#     for ii in range(0, len(amplitudes))
# ]
# composite_y_value = np.sum(y_values, axis=0)

# print(get_features(get_fft_values(composite_y_value)))
x = [1] * 100
print(get_features(get_fft_values(x)))
