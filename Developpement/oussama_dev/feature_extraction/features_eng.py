from collections import Counter
import numpy as np
from scipy import stats
from scipy.fft import fft
from scipy.signal import welch, find_peaks
import pywt
from sklearn.base import BaseEstimator, TransformerMixin
from dataclasses import dataclass


@dataclass
class FeatureExtractor(BaseEstimator, TransformerMixin):
    no_peaks: int = 2
    waveletname: str = "db2"

    def dwt_transform(self, signal: np.ndarray) -> list[np.ndarray]:

        coeffs = pywt.wavedec(signal, self.waveletname)
        # reconstructed_signal = pywt.waverec(coeffs, waveletname)
        return coeffs

    def calculate_statistics(self, list_values: list[float]) -> list[float]:
        n5 = np.nanpercentile(list_values, 5)
        n25 = np.nanpercentile(list_values, 25)
        n75 = np.nanpercentile(list_values, 75)
        n95 = np.nanpercentile(list_values, 95)
        median = np.nanpercentile(list_values, 50)
        mean = np.nanmean(list_values)
        std = np.nanstd(list_values)
        var = np.nanvar(list_values)
        rms = np.sqrt(np.nanmean(np.square(list_values)))
        return [n5, n25, n75, n95, median, mean, std, var, rms]

    def calculate_crossings(self, list_values: list[float]) -> list[float]:
        values_array = np.array(list_values)
        # Calculate the mean value of the list values
        mean_value = np.nanmean(values_array)
        # Calculate zero crossings
        zero_crossing_indices = np.nonzero(np.diff(values_array > 0))[0]
        no_zero_crossings = len(zero_crossing_indices)
        # Calculate mean crossings
        mean_crossing_indices = np.nonzero(np.diff(values_array > mean_value))[0]
        no_mean_crossings = len(mean_crossing_indices)
        return [no_zero_crossings, no_mean_crossings]

    def calculate_entropy(self, list_values: list[float]) -> list[float]:
        counter_values = Counter(list_values).most_common()
        probabilities = [elem[1] / len(list_values) for elem in counter_values]
        entropy = stats.entropy(probabilities)
        return [entropy]

    def get_dwt_features(self, signal: np.ndarray):
        dwt_features = []
        coeffs = self.dwt_transform(signal)
        for coeff in coeffs:
            dwt_features += self.calculate_statistics(coeff)
            dwt_features += self.calculate_crossings(coeff)
            dwt_features += self.calculate_entropy(coeff)
        return dwt_features

    def get_fft_values(self, x):
        fxt = fft(x)
        fxt = 2.0 / len(fxt) * np.abs(fxt[0 : len(fxt) // 2])
        return fxt

    def get_psd_values(self, x, f_s=1):
        # fs : sampling freq by default 1
        _, psd_values = welch(x, fs=f_s)
        return psd_values

    def get_first_n_peaks(self, x):
        x_ = list(x)

        if len(x_) >= self.no_peaks:
            return x_[: self.no_peaks]
        else:
            missing_no_peaks = self.no_peaks - len(x_)
            return x_ + [0] * missing_no_peaks

    def get_features(self, x):
        indices_peaks, _ = find_peaks(x)
        peaks_values = self.get_first_n_peaks(np.array(x)[indices_peaks])
        return peaks_values

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        list_of_features = []
        for signal_no in range(0, len(X)):
            features = []

            signal = X[signal_no, :]
            features += self.get_features(self.get_fft_values(signal))
            features += self.get_features(self.get_psd_values(signal))
            features += self.calculate_statistics(signal)
            features += self.get_dwt_features(signal)
            list_of_features.append(features)
        return np.array(list_of_features)
