import numpy as np
from scipy.fft import fft
from scipy.signal import welch, find_peaks
import pywt


# os.environ["KERAS_BACKEND"] = "torch"
# from keras.utils import to_categorical


class FeatureExtractor:
    def __init__(self, dataset, lables) -> None:
        self.dataset = dataset
        self.lables = lables
        self._X, self._y = self.extract_features_lables()

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    def dwt_transform(self, signal: np.ndarray, waveletname="db2") -> list[np.ndarray]:

        coeffs = pywt.wavedec(signal, waveletname)
        # reconstructed_signal = pywt.waverec(coeffs, waveletname)
        return coeffs

    def calculate_statistics(self, list_values):
        n5 = np.nanpercentile(list_values, 5)
        n25 = np.nanpercentile(list_values, 25)
        n75 = np.nanpercentile(list_values, 75)
        n95 = np.nanpercentile(list_values, 95)
        median = np.nanpercentile(list_values, 50)
        mean = np.nanmean(list_values)
        std = np.nanstd(list_values)
        var = np.nanvar(list_values)
        rms = np.nanmean(np.sqrt(list_values**2))
        return [n5, n25, n75, n95, median, mean, std, var, rms]

    def get_dwt_features(self, signal: np.ndarray):
        dwt_features = []
        coeffs = self.dwt_transform(signal)
        for coeff in coeffs:
            dwt_features += self.calculate_statistics(coeff)
        return dwt_features

    def get_fft_values(self, x):
        fxt = fft(x)
        fxt = 2.0 / len(fxt) * np.abs(fxt[0 : len(fxt) // 2])
        return fxt

    def get_psd_values(self, x, f_s=1):
        # fs : sampling freq by default 1
        _, psd_values = welch(x, fs=f_s)
        return psd_values

    def get_first_n_peaks(self, x, no_peaks=2):
        x_ = list(x)

        if len(x_) >= no_peaks:
            return x_[:no_peaks]
        else:
            missing_no_peaks = no_peaks - len(x_)
            return x_ + [0] * missing_no_peaks

    def get_features(self, x):
        indices_peaks, _ = find_peaks(x)
        peaks_values = self.get_first_n_peaks(np.array(x)[indices_peaks])
        return peaks_values

    def extract_features_lables(self):
        list_of_features = []
        list_of_lables = []
        for signal_no in range(0, len(self.dataset)):
            features = []
            list_of_lables.append(self.lables[signal_no])

            signal = self.dataset[signal_no, :]
            features += self.get_features(self.get_fft_values(signal))
            features += self.get_features(self.get_psd_values(signal))
            features += self.calculate_statistics(signal)
            features += self.get_dwt_features(signal)
            list_of_features.append(features)
        return np.array(list_of_features), np.array(list_of_lables)
