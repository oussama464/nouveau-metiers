import glob
import pathlib
from enum import StrEnum
import numpy as np
from scipy.fftpack import fft
from scipy.signal import welch, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pywt

BASE_PATH = pathlib.Path(__file__).parent.joinpath("UCI_HAR_Dataset")


def read_signals(filename):
    with open(filename, "r") as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.strip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
        data = np.array(data, dtype=np.float32)
    return data


def read_lables(filename):
    with open(filename, "r") as fp:
        activities = fp.read().splitlines()
        activities = list(map(int, activities))
    return np.array(activities)


class ALLOWED_GROUPS(StrEnum):
    TEST = "test"
    TRAIN = "train"


class DataPreper:
    def __init__(self, group: ALLOWED_GROUPS) -> None:
        if group not in ALLOWED_GROUPS:
            raise ValueError(f"allowed groups are {ALLOWED_GROUPS}")
        else:
            self.group = group
        self._X, self._y = self.get_data_and_lables()

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    def get_group_file_paths(self):

        data_path = BASE_PATH.joinpath(
            f"{self.group}/Inertial_Signals/*.txt"
        ).as_posix()
        labels_path = BASE_PATH.joinpath(f"{self.group}/y_{self.group}.txt").as_posix()
        data_files_paths = glob.glob(data_path)
        return data_files_paths, labels_path

    def get_data_and_lables(self):
        data_files_paths, labels_path = self.get_group_file_paths()
        signals = [read_signals(input_file) for input_file in data_files_paths]
        signals = np.transpose(np.array(signals), (1, 2, 0))
        labels = read_lables(labels_path)
        return signals, labels


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

    def cwt_transform(self, signal: np.ndarray, waveletname="morl"):
        waveletname = "morl"
        scales = np.arange(1, 128)
        coefficients, frequencies = pywt.cwt(signal, scales, waveletname)
        return coefficients, frequencies

    def dwt_transform(
        self, signal: np.ndarray, waveletname="db2", level=8
    ) -> list[np.ndarray]:

        coeffs = pywt.wavedec(signal, waveletname, level=level)
        reconstructed_signal = pywt.waverec(coeffs, waveletname)
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
        # rms = np.nanmean(np.sqrt(list_values**2))
        return [n5, n25, n75, n95, median, mean, std, var]

    def get_fft_values(self, x):
        fxt = fft(x)
        fxt = 2.0 / len(fxt) * np.abs(fxt[0 : len(fxt) // 2])
        return fxt

    def get_psd_values(self, x, f_s=1):
        # fs : sampling freq by default 1
        _, psd_values = welch(x, fs=f_s)
        return psd_values

    def get_first_n_peaks(self, x, no_peaks=5):
        x_ = list(x)

        if len(x_) >= no_peaks:
            return x_[:no_peaks]
        else:
            missing_no_peaks = no_peaks - len(x_)
            return x_ + [0] * missing_no_peaks

    def get_features(self, x):
        indices_peaks, _ = find_peaks(x)
        peaks_x = self.get_first_n_peaks(x[indices_peaks])
        return peaks_x

    def extract_features_lables(self):
        list_of_features = []
        list_of_lables = []
        for signal_no in range(0, len(self.dataset)):
            features = []
            list_of_lables.append(self.lables[signal_no])
            for signal_comp in range(0, self.dataset.shape[2]):
                signal = self.dataset[signal_no, :, signal_comp]
                features += self.get_features(self.get_fft_values(signal))
                # features += self.get_features(self.get_psd_values(signal))
            list_of_features.append(features)
        return np.array(list_of_features), np.array(list_of_lables)


preper_tarin = DataPreper("train")
preper_test = DataPreper("test")
featurizer_tain, featurizer_test = FeatureExtractor(
    preper_tarin.X, preper_tarin.y
), FeatureExtractor(preper_test.X, preper_test.y)
X_train, Y_train = featurizer_tain.X, featurizer_tain.y
X_test, Y_test = featurizer_test.X, featurizer_test.y


####


clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, Y_train)
print("Accuracy on training set is : {}".format(clf.score(X_train, Y_train)))
print("Accuracy on test set is : {}".format(clf.score(X_test, Y_test)))
Y_test_pred = clf.predict(X_test)
print(classification_report(Y_test, Y_test_pred))
