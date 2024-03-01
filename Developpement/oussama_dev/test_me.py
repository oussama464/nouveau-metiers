import os
from enum import StrEnum
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Any, Callable, Optional
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import matplotlib.pyplot as plt
from scipy.fft import fft

# from scipy.fftpack import fft, fftfreq
from scipy.signal import welch, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
import pywt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics


os.environ["KERAS_BACKEND"] = "torch"
from keras.utils import to_categorical


FILE_PATH = "/home/bobo/Desktop/nouveau_metier/Developpement/raw_csv/extracted_jobs.csv"


def get_formatted_raw_data(file_path: str) -> list[list[Any]]:

    df = pd.read_csv(file_path)
    df_grouped = df.groupby(["job_code", "is_emerging_job"]).agg(list).reset_index()

    # Transforming the grouped dataframe to the desired format
    training_data = df_grouped.apply(
        lambda row: [row["job_code"], row["rawpop"], row["is_emerging_job"]], axis=1
    ).tolist()
    return training_data


class PaddingType(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"
    MAX_SIZE_NO_PAD = "no_padding"


@dataclass
class DataPreper:
    padding_type: PaddingType
    padding_strategy: Optional[Callable[[list[list[Any]]], list[list[Any]]]] = None

    def __post_init__(self):
        strategy_map = {
            PaddingType.MEAN: self.mean_padding_strategy,
            PaddingType.MEDIAN: self.median_padding_strategy,
            PaddingType.MAX_SIZE_NO_PAD: self.no_padding_strategy_max_sizes_only,
        }
        if self.padding_type in strategy_map:
            self.padding_strategy = strategy_map[self.padding_type]
        else:
            raise ValueError(
                f"Unrecognized padding type: {self.padding_type} available options are {PaddingType}"
            )

    def get_min_max_sample_size(self, data: list[list[Any]]) -> tuple[int, int]:
        sizes = list(map(lambda x: len(x[1]), data))
        return min(sizes), max(sizes)

    # padding strategy for missing mesurment
    # zero padding , mean ,median ,last_value ,no_padding
    def no_padding_strategy_max_sizes_only(
        self, data: list[list[Any]]
    ) -> list[list[Any]]:
        _, max_size = self.get_min_max_sample_size(data)
        return list(filter(lambda x: len(x[1]) == max_size, data))

    def mean_padding_strategy(self, data: list[list[Any]]) -> list[list[Any]]:
        new_padded_data = []
        _, max_size = self.get_min_max_sample_size(data)
        for item in data:
            if len(item[1]) < max_size:
                mean_val = np.mean(item[1])
                padded_item = item[1] + [mean_val] * (max_size - len(item[1]))
                new_padded_data.append([item[0], padded_item, item[2]])
            else:
                new_padded_data.append(item)
        return new_padded_data

    def median_padding_strategy(self, data: list[list[Any]]) -> list[list[Any]]:
        new_padded_data = []
        _, max_size = self.get_min_max_sample_size(data)
        for item in data:
            if len(item[1]) < max_size:
                mean_val = np.median(item[1])
                padded_item = item[1] + [mean_val] * (max_size - len(item[1]))
                new_padded_data.append([item[0], padded_item, item[2]])
            else:
                new_padded_data.append(item)
        return new_padded_data

    def rolling_window_smooting_pandas(self, data, window_size=3):
        rolling_avrage_smoothed = []
        for item in data:
            job_code, values, is_emerging = item
            series = pd.Series(values)
            rolling_avrage = series.rolling(window=window_size).mean()
            # rolling_avrage_smoothed[job_code] = rolling_avrage.to_list()
            rolling_avrage.dropna(inplace=True)
            rolling_avrage_smoothed.append(
                [job_code, rolling_avrage.to_list(), is_emerging]
            )
        return rolling_avrage_smoothed

    def rolling_window_smoothing(self, data, window_size):
        rolling_avrage_smoothed = []
        for item in data:
            job_code, values, is_emerging = item
            adjusted_values = []
            for i in range(len(values)):
                window = values[max(0, i - window_size + 1) : i + 1]
                adjusted_values.append(sum(window) / len(window))
            rolling_avrage_smoothed.append([job_code, adjusted_values, is_emerging])
        return rolling_avrage_smoothed

    def split_dataset_into_train_test(
        self, dataset: list[list]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.padding_strategy:
            padded_data = self.padding_strategy(dataset)
        else:
            raise ValueError("padding strategy was not set")
        X = [item[1] for item in padded_data]
        y = [item[2] for item in padded_data]

        # Convert lists to numpy arrays
        X_array = np.array(X)
        y_array = np.array(y, dtype=bool)
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array, test_size=0.33, random_state=42, stratify=y_array
        )
        # print(Counter(y))
        # print(Counter(y_train))
        # print(Counter(y_test))
        # one hot encode target
        # y_train = to_categorical(y_train)
        # y_test = to_categorical(y_test)
        # y_test = np.argmax(y_test, axis=1)
        # y_train = np.argmax(y_train, axis=1)

        return X_train, X_test, y_train, y_test


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


@dataclass
class FeatureSelector:
    target: np.ndarray
    anova_k_features: int | str = "all"
    pca_variance_threshold: float = 0.95

    def __post_init__(self):
        "inverse the to_categorical transform in order to use select features"
        if self.target.ndim != 1:
            self.target = np.argmax(self.target, axis=1)

    def select_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        # configure to select all features
        fs = SelectKBest(score_func=f_classif, k=self.anova_k_features)
        # learn relationship from the training data
        fs.fit(X_train, self.target)
        # transform the train input data
        X_train_fs = fs.transform(X_train)
        # transform the test input data
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs.scores_.tolist()

    # what are the scores for the features
    def bar_plot_features_scores(self, scores: list[float]) -> None:
        for i in range(len(scores)):
            print("feature %d: %f" % (i, scores[i]))

        plt.bar([i for i in range(len(scores))], scores)
        plt.show()

    def apply_pca(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pca = PCA(n_components=self.pca_variance_threshold)
        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)
        return X_train_reduced, X_test_reduced


def prepare_model_data():
    perepr = DataPreper(PaddingType.MEAN)
    data = get_formatted_raw_data(FILE_PATH)
    smoothed_data = perepr.rolling_window_smoothing(data, 3)
    X_train, X_test, y_train, y_test = perepr.split_dataset_into_train_test(
        smoothed_data
    )
    f_extract_test, f_extract_train = FeatureExtractor(
        X_test, y_test
    ), FeatureExtractor(X_train, y_train)
    X_futures_test, y_test = f_extract_test.X, f_extract_test.y
    X_futures_train, y_train = f_extract_train.X, f_extract_train.y
    f_selector = FeatureSelector(
        y_train, anova_k_features=10, pca_variance_threshold=0.99
    )
    X_train_fs, X_test_fs, scores = f_selector.select_features(
        X_futures_train, X_futures_test
    )
    # f_selector.bar_plot_features_scores(scores)
    X_train_reduced, X_test_reduced = f_selector.apply_pca(X_train_fs, X_test_fs)
    return X_train_reduced, X_test_reduced, y_train, y_test


def evaluate_model(X_train, X_test, y_train, y_test):
    logistic_reg = LogisticRegression()
    logistic_reg.fit(X_train, y_train)
    predictions = logistic_reg.predict(X_test)
    score = logistic_reg.score(X_test, y_test)
    return predictions, score


def plot_model_confusion_matrix(y_test, predictions, score):
    cm = metrics.confusion_matrix(y_test, predictions, labels=[True, False])
    plt.figure(figsize=(9, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        linewidths=0.5,
        square=True,
        cmap="Blues_r",
        xticklabels=["True", "False"],
        yticklabels=["True", "False"],
    )
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    all_sample_title = f"Accuracy Score: {score}"
    plt.title(all_sample_title, size=15)
    plt.show()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_model_data()
    predictions, score = evaluate_model(X_train, X_test, y_train, y_test)
    plot_model_confusion_matrix(y_test, predictions, score)
