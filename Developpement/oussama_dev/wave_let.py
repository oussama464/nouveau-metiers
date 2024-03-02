import os
import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any
from data_preparation.data_prep import DataPreper, PaddingType
import torch

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


# plot the scaelogram
def plot_cwt_coeffs(coef):
    plt.figure(figsize=(15, 20))
    plt.imshow(
        coef,
        extent=[0, 200, 31, 1],
        interpolation="bilinear",
        cmap="jet",
        aspect="auto",
        vmax=abs(coef).max(),
        vmin=-abs(coef).max(),
    )
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(1, 31, 1))
    plt.xticks(np.arange(0, 201, 10))
    plt.show()


# plot the signal
def plot_signal(signal):
    t = np.arange(2015, 2024, 1)
    plt.figure(figsize=(15, 10))
    plt.plot(t, signal)
    plt.grid(color="gray", linestyle=":", linewidth=0.5)
    plt.show()


# 'mexh', 'morl'
def reshape_data_for_cnn(X, scales, waveletname):
    new_X = []
    for sample_no in range(len(X)):
        item = X[sample_no, :]
        coefs, freqs = pywt.cwt(item, scales, wavelet=waveletname)
        coefs = coefs.reshape(-1)
        new_X.append(coefs)
    new_X = np.array(new_X)
    reshaped = new_X.reshape((new_X.shape[0], 9, 9, 1))
    return reshaped


if __name__ == "__main__":
    data = get_formatted_raw_data(FILE_PATH)

    perepr = DataPreper(PaddingType.MEAN)
    data = get_formatted_raw_data(FILE_PATH)
    X_train, X_test, y_train, y_test = perepr.split_dataset_into_train_test(data)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    scales = np.arange(1, 10)
    waveletname = "morl"
    X_train, X_test = reshape_data_for_cnn(
        X_train, scales, waveletname
    ), reshape_data_for_cnn(X_test, scales, waveletname)
    print(X_train.shape, X_test.shape)
